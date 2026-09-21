"""Compare pinned Jev with a completed adapter run on its exact held-out inputs."""
import argparse
import concurrent.futures
import json
import math
import os
import time
import urllib.error
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from nimble.paths import PROJECT_ROOT
from nimble.datasets.dataset_io import request_for, validate_teacher
from nimble.datasets.typesafe_curator_bridge import call_typesafe
from nimble.evaluation.evaluate_pilot import assess
from nimble.training.schema_data import fingerprint, read_rows


def summarize(rows, lookup):
    result = {}
    for kind in ('all', 'choice', 'noul', 'score'):
        selected = [r for r in rows if kind == 'all' or r['kind'] == kind]
        group = {'count': len(selected), 'correct': sum(r['correct'] for r in selected)}
        group['accuracy'] = group['correct'] / group['count']
        for metric in ('nll', 'brier', 'score_absolute_error'):
            if all(metric in r for r in selected):
                group[metric] = sum(r[metric] for r in selected) / len(selected)
        result[kind] = group
    pairs = defaultdict(list)
    for r in rows:
        pairs[lookup[r['id']]['family']].append(r)
    if not all(len(p) == 2 for p in pairs.values()):
        raise ValueError('Expected complete evaluation pairs')
    result['pairs'] = {'count': len(pairs), 'both_correct': sum(all(r['correct'] for r in p) for p in pairs.values())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model', default='jev-1.13.0')
    parser.add_argument('--output-dir', type=Path, help='Optional independent directory for this evaluation')
    args = parser.parse_args()
    rows = read_rows(args.data)
    audit = json.loads((args.run_dir / 'data_audit.json').read_text())
    if fingerprint(rows) != audit['validation_fingerprint']:
        raise ValueError('Data differs from adapter evaluation set')
    lookup = {r['id']: r for r in rows}
    output = args.output_dir or args.run_dir / args.model
    output.mkdir(parents=True, exist_ok=True)
    config = {'model': args.model, 'dataset_fingerprint': fingerprint(rows),
              'dataset': str(args.data.resolve()), 'request_fields': ['model', 'state', 'questions'],
              'selection': 'All fixed evaluation rows; no prompt tuning or retraining'}
    settings = output / 'settings.json'
    if settings.exists() and json.loads(settings.read_text()) != config:
        raise ValueError('Existing evaluation configuration differs')
    settings.write_text(json.dumps(config, indent=2) + '\n')
    progress = output / 'responses.jsonl'
    saved = read_rows(progress) if progress.exists() and progress.stat().st_size else []
    for r in saved:
        if r['id'] not in lookup or r['input_fingerprint'] != fingerprint(lookup[r['id']]['input']):
            raise ValueError('Saved response input changed')
        validate_teacher(lookup[r['id']], r['response'], args.model)
    load_dotenv(PROJECT_ROOT / '.env')
    key = os.environ['TYPESAFE_API_KEY']

    def request(row):
        started = time.monotonic()
        payload = request_for(row, args.model)
        for attempt in range(4):
            try:
                response = call_typesafe(payload, key)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 529, 502, 503, 504) or attempt == 3:
                    raise RuntimeError(f'TypeSafe HTTP {exc.code}') from None
                time.sleep(2 ** attempt)
        validate_teacher(row, response, args.model)
        return {'id': row['id'], 'input_fingerprint': fingerprint(row['input']),
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': time.monotonic() - started, 'response': response}

    pending = [r for r in rows if r['id'] not in {s['id'] for s in saved}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool, progress.open('a') as stream:
        for future in concurrent.futures.as_completed([pool.submit(request, r) for r in pending]):
            result = future.result()
            saved.append(result)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            if len(saved) % 10 == 0:
                print(f'Jev evaluation: {len(saved)}/{len(rows)}', flush=True)
    responses = {r['id']: r['response'] for r in saved}
    evaluated, zero_gold = [], 0
    for row in rows:
        kind = row['input']['questions']['decision']['type']
        answer = responses[row['id']]['answers']['decision']
        if kind == 'noul':
            probs = {'false': 1 - answer['noul'], 'true': answer['noul']}
        else:
            # Preserve criterion order for score ties; normalize API rounding drift.
            labels = list(row['input']['questions']['decision']['criteria']) if kind == 'choice' else [
                str(i) for i in range(len(row['input']['questions']['decision']['criteria']))]
            total = sum(answer['probabilities'].values())
            probs = {k: answer['probabilities'][k] / total for k in labels}
        assessment = assess(probs, row['reference']['target'], kind)
        prediction = (answer['choice'] if kind == 'choice' else answer['noul'] > .5 if kind == 'noul'
                      else assessment['prediction'])
        r = {'id': row['id'], 'kind': kind, 'family': row['source_family'], 'target': row['reference']['target'],
             'prediction': prediction, 'correct': prediction == row['reference']['target'],
             'probabilities': probs, 'nll': assessment['negative_log_likelihood'],
             'brier': assessment['multiclass_brier']}
        if kind == 'score':
            r.update(expected_score=assessment['expected_score'], score_absolute_error=assessment['absolute_score_error'])
        zero_gold += assessment['reference_probability'] == 0
        evaluated.append(r)
    base, trained = [json.loads((args.run_dir / f).read_text())['rows'] for f in ('before.json', 'after.json')]
    for records in (base, trained):
        if [r['id'] for r in records] != [r['id'] for r in rows]:
            raise ValueError('Model evaluation IDs differ')
        for r in records:
            assert r['target'] == lookup[r['id']]['reference']['target']
            gold = str(r['target']).lower() if isinstance(r['target'], bool) else str(r['target'])
            # Apply the same floor to all models for reported probability NLL.
            r['nll'] = -math.log(max(r['probabilities'][gold], 1e-15))
    summaries = {'base': summarize(base, lookup), 'trained': summarize(trained, lookup),
                 'jev': summarize(evaluated, lookup)}
    paired = {'both_correct': 0, 'trained_only_correct': 0, 'jev_only_correct': 0, 'neither_correct': 0}
    for a, b in zip(trained, evaluated):
        paired['both_correct' if a['correct'] and b['correct'] else 'trained_only_correct' if a['correct']
               else 'jev_only_correct' if b['correct'] else 'neither_correct'] += 1
    result = {'model': args.model, 'summary': summaries, 'paired_comparison': paired, 'jev_rows': evaluated,
              'zero_reference_probabilities_jev': zero_gold, 'nll_floor': 1e-15,
              'input_tokens': sum(r['usage']['input_tokens'] for r in responses.values()),
              'output_tokens': sum(r['usage']['output_tokens'] for r in responses.values()),
              'method': 'Same semantic inputs and reference labels; native TypeSafe API versus Qwen schema prompt. API probabilities renormalized; Choice uses API choice, Noul uses >0.5, Score uses modal level in criterion order. No tuning.'}
    (output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = [f'# Jev comparison: same {len(rows)}-example curated holdout', '',
             f'| Metric | Base Qwen | Trained Qwen | {args.model} |', '| --- | ---: | ---: | ---: |']
    for kind, name in [('all','Overall matches'),('choice','Choice'),('noul','Boolean'),('score','Score level')]:
        lines.append('| ' + name + ' | ' + ' | '.join(f"{s[kind]['correct']}/{s[kind]['count']}" for s in summaries.values()) + ' |')
    for kind, key, name in [('all','nll','NLL (lower better)'),('all','brier','Brier (lower better)'),('score','score_absolute_error','Expected score MAE')]:
        lines.append('| ' + name + ' | ' + ' | '.join(f"{s[kind][key]:.4f}" for s in summaries.values()) + ' |')
    lines += ['| Both pair members correct | ' + ' | '.join(f"{s['pairs']['both_correct']}/{s['pairs']['count']}" for s in summaries.values()) + ' |', '',
              f'All models use the same {len(rows)} held-out examples and provisional labels. Jev receives only the original state and question, never references or evidence certificates. Jev was not used to generate or verify these labels. No model or prompt was tuned for this comparison.', '',
              result['method'], '',
              f"Jev assigned zero returned probability to the reference on {zero_gold} rows. Returned probabilities can be rounded; NLL uses a 1e-15 floor for every model, so zero-probability errors can dominate it. Brier is less sensitive to this issue.", '',
              f"These are {summaries['jev']['pairs']['count']} correlated pairs across {len({r['source_family'] for r in rows})} source families with synthetic, model-checked labels, not human-reviewed ground truth. This comparison establishes performance on this holdout, not general superiority or calibration. No comparable latency benchmark was run.", '',
              '## Paired correctness', '', '```json', json.dumps(paired, indent=2), '```', '',
              f"API usage: {result['input_tokens']:,} input tokens, {result['output_tokens']:,} output tokens. Raw responses and request-input fingerprints are retained in responses.jsonl."]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'jev_rows'}, indent=2))


if __name__ == '__main__':
    main()
