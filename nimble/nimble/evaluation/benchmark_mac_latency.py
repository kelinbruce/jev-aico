"""Benchmark a merged local checkpoint on the complete frozen holdout with MLX."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from nimble.evaluation.benchmark_decision_latency import fingerprint, timing_summary
from nimble.evaluation.evaluate_pilot import adapt_input, assess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/eval.jsonl'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    from nimble.scoring.parallel_scorer import ParallelScorer

    if (args.output_dir/'responses.jsonl').exists():
        raise FileExistsError('Use a fresh output directory for each benchmark')
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    contract = json.loads((args.model_dir/'schema_config.json').read_text())
    merge = json.loads((args.model_dir/'READY.json').read_text())
    assert len(rows) == len({r['id'] for r in rows}) == contract['data_audit']['validation_rows'] == 324
    assert fingerprint(rows) == contract['data_audit']['validation_fingerprint']
    assert contract['prompt_code_sha256'] == hashlib.sha256(Path('nimble/scoring/parallel_schema.py').read_bytes()).hexdigest()
    inputs = []
    for row in rows:
        context, schema = adapt_input(row['input'])
        inputs.append({'id': row['id'], 'context': context, 'schema': schema,
                       'input_fingerprint': fingerprint(row['input']),
                       'kind': row['input']['questions']['decision']['type']})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scorer = ParallelScorer(model_path=args.model_dir, max_input_tokens=contract['max_length'],
                            model_id='nimble-diverse9b-v2', revision=merge['adapter_sha256'])
    mx.eval(scorer.model.parameters())
    mx.synchronize()
    load_seconds = time.perf_counter()-started
    runtime = {'model': 'nimble-diverse9b-v2', 'training_examples': merge['training_examples'],
               'heldout_examples': 324, 'hardware': mx.device_info(), 'platform': platform.platform(),
               'macos': platform.mac_ver()[0],
               'ram_bytes': int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
               'versions': {p: version(p) for p in ('mlx', 'mlx-lm', 'transformers')},
               'merge': merge, 'model_load_seconds': load_seconds,
               'resident_mlx_gib': mx.get_active_memory()/2**30,
               'head_dtype': str(scorer.head_weight.dtype),
               'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
               'input_fingerprint': fingerprint(inputs),
               'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'scorer_sha256': hashlib.sha256(Path('nimble/scoring/parallel_scorer.py').read_bytes()).hexdigest(),
               'batch_size': 1, 'concurrency': 1, 'mode': 'independent', 'warmups': 3,
               'requests_per_example': 1, 'temperature': 1,
               'timing_boundary': 'Synchronized local model.score including prompt tokenization and probability formatting; no network, model loading, or input adaptation',
               'projection': 'FP32 candidate-only output head; merged BF16 model',
               'autoregressive_tokens': 0, 'forward_passes_per_example': 1}
    (args.output_dir/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
    print(f'Model loaded in {load_seconds:.2f}s; warming up', flush=True)

    def score(row):
        mx.synchronize()
        started = time.perf_counter()
        result = scorer.score(row['context'], row['schema'], mode='independent')
        mx.synchronize()
        elapsed = time.perf_counter()-started
        return {'id': row['id'], 'kind': row['kind'], 'input_fingerprint': row['input_fingerprint'],
                'elapsed_seconds': elapsed, 'result': result}

    with (args.output_dir/'warmups.jsonl').open('w') as stream:
        for kind in ('choice', 'noul', 'score'):
            result = score(next(row for row in inputs if row['kind'] == kind))
            stream.write(json.dumps(result, allow_nan=False)+'\n')
            stream.flush()
    records = []
    with (args.output_dir/'responses.jsonl').open('w') as stream:
        for i, row in enumerate(inputs):
            result = score(row)
            stream.write(json.dumps(result, allow_nan=False)+'\n')
            stream.flush()
            records.append(result)
            if (i+1) % 20 == 0 or i+1 == len(inputs):
                print(f'Mac latency: {i+1}/{len(inputs)}', flush=True)
    summary = {'runtime': runtime, 'scoring': timing_summary([r['elapsed_seconds'] for r in records]),
               'by_kind': {kind: timing_summary([r['elapsed_seconds'] for r in records if r['kind'] == kind])
                           for kind in ('choice', 'noul', 'score')},
               'peak_mlx_active_gib': max(r['result']['metrics']['mlx_peak_active_gib'] for r in records)}
    # Only after all predictions are frozen do we join gold for an integrity check.
    lookup = {r['id']: r for r in rows}
    assessed = []
    for record in records:
        probs = record['result']['fields']['decision']['scores']
        total = sum(probs.values())
        assessed.append({'id': record['id'], **assess({k: v/total for k,v in probs.items()},
            lookup[record['id']]['reference']['target'], record['kind'])})
    summary['reference_matches'] = sum(r['correct'] for r in assessed)
    summary['reference_agreement'] = summary['reference_matches']/len(rows)
    (args.output_dir/'assessments.json').write_text(json.dumps(assessed, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'runtime'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
