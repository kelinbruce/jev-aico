"""Evaluate pinned baseline checkpoints on evidence inputs without teacher data."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from nimble.paths import PROJECT_ROOT
from nimble.evaluation.evaluate_models import MODELS
from nimble.evaluation.evaluate_pilot import adapt_input, assess
from nimble.evaluation.evaluate_jev_holdout import summarize
from nimble.scoring.cuda_scorer import CudaCandidateScorer
from nimble.scoring.parallel_schema import SYSTEM_PROMPT
from nimble.training.schema_data import fingerprint, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), required=True)
    args = parser.parse_args()
    rows = read_rows(args.data)
    lookup = {r['id']: r for r in rows}
    for name in args.models:
        config = MODELS[name]
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        settings = {**config, 'dataset_fingerprint': fingerprint(rows), 'count': len(rows),
                    'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                    'prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                    'method': 'BF16 checkpoint, independent full prefill, FP32 candidate projection, temperature 1, no reasoning or tuning',
                    'max_input_tokens': 2048}
        if (out / 'settings.json').exists():
            assert json.loads((out / 'settings.json').read_text()) == settings
        (out / 'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
        progress = out / 'rows.jsonl'
        saved = read_rows(progress) if progress.exists() and progress.stat().st_size else []
        assert [r['id'] for r in saved] == [r['id'] for r in rows[:len(saved)]]
        for r in saved:
            assert r['input_fingerprint'] == fingerprint(lookup[r['id']]['input'])
        if len(saved) < len(rows):
            print(f'Downloading {name}', flush=True)
            checkpoint = snapshot_download(config['model'], revision=config['revision'],
                cache_dir=str(PROJECT_ROOT / '.cache/huggingface/hub'),
                allow_patterns=['*.json', '*.jinja', '*.safetensors', '*.txt', '*.model'])
            info = HfApi().model_info(config['model'], revision=config['revision'], files_metadata=True)
            assert info.sha == config['revision']
            expected = {f.rfilename: f.lfs.sha256 for f in info.siblings if f.lfs}
            integrity = {}
            for p in sorted(Path(checkpoint).glob('*.safetensors')):
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for chunk in iter(lambda: f.read(8*1024*1024), b''):
                        h.update(chunk)
                digest = h.hexdigest()
                # Verify against revision metadata, independent of Hub cache layout.
                assert expected[p.name] == digest, p.name
                integrity[p.name] = digest
            assert integrity
            (out / 'weight_sha256.json').write_text(json.dumps(integrity, indent=2)+'\n')
            print(f'Loading {name}', flush=True)
            scorer = CudaCandidateScorer(checkpoint, config['model'], config['revision'], max_input_tokens=2048)
            with progress.open('a') as stream:
                for row in rows[len(saved):]:
                    started = time.monotonic()
                    context, schema = adapt_input(row['input'])
                    prediction = scorer.score(context, schema)
                    field = prediction['fields']['decision']
                    total = sum(field['scores'].values())
                    probs = {k: p/total for k,p in field['scores'].items()}
                    kind = row['input']['questions']['decision']['type']
                    a = assess(probs, row['reference']['target'], kind)
                    r = {'id': row['id'], 'kind': kind, 'family': row['source_family'],
                         'input_fingerprint': fingerprint(row['input']), 'target': row['reference']['target'],
                         'prediction': a['prediction'], 'correct': a['correct'], 'probabilities': probs,
                         'nll': a['negative_log_likelihood'], 'brier': a['multiclass_brier'],
                         'logits': field['logits'], 'prompt_token_sha256': field['prompt_token_sha256'],
                         'prompt_token_count': field['prompt_token_count'], 'elapsed_seconds': time.monotonic()-started}
                    if kind == 'score':
                        r.update(expected_score=a['expected_score'], score_absolute_error=a['absolute_score_error'])
                    stream.write(json.dumps(r, allow_nan=False)+'\n'); stream.flush()
                    saved.append(r)
                    if len(saved) % 20 == 0 or len(saved) == len(rows):
                        print(f'{name}: {len(saved)}/{len(rows)}', flush=True)
            del scorer
            gc.collect(); torch.cuda.empty_cache()
        result = {'settings': settings, 'summary': summarize(saved, lookup), 'rows': saved}
        (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'model': name, 'summary': result['summary']}), flush=True)


if __name__ == '__main__':
    main()
