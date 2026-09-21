"""Measure MLX/CUDA agreement on identical saved datasets and checkpoints."""

import argparse
import json
import statistics
from pathlib import Path

from nimble.evaluation.evaluate_models import MODELS


def compare(left_root, right_root, output):
    comparisons = {}
    for name in MODELS:
        left, right = left_root / name, right_root / name
        a = json.loads((left / 'settings.json').read_text())
        b = json.loads((right / 'settings.json').read_text())
        for key in ('dataset_sha256', 'model', 'revision', 'temperature', 'prompt_sha256', 'prompt_roles', 'mode'):
            if a[key] != b[key]:
                raise ValueError(f'{name}: incompatible {key}')
        rows_a = [json.loads(line) for line in (left / 'rows.jsonl').read_text().splitlines()]
        rows_b = [json.loads(line) for line in (right / 'rows.jsonl').read_text().splitlines()]
        if [r['id'] for r in rows_a] != [r['id'] for r in rows_b] or not rows_a:
            raise ValueError('Row identities or order differ')
        flips, differences, logit_differences = [], [], []
        for x, y in zip(rows_a, rows_b):
            fx, fy = x['raw_student']['fields']['decision'], y['raw_student']['fields']['decision']
            if fx['candidate_token_ids'] != fy['candidate_token_ids'] or fx['code_to_choice'] != fy['code_to_choice']:
                raise ValueError(f'{name}/{x["id"]}: candidate encoding differs')
            mlx_metrics = x['raw_student']['metrics']
            if mlx_metrics['prefix_tokens'] + mlx_metrics['suffix_tokens'][0] != fy['prompt_token_count']:
                raise ValueError(f'{name}/{x["id"]}: prompt token count differs')
            if x['reference'] != y['reference']:
                raise ValueError('References differ')
            differences += [abs(fx['scores'][k] - fy['scores'][k]) for k in fx['scores']]
            logit_differences += [abs(fx['logits'][k] - fy['logits'][k]) for k in fx['logits']]
            if fx['value'] != fy['value']:
                flips.append({'id': x['id'], 'mlx': fx['value'], 'cuda': fy['value'],
                              'reference': x['reference']['target']})
        comparisons[name] = {'count': len(rows_a), 'candidate_encodings_match': True,
                             'prompt_token_counts_match': True,
                             'prediction_agreement': len(rows_a) - len(flips), 'prediction_changes': flips,
                             'mean_absolute_probability_difference': statistics.mean(differences),
                             'max_absolute_probability_difference': max(differences),
                             'mean_absolute_logit_difference': statistics.mean(logit_differences),
                             'max_absolute_logit_difference': max(logit_differences)}
    output.write_text(json.dumps(comparisons, indent=2) + '\n')
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mlx-dir', type=Path, required=True)
    parser.add_argument('--cuda-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.mlx_dir, args.cuda_dir, args.output), indent=2))


if __name__ == '__main__':
    main()
