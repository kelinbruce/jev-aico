"""Create a blind human-review worksheet and a separate model-label key."""
import argparse
import json
import random
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.datasets.create_diverse_dataset import read_jsonl
from nimble.datasets.dataset_io import write_jsonl
from nimble.datasets.contrastive_data import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=PROJECT_ROOT / 'data/evidence_curated_1000_gpt56')
    parser.add_argument('--limit', type=int, default=30)
    args = parser.parse_args()
    records = read_jsonl(args.data / 'train.jsonl')
    if not records:
        raise ValueError('No accepted rows to review')
    if args.limit < 1:
        raise ValueError('Review limit must be positive')
    # Seeded random sample from accepted rows, no selection based on model labels.
    random.Random(17).shuffle(records)
    records = records[:args.limit]
    blind, key = [], []
    pages = ['# Blind label review\n\nRead each context and apply only its question and criteria. '
             'Record one semantic answer, or mark it ambiguous and explain. Do not consult '
             '`human_review_key.jsonl` until your judgments are saved. These are synthetic '
             'examples with provisional model labels. This worksheet has not been completed by a human.\n']
    for i, row in enumerate(records):
        review_id = f'review-{i + 1:03d}'
        blind.append({'review_id': review_id, 'input': row['input'],
                      'human_target': None, 'ambiguous': None, 'reviewer_notes': '', 'human_reviewed': False})
        key.append({'review_id': review_id, 'example_id': row['id'], 'source_family': row['source_family'],
                    'model_target': row['reference']['target'], 'input_sha256': fingerprint(row['input'])})
        pages.append(f'## {review_id}\n\n```json\n{json.dumps(row["input"], ensure_ascii=False, indent=2)}\n```\n\n'
                     '**Your answer:**\n\n**Ambiguous? Why?**\n\n')
    write_jsonl(args.data / 'human_review_blind.jsonl', blind)
    write_jsonl(args.data / 'human_review_key.jsonl', key)
    (args.data / 'HUMAN_REVIEW.md').write_text('\n'.join(pages))
    print(json.dumps({'review_examples': len(records), 'human_reviewed': False}))


if __name__ == '__main__':
    main()
