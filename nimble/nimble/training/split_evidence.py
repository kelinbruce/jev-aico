"""Create a fixed, source-family-disjoint training/evaluation evidence split."""
import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from nimble.datasets.scaled_evidence import audit_release
from nimble.training.evidence_data import validate_evidence_data
from nimble.training.schema_data import fingerprint, read_rows, validate_separation


def select_families(rows, eval_count, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source_family']].append(row)
    # Selection depends on metadata only, never model predictions or correctness.
    families = sorted(grouped, key=lambda f: hashlib.sha256(f'{seed}:{f}'.encode()).hexdigest())
    sizes = [len(grouped[f]) for f in families]
    domains = [grouped[f][0]['domain'] for f in families]
    kinds = [Counter(r['input']['questions']['decision']['type'] for r in grouped[f]) for f in families]
    maximum = next((i for i in range(1, len(sizes) + 1)
                    if sum(sorted(sizes)[:i]) > eval_count), len(sizes) + 1) - 1
    best, best_key = None, None
    for count in range(maximum, 0, -1):
        for indices in itertools.combinations(range(len(families)), count):
            if sum(sizes[i] for i in indices) != eval_count:
                continue
            coverage = len({domains[i] for i in indices})
            totals = {k: sum(kinds[i][k] for i in indices) for k in ('choice', 'noul', 'score')}
            key = (-coverage, sum((3 * n - eval_count) ** 2 for n in totals.values()), indices)
            if best_key is None or key < best_key:
                best_key, best = key, {families[i] for i in indices}
        # Smaller combinations cannot match the best domain coverage.
        if best_key is not None and -best_key[0] > count - 1:
            break
    if best is None:
        raise ValueError('Exact evaluation count is impossible without splitting source families')
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--eval-count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--accepted-file', default='accepted_progress.jsonl',
                        help='Accepted raw file within source-dir; use train.jsonl for a nested split')
    parser.add_argument('--for-tuning', action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a new split output directory')
    source = args.source_dir
    rows = read_rows(source / args.accepted_file)
    manifest = json.loads((source / 'manifest.json').read_text())
    token_manifest = json.loads((source / 'train_tokens.manifest.json').read_text())
    scoring, tokens = read_rows(source / 'train_scoring.jsonl'), read_rows(source / 'train_tokens.jsonl')
    if fingerprint(rows) != manifest['train_sha256'] or rows != read_rows(source / 'train.jsonl'):
        raise ValueError('Accepted progress differs from audited release')
    if fingerprint(scoring) != token_manifest['source_sha256'] or fingerprint(tokens) != token_manifest['records_sha256']:
        raise ValueError('Export fingerprints differ')
    if not [r['id'] for r in rows] == [r['id'] for r in scoring] == [r['id'] for r in tokens]:
        raise ValueError('Export IDs or order differ')
    validate_evidence_data(rows, manifest)
    heldout_families = select_families(rows, args.eval_count, args.seed)
    training = [r for r in rows if r['source_family'] not in heldout_families]
    heldout_original = [r for r in rows if r['source_family'] in heldout_families]
    # Preserve provenance and pair ID; the loader's validation role means no gradients.
    heldout = [{**r, 'split': 'validation'} for r in heldout_original]
    validate_separation(training, heldout)
    train_ids = {r['id'] for r in training}
    scoring = [r for r in scoring if r['id'] in train_ids]
    tokens = [r for r in tokens if r['id'] in train_ids]
    sources = read_rows(Path(manifest['source']))
    original_fingerprint = manifest['train_sha256']
    manifest = {**manifest, 'target': len(training), 'training': audit_release(training, sources),
                'train_sha256': fingerprint(training), 'parent_train_sha256': original_fingerprint,
                'notes': ['Derived subset; original release preserved. Whole source families held out.',
                          'All evidence certificates validated before splitting.']}
    token_manifest = {**token_manifest, 'rows': len(tokens), 'source_sha256': fingerprint(scoring),
                      'records_sha256': fingerprint(tokens),
                      'max_prompt_tokens': max(len(r['prompt_token_ids']) for r in tokens)}
    split = {'seed': args.seed, 'source_file': str((source / args.accepted_file).resolve()),
             'source_fingerprint': original_fingerprint, 'training_rows': len(training),
             'evaluation_rows': len(heldout), 'evaluation_fingerprint': fingerprint(heldout),
             'evaluation_source_families': sorted(heldout_families),
             'evaluation_coverage': audit_release(heldout_original, sources),
             'training_ids': sorted(train_ids), 'evaluation_ids': [r['id'] for r in heldout],
             'selection': 'Exact whole-family count; maximize domain coverage, then balance primitives; seeded hash tie-break',
             'evaluation_role': ('Inner validation for hyperparameter/epoch selection; separate outer holdout required'
                                 if args.for_tuning else 'Fixed held-out evaluation before and after; no tuning or checkpoint selection'),
             'initialization_required': 'Fresh base model; older adapters saw related source families'}
    args.output_dir.mkdir(parents=True)
    for name, records in [('train', training), ('train_scoring', scoring), ('train_tokens', tokens), ('eval', heldout)]:
        (args.output_dir / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    for name, value in [('manifest', manifest), ('train_tokens.manifest', token_manifest), ('split_manifest', split)]:
        (args.output_dir / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: v for k, v in split.items() if k not in ('training_ids', 'evaluation_ids')}, indent=2))


if __name__ == '__main__':
    main()
