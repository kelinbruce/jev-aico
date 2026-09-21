"""Append blind-reviewed examples without weakening evidence certificate checks."""
import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.training.schema_data import as_scoring, fingerprint, read_rows, validate_separation
from nimble.training.evidence_data import validate_evidence_data

VERSION = 'evidence-plus-reviewed-v1'


def assigned(row):
    row = copy.deepcopy(row)
    if row['split'] != 'unassigned':
        raise ValueError('Supplement must be an unassigned release')
    row['split'] = 'train'
    row['provenance']['source_id'] = row['id']
    return row


def validate_augmented_data(raw, manifest):
    from nimble.datasets.create_complex_segments import validate_content, validate_review
    if manifest['pipeline_version'] != VERSION:
        raise ValueError('Unsupported augmented data version')
    n = manifest['evidence_rows']
    evidence, added = raw[:n], raw[n:]
    if len(added) != manifest['reviewed_rows'] or fingerprint(raw) != manifest['train_sha256']:
        raise ValueError('Augmented training rows changed')
    if fingerprint(evidence) != manifest['evidence_manifest']['train_sha256']:
        raise ValueError('Original evidence rows changed')
    groups = validate_evidence_data(evidence, manifest['evidence_manifest'])
    if manifest.get('reviewed_source_storage') == 'reconstruct-from-training-v1':
        original = copy.deepcopy(added)
        for row in original:
            if row['split'] != 'train' or row['provenance']['source_id'] != row['id']:
                raise ValueError('Reviewed example assignment changed')
            row['split'] = 'unassigned'
            del row['provenance']['source_id']
        source_bytes = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
                               for r in original).encode()
    else:
        source = PROJECT_ROOT / manifest['reviewed_source']
        original = read_rows(source)
        source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != manifest['reviewed_source_sha256']:
        raise ValueError('Reviewed release bytes changed')
    if fingerprint(original) != manifest['reviewed_record_fingerprint']:
        raise ValueError('Reviewed release fingerprint changed')
    if added != [assigned(r) for r in original]:
        raise ValueError('Supplement differs from its reviewed source')
    for row in original:
        if (row['quality_status'] != 'model_checked' or not row['provenance']['synthetic']
                or not row['provenance']['review_reference_hidden']):
            raise ValueError('Supplement lacks accepted blind review')
        validate_content(row)
        validate_review(row, row['review'])
        if row['family'] in groups:
            raise ValueError('Supplement overlaps an evidence family')
    extra = defaultdict(list)
    for row in added:
        extra[row['family']].append('blind_reviewed_singleton')
    return {**groups, **extra}


def build(base, supplement, output):
    from nimble.datasets.create_complex_segments import audit
    if output.exists():
        raise ValueError('Use a new output directory')
    original = read_rows(base / 'train.jsonl')
    heldout = read_rows(base / 'eval.jsonl')
    added = read_rows(supplement / 'all.jsonl')
    source_manifest = json.loads((supplement / 'manifest.json').read_text())
    if fingerprint(added) != source_manifest['record_fingerprint']:
        raise ValueError('Reviewed source fingerprint mismatch')
    if hashlib.sha256((supplement / 'all.jsonl').read_bytes()).hexdigest() != source_manifest['dataset_sha256']:
        raise ValueError('Reviewed source bytes mismatch')
    if len(original) != 2676 or len(heldout) != 324 or len(added) != 150:
        raise ValueError('Unexpected release sizes')
    # The newer 120-example evaluation remains excluded too.
    domain_eval = read_rows(PROJECT_ROOT / 'data/lint_gaming_eval_v1/eval.jsonl')
    duplicate_audit = audit(added, [*original, *heldout, *domain_eval])
    training = original + [assigned(r) for r in added]
    if len({r['id'] for r in training}) != len(training):
        raise ValueError('Duplicate training IDs')
    validate_separation(training, heldout)
    for excluded in (heldout, domain_eval):
        if {r['id'] for r in training} & {r['id'] for r in excluded}:
            raise ValueError('Training/evaluation ID overlap')
        if {r['source_family'] for r in training} & {r.get('source_family', r['family']) for r in excluded}:
            raise ValueError('Training/evaluation family overlap')
    manifest = {
        'pipeline_version': VERSION, 'train_sha256': fingerprint(training),
        'evidence_rows': len(original), 'reviewed_rows': len(added),
        'evidence_manifest': json.loads((base / 'manifest.json').read_text()),
        'reviewed_source': str((supplement / 'all.jsonl').resolve().relative_to(PROJECT_ROOT)),
        'reviewed_source_sha256': source_manifest['dataset_sha256'],
        'reviewed_record_fingerprint': fingerprint(added),
        'eval_sha256': hashlib.sha256((base / 'eval.jsonl').read_bytes()).hexdigest(),
        'eval_record_fingerprint': fingerprint(heldout), 'eval_rows': len(heldout),
        'duplicate_audit': duplicate_audit, 'reference_human_reviewed': False,
        'notes': ['Original evidence certificates remain required and reconstructed.',
                  '150 separately blind-reviewed examples are not evidence-certified pairs.',
                  'All 30 new source families assigned wholly to training.',
                  '324-example evaluation copied byte-for-byte; no holdout-driven selection.'],
    }
    validate_augmented_data(training, manifest)
    output.mkdir(parents=True)
    for name, rows in [('train.jsonl', training), ('train_scoring.jsonl', [as_scoring(r, True) for r in training])]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    (output / 'eval.jsonl').write_bytes((base / 'eval.jsonl').read_bytes())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'training_rows':len(training), 'frozen_eval_rows':len(heldout), 'eval_sha256':manifest['eval_sha256'],
                      'new_examples':len(added), 'duplicate_audit':duplicate_audit}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=PROJECT_ROOT / 'data/openjeff_mixed_3000_qwen9b_v1/full')
    parser.add_argument('--supplement', type=Path, default=PROJECT_ROOT / 'data/complex_segments_150_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.supplement, args.output)


if __name__ == '__main__':
    main()
