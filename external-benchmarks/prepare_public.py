#!/usr/bin/env python3
"""Rebuild the seven fixed Nimble public subsets from downloaded upstream JSONL.

Usage: python prepare_public.py --raw /path/to/downloads --nimble-root ../nimble
Raw file names: boolq.jsonl, paws.jsonl, squad2.jsonl, pubmedqa.jsonl,
helpsteer2.jsonl, summeval.jsonl. Original Hugging Face fields must be retained.
"""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
NAMES = ['boolq', 'paws', 'squad2', 'pubmedqa', 'helpsteer2', 'summeval-relevance', 'summeval-consistency']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def convert(row, source):
    assert set(row['input']['questions']) == {'decision'}
    result = copy.deepcopy(row['input'])
    result['questions']['decision'].update(label=row['reference']['target'], src=source)
    result['_meta'] = {'id': row['id'], 'source': source, 'family': row['family'],
                       'group_id': row['family'], 'variant': 'clean',
                       'human_reviewed': row['reference'].get('human_reviewed', False)}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', required=True, type=Path)
    ap.add_argument('--nimble-root', type=Path, default=ROOT.parent / 'nimble')
    args = ap.parse_args()
    sys.path.insert(0, str(args.nimble_root.resolve()))
    from nimble.datasets.public_benchmarks import jsonl_line
    for name in NAMES:
        module_name, _, subset = name.partition('-')
        module = importlib.import_module('nimble.datasets.public_sources.' + module_name)
        manifest_path = args.nimble_root / 'docs/assets/public-benchmarks/subsets' / (name + '-manifest.json')
        manifest = json.loads(manifest_path.read_text())
        wanted = set(manifest['ids'])
        selected = []
        for raw in module.rows(args.raw / (module_name + '.jsonl'), subset):
            row = module.record(raw, subset)
            if row and row['id'] in wanted:
                selected.append(row)
        selected.sort(key=lambda r: r['id'])
        assert len(selected) == len(wanted) == manifest['count'], name + ': missing/duplicate IDs'
        original = ''.join(jsonl_line(r) for r in selected).encode()
        assert digest(original) == manifest['dataset_sha256'], name + ': upstream subset checksum differs; do not claim matching baseline'
        out = ROOT / 'datasets' / name
        out.mkdir(parents=True, exist_ok=True)
        converted = ''.join(jsonl_line(convert(r, name)) for r in selected).encode()
        (out / 'data.jsonl').write_bytes(converted)
        shutil.copy2(manifest_path, out / 'upstream-manifest.json')
        metadata = {'name': name, 'records': len(selected), 'questions': len(selected),
                    'sha256': digest(converted), 'license': manifest['license'],
                    'source_url': manifest['source_url'],
                    'upstream_subset_sha256': digest(original), 'upstream_subset_verified': True,
                    'conversion': 'Keep state/instructions/criteria/order and labels; remove non-input annotations from request.',
                    'converter_sha256': digest(Path(module.__file__).read_bytes())}
        download = args.raw / (module_name + '-download.json')
        if download.exists(): metadata['download'] = json.loads(download.read_text())
        write_json(out / 'manifest.json', metadata)
        print(name, len(selected), 'original subset SHA-256 MATCH', flush=True)


if __name__ == '__main__':
    main()
