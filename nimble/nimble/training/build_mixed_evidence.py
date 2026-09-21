"""Combine audited evidence releases while retaining the existing family holdouts."""
import argparse
import json
from pathlib import Path

from nimble.datasets.scaled_evidence import audit_release
from nimble.training.evidence_data import validate_evidence_data
from nimble.training.schema_data import as_scoring, fingerprint, read_rows, validate_separation


def split_rows(rows, outer_families, inner_families):
    if set(outer_families) & set(inner_families):
        raise ValueError('Outer and inner source families overlap')
    full = [r for r in rows if r['source_family'] not in outer_families]
    outer = [r for r in rows if r['source_family'] in outer_families]
    inner_train = [r for r in full if r['source_family'] not in inner_families]
    inner_eval = [r for r in full if r['source_family'] in inner_families]
    for train, evaluation in [(full, outer), (inner_train, inner_eval)]:
        validate_separation(train, [{**r, 'split':'validation'} for r in evaluation])
    return full, outer, inner_train, inner_eval


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,required=True)
    parser.add_argument('--release',type=Path,action='append',required=True)
    parser.add_argument('--inner-split-manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a new output directory')
    original_eval=read_rows(args.base_dir/'eval.jsonl')
    outer_families={r['source_family'] for r in original_eval}
    inner_families=set(json.loads(args.inner_split_manifest.read_text())['evaluation_source_families'])
    rows=[];lineage=[]
    for path in [args.base_dir,*args.release]:
        raw=read_rows(path/'train.jsonl');manifest=json.loads((path/'manifest.json').read_text())
        assert fingerprint(raw)==manifest['train_sha256']
        validate_evidence_data(raw,manifest)
        rows.extend(raw)
        lineage.append({'directory':str(path.resolve()),'rows':len(raw),'training_fingerprint':fingerprint(raw),
                        'manifest_fingerprint':fingerprint(manifest),
                        'generator_model':manifest.get('generator_model',manifest.get('model')),
                        'outer_family_rows':sum(r['source_family'] in outer_families for r in raw)})
    if len({r['id'] for r in rows})!=len(rows) or len({fingerprint(r['input']) for r in rows})!=len(rows):
        raise ValueError('Duplicate ID or input in combined release')
    source_path=Path(json.loads((args.base_dir/'manifest.json').read_text())['source'])
    sources=read_rows(source_path)
    full, new_outer, inner_train, inner_eval=split_rows(rows,outer_families,inner_families)
    outer=[*original_eval,*[{**r,'split':'validation'} for r in new_outer]]
    assert len({r['id'] for r in outer})==len(outer)
    validate_separation(full,outer)
    # Audit original held-out examples as their source release's training rows.
    outer_as_train=[{**r,'split':'train'} for r in outer]
    audit_release([*full,*outer_as_train],sources)
    args.output_dir.mkdir(parents=True)
    def write_jsonl(path,values):
        path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in values))
    for name,training,evaluation in [('inner',inner_train,[{**r,'split':'validation'} for r in inner_eval]),
                                     ('full',full,outer)]:
        directory=args.output_dir/name;directory.mkdir()
        manifest={'pipeline_version':'evidence-curation-v3','source':str(source_path.resolve()),
                  'source_sha256':fingerprint(sources),'training':audit_release(training,sources),
                  'train_sha256':fingerprint(training),'target':len(training),'reference_human_reviewed':False,
                  'lineage':lineage,'notes':['Mixed Sol, Luna and Sonnet evidence releases; each certificate reconstructed.',
                    'All source families of the original outer holdout remain outside training.',
                    'Sonnet includes 22 rows drafted using same-source Luna illustrations; releases are not independent.']}
        validate_evidence_data(training,manifest)
        for filename,values in [('train.jsonl',training),('eval.jsonl',evaluation),
                                ('train_scoring.jsonl',[as_scoring(r,True) for r in training])]:
            write_jsonl(directory/filename,values)
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    write_jsonl(args.output_dir/'original_eval.jsonl',original_eval)
    write_jsonl(args.output_dir/'new_heldout.jsonl',[{**r,'split':'validation'} for r in new_outer])
    split={'raw_combined_training_release_rows':len(rows),'new_release_rows':sum(x['rows'] for x in lineage[1:]),
           'full_training_rows':len(full),'inner_training_rows':len(inner_train),'inner_validation_rows':len(inner_eval),
           'outer_evaluation_rows':len(outer),'original_outer_rows':len(original_eval),'new_outer_rows':len(new_outer),
           'outer_source_families':sorted(outer_families),'inner_source_families':sorted(inner_families),
           'full_training_fingerprint':fingerprint(full),'inner_training_fingerprint':fingerprint(inner_train),
           'inner_validation_fingerprint':fingerprint([{**r,'split':'validation'} for r in inner_eval]),
           'outer_evaluation_fingerprint':fingerprint(outer),'original_outer_fingerprint':fingerprint(original_eval),
           'new_outer_fingerprint':fingerprint([{**r,'split':'validation'} for r in new_outer]),
           'selection':'Reuse previous source-family assignments; no model scores enter the split',
           'lineage':lineage}
    (args.output_dir/'split_manifest.json').write_text(json.dumps(split,indent=2)+'\n')
    (args.output_dir/'README.md').write_text(
        '# Mixed evidence data for Qwen3.5-9B\n\n'
        f"{len(full)} full training rows; {len(inner_train)} inner training / {len(inner_eval)} inner validation. "
        f"Outer evaluation contains the original {len(original_eval)} rows plus {len(new_outer)} new rows from those same held-out source families.\n\n"
        'Whole source families and contrast pairs remain together. Original releases are unchanged. '
        'Labels are synthetic and model-checked, not human-reviewed. Tokens must be exported with the pinned 9B tokenizer.\n')
    print(json.dumps({k:v for k,v in split.items() if k!='lineage'},indent=2))


if __name__=='__main__':
    main()
