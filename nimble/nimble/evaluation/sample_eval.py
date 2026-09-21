"""Select a reproducible, domain/primitive-stratified evaluation subset without labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def sample(source, output, seed=20260917, per_domain=10):
    raw=source.read_bytes()
    rows=[json.loads(line) for line in raw.decode().splitlines() if line]
    rng=random.Random(seed); selected=[]
    for index,domain in enumerate(sorted({row['domain'] for row in rows})):
        for kind_index,kind in enumerate(('choice','noul','score')):
            pool=[r for r in rows if r['domain']==domain and r['input']['questions']['decision']['type']==kind]
            count=per_domain//3 + int((kind_index-index)%3 < per_domain%3)
            selected.extend(rng.sample(pool,count))
    selected.sort(key=lambda r:r['id'])
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected).encode()
    output.mkdir(parents=True,exist_ok=True)
    path=output/'all.jsonl'
    if path.exists() and path.read_bytes()!=payload:
        raise ValueError('Existing sample differs; choose a new directory')
    path.write_bytes(payload)
    manifest={'source':str(source),'source_sha256':hashlib.sha256(raw).hexdigest(),
              'dataset_sha256':hashlib.sha256(payload).hexdigest(),'seed':seed,
              'selection':'Stratified random sample without replacement; only domain and primitive type determine strata; model outputs and reference values are not inspected',
              'count':len(selected),'per_domain':per_domain,
              'domains':dict(Counter(r['domain'] for r in selected)),
              'primitives':dict(Counter(r['input']['questions']['decision']['type'] for r in selected)),
              'ids':[r['id'] for r in selected]}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260917)
    p.add_argument('--per-domain',type=int,default=10)
    args=p.parse_args()
    print(json.dumps(sample(args.source,args.output_dir,args.seed,args.per_domain),indent=2))

if __name__=='__main__':main()
