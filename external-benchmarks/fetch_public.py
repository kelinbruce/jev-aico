import urllib.request,urllib.parse,json,concurrent.futures,pathlib,hashlib
import argparse
ap=argparse.ArgumentParser(description='Download upstream public datasets; requires pyarrow.')
ap.add_argument('--raw',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parent/'raw')
root=ap.parse_args().raw
root.mkdir(parents=True,exist_ok=True)
specs=[('boolq','google/boolq','default','validation'),('paws','google-research-datasets/paws','labeled_final','test'),('squad2','rajpurkar/squad_v2','squad_v2','validation'),('pubmedqa','qiaojin/PubMedQA','pqa_labeled','train'),('helpsteer2','nvidia/HelpSteer2','default','validation'),('summeval','mteb/summeval','default','test')]
def fetch(s):
 name,repo,config,split=s
 url='https://datasets-server.huggingface.co/parquet?'+urllib.parse.urlencode({'dataset':repo})
 info=json.load(urllib.request.urlopen(url,timeout=60))
 files=[f for f in info['parquet_files'] if f['config']==config and f['split']==split]
 if not files: raise ValueError((name,[(f['config'],f['split']) for f in info['parquet_files']]))
 records=[]; provenance=[]
 import pyarrow.parquet as pq
 for i,f in enumerate(files):
  p=root/f'{name}-{i}.parquet'
  if not p.exists():
   with urllib.request.urlopen(f['url'],timeout=90) as r: p.write_bytes(r.read())
  records.extend(pq.read_table(p).to_pylist())
  provenance.append({'url':f['url'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 (root/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
 (root/f'{name}-download.json').write_text(json.dumps({'dataset':repo,'config':config,'split':split,'files':provenance},indent=2))
 return name,len(records)
failed=False
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 fs={pool.submit(fetch,s):s[0] for s in specs}
 for f in concurrent.futures.as_completed(fs):
  try: print(f.result(),flush=True)
  except Exception as e:
   failed=True
   print(fs[f],repr(e),flush=True)

raise SystemExit(int(failed))
