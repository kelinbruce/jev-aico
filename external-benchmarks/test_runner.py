"""Offline scoring tests and a local mock-HTTP integration test. No real model is called."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typesafe_sdk._core.response_types import SystemOneResponse

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runner', ROOT.parent / 'decision-v7/run_test.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def response(answers):
    return SystemOneResponse.model_validate({'model':'mock', 'answers':answers,
        'usage': {'input_tokens':1, 'output_tokens':1}})


class Scoring(unittest.TestCase):
    def test_score_mode_not_rounded_expectation(self):
        r = response({'d': {'type':'score','score':1.1,'confidence':0.1,
                           'legend':{0:'low',1:'medium',2:'high'},
                           'probabilities':{0:.35,1:.20,2:.45}}})
        self.assertEqual(runner.predict(r,'d',{'type':'score','criteria':['low','medium','high']}),2)

    def test_noul_tie_and_error_denominator(self):
        r = response({'d':{'type':'noul','noul':.5}})
        self.assertFalse(runner.predict(r,'d',{'type':'noul'}))
        def q(correct,error,variant,gold):
            return dict(type='choice',source='a',family='f',gold=gold,correct=correct,error=error,variant=variant)
        result=runner.summarize([{'latency_ms':1,'questions':[
            q(True,None,'clean','yes'),q(False,'ConnectionError','clean','yes'),q(False,None,'permuted','no')]}])
        self.assertAlmostEqual(result['accuracy'],1/3)
        self.assertEqual(result['clean']['questions'],2)
        self.assertEqual(result['clean']['accuracy'],.5)
        self.assertEqual(result['family_balanced_accuracy'],.25)

    def test_no_labels_in_any_request(self):
        n=0
        for path in (ROOT/'datasets').glob('*/data.jsonl'):
            for line in path.read_text(encoding='utf-8').split('\n'):
                if not line.strip(): continue
                row=json.loads(line)
                payload=runner.make_questions(row)
                for qid,q in payload.items():
                    self.assertNotIn('label',q.model_dump())
                    self.assertNotIn('src',q.model_dump())
                    self.assertEqual(q.instructions,row['questions'][qid]['instructions'])
                    self.assertEqual(q.criteria,row['questions'][qid]['criteria'])
                n+=1
        self.assertEqual(n,2308)

    def test_first_error_stops_before_concurrent_batch(self):
        called=[]
        def execute(item):
            called.append(item)
            return {"request_error":"ConnectionError"}
        self.assertEqual(len(list(runner.run_records(range(10),execute,4))),1)
        self.assertEqual(called,[0])

    def test_invalid_workers(self):
        for script in (ROOT/'run_benchmarks.py', ROOT.parent/'decision-v7/run_test.py'):
            for workers in ('0','-2'):
                p=subprocess.run([sys.executable,str(script),'--workers',workers,'--dry-run'],capture_output=True,text=True)
                self.assertEqual(p.returncode,2)
                self.assertIn('--workers must be positive',p.stderr)

    def test_mock_http_all_suites(self):
        captured=[]
        lock=threading.Lock()
        active=0
        peak=0
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(.05)
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append((self.path,payload))
                answers={}
                for key,q in payload['questions'].items():
                    kind=q['type']
                    if kind=='noul': a={'type':kind,'noul':.7}
                    elif kind=='choice':
                        keys=list(q['criteria'])
                        a={'type':kind,'choice':keys[0],'confidence':1.,'probabilities':{k:float(k==keys[0]) for k in keys}}
                    else:
                        keys=list(range(len(q['criteria'])))
                        a={'type':kind,'score':0.,'confidence':1.,'legend':dict(enumerate(q['criteria'])),
                           'probabilities':{k:float(k==0) for k in keys}}
                    answers[key]=a
                data=json.dumps({'model':'mock','usage':{'input_tokens':1,'output_tokens':1},'answers':answers}).encode()
                self.send_response(200); self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
                with lock: active -= 1
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            summaries=[]
            for workers in (1,4):
                with lock: peak=0
                with tempfile.TemporaryDirectory() as tmp:
                    out=Path(tmp)/'run'
                    p=subprocess.run([sys.executable,str(ROOT/'run_benchmarks.py'),'--limit','8',
                        '--workers',str(workers),
                        '--base-url',f'http://127.0.0.1:{server.server_port}','--model','mock','--out',str(out)],capture_output=True,text=True)
                    self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                    comparison=json.loads((out/'comparison.json').read_text())
                    self.assertEqual(len(comparison),10)
                    summaries.append({})
                    for name,row in comparison.items():
                        self.assertFalse(row['full_run_comparable'])
                        self.assertEqual(row['summary']['errors'],0)
                        self.assertEqual(row['summary']['records'],8)
                        summaries[-1][name]=row['summary']['accuracy']
                        saved=[json.loads(l) for l in (out/name/'results.jsonl').read_text().splitlines()]
                        self.assertEqual(sorted(r['index'] for r in saved),list(range(1,9)))
                        self.assertEqual(len({r['id'] for r in saved}),8)
                        self.assertEqual(json.loads((out/name/'run.json').read_text())['workers'],workers)
                        for baseline in row['published_baselines']:self.assertIsNone(baseline['difference'])
                    self.assertTrue((out/'REPORT.md').exists())
                self.assertLessEqual(peak,workers)
                if workers==1:self.assertEqual(peak,1)
                else:self.assertGreater(peak,1)
            self.assertEqual(summaries[0],summaries[1])
            self.assertEqual(len(captured),160)
            for path,request in captured:
                self.assertEqual(path,'/v1/systemone')
                self.assertEqual(set(request),{'model','state','questions'})
                for q in request['questions'].values():
                    self.assertNotIn('label',q); self.assertNotIn('src',q)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__=='__main__':unittest.main()
