"""Offline scoring tests and a local mock-HTTP integration test. No real model is called."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
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

    def test_mock_http_all_suites(self):
        captured=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
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
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'run'
                p=subprocess.run([sys.executable,str(ROOT/'run_benchmarks.py'),'--limit','1',
                    '--base-url',f'http://127.0.0.1:{server.server_port}','--model','mock','--out',str(out)],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                comparison=json.loads((out/'comparison.json').read_text())
                self.assertEqual(len(comparison),10)
                for row in comparison.values():
                    self.assertFalse(row['full_run_comparable'])
                    self.assertEqual(row['summary']['errors'],0)
                    for b in row['published_baselines']:self.assertIsNone(b['difference'])
                self.assertTrue((out/'REPORT.md').exists())
            self.assertEqual(len(captured),10)
            for path,request in captured:
                self.assertEqual(path,'/v1/systemone')
                self.assertEqual(set(request),{'model','state','questions'})
                for q in request['questions'].values():
                    self.assertNotIn('label',q); self.assertNotIn('src',q)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__=='__main__':unittest.main()
