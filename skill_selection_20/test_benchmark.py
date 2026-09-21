import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_benchmark as b


class PortableTests(unittest.TestCase):
    def test_twenty_questions_only(self):
        rows = b.read_jsonl(b.ROOT/'questions.jsonl')
        self.assertEqual(len(rows), 20)
        for row in rows:
            self.assertEqual(row['payload']['state'], {'question':row['question']})
            self.assertNotIn('baseline_skill', row['payload'])

    def test_sdk_cli_run_resume_report(self):
        calls = []
        class Client:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def system_one(self, **kwargs):
                calls.append((self.kwargs,kwargs))
                return SimpleNamespace(choices={'skill':SimpleNamespace(choice='query_param',confidence=1)})
        sdk=SimpleNamespace(TypeSafeClient=Client, Choice=lambda **kwargs:SimpleNamespace(**kwargs))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch.dict(sys.modules, {'typesafe_sdk':sdk}), patch.dict(b.os.environ,{},clear=True):
            with patch.object(sys,'argv',['run_benchmark.py','run','--out',tmp]):
                self.assertEqual(b.main(),0)
                self.assertEqual(len(calls),20)
                self.assertEqual(b.main(),0)
                self.assertEqual(len(calls),20)
            for config, request in calls:
                self.assertEqual(config,{'api_key':'local','base_url':'http://71.77.153.223:55332','model':'kev-latest'})
                self.assertEqual(set(request['state']),{'question'})
                self.assertEqual(set(request['questions']),{'skill'})
            with patch.object(sys,'argv',['run_benchmark.py','report','--out',tmp]):
                self.assertEqual(b.main(),0)
            summary=json.loads((Path(tmp)/'summary.json').read_text())
            self.assertEqual(summary['agreement_all'],1)
            self.assertIsNone(summary['local_accuracy'])

    def test_invalid_skill_and_sdk_failure(self):
        class Client:
            def __init__(self,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def system_one(self,**kwargs):
                return SimpleNamespace(choices={'skill':SimpleNamespace(choice='invalid')})
        sdk=SimpleNamespace(TypeSafeClient=Client,Choice=lambda **kwargs:kwargs)
        payload=b.read_jsonl(b.ROOT/'questions.jsonl')[0]['payload']
        with patch.dict(sys.modules,{'typesafe_sdk':sdk}):
            self.assertEqual(b.request_jev(payload,'local')['error'],'ValueError')
            with patch.object(Client,'system_one',side_effect=TimeoutError):
                self.assertEqual(b.request_jev(payload,'local')['error'],'TimeoutError')


if __name__ == '__main__':
    unittest.main()
