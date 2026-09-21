import json
from pathlib import Path
import tempfile
import unittest
from nimble.evaluation.sample_eval import sample

class SampleTests(unittest.TestCase):
    def test_references_do_not_influence_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=[]
            for domain in ('a','b','c'):
                for kind in ('choice','noul','score'):
                    for i in range(7):
                        rows.append({'id':f'{domain}-{kind}-{i}','domain':domain,
                                     'input':{'questions':{'decision':{'type':kind}}},'reference':{'target':i}})
            source=root/'input.jsonl'
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            before=sample(source,root/'before')
            for row in rows: row['reference']['target']='changed'
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            after=sample(source,root/'after')
            self.assertEqual(before['ids'],after['ids'])
            self.assertEqual(before['domains'],{'a':10,'b':10,'c':10})
            self.assertEqual(before['primitives'],{'choice':10,'noul':10,'score':10})
            self.assertEqual(len(set(before['ids'])),30)
    def test_existing_different_sample_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'input.jsonl'; out=root/'out';out.mkdir()
            (out/'all.jsonl').write_text('existing')
            source.write_text(json.dumps({'id':'x','domain':'a','input':{'questions':{'decision':{'type':'choice'}}}}))
            with self.assertRaises(ValueError):sample(source,out,per_domain=1)
            self.assertEqual((out/'all.jsonl').read_text(),'existing')

if __name__=='__main__':unittest.main()
