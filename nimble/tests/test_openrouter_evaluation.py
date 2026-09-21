import json
import unittest
from pathlib import Path
from nimble.evaluation.evaluate_openrouter import make_messages, parse_answer, summarize, charged_or_reserved
from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.scoring.parallel_schema import prepare_prompts

class CaptureTokenizer:
    all_special_ids = []
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return messages[0]['content'] + '\n' + messages[1]['content'] + '\n'
    def encode(self, text, **kwargs):
        return list(text.encode())

class EvaluationTests(unittest.TestCase):
    def test_prompt_matches_cuda_content_for_every_sample(self):
        path=Path(__file__).resolve().parents[1]/'data/eval.jsonl'
        for row in map(json.loads,path.read_text().splitlines()):
            tok=CaptureTokenizer();context,schema=adapt_input(row['input'])
            prepare_prompts(tok,context,schema,50000)
            messages,mapping=make_messages(row['input'])
            expected=tok.messages
            expected[1]['content']=expected[1]['content'].replace('__PARALLEL_FIELD_TARGET__','"decision"')
            self.assertEqual(messages,expected)
            self.assertEqual(len(mapping),len(schema['decision']['choices']))
    def test_strict_output_and_native_types(self):
        self.assertIs(parse_answer(' B\n','stop',{'A':False,'B':True},'noul'),True)
        self.assertEqual(parse_answer('A','stop',{'A':'2'},'score'),2)
        for text,reason in [('Answer: A','stop'),('A or B','stop'),('a','stop'),('C','stop'),('A','length')]:
            with self.assertRaises(ValueError):parse_answer(text,reason,{'A':'x','B':'y'},'choice')
    def test_failures_stay_in_denominator(self):
        rows=[{'type':'choice','valid':True,'correct':True,'teacher_agreement':True},
              {'type':'choice','valid':False,'correct':False,'teacher_agreement':False}]
        self.assertEqual(summarize(rows)['all']['accuracy'],.5)
    def test_unknown_cost_is_reserved(self):
        self.assertEqual(charged_or_reserved({'reservation_usd':.1}),.1)
        self.assertEqual(charged_or_reserved({'reservation_usd':.1,'response':{'usage':{'cost':.02}}}),.02)

if __name__=='__main__': unittest.main()
