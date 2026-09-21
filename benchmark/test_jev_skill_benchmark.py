import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jev_skill_benchmark as b


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = b.read_jsonl(b.DEFAULT_INPUT)

    def test_dataset_counts_and_decoding(self):
        rows = [b.convert(r, i) for i, r in enumerate(self.raw, 1)]
        self.assertEqual(len(rows), 41)
        self.assertEqual(sum(r['phase'] == 'initial' for r in rows), 20)
        self.assertEqual(sum(r['baseline_skill'] == b.NO_SKILL for r in rows), 19)
        self.assertEqual(rows[0]['question'], '列出"区域01"这个区域内室外小区的详细清单，包括小区标识和小区名称')

    def test_target_response_never_enters_payload(self):
        row = json.loads(json.dumps(self.raw[0]))
        expected = b.convert(row, 1)['payload']
        row['response_body']['choices'][0]['message']['tool_calls'][0]['function']['name'] = 'query_pm'
        row['extracted_answer'] = 'SECRET_TARGET_ANSWER'
        self.assertEqual(b.convert(row, 1)['payload'], expected)

    def test_history_and_question_mode(self):
        row = b.convert(self.raw[10], 11)
        self.assertEqual(row['payload']['state']['history'][-1]['role'], 'tool')
        self.assertEqual(row['baseline_skill'], b.NO_SKILL)
        question = b.convert(self.raw[0], 1, 'question')['payload']['state']
        self.assertEqual(set(question), {'question'})
        self.assertNotIn('query_param', question['question'])
        self.assertEqual(b.convert(self.raw[0], 1, 'full')['payload']['state']['messages'],
                         self.raw[0]['request_body']['messages'])

    def test_missing_task_and_multiple_skills_are_not_silently_scored(self):
        with self.assertRaises(ValueError):
            b.extract_question([{'role':'user', 'content':'no task'}])
        row = json.loads(json.dumps(self.raw[0]))
        row['response_body']['choices'][0]['message']['tool_calls'].append({'function':{'name':'query_pm'}})
        with self.assertRaises(ValueError):
            b.baseline_label(row)

    def test_accuracy_not_agreement_and_errors_count(self):
        rows = [b.convert(self.raw[0], 1), b.convert(self.raw[1], 2)]
        results = {rows[0]['id']: {'prediction':'query_param', 'latency_ms':123},
                   rows[1]['id']: {'prediction':None, 'latency_ms':100}}
        stats = b.metrics(rows, results, {})
        self.assertEqual(stats['agreement_all'], .5)
        self.assertEqual(stats['agreement_success_only'], 1)
        self.assertIsNone(stats['jev_accuracy'])
        gold = {r['id']:'query_pm' for r in rows}
        stats = b.metrics(rows, results, gold)
        self.assertEqual(stats['jev_accuracy'], 0)
        self.assertEqual(stats['baseline_accuracy'], 0)

    def test_real_response_contract_and_unknown_choice(self):
        payload = b.convert(self.raw[0], 1)['payload']
        raw = {'answers':{'skill':{'choice':'query_param', 'confidence':.9}}, 'model':'test'}
        with patch.object(b, 'urlopen', return_value=io.BytesIO(json.dumps(raw).encode())):
            result = b.request_jev(payload, 'fake-test-key')
        self.assertEqual(result['prediction'], 'query_param')
        raw['answers']['skill']['choice'] = 'invented'
        with patch.object(b, 'urlopen', return_value=io.BytesIO(json.dumps(raw).encode())):
            self.assertIsNotNone(b.request_jev(payload, 'fake-test-key')['error'])

    def test_retry_and_no_retry_on_auth_failure(self):
        payload = b.convert(self.raw[0], 1)['payload']
        raw = {'answers':{'skill':{'choice':'query_param'}}}
        error = HTTPError(b.API_URL, 429, 'rate limited', {'Retry-After':'0'}, None)
        with patch.object(b, 'urlopen', side_effect=[error, io.BytesIO(json.dumps(raw).encode())]), patch.object(b.time, 'sleep'):
            result = b.request_jev(payload, 'fake-test-key')
        self.assertEqual(result['attempts'], 2)
        with patch.object(b, 'urlopen', side_effect=HTTPError(b.API_URL, 401, 'auth', {}, None)) as request:
            result = b.request_jev(payload, 'fake-test-key')
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result['error'], 'HTTP 401')


if __name__ == '__main__':
    unittest.main()
