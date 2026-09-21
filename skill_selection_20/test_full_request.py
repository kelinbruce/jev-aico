import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import run_full_request as b


class FullRequestTests(unittest.TestCase):
    def setUp(self):
        self.body = {'model': 'original', 'verify_ssl': 'false', 'timeout': '120',
                     'temperature': 0.2, 'max_tokens': 2048, 'messages': [
            {'role': 'system', 'content': '完整系统规则'},
            {'role': 'user', 'content': '完整用户输入'},
            {'role': 'assistant', 'tool_calls': [{'id': 'x', 'function': {'name': 'query'}}]},
            {'role': 'tool', 'tool_call_id': 'x', 'content': '历史结果'}],
            'tools': [{'type': 'function', 'function': {'name': 'query', 'description': '查询',
                       'parameters': {'type': 'object', 'properties': {'area': {'type': 'string'}}}}}],
            'tool_choice': 'auto'}

    def test_lossless_and_no_answer_leakage(self):
        payload = b.build_payload({'request_body': self.body, 'response_body': 'SECRET_ANSWER'}, 'kev-latest')
        self.assertEqual(payload['state']['request_body'],
                         {name: self.body[name] for name in ('messages', 'tools')})
        self.assertNotIn('SECRET_ANSWER', json.dumps(payload))
        payload['state']['request_body']['messages'][0]['content'] = 'changed'
        self.assertEqual(self.body['messages'][0]['content'], '完整系统规则')

    def test_real_source_and_wire_payload(self):
        with b.DEFAULT_INPUT.open() as f:
            record = json.loads(next(f))
        payload = b.build_payload(record, 'kev-latest')
        response = io.BytesIO(b'{"answers":{"skill":{"choice":"query_param","confidence":0.9}}}')
        response.status = 200
        response.headers = {}
        with patch.object(b, 'urlopen', return_value=response) as send:
            result = b.call_service(payload, 'http://localhost:1234', 'local', 60)
        req = send.call_args.args[0]
        self.assertEqual(req.full_url, 'http://localhost:1234/v1/systemone')
        self.assertEqual(json.loads(req.data)['state']['request_body'],
                         {name: record['request_body'][name] for name in ('messages', 'tools')})
        self.assertEqual(json.loads(req.data)['model'], 'kev-latest')
        self.assertEqual(result['prediction'], 'query_param')

    def test_initial_selection_matches_original_twenty(self):
        with b.DEFAULT_INPUT.open() as src:
            records = [json.loads(line) for line in src if line.strip()]
        selected = [r for r in records if b.is_initial_request(r)]
        with (b.ROOT/'questions.jsonl').open() as src:
            expected = {json.loads(line)['id'] for line in src if line.strip()}
        self.assertEqual(len(selected), 20)
        self.assertEqual({r['proxy_request_id'] for r in selected}, expected)
        self.assertFalse(b.is_initial_request({'request_body': self.body}))

    def test_http_error_details(self):
        error = HTTPError('http://localhost', 503, 'unavailable', {'x-typesafe-request-id': 'trace'},
                          io.BytesIO(b'{"error":"backend unavailable secret-key"}'))
        with patch.object(b, 'urlopen', side_effect=error) as send:
            result = b.call_service(b.build_payload({'request_body': self.body}, 'kev-latest'),
                                    'http://localhost', 'secret-key', 60)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(result['http_status'], 503)
        self.assertEqual(result['request_id'], 'trace')
        self.assertNotIn('secret-key', result['error_detail'])


if __name__ == '__main__':
    unittest.main()
