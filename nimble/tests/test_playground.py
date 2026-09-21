import concurrent.futures
import json
import tempfile
import threading
import unittest
from pathlib import Path

from nimble.playground.server import Playground, normalized_answer, sample_examples, validate_input
from nimble.playground.mac_worker import normalized_field


def example():
    return {'state': 'The customer requests a refund.', 'questions': {'decision': {
        'type': 'choice', 'instructions': 'Route the request.',
        'criteria': {'billing': 'Payment issues', 'technical': 'Software issues'}}}}


class PlaygroundTests(unittest.TestCase):
    def test_fast_result_is_emitted_while_mac_is_still_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Playground.__new__(Playground)
            app.lookup = {'case': {'input': example(), 'reference': 'billing'}}
            app.lock = threading.Lock()
            app.saved = []
            app.history = Path(directory)/'results.jsonl'
            release_mac, jev_received = threading.Event(), threading.Event()
            events = []

            def invoke(name, value):
                if name == 'mac':
                    if not release_mac.wait(5):
                        raise TimeoutError('Test did not release Mac')
                return {'prediction': 'billing', 'probabilities': {'billing': 1.}}

            def receive(event):
                events.append(json.loads(json.dumps(event)))
                if event['type'] == 'result' and event['provider'] == 'jev':
                    jev_received.set()

            app.invoke = invoke
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool, concurrent.futures.ThreadPoolExecutor(max_workers=1) as runner:
                app.pool = pool
                task = runner.submit(app.compare, 'case', example(), receive)
                try:
                    self.assertTrue(jev_received.wait(2), 'Jev was held behind the slow Mac')
                    self.assertFalse(task.done())
                    self.assertFalse(app.history.exists())
                    self.assertEqual(events[0]['type'], 'start')
                    self.assertEqual(events[0]['record']['results'], {})
                    self.assertFalse(any(e.get('provider') == 'mac' for e in events))
                finally:
                    release_mac.set()
                result = task.result(timeout=3)
            self.assertEqual(events[-1]['type'], 'complete')
            self.assertTrue(result['agreement'])
            self.assertEqual(len(app.history.read_text().splitlines()), 1)
            self.assertFalse(app.lock.locked())

    def test_mac_score_and_boolean_keep_typed_predictions(self):
        score = normalized_field({'value': '2', 'scores': {'0': .1, '1': .2, '2': .7}}, {}, ['decision'])
        self.assertEqual(score['prediction'], 2)
        self.assertIsInstance(score['prediction'], int)
        boolean = normalized_field({'value': False, 'scores': {'false': .8, 'true': .2}}, {}, [])
        self.assertIs(boolean['prediction'], False)
        self.assertEqual(boolean['probabilities'], {'false': .8, 'true': .2})

    def test_inference_contract_rejects_reference_fields(self):
        self.assertEqual(validate_input(example()), example())
        with self.assertRaises(ValueError):
            validate_input({**example(), 'reference': 'billing'})

    def test_candidate_limit(self):
        value = example()
        value['questions']['decision']['criteria'] = {str(i): str(i) for i in range(27)}
        with self.assertRaises(ValueError):
            validate_input(value)

    def test_weighted_score_and_tie_order(self):
        result = normalized_answer({'0': .5, '1': .5}, 'score')
        self.assertEqual(result['prediction'], 0)
        self.assertEqual(result['expected_score'], .5)
        with self.assertRaises(ValueError):
            normalized_answer({'0': float('nan')}, 'score')

    def test_edited_input_does_not_inherit_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Playground.__new__(Playground)
            app.lookup = {'case': {'input': example(), 'reference': 'billing'}}
            app.lock = threading.Lock()
            app.saved = []
            app.history = Path(directory)/'results.jsonl'
            seen = []
            def invoke(name, value):
                seen.append(value)
                return {'prediction': 'billing', 'probabilities': {'billing': 1., 'technical': 0.}}
            app.invoke = invoke
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                app.pool = pool
                original = app.compare('case', example())
                self.assertEqual(set(original['results']), {'nimble', 'mac', 'jev'})
                self.assertTrue(original['agreement'])
                self.assertTrue(original['results']['nimble']['correct'])
                edited = app.compare('case', {**example(), 'state': 'Changed context'})
                self.assertFalse(edited['reference_available'])
                self.assertNotIn('reference', edited)
                self.assertNotIn('correct', edited['results']['nimble'])
                app.invoke = lambda name, value: {'prediction': 'technical' if name == 'mac' else 'billing'}
                self.assertFalse(app.compare('case', example())['agreement'])
                app.invoke = lambda name, value: {'error': 'Offline'} if name == 'mac' else {'prediction': 'billing'}
                self.assertIsNone(app.compare('case', example())['agreement'])
                app.providers = ('nimble', 'jev')
                pair = app.compare('case', example())
                self.assertEqual(set(pair['results']), {'nimble', 'jev'})
                self.assertTrue(pair['agreement'])
                app.providers = ('mac', 'jev')
                app.invoke = invoke
                pair = app.compare('case', example())
                self.assertEqual(set(pair['results']), {'mac', 'jev'})
                self.assertTrue(pair['agreement'])
            self.assertTrue(all(set(v) == {'state', 'questions'} for v in seen))

    def test_sample_is_reproducible_and_balanced(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for kind in ('choice', 'noul', 'score'):
                for i in range(15):
                    value = example()
                    value['questions']['decision']['type'] = kind
                    rows.append({'id': kind+str(i), 'input': value, 'reference': {'target': 'billing'}})
            path = Path(directory)/'data.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            first, second = sample_examples(path), sample_examples(path)
            self.assertEqual(first, second)
            self.assertEqual(len({r['id'] for r in first}), 30)
            for kind in ('choice', 'noul', 'score'):
                self.assertEqual(sum(r['kind'] == kind for r in first), 10)


if __name__ == '__main__':
    unittest.main()
