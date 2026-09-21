"""Coverage and cost-accounting checks for the fresh evaluation dataset."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from nimble.datasets.create_eval_dataset import eval_plans, generator_usage
from nimble.datasets.diversity_plan import DOMAINS


class FreshEvalTests(unittest.TestCase):
    def test_balanced_coverage_and_held_out_families(self):
        subjects = [{'domain': domain, 'domain_index': i, 'subtopic_index': j,
                     'group_id': f'{domain}-{j}'}
                    for i, (domain, _) in enumerate(DOMAINS) for j in range(5)]
        plans = eval_plans(subjects, 123)
        slots = [slot for row in plans for slot in json.loads(row['slots_json'])]
        self.assertEqual(len(slots), 1000)
        self.assertEqual({r['split'] for r in plans}, {'eval'})
        self.assertEqual(len({r['group_id'] for r in plans}), 50)
        self.assertFalse({r['group_id'] for r in subjects} & {r['group_id'] for r in plans})
        self.assertEqual(Counter(s['type'] for s in slots), {'choice': 334, 'noul': 333, 'score': 333})
        self.assertEqual(set(Counter(s['mechanism'] for s in slots).values()), {100})
        self.assertEqual(Counter(s['target'] for s in slots if s['type'] == 'noul'), {False: 167, True: 166})
        self.assertTrue(all(0 <= s['target'] < s['score_levels'] for s in slots if s['type'] == 'score'))

    def test_cost_includes_reasoning_output_and_deduplicates_response_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = {'raw_response': {'id': 'response-1'}, 'token_usage': {'input': 1000, 'output': 2000}}
            for i in range(2):
                (root / f'responses_{i}.jsonl').write_text(json.dumps(response) + '\n')
            usage = generator_usage(root)
            self.assertEqual(usage['responses'], 1)
            self.assertEqual(usage['output_tokens'], 2000)
            self.assertAlmostEqual(usage['estimated_usd'], .044)


if __name__ == '__main__':
    unittest.main()
