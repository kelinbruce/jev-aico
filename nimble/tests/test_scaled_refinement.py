import copy
import unittest
from collections import Counter

from tests.test_scaled_evidence import fixture
from nimble.datasets.scaled_evidence_refinement import select_refinements, meaning


class RefinementTests(unittest.TestCase):
    def test_refine_only_repeated_failures_in_underfilled_categories(self):
        source, plan = fixture()
        sources = {sid: {**source, 'id': sid} for sid in ('a', 'b', 'c')}
        plans = {sid: {'plan': plan} for sid in sources}
        attempts = Counter(a=2, b=2, c=1)
        successes = Counter(b=1)
        bucket = ('science', 'noul')
        self.assertEqual(select_refinements(plans, sources, attempts, successes,
                                           Counter({bucket: 2}), {bucket: 17}), ['a'])
        self.assertEqual(select_refinements(plans, sources, attempts, successes,
                                           Counter({bucket: 10}), {bucket: 17}), [])
        sources['a']['split'] = 'eval'
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            select_refinements(plans, sources, attempts, successes, Counter({bucket: 2}), {bucket: 17})

    def test_changing_only_justification_is_not_a_new_contrast(self):
        _, plan = fixture()
        changed = copy.deepcopy(plan)
        changed['rules'][0]['justification'] = 'A longer explanation of exactly the same rule.'
        self.assertEqual(meaning(plan), meaning(changed))
        changed['atoms'][0]['statement'] = 'The item is documented by a second independent record.'
        self.assertNotEqual(meaning(plan), meaning(changed))


if __name__ == '__main__':
    unittest.main()
