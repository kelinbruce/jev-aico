import copy
import itertools
import unittest

from nimble.datasets.scaled_evidence import normalize_plan, validate_plan, quotas, make_rows
from nimble.datasets.evidence_curation import decide


def fixture():
    source = {'id': 's', 'split': 'train', 'family': 'f', 'domain': 'science',
              'input': {'state': 'Original synthetic context.', 'questions': {'decision': {
                  'type': 'noul', 'instructions': 'Is the current item both qualified and documented?',
                  'criteria': {'true': 'Both conditions hold.', 'false': 'A condition is explicitly false.'}}}}}
    plan = {'atoms': [{'id': 'a', 'statement': 'The item is qualified.'},
                      {'id': 'b', 'statement': 'The item is documented.'}],
            'focus_atom': 'a', 'policy_evidence': [], 'rules': [
                {'when': [{'atom_id': 'a', 'state': 'supported'}, {'atom_id': 'b', 'state': 'supported'}],
                 'target': 'true', 'justification': 'Both.'},
                {'when': [{'atom_id': 'a', 'state': 'refuted'}], 'target': 'false', 'justification': 'Fails.'}],
            'base_states': [{'atom_id': 'a', 'state': 'supported'}, {'atom_id': 'b', 'state': 'supported'}],
            'counter_states': [{'atom_id': 'a', 'state': 'refuted'}, {'atom_id': 'b', 'state': 'supported'}]}
    return source, plan


class ScaledEvidenceTests(unittest.TestCase):
    def test_citation_normalization_requires_literal_original_quotes(self):
        source, plan = fixture()
        plan['policy_evidence'] = [
            {'path': ['input', 'state'], 'text': 'Original synthetic context.'},
            {'path': ['questions', 'decision', 'criteria', 'true'], 'text': 'Both conditions hold.'}]
        result = normalize_plan(plan, source)
        self.assertEqual(result['policy_evidence'], [{'path': [], 'text': 'Original synthetic context.'}])
        self.assertEqual(plan['policy_evidence'][0]['path'], ['input', 'state'])
        plan['policy_evidence'][1]['text'] = 'A fabricated quotation.'
        with self.assertRaises(ValueError):
            normalize_plan(plan, source)

    def test_partial_rules_do_not_treat_unknown_as_false(self):
        source, plan = fixture()
        validate_plan(plan, source)
        self.assertIsNone(decide(plan, {'a': 'unknown', 'b': 'supported'}, source['input']['questions']['decision']))

    def test_conflicting_partial_rules_rejected(self):
        source, plan = fixture()
        plan['rules'][1]['when'] = [{'atom_id': 'b', 'state': 'supported'}]
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            validate_plan(plan, source)

    def test_twenty_atoms_without_exponential_enumeration(self):
        source, plan = fixture()
        for i in range(18):
            atom = 'x' + str(i)
            plan['atoms'].append({'id': atom, 'statement': atom + ' is present.'})
            for key in ('base_states', 'counter_states'):
                plan[key].append({'atom_id': atom, 'state': 'supported'})
        validate_plan(plan, source)

    def test_heldout_or_nonlocal_change_rejected(self):
        source, plan = fixture()
        source['split'] = 'eval'
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            validate_plan(plan, source)
        source['split'] = 'train'
        plan['counter_states'][1]['state'] = 'refuted'
        with self.assertRaisesRegex(ValueError, 'Only focus'):
            validate_plan(plan, source)

    def test_balanced_thousand_keeps_complete_pairs(self):
        sources = [{'domain': f'd{i}', 'split': 'train', 'input': {'questions': {'decision': {'type': t}}}}
                   for i, t in itertools.product(range(10), ('choice', 'noul', 'score'))]
        counts = quotas(sources, 1000)
        self.assertEqual(sum(counts.values()), 500)
        for i in range(10):
            self.assertEqual(sum(n for (d, _), n in counts.items() if d == f'd{i}'), 50)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        with self.assertRaises(ValueError):
            quotas(sources, 999)

    def test_full_context_redundancy_and_nonfocus_changes_rejected(self):
        source, plan = fixture()
        group = {'id': 'g', 'spec': plan, 'question': source['input']['questions']['decision']}
        audit = {k: True for k in ('atoms_are_atomic', 'policy_complete', 'focus_is_factual', 'assignments_are_realizable')}
        audit['rule_checks'] = [{'rule_index': i, 'sound': True} for i in range(2)]
        context_audit = {k: True for k in ('policy_preserved', 'question_bindings_preserved',
                                         'evidence_is_two_factual_sentences', 'counterfactual_is_coherent', 'no_answer_leakage')}
        pairs = {k: {'a': s} for k, s in {'left': 'unknown', 'right': 'unknown',
                 'negative_sentence': 'unknown', 'positive_pair': 'supported', 'negative_pair': 'refuted'}.items()}
        full = {'base': {'a': 'supported', 'b': 'supported'}, 'counterfactual': {'a': 'refuted', 'b': 'supported'},
                'remove_left': {'a': 'supported'}, 'remove_right': {'a': 'unknown'}}
        with self.assertRaisesRegex(ValueError, 'necessity'):
            make_rows(group, audit, {}, pairs, context_audit, full)
        full['remove_left']['a'] = 'unknown'
        full['counterfactual']['b'] = 'refuted'
        with self.assertRaisesRegex(ValueError, 'audited contrast'):
            make_rows(group, audit, {}, pairs, context_audit, full)


if __name__ == '__main__':
    unittest.main()
