import copy
import unittest

from tests.test_evidence_curation import fixture
from nimble.datasets.curate_evidence_counterfactuals import (
    assess_revision, eligible_groups, revised_context,
)


class CounterfactualCurationTests(unittest.TestCase):
    def prepare(self):
        _, _, group, audit, facts = fixture()
        group['question']['criteria']['denied'] = 'Approval is recorded, but Mira does not handle Q.'
        group['spec']['rules'].append({'when': [{'atom_id': 'a1', 'state': 'refuted'},
                                              {'atom_id': 'a2', 'state': 'supported'}],
                                      'target': 'denied', 'justification': 'Explicit negation.'})
        audit['rule_checks'].append({'rule_index': 2, 'sound': True, 'reason': 'Explicit criterion.'})
        revision = {'sentence_index': 0, 'replacement': 'Mira handles only accounts assigned to the day desk.',
                    'explanation': 'Change the assignment.'}
        review = {'policy_preserved': True, 'single_coherent_factual_change': True,
                  'no_answer_instruction': True, 'explanation': 'A factual change.'}
        gid = group['id']
        quotes = [revision['replacement'], group['spec']['focus_evidence'][1]['text']]
        cf = {gid + ':' + case: {'judgments': [{'atom_id': 'a1', 'state': state,
                    'quotes': quotes if state == 'refuted' else [], 'reason': 'Assessed evidence.'}]}
              for case, state in [('changed_sentence', 'unknown'), ('counter_pair', 'refuted'),
                                  ('counterfactual', 'refuted')]}
        cf[gid + ':counterfactual']['judgments'].append({'atom_id': 'a2', 'state': 'supported',
                        'quotes': ['Approval was recorded by compliance.'], 'reason': 'Recorded.'})
        return group, audit, facts, revision, review, cf

    def test_explicit_refutation_derives_different_schema_answer(self):
        group, audit, facts, revision, review, cf = self.prepare()
        eligible, _ = eligible_groups({group['id']: group}, {group['id']: audit}, facts)
        self.assertEqual(len(eligible), 1)
        base, counter = assess_revision(group, revision, review, cf, {group['id']: audit}, facts)
        self.assertEqual(base['reference']['target'], 'verified')
        self.assertEqual(counter['reference']['target'], 'denied')
        original = copy.deepcopy(group['states']['base'])
        new, _ = revised_context(group, revision)
        self.assertEqual(group['states']['base'], original)
        self.assertIn('Approval was recorded by compliance.', new)

    def test_unknown_and_single_sentence_shortcuts_rejected(self):
        group, audit, facts, revision, review, cf = self.prepare()
        cf[group['id'] + ':counter_pair']['judgments'][0].update(state='unknown', quotes=[])
        with self.assertRaisesRegex(ValueError, 'explicit refutation'):
            assess_revision(group, revision, review, cf, {group['id']: audit}, facts)
        group, audit, facts, revision, review, cf = self.prepare()
        cf[group['id'] + ':changed_sentence']['judgments'][0].update(state='refuted', quotes=[revision['replacement']])
        with self.assertRaisesRegex(ValueError, 'alone'):
            assess_revision(group, revision, review, cf, {group['id']: audit}, facts)

    def test_failed_parent_or_edit_review_cannot_enter_training(self):
        group, audit, facts, revision, review, cf = self.prepare()
        audit['rule_checks'][0]['sound'] = False
        self.assertFalse(eligible_groups({group['id']: group}, {group['id']: audit}, facts)[0])
        review['policy_preserved'] = False
        with self.assertRaisesRegex(ValueError, 'audit failed'):
            assess_revision(group, revision, review, cf, {group['id']: audit}, facts)


if __name__ == '__main__':
    unittest.main()
