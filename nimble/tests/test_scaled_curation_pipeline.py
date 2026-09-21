"""Exercise a format repair through every unchanged evidence gate without API calls."""
import json
import unittest

from tests.test_scaled_evidence import fixture
from nimble.datasets.scale_evidence_dataset import construct_batch, scoring_exports
from nimble.datasets.dataset_io import canonical
from nimble.training.schema_data import as_scoring


class PipelineTests(unittest.TestCase):
    def test_structured_context_matches_reloaded_training_prompt(self):
        row = {'id': 'example', 'family': 'group', 'source_family': 'source',
               'input': {'state': {'z_last': ['A fact.'], 'a_first': {'z': 1, 'a': 2}},
                         'questions': {'decision': {'type': 'noul', 'instructions': 'Is it supported?',
                                                    'criteria': {'true': 'Supported', 'false': 'Refuted'}}}},
               'reference': {'target': True}}
        original = json.dumps(row)
        saved_row = json.loads(canonical(row))
        self.assertEqual(scoring_exports([row]), [as_scoring(saved_row, True)])
        self.assertEqual(scoring_exports([row]), scoring_exports([saved_row]))
        self.assertEqual(json.dumps(row), original)

    def test_one_format_repair_then_full_verification(self):
        source, plan = fixture()
        pair = {'left': 'The current item is listed in roster R.',
                'right': 'Every entry on roster R is qualified.',
                'negative_left': 'The current item is listed in roster R.',
                'negative_right': 'No entry on roster R is qualified.'}
        context = (pair['left'] + ' ' + pair['right'] + ' The current item is documented. '
                   'An administrative case note was added to the paper file on Tuesday '
                   'afternoon, and the records clerk retained the signed original.')
        audit = {k: True for k in ('atoms_are_atomic', 'policy_complete', 'focus_is_factual', 'assignments_are_realizable')}
        audit['rule_checks'] = [{'rule_index': i, 'sound': True} for i in range(2)]
        calls = []

        def stage(cls, schema, requests, name):
            calls.append(name)
            responses = {}
            for req in requests:
                payload = json.loads(req['payload_json'])
                if cls.__name__ == 'PairGenerator':
                    value = pair
                elif cls.__name__ == 'DocumentGenerator':
                    repairing = name.endswith('_document_repairs')
                    if repairing:
                        self.assertEqual(payload['required_state_kind'], 'str')
                        self.assertIn('Context kind', payload['structural_error'])
                    value = {'base_state_json': json.dumps(context if repairing else {'note': context}),
                             'focus_evidence': [{'path': [] if repairing else ['note'], 'text': pair[k]}
                                                for k in ('left', 'right')]}
                elif cls.__name__ == 'ContextReviewer':
                    value = {k: True for k in ('policy_preserved', 'question_bindings_preserved',
                        'evidence_is_two_factual_sentences', 'counterfactual_is_coherent', 'no_answer_leakage')}
                    value['explanation'] = 'Controlled test fixture.'
                else:
                    case = req['job_id'].rsplit(':', 1)[1]
                    focus = {'left': 'unknown', 'right': 'unknown', 'negative_sentence': 'unknown',
                             'positive_pair': 'supported', 'negative_pair': 'refuted', 'base': 'supported',
                             'counterfactual': 'refuted', 'remove_left': 'unknown', 'remove_right': 'unknown'}[case]
                    value = {'judgments': []}
                    for atom in payload['propositions']:
                        state = focus if atom['id'] == 'a' else 'supported'
                        value['judgments'].append({'atom_id': atom['id'], 'state': state,
                            'quotes': [] if state == 'unknown' else [payload['context']], 'reason': 'Controlled test fixture.'})
                responses[req['job_id']] = value
            return responses

        accepted, review = construct_batch([{'id': 'g', 'source_id': source['id'], 'variation': {'ordinal': 1}}],
            {source['id']: {'plan': plan, 'audit': audit}}, {source['id']: source}, stage, 0)
        self.assertEqual(len(accepted[0]['rows']), 2)
        self.assertEqual(calls.count('round_00_document_repairs'), 1)
        self.assertIn('round_00_full_checks', calls)
        self.assertEqual([r['stage'] for r in review], ['document_structure'])


if __name__ == '__main__':
    unittest.main()
