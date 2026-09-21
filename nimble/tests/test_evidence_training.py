import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_scaled_evidence import fixture
from nimble.datasets.contrastive_data import fingerprint
from nimble.datasets.scaled_evidence import build_group, make_rows, audit_release
from nimble.training.evidence_data import validate_evidence_data


class EvidenceTrainingTests(unittest.TestCase):
    def examples(self):
        source, plan = fixture()
        pair = {'left': 'The current item is listed in roster R.',
                'right': 'Every entry on roster R is qualified.',
                'negative_left': 'The current item is listed in roster R.',
                'negative_right': 'No entry on roster R is qualified.'}
        state = (pair['left'] + ' ' + pair['right'] + ' The current item is documented. '
                 'The administrative review was recorded on Tuesday afternoon, and the original '
                 'paperwork was filed with the desk that handles this particular case.')
        doc = {'base_state_json': json.dumps(state), 'focus_evidence': [
            {'path': [], 'text': pair['left']}, {'path': [], 'text': pair['right']}]}
        group = build_group('g', source, plan, doc, pair)
        audit = {k: True for k in ('atoms_are_atomic', 'policy_complete', 'focus_is_factual', 'assignments_are_realizable')}
        audit['rule_checks'] = [{'rule_index': i, 'sound': True} for i in range(2)]
        context = {k: True for k in ('policy_preserved', 'question_bindings_preserved',
                                   'evidence_is_two_factual_sentences', 'counterfactual_is_coherent', 'no_answer_leakage')}
        pairs = {k: {'a': s} for k, s in {'left': 'unknown', 'right': 'unknown',
                 'negative_sentence': 'unknown', 'positive_pair': 'supported', 'negative_pair': 'refuted'}.items()}
        full = {'base': {'a': 'supported', 'b': 'supported'}, 'counterfactual': {'a': 'refuted', 'b': 'supported'},
                'remove_left': {'a': 'unknown'}, 'remove_right': {'a': 'unknown'}}
        return source, make_rows(group, audit, pair, pairs, context, full)

    def test_reconstructs_every_variant_and_rejects_context_or_certificate_changes(self):
        source, rows = self.examples()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'all.jsonl'
            path.write_text(json.dumps(source) + '\n')
            manifest = {'pipeline_version': 'evidence-curation-v3', 'source': str(path),
                        'source_sha256': fingerprint([source]), 'training': audit_release(rows, [source])}
            self.assertEqual(validate_evidence_data(rows, manifest), {'g': ['base', 'counterfactual']})
            embedded = {**manifest, 'source_records': [source]}
            path.unlink()
            self.assertEqual(validate_evidence_data(rows, embedded), {'g': ['base', 'counterfactual']})
            corrupted = copy.deepcopy(embedded)
            corrupted['source_records'][0]['input']['state'] += ' Changed source.'
            with self.assertRaisesRegex(ValueError, 'source fingerprint'):
                validate_evidence_data(rows, corrupted)
            manifest = embedded
            changed = copy.deepcopy(rows)
            changed[1]['input']['state'] += ' An extra unsupported fact.'
            with self.assertRaisesRegex(ValueError, 'reconstruction'):
                validate_evidence_data(changed, manifest)
            changed = copy.deepcopy(rows)
            changed[0]['evidence_certificate']['full_context_fact_states']['remove_left']['a'] = 'supported'
            with self.assertRaisesRegex(ValueError, 'necessity'):
                validate_evidence_data(changed, manifest)


if __name__ == '__main__':
    unittest.main()
