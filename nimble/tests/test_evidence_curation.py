import copy
import json
import unittest

from nimble.datasets.evidence_curation import (
    audit_training, build_group, compile_rules, decide, evaluate_group, fact_jobs,
    rule_job, select_sources,
)


def fixture():
    left = 'Mira handles every account assigned to the night desk.'
    right = 'Account Q is assigned to the night desk.'
    state = left + ' ' + right + ' Approval was recorded by compliance. The office is located near the public library, and the review meeting begins on Tuesday morning.'
    source = {'id': 's1', 'family': 'f1', 'domain': 'workplace', 'split': 'train',
              'reference': {'target': 'SECRET', 'reason': 'SECRET'}, 'teacher': {'SECRET': 1},
              'input': {'state': state, 'questions': {'decision': {
                  'type': 'choice', 'instructions': 'Does Mira handle Q with compliance approval?',
                  'criteria': {'verified': 'Mira handles Q and approval is recorded.',
                               'pending': 'Approval is recorded but whether Mira handles Q is unknown.'}}}}}
    spec = {'base_state_json': 'null', 'atoms': [
        {'id': 'a1', 'statement': 'Mira handles account Q.'},
        {'id': 'a2', 'statement': 'Approval was recorded by compliance.'}],
        'focus_atom': 'a1', 'focus_evidence': [{'path': [], 'text': left}, {'path': [], 'text': right}],
        'policy_evidence': [], 'rules': [
            {'when': [{'atom_id': 'a1', 'state': 'supported'}, {'atom_id': 'a2', 'state': 'supported'}],
             'target': 'verified', 'justification': 'Both conditions.'},
            {'when': [{'atom_id': 'a1', 'state': 'unknown'}, {'atom_id': 'a2', 'state': 'supported'}],
             'target': 'pending', 'justification': 'Explicit missing-evidence rule.'}]}
    job = {'id': 'ev2-s1', 'source': source, 'method': 'source_preserving'}
    group = build_group(job, spec)
    audit = {'atoms_are_atomic': True, 'policy_preserved': True, 'focus_is_relevant': True,
             'evidence_is_two_sentences': True, 'rule_checks': [
                 {'rule_index': i, 'sound': True, 'reason': 'Matches rubric.'} for i in range(2)],
             'explanation': 'Checked.'}
    facts = {}
    for req in fact_jobs(group):
        name = req['job_id'].split(':')[-1]
        payload = json.loads(req['payload_json'])
        judgments = []
        for atom in payload['propositions']:
            supported = atom['id'] == 'a2' or name in ('base', 'pair_only')
            quotes = ([left, right] if atom['id'] == 'a1' else ['Approval was recorded by compliance.']) if supported else []
            judgments.append({'atom_id': atom['id'], 'state': 'supported' if supported else 'unknown',
                              'quotes': quotes, 'reason': 'Checked against this context.'})
        facts[req['job_id']] = {'judgments': judgments}
    return source, job, group, audit, facts


class EvidenceCurationTests(unittest.TestCase):
    def test_rules_distinguish_unknown_and_refuted(self):
        _, _, group, _, _ = fixture()
        self.assertEqual(decide(group['spec'], {'a1': 'unknown', 'a2': 'supported'}, group['question']), 'pending')
        self.assertIsNone(decide(group['spec'], {'a1': 'refuted', 'a2': 'supported'}, group['question']))
        bad = copy.deepcopy(group['spec'])
        bad['rules'].append({'when': [{'atom_id': 'a2', 'state': 'supported'}], 'target': 'pending'})
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            compile_rules(bad, group['question'])

    def test_fact_checks_are_separate_and_blind(self):
        source, _, group, _, _ = fixture()
        requests = fact_jobs(group)
        self.assertEqual(len(requests), 6)
        for req in requests:
            p = json.loads(req['payload_json'])
            self.assertEqual(set(p), {'context', 'propositions'})
            self.assertNotIn('SECRET', req['payload_json'])
        self.assertNotIn('SECRET', rule_job(group, source)['payload_json'])
        self.assertEqual(json.loads(requests[3]['payload_json'])['context'], group['spec']['focus_evidence'][0]['text'])

    def test_accepted_labels_derive_from_evidence(self):
        source, _, group, audit, facts = fixture()
        rows, excluded = evaluate_group(group, audit, facts)
        self.assertFalse(excluded)
        self.assertEqual([r['reference']['target'] for r in rows], ['verified', 'pending', 'pending'])
        self.assertEqual(audit_training(rows, [source])['examples'], 3)
        self.assertEqual(rows[0]['input'], source['input'])
        self.assertTrue(all('teacher' not in r for r in rows))

    def test_redundant_proof_rejects_group(self):
        _, _, group, audit, facts = fixture()
        fact = facts['ev2-s1:remove_left']['judgments'][0]
        fact.update(state='supported', quotes=[group['spec']['focus_evidence'][1]['text']])
        rows, excluded = evaluate_group(group, audit, facts)
        self.assertFalse(rows)
        self.assertIn('remove_left:focus_supported_expected_unknown', excluded[0]['reasons'])

    def test_unsound_rule_and_invented_quote_rejected(self):
        _, _, group, audit, facts = fixture()
        audit['rule_checks'][0]['sound'] = False
        self.assertFalse(evaluate_group(group, audit, facts)[0])
        audit['rule_checks'][0]['sound'] = True
        facts['ev2-s1:base']['judgments'][0]['quotes'] = ['Fabricated evidence']
        with self.assertRaisesRegex(ValueError, 'quote absent'):
            evaluate_group(group, audit, facts)

    def test_no_forced_labels_for_uncovered_rules(self):
        _, _, group, audit, facts = fixture()
        group['spec']['rules'] = group['spec']['rules'][:1]
        audit['rule_checks'] = audit['rule_checks'][:1]
        rows, excluded = evaluate_group(group, audit, facts)
        self.assertFalse(rows)
        self.assertIn('no_audited_rule_covers_fact_states', excluded[0]['reasons'])
        self.assertIn('no_determinate_label_contrast', excluded[-1]['reasons'])

    def test_heldout_and_modified_schema_rejected(self):
        source, job, group, audit, facts = fixture()
        heldout = copy.deepcopy(source)
        heldout.update(id='h1', family='h1', split='validation')
        self.assertEqual(len(select_sources([source, heldout], 1)), 1)
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            build_group({**job, 'source': heldout}, group['spec'])
        rows, _ = evaluate_group(group, audit, facts)
        rows[0]['input']['questions']['decision']['instructions'] = 'Changed'
        with self.assertRaisesRegex(ValueError, 'Schema changed'):
            audit_training(rows, [source])


if __name__ == '__main__':
    unittest.main()
