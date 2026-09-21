"""Offline checks for coverage and label-blind complex-example review."""
import copy
import json
import unittest
from collections import Counter

from nimble.datasets.create_complex_segments import (
    GATES, TOPICS, plans, reorder, review_prompt, validate_review,
)


class ComplexSegmentTests(unittest.TestCase):
    def test_exact_segment_and_primitive_quotas(self):
        jobs = plans()
        self.assertEqual(len({r['example_id'] for r in jobs}), 150)
        self.assertEqual(Counter(r['domain'] for r in jobs), {d:30 for d in TOPICS})
        self.assertEqual(len({r['group_id'] for r in jobs}), 30)
        self.assertEqual(set(r['split'] for r in jobs), {'unassigned'})
        for domain in TOPICS:
            slots = [json.loads(r['slot_json']) for r in jobs if r['domain']==domain]
            self.assertEqual(Counter(s['type'] for s in slots), {'choice':10,'noul':10,'score':10})
            self.assertEqual(Counter(s['target'] for s in slots if s['type']=='noul'), {True:5,False:5})

    def test_review_hides_labels_and_generation_metadata(self):
        row = {'domain':'gaming','subtopic':'quest_dependencies','input':{'state':'Supplied facts'},
               'reference':{'target':'SECRET_TARGET','reason':'SECRET_RATIONALE'},
               'slot_json':'SECRET_SLOT','provenance':{'hidden':'SECRET_PROVENANCE'}}
        prompt = review_prompt(row)
        self.assertIn('Supplied facts',prompt)
        self.assertNotIn('SECRET',prompt)
        other = copy.deepcopy(row)
        other['reference']['target']='A DIFFERENT ANSWER'
        self.assertEqual(review_prompt(other),prompt)

    def test_review_requires_agreement_all_gates_and_real_evidence(self):
        row = {'input':{'state':'The bridge is closed. The boat leaves at noon.'}, 'reference':{'target':True}}
        review = dict(target_key='true', evidence_quotes=['The bridge is closed.','The boat leaves at noon.'],
                      reason='Two facts determine the permitted route.', **{gate:True for gate in GATES})
        validate_review(row,review)
        for update in ({'target_key':'false'}, {'requires_combining_evidence':False},
                       {'evidence_quotes':['Invented evidence','The boat leaves at noon.']},
                       {'evidence_quotes':['The bridge is closed.']},
                       {'evidence_quotes':['The bridge is closed.','The bridge is closed.']}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_review(row,{**review,**update})

    def test_choice_reordering_keeps_meanings_and_requested_gold_position(self):
        row = {'id':'case','input':{'questions':{'decision':{'type':'choice','criteria':{
            'one':'Meaning 1','two':'Meaning 2','three':'Meaning 3','four':'Meaning 4'}}}},
            'reference':{'target':'two'}}
        before = copy.deepcopy(row['input']['questions']['decision']['criteria'])
        result = reorder(row,{'choice_target_position':3})
        after = result['input']['questions']['decision']['criteria']
        self.assertEqual(after,before)
        self.assertEqual(list(after)[3],'two')


if __name__=='__main__':
    unittest.main()
