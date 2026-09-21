import copy
import unittest

from nimble.datasets.curate_paired_evidence import pair_requests, pair_passes


class PairedEvidenceTests(unittest.TestCase):
    def test_pair_must_change_exactly_one_sentence(self):
        atom = {'id': 'a1', 'statement': 'Parcel K exceeds its route weight limit.'}
        pair = {'left': 'Parcel K uses route R with a twelve kilogram limit.',
                'right': 'Parcel K weighs fifteen kilograms.',
                'negative_left': 'Parcel K uses route R with a twelve kilogram limit.',
                'negative_right': 'Parcel K weighs nine kilograms.'}
        reqs = pair_requests('g', atom, pair)
        self.assertEqual(len(reqs), 5)
        expected = {'left': 'unknown', 'right': 'unknown', 'positive_pair': 'supported',
                    'negative_sentence': 'unknown', 'negative_pair': 'refuted'}
        responses = {}
        for req in reqs:
            name = req['job_id'].split(':')[-1]
            quotes = []
            if name == 'positive_pair': quotes = [pair['left'], pair['right']]
            if name == 'negative_pair': quotes = [pair['negative_left'], pair['negative_right']]
            responses[req['job_id']] = {'judgments': [{'atom_id': 'a1', 'state': expected[name],
                                                     'quotes': quotes, 'reason': 'Check source.'}]}
        self.assertTrue(pair_passes('g', atom, pair, responses))
        responses['g:negative_pair']['judgments'][0].update(state='unknown', quotes=[])
        self.assertFalse(pair_passes('g', atom, pair, responses))
        bad = copy.deepcopy(pair)
        bad['negative_left'] = 'Parcel K uses a different route with a higher limit.'
        with self.assertRaises(ValueError): pair_requests('g', atom, bad)


if __name__ == '__main__':
    unittest.main()
