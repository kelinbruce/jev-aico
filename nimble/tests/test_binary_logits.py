"""Offline checks of score semantics for the native Qwen runner."""

import math
import unittest

import torch

from nimble.scoring.binary_logits import summarize_logits


class ScoreTests(unittest.TestCase):
    def test_label_probability_is_not_full_vocabulary_probability(self):
        score = summarize_logits(torch.tensor([0.0, 2.0, 10.0]), 0, 1, 4)
        self.assertAlmostEqual(score.support_logit, 2)
        self.assertAlmostEqual(score.support_probability, 1 / (1 + math.exp(-2)), places=6)
        self.assertLess(score.yes_probability_full_vocab, 0.001)
        self.assertLess(score.label_probability_mass, 0.001)

    def test_common_logit_shift_preserves_margin_and_probability(self):
        logits = torch.tensor([3.0, -1.0, 2.0])
        before = summarize_logits(logits, 0, 1, 1)
        after = summarize_logits(logits + 1000, 0, 1, 1)
        self.assertEqual(before.support_logit, after.support_logit)
        self.assertEqual(before.support_probability, after.support_probability)
        self.assertEqual(before.predicted_label, 0)

    def test_tie_and_extreme_logits(self):
        self.assertEqual(summarize_logits(torch.zeros(2), 0, 1, 1).predicted_label, 0)
        for yes, expected in [(1000.0, 1.0), (-1000.0, 0.0)]:
            result = summarize_logits(torch.tensor([0.0, yes]), 0, 1, 1)
            self.assertEqual(result.support_probability, expected)

    def test_nonfinite_output_fails(self):
        with self.assertRaises(ValueError):
            summarize_logits(torch.tensor([0.0, float('nan')]), 0, 1, 1)
