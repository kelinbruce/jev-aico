"""Numerical checks for the extra model architectures and shared metrics."""

import unittest
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx_lm.models.gemma3_text import Model as Gemma, ModelArgs as GemmaArgs
from mlx_lm.models.qwen3_5 import Model as Qwen, ModelArgs as QwenArgs

from nimble.scoring.parallel_schema import PreparedPrompts
from nimble.scoring.parallel_scorer import ParallelScorer
from nimble.evaluation.evaluate_models import teacher_assessment


class ModelEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.device = mx.default_device()
        mx.set_default_device(mx.cpu)
        mx.random.seed(12)

    def tearDown(self):
        mx.set_default_device(self.device)

    def check_projection(self, model, language_model):
        model.eval()
        scorer = ParallelScorer.__new__(ParallelScorer)
        scorer.model = model
        scorer.backbone = language_model.model
        scorer.head_weight = language_model.lm_head.weight
        scorer.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
        scorer.temperature = 1.0
        ids = list(range(1, 14))
        candidates = [15, 16, 17]
        prepared = PreparedPrompts(['decision'], [['a', 'b', 'c']], ids[:5],
                                   [ids[5:]], [ids], [candidates])
        actual, _ = scorer.evaluate(prepared, mode='independent')
        full = model(mx.array([ids]), cache=model.make_cache())[0, -1, candidates]
        mx.eval(actual, full)
        np.testing.assert_allclose(np.array(actual[0]), np.array(full), rtol=1e-4, atol=1e-4)

    def test_gemma_sliding_window_candidate_logits_match_full_model(self):
        model = Gemma(GemmaArgs(model_type='gemma3_text', hidden_size=32,
                              num_hidden_layers=6, intermediate_size=64,
                              num_attention_heads=2, num_key_value_heads=1,
                              head_dim=16, vocab_size=32, sliding_window=8))
        self.check_projection(model, model)

    def test_qwen_untied_candidate_logits_match_full_model(self):
        model = Qwen(QwenArgs(model_type='qwen3_5', text_config={
            'model_type': 'qwen3_5_text', 'hidden_size': 32, 'intermediate_size': 64,
            'num_hidden_layers': 4, 'num_attention_heads': 2, 'num_key_value_heads': 1,
            'head_dim': 16, 'vocab_size': 32, 'full_attention_interval': 4,
            'linear_num_value_heads': 2, 'linear_num_key_heads': 2,
            'linear_key_head_dim': 128, 'linear_value_head_dim': 128,
            'linear_conv_kernel_dim': 4, 'tie_word_embeddings': False,
            'rope_parameters': {'rope_theta': 10000., 'partial_rotary_factor': .5},
        }))
        self.check_projection(model, model.language_model)

    def test_teacher_ties_preserve_saved_choice_and_false_noul(self):
        row = {'reference': {'target': 'b'}, 'teacher_targets': {'probabilities': {'a': .5, 'b': .5}},
               'teacher': {'answers': {'decision': {'type': 'choice', 'choice': 'b'}}}}
        self.assertEqual(teacher_assessment(row, 'choice')['prediction'], 'b')
        row = {'reference': {'target': False}, 'teacher_targets': {'probabilities': {'true': .5, 'false': .5}},
               'teacher': {'answers': {'decision': {'type': 'noul', 'noul': .5}}}}
        self.assertIs(teacher_assessment(row, 'noul')['prediction'], False)


if __name__ == '__main__':
    unittest.main()
