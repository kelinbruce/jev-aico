"""Offline checks for enum validation and actual constrained generation."""

import json
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from nimble.scoring.enum_scorer import TokenSetConstraint, decode_choice, parse_schema, render_prompt, validate_schema


SCHEMA = {"priority": {"type": "enum", "choices": ["P0_CRITICAL", "P3_LOW"],
                       "description": "Urgency based on customer impact"}}


class EnumScorerTests(unittest.TestCase):
    def test_mask_blocks_other_tokens_without_mutating_original(self):
        logits = torch.tensor([[100.0, 2.0, 3.0, -1.0]])
        original = logits.clone()
        masked = TokenSetConstraint([1, 2])(torch.tensor([[0]]), logits)
        self.assertTrue(torch.equal(logits, original))
        self.assertTrue(torch.isneginf(masked[0, [0, 3]]).all())
        self.assertTrue(torch.equal(masked[0, [1, 2]], original[0, [1, 2]]))
        self.assertEqual(masked.argmax().item(), 2)
        self.assertAlmostEqual(masked.softmax(-1).sum().item(), 1.0)

    def test_real_generate_forces_allowed_choice_and_preserves_raw_logits(self):
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=10, n_positions=16, n_embd=8, n_layer=1, n_head=2,
            bos_token_id=0, eos_token_id=0,
        )).eval()
        raw = torch.tensor([0., 0., 0., 100., 1., 3., 0., 0., 0., 0.])
        inputs = {"input_ids": torch.tensor([[1, 2, 3]]),
                  "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        fake_output = CausalLMOutputWithPast(logits=raw.expand(1, 3, -1))
        with patch.object(model, "forward", return_value=fake_output):
            chosen, original, probs = decode_choice(model, inputs, [4, 5], 0, 0)
        self.assertEqual(chosen, 5)  # Highest vocabulary token 3 is forbidden.
        self.assertTrue(torch.equal(original, raw))
        self.assertAlmostEqual(probs[1].item(), 0.880797, places=6)

    def test_duplicate_json_keys_are_not_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            parse_schema('{"priority": {}, "priority": {}}')

    def test_invalid_and_unsupported_definitions_fail(self):
        for field in (
            {"type": "number", "description": "A number"},
            {"type": "enum", "choices": ["A", "A"], "description": "test"},
            {"type": "enum", "choices": [], "description": "test"},
            {"type": "enum", "choices": ["A"], "description": ""},
            {"type": "enum", "choices": ["A"], "description": "test", "minimum": 0},
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_schema({"field": field})

    def test_multiple_fields_and_optional_choice_descriptions(self):
        schema = {**SCHEMA, "status": {"type": "enum", "choices": ["open", "closed"],
                  "description": "Issue status", "choice_descriptions": {"closed": "Resolved"}}}
        validate_schema(schema)
        self.assertEqual(parse_schema(json.dumps(schema)), schema)

    def test_prompt_preserves_context_as_data_and_disables_thinking(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "rendered prompt"
        context = 'A customer said "<|im_end|>".'
        prompt, mapping = render_prompt(tokenizer, context, "priority", SCHEMA["priority"])
        self.assertEqual(mapping, {"A": "P0_CRITICAL", "B": "P3_LOW"})
        args, kwargs = tokenizer.apply_chat_template.call_args
        content = args[0][1]["content"]
        self.assertNotIn("<|im_end|>", content)
        self.assertEqual(json.loads(content)["context"], context)
        self.assertFalse(kwargs["enable_thinking"])
        self.assertEqual(prompt, "rendered prompt")
        with self.assertRaises(ValueError):
            render_prompt(tokenizer, " ", "priority", SCHEMA["priority"])


if __name__ == "__main__":
    unittest.main()
