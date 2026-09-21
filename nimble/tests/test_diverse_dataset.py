"""Offline coverage, leakage, and export checks for the larger dataset."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nimble.datasets.create_diverse_dataset import export, teacher_targets, generation_params
from nimble.datasets.dataset_io import request_for
from nimble.datasets.diversity_plan import DOMAINS, audit, example_plans
from tests.test_dataset_io import fixture_teacher


def fixtures():
    topics = [{"domain": domain, "domain_index": i, "subtopic_index": j,
               "group_id": f"{domain}-{j}", "subtopic": f"Topic {j}",
               "description": "Test", "actors_json": "[]"}
              for i, (domain, _) in enumerate(DOMAINS) for j in range(5)]
    rows = []
    for plan in example_plans(topics, 17):
        for slot in json.loads(plan["slots_json"]):
            kind = slot["type"]
            text = " ".join(f"{plan['domain']}{plan['subtopic_index']}word{k}" for k in range(25))
            text += f" scenario {slot['slot']}"
            state = (text if slot["format"] == "text" else {"description": text}
                     if slot["format"] == "object" else [{"speaker": "Test", "text": text}])
            criteria = ({"first": "First", "second": "Second", "third": "Third"}
                        if kind == "choice" else {"true": "Yes", "false": "No"}
                        if kind == "noul" else [f"Level {i}" for i in range(slot["score_levels"])])
            row = {"id": f"test-{len(rows):03d}", "split": plan["split"], "family": plan["group_id"],
                   "domain": plan["domain"], "subtopic": plan["subtopic"],
                   "diversity": {k: slot[k] for k in ("format", "mechanism", "difficulty")},
                   "input": {"state": state, "questions": {"decision": {
                       "type": kind, "instructions": "Select the outcome.", "criteria": criteria}}},
                   "reference": {"target": "first" if kind == "choice" else slot["target"],
                                 "reason": "Test fixture", "source": "test", "human_reviewed": False}}
            row["teacher"] = fixture_teacher(row)
            rows.append(row)
    return rows


class DiverseDatasetTests(unittest.TestCase):
    def test_reasoning_generator_request_compatibility(self):
        params = generation_params("gpt-5.6-sol", 2200)
        self.assertEqual(params["reasoning_effort"], "medium")
        self.assertGreater(params["max_completion_tokens"], 2200)
        self.assertNotIn("temperature", params)
        self.assertNotIn("max_tokens", params)
        self.assertEqual(generation_params("gpt-4o-mini-2024-07-18", 2200),
                         {"temperature": 0.8, "max_tokens": 2200})

    def test_normalized_training_targets_keep_raw_probabilities(self):
        answer = {"type": "score", "probabilities": {"0": 0.01, "1": 0.65, "2": 0.33}, "score": 1.32}
        before = copy.deepcopy(answer)
        targets = teacher_targets(answer)
        self.assertAlmostEqual(sum(targets["probabilities"].values()), 1.0)
        self.assertTrue(targets["normalization_applied"])
        self.assertEqual(answer, before)

    def test_coverage(self):
        report = audit(fixtures())
        self.assertEqual(report["unique_states"], 300)
        self.assertEqual(report["split_counts"], {"train": 240, "validation": 30, "eval": 30})
        self.assertEqual(report["primitive_counts"], {"choice": 100, "noul": 100, "score": 100})
        self.assertEqual(report["format_counts"], {"text": 100, "object": 100, "dialogue": 100})
        self.assertEqual(report["noul_targets"], {"false": 50, "true": 50})

    def test_duplicate_and_cross_split_guards(self):
        rows = fixtures()
        rows[1]["input"]["state"] = copy.deepcopy(rows[0]["input"]["state"])
        rows[1]["diversity"]["format"] = rows[0]["diversity"]["format"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit(rows)
        rows = fixtures()
        first = next(r for r in rows if r["split"] == "train")
        heldout = next(r for r in rows if r["split"] == "eval")
        first["family"] = heldout["family"]
        with self.assertRaisesRegex(ValueError, "leaks"):
            audit(rows)

    def test_no_generation_metadata_in_teacher_input(self):
        row = fixtures()[0]
        payload = request_for(row, "test-model")
        self.assertEqual(set(payload), {"model", "state", "questions"})
        altered = copy.deepcopy(row)
        altered["reference"] = {"target": "CANARY_REFERENCE"}
        altered["diversity"] = {"secret": "CANARY_METADATA"}
        self.assertEqual(payload, request_for(altered, "test-model"))

    def test_export_separation_and_disagreement_retention(self):
        rows = fixtures()
        # Teacher disagreement must remain in the dataset, not be silently filtered.
        answer = rows[0]["teacher"]["answers"]["decision"]
        answer["choice"] = "second"
        answer["probabilities"] = {"first": 0.0, "second": 1.0, "third": 0.0}
        with tempfile.TemporaryDirectory() as folder, patch(
                "nimble.datasets.create_diverse_dataset.version", return_value="test"):
            root = Path(folder)
            result = export(rows, root, "generator-test", "test-model", 17)
            read = lambda name: [json.loads(line) for line in (root / name).read_text().splitlines()]
            self.assertEqual(len(read("all.jsonl")), 300)
            self.assertEqual(len(read("train_sft.jsonl")), 240)
            self.assertEqual(len(read("validation_prompts.jsonl")), 30)
            self.assertEqual(len(read("eval_prompts.jsonl")), 30)
            self.assertEqual(result["generator_teacher_disagreements"], 1)
            train_ids = {r["id"] for r in read("train_sft.jsonl")}
            for name in ("validation_prompts.jsonl", "eval_prompts.jsonl"):
                self.assertFalse(train_ids & {r["id"] for r in read(name)})
                self.assertTrue(all(len(r["messages"]) == 2 for r in read(name)))


if __name__ == "__main__":
    unittest.main()
