import json
import math
import unittest
from pathlib import Path

from nimble.paths import PROJECT_ROOT

from nimble.evaluation.evaluate_pilot import adapt_input, assess, markdown_report, summarize


class PilotEvaluationTests(unittest.TestCase):
    def test_all_input_types_preserve_criteria_without_reference_leakage(self):
        rows = [json.loads(line) for line in (PROJECT_ROOT / "data/eval.jsonl").read_text().splitlines()]
        for row in rows:
            context, schema = adapt_input(row["input"])
            question = row["input"]["questions"]["decision"]
            self.assertEqual(schema["decision"]["description"], question["instructions"])
            criteria = question["criteria"]
            expected = dict(enumerate(criteria)) if isinstance(criteria, list) else criteria
            self.assertEqual(schema["decision"]["choice_descriptions"], {str(k): v for k, v in expected.items()})
            if not isinstance(row["input"]["state"], str):
                self.assertEqual(json.loads(context), row["input"]["state"])
            else:
                self.assertEqual(context, row["input"]["state"])
            # Poison metadata: the actual inference adapter must remain unaffected.
            row["reference"] = {"target": "LEAK_SENTINEL"}
            row["teacher"] = {"answer": "LEAK_SENTINEL"}
            self.assertEqual(adapt_input(row["input"]), (context, schema))

    def test_metrics_against_hand_computed_results(self):
        binary = assess({"false": 0.25, "true": 0.75}, False, "noul")
        self.assertFalse(binary["correct"])
        self.assertAlmostEqual(binary["binary_brier"], 0.5625)
        self.assertAlmostEqual(binary["multiclass_brier"], 1.125)
        self.assertAlmostEqual(binary["negative_log_likelihood"], -math.log(0.25))
        ordered = assess({"0": 0.1, "1": 0.2, "2": 0.7}, 2, "score")
        self.assertEqual(ordered["prediction"], 2)
        self.assertAlmostEqual(ordered["expected_score"], 1.6)
        self.assertAlmostEqual(ordered["absolute_score_error"], 0.4)
        summary = summarize([{"type": "noul", "qwen": binary}, {"type": "score", "qwen": ordered}], "qwen")
        self.assertEqual(summary["all"]["accuracy"], 0.5)
        self.assertEqual(summary["score"]["accuracy"], 1)

    def test_invalid_distributions_and_labels_fail(self):
        for probs, target in [({"a": 0.2, "b": 0.2}, "a"),
                              ({"a": float("nan"), "b": 0.5}, "a"),
                              ({"a": 0.5, "b": 0.5}, "missing")]:
            with self.assertRaises(ValueError):
                assess(probs, target, "choice")

    def test_report_uses_dataset_size_and_reference_provenance(self):
        rows = []
        for kind, target, probabilities in [
            ("choice", "a", {"a": 0.9, "b": 0.1}),
            ("noul", False, {"false": 0.9, "true": 0.1}),
            ("score", 1, {"0": 0.1, "1": 0.9}),
        ]:
            rows.append({"id": kind, "family": "example", "type": kind,
                         "reference": {"target": target, "source": "generator_model", "human_reviewed": False},
                         "qwen": assess(probabilities, target, kind),
                         "teacher": assess(probabilities, target, kind)})
        report = markdown_report({"dataset": "/data/different_dataset/all.jsonl", "rows": rows,
                                  "summary": {name: summarize(rows, name) for name in ("qwen", "teacher")}})
        self.assertIn("Evaluated 3 examples", report)
        self.assertIn("generator_model", report)
        self.assertIn("Human-reviewed: 0/3", report)
        self.assertIn("--data /data/different_dataset/all.jsonl", report)
        self.assertNotIn("ten-example", report)


if __name__ == "__main__":
    unittest.main()
