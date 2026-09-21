"""Offline checks for label leakage, split contamination, and teacher validation."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nimble.datasets.dataset_io import request_for, training_record, validate_teacher
from tests.dataset_fixtures import seeds
from nimble.datasets.typesafe_curator_bridge import completion_envelope


def fixture_teacher(row):
    q = row["input"]["questions"]["decision"]
    target = row["reference"]["target"]
    kind = q["type"]
    answer = {"type": kind}
    if kind == "noul":
        answer["noul"] = float(target)
    elif kind == "choice":
        answer.update(choice=target, confidence=1.0,
                      probabilities={key: float(key == target) for key in q["criteria"]})
    else:
        answer.update(score=float(target), confidence=1.0,
                      probabilities={str(i): float(i == target) for i in range(len(q["criteria"]))},
                      legend={str(i): value for i, value in enumerate(q["criteria"])})
    return {"model": "test-model", "answers": {"decision": answer},
            "usage": {"input_tokens": 10, "output_tokens": 5}}


class DatasetTests(unittest.TestCase):
    def test_no_reference_leakage(self):
        for row in seeds():
            original = request_for(row, "test-model")
            changed = copy.deepcopy(row)
            changed.update(reference={"target": "SECRET_LABEL", "reason": "SECRET_REASON"},
                           split="SECRET_SPLIT", id="SECRET_ID")
            self.assertEqual(original, request_for(changed, "test-model"))
            self.assertEqual(set(original), {"model", "state", "questions"})


    def test_invalid_probabilities_rejected(self):
        row = seeds()[0]
        teacher = fixture_teacher(row)
        validate_teacher(row, teacher, "test-model")
        for bad in (float("nan"), -0.1, 1.1, 0.5):
            changed = copy.deepcopy(teacher)
            changed["answers"]["decision"]["probabilities"]["returns"] = bad
            with self.assertRaises(ValueError):
                validate_teacher(row, changed, "test-model")

    def test_score_and_model_guards(self):
        row = seeds()[-1]
        teacher = fixture_teacher(row)
        teacher["answers"]["decision"]["score"] = 0.0
        with self.assertRaisesRegex(ValueError, "weighted"):
            validate_teacher(row, teacher, "test-model")
        with self.assertRaisesRegex(ValueError, "pinned"):
            validate_teacher(row, fixture_teacher(row), "different-model")

    def test_api_rounding_is_preserved(self):
        row = seeds()[-1]
        teacher = fixture_teacher(row)
        answer = teacher["answers"]["decision"]
        answer["probabilities"] = {"0": 0.01, "1": 0.65, "2": 0.33}
        answer["score"] = 1.32
        before = copy.deepcopy(teacher)
        validate_teacher(row, teacher, "test-model")
        self.assertEqual(teacher, before)
        answer["score"] = 1.8
        with self.assertRaisesRegex(ValueError, "weighted"):
            validate_teacher(row, teacher, "test-model")


    def test_transport_preserves_teacher_output(self):
        teacher = fixture_teacher(seeds()[0])
        envelope = completion_envelope(teacher)
        self.assertEqual(json.loads(envelope["choices"][0]["message"]["content"]), teacher)
        self.assertEqual(envelope["usage"]["total_tokens"], 15)


if __name__ == "__main__":
    unittest.main()
