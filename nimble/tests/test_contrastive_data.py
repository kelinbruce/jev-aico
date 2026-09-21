import copy
import json
import unittest

from nimble.datasets.contrastive_data import (
    apply_edit, assess_verification, audit_export, build_group, generation_jobs,
    scoring_record, verification_input, decode_verifier_target,
)


def fixture():
    state = "The shipment has an intact seal. Inspection passed. The clerk logged the arrival at noon."
    source = {"id": "s1", "family": "f1", "split": "train", "domain": "commerce", "subtopic": "Receiving",
              "input": {"state": state, "questions": {"decision": {
                  "type": "noul", "instructions": "Is the shipment eligible? Require both an intact seal and a passed inspection.",
                  "criteria": {"true": "Both conditions are established.", "false": "Either condition fails or is not established."}}}},
              "reference": {"target": True, "reason": "Both conditions are met."}}
    job = generation_jobs([source])[0] | {"method": "d2c"}
    draft = {"base_state_json": "null", "base_reason": "Both conditions hold.",
             "atomic_facts": [{"fact": "Seal is intact", "path": [], "evidence": "an intact seal"},
                              {"fact": "Inspection passed", "path": [], "evidence": "Inspection passed."}],
             "removal": {"path": [], "old": "Inspection passed.", "new": ""},
             "removed_target": "false", "removed_reason": "Inspection is not established.",
             "counterfactual": {"path": [], "old": "an intact seal", "new": "a broken seal"},
             "counterfactual_target": "false", "counterfactual_reason": "Seal fails.",
             "paraphrase_edits": [{"path": [], "old": "Inspection passed.", "new": "The inspection was successful."}],
             "paraphrase_reason": "Both facts preserved.", "distractor": "The clerk wore a blue jacket.",
             "distractor_reason": "Clothes do not affect eligibility."}
    return source, job, draft


class ContrastiveDataTests(unittest.TestCase):
    def test_verifier_letters_require_unique_explicit_labels(self):
        question = {"type": "choice", "criteria": {"B — Green": "Go", "A — Red": "Stop"}}
        self.assertEqual(decode_verifier_target("A", question), "A — Red")
        self.assertEqual(decode_verifier_target("B — Green", question), "B — Green")
        for criteria in ({"Red": "Stop", "Green": "Go"}, {"A — Red": "Stop", "A — Green": "Go"}):
            with self.assertRaises(ValueError):
                decode_verifier_target("A", {"type": "choice", "criteria": criteria})

    def test_group_edits_are_exact_and_retain_schema(self):
        source, job, draft = fixture()
        rows = build_group(job, draft, "test")
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["input"], source["input"])
        self.assertEqual(rows[1]["input"]["state"], source["input"]["state"].replace("Inspection passed.", ""))
        self.assertEqual([r["reference"]["target"] for r in rows], [True, False, False, True, True])
        self.assertTrue(all(r["input"]["questions"] == source["input"]["questions"] for r in rows))
        self.assertTrue(all("teacher" not in r for r in rows))

    def test_nested_edits_do_not_mutate_original(self):
        original = {"evidence": ["A", "B"]}
        actual = apply_edit(original, {"path": ["evidence", "1"], "old": "B", "new": "C"})
        self.assertEqual(original, {"evidence": ["A", "B"]})
        self.assertEqual(actual, {"evidence": ["A", "C"]})
        with self.assertRaises(ValueError):
            apply_edit("A A", {"path": [], "old": "A", "new": "B"})

    def test_wrapper_paths_are_resolved_only_when_literal_span_matches(self):
        _, job, draft = fixture()
        expected = build_group(job, draft, "test")
        draft["removal"]["path"] = ["input", "state"]
        actual = build_group(job, draft, "test")
        self.assertEqual(actual[1]["input"], expected[1]["input"])
        self.assertEqual(actual[1]["construction"]["path_repairs"],
                         [{"original": ["input", "state"], "resolved": []}])
        draft["removal"]["old"] = "Not a source span"
        with self.assertRaises((ValueError, TypeError)):
            build_group(job, draft, "test")

    def test_verification_blinds_labels_and_quarantines_disagreement(self):
        source, job, draft = fixture()
        rows = build_group(job, draft, "test")
        payload, mapping = verification_input(rows)
        poisoned = copy.deepcopy(rows)
        for row in poisoned:
            row["reference"] = {"target": "SECRET", "reason": "SECRET"}
            row["teacher"] = {"choice": "SECRET"}
        self.assertEqual(verification_input(poisoned), (payload, mapping))
        self.assertNotIn("SECRET", json.dumps(payload))
        lookup = {r["id"]: r for r in rows}
        response = {"judgments": [{"id": key, "target": str(lookup[value]["reference"]["target"]).lower(),
                                    "unambiguous": True, "reason": "Evidence checked."}
                                   for key, value in mapping.items()]}
        verified, issues = assess_verification(rows, response, "test")
        self.assertFalse(issues)
        self.assertEqual(audit_export(verified, [source])["examples"], 5)
        response["judgments"][0]["unambiguous"] = False
        disputed, issues = assess_verification(rows, response, "test")
        self.assertEqual(len(issues), 1)
        with self.assertRaises(ValueError):
            audit_export(disputed, [source])

    def test_heldout_sources_excluded_and_whole_groups_required(self):
        source, job, draft = fixture()
        heldout = copy.deepcopy(source)
        heldout.update(id="heldout", split="eval")
        self.assertEqual(len(generation_jobs([source, heldout])), 1)
        with self.assertRaises(ValueError):
            build_group({**job, "source": heldout}, draft, "test")
        rows = build_group(job, draft, "test")
        for row in rows:
            row["verification"] = {"agrees": True, "unambiguous": True}
        with self.assertRaises(ValueError):
            audit_export(rows[:-1], [source])

    def test_export_has_only_input_and_target_not_rationale(self):
        _, job, draft = fixture()
        row = build_group(job, draft, "test")[0]
        exported = scoring_record(row)
        self.assertTrue(exported["target"])
        self.assertEqual(exported["target_key"], "true")
        self.assertNotIn("reference", exported)
        self.assertNotIn("construction", exported)
        self.assertEqual(exported["schema"]["decision"]["choices"], [False, True])


if __name__ == "__main__":
    unittest.main()
