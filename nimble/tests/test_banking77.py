import unittest

from nimble.evaluation.evaluate_banking77 import chat_prompt, parse_response, summarize


class Banking77Tests(unittest.TestCase):
    def test_prompt_contains_every_label_and_request(self):
        labels = [f"label_{i:02d}" for i in range(77)]
        prompt = chat_prompt("Test request", labels)
        for label in labels:
            self.assertEqual(prompt.count('"'+label+'"'), 1)
        self.assertIn("Request:\nTest request", prompt)

    def test_response_validation(self):
        self.assertEqual(parse_response('```json\n{"label":"x","confidence":75}\n```', ["x"]),
                         {"prediction": "x", "confidence": .75})
        for text in ('{"label":"y","confidence":75}', '{"label":"x","confidence":101}',
                     '{"label":"x","confidence":NaN}', 'A'):
            with self.assertRaises(ValueError):
                parse_response(text, ["x"])

    def test_metrics_and_failures(self):
        rows = [{"correct": True, "confidence": .75, "latency_s": 1},
                {"correct": False, "confidence": .25, "latency_s": 3}]
        result = summarize(rows)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["brier"], .0625)
        self.assertEqual(result["ece10"], .25)
        self.assertEqual(result["auroc"], 1)
        rows.append({"correct": False, "error": "invalid", "latency_s": 2})
        result = summarize(rows)
        self.assertEqual(result["accuracy"], 1/3)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["calibration_n"], 2)


if __name__ == "__main__":
    unittest.main()
