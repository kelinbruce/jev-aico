import unittest

from nimble.evaluation.benchmark_decision_latency import fingerprint, timing_summary


class DecisionLatencyTests(unittest.TestCase):
    def test_milliseconds_and_interpolated_percentile(self):
        result = timing_summary([.1, .3, .2])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["median_ms"], 200)
        self.assertEqual(result["mean_ms"], 200)
        self.assertEqual(result["p95_ms"], 290)

    def test_rejects_invalid_timings(self):
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                timing_summary(values)

    def test_fingerprint_ignores_object_key_order(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(fingerprint(["a", "b"]), fingerprint(["b", "a"]))


if __name__ == "__main__":
    unittest.main()
