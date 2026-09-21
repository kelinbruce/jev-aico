"""Compare identical prompts across independent, cached-serial and batched paths."""

import argparse
import gc
import json
import platform
import statistics
from importlib.metadata import version
from pathlib import Path

from nimble.paths import PROJECT_ROOT

import mlx.core as mx

from nimble.scoring.parallel_schema import parse_schema
from nimble.scoring.parallel_scorer import ParallelScorer


def compare(actual, expected):
    errors = []
    for a, b in zip(actual, expected):
        errors.append({
            "max_logit_difference": mx.max(mx.abs(a - b)).item(),
            "max_probability_difference": mx.max(mx.abs(mx.softmax(a) - mx.softmax(b))).item(),
            "same_choice": mx.argmax(a).item() == mx.argmax(b).item(),
        })
    return {"max_logit_difference": max(x["max_logit_difference"] for x in errors),
            "max_probability_difference": max(x["max_probability_difference"] for x in errors),
            "all_choices_agree": all(x["same_choice"] for x in errors), "fields": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=(PROJECT_ROOT / "examples/parallel_benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    base = parse_schema((PROJECT_ROOT / "examples/parallel_schema.json").read_text())
    scorer = ParallelScorer(max_input_tokens=4096)
    short = "Our production payment service is completely down for every customer, and no one can complete a purchase."
    background = "Background: The company sells products online and customers normally use a web checkout to pay for their orders. " * 50
    cases = [("one_field_short", 1, short), ("three_fields_short", 3, short),
             ("eight_fields_short", 8, short), ("three_fields_long", 3, background + "Current incident: " + short),
             ("eight_fields_long", 8, background + "Current incident: " + short)]
    modes = ["parallel", "cached_serial", "independent"]
    records = []
    for case_name, count, context in cases:
        items = list(base.items())
        schema = {((items[i % len(items)][0] if i < len(items) else f"{items[i % len(items)][0]}_{i}")):
                  items[i % len(items)][1] for i in range(count)}
        prepared = scorer.prepare(context, schema)
        # Warm all shapes/kernels once. The benchmark excludes model loading and tokenization.
        warm = {}
        for mode in modes:
            rows, _ = scorer.evaluate(prepared, mode)
            warm[mode] = rows
        checks = {mode: compare(warm["parallel"], warm[mode]) for mode in modes[1:]}
        # Numerical tolerances are declared; do not describe BF16 scores as bitwise equal.
        for mode, check in checks.items():
            if not check["all_choices_agree"] or check["max_probability_difference"] > 0.03:
                raise AssertionError(f"{case_name}/{mode}: {check}")
        del warm, rows
        measurements = {mode: [] for mode in modes}
        for repeat in range(args.repeats):
            # Rotate execution order to avoid always favoring the same mode.
            for mode in modes[repeat % 3:] + modes[:repeat % 3]:
                gc.collect()
                mx.clear_cache()
                rows, metrics = scorer.evaluate(prepared, mode)
                measurements[mode].append(metrics)
                del rows
        medians = {mode: statistics.median(m["total_seconds"] for m in values)
                   for mode, values in measurements.items()}
        record = {"case": case_name, "fields": count,
                  "context_tokens": len(scorer.tokenizer.encode(context, add_special_tokens=False)),
                  "prefix_tokens": len(prepared.prefix_ids),
                  "suffix_tokens": list(map(len, prepared.suffix_ids)),
                  "median_seconds": medians,
                  "speedup_vs_independent": medians["independent"] / medians["parallel"],
                  "speedup_vs_cached_serial": medians["cached_serial"] / medians["parallel"],
                  "validation": checks, "runs": measurements}
        records.append(record)
        result = {"machine": "Apple M5 Pro, 64 GB unified memory", "python": platform.python_version(),
                  "mlx": version("mlx"), "mlx_lm": version("mlx-lm"), "precision": "BF16 weights; FP32 candidate head",
                  "repeats": args.repeats, "probability_tolerance": 0.03,
                  "timing_scope": "Warm kernels; model loading and tokenization excluded; prefill and branch evaluation included",
                  "cases": records}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: record[k] for k in ("case", "prefix_tokens", "median_seconds", "speedup_vs_independent", "speedup_vs_cached_serial")}), flush=True)


if __name__ == "__main__":
    main()
