"""Join frozen BANKING77 gold labels after inference and compare published results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from nimble.evaluation.evaluate_banking77 import UPSTREAM_REVISION, summarize

CASES_SHA256 = "e9166d12ac05782c68a5407bcc0e9ac630badf3a2cb8f3bdd3d5a59edd53389f"
RESULTS_SHA256 = "c50ebbd6b315ed79cbc551f4f4b278382cdca0e689ba25b5a8a32f5f17410105"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases_path = args.upstream_dir/"data/cases.csv"
    reference_path = args.upstream_dir/"data/results.jsonl"
    assert hashlib.sha256(cases_path.read_bytes()).hexdigest() == CASES_SHA256
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == RESULTS_SHA256
    cases = list(csv.DictReader(cases_path.open()))
    gold = {r["id"]: r["category"] for r in cases}
    predictions = [json.loads(line) for line in (args.run_dir/"predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == len({r["id"] for r in predictions}) == len(gold) == 77
    assert {r["id"] for r in predictions} == set(gold)
    scored = [{**r, "gold": gold[r["id"]], "correct": not r.get("error") and r["prediction"] == gold[r["id"]]} for r in predictions]
    runtime = json.loads((args.run_dir/"runtime.json").read_text())
    models = json.loads((args.upstream_dir/"models.json").read_text())
    reference = {}
    for line in reference_path.read_text().splitlines():
        row = json.loads(line)
        reference.setdefault((row["model_key"], row["id"]), row)
    comparisons = []
    published = {r["key"]: r for r in json.loads((args.upstream_dir/"summary.json").read_text())}
    for key, spec in models.items():
        rows = [reference[key, r["id"]] for r in cases]
        assert all(r["gold"] == gold[r["id"]] and bool(r["correct"]) == (r["prediction"] == gold[r["id"]]) for r in rows)
        summary = summarize(rows)
        # Check our independent metric implementation against the site's snapshot.
        for metric in ("accuracy", "mean_confidence", "brier", "ece10", "auroc", "median_latency_s"):
            assert math.isclose(summary[metric], published[key][metric], abs_tol=1e-12), (key, metric)
        comparisons.append({"key": key, "name": spec["name"], "source": "upstream_saved_results", **summary})
    ours = {"key": "nimble_chat", "name": "Nimble 9B (generative transfer)", "source": "new_local_gpu_run", **summarize(scored)}
    comparisons.append(ours)
    comparisons.sort(key=lambda x: x["accuracy"], reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"upstream_revision": UPSTREAM_REVISION, "cases_sha256": CASES_SHA256,
              "reference_results_sha256": RESULTS_SHA256, "runtime": runtime, "models": comparisons}
    (args.output_dir/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (args.output_dir/"scored.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in scored))
    table = ["| Model | Correct | Accuracy | Brier ↓ | ECE ↓ | Median latency |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in comparisons:
        brier = f"{r['brier']:.3f}" if "brier" in r else "N/A"
        ece = f"{r['ece10']:.3f}" if "ece10" in r else "N/A"
        table.append(f"| {r['name']} | {r['correct']}/77 | {r['accuracy']:.1%} | {brier} | {ece} | {r['median_latency_s']:.3f} s |")
    text = f'''# Nimble on the external BANKING77 pilot

Nimble answered **{ours['correct']}/77 correctly ({ours['accuracy']:.1%})**, with
**{ours['valid']}/77 responses satisfying the label/confidence contract**. The accuracy Wilson 95% interval is
{ours['accuracy_ci95'][0]:.1%}–{ours['accuracy_ci95'][1]:.1%}.

{chr(10).join(table)}

## Protocol and interpretation

- Dataset: the [sanand0 Jev pilot](https://sanand0.github.io/llmevals/jev/), pinned
  to [revision {UPSTREAM_REVISION[:12]}](https://github.com/sanand0/llmevals/tree/{UPSTREAM_REVISION}/jev).
  There are 77 requests, one per BANKING77 intent, with all 77 labels offered every time.
- Checkpoint: published `bespokelabs/Bespoke-Nimble-9B` adapter over its pinned
  Qwen3.5-9B base. No new training, calibration, prompt search, or answer shortlisting.
- **This is a generative transfer test, not the native Nimble scoring path.**
  The native scorer supports at most 26 A–Z choices; the external task needs 77.
  We use the upstream chat prompt verbatim, asking for a label and a confidence
  integer, with greedy decoding, thinking disabled, and a 256-token output cap.
  Nimble was trained for candidate scoring, not this JSON generation protocol.
  These results cannot establish the accuracy or calibration of a native 77-way
  Nimble scorer, which the published interface does not currently implement.
- Confidence here is the generated self-report divided by 100, **not** a softmax
  from Nimble's candidate head. Jev's reference confidence is its chosen-label probability.
- Exact-match scoring uses the dataset gold after inference. The GPU receives only
  request text, IDs, and the global label catalog. Invalid outputs count as failures
  in the 77-example accuracy; calibration uses valid responses only
  ({ours['calibration_n']}/77). There are no retries or result-dependent prompt changes.
- Brier is the upstream binary correctness Brier, not multiclass Brier. ECE uses
  ten equal-width confidence bins. The independent implementation reproduces all
  nine published models' accuracy, Brier, ECE, AUROC, confidence, and median latency.
- Nimble latency measures synchronized local GPU generation at concurrency one,
  after one unscored one-token warmup; it excludes loading, tokenization, and network
  transit. Other rows are the site's saved API latencies. **They are not equivalent
  serving benchmarks**, and this generation latency is not native scorer latency.
- This is a small pilot with one example per class, not a robust banking benchmark.
  Possible base-model pretraining exposure is unknown. Reference models were not
  rerun: their values come from the pinned public results.

Full predictions, token counts, runtime versions, weight and input fingerprints,
and machine-readable metrics are saved beside this report.
'''
    (args.output_dir/"REPORT.md").write_text(text)
    print(json.dumps(ours, indent=2))


if __name__ == "__main__":
    main()
