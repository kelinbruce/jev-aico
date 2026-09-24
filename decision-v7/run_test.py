#!/usr/bin/env python3
"""Run decision-v7 through the TypeSafe SDK (Python 3.10+)."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

BASE_URL = "http://127.0.0.1:55733"
ROOT = Path(__file__).resolve().parent


def make_questions(row):
    from typesafe_sdk import Choice, Noul, Score
    classes = {"choice": Choice, "noul": Noul, "score": Score}
    return {qid: classes[q["type"]](**{k: q[k] for k in ("instructions", "criteria") if k in q})
            for qid, q in row["questions"].items()}


def predict(response, qid, q):
    if q["type"] == "choice":
        value = response.choices[qid].choice
        if value not in q["criteria"]:
            raise ValueError("Unknown choice")
        return value
    if q["type"] == "noul":
        value = float(response.nouls[qid].noul)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid Noul probability")
        return value > 0.5  # argmax([p(false), p(true)]): ties choose false.
    probabilities = response.scores[qid].probabilities
    values = [float(probabilities[i] if i in probabilities else probabilities[str(i)])
              for i in range(len(q["criteria"]))]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values) or sum(values) <= 0:
        raise ValueError("Invalid Score probabilities")
    return max(range(len(values)), key=values.__getitem__)


def summarize(results):
    def metrics(items):
        correct = sum(x["correct"] for x in items)
        successful = sum(x["error"] is None for x in items)
        errors = [x["score_absolute_error"] for x in items if x.get("score_absolute_error") is not None]
        return {"questions": len(items), "correct": correct, "errors": len(items) - successful,
                "accuracy": correct / len(items) if items else None,
                "accuracy_success_only": correct / successful if successful else None,
                "score_mae": sum(errors) / len(errors) if errors else None,
                "score_mae_n": len(errors)}
    questions = [q for r in results for q in r["questions"]]
    groups = {}
    for key in ("source", "type"):
        buckets = defaultdict(list)
        for q in questions:
            buckets[q[key]].append(q)
        groups["by_" + key] = {k: metrics(v) for k, v in sorted(buckets.items())}
    elapsed = sorted(r["latency_ms"] for r in results)
    clean = [q for q in questions if q.get("variant", "clean") == "clean"]
    families = defaultdict(lambda: defaultdict(list))
    for q in questions:
        families[q.get("family", q["source"])][str(q["gold"])].append(q["correct"])
    balanced = [sum(sum(v) / len(v) for v in labels.values()) / len(labels)
                for labels in families.values()]
    return {"records": len(results), **metrics(questions), **groups,
            "clean": metrics(clean),
            "family_balanced_accuracy": sum(balanced) / len(balanced),
            "latency_ms": {"mean": sum(elapsed) / len(elapsed),
                           "p50": elapsed[math.ceil(len(elapsed) * .5) - 1],
                           "p95": elapsed[math.ceil(len(elapsed) * .95) - 1]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default="kev-latest")
    parser.add_argument("--data", type=Path, default=ROOT / "development.jsonl")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, help="Run only the first N records")
    parser.add_argument("--dry-run", action="store_true", help="Validate requests without calling the server")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    from typesafe_sdk import TypeSafeClient
    from typesafe_sdk import RetryPolicy

    data = args.data.read_bytes()
    rows = [json.loads(line) for line in data.split(b"\n") if line.strip()]
    if args.data.resolve() == (ROOT / "development.jsonl").resolve():
        manifest = json.loads((ROOT / "manifest.json").read_text())
        if hashlib.sha256(data).hexdigest() != manifest["files"]["development.jsonl"]["sha256"]:
            raise ValueError("Dataset SHA-256 mismatch")
    rows = rows[:args.limit]
    if not rows:
        parser.error("Dataset is empty")
    # Validate every request before starting a potentially long run.
    requests = [make_questions(row) for row in rows]
    if args.dry_run:
        print(f"Validated {len(rows)} records / {sum(len(r['questions']) for r in rows)} questions; no requests sent.")
        return 0
    out = args.out or ROOT / "output" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    config = {"base_url": args.base_url, "model": args.model, "dataset_sha256": hashlib.sha256(data).hexdigest(),
              "records": len(rows), "timeout": 300.0, "max_retries": 0}
    (out / "run.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    results = []
    interrupted = False
    try:
        with TypeSafeClient(base_url=args.base_url, model=args.model,
                            api_key=os.environ.get("LOCAL_MODEL_API_KEY", "local"),
                            retry=RetryPolicy(max_retries=0, timeout=300.0)) as client, \
                (out / "results.jsonl").open("w", encoding="utf-8") as f:
            for index, (row, questions) in enumerate(zip(rows, requests), 1):
                start = time.perf_counter()
                response, error, raw = None, None, None
                try:
                    response = client.system_one(state=row["state"], questions=questions)
                    raw = response.model_dump(mode="json")
                except Exception as exc:
                    error = type(exc).__name__
                latency = (time.perf_counter() - start) * 1000
                answers = []
                for qid, q in row["questions"].items():
                    prediction, qerror, score_error = None, error, None
                    if qerror is None:
                        try:
                            prediction = predict(response, qid, q)
                            if q["type"] == "score":
                                probs = response.scores[qid].probabilities
                                total = sum(probs.values())
                                expected = sum(int(k) * v for k, v in probs.items()) / total
                                score_error = abs(expected - q["label"])
                        except Exception as exc:
                            qerror = type(exc).__name__
                    answers.append({"id": qid, "type": q["type"], "source": q.get("src", "unknown"),
                                    "family": row.get("_meta", {}).get("family", q.get("src", "unknown")),
                                    "variant": row.get("_meta", {}).get("variant", "clean"),
                                    "score_absolute_error": score_error,
                                    "gold": q["label"], "prediction": prediction,
                                    "correct": qerror is None and prediction == q["label"], "error": qerror})
                result = {"index": index, "id": row.get("_meta", {}).get("id", str(index)),
                          "latency_ms": latency, "questions": answers, "response": raw}
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{index}/{len(rows)}] correct={sum(q['correct'] for q in answers)}/{len(answers)} "
                      f"errors={sum(q['error'] is not None for q in answers)} latency={latency:.0f}ms", flush=True)
                if index == 1 and error:
                    print(f"First request failed ({error}); stopping. Check server URL and authentication.")
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; saving completed results.")
    if results:
        summary = {**summarize(results), "requested_records": len(rows),
                   "complete": len(results) == len(rows) and not interrupted}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {out.resolve()}")
    return int(interrupted or len(results) != len(rows) or any(q["error"] for r in results for q in r["questions"]))


if __name__ == "__main__":
    raise SystemExit(main())
