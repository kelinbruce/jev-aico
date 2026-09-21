"""Compare pinned MLX or CUDA models on saved decisions, without training or API calls."""

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.evaluation.evaluate_pilot import adapt_input, assess, summarize
from nimble.scoring.parallel_schema import SYSTEM_PROMPT

MODELS = {
    "qwen38_27b": {
        "model": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "note": "Official BF16 checkpoint; CUDA candidate scoring without generated reasoning.",
    },
    "gemma3_270m": {
        "model": "mlx-community/gemma-3-270m-it-bf16",
        "revision": "c806ef3a4ed971bd75aaee3346e0fef808512f03",
        "note": "Public BF16 MLX conversion of google/gemma-3-270m-it; not a quantized model.",
    },
    "qwen35_08b": {
        "model": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "note": "Closest Qwen3.5 size to the requested 1B.",
    },
    "qwen35_4b": {
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "note": "Existing baseline, rerun on the same dataset as the other models.",
    },
    "qwen35_9b": {
        "model": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "note": "Official BF16 checkpoint with an untied output head.",
    },
}


def teacher_assessment(row, kind):
    answer = row["teacher"]["answers"]["decision"]
    probabilities = row["teacher_targets"]["probabilities"]
    result = assess(probabilities, row["reference"]["target"], kind)
    if kind in {"choice", "noul"}:
        prediction = answer["choice"] if kind == "choice" else answer["noul"] > 0.5
        result.update(prediction=prediction, correct=prediction == row["reference"]["target"])
    return result


def evaluate_model(name, data, output_root, backend="mlx"):
    if backend == "mlx":
        from nimble.scoring.parallel_scorer import ParallelScorer as Scorer
        import mlx.core as mx
        packages = ("mlx", "mlx-lm", "transformers")
    elif backend == "cuda":
        from nimble.scoring.cuda_scorer import CudaCandidateScorer as Scorer
        import torch
        packages = ("torch", "transformers", "accelerate")
    else:
        raise ValueError("Unknown backend")

    config = MODELS[name]
    checkpoint = (PROJECT_ROOT / ".cache/huggingface/hub" /
                  ("models--" + config["model"].replace("/", "--")) / "snapshots" / config["revision"])
    raw = data.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Dataset must have unique IDs and at least one record")
    settings = {
        **config, "dataset": str(data.resolve()), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "temperature": 1.0, "temperature_fitted": False, "max_input_tokens": 4096,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "scoring": "One-letter candidate logits, FP32 output projection, no generated reasoning",
        "prompt_roles": "user only, instructions prepended" if name == "gemma3_270m" else "system and user",
        "mode": "independent", "split_policy": "all supplied rows; per-split metrics also reported",
        "versions": {"python": platform.python_version(), **{k: version(k) for k in packages}},
    }
    if backend == "cuda":
        settings.update(backend=backend, device=torch.cuda.get_device_name(),
                        cuda_version=torch.version.cuda,
                        attention="eager" if name == "gemma3_270m" else "sdpa", tf32=False)
        from importlib.util import find_spec
        settings["optional_kernels"] = {name: find_spec(name) is not None
                                        for name in ("causal_conv1d", "fla")}
    output = output_root / name
    output.mkdir(parents=True, exist_ok=True)
    settings_path, progress_path = output / "settings.json", output / "rows.jsonl"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("Existing run settings differ; use a new output directory")
    elif progress_path.exists():
        raise ValueError("Progress exists without provenance; use a new output directory")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    rows = [json.loads(l) for l in progress_path.read_text().splitlines()] if progress_path.exists() else []
    if [r["id"] for r in rows] != [r["id"] for r in records[:len(rows)]]:
        raise ValueError("Saved progress does not match dataset order")
    scorer = None
    load_seconds = None
    try:
        if len(rows) < len(records):
            started = time.perf_counter()
            scorer = Scorer(model_path=checkpoint, model_id=config["model"],
                                       revision=config["revision"], temperature=1.0)
            load_seconds = time.perf_counter() - started
        with progress_path.open("a") as stream:
            for row in records[len(rows):]:
                if list(row["input"]["questions"]) != ["decision"]:
                    raise ValueError("Exactly one decision field is required")
                context, schema = adapt_input(row["input"])
                kind = row["input"]["questions"]["decision"]["type"]
                started = time.perf_counter()
                # Every model uses a full independent prefill for a fair one-field comparison.
                prediction = scorer.score(context, schema, mode="independent")
                seconds = time.perf_counter() - started
                field = prediction["fields"]["decision"]
                teacher = teacher_assessment(row, kind)
                if set(field["scores"]) != set(teacher["probabilities"]):
                    raise ValueError("Student and teacher candidates differ")
                result = {"id": row["id"], "type": kind, "domain": row["domain"], "split": row["split"],
                          "reference": row["reference"], "student": assess(field["scores"], row["reference"]["target"], kind),
                          "teacher": teacher, "raw_student": prediction, "elapsed_seconds": seconds}
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                rows.append(result)
                if len(rows) % 10 == 0 or len(rows) == len(records):
                    print(f"{name}: {len(rows)}/{len(records)}", flush=True)
    finally:
        del scorer
        gc.collect()
        if backend == "mlx":
            mx.clear_cache()
        else:
            torch.cuda.empty_cache()
    times = sorted(r["elapsed_seconds"] for r in rows)
    code_counts = Counter()
    for row in rows:
        field = row["raw_student"]["fields"]["decision"]
        code_counts.update([next(code for code, value in field["code_to_choice"].items()
                                 if value == field["value"])])
    report = {**settings, "created_utc": datetime.now(timezone.utc).isoformat(), "count": len(rows),
              "selected_code_counts": dict(sorted(code_counts.items())),
              "load_seconds_this_run": load_seconds,
              "timing": {"total_scoring_seconds": sum(times), "median_seconds": statistics.median(times),
                         "p95_seconds": times[math.ceil(.95 * len(times)) - 1],
                         "peak_active_gib": max(r["raw_student"]["metrics"][f"{backend}_peak_active_gib"] for r in rows)},
              "summary": {who: summarize(rows, who) for who in ("student", "teacher")},
              "teacher_agreement": sum(r["student"]["prediction"] == r["teacher"]["prediction"] for r in rows),
              "by_split": {split: {who: summarize([r for r in rows if r["split"] == split], who)
                                    for who in ("student", "teacher")} for split in sorted({r["split"] for r in rows})},
              "by_domain": {domain: summarize([r for r in rows if r["domain"] == domain], "student")
                            for domain in sorted({r["domain"] for r in rows})}}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "disagreements.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                                                       for r in rows if not r["student"]["correct"] or not r["teacher"]["correct"]))
    return report


def write_comparison(output_root):
    results = {name: json.loads((output_root / name / "results.json").read_text())
               for name in MODELS if (output_root / name / "results.json").exists()}
    if not results:
        raise ValueError("No completed evaluations to summarize")
    if len({r["dataset_sha256"] for r in results.values()}) != 1:
        raise ValueError("Cannot compare evaluations of different datasets")
    lines = ["# Model comparison", "", "All supplied examples are evaluated without training or tuning. "
             "References were generated by GPT-5.6 Sol and have not been human-reviewed. "
             "These are reference-agreement measurements, not established accuracy.", "",
             "| Model | Choice | Noul | Score | Overall | Agreement with Jev | Median seconds/example |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in results.items():
        summary = r["summary"]["student"]
        cells = [f"{summary[k]['correct']}/{summary[k]['count']}" for k in ("choice", "noul", "score")]
        lines.append(f"| {r['model']} | " + " | ".join(cells) +
                     f" | {summary['all']['accuracy']:.1%} | {r['teacher_agreement']}/{r['count']} | {r['timing']['median_seconds']:.3f} |")
    teacher = next(iter(results.values()))["summary"]["teacher"]
    lines += [f"| Saved Jev-1.13.0 | {teacher['choice']['correct']}/{teacher['choice']['count']} | "
              f"{teacher['noul']['correct']}/{teacher['noul']['count']} | {teacher['score']['correct']}/{teacher['score']['count']} | "
              f"{teacher['all']['accuracy']:.1%} | — | — |", "",
              "## Probability metrics and resources", "",
              "Lower is better for Brier, NLL, and expected-score MAE. NLL clips probabilities at 1e-15.", "",
              "| Model | Brier | NLL | Score MAE | Scoring seconds | Peak active GiB |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in results.items():
        s = r["summary"]["student"]
        lines.append(f"| {name} | {s['all']['mean_multiclass_brier']:.4f} | {s['all']['mean_negative_log_likelihood']:.4f} | "
                     f"{s['score']['mean_absolute_score_error']:.4f} | {r['timing']['total_scoring_seconds']:.1f} | {r['timing']['peak_active_gib']:.2f} |")
    lines += ["", "## Reserved eval split", "", "The main table covers every supplied row, including any training and validation splits. "
              "The table below isolates rows marked eval; no model was trained here.", "",
              "| Model | Eval reference agreement |", "| --- | ---: |"]
    for name, r in results.items():
        s = r["by_split"].get("eval", {}).get("student", {}).get("all")
        if s:
            lines.append(f"| {name} | {s['correct']}/{s['count']} ({s['accuracy']:.1%}) |")
    lines += ["", "## Answer-position diagnostic", "",
              "Option order is preserved, with A denoting the first candidate. "
              "Frequent selection of A can reveal a position preference under this prompt and scoring protocol.", "",
              "| Model | First candidate selected |", "| --- | ---: |"]
    for name, r in results.items():
        if "selected_code_counts" in r:
            lines.append(f"| {name} | {r['selected_code_counts'].get('A', 0)}/{r['count']} |")
    lines += ["", "## Method and provenance", "",
              "BF16, unquantized checkpoints with FP32 candidate projection, temperature 1.0, "
              "one-letter answer codes and no generated reasoning. All models use independent full-prompt prefill. "
              "Gemma uses a user-only chat with classification instructions prepended; Qwen uses system/user roles "
              "with thinking disabled. The candidate code is verified to be one token at each model's answer boundary. "
              "The same state, instructions, criteria, and option order are used for every model. "
              "Jev outputs are reused locally; no teacher API calls were made.", "",
              "Probabilities are normalized over the allowed candidates, not calibrated confidence. "
              "Score outputs are distributions over levels with probability-weighted expected scores. "
              "Timing includes tokenization and scoring, excludes model loading, and includes first-example effects; "
              "these single-run timings are not controlled performance benchmarks.", "",
              f"Dataset SHA-256: `{next(iter(results.values()))['dataset_sha256']}`", ""]
    for name, r in results.items():
        lines += [f"- **{name}**: `{r['model']}` at `{r['revision']}`. {r['note']} "
                  f"Backend: `{r.get('backend', 'mlx')}`; device: `{r.get('device', 'Apple Metal')}`. "
                  + (f"Attention: `{r['attention']}`. " if r.get('attention') else "") +
                  f"[Metrics]({name}/results.json), [per-example logits/probabilities]({name}/rows.jsonl), "
                  f"[disagreements]({name}/disagreements.jsonl)."]
    (output_root / "report.md").write_text("\n".join(lines) + "\n")
    (output_root / "comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=[name for name in MODELS if name != "qwen38_27b"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluations/typesafe_diverse_300_gpt56")
    parser.add_argument("--backend", choices=["mlx", "cuda"], default="mlx")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--download", action="store_true", help="Download pinned checkpoints before evaluation")
    args = parser.parse_args()
    if args.download and not args.summarize_only:
        from huggingface_hub import snapshot_download
        for name in args.models:
            config = MODELS[name]
            snapshot_download(config["model"], revision=config["revision"],
                              cache_dir=str(PROJECT_ROOT / ".cache/huggingface/hub"),
                              allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt", "*.model", "README.md"])
    if not args.summarize_only:
        for name in args.models:
            evaluate_model(name, args.data, args.output_dir, args.backend)
            write_comparison(args.output_dir)
    else:
        write_comparison(args.output_dir)


if __name__ == "__main__":
    main()
