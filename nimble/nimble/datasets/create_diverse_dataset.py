"""Generate 300 diverse decisions with Curator, then annotate them with Jev."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.datasets.dataset_io import (
    canonical, request_for, training_record, validate_teacher, write_jsonl,
)
from nimble.datasets.diversity_plan import audit, example_plans, subject_inputs
from nimble.datasets.typesafe_curator_bridge import bridge

GENERATOR_MODEL = "gpt-5.6-sol"


def generation_params(model, output_tokens):
    """Reasoning models need a completion budget that also covers reasoning."""
    if model.startswith(("gpt-5", "gpt-6")):
        return {"reasoning_effort": "medium", "max_completion_tokens": 8192}
    return {"temperature": 0.8, "max_tokens": output_tokens}


def configure():
    os.environ["TELEMETRY_ENABLED"] = "false"
    os.environ["CURATOR_VIEWER"] = "0"
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "tiktoken"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def generator_backend():
    # Use only the explicit official endpoint; never inherit a different base URL.
    return {"base_url": "https://api.openai.com/v1", "max_concurrent_requests": 16,
            "max_requests_per_minute": 120, "max_tokens_per_minute": 1_000_000,
            "max_retries": 2, "request_timeout": 180, "require_all_responses": True}


def generate_sources(output, model, seed, *, plan_builder=example_plans, auditor=audit,
                     id_prefix="diverse", cache=None, runner=None, exclusions=None,
                     backend_overrides=None):
    from nimble.datasets.diverse_stages import (
        SubsubjectGenerator, ScenarioGenerator, ChoiceDraft, NoulDraft, ScoreDraft, unpack_example,
    )
    cache = cache or PROJECT_ROOT / ".cache" / "curator-diverse"
    run = runner or (lambda generator, rows, working_dir: generator(rows, working_dir=working_dir))
    backend = {**generator_backend(), **(backend_overrides or {})}
    plan_path = output / "generation_plan.jsonl"
    if plan_path.exists():
        plans = read_jsonl(plan_path)
        if len(plans) != 50 or any(row["seed"] != seed for row in plans):
            raise ValueError("Saved generation plan differs; use a new output directory")
        print("Stage 1/3: reusing fifty saved subsubject plans.", flush=True)
    else:
        print("Stage 1/3: generating five subsubjects in each of ten domains.", flush=True)
        subjects = SubsubjectGenerator(model_name=model, backend="openai", backend_params=backend,
                                       generation_params=generation_params(model, 2500))
        inputs = subject_inputs(seed)
        if exclusions:
            inputs = [{**row, "exclude_subtopics": exclusions.get(row["domain"], [])} for row in inputs]
        result = run(subjects, inputs, working_dir=str(cache / "subsubjects"))
        subtopics = []
        for row in result.dataset:
            values = json.loads(row["subsubjects_json"])["subsubjects"]
            if len(values) != 5 or len({v["title"].strip().lower() for v in values}) != 5:
                raise ValueError("Generator must produce exactly five distinct subsubjects per domain")
            for index, value in enumerate(values):
                subtopics.append({"domain": row["domain"], "domain_index": row["domain_index"],
                                  "subtopic_index": index, "group_id": f"{row['domain']}-{index + 1:02d}",
                                  "subtopic": value["title"], "description": value["description"],
                                  "actors_json": canonical(value["actors"])})
        plans = plan_builder(subtopics, seed)
        write_jsonl(plan_path, plans)
    print("Stage 2/3: generating one scenario per request from the saved coverage plan.", flush=True)
    jobs = []
    for plan in plans:
        for slot in json.loads(plan["slots_json"]):
            jobs.append({**{k: v for k, v in plan.items() if k != "slots_json"},
                         "example_id": f"{id_prefix}-{len(jobs) + 1:03d}",
                         "slot_json": canonical(slot)})
    progress_path = output / "source_progress.jsonl"
    attempts_path = output / "generation_attempts.jsonl"
    accepted = {row["id"]: row for row in read_jsonl(progress_path)} if progress_path.exists() else {}
    attempts = read_jsonl(attempts_path) if attempts_path.exists() else []
    job_by_id = {row["example_id"]: row for row in jobs}
    from nimble.datasets.diversity_plan import validate_row
    for row in accepted.values():
        validate_row(row)
        if (row["id"] not in job_by_id or row["reference"]["model"] != model
                or row["family"] != job_by_id[row["id"]]["group_id"]):
            raise ValueError("Saved generation progress does not match this run")
    for kind, schema in (("choice", ChoiceDraft), ("noul", NoulDraft), ("score", ScoreDraft)):
        generator = ScenarioGenerator(model_name=model, response_format=schema, backend="openai",
                                      backend_params=backend, generation_params={
                                          **generation_params(model, 2200),
                                          "response_format": {"type": "json_schema", "json_schema": {
                                              "name": "typed_scenario", "strict": True,
                                              "schema": schema.model_json_schema()}}})
        pending = [row for row in jobs if json.loads(row["slot_json"])["type"] == kind
                   and row["example_id"] not in accepted]
        for attempt in range(3):
            if not pending:
                break
            result = run(generator, pending, working_dir=str(cache / "individual_scenarios" / kind))
            retry = []
            for row in result.dataset:
                try:
                    record = unpack_example(row, model)
                    accepted[record["id"]] = record
                    number = 1 + sum(item.get("id") == row["example_id"] for item in attempts)
                    attempts.append({"id": row["example_id"], "attempt": number, "accepted": True})
                except (ValueError, KeyError, TypeError) as error:
                    number = 1 + sum(item.get("id") == row["example_id"] for item in attempts)
                    attempts.append({"id": row["example_id"], "attempt": number,
                                     "accepted": False, "error": str(error)})
                    retry.append({**{k: v for k, v in row.items() if k != "draft_json"},
                                  "repair_error": str(error), "attempt": attempt + 1})
            pending = retry
            write_jsonl(output / "generation_attempts.jsonl", attempts)
            write_jsonl(output / "source_progress.jsonl", sorted(accepted.values(), key=lambda r: r["id"]))
            print(f"Accepted {len(accepted)}/{len(jobs)} examples; {len(pending)} {kind} examples need repair.", flush=True)
        if pending:
            raise ValueError(f"Could not generate all {kind} examples after bounded repairs")
    records = sorted(accepted.values(), key=lambda r: r["id"])
    auditor(records)
    write_jsonl(output / "sources.jsonl", records)
    (output / "source_manifest.json").write_text(json.dumps({
        "generator_model": model, "seed": seed, "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_parameters": generation_params(model, 2200),
        "source_sha256": hashlib.sha256(canonical(records).encode()).hexdigest(),
        "generation": "Curator: fixed domains -> generated subsubjects -> individually generated scenarios",
    }, indent=2) + "\n")
    return records


def annotate(records, model, api_key):
    from nimble.datasets.diverse_stages import JevLabeler
    import litellm
    print(f"Stage 3/3: labeling {len(records)} examples with Jev through Curator.", flush=True)
    inputs = [{"request_json": canonical(request_for(row, model)), "source_json": canonical(row)} for row in records]
    with bridge(api_key) as (base_url, adapter_token):
        labeler = JevLabeler(model_name=model, backend="openai", backend_params={
            "base_url": base_url, "api_key": adapter_token, "max_concurrent_requests": 2,
            "max_requests_per_minute": 60, "max_tokens_per_minute": 100_000,
            "max_retries": 2, "request_timeout": 90, "require_all_responses": True,
            "in_mtok_cost": 0, "out_mtok_cost": 0,
        })
        # Compatibility with Curator 0.1.29's LiteLLM token estimator; not an API limit.
        litellm.register_model({model: {"max_output_tokens": 4096}})
        result = labeler(inputs, working_dir=str(PROJECT_ROOT / ".cache" / "curator-diverse" / "teacher"))
    return sorted([json.loads(row["record_json"]) for row in result.dataset], key=lambda r: r["id"])


def teacher_targets(answer):
    """Derive normalized training targets while retaining the raw API answer."""
    if answer["type"] == "noul":
        probabilities = {"false": 1 - answer["noul"], "true": answer["noul"]}
    else:
        probabilities = answer["probabilities"]
    mass = sum(probabilities.values())
    normalized = {key: value / mass for key, value in probabilities.items()}
    result = {"probabilities": normalized, "raw_probability_sum": mass,
              "normalization_applied": abs(mass - 1.0) > 1e-8}
    if answer["type"] == "score":
        result["expected_score_from_normalized_probabilities"] = sum(int(key) * p for key, p in normalized.items())
    return result


def export(records, output, generator_model, teacher_model, seed, auditor=audit):
    coverage = auditor(records)
    review = []
    for row in records:
        validate_teacher(row, row["teacher"], teacher_model)
        answer = row["teacher"]["answers"]["decision"]
        row["teacher_targets"] = teacher_targets(answer)
        kind = answer["type"]
        predicted = (answer["choice"] if kind == "choice" else answer["noul"] > 0.5
                     if kind == "noul" else int(max(answer["probabilities"], key=answer["probabilities"].get)))
        if predicted != row["reference"]["target"]:
            review.append({"id": row["id"], "split": row["split"], "reason": "generator_teacher_disagreement",
                           "reference_target": row["reference"]["target"], "teacher_prediction": predicted,
                           "input": row["input"], "reference": row["reference"], "teacher": row["teacher"]})
    coverage["record_sha256"] = hashlib.sha256(canonical(records).encode()).hexdigest()
    write_jsonl(output / "all.jsonl", records)
    for split in ("train", "validation", "eval"):
        selected = [row for row in records if row["split"] == split]
        write_jsonl(output / f"{split}.jsonl", selected)
        if split != "train":
            write_jsonl(output / f"{split}_prompts.jsonl", [
                {"id": row["id"], "messages": training_record(row)["messages"][:2],
                 "expected_target": row["reference"]["target"]} for row in selected])
    write_jsonl(output / "train_sft.jsonl", [training_record(row) for row in records if row["split"] == "train"])
    write_jsonl(output / "review_queue.jsonl", review)
    (output / "diversity_report.json").write_text(json.dumps(coverage, indent=2) + "\n")
    manifest = {
        "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "curator_version": version("bespokelabs-curator"), "seed": seed,
        "generator_model": generator_model, "teacher_model_requested": teacher_model,
        "generator_parameters": generation_params(generator_model, 2200),
        "teacher_models_returned": sorted({r["teacher"]["model"] for r in records}),
        "counts": coverage["split_counts"], "primitive_counts": coverage["primitive_counts"],
        "teacher_usage": {k: sum(r["teacher"]["usage"][k] for r in records)
                          for k in ("input_tokens", "output_tokens")},
        "source_sha256": hashlib.sha256(canonical([{k: v for k, v in r.items()
                                                    if k not in {"teacher", "teacher_targets", "teacher_validation_error"}}
                                                   for r in records]).encode()).hexdigest(),
        "generator_teacher_disagreements": len(review), "human_reviewed": False,
        "teacher_cost_usd": None,
        "generation_reference": "https://github.com/bespokelabsai/curator/blob/main/examples/ungrounded-qa/ungrounded_qa.py",
        "notes": ["Generator references are provisional; agreement with Jev is not proof of truth.",
                  "All six examples from each subsubject remain in the same split.",
                  "Inputs and rubrics are synthetic. Dataset requires human review before use as a benchmark.",
                  "No examples are removed merely because Jev disagrees; disagreements are queued for review.",
                  "Coverage tags record requested generation controls, not independently verified difficulty.",
                  "Text Jaccard checks do not guarantee absence of semantic duplicates.",
                  "Reference hard labels populate SFT; full training records retain teacher distributions.",
                  "Raw teacher numbers are preserved; teacher_targets contains normalized copies for training.",
                  "Numeric validation allows the bounded effect of two-decimal API rounding.",
                  "Teacher usage describes saved responses, not incremental usage of cached reruns."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "typesafe_diverse_300_gpt56")
    parser.add_argument("--generator-model", default=GENERATOR_MODEL)
    parser.add_argument("--teacher-model", default="jev-1.13.0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--generate-only", action="store_true", help="Generate sources without Jev calls")
    args = parser.parse_args()
    if args.offline:
        records = read_jsonl(args.output / "all.jsonl")
        report = audit(records)
        for row in records:
            validate_teacher(row, row["teacher"], args.teacher_model)
        print(json.dumps(report, indent=2))
        return
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=False)
    source_path = args.output / "sources.jsonl"
    if not source_path.exists() and not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required to generate scenarios")
    if not args.generate_only and not os.environ.get("TYPESAFE_API_KEY"):
        parser.error("TYPESAFE_API_KEY is required to label scenarios")
    configure()
    args.output.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        records = read_jsonl(source_path)
        metadata = json.loads((args.output / "source_manifest.json").read_text())
        if metadata["generator_model"] != args.generator_model or metadata["seed"] != args.seed:
            parser.error("Existing source settings differ; choose a new output directory")
        if hashlib.sha256(canonical(records).encode()).hexdigest() != metadata["source_sha256"]:
            parser.error("Source checksum mismatch; review edits before relabeling")
        audit(records)
        print("Using saved, validated source scenarios.", flush=True)
    else:
        records = generate_sources(args.output, args.generator_model, args.seed)
    if not args.generate_only:
        records = annotate(records, args.teacher_model, os.environ["TYPESAFE_API_KEY"])
        print(json.dumps(export(records, args.output, args.generator_model, args.teacher_model, args.seed), indent=2))
    print(f"Saved output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
