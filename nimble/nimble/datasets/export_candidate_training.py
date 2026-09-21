"""Tokenize verified training inputs for single-token candidate cross-entropy."""

import argparse
import json
import random
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.scoring.parallel_schema import MODEL_ID, REVISION, prepare_prompts
from nimble.datasets.contrastive_data import fingerprint


def tokenize_record(row, tokenizer, seed, max_input_tokens):
    schema = json.loads(json.dumps(row["schema"]))
    field = schema[row["field"]]
    # Gold values remain semantic values. The correct answer code is derived only
    # after permuting choices, so learning A/B/C positions cannot solve the task.
    rng = random.Random(f"{seed}:{row['id']}")
    rng.shuffle(field["choices"])
    prepared = prepare_prompts(tokenizer, row["context"], schema, max_input_tokens)
    index = prepared.names.index(row["field"])
    target_index = prepared.choices[index].index(row["target"])
    return {"id": row["id"], "group_id": row["group_id"], "source_family": row["source_family"],
            "prompt_token_ids": prepared.full_ids[index],
            "candidate_token_ids": prepared.candidate_ids[index],
            "target_index": target_index,
            "target_token_id": prepared.candidate_ids[index][target_index],
            "code_to_choice": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", prepared.choices[index])),
            "loss": "candidate_cross_entropy_at_last_prompt_position"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/evidence_curated_1000_gpt56/train_scoring.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--revision', help='Required for a non-default model')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from nimble.training.schema_data import resolve_checkpoint
    model_id, revision = resolve_checkpoint(args.model, args.revision)
    if model_id == MODEL_ID and revision == REVISION:
        model_path = PROJECT_ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots" / REVISION
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No verified training rows to tokenize")
    output = args.output or args.data.parent / "train_tokens.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Complete validation before writing, so errors cannot leave a partial export.
    records = [tokenize_record(row, tokenizer, args.seed, args.max_input_tokens) for row in rows]
    output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records))
    manifest = {"model": model_id, "revision": revision, "rows": len(records),
                "source_sha256": fingerprint(rows), "records_sha256": fingerprint(records),
                "shuffle_seed": args.seed, "max_prompt_tokens": max(len(r["prompt_token_ids"]) for r in records),
                "loss": "cross entropy over candidate logits at final prompt position",
                "label_or_rationale_in_prompt": False}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
