"""Run Nimble on the frozen sanand0 BANKING77 pilot's original chat prompt.

This is a generative transfer evaluation, outside the native <=26-choice scorer.
Gold labels are kept out of inference inputs; join them only during analysis.
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path

UPSTREAM_REVISION = "3f03e83cf04985f7b10843b787cdf3cbc946aeb4"


def chat_prompt(text, labels):
    allowed = json.dumps(labels, separators=(",", ":"))
    return f'''Classify this customer request into exactly one of the allowed labels below.

Allowed labels:
{allowed}

Request:
{text}

Return ONLY JSON with label (exactly one allowed label) and confidence (integer 0-100): your probability that your chosen label exactly matches the gold routing label. Treat confidence as a probability of correctness, not a vague feeling.'''


def parse_response(text, labels):
    """Match upstream fence/object extraction; also flag invalid contract outputs."""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("No JSON object in response")
        obj = json.loads(match.group())
    if not isinstance(obj, dict) or obj.get("label") not in labels:
        raise ValueError("Missing or invalid routing label")
    confidence = obj.get("confidence")
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 100):
        raise ValueError("Confidence must be a finite number in [0, 100]")
    return {"prediction": obj["label"], "confidence": confidence / 100}


def summarize(rows):
    """Upstream binary correctness Brier/ECE/AUROC; failures count in accuracy."""
    n = len(rows)
    if not n:
        raise ValueError("Empty evaluation")
    valid = [r for r in rows if not r.get("error")]
    correct = sum(bool(r["correct"]) for r in rows)
    accuracy = correct / n
    z = 1.96
    denominator = 1 + z*z/n
    center = (accuracy + z*z/(2*n))/denominator
    half = z*math.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n))/denominator
    result = {"n": n, "valid": len(valid), "invalid": n-len(valid),
              "correct": correct, "accuracy": accuracy,
              "accuracy_ci95": [center-half, center+half],
              "median_latency_s": statistics.median(r["latency_s"] for r in rows),
              "calibration_n": len(valid)}
    if not valid:
        return result
    y, p = [int(r["correct"]) for r in valid], [r["confidence"] for r in valid]
    ece = 0.0
    for i in range(10):
        indexes = [j for j, x in enumerate(p) if i/10 <= x and (x < (i+1)/10 or i == 9 and x <= 1)]
        if indexes:
            ece += abs(sum(p[j]-y[j] for j in indexes))/len(valid)
    positive = [x for x, outcome in zip(p, y) if outcome]
    negative = [x for x, outcome in zip(p, y) if not outcome]
    auc = (sum((a > b) + 0.5*(a == b) for a in positive for b in negative)
           / (len(positive)*len(negative))) if positive and negative else None
    result.update(mean_confidence=statistics.mean(p),
                  confidence_gap=statistics.mean(p)-statistics.mean(y),
                  brier=statistics.mean((a-b)**2 for a, b in zip(p, y)),
                  ece10=ece, auroc=auc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="JSON: labels and rows containing only id/text")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from importlib.metadata import version
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    data = json.loads(args.inputs.read_text())
    labels, rows = data["labels"], data["rows"]
    assert len(labels) == len(set(labels)) == len(rows) == 77
    assert labels == sorted(labels) and len({r["id"] for r in rows}) == 77
    assert all(set(r) == {"id", "text"} for r in rows)
    contract = json.loads((args.adapter / "schema_config.json").read_text())
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "predictions.jsonl"
    if result_path.exists():
        raise FileExistsError("Use a fresh output directory to prevent mixing runs")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": chat_prompt(r["text"], labels)}],
               tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    encoded = [tokenizer(p, return_tensors="pt", add_special_tokens=False) for p in prompts]
    metadata = {"upstream_revision": UPSTREAM_REVISION, "protocol": "original_chat_prompt",
                "model": "bespokelabs/Bespoke-Nimble-9B", "base_model": contract["model"],
                "base_revision": contract["revision"], "adapter_merged": False,
                "adapter_sha256": hashlib.sha256((args.adapter/"adapter_model.safetensors").read_bytes()).hexdigest(),
                "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16", "attention": "sdpa",
                "versions": {p: version(p) for p in ("torch", "transformers", "peft", "accelerate")},
                "do_sample": False, "max_new_tokens": 256, "enable_thinking": False,
                "retries": 0, "warmup_runs": 1, "concurrency": 1,
                "confidence_source": "generated self-reported confidence, not candidate probabilities",
                "prompt_tokens_min": min(x["input_ids"].shape[1] for x in encoded),
                "prompt_tokens_max": max(x["input_ids"].shape[1] for x in encoded),
                "native_scorer": False, "native_max_choices": 26}
    (args.output_dir/"runtime.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("Loading frozen base and published adapter", flush=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(contract["model"], revision=contract["revision"],
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    model.config.use_cache = True
    generation = dict(do_sample=False, max_new_tokens=256, use_cache=True,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        warmup = {k: v.to("cuda") for k, v in encoded[0].items()}
        model.generate(**warmup, **{**generation, "max_new_tokens": 1})
        torch.cuda.synchronize()
        with result_path.open("w") as stream:
            for row, prompt, tokens in zip(rows, prompts, encoded):
                inputs = {k: v.to("cuda") for k, v in tokens.items()}
                torch.cuda.synchronize()
                started = time.perf_counter()
                sequence = model.generate(**inputs, **generation)
                torch.cuda.synchronize()
                latency = time.perf_counter()-started
                generated = sequence[0, inputs["input_ids"].shape[1]:].tolist()
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                result = {"id": row["id"], "raw": raw, "latency_s": latency,
                          "input_tokens": inputs["input_ids"].shape[1], "output_tokens": len(generated),
                          "hit_token_limit": len(generated) == 256 and generated[-1] != tokenizer.eos_token_id,
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
                try:
                    result.update(parse_response(raw, labels))
                except (ValueError, TypeError, KeyError) as exc:
                    result.update(prediction=None, confidence=None, error=str(exc))
                stream.write(json.dumps(result, allow_nan=False)+"\n")
                stream.flush()
                print(f"Completed {row['id']} ({len(generated)} tokens)", flush=True)
    (args.output_dir/"complete.json").write_text(json.dumps({"count": len(rows)}))


if __name__ == "__main__":
    main()
