"""Run the official unquantized Qwen3.5-4B checkpoint on Apple's Metal GPU."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

from nimble.paths import PROJECT_ROOT

import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="In two short sentences, explain what a language model does.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluations/native/qwen-native-results.json")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal GPU is unavailable. Run in a native macOS terminal with GPU access.")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK to verify native GPU execution.")
    cache = PROJECT_ROOT / ".cache" / "huggingface" / "hub"
    options = dict(revision=REVISION, cache_dir=str(cache), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
    print("Loading original BF16 checkpoint into CPU memory, then moving to Metal GPU...", flush=True)
    started = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager", **options,
    ).eval()
    print("Moving model to Metal GPU...", flush=True)
    model = model.to("mps")
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - started
    devices = sorted({str(p.device) for p in model.parameters()})
    if devices != ["mps:0"]:
        raise RuntimeError(f"Expected all model parameters on MPS, got {devices}.")
    print(f"Loaded in {load_seconds:.2f}s; devices={devices}", flush=True)

    def encode(prompt):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        return tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to("mps")

    inputs = encode(args.prompt)
    torch.mps.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
    torch.mps.synchronize()
    generation_seconds = time.perf_counter() - started
    generated = output[0, inputs["input_ids"].shape[1]:].tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"Generated: {text}", flush=True)

    # Also verify access to next-token logits for the intended fact-checker.
    from nimble.scoring.binary_logits import INSTRUCTION, summarize_logits
    document = "The students are studying for their final exams."
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in ("No", "Yes")]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError(f"Expected single-token labels, got {label_ids}.")
    scores = []
    for claim in ("The students are preparing for an examination.", "The students are on vacation."):
        inputs = encode(f"{INSTRUCTION}\n\nDocument:\n{document}\n\nClaim:\n{claim}")
        torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
        torch.mps.synchronize()
        seconds = time.perf_counter() - started
        score = summarize_logits(logits, label_ids[0][0], label_ids[1][0], inputs["input_ids"].shape[1])
        from dataclasses import asdict
        scores.append({"claim": claim, "seconds": seconds, **asdict(score)})
        print(json.dumps(scores[-1]), flush=True)

    result = {
        "model": MODEL_ID, "revision": REVISION,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "device": devices,
        "dtype": str(next(model.parameters()).dtype), "quantized": False,
        "cpu_fallback_enabled": False, "thinking_enabled": False,
        "load_seconds": load_seconds, "prompt": args.prompt, "response": text,
        "generated_tokens_including_eos": len(generated),
        "generation_seconds_including_prefill": generation_seconds,
        "tokens_per_second_including_prefill": len(generated) / generation_seconds,
        "mps_allocated_gib_at_end": torch.mps.current_allocated_memory() / 2**30,
        "mps_driver_gib_at_end": torch.mps.driver_allocated_memory() / 2**30,
        "fact_check_scores": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
