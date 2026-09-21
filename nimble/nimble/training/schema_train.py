"""Train and evaluate Qwen LoRA adapters for schema-conditioned decisions."""

import argparse
import hashlib
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.modeling_outputs import SequenceClassifierOutput

from nimble.scoring.parallel_schema import MODEL_ID, REVISION, SYSTEM_PROMPT, choice_key, prepare_prompts
from nimble.training.model_loading import load_base
from nimble.training.schema_data import prepare_data, resolve_checkpoint

MAX_CHOICES = 26


class CandidateCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        length = max(len(r["input_ids"]) for r in rows)
        for r in rows:
            n = len(r["candidate_ids"])
            if not 1 <= n <= MAX_CHOICES or len(set(r["candidate_ids"])) != n:
                raise ValueError("Invalid candidate token set")
            if "labels" in r and not 0 <= r["labels"] < n:
                raise ValueError("Gold label points outside the real candidates")
        batch = {
            "input_ids": torch.tensor([[self.pad_id] * (length - len(r["input_ids"])) + r["input_ids"] for r in rows]),
            "attention_mask": torch.tensor([[0] * (length - len(r["input_ids"])) + [1] * len(r["input_ids"]) for r in rows]),
            "candidate_ids": torch.tensor([r["candidate_ids"] + [0] * (MAX_CHOICES - len(r["candidate_ids"])) for r in rows]),
            "candidate_mask": torch.tensor([[True] * len(r["candidate_ids"]) + [False] * (MAX_CHOICES - len(r["candidate_ids"])) for r in rows]),
        }
        if "labels" in rows[0]:
            batch["labels"] = torch.tensor([r["labels"] for r in rows])
        return batch


def candidate_logits(model, inputs):
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                   use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    selected = logits.gather(1, inputs["candidate_ids"])
    return selected.masked_fill(~inputs["candidate_mask"], -torch.inf)


class CandidateTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits = candidate_logits(model, inputs)
        loss = torch.nn.functional.cross_entropy(logits, inputs["labels"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite candidate loss")
        return (loss, SequenceClassifierOutput(loss=loss, logits=logits)) if return_outputs else loss


def decision_result(row, logits):
    logits = logits[:len(row["choices"])].double()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite candidate logits")
    probs = logits.softmax(-1)
    index = logits.argmax().item()
    gold = row.get("labels")
    choices = row["choices"]
    result = {"prediction": choices[index], "probabilities": {choice_key(k): v for k, v in zip(choices, probs.tolist())},
              "logits": {choice_key(k): v for k, v in zip(choices, logits.tolist())}}
    if gold is not None:
        result.update(correct=index == gold, nll=-logits.log_softmax(-1)[gold].item(),
                      brier=sum((p - int(i == gold)) ** 2 for i, p in enumerate(probs.tolist())))
    if row.get("kind") == "score":
        result["prediction"] = int(choices[index])
        result["expected_score"] = sum(int(v) * p for v, p in zip(choices, probs.tolist()))
        if gold is not None:
            result["score_absolute_error"] = abs(result["expected_score"] - int(choices[gold]))
    if row.get("kind") == "noul":
        result["probability_true"] = result["probabilities"]["true"]
    return result


def summarize(rows):
    summary = {}
    for kind in ("all", "choice", "noul", "score"):
        selected = [r for r in rows if kind == "all" or r["kind"] == kind]
        if not selected:
            continue
        group = {"count": len(selected), "correct": sum(r["correct"] for r in selected)}
        group["accuracy"] = group["correct"] / len(selected)
        for metric in ("nll", "brier", "score_absolute_error"):
            if all(metric in r for r in selected):
                group[metric] = sum(r[metric] for r in selected) / len(selected)
        summary[kind] = group
    return summary


def evaluate(model, data, collator, output):
    model.eval()
    rows = []
    # Match Trainer's BF16 autocast in standalone evaluation and after reload.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for r in data:
            batch = {k: v.to("cuda") for k, v in collator([r]).items()}
            result = decision_result(r, candidate_logits(model, batch)[0].cpu())
            rows.append({"id": r["id"], "family": r["family"], "kind": r["kind"],
                         "target": r["target"], **result})
    report = {"summary": summarize(rows), "rows": rows}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"evaluation": output.name, "summary": report["summary"]}), flush=True)
    return report


class SaveSchemaContract(TrainerCallback):
    def __init__(self, contract):
        self.contract = contract

    def on_save(self, args, state, control, **kwargs):
        path = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "schema_config.json"
        path.write_text(json.dumps(self.contract, indent=2) + "\n")


class EpochValidation(TrainerCallback):
    """Measure inner validation after full epochs without consulting outer eval."""
    def __init__(self, validation, collator, output_dir, stop_after_epochs=None):
        self.validation, self.collator = validation, collator
        self.output_dir = Path(output_dir)
        self.stop_after_epochs = stop_after_epochs
        path = self.output_dir / 'epoch_metrics.json'
        self.history = json.loads(path.read_text()) if path.exists() else []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if math.isclose(state.epoch, epoch, abs_tol=1e-6) and epoch > 0:
            if self.validation is not None and not any(r['epoch'] == epoch for r in self.history):
                was_training = model.training
                report = evaluate(model, self.validation, self.collator,
                                  self.output_dir / f'epoch-{epoch}.json')
                model.train(was_training)
                self.history.append({'epoch': epoch, 'step': state.global_step, 'summary': report['summary']})
                (self.output_dir / 'epoch_metrics.json').write_text(json.dumps(self.history, indent=2) + '\n')
            if self.stop_after_epochs and epoch >= self.stop_after_epochs:
                control.should_training_stop = True
        return control


def train(args):
    if any(v <= 0 for v in (args.max_steps, args.learning_rate, args.batch_size, args.gradient_accumulation,
                            args.max_length, args.lora_rank, args.save_steps)):
        raise ValueError("Training sizes, steps and learning rate must be positive")
    if not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError("Warmup must be nonnegative and smaller than max-steps")
    if args.stop_after_epochs is not None and args.stop_after_epochs <= 0:
        raise ValueError('stop-after-epochs must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError("Use a new output directory or explicitly resume a checkpoint")
    set_seed(args.seed)
    model_id, revision = resolve_checkpoint(args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    training, validation, audit = prepare_data(args.data_dir, args.validation, tokenizer, args.max_length, model_id, revision)
    prompt_hash = hashlib.sha256((Path(__file__).parents[1] / "scoring/parallel_schema.py").read_bytes()).hexdigest()
    contract = {"task": "schema_candidate_classification_v1", "model": model_id, "revision": revision,
                "system_prompt": SYSTEM_PROMPT, "prompt_code_sha256": prompt_hash, "max_length": args.max_length,
                "lora_rank": args.lora_rank, "seed": args.seed, "data_audit": audit,
                "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation, "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps, "lr_scheduler": "linear", "weight_decay": 0.0}
    contract["inference_precision"] = "bf16_autocast"
    if args.eval_each_epoch or args.stop_after_epochs:
        contract.update(eval_each_epoch=args.eval_each_epoch, stop_after_epochs=args.stop_after_epochs)
    if args.resume_from_checkpoint:
        previous = json.loads((Path(args.resume_from_checkpoint) / "schema_config.json").read_text())
        # The first pilot already used Trainer's BF16 autocast before this was explicit.
        previous.setdefault("inference_precision", "bf16_autocast")
        if any(previous.get(k) != v for k, v in contract.items()):
            raise ValueError("Resume contract differs; preserve the original data and optimization settings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    base = load_base(model_id, revision)
    targets = [name for name, layer in base.named_modules() if isinstance(layer, torch.nn.Linear) and ".language_model." in name]
    if not targets:
        raise RuntimeError("No language linear layers matched")
    model = get_peft_model(base, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                                          lora_dropout=.05, target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    contract.update(versions={p: version(p) for p in ("torch", "transformers", "peft", "accelerate")},
                    gpu=torch.cuda.get_device_name(), target_modules=targets)
    (args.output_dir / "schema_config.json").write_text(json.dumps(contract, indent=2) + "\n")
    collator = CandidateCollator(tokenizer.pad_token_id)
    initial = evaluate(model, validation, collator, args.output_dir / "before.json") if not args.resume_from_checkpoint else None
    args_hf = TrainingArguments(
        output_dir=str(args.output_dir), max_steps=args.max_steps, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps, lr_scheduler_type="linear", weight_decay=0.,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no", save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        logging_steps=1, report_to="none", remove_unused_columns=False, label_names=["labels"],
        seed=args.seed, data_seed=args.seed, dataloader_num_workers=0, max_grad_norm=1.,
        optim="adamw_torch", logging_nan_inf_filter=False,
    )
    callbacks = [SaveSchemaContract(contract)]
    if args.eval_each_epoch or args.stop_after_epochs:
        callbacks.append(EpochValidation(validation if args.eval_each_epoch else None, collator,
                                         args.output_dir, args.stop_after_epochs))
    trainer = CandidateTrainer(model=model, args=args_hf, train_dataset=training, data_collator=collator,
                               processing_class=tokenizer, callbacks=callbacks)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elapsed = time.monotonic() - started
    final = evaluate(model, validation, collator, args.output_dir / "after.json")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    report = {"before": initial["summary"] if initial else None, "after": final["summary"],
              "training": result.metrics, "optimizer_steps": trainer.state.global_step,
              "training_wall_seconds": elapsed, "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "limitations": ["Model-generated, non-human-reviewed labels", f"Only {len(validation)} exploratory validation examples",
                              "Test split not used", "Single seeded run; no hyperparameter search or calibration",
                              "Schema classification data, not the original MiniCheck curriculum"]}
    if args.eval_each_epoch:
        report['limitations'][3] = 'Hyperparameter selection uses this inner validation set; requires separate outer evaluation'
    elif args.stop_after_epochs:
        report['limitations'][3] = 'Fixed epoch count; no outer-evaluation checkpoint selection; consult run plan for recipe provenance'
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def score(args):
    contract = json.loads((args.adapter / "schema_config.json").read_text())
    prompt_hash = hashlib.sha256((Path(__file__).parents[1] / "scoring/parallel_schema.py").read_bytes()).hexdigest()
    if contract["task"] != "schema_candidate_classification_v1" or contract["prompt_code_sha256"] != prompt_hash:
        raise ValueError("Saved adapter task or prompt implementation differs")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    schema = json.loads(args.schema_file.read_text())
    prepared = prepare_prompts(tokenizer, args.context, schema, contract["max_length"])
    model = PeftModel.from_pretrained(load_base(contract["model"], contract["revision"]), args.adapter).eval()
    collator = CandidateCollator(tokenizer.pad_token_id)
    fields = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, name in enumerate(prepared.names):
            row = {"input_ids": prepared.full_ids[i], "candidate_ids": prepared.candidate_ids[i], "choices": prepared.choices[i]}
            if schema[name]["type"] == "boolean":
                row["kind"] = "noul"
            batch = {k: v.to("cuda") for k, v in collator([row]).items()}
            fields[name] = decision_result(row, candidate_logits(model, batch)[0].cpu())
    print(json.dumps({"output": {name: r["prediction"] for name, r in fields.items()}, "fields": fields}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument('--model', default=MODEL_ID)
    fit.add_argument('--revision', help='Pinned model revision; required when changing the default model')
    fit.add_argument("--data-dir", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--max-steps", type=int, default=30)
    fit.add_argument("--warmup-steps", type=int, default=3)
    fit.add_argument("--batch-size", type=int, default=1)
    fit.add_argument("--gradient-accumulation", type=int, default=8)
    fit.add_argument("--learning-rate", type=float, default=2e-5)
    fit.add_argument("--max-length", type=int, default=1024)
    fit.add_argument("--lora-rank", type=int, default=16)
    fit.add_argument("--save-steps", type=int, default=10)
    fit.add_argument("--seed", type=int, default=17)
    fit.add_argument("--resume-from-checkpoint")
    fit.add_argument('--eval-each-epoch', action='store_true', help='Measure validation at complete epoch boundaries')
    fit.add_argument('--stop-after-epochs', type=int, help='Stop at an epoch boundary while retaining the max-steps LR schedule')
    predict = commands.add_parser("score")
    predict.add_argument("--adapter", type=Path, required=True)
    predict.add_argument("--context", required=True)
    predict.add_argument("--schema-file", type=Path, required=True)
    args = parser.parse_args()
    train(args) if args.command == "train" else score(args)


if __name__ == "__main__":
    main()
