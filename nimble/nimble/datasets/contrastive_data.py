"""Deterministic construction and audits for schema-conditioned contrast groups."""

import copy
import hashlib
import json
from collections import Counter

from nimble.datasets.dataset_io import canonical
from nimble.evaluation.evaluate_pilot import adapt_input, target_key

VARIANTS = ("base", "evidence_removed", "counterfactual", "paraphrase", "distractor")


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def decode_target(value, question):
    kind, criteria = question["type"], question["criteria"]
    if kind == "choice" and isinstance(value, str) and value in criteria:
        return value
    if kind == "noul" and value in ("false", "true"):
        return value == "true"
    if kind == "score" and isinstance(value, str) and value in map(str, range(len(criteria))):
        return int(value)
    raise ValueError(f"Invalid {kind} target: {value!r}")


def decode_verifier_target(value, question):
    """Accept an explicit unique letter label, never infer a choice by position."""
    if question["type"] == "choice" and isinstance(value, str) and len(value) == 1:
        matches = [key for key in question["criteria"] if key.startswith(value + " — ")]
        if value not in question["criteria"] and len(matches) == 1:
            return matches[0]
    return decode_target(value, question)


def text_at(state, path):
    node = state
    for key in path:
        node = node[int(key)] if isinstance(node, list) else node[key]
    if not isinstance(node, str):
        raise ValueError("An edit path must identify a string")
    return node


def apply_edit(state, edit):
    """Change exactly one literal span, preserving structured state everywhere else."""
    result = copy.deepcopy(state)
    path, old, new = edit["path"], edit["old"], edit["new"]
    original = text_at(result, path)
    if not old.strip() or original.count(old) != 1 or old == new:
        raise ValueError("Edit must match exactly once and change a nonempty span")
    updated = original.replace(old, new, 1)
    if not path:
        return updated
    parent = result
    for key in path[:-1]:
        parent = parent[int(key)] if isinstance(parent, list) else parent[key]
    key = int(path[-1]) if isinstance(parent, list) else path[-1]
    parent[key] = updated
    return result


def append_distractor(state, text):
    if not text.strip():
        raise ValueError("Missing distractor")
    result = copy.deepcopy(state)
    if isinstance(result, str):
        return result + "\n\n" + text
    if isinstance(result, dict):
        if "additional_background" in result:
            raise ValueError("Distractor field would overwrite existing data")
        result["additional_background"] = text
        return result
    if isinstance(result, list):
        result.append({"speaker": "Background", "text": text})
        return result
    raise ValueError("Unsupported state")


def normalize_wrapper_paths(draft, state):
    """Resolve only unambiguous request-wrapper prefixes; never guess an evidence leaf."""
    draft = copy.deepcopy(draft)
    repairs = []
    entries = [*draft["atomic_facts"], draft["removal"], draft["counterfactual"], *draft["paraphrase_edits"]]
    for entry in entries:
        path = entry["path"]
        try:
            text_at(state, path)
            continue
        except (ValueError, KeyError, TypeError, IndexError):
            pass
        for prefix in (["input", "state"], ["state"]):
            if path[:len(prefix)] != prefix:
                continue
            candidate = path[len(prefix):]
            try:
                leaf = text_at(state, candidate)
            except (ValueError, KeyError, TypeError, IndexError):
                continue
            span = entry.get("old", entry.get("evidence", ""))
            if span and span in leaf:
                entry["path"] = candidate
                repairs.append({"original": path, "resolved": candidate})
                break
    return draft, repairs


def generation_jobs(sources):
    """Only original training rows can seed new groups; balance method within type."""
    selected = sorted((r for r in sources if r["split"] == "train"), key=lambda r: r["id"])
    if not selected or len({r["id"] for r in selected}) != len(selected):
        raise ValueError("Expected unique, nonempty training sources")
    counts = Counter()
    jobs = []
    for source in selected:
        kind = source["input"]["questions"]["decision"]["type"]
        method = "c2d" if counts[kind] % 2 == 0 else "d2c"
        counts[kind] += 1
        jobs.append({"group_id": "mc-" + source["id"], "method": method,
                     "source": source, "source_sha256": fingerprint(source)})
    return jobs


def build_group(job, draft, model):
    source = job["source"]
    if source["split"] != "train":
        raise ValueError("Held-out sources cannot be augmented for training")
    question = source["input"]["questions"]["decision"]
    # Rubrics are reused verbatim. C2D synthesizes a new supporting context for
    # the source decision; D2C retains the exact source state, including its type.
    base = (json.loads(draft["base_state_json"]) if job["method"] == "c2d" else copy.deepcopy(source["input"]["state"]))
    if job["method"] == "c2d":
        if not isinstance(base, type(source["input"]["state"])) or len(canonical(base).split()) < 30:
            raise ValueError("C2D requires a substantive new context")
        if canonical(base) == canonical(source["input"]["state"]):
            raise ValueError("C2D must generate a new context")
    draft, path_repairs = normalize_wrapper_paths(draft, base)
    base_target = source["reference"]["target"]
    if not 2 <= len(draft["atomic_facts"]) <= 6:
        raise ValueError("Expected two to six atomic facts with source evidence")
    for fact in draft["atomic_facts"]:
        if not fact["fact"].strip() or not fact["evidence"].strip():
            raise ValueError("Atomic facts require text and evidence")
        if fact["evidence"] not in text_at(base, fact["path"]):
            raise ValueError("Atomic fact evidence is not a literal span in its state path")
    removal = draft["removal"]
    if removal["new"] != "":
        raise ValueError("Evidence removal must be a deletion, not a rewrite")
    states = {"base": base, "evidence_removed": apply_edit(base, removal),
              "counterfactual": apply_edit(base, draft["counterfactual"])}
    if not draft["counterfactual"]["new"].strip():
        raise ValueError("Counterfactual replacement cannot be empty")
    if not draft["paraphrase_edits"]:
        raise ValueError("Paraphrase needs at least one edit")
    paraphrase = base
    for edit in draft["paraphrase_edits"]:
        paraphrase = apply_edit(paraphrase, edit)
    states.update(paraphrase=paraphrase, distractor=append_distractor(base, draft["distractor"]))
    if len({canonical(state) for state in states.values()}) != len(VARIANTS):
        raise ValueError("Every variant must have a distinct context")
    targets = {variant: base_target for variant in VARIANTS}
    targets["evidence_removed"] = decode_target(draft["removed_target"], question)
    targets["counterfactual"] = decode_target(draft["counterfactual_target"], question)
    if targets["counterfactual"] == base_target:
        raise ValueError("Counterfactual must change the target")
    reasons = {"base": draft["base_reason"], "evidence_removed": draft["removed_reason"],
               "counterfactual": draft["counterfactual_reason"], "paraphrase": draft["paraphrase_reason"],
               "distractor": draft["distractor_reason"]}
    rows = []
    for variant in VARIANTS:
        if not reasons[variant].strip():
            raise ValueError("Missing reference rationale")
        input_data = {"state": states[variant], "questions": {"decision": copy.deepcopy(question)}}
        adapt_input(input_data)
        row = {"id": job["group_id"] + "-" + variant, "family": job["group_id"],
               "source_family": source["family"], "split": "train", "method": job["method"],
               "domain": source["domain"], "subtopic": source["subtopic"], "variant": variant,
               "input": input_data,
               "reference": {"target": targets[variant], "reason": reasons[variant],
                             "source": "minicheck_style_generator", "model": model, "human_reviewed": False},
               "provenance": {"source_id": source["id"], "source_split": source["split"],
                              "source_sha256": job["source_sha256"], "source_is_synthetic": True},
               "construction": {"atomic_facts": draft["atomic_facts"],
                                "removal": removal, "counterfactual": draft["counterfactual"],
                                "path_repairs": path_repairs}}
        # These probabilities are reusable ONLY for the unmodified D2C base.
        if job["method"] == "d2c" and variant == "base" and "teacher" in source:
            row["teacher"] = source["teacher"]
            row["teacher_targets"] = source["teacher_targets"]
        rows.append(row)
    return rows


def verification_input(rows):
    """Only questions and randomized opaque-ID contexts; never proposed labels."""
    # Sorting by input hash hides the base/edited order without introducing RNG state.
    ordered = sorted(rows, key=lambda row: fingerprint(row["input"]))
    payload = {"questions": rows[0]["input"]["questions"],
               "contexts": [{"id": f"case-{i}", "state": row["input"]["state"]}
                            for i, row in enumerate(ordered)]}
    mapping = {f"case-{i}": row["id"] for i, row in enumerate(ordered)}
    return payload, mapping


def assess_verification(rows, response, model):
    _, mapping = verification_input(rows)
    judgments = response["judgments"]
    if len(judgments) != len(rows) or {j["id"] for j in judgments} != set(mapping):
        raise ValueError("Verifier must return exactly one judgment for each opaque case ID")
    by_id = {mapping[j["id"]]: j for j in judgments}
    reasons = []
    result = copy.deepcopy(rows)
    for row in result:
        judgment = by_id[row["id"]]
        target = decode_verifier_target(judgment["target"], row["input"]["questions"]["decision"])
        agrees = target == row["reference"]["target"]
        row["verification"] = {**judgment, "target": target, "model": model, "agrees": agrees,
                               "labels_hidden": True, "independent_model": False}
        if target != judgment["target"] and row["input"]["questions"]["decision"]["type"] == "choice":
            row["verification"]["raw_target"] = judgment["target"]
        if not agrees or not judgment["unambiguous"]:
            reasons.append({"id": row["id"], "issue": "label_disagreement" if not agrees else "ambiguous",
                            "proposed": row["reference"]["target"], "verified": target,
                            "reason": judgment["reason"]})
    return result, reasons


def scoring_record(row):
    context, schema = adapt_input(row["input"])
    field = schema["decision"]
    target = row["reference"]["target"]
    value = str(target) if row["input"]["questions"]["decision"]["type"] == "score" else target
    return {"id": row["id"], "group_id": row["family"], "source_family": row["source_family"],
            "context": context, "schema": schema, "field": "decision", "target": value,
            "target_key": target_key(target, row["input"]["questions"]["decision"]["type"]),
            "loss": "candidate_cross_entropy"}


def audit_export(rows, sources):
    source_by_id = {r["id"]: r for r in sources}
    held_out = {canonical(r["input"]["state"]) for r in sources if r["split"] != "train"}
    groups, inputs = {}, set()
    for row in rows:
        source = source_by_id[row["provenance"]["source_id"]]
        if source["split"] != "train" or row["split"] != "train":
            raise ValueError("Training export includes a held-out source")
        if canonical(row["input"]["state"]) in held_out:
            raise ValueError("Training context duplicates a held-out source")
        sig = canonical(row["input"])
        if sig in inputs:
            raise ValueError("Duplicate training input")
        inputs.add(sig)
        groups.setdefault(row["family"], []).append(row)
        if not row["verification"]["agrees"] or not row["verification"]["unambiguous"]:
            raise ValueError("Unverified example in training export")
        scoring_record(row)
    for group in groups.values():
        if Counter(r["variant"] for r in group) != Counter(VARIANTS):
            raise ValueError("Training exports must retain whole contrast groups")
    return {"examples": len(rows), "groups": len(groups),
            "method_counts": dict(Counter(r["method"] for r in rows)),
            "primitive_counts": dict(Counter(r["input"]["questions"]["decision"]["type"] for r in rows)),
            "variant_counts": dict(Counter(r["variant"] for r in rows)),
            "domain_counts": dict(Counter(r["domain"] for r in rows))}
