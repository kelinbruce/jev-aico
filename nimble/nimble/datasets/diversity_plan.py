"""Deterministic coverage plan and validation for 300 synthetic decisions."""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict

from nimble.datasets.dataset_io import canonical

DOMAINS = [
    ("commerce", "Retail, marketplaces, subscriptions, and customer service"),
    ("workplace", "Team coordination, recruiting operations, and project delivery"),
    ("education", "Learning, libraries, course administration, and classroom logistics"),
    ("travel", "Transport, accommodation, itineraries, and accessibility needs"),
    ("software", "Software products, developer workflows, reliability, and data quality"),
    ("public_services", "Municipal services, community facilities, and civic information"),
    ("home", "Household maintenance, food preparation, hobbies, and daily planning"),
    ("media", "Publishing, event production, content organization, and creative collaboration"),
    ("science", "Research workflows, experimental evidence, ecology, and lab logistics"),
    ("supply_chain", "Procurement, inventory, manufacturing, and delivery operations"),
]
MECHANISMS = ["explicit evidence", "negation", "conditional intent", "temporal update",
              "conflicting evidence", "missing information", "scope and exceptions",
              "coreference", "distractor facts", "paraphrase"]
FORMATS = ["text", "object", "dialogue"]
KINDS = ["choice", "choice", "noul", "noul", "score", "score"]
PLAN_VERSION = "diverse-v1"


def subject_inputs(seed):
    return [{"domain": domain, "description": description, "domain_index": i,
             "seed": seed, "plan_version": PLAN_VERSION} for i, (domain, description) in enumerate(DOMAINS)]


def example_plans(subjects, seed):
    if len(subjects) != 50 or len({r["group_id"] for r in subjects}) != 50:
        raise ValueError("Expected five unique subsubjects for each of ten domains")
    counts = Counter(r["domain"] for r in subjects)
    if counts != {domain: 5 for domain, _ in DOMAINS}:
        raise ValueError("Invalid domain coverage")
    plans = []
    for row in sorted(subjects, key=lambda r: (r["domain_index"], r["subtopic_index"])):
        domain_index, subtopic_index = row["domain_index"], row["subtopic_index"]
        ordinal = domain_index * 5 + subtopic_index
        # Four training groups and one held-out group per domain. Five domains
        # supply validation groups; the other five supply evaluation groups.
        indices = list(range(5))
        random.Random(seed + domain_index).shuffle(indices)
        split = ("validation" if domain_index % 2 == 0 else "eval") if subtopic_index == indices[0] else "train"
        slots = []
        for slot, kind in enumerate(KINDS):
            levels = 3 + ordinal % 3
            target = ((ordinal + slot - 4) % levels if kind == "score"
                      else (slot == 3) if kind == "noul" else None)
            slots.append({"slot": slot, "type": kind,
                          "format": FORMATS[(ordinal + slot) % len(FORMATS)],
                          "mechanism": MECHANISMS[(ordinal + slot) % len(MECHANISMS)],
                          "difficulty": ["straightforward", "contextual", "boundary_case"][(ordinal + slot) % 3],
                          "target": target,
                          "score_levels": levels if kind == "score" else None,
                          "choice_options": 3 + (ordinal + slot) % 4 if kind == "choice" else None,
                          "include_no_match": (ordinal + slot) % 3 == 0 if kind == "choice" else False})
        plans.append({**row, "split": split, "slots_json": canonical(slots),
                      "seed": seed, "plan_version": PLAN_VERSION})
    return plans


def normalize_state(state):
    return " ".join(re.findall(r"\w+", canonical(state).lower()))


def state_tokens(state):
    return set(normalize_state(state).split())


def validate_row(row):
    state = row["input"]["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("State must be nonempty text, object, or list")
    if len(normalize_state(state).split()) < 15:
        raise ValueError("State is too short to supply a useful scenario")
    expected_format = row["diversity"]["format"]
    expected_class = {"text": str, "object": dict, "dialogue": list}[expected_format]
    if not isinstance(state, expected_class):
        raise ValueError(f"Expected {expected_format} state")
    question = row["input"]["questions"]["decision"]
    if not isinstance(question["instructions"], str) or not question["instructions"].strip():
        raise ValueError("Missing question instructions")
    kind, criteria, target = question["type"], question["criteria"], row["reference"]["target"]
    if kind == "choice":
        if not isinstance(criteria, dict) or not 3 <= len(criteria) <= 6 or target not in criteria:
            raise ValueError("Invalid choice options or target")
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip() for k, v in criteria.items()):
            raise ValueError("Choice options need keys and descriptions")
    elif kind == "noul":
        if type(target) is not bool or set(criteria) != {"true", "false"}:
            raise ValueError("Invalid Noul reference or criteria")
        if not all(isinstance(v, str) and v.strip() for v in criteria.values()):
            raise ValueError("Missing Noul rubric")
    elif kind == "score":
        if (not isinstance(criteria, list) or not 3 <= len(criteria) <= 5
                or type(target) is not int or not 0 <= target < len(criteria)
                or any(not isinstance(v, str) or not v.strip() for v in criteria)):
            raise ValueError("Invalid score levels or reference")
    else:
        raise ValueError("Unknown primitive")
    if not row["reference"]["reason"].strip():
        raise ValueError("Reference needs a labeling rationale")


def audit(rows):
    if len(rows) != 300 or len({r["id"] for r in rows}) != 300:
        raise ValueError("Expected exactly 300 unique examples")
    if Counter(r["split"] for r in rows) != {"train": 240, "validation": 30, "eval": 30}:
        raise ValueError("Invalid split sizes")
    if Counter(r["input"]["questions"]["decision"]["type"] for r in rows) != {k: 100 for k in ("choice", "noul", "score")}:
        raise ValueError("Expected 100 examples per primitive")
    groups, states, group_counts = {}, set(), Counter()
    for row in rows:
        validate_row(row)
        group = row["family"]
        if groups.setdefault(group, row["split"]) != row["split"]:
            raise ValueError("Subsubject group leaks across splits")
        group_counts[group] += 1
        normalized = normalize_state(row["input"]["state"])
        if normalized in states:
            raise ValueError("Duplicate normalized state")
        states.add(normalized)
    if len(groups) != 50 or set(group_counts.values()) != {6}:
        raise ValueError("Expected 50 groups of six examples")
    near_duplicates = []
    tokens = [state_tokens(r["input"]["state"]) for r in rows]
    for i, left in enumerate(rows):
        for j in range(i + 1, len(rows)):
            similarity = len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j])
            if similarity >= 0.8:
                near_duplicates.append({"left": left["id"], "right": rows[j]["id"],
                                        "token_jaccard": round(similarity, 4),
                                        "cross_split": left["split"] != rows[j]["split"]})
    if any(pair["cross_split"] for pair in near_duplicates):
        raise ValueError("Near-duplicate states cross split boundaries")
    by_split = defaultdict(Counter)
    for row in rows:
        by_split[row["split"]][row["input"]["questions"]["decision"]["type"]] += 1
    return {"examples": len(rows), "unique_states": len(states), "groups": len(groups),
            "split_counts": dict(Counter(r["split"] for r in rows)),
            "primitive_counts": dict(Counter(r["input"]["questions"]["decision"]["type"] for r in rows)),
            "split_primitives": dict(by_split),
            "domain_counts": dict(Counter(r["domain"] for r in rows)),
            "format_counts": dict(Counter(r["diversity"]["format"] for r in rows)),
            "mechanism_counts": dict(Counter(r["diversity"]["mechanism"] for r in rows)),
            "difficulty_counts": dict(Counter(r["diversity"]["difficulty"] for r in rows)),
            "noul_targets": dict(Counter(str(r["reference"]["target"]).lower() for r in rows
                                         if r["input"]["questions"]["decision"]["type"] == "noul")),
            "score_targets": dict(Counter(str(r["reference"]["target"]) for r in rows
                                          if r["input"]["questions"]["decision"]["type"] == "score")),
            "near_duplicate_pairs": near_duplicates,
            "record_sha256": hashlib.sha256(canonical(rows).encode()).hexdigest()}
