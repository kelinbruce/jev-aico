"""Curator stages for hierarchical scenario generation and Jev annotation.

Import after configuring local cache/viewer/telemetry settings.
"""

import json

from bespokelabs import curator
from pydantic import BaseModel, ConfigDict

from nimble.datasets.dataset_io import canonical, validate_teacher
from nimble.datasets.diversity_plan import validate_row


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Subsubject(StrictModel):
    title: str
    description: str
    actors: list[str]


class Subsubjects(StrictModel):
    subsubjects: list[Subsubject]


class StructuredState(StrictModel):
    context: str
    evidence: list[str]
    request: str


class DialogueTurn(StrictModel):
    speaker: str
    text: str


class BaseDraft(StrictModel):
    state: str | StructuredState | list[DialogueTurn]
    instructions: str
    reason: str


class Option(StrictModel):
    key: str
    description: str


class ChoiceDraft(BaseDraft):
    options: list[Option]
    target: str


class NoulDraft(BaseDraft):
    yes_description: str
    no_description: str
    target: bool


class ScoreDraft(BaseDraft):
    levels: list[str]
    target: int


class SubsubjectGenerator(curator.LLM):
    response_format = Subsubjects

    def prompt(self, row):
        return (
            "Design exactly five distinct subsubjects for a synthetic typed-decision dataset. "
            "Each must support practical routing, evidence checking, yes/no judgments, and ordered ratings. "
            "Choose concrete, non-overlapping situations, not synonyms. Include different actors and "
            "operational settings. All facts in later examples will be fictional and self-contained; "
            "avoid tasks requiring external knowledge, medical decisions, legal advice, or current news. "
            f"Domain: {row['domain']}. Scope: {row['description']}. "
            f"Diversity seed: {row['seed']}. Return a title, description, and 2-4 actor roles for each."
            + (" Avoid these existing subsubjects: " + json.dumps(row["exclude_subtopics"])
               if row.get("exclude_subtopics") else "")
        )

    def parse(self, row, response):
        return {**row, "subsubjects_json": response.model_dump_json()}


class ScenarioGenerator(curator.LLM):
    def prompt(self, row):
        slot = json.loads(row["slot_json"])
        if slot["type"] == "choice":
            requirement = f"Create exactly {slot['choice_options']} TOTAL options, INCLUDING any none_of_above option in that total, with descriptive keys and mutually exclusive rubrics. "
            requirement += ("Include key none_of_above and make it the correct target; none of the substantive options fits the evidence."
                            if slot["include_no_match"] else "Make one substantive option unambiguously correct; vary option position.")
        elif slot["type"] == "noul":
            requirement = f"Design a yes/no question whose correct target is {str(slot['target']).lower()}. Supply explicit yes_description and no_description."
        else:
            requirement = (f"Create exactly {slot['score_levels']} ordered levels, indexed from 0. "
                           f"Design evidence whose correct target index is {slot['target']}. "
                           "Write a concrete, self-contained definition for EVERY level, low to high; do not reverse the scale.")
        state_format = {
            "text": "state MUST be a plain STRING of natural text, not an object or list.",
            "object": "state MUST be an OBJECT with context (string), evidence (list of strings), and request (string).",
            "dialogue": "state MUST be a LIST of 3-5 dialogue turns; each turn has speaker and text. Do not wrap it in an object.",
        }[slot["format"]]
        return (
            f"Create ONE fictional, self-contained {slot['type']} decision example.\n"
            f"Domain: {row['domain']}. Subsubject: {row['subtopic']}. Setting: {row['description']}.\n"
            f"Actor roles: {row['actors_json']}.\n"
            f"Required evidence mechanism: {slot['mechanism']}; requested difficulty: {slot['difficulty']}. "
            "Make the mechanism actually matter to the answer. A boundary case still needs a defensible target under an explicit rubric.\n"
            + state_format + " Aim for 45-130 words of state. Use concrete details.\n"
            + requirement + "\n"
            "Choose a domain-relevant semantic task: routing, extraction among candidates, verification, intent, readiness, "
            "relevance, preference, impact, or completeness. Do not make every task urgency or customer support. "
            "All necessary facts, definitions, and policies must be provided. No outside knowledge is required. "
            "Use different scenarios, actors, and facts across variants. "
            "Question instructions and option/level descriptions must be clear and nonempty. "
            "Do not embed model instructions, reference labels, or grading notes in the state. "
            "Return the hard reference target and a concise reason citing evidence; do not generate probabilities.\n"
            f"Variation identifier: {row['example_id']}."
            + ("\nRepair the prior validation error: " + row["repair_error"] if row.get("repair_error") else "")
        )

    def parse(self, row, response):
        return {**row, "draft_json": response.model_dump_json()}


def unpack_example(row, generator_model):
    slot = json.loads(row["slot_json"])
    kind = slot["type"]
    schema = {"choice": ChoiceDraft, "noul": NoulDraft, "score": ScoreDraft}[kind]
    draft = schema.model_validate_json(row["draft_json"])
    if kind == "choice":
        criteria = {opt.key: opt.description for opt in draft.options}
        if len(criteria) != len(draft.options) or len(criteria) != slot["choice_options"]:
            raise ValueError(f"Expected {slot['choice_options']} total unique options including none_of_above if present; got {len(draft.options)} entries and {len(criteria)} unique keys")
        if slot["include_no_match"] and draft.target != "none_of_above":
            raise ValueError("Required no-match example must select none_of_above")
    elif kind == "noul":
        criteria = {"true": draft.yes_description, "false": draft.no_description}
    else:
        criteria = draft.levels
        if len(criteria) != slot["score_levels"]:
            raise ValueError("Wrong number of score levels")
    if slot["target"] is not None and draft.target != slot["target"]:
        raise ValueError("Reference target does not match the coverage plan")
    record = {
        "id": row["example_id"], "split": row["split"], "family": row["group_id"],
        "domain": row["domain"], "subtopic": row["subtopic"],
        "diversity": {"format": slot["format"], "mechanism": slot["mechanism"],
                      "difficulty": slot["difficulty"], "plan_version": row["plan_version"]},
        "input": {"state": draft.model_dump()["state"], "questions": {"decision": {
            "type": kind, "instructions": draft.instructions, "criteria": criteria,
        }}},
        "reference": {"target": draft.target, "reason": draft.reason,
                      "source": "generator_model", "model": generator_model, "human_reviewed": False},
    }
    validate_row(record)
    return record


class JevLabeler(curator.LLM):
    def prompt(self, row):
        return row["request_json"]

    def parse(self, row, response):
        source, teacher = json.loads(row["source_json"]), json.loads(response)
        source["teacher"] = teacher
        # Persist successful API responses even when validation flags them.
        # The export validates again and fails explicitly; Curator must not drop
        # the only copy of a paid response in its parse callback.
        try:
            validate_teacher(source, teacher, json.loads(row["request_json"])["model"])
        except (ValueError, KeyError, TypeError) as error:
            source["teacher_validation_error"] = str(error)
        return {"id": source["id"], "record_json": canonical(source)}
