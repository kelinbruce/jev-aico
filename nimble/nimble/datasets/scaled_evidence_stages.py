"""Versioned prompts for training-only, rule-first evidence curation."""
from bespokelabs import curator
from nimble.datasets.evidence_stages import Strict, Atom, Rule, Span, RuleCheck, Condition


class RulePlan(Strict):
    atoms: list[Atom]
    focus_atom: str
    policy_evidence: list[Span]
    rules: list[Rule]
    base_states: list[Condition]
    counter_states: list[Condition]


class PlanAudit(Strict):
    atoms_are_atomic: bool
    policy_complete: bool
    focus_is_factual: bool
    assignments_are_realizable: bool
    rule_checks: list[RuleCheck]
    explanation: str


class ContextAudit(Strict):
    policy_preserved: bool
    question_bindings_preserved: bool
    evidence_is_two_factual_sentences: bool
    counterfactual_is_coherent: bool
    no_answer_leakage: bool
    explanation: str


class PlanGenerator(curator.LLM):
    def prompt(self, row):
        return """Design a SMALL auditable classification contrast for the supplied training
input. Preserve the question, options, criteria, and any governing policy EXACTLY.
You are designing rules, not writing the training context yet. Source answers are
not provided. Use 2-20 individually atomic factual propositions, each ONE relation
or measurement, not a bundle of conditions or the final classification. Separate
every independently variable requirement. Universal quantification of ONE relation
is allowed ('Every submitted photo has an attached permission'); joining unrelated
requirements in one atom is not. Encode all entities, times and scopes explicitly.

Choose one factual focus atom whose supported vs explicitly refuted state can change
the schema answer, with all other atom states unchanged. It must be possible to prove
it by joining two concrete factual sentences, neither sufficient alone. Prefer record
joins, identity/coreference, comparison of two observed measurements, dates, counts,
or comparisons between submitted and signed specifications. The two sentences must
be observations, not rules or policies. Do not choose a focus that is itself a policy,
subjective final judgment, conjunction, or simply a fact directly quoted in one sentence.

Return 2-8 UNORDERED sufficient-condition rules, with targets as strings (exact choice
key, true/false, or zero-based score index). Every rule must be valid for ALL scenarios
satisfying its ANDed conditions under the original rubric, not only the source case.
Include every required qualification, exception, scope and competing-outcome exclusion.
Supported means entailed; refuted means negation entailed; unknown is neither and MUST
NOT be treated as false. Never invent missing-information defaults or priorities.
Partial tables are encouraged: only cover a useful two-answer contrast. Do not try to
cover the entire schema. Different-target rules must not overlap. Two full conjunctions
of known factual states are often easiest. You may add a deletion/unknown rule ONLY
if explicitly justified by the original rubric; otherwise leave it uncovered.

base_states and counter_states are arrays of objects with exactly atom_id and state
keys, one per atom. States are supported/refuted/unknown. Base focus is supported,
counter focus is refuted, and ALL other assignments are identical. Each assignment
must match a rule, with DIFFERENT answers. They must be logically realizable together
with the policy. Don't use mutually dependent atoms that necessarily change together.

policy_evidence must quote EVERY governing policy/requirement from the ORIGINAL state
needed to interpret the unchanged question. Use exact spans and paths to string leaves
(string state path []; array indices as strings). Include original requested thresholds,
scope and conditions if criteria refer to them. Do not mistake original observed facts
for policy. An empty list is allowed only if the complete policy is in the question.
These quotes will be copied unchanged into every constructed context.

If previous_proposal and reviewer_feedback are supplied, make a substantive revision
addressing the failure, possibly selecting a simpler contrast. Do not merely restate
the rejected rules. Do not alter the question or request a more lenient reviewer.
""" + row['payload_json']

    def parse(self, row, response):
        return {'job_id': row['job_id'], 'result_json': response.model_dump_json()}


class PlanReviewer(curator.LLM):
    def prompt(self, row):
        return """Independently audit the proposed rule plan against the ORIGINAL input's
question, criteria and explicit policy. Do not trust the proposal's justifications.
Each rule is an unordered conjunction and must be a sufficient condition for its
target for ALL scenarios satisfying it. Partial rule tables are valid; uncovered
assignments abstain. A supported atom is entailed, refuted means its negation is
entailed, and unknown is neither. Unknown is never implicitly false. Reject unstated
assumptions, missing competing-outcome exclusions, invented defaults or priorities.

Atomic propositions each express ONE factual relationship, even if universally
quantified over an explicit set. Reject bundles of unrelated requirements and final
schema classifications. Check the focus is a factual relation rather than policy.
Check base/counter assignments are realizable under the original policy with only
the focus changing.

CRITICAL POLICY-COMPLETENESS CONTRACT: original_input.questions is ALWAYS retained
verbatim and supplied to the trained model alongside every generated context. Thus
ALL instructions, criteria, priorities, thresholds, scope and exceptions written in
that questions object are ALREADY PRESERVED AUTOMATICALLY. They MUST NOT appear in
policy_evidence, which contains citations from original_input.state ONLY. Do NOT
mark policy_complete false for missing quotations of material in questions.
An EMPTY policy_evidence is CORRECT when all governing rules are in questions.

For policy_complete, check ONLY whether policy_evidence preserves substantive rules,
requested requirements, scope or bindings that originate in the ORIGINAL STATE and
are needed to interpret the unchanged question. A generic state request such as
'classify this case' need not be quoted when the question already specifies the task.
Do not demand verbatim preservation of original case observations, incidental
identifiers or workflow narrative: NEW synthetic observations will be written.
Return a check for EVERY rule index, including unsound rules; do not repair anything.
""" + row['payload_json']

    def parse(self, row, response):
        return {'job_id': row['job_id'], 'result_json': response.model_dump_json()}


class ContextReviewer(curator.LLM):
    def prompt(self, row):
        return """Audit a newly constructed context and its single-sentence counterfactual
against the original question and policy. Original case observations may change,
but governing policy, request scope and question entity/path/time bindings may not.
Both contexts must preserve every governing policy, without inventing exceptions,
priorities or missing-evidence defaults. Evidence spans must be complete factual
sentences, not policy definitions or instructions. The counterfactual must remain
coherent with all unchanged context; reject contradictory duplicate measurements,
counts or assertions. Check neither context embeds a gold answer, answer code, rule
table, proposition IDs, label rationale or instruction telling the classifier what
to output. Natural policy terminology is allowed. Audit both full contexts. Do not
infer correctness from the generator's intended label (none is supplied).
""" + row['payload_json']

    def parse(self, row, response):
        return {'job_id': row['job_id'], 'result_json': response.model_dump_json()}


class FullInputContextReviewer(ContextReviewer):
    def prompt(self, row):
        original = super().prompt(row)
        return original[:-len(row['payload_json'])] + """
POLICY PRESERVATION CONTRACT: Every generated model input includes the ORIGINAL
questions object VERBATIM alongside the generated state. The question instructions
and every choice/score criterion are therefore already preserved automatically.
For policy_preserved, assess the WHOLE input (unchanged questions + each context).
Do not demand that question-only rules, score descriptions or thresholds be copied
into the context again. Reject lost or changed governing policy originating in the
ORIGINAL STATE when it is not already fully preserved in the unchanged question.
All other requirements, including factual coherence and question bindings, apply.
""" + row['payload_json']
