"""Create 150 complex, blind-reviewed decisions in five new coverage segments.

Uses the existing Curator draft schemas and cached asynchronous curation transport.
New examples remain unassigned to training/evaluation until a family-level split.
"""
import argparse
import asyncio
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from nimble.paths import PROJECT_ROOT
from nimble.datasets.create_diverse_dataset import configure
from nimble.datasets.dataset_io import canonical
from nimble.datasets.diversity_plan import normalize_state, state_tokens, validate_row
from nimble.training.schema_data import as_scoring, fingerprint

VERSION = 'complex-segments-v1'
MODEL = 'gpt-5.6-sol'
SEED = 20260918
USE_CASE_URL = 'https://docs.typesafe.ai/concepts/use-case-map'
TOPICS = {
    'gaming': [
        ('cooperative_tactics', 'Choose or verify a cooperative action using cooldowns, objective priority, ally intent, and turn timing.'),
        ('hidden_information', 'Distinguish observation from inference in fog-of-war, bluffing, and incomplete scouting; supplied rules define what is knowable.'),
        ('quest_dependencies', 'Resolve branching quests, faction commitments, consumable resources, and prerequisite exceptions in fictional games.'),
        ('chat_and_reports', 'Interpret quoted versus directed chat, consent, repeated conduct, and corroborating evidence under explicit community rules.'),
        ('player_retention', 'Interpret player engagement, frustration, matchmaking complaints and conditional departure; avoid equating complaints with churn.'),
        ('npc_negotiation', 'Resolve indirect player intent, dialogue history, conflicting requests, trust, and permitted NPC actions.'),
    ],
    'browser_website': [
        ('ambiguous_controls', 'Resolve repeated button labels using role, accessible name, parent panel, target item, enabled state, and user goal.'),
        ('forms_and_validation', 'Select the next form action with dependent fields, validation messages, unsaved edits and submission prerequisites.'),
        ('modals_and_overlays', 'Reason about focus, foreground dialogs, disabled background controls, and dismissal versus confirmation semantics.'),
        ('tabs_and_navigation', 'Disambiguate tabs, account/workspace scope, stale pages, navigation history, and actions that preserve the user goal.'),
        ('asynchronous_state', 'Use loading indicators, disabled controls, successful or failed prior actions, and current evidence to select click, wait, or clarify.'),
        ('filters_and_preferences', 'Interpret filter scope, cookie preferences, selection toggles, pagination and save/apply controls without unintended changes.'),
    ],
    'demand_forecasting': [
        ('purchase_commitment', 'Separate signed commitments, conditional purchase intent, budgets, generic interest, cancellations and revised quantities.'),
        ('stockout_censoring', 'Distinguish low sales caused by stockouts or fulfillment restrictions from falling demand, using customer messages and availability.'),
        ('seasonality_and_events', 'Identify time-local seasonal, event, promotional and enduring demand signals without extrapolating them beyond the horizon.'),
        ('substitution_and_competition', 'Separate substitution, competitor outages, cannibalization and net incremental demand using channel and product context.'),
        ('lead_times_and_backlog', 'Interpret requested delivery dates, backlog, duplicated orders, supply concerns and customers pulling orders forward.'),
        ('aggregation_and_revisions', 'Join customer identities, channel reports, revised opportunities and duplicate inquiries to classify evidence without double counting.'),
    ],
    'typesafe_other_use_cases': [
        ('retrieval_and_ranking', 'Compare passage relevance to a multi-constraint query; separate keyword overlap from evidence that answers the actual request.'),
        ('scientific_review', 'Screen fictional study descriptions against inclusion/exclusion rules or verify whether a cited passage supports a claim.'),
        ('model_and_tool_routing', 'Select a model or tool using the requested operation, supported capabilities, privacy constraints, and explicit escalation rules.'),
        ('guardrails_and_verification', 'Distinguish untrusted retrieved instructions from user intent and verify tool arguments, output claims, or sensitive-data exposure.'),
        ('semantic_code_linting', 'Judge a small real Python snippet against a supplied API contract: retries, error semantics, resource ownership or logging conventions.'),
        ('knowledge_graphs', 'Resolve entity identity, direction of relationships, temporal validity and conflicting records before adding or verifying a graph edge.'),
    ],
    'ancient_history': [
        ('mesopotamian_administration', 'Interpret fictional teaching dossiers about Mesopotamian administration: distinguish recorded allocation, delivery, debt and inference.'),
        ('egyptian_chronology', 'Use supplied chronological ranges, stratigraphy, reused objects and later copies to distinguish object age, deposition and event date.'),
        ('indus_evidence_limits', 'Evaluate archaeological provenance, seal impressions and exchange hypotheses while respecting the limits of an undeciphered script.'),
        ('shang_divination', 'Distinguish a divination question, prediction, and separately recorded outcome; account for damage, copying and provenance.'),
        ('greek_myth_and_civic_memory', 'Distinguish mythic precedent, later commemoration, contemporary evidence and political claims in Greek historical interpretation.'),
        ('roman_images_and_testimony', 'Compare official representation, later testimony and material context; distinguish an imperial image from evidence of an actual event.'),
    ],
}
HISTORY_SOURCES = {
    'mesopotamian_administration': {
        'url': 'https://www.metmuseum.org/de/essays/uruk-the-first-city',
        'background': 'Early Uruk pictographs recorded goods management and worker rations. Later traditions about a ruler are not equivalent to contemporary administrative records.'},
    'egyptian_chronology': {
        'url': 'https://www.ucl.ac.uk/museums-static/digitalegypt/chronology/index.html',
        'background': 'Egyptian chronology distinguishes successive historical periods. Supply all dates and chronological assumptions needed for the exercise; do not invent real reign dates.'},
    'indus_evidence_limits': {
        'url': 'https://www.metmuseum.org/art/collection/search/324062',
        'background': 'Harappan stamp seals often carry animal imagery and short inscriptions. Their script is not fully deciphered; a proposed reading must not be treated as an established translation.'},
    'shang_divination': {
        'url': 'https://collections.penn.museum/collections/object/76960',
        'background': 'Shang oracle bones were used for divination about political, social and personal matters. A question about a future event alone does not establish that the event occurred.'},
    'greek_myth_and_civic_memory': {
        'url': 'https://www.metmuseum.org/it/essays/theseus-hero-of-athens',
        'background': 'Greek myth could provide precedent for political programs. Later political use of a myth is distinct from direct evidence for the events the myth describes.'},
    'roman_images_and_testimony': {
        'url': 'https://www.metmuseum.org/ja/essays/roman-portrait-sculpture-the-stylistic-cycle',
        'background': 'Official Roman portrait types could idealize rulers and communicate dynastic values. An idealized image is not a literal record of every depicted attribute or event.'},
}
DOMAIN_RULES = {
    'gaming': 'Use fictional games. State the rules, current turn/time and relevant observations; do not require real-game knowledge or infer cheating from skill alone.',
    'browser_website': 'Use a synthetic text DOM/accessibility snapshot, never an image. Include the user goal and at least four uniquely identified controls with role, accessible name, container, visibility/enabled state, and relevant effects. Resolve the current page state and scope. Choice outputs select a listed control/action or explicit wait/clarify action. Boolean/score cases evaluate a named proposed action. No real account, website interaction or purchase is performed.',
    'demand_forecasting': 'Extract a semantic feature for a downstream forecast, not an invented numerical prediction. Include forecast cutoff, horizon, product/channel scope and evidence timestamps. Avoid future-outcome leakage; distinguish requested quantities from fulfilled sales. Supply any arithmetic already computed. Target correctness follows the explicit feature definition, not unknowable future demand.',
    'typesafe_other_use_cases': 'Use only synthetic task data. For semantic_code_linting include one syntactically valid Python fenced snippet and all nonstandard API semantics; code is read, never executed. For verification keep quoted source content separate from the governing instructions.',
    'ancient_history': 'Use the verified background only as orientation. Make a SYNTHETIC TEACHING DOSSIER, and explicitly label it that way in the state. Invented excavation logs, inscriptions and translated text are exercise material, never authentic quotations, real museum accessions or discoveries. Put all needed evidence inside the state; no external historical recall. Distinguish fact, source claim, inference and uncertainty. Do not claim an undeciphered text is translated. Do not invent real rulers, dates, events or scholarly consensus. A decision must require interpreting the dossier, not simply noticing the synthetic label.',
}
MECHANISMS = [
    'Resolve a later correction against earlier records and a still-valid exception.',
    'Join two records by identity or scope, then apply an exception with a plausible distractor.',
    'Distinguish a conditional or quoted statement from an observed event using attribution and timing.',
    'Compare conflicting evidence with an explicit priority rule and explain why the tempting alternative fails.',
    'Handle missing evidence or a boundary condition with an explicit allowed outcome; absence is not automatically false.',
]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def state_text(state):
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return '\n'.join(state_text(v) for v in state)
    return '\n'.join(state_text(v) for v in state.values())


def plans():
    jobs = []
    for domain, topics in TOPICS.items():
        for topic_index, (topic, description) in enumerate(topics):
            for i in range(5):
                n = topic_index * 5 + i
                kind = ('choice', 'noul', 'score')[n % 3]
                fmt = 'object' if domain == 'browser_website' else ('text', 'object', 'dialogue')[(n // 3) % 3]
                slot = {'type': kind, 'format': fmt, 'mechanism': MECHANISMS[i], 'difficulty': 'complex',
                        'target': bool((n // 3) % 2) if kind == 'noul' else (n // 3) % 4 if kind == 'score' else None,
                        'choice_options': 4 if kind == 'choice' else None,
                        'score_levels': 4 if kind == 'score' else None,
                        'include_no_match': kind == 'choice' and n // 3 in (4, 9)}
                jobs.append({'example_id': f'{VERSION}-{domain}-{n+1:03d}', 'domain': domain,
                             'group_id': f'{VERSION}-{domain}-{topic}', 'subtopic': topic,
                             'description': description, 'actors_json': '["case author", "decision maker", "record owner"]',
                             'split': 'unassigned', 'slot_json': canonical(slot), 'seed': SEED,
                             'choice_target_position': (n // 3) % 4, 'plan_version': VERSION})
    return jobs


class ComplexReview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    target_key: str
    unambiguous: bool
    rubric_complete: bool
    domain_relevant: bool
    requires_combining_evidence: bool
    no_external_knowledge_needed: bool
    no_answer_leakage: bool
    domain_constraints_met: bool
    evidence_quotes: list[str]
    reason: str


GATES = ('unambiguous', 'rubric_complete', 'domain_relevant', 'requires_combining_evidence',
         'no_external_knowledge_needed', 'no_answer_leakage', 'domain_constraints_met')


def draft_prompt(job):
    from nimble.datasets.diverse_stages import ScenarioGenerator
    prompt = object.__new__(ScenarioGenerator).prompt(job)
    prompt = prompt.replace('Aim for 45-130 words of state.', 'Aim for 200-320 words of state (hard bounds 150-430).')
    prompt += ('\nCOMPLEXITY: Include at least three relevant observations, a temporal/scope/exception constraint, '
               'and at least one plausible but nondecisive distractor. At least TWO distinct evidence items must '
               'be combined to decide; a single keyword, explicit verdict, or simple arithmetic must not solve it. '
               'Make one narrow decision with a uniquely supported target. Questions and criteria must not reveal '
               'the answer by repeating case identifiers or the decisive case fact. Missing information needs a '
               'defined outcome. Vary entities, settings, evidence structure and scenario trajectory. '
               'Reference reason should cite decisive evidence and reject the closest alternative.\n' + DOMAIN_RULES[job['domain']])
    if job['domain'] == 'ancient_history':
        prompt += '\nVerified orientation: ' + canonical(HISTORY_SOURCES[job['subtopic']])
    prompt += f"\nConstruction attempt: {job.get('attempt', 0)}. Write a fresh case for this slot."
    return prompt


def review_prompt(row):
    # Deliberate allowlist: never disclose target, generator rationale, slot, or desired label.
    return ('Blindly solve and audit this synthetic typed-decision example. Return target_key as an option '
            'key, lowercase true/false, or zero-based integer string for score. Independently derive the answer. '
            'Check every named boolean gate. Complexity means combining at least two distinct facts with a '
            'temporal, scope, attribution or exception constraint, not just long text or counting. '
            'Reject incomplete/overlapping rubrics, required unstated assumptions, factual contradictions, '
            'single-keyword shortcuts, and criteria that leak the case answer. Give 2-4 SHORT exact quotes '
            'from distinct state evidence items that jointly determine the answer, each at most 180 characters, '
            'plus a concise rationale. Quotes must be literal state substrings, not invented paraphrases. '
            'Evaluate these domain constraints too: ' + DOMAIN_RULES[row['domain']] + '\n' +
            canonical({'domain': row['domain'], 'subtopic': row['subtopic'], 'input': row['input']}))


def validate_content(row):
    import ast
    import re
    validate_row(row)
    text = state_text(row['input']['state'])
    if not 150 <= len(text.split()) <= 430:
        raise ValueError(f'State word count outside 150-430: {len(text.split())}')
    if row['subtopic'] == 'semantic_code_linting':
        snippets = re.findall(r'```python\s*\n(.*?)```', text, re.S)
        if len(snippets) != 1:
            raise ValueError('Expected one Python snippet')
        ast.parse(snippets[0].strip())
    if row['domain'] == 'ancient_history' and 'synthetic' not in text.lower():
        raise ValueError('History dossier must disclose synthetic status')


def validate_review(row, review):
    ComplexReview.model_validate(review)
    target = str(row['reference']['target']).lower() if isinstance(row['reference']['target'], bool) else str(row['reference']['target'])
    if review['target_key'] != target or not all(review[k] for k in GATES):
        raise ValueError('Blind review rejected: ' + canonical(review))
    quotes = review['evidence_quotes']
    if not 2 <= len(set(quotes)) == len(quotes) <= 4:
        raise ValueError('Review requires 2-4 distinct evidence quotes')
    text = state_text(row['input']['state'])
    if any(not q.strip() or len(q) > 180 or q not in text for q in quotes):
        raise ValueError('Review evidence must be short literal state excerpts')


def reorder(row, job):
    q = row['input']['questions']['decision']
    if q['type'] == 'choice':
        gold = row['reference']['target']
        others = sorted(k for k in q['criteria'] if k != gold)
        random.Random(f"{SEED}:{row['id']}").shuffle(others)
        others.insert(job['choice_target_position'], gold)
        q['criteria'] = {key: q['criteria'][key] for key in others}
    return row


def audit(rows, previous):
    if len(rows) != 150 or len({r['id'] for r in rows}) != 150:
        raise ValueError('Expected 150 unique examples')
    if Counter(r['domain'] for r in rows) != {d: 30 for d in TOPICS}:
        raise ValueError('Expected 30 per segment')
    for domain in TOPICS:
        subset = [r for r in rows if r['domain'] == domain]
        if Counter(r['input']['questions']['decision']['type'] for r in subset) != {k: 10 for k in ('choice','noul','score')}:
            raise ValueError('Primitive quota differs')
    if len({normalize_state(r['input']['state']) for r in rows}) != 150:
        raise ValueError('Duplicate normalized state')
    tokens = [state_tokens(r['input']['state']) for r in previous]
    maximum = 0.0
    for row in rows:
        validate_content(row); validate_review(row, row['review'])
        t = state_tokens(row['input']['state'])
        for other in tokens:
            score = len(t & other) / len(t | other)
            maximum = max(score, maximum)
            if score >= .8:
                raise ValueError('Near-duplicate: ' + row['id'])
        tokens.append(t)
    return {'examples': len(rows), 'domain_counts': dict(Counter(r['domain'] for r in rows)),
            'primitive_counts': dict(Counter(r['input']['questions']['decision']['type'] for r in rows)),
            'families': len({r['family'] for r in rows}), 'prior_examples_checked': len(previous),
            'maximum_token_jaccard': maximum, 'near_duplicate_threshold': .8,
            'word_count_range': [min(len(state_text(r['input']['state']).split()) for r in rows),
                                 max(len(state_text(r['input']['state']).split()) for r in rows)]}


def previous_rows():
    paths = ['data/typesafe_diverse_300_gpt56/all.jsonl', 'data/typesafe_eval_1000_gpt56/all.jsonl',
             'data/openjeff_mixed_3000_qwen9b_v1/full/train.jsonl',
             'data/openjeff_mixed_3000_qwen9b_v1/full/eval.jsonl', 'data/lint_gaming_eval_v1/eval.jsonl']
    return [r for name in paths if (PROJECT_ROOT / name).exists() for r in read_rows(PROJECT_ROOT / name)]


def markdown(rows):
    chunks = []
    for row in rows:
        q = row['input']['questions']['decision']
        chunks += [f"## {row['id']}\n", f"**{row['subtopic']} · {q['type']}**\n",
                   '```json\n' + json.dumps(row['input']['state'], ensure_ascii=False, indent=2) + '\n```\n',
                   q['instructions'] + '\n',
                   '```json\n' + json.dumps(q['criteria'], ensure_ascii=False, indent=2) + '\n```\n',
                   f"**Reference:** `{json.dumps(row['reference']['target'])}`\n",
                   row['reference']['reason'] + '\n', '**Blind reviewer:** ' + row['review']['reason'] + '\n']
    return '\n'.join(chunks)


async def curate(args):
    from nimble.datasets.fast_training_dataset import AsyncStages, DraftRejected
    from nimble.datasets.curation_providers import CurationClient
    from nimble.datasets.diverse_stages import ChoiceDraft, NoulDraft, ScoreDraft, unpack_example

    class Stages(AsyncStages):
        def request_spec(self, cls, schema, row):
            spec = {'model': MODEL, 'reasoning_effort': 'medium', 'max_completion_tokens': 6144,
                    'messages': [{'role':'user', 'content': draft_prompt(row) if cls == 'draft' else review_prompt(row)}],
                    'response_format': {'type':'json_schema', 'json_schema': {
                        'name': schema.__name__, 'strict': True, 'schema': schema.model_json_schema()}}}
            path = self.output / 'request_specs' / (fingerprint(spec) + '.json')
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                path.write_text(json.dumps(spec, indent=2) + '\n')
            return spec

    jobs = plans()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    prior = previous_rows()
    settings = {'version': VERSION, 'seed': SEED, 'model': MODEL, 'examples': 150,
                'plan_fingerprint': fingerprint(jobs), 'history_sources': HISTORY_SOURCES,
                'use_case_source': USE_CASE_URL, 'prior_input_fingerprint': fingerprint([r['input'] for r in prior]),
                'pipeline_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'split': 'unassigned', 'max_attempts_per_slot': 4,
                'endpoint': 'https://api.openai.com/v1',
                'label_method': 'Generator reference plus separate blind review; no Jev or evaluated-model outputs used'}
    path = output / 'settings.json'
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError('Run settings changed; use a new output directory')
    path.write_text(json.dumps(settings, indent=2) + '\n')
    write_rows(output / 'plan.jsonl', jobs)
    if args.plan_only:
        print(json.dumps(settings, indent=2)); return
    options = SimpleNamespace(concurrency=12, requests_per_minute=120, tokens_per_minute=600000, offline=args.offline)
    client = None if args.offline else CurationClient([MODEL])
    stage = Stages(output, options, client)
    accepted_path = output / 'accepted_progress.jsonl'
    accepted = {r['id']: r for r in read_rows(accepted_path)} if accepted_path.exists() else {}
    attempt_path = output / 'attempts.jsonl'
    attempts = read_rows(attempt_path) if attempt_path.exists() else []
    lock = asyncio.Lock()
    old_tokens = [state_tokens(r['input']['state']) for r in prior]

    async def one(job):
        if job['example_id'] in accepted:
            return
        kind = json.loads(job['slot_json'])['type']
        schema = {'choice': ChoiceDraft, 'noul': NoulDraft, 'score': ScoreDraft}[kind]
        for attempt in range(4):
            request = {**job, 'attempt': attempt}
            try:
                draft = await stage.one('draft', schema, request)
                row = reorder(unpack_example({**request, 'draft_json': canonical(draft)}, MODEL), job)
                validate_content(row)
                review = await stage.one('review', ComplexReview, row)
                validate_review(row, review)
                row.update(review=review, source_family=row['family'], quality_status='model_checked')
                row['provenance'] = {'synthetic': True, 'attempt': attempt, 'plan_fingerprint': fingerprint(job),
                                     'generation_request': fingerprint(stage.request_spec('draft', schema, request)),
                                     'review_request': fingerprint(stage.request_spec('review', ComplexReview, row)),
                                     'review_reference_hidden': True}
                if row['domain'] == 'ancient_history':
                    row['provenance']['historical_background'] = HISTORY_SOURCES[row['subtopic']]
                async with lock:
                    tokens = state_tokens(row['input']['state'])
                    compare = old_tokens + [state_tokens(r['input']['state']) for r in accepted.values()]
                    if any(len(tokens & t) / len(tokens | t) >= .8 for t in compare):
                        raise ValueError('Near-duplicate of prior/accepted state')
                    accepted[row['id']] = row
                    attempts.append({'id': row['id'], 'attempt': attempt, 'accepted': True})
                    write_rows(accepted_path, sorted(accepted.values(), key=lambda r:r['id']))
                    write_rows(attempt_path, attempts)
                    print(f"Accepted {len(accepted)}/150: {row['id']}", flush=True)
                return
            except (ValueError, SyntaxError, DraftRejected) as error:
                async with lock:
                    attempts.append({'id': job['example_id'], 'attempt': attempt, 'accepted': False, 'reason': str(error)})
                    write_rows(attempt_path, attempts)
        raise RuntimeError('Unfilled slot after four checked drafts: ' + job['example_id'])

    try:
        outcomes = await asyncio.gather(*(one(job) for job in jobs), return_exceptions=True)
        errors = [str(value) for value in outcomes if isinstance(value, BaseException)]
        if errors:
            raise RuntimeError(canonical(errors))
        rows = sorted(accepted.values(), key=lambda r:r['id'])
        by_id = {j['example_id']:j for j in jobs}
        for row in rows:
            job = by_id[row['id']]
            evidence = {}
            for name in ('generation', 'review'):
                key = row['provenance'][name + '_request']
                saved = json.loads((output / 'requests' / (key + '.json')).read_text())
                spec = json.loads((output / 'request_specs' / (key + '.json')).read_text())
                if fingerprint(spec) != key or fingerprint(saved['response']) != saved['response_sha256']:
                    raise ValueError('Cached provenance mismatch')
                if saved['model_returned'] != MODEL:
                    raise ValueError('Unexpected returned model')
                evidence[name] = saved['response']
            reconstructed = reorder(unpack_example({**job, 'draft_json':canonical(evidence['generation'])}, MODEL), job)
            if reconstructed['input'] != row['input'] or reconstructed['reference'] != row['reference'] or evidence['review'] != row['review']:
                raise ValueError('Accepted row differs from saved evidence')
        report = audit(rows, prior)
        usage = Counter()
        for folder in ('requests', 'rejected_requests'):
            for p in (output / folder).glob('*.json'):
                saved = json.loads(p.read_text())
                usage.update({k:saved.get('usage',{}).get(k,0) for k in ('prompt_tokens','completion_tokens')})
        manifest = {**settings, **report, 'usage':dict(usage), 'record_fingerprint':fingerprint(rows),
                    'rejected_attempts':sum(not r['accepted'] for r in attempts),
                    'limitations':['Synthetic and model-checked, not human-reviewed',
                                   'Generator and blind reviewer are separate calls to the same model',
                                   'Agreement filtering may favor cases this model finds easier',
                                   'No contrastive evidence/deletion certificate; not directly accepted by the certified trainer',
                                   'Do not use the same source family for both training and evaluation']}
        write_rows(output / 'all.jsonl', rows)
        write_rows(output / 'scoring.jsonl', [as_scoring(r, False) for r in rows])
        for domain in TOPICS:
            subset = [r for r in rows if r['domain'] == domain]
            write_rows(output / (domain + '.jsonl'), subset)
            (output / (domain + '.md')).write_text('# ' + domain.replace('_',' ').title() + '\n\n' + markdown(subset))
        manifest['dataset_sha256'] = hashlib.sha256((output / 'all.jsonl').read_bytes()).hexdigest()
        if (output / 'manifest.json').exists():
            if json.loads((output / 'manifest.json').read_text()) != manifest:
                raise ValueError('Frozen release differs')
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (output / 'README.md').write_text(
            '# Complex segment expansion: 150 examples\n\n' +
            '30 new cases per segment; 10 Choice, 10 Noul and 10 Score per segment. '
            'Six topic families per segment, five cases per family. Every case passes a blind answer/complexity review.\n\n' +
            '\n'.join(f'- [{d.replace("_"," ").title()}]({d}.md): [JSONL]({d}.jsonl)' for d in TOPICS) +
            '\n\n`all.jsonl` includes inputs, references, evidence quotes and provenance. `scoring.jsonl` is the candidate-scoring export. '
            '`manifest.json` records coverage, hashes and API token usage. Raw request specifications and responses permit offline replay.\n\n'
            'All splits are **unassigned**. Freeze a family-level train/validation/evaluation split before using the pool; '
            'do not train on a case and later report it as held out. These are distinct scenarios, not contrast pairs. '
            'They lack the contrastive certificates required by the current certified training loader.\n\n'
            'Browser examples use synthetic textual accessibility/page states; they do not test vision or interact with real sites. '
            'Demand cases extract forecasting features; they do not claim to predict future sales. Ancient-history cases '
            'are explicitly synthetic source-analysis exercises with linked background references, not authentic inscriptions '
            'or a historical fact benchmark.\n\n'
            f'Use-case inspiration: [{USE_CASE_URL}]({USE_CASE_URL}). Historical background links are in settings and per-row provenance.\n\n'
            'Labels are generated by GPT-5.6 Sol and checked in separate blind Sol calls; the reviewer never sees the reference. '
            'Same-model errors can remain correlated. No human review, Jev labeling or model-quality evaluation has been performed.\n\n'
            'Reproduce/resume: `.venv-curator/bin/python -m nimble.datasets.create_complex_segments`\n\n'
            'Verify saved data without API calls: add `--offline`.\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if client:
            await client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/complex_segments_150_v1')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    configure(); load_dotenv(PROJECT_ROOT / '.env')
    asyncio.run(curate(args))


if __name__ == '__main__':
    main()
