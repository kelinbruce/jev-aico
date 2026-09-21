"""Staged follow-up: verify premise pairs before writing their surrounding contexts."""
import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path

from bespokelabs import curator
from pydantic import BaseModel, ConfigDict

from nimble.paths import PROJECT_ROOT
from nimble.datasets.create_diverse_dataset import configure, read_jsonl
from nimble.datasets.dataset_io import canonical, write_jsonl
from nimble.datasets.curate_evidence_dataset import run_stage
from nimble.datasets.curate_evidence_counterfactuals import (
    ReviewRevision, RevisionAudit, make_row,
)
from nimble.datasets.evidence_curation import (
    assess_facts, build_group, decide, fact_jobs, rule_job,
)
from nimble.datasets.evidence_stages import FactAudit, FactReviewer, RuleAudit, RuleReviewer, Span
from nimble.datasets.contrastive_data import apply_edit, fingerprint, scoring_record


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Pair(Strict):
    left: str
    right: str
    negative_left: str
    negative_right: str


class Document(Strict):
    base_state_json: str
    focus_evidence: list[Span]


class PairGenerator(curator.LLM):
    def prompt(self, row):
        return """Generate a rigorously controlled factual evidence pair for the given
standalone proposition. left+right must entail the proposition, but left alone and
right alone must each leave it UNKNOWN (neither supported nor refuted). Then change
exactly ONE of those sentences to form negative_left+negative_right which entails
its NEGATION. The changed negative sentence must also leave the proposition UNKNOWN
alone. The unchanged sentence must be identical, including punctuation. Each string
is one complete factual sentence, not a policy definition, question, or instruction.
Use self-contained evidence with unambiguous entities, timestamps, scopes, and no
external information. No sentence may directly assert the focus proposition or its
negation. The pair must supply every fact needed; no unspecified background is allowed.

Examples of the structure, not content to copy:
Proposition: 'Parcel K exceeds its route weight limit.'
left: 'Parcel K is assigned to route R, whose parcel weight limit is 12 kilograms.'
right: 'The calibrated scale records parcel K at 15 kilograms.'
negative_right: 'The calibrated scale records parcel K at 9 kilograms.'
negative_left: identical to left.
Proposition: 'Aster's submitted width conflicts with the signed specification.'
left: 'Aster's submitted width is 18 centimeters.'
right: 'The signed specification lists Aster's width as 21 centimeters.'
negative_right: 'The signed specification lists Aster's width as 18 centimeters.'
negative_left: identical to left.
Use the exact entities/quantities already named in the supplied proposition where
needed. Avoid encoding the answer in labels or names. Supply only the four sentences.
""" + row['payload_json']

    def parse(self, row, response):
        return {'job_id': row['job_id'], 'result_json': response.model_dump_json()}


class DocumentGenerator(curator.LLM):
    def prompt(self, row):
        return """Write a new concise natural context for the supplied unchanged question
and rubric. Two evidence sentences have ALREADY been independently verified. Include
both positive sentences VERBATIM and return their exact string-leaf paths and text
in focus_evidence, in left/right order. The base_state_json is a JSON serialization
of a string, object, or dialogue matching the original input's kind. Preserve entities
referenced by the question and propositions. Copy the listed original policy spans
VERBATIM. Do not invent policy, default assumptions, priorities, or answer choices.

Provide enough additional observed facts to determine the other listed propositions
and permit at least one of the supplied audited sufficient rules to apply. Do not
add any direct statement, paraphrase, or alternate proof of the focus proposition.
If either pair sentence is deleted, focus support must disappear. Do not repeat its
underlying values in summaries, titles, role labels, or other context fields. Retain
the pair's logical coherence with every added fact. Its single-sentence counterfactual
must also remain coherent with those facts. Do not include the proposition IDs, rule
table, derived answers, or rationale explaining the classification in the context.
A normal narrative, case note, structured evidence list or dialogue is appropriate.
Aim for 100-180 words plus any verbatim policy. Return only the serialized context
and its two exact evidence spans. The original input is background inspiration,
not a source to copy wholesale; copying it may reintroduce redundant proofs.
""" + row['payload_json']

    def parse(self, row, response):
        return {'job_id': row['job_id'], 'result_json': response.model_dump_json()}


def pair_requests(gid, atom, pair):
    differences = [i for i, name in enumerate(('left', 'right')) if pair[name] != pair['negative_' + name]]
    if len(differences) != 1 or any(len(v.split()) < 4 for v in pair.values()):
        raise ValueError('Need two substantive pairs differing by exactly one sentence')
    changed = ('left', 'right')[differences[0]]
    return [{'job_id': gid + ':' + name, 'payload_json': canonical({'context': text, 'propositions': [atom]})}
            for name, text in [('left', pair['left']), ('right', pair['right']),
                               ('positive_pair', pair['left'] + '\n' + pair['right']),
                               ('negative_sentence', pair['negative_' + changed]),
                               ('negative_pair', pair['negative_left'] + '\n' + pair['negative_right'])]]


def pair_passes(gid, atom, pair, results):
    expectations = {'left': 'unknown', 'right': 'unknown', 'positive_pair': 'supported',
                    'negative_sentence': 'unknown', 'negative_pair': 'refuted'}
    for req in pair_requests(gid, atom, pair):
        statuses = assess_facts(req, results[req['job_id']])
        if statuses[atom['id']] != expectations[req['job_id'].split(':')[-1]]:
            return False
    return True


def audit_passes(audit, count):
    checks = audit['rule_checks']
    return (len(checks) == count and {r['rule_index'] for r in checks} == set(range(count))
            and all(audit[k] for k in ('atoms_are_atomic', 'policy_preserved', 'focus_is_relevant', 'evidence_is_two_sentences'))
            and all(r['sound'] for r in checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PROJECT_ROOT / 'data/evidence_curated_gpt56_v2')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/evidence_curated_gpt56_v2_staged')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    parent_config = json.loads((args.parent / 'run_config.json').read_text())
    sources = read_jsonl(Path(parent_config['source']))
    source_map = {r['id']: r for r in sources}
    if fingerprint(sources) != parent_config['source_sha256']:
        raise ValueError('Source changed')
    old_groups = {r['id']: r for r in read_jsonl(args.parent / 'constructed_groups.jsonl')}
    old_audits = {r['job_id']: r['response'] for r in read_jsonl(args.parent / 'rule_audits.jsonl')}
    candidates = {'staged-' + gid: g for gid, g in old_groups.items()
                  if audit_passes(old_audits[gid], len(g['spec']['rules']))}
    config = {'pipeline_version': 'evidence-curation-v2-staged', 'model': parent_config['model'],
              'parent_sha256': fingerprint([parent_config, old_groups, old_audits]),
              'implementation_sha256': fingerprint(Path(__file__).read_text()),
              'source_sha256': fingerprint(sources), 'selected_groups': sorted(candidates)}
    if args.offline:
        if json.loads((args.output / 'run_config.json').read_text()) != config:
            raise ValueError('Source/configuration changed')
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / 'run_config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Output belongs to another run')
        path.write_text(json.dumps(config, indent=2) + '\n')
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / '.env', override=False)
        if not os.environ.get('OPENAI_API_KEY'):
            parser.error('OPENAI_API_KEY is required')
        configure()

    def stage(cls, schema, reqs, name):
        if args.offline:
            saved = {r['job_id']: r for r in read_jsonl(args.output / (name + '.jsonl'))}
            for req in reqs:
                if saved[req['job_id']]['request_sha256'] != fingerprint(req):
                    raise ValueError('Saved stage request changed')
            return {r['job_id']: saved[r['job_id']]['response'] for r in reqs}
        return run_stage(cls, schema, reqs, args.output, config['model'], name)

    atoms = {gid: next(a for a in g['spec']['atoms'] if a['id'] == g['spec']['focus_atom'])
             for gid, g in candidates.items()}
    pairs = stage(PairGenerator, Pair, [{'job_id': gid, 'payload_json': canonical({'proposition': a['statement']})}
                                      for gid, a in atoms.items()], 'premise_pairs')
    requests, review = [], []
    for gid, pair in pairs.items():
        try:
            requests.extend(pair_requests(gid, atoms[gid], pair))
        except ValueError as error:
            review.append({'group_id': gid, 'reasons': [str(error)]})
    checks = stage(FactReviewer, FactAudit, requests, 'pair_checks')
    passed = {}
    for gid, pair in pairs.items():
        try:
            if pair_passes(gid, atoms[gid], pair, checks):
                passed[gid] = pair
            else:
                review.append({'group_id': gid, 'reasons': ['isolated_pair_entailment_failure']})
        except (ValueError, KeyError, TypeError) as error:
            review.append({'group_id': gid, 'reasons': ['invalid_pair_check'], 'detail': str(error)})
    print(f'Verified premise pairs: {len(passed)}/{len(candidates)}.', flush=True)
    doc_requests = [{'job_id': gid, 'payload_json': canonical({
        'original_input': source_map[candidates[gid]['source_id']]['input'],
        'atoms': candidates[gid]['spec']['atoms'], 'rules': candidates[gid]['spec']['rules'],
        'focus_atom': candidates[gid]['spec']['focus_atom'], 'verified_pair': pair,
        'policy_evidence': candidates[gid]['spec']['policy_evidence']})} for gid, pair in passed.items()]
    documents = stage(DocumentGenerator, Document, doc_requests, 'paired_documents')
    built, negatives, edit_requests = {}, {}, []
    for gid, doc in documents.items():
        try:
            parent = candidates[gid]
            spec = {**copy.deepcopy(parent['spec']), **doc}
            if [s['text'] for s in spec['focus_evidence']] != [passed[gid]['left'], passed[gid]['right']]:
                raise ValueError('Verified sentences changed in document')
            group = build_group({'id': gid, 'source': source_map[parent['source_id']], 'method': 'c2d'}, spec)
            pair = passed[gid]
            index = 0 if pair['left'] != pair['negative_left'] else 1
            replacement = pair['negative_left'] if index == 0 else pair['negative_right']
            span = spec['focus_evidence'][index]
            negative = apply_edit(group['states']['base'], {'path': span['path'], 'old': span['text'], 'new': replacement})
            built[gid], negatives[gid] = group, negative
            edit_requests.append({'job_id': gid, 'payload_json': canonical({
                'question': group['question'], 'original_context': group['states']['base'],
                'revised_context': negative, 'edit': {'sentence_index': index, 'replacement': replacement}})})
        except (ValueError, KeyError, TypeError, IndexError) as error:
            review.append({'group_id': gid, 'reasons': ['invalid_document'], 'detail': str(error)})
    audits = stage(RuleReviewer, RuleAudit, [rule_job(g, source_map[g['source_id']]) for g in built.values()], 'staged_rule_audits')
    edit_audits = stage(ReviewRevision, RevisionAudit, edit_requests, 'staged_edit_audits')
    full_requests = [req for g in built.values() for req in fact_jobs(g)[:3]]
    full_requests += [{'job_id': gid + ':counterfactual', 'payload_json': canonical({
        'context': negatives[gid], 'propositions': g['spec']['atoms']})} for gid, g in built.items()]
    full_checks = stage(FactReviewer, FactAudit, full_requests, 'staged_full_checks')
    rows = []
    for gid, group in built.items():
        try:
            if not audit_passes(audits[gid], len(group['spec']['rules'])):
                raise ValueError('Context/rule audit failed')
            if not all(edit_audits[gid][k] for k in ('policy_preserved', 'single_coherent_factual_change', 'no_answer_instruction')):
                raise ValueError('Counterfactual coherence/policy audit failed')
            matrices = {req['job_id'].split(':')[-1]: assess_facts(req, full_checks[req['job_id']])
                        for req in full_requests if req['job_id'].startswith(gid + ':')}
            focus = group['spec']['focus_atom']
            for case, expected in [('base', 'supported'), ('remove_left', 'unknown'), ('remove_right', 'unknown'), ('counterfactual', 'refuted')]:
                if matrices[case][focus] != expected or 'inconsistent' in matrices[case].values():
                    raise ValueError('Full-context evidence gate failed: ' + case)
            group_rows = []
            for variant, state in {**group['states'], 'counterfactual': negatives[gid]}.items():
                if decide(group['spec'], matrices[variant], group['question']) is not None:
                    group_rows.append(make_row(group, state, variant, matrices[variant], audits[gid],
                        {'verified_pair': passed[gid], 'edit_audit': edit_audits[gid],
                         'full_context_fact_states': matrices, 'construction': 'pair_checked_before_document'}))
            if not any(r['variant'] == 'base' for r in group_rows) or len({canonical(r['reference']['target']) for r in group_rows}) < 2:
                raise ValueError('No determinate schema contrast')
            rows.extend(group_rows)
        except (ValueError, KeyError, TypeError) as error:
            review.append({'group_id': gid, 'reasons': [str(error)]})
    rows.sort(key=lambda r: r['id'])
    heldout = [r for r in sources if r['split'] != 'train']
    for row in rows:
        source = source_map[row['provenance']['source_id']]
        if source['split'] != 'train' or row['input']['questions'] != source['input']['questions']:
            raise ValueError('Invalid source/schema')
        if row['source_family'] in {r['family'] for r in heldout} or fingerprint(row['input']['state']) in {fingerprint(r['input']['state']) for r in heldout}:
            raise ValueError('Held-out overlap')
    if len({fingerprint(r['input']) for r in rows}) != len(rows):
        raise ValueError('Duplicate training inputs')
    manifest = {**config, 'training': {'examples': len(rows), 'groups': len({r['family'] for r in rows}),
                 'domains': dict(Counter(r['domain'] for r in rows)),
                 'primitives': dict(Counter(r['input']['questions']['decision']['type'] for r in rows)),
                 'variants': dict(Counter(r['variant'] for r in rows))},
                 'verified_pairs': len(passed), 'constructed_groups': len(built),
                 'train_sha256': fingerprint(rows), 'review_entries': len(review),
                 'reference_human_reviewed': False, 'held_out_overlap': False,
                 'notes': ['One new construction attempt per previously audited rule set; original failed proposals retained.',
                           'Pair support/refutation and individual-sentence insufficiency checked before document generation.',
                           'All sentence necessities rechecked in full contexts; unchanged source schema and rules.',
                           'Labels are derived from rules, not generated final-label guesses.',
                           'All sources synthetic; same-model verification; no measured training improvement.']}
    if args.offline:
        if rows != read_jsonl(args.output / 'train.jsonl') or manifest != json.loads((args.output / 'manifest.json').read_text()):
            raise ValueError('Evidence reconstruction mismatch')
    else:
        write_jsonl(args.output / 'train.jsonl', rows)
        write_jsonl(args.output / 'train_scoring.jsonl', [scoring_record(r) for r in rows])
        write_jsonl(args.output / 'review_queue.jsonl', review)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
