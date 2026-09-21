"""Add verified contrary evidence to v2 groups with underspecified deletion labels."""
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
from nimble.datasets.curation_options import llm_options
from nimble.datasets.curate_evidence_dataset import run_stage
from nimble.datasets.evidence_curation import (
    assess_facts, build_group, decide, evaluate_group, fact_jobs, select_sources,
)
from nimble.datasets.evidence_stages import FactAudit, FactReviewer
from nimble.datasets.contrastive_data import apply_edit, fingerprint, scoring_record


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Revision(Strict):
    sentence_index: int
    replacement: str
    explanation: str


class RevisionAudit(Strict):
    policy_preserved: bool
    single_coherent_factual_change: bool
    no_answer_instruction: bool
    explanation: str


class Revise(curator.LLM):
    def prompt(self, row):
        return """Change exactly ONE of the two supplied evidence sentences so their
combined evidence REFUTES the focus proposition (its negation is established).
Neither sentence ALONE may support or refute that proposition. Preserve every other
sentence, identity, governing policy, and schema. Choose sentence_index 0 or 1 and
return the complete replacement sentence. Change an observed fact or relation, not
a policy, missing-information rule, or output instruction. Do not simply insert the
focus conclusion or 'not' before the conclusion; keep a natural two-sentence inference.
The whole revised context must remain logically coherent. Do not assert a final
schema label. If an exclusive mapping is needed to establish the negative, ensure
it is actually present in the context; absence is never refutation. You receive no
expected schema labels. Later independent checks will reject invalid proposals.
""" + row['payload_json']

    def parse(self, row, response):
        return {'job_id': row['job_id'], 'result_json': response.model_dump_json()}


class ReviewRevision(curator.LLM):
    def prompt(self, row):
        return """Audit this single-sentence edit. Does it preserve all governing policy
and question meaning, make one coherent factual change without contradicting other
facts, and avoid instructions or explicit answer labels? Do not repair the edit.
Flag invented policy, changed entity scope, incompatible facts, and classifier-facing
answer hints. A changed observed value or relation is allowed. Return all three
checks, marking false when uncertain. You are not given an expected schema label.
""" + row['payload_json']

    def parse(self, row, response):
        return {'job_id': row['job_id'], 'result_json': response.model_dump_json()}


def load_stage(path):
    return {r['job_id']: r['response'] for r in read_jsonl(path)} if path.exists() else {}


def eligible_groups(groups, rules, facts):
    eligible, excluded = {}, []
    for gid, group in groups.items():
        try:
            _, issues = evaluate_group(group, rules[gid], facts)
            reasons = {r for issue in issues for r in issue['reasons']}
            if reasons - {'no_audited_rule_covers_fact_states', 'no_determinate_label_contrast'}:
                excluded.append({'group_id': gid, 'reasons': sorted(reasons)})
                continue
            job = next(r for r in fact_jobs(group) if r['job_id'].endswith(':base'))
            base_states = assess_facts(job, facts[job['job_id']])
            target = decide(group['spec'], base_states, group['question'])
            if target is None:
                excluded.append({'group_id': gid, 'reasons': ['base_has_no_justified_label']})
                continue
            eligible[gid] = group
        except (ValueError, KeyError, TypeError) as error:
            excluded.append({'group_id': gid, 'reasons': ['invalid_parent_audit'], 'detail': str(error)})
    return eligible, excluded


def revised_context(group, revision):
    index = revision['sentence_index']
    if type(index) is not int or index not in (0, 1):
        raise ValueError('Replacement index must be 0 or 1')
    text = revision['replacement']
    if len(text.split()) < 4:
        raise ValueError('Replacement must be a substantive sentence')
    pair = [s['text'] for s in group['spec']['focus_evidence']]
    span = group['spec']['focus_evidence'][index]
    state = apply_edit(group['states']['base'], {'path': span['path'], 'old': span['text'], 'new': text})
    pair[index] = text
    return state, pair


def revision_requests(group, revision):
    state, pair = revised_context(group, revision)
    focus = [a for a in group['spec']['atoms'] if a['id'] == group['spec']['focus_atom']]
    contexts = [('counterfactual', state, group['spec']['atoms']),
                ('changed_sentence', pair[revision['sentence_index']], focus),
                ('counter_pair', '\n'.join(pair), focus)]
    return [{'job_id': group['id'] + ':' + name,
             'payload_json': canonical({'context': context, 'propositions': atoms})}
            for name, context, atoms in contexts]


def make_row(group, state, variant, statuses, rule_audit, additional=None):
    target = decide(group['spec'], statuses, group['question'])
    if target is None:
        raise ValueError('No justified label')
    certificate = {'rule_audit': rule_audit, 'fact_states': statuses, 'spec': group['spec'],
                   'necessity_checks_passed': True, 'verifier_independent_model': False}
    if additional:
        certificate.update(additional)
    return {'id': group['id'] + '-' + variant, 'family': group['id'], 'source_family': group['source_family'],
            'split': 'train', 'domain': group['domain'], 'method': group['method'], 'variant': variant,
            'input': {'state': state, 'questions': {'decision': group['question']}},
            'reference': {'target': target, 'source': 'audited_rule_over_verified_facts', 'human_reviewed': False},
            'provenance': {'source_id': group['source_id'], 'source_split': 'train',
                           'source_sha256': group['source_sha256'], 'source_is_synthetic': True},
            'quality_status': 'evidence_model_checked', 'evidence_certificate': certificate}


def assess_revision(group, revision, audit, facts, parent_rules, parent_facts):
    if not all(audit[k] for k in ('policy_preserved', 'single_coherent_factual_change', 'no_answer_instruction')):
        raise ValueError('Counterfactual policy/coherence audit failed')
    matrices = {req['job_id'].split(':')[-1]: assess_facts(req, facts[req['job_id']])
                for req in revision_requests(group, revision)}
    focus = group['spec']['focus_atom']
    if matrices['changed_sentence'][focus] != 'unknown':
        raise ValueError('Changed sentence alone establishes the focus')
    if any(matrices[name][focus] != 'refuted' for name in ('counter_pair', 'counterfactual')):
        raise ValueError('Counterfactual does not establish explicit refutation')
    if any('inconsistent' in m.values() for m in matrices.values()):
        raise ValueError('Counterfactual contains inconsistent evidence')
    base_req = next(r for r in fact_jobs(group) if r['job_id'].endswith(':base'))
    base_states = assess_facts(base_req, parent_facts[base_req['job_id']])
    base = make_row(group, group['states']['base'], 'base', base_states, parent_rules[group['id']])
    state, _ = revised_context(group, revision)
    counter = make_row(group, state, 'counterfactual', matrices['counterfactual'], parent_rules[group['id']],
                       {'revision': revision, 'revision_audit': audit, 'counterfactual_fact_states': matrices})
    if counter['reference']['target'] == base['reference']['target']:
        raise ValueError('Counterfactual has no schema label change')
    return base, counter


def assemble(groups, eligible, revisions, audits, facts, rules, parent_facts):
    rows, review = {}, []
    # Keep the stricter deletion contrasts wherever they independently passed.
    for gid, group in groups.items():
        try:
            accepted, _ = evaluate_group(group, rules[gid], parent_facts)
            rows.update((r['id'], r) for r in accepted)
        except (ValueError, KeyError, TypeError):
            pass
    for gid, group in eligible.items():
        try:
            base, counter = assess_revision(group, revisions[gid], audits[gid], facts, rules, parent_facts)
            rows.setdefault(base['id'], base)
            rows[counter['id']] = counter
        except (ValueError, KeyError, TypeError, IndexError) as error:
            review.append({'group_id': gid, 'reasons': [str(error)]})
    return sorted(rows.values(), key=lambda r: r['id']), review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PROJECT_ROOT / 'data/evidence_curated_gpt56_v2')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/evidence_curated_gpt56_v2_expanded')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    parent_config = json.loads((args.parent / 'run_config.json').read_text())
    source_path = Path(parent_config['source'])
    sources = read_jsonl(source_path)
    if fingerprint(sources) != parent_config['source_sha256']:
        raise ValueError('Source changed')
    jobs = {j['id']: j for j in select_sources(sources, parent_config['source_seeds'])}
    groups = {g['id']: g for g in read_jsonl(args.parent / 'constructed_groups.jsonl')}
    for gid, group in groups.items():
        if build_group(jobs[gid], group['spec']) != group:
            raise ValueError('Parent context changed')
    rules, parent_facts = (load_stage(args.parent / (s + '.jsonl')) for s in ('rule_audits', 'fact_audits'))
    eligible, rejected = eligible_groups(groups, rules, parent_facts)
    config = {'pipeline_version': 'evidence-curation-v2-counterfactuals',
              'parent_config_sha256': fingerprint(parent_config),
              'parent_artifacts_sha256': fingerprint([groups, rules, parent_facts]),
              'implementation_sha256': fingerprint(Path(__file__).read_text()),
              'model': parent_config['model'], 'source_sha256': fingerprint(sources)}
    print(json.dumps({'eligible_groups': len(eligible), 'parent_groups': len(groups)}), flush=True)
    if args.offline:
        if json.loads((args.output / 'run_config.json').read_text()) != config:
            raise ValueError('Configuration/source changed')
        revisions, audits, facts = (load_stage(args.output / (s + '.jsonl'))
                                    for s in ('revisions', 'revision_audits', 'revision_facts'))
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
        requests = [{'job_id': gid, 'payload_json': canonical({'input': {
            'state': g['states']['base'], 'questions': {'decision': g['question']}},
            'focus_proposition': next(a['statement'] for a in g['spec']['atoms'] if a['id'] == g['spec']['focus_atom']),
            'evidence_pair': [s['text'] for s in g['spec']['focus_evidence']]})} for gid, g in eligible.items()]
        revisions = run_stage(Revise, Revision, requests, args.output, config['model'], 'revisions')
        valid = {}
        for gid, revision in revisions.items():
            try:
                state, pair = revised_context(eligible[gid], revision)
                valid[gid] = state
            except (ValueError, TypeError, KeyError, IndexError) as error:
                print(f'Invalid revision for {gid}: {error}', flush=True)
        audits = run_stage(ReviewRevision, RevisionAudit, [{'job_id': gid, 'payload_json': canonical({
            'question': eligible[gid]['question'], 'original_context': eligible[gid]['states']['base'],
            'revised_context': state, 'edit': revisions[gid]})} for gid, state in valid.items()],
            args.output, config['model'], 'revision_audits')
        facts = run_stage(FactReviewer, FactAudit,
                         [req for gid in valid for req in revision_requests(eligible[gid], revisions[gid])],
                         args.output, config['model'], 'revision_facts')
    rows, review = assemble(groups, eligible, revisions, audits, facts, rules, parent_facts)
    source_map = {r['id']: r for r in sources}
    held_out = [r for r in sources if r['split'] != 'train']
    held_out_states = {fingerprint(r['input']['state']) for r in held_out}
    held_out_families = {r['family'] for r in held_out}
    signatures = set()
    for row in rows:
        source = source_map[row['provenance']['source_id']]
        if source['split'] != 'train' or row['source_family'] in held_out_families or fingerprint(row['input']['state']) in held_out_states:
            raise ValueError('Held-out contamination')
        if row['input']['questions'] != source['input']['questions']:
            raise ValueError('Question changed')
        sig = fingerprint(row['input'])
        if sig in signatures:
            raise ValueError('Duplicate training input')
        signatures.add(sig)
    manifest = {**config, 'training': {'examples': len(rows), 'groups': len({r['family'] for r in rows}),
                'source_families': len({r['source_family'] for r in rows}),
                'primitives': dict(Counter(r['input']['questions']['decision']['type'] for r in rows)),
                'methods': dict(Counter(r['method'] for r in rows)),
                'variants': dict(Counter(r['variant'] for r in rows)),
                'domains': dict(Counter(r['domain'] for r in rows))},
                'train_sha256': fingerprint(rows), 'eligible_counterfactual_groups': len(eligible),
                'counterfactual_review_entries': len(review), 'parent_gate_rejections': len(rejected),
                'held_out_overlap': False, 'reference_human_reviewed': False,
                'notes': ['Includes accepted deletion contrasts from parent plus accepted counterfactual contrasts.',
                          'Missing-evidence cases without an audited outcome stay excluded.',
                          'Counterfactuals must establish refutation jointly, with neither sentence alone sufficient.',
                          'Rules, evidence, and edits are separately audited by the same model; errors may correlate.',
                          'Sources are synthetic. This does not establish human correctness or improved model accuracy.',
                          'Existing v1 trainer requires a loader update for these v2 certificates and variable-size groups.']}
    if args.offline:
        if rows != read_jsonl(args.output / 'train.jsonl'):
            raise ValueError('Stored rows differ from complete evidence reconstruction')
        saved = json.loads((args.output / 'manifest.json').read_text())
        if manifest != saved:
            raise ValueError('Manifest changed')
        print(json.dumps(manifest['training'], indent=2))
        return
    write_jsonl(args.output / 'train.jsonl', rows)
    write_jsonl(args.output / 'train_scoring.jsonl', [scoring_record(r) for r in rows])
    write_jsonl(args.output / 'review_queue.jsonl', review)
    write_jsonl(args.output / 'parent_gate_rejections.jsonl', rejected)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
