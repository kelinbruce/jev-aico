"""Curate 1,000 evidence-checked rows, with resumable API stages and offline replay."""
import argparse
import copy
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.datasets.create_diverse_dataset import configure, read_jsonl
from nimble.datasets.dataset_io import canonical, write_jsonl
from nimble.datasets.curation_options import llm_options
from nimble.datasets.contrastive_data import fingerprint, scoring_record
from nimble.datasets.evidence_curation import assess_facts
from nimble.datasets.evidence_stages import FactReviewer, FactAudit
from nimble.datasets.curate_paired_evidence import Pair, PairGenerator, Document, DocumentGenerator, pair_requests
from nimble.datasets.scaled_evidence_stages import (
    RulePlan, PlanAudit, PlanGenerator, PlanReviewer, ContextAudit, ContextReviewer, FullInputContextReviewer,
)
from nimble.datasets.scaled_evidence import (
    normalize_plan, validate_plan, plan_passes, build_group, full_requests, make_rows, quotas, audit_release,
)


VERSION = 'evidence-curation-v3'


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def scoring_exports(rows):
    # Match the key order of persisted train.jsonl before serializing a structured
    # state into prompt text. Otherwise fresh API object order changes the prompt
    # relative to reloaded rows, even though their JSON values are identical.
    return [scoring_record(json.loads(canonical(row))) for row in rows]


class Stages:
    def __init__(self, output, model, offline, concurrency, requests_per_minute=240):
        self.output, self.model, self.offline, self.concurrency = output, model, offline, concurrency
        self.requests_per_minute = requests_per_minute

    def __call__(self, cls, schema, requests, name):
        path = self.output / (name + '.jsonl')
        saved = {r['job_id']: r for r in read_jsonl(path)} if path.exists() else {}
        if len({r['job_id'] for r in requests}) != len(requests):
            raise ValueError('Duplicate request ID')
        for req in requests:
            if req['job_id'] in saved and saved[req['job_id']]['request_sha256'] != fingerprint(req):
                raise ValueError('Cached request changed: ' + req['job_id'])
        pending = [r for r in requests if r['job_id'] not in saved]
        if self.offline and pending:
            raise ValueError('Offline replay lacks stage: ' + name)
        if pending:
            print(f'{name}: {len(pending)} API requests; {len(saved)} cached.', flush=True)
            options = llm_options(self.model, schema)
            options['backend_params']['max_concurrent_requests'] = self.concurrency
            options['backend_params']['max_requests_per_minute'] = self.requests_per_minute
            if schema in (RulePlan, PlanAudit):
                options['generation_params']['max_completion_tokens'] = 16384
            result = cls(**options)(pending, working_dir=str(PROJECT_ROOT / '.cache/curator-evidence-v3' / name))
            lookup = {r['job_id']: r for r in pending}
            for item in result.dataset:
                response = json.loads(item['result_json'])
                schema.model_validate(response)
                saved[item['job_id']] = {'job_id': item['job_id'],
                    'request_sha256': fingerprint(lookup[item['job_id']]), 'response': response}
            temp = path.with_suffix('.jsonl.tmp')
            write_jsonl(temp, sorted(saved.values(), key=lambda r: r['job_id']))
            temp.replace(path)
        if not self.offline and not path.exists():
            write_jsonl(path, [])
        return {r['job_id']: saved[r['job_id']]['response'] for r in requests}


def request(gid, payload):
    return {'job_id': gid, 'payload_json': canonical(payload)}


def prepare_plans(sources, stage, attempts):
    pending = {r['id']: r for r in sources if r['split'] == 'train'}
    accepted, history, feedback, previous = {}, [], {}, {}
    for attempt in range(attempts):
        drafts = stage(PlanGenerator, RulePlan, [request(sid, {
            'original_input': source['input'],
            **({'previous_proposal': previous[sid], 'reviewer_feedback': feedback[sid]} if sid in previous else {})})
            for sid, source in sorted(pending.items())], f'plan_{attempt}_generation')
        valid = {}
        for sid, plan in drafts.items():
            previous[sid] = plan
            try:
                plan = normalize_plan(plan, pending[sid])
                validate_plan(plan, pending[sid])
                valid[sid] = plan
            except (ValueError, KeyError, TypeError, IndexError) as error:
                feedback[sid] = {'structural_error': str(error)}
                history.append({'source_id': sid, 'attempt': attempt, 'reason': str(error)})
        audits = stage(PlanReviewer, PlanAudit, [request(sid, {
            'original_input': pending[sid]['input'], 'plan': plan}) for sid, plan in sorted(valid.items())],
            f'plan_{attempt}_audits')
        for sid, audit in audits.items():
            if plan_passes(audit, valid[sid]):
                accepted[sid] = {'source_id': sid, 'plan': valid[sid], 'audit': audit, 'attempt': attempt}
            else:
                feedback[sid] = audit
                history.append({'source_id': sid, 'attempt': attempt, 'reason': 'rule_audit_failed', 'audit': audit})
        pending = {sid: s for sid, s in pending.items() if sid not in accepted}
        print(f'Rule pass {attempt + 1}: {len(accepted)}/240 accepted plans; {len(pending)} remaining.', flush=True)
    return accepted, history


def construct_batch(jobs, plans, source_map, stage, index):
    prefix = f'round_{index:02d}'
    atoms = {j['id']: next(a for a in plans[j['source_id']]['plan']['atoms']
                           if a['id'] == plans[j['source_id']]['plan']['focus_atom']) for j in jobs}
    pairs = stage(PairGenerator, Pair, [request(j['id'], {
        'proposition': atoms[j['id']]['statement'],
        'original_question': source_map[j['source_id']]['input']['questions']['decision'],
        'unchanged_policy': plans[j['source_id']]['plan']['policy_evidence'],
        'background_fact_assignments': plans[j['source_id']]['plan'],
        'variation': j['variation'],
        'diversity_instruction': 'Use a fresh pair of observed facts, with different incidental numbers, '
            'identifiers and wording from previous variants. Preserve proposition bindings and policy. '
            'Do not make either sentence a policy definition. Realize the base and counter factual states.'})
        for j in jobs], prefix + '_pairs')
    pair_reqs, review, matrices = [], [], {}
    for j in jobs:
        try:
            pair_reqs.extend(pair_requests(j['id'], atoms[j['id']], pairs[j['id']]))
        except (ValueError, TypeError, KeyError) as error:
            review.append({'group_id': j['id'], 'stage': 'pair_structure', 'reason': str(error)})
    pair_checks = stage(FactReviewer, FactAudit, pair_reqs, prefix + '_pair_checks')
    expected = {'left': 'unknown', 'right': 'unknown', 'positive_pair': 'supported',
                'negative_sentence': 'unknown', 'negative_pair': 'refuted'}
    passed = []
    for j in jobs:
        gid = j['id']
        try:
            matrix = {r['job_id'].split(':')[-1]: assess_facts(r, pair_checks[r['job_id']])
                      for r in pair_requests(gid, atoms[gid], pairs[gid])}
            if any(matrix[k][atoms[gid]['id']] != v for k, v in expected.items()):
                raise ValueError('Pair entailment or sentence insufficiency failed: ' + canonical(matrix))
            matrices[gid] = matrix
            passed.append(j)
        except (ValueError, TypeError, KeyError) as error:
            review.append({'group_id': gid, 'stage': 'pair_facts', 'reason': str(error)})
    print(f'{prefix}: {len(passed)}/{len(jobs)} evidence pairs passed.', flush=True)
    doc_requests = [request(j['id'], {
        'original_input': source_map[j['source_id']]['input'],
        **plans[j['source_id']]['plan'], 'verified_pair': pairs[j['id']], 'variation': j['variation'],
        'required_fact_assignments': plans[j['source_id']]['plan']['base_states'],
        'instruction': 'Establish every base_states assignment explicitly. Keep all non-focus facts '
            'unchanged under the counterfactual. Do not introduce extra support for the focus.'})
        for j in passed]
    docs = stage(DocumentGenerator, Document, doc_requests, prefix + '_documents')
    built = {}
    repairs = []
    original_requests = {r['job_id']: r for r in doc_requests}
    jobs_by_id = {j['id']: j for j in passed}
    for j in passed:
        try:
            built[j['id']] = build_group(j['id'], source_map[j['source_id']], plans[j['source_id']]['plan'],
                                        docs[j['id']], pairs[j['id']])
        except (ValueError, KeyError, TypeError, IndexError) as error:
            review.append({'group_id': j['id'], 'stage': 'document_structure', 'reason': str(error)})
            repairs.append(request(j['id'], {
                **json.loads(original_requests[j['id']]['payload_json']),
                'previous_document': docs[j['id']], 'structural_error': str(error),
                'required_state_kind': type(source_map[j['source_id']]['input']['state']).__name__,
                'repair_instructions': 'Repair the context serialization/format. base_state_json '
                    'must decode to the required_state_kind of original_input.state, NOT to the '
                    'original_input wrapper. For str, produce a natural-language paragraph. For '
                    'list, produce a dialogue array. For dict, produce only the context object. '
                    'NEVER include questions or an input wrapper inside the context. Keep the '
                    'verified evidence sentences and quoted governing policy VERBATIM. Return '
                    'correct string-leaf paths relative to the context itself; for str use []. '
                    'Preserve the factual contrast and all required fact assignments.'}))
    repaired = stage(DocumentGenerator, Document, repairs, prefix + '_document_repairs')
    for gid, document in repaired.items():
        j = jobs_by_id[gid]
        try:
            built[gid] = build_group(gid, source_map[j['source_id']], plans[j['source_id']]['plan'],
                                     document, pairs[gid])
        except (ValueError, KeyError, TypeError, IndexError) as error:
            review.append({'group_id': gid, 'stage': 'document_structure_after_repair', 'reason': str(error)})
    reviewer = FullInputContextReviewer if index >= 5 else ContextReviewer
    audits = stage(reviewer, ContextAudit, [request(gid, {
        'original_input': source_map[g['source_id']]['input'],
        'base_context': g['states']['base'], 'counterfactual_context': g['states']['counterfactual'],
        'focus_evidence': g['spec']['focus_evidence']}) for gid, g in sorted(built.items())], prefix + '_context_audits')
    viable = {}
    for gid, group in built.items():
        if all(v for k, v in audits[gid].items() if k != 'explanation'):
            viable[gid] = group
        else:
            review.append({'group_id': gid, 'stage': 'context_audit', 'reason': audits[gid]['explanation'], 'audit': audits[gid]})
    facts = stage(FactReviewer, FactAudit, [r for g in viable.values() for r in full_requests(g)], prefix + '_full_checks')
    accepted = []
    for gid, group in viable.items():
        try:
            full = {r['job_id'].split(':')[-1]: assess_facts(r, facts[r['job_id']]) for r in full_requests(group)}
            rows = make_rows(group, plans[group['source_id']]['audit'], pairs[gid], matrices[gid], audits[gid], full)
            accepted.append({'group_id': gid, 'rows': rows})
        except (ValueError, KeyError, TypeError) as error:
            review.append({'group_id': gid, 'stage': 'full_facts', 'reason': str(error)})
    print(f'{prefix}: {len(accepted)}/{len(jobs)} complete contrast groups accepted.', flush=True)
    return accepted, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT_ROOT / 'data/typesafe_diverse_300_gpt56/all.jsonl')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/evidence_curated_1000_gpt56')
    parser.add_argument('--target', type=int, default=1000)
    parser.add_argument('--model', default='gpt-5.6-sol')
    parser.add_argument('--plan-attempts', type=int, default=3)
    parser.add_argument('--max-rounds', type=int, default=24)
    parser.add_argument('--per-bucket', type=int, default=4)
    parser.add_argument('--concurrency', type=int, default=64)
    parser.add_argument('--requests-per-minute', type=int, default=240)
    parser.add_argument('--phase', choices=['plans', 'all'], default='all')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    sources = read_jsonl(args.source)
    source_map = {r['id']: r for r in sources}
    limits = quotas(sources, args.target)
    code_files = [Path(__file__), Path(__file__).with_name('scaled_evidence.py'),
                  Path(__file__).with_name('scaled_evidence_stages.py'),
                  Path(__file__).with_name('scaled_evidence_refinement.py'),
                  Path(__file__).with_name('evidence_stages.py'), Path(__file__).with_name('curate_paired_evidence.py')]
    config = {'pipeline_version': VERSION, 'source': str(args.source.resolve()), 'source_sha256': fingerprint(sources),
              'model': args.model, 'target': args.target, 'plan_attempts': args.plan_attempts,
              'max_rounds': args.max_rounds, 'per_bucket': args.per_bucket,
              'implementation_sha256': fingerprint({p.name: p.read_text() for p in code_files}),
              'group_quotas': {d + '/' + t: n for (d, t), n in limits.items()}}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Output belongs to a different source/configuration/implementation')
    if not args.offline:
        atomic_json(config_path, config)
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / '.env', override=False)
        if not os.environ.get('OPENAI_API_KEY'):
            parser.error('OPENAI_API_KEY is required')
        configure()
    stage = Stages(args.output, args.model, args.offline, args.concurrency, args.requests_per_minute)
    plans, plan_review = prepare_plans(sources, stage, args.plan_attempts)
    if not args.offline:
        write_jsonl(args.output / 'accepted_plans.jsonl', [plans[k] for k in sorted(plans)])
        write_jsonl(args.output / 'plan_review.jsonl', plan_review)
    if args.phase == 'plans':
        print(json.dumps({'plans': len(plans), 'source_families': len({source_map[s]['family'] for s in plans})}), flush=True)
        return
    pools = defaultdict(list)
    for sid in sorted(plans):
        s = source_map[sid]
        pools[(s['domain'], s['input']['questions']['decision']['type'])].append(sid)
    missing = [key for key, n in limits.items() if n and not pools[key]]
    if missing:
        raise ValueError('No audited plans for required categories: ' + repr(missing))
    rows, review, used_inputs = [], [], set()
    attempted, selected, selected_sources = Counter(), Counter(), Counter()
    selected_families = Counter()
    completed_rounds = 0
    for index in range(args.max_rounds):
        if index == 5 and len(rows) < args.target:
            from nimble.datasets.scaled_evidence_refinement import refine_plans
            plans = refine_plans(plans, source_map, attempted, selected_sources, selected, limits,
                                 review, stage, index)
        jobs = []
        for bucket, quota in limits.items():
            remaining = quota - selected[bucket]
            for _ in range(min(args.per_bucket, remaining)):
                sid = min(pools[bucket], key=lambda s: (attempted[s], selected_families[source_map[s]['family']],
                                                       selected_sources[s], s))
                attempted[sid] += 1
                ordinal = attempted[sid]
                jobs.append({'id': f'scale-{sid}-{ordinal:03d}', 'source_id': sid,
                             'variation': {'ordinal': ordinal, 'seed': 17 + ordinal * 997,
                                           'style': ('concise field note', 'chronological account',
                                                     'evidence reconciliation', 'operational handoff')[ordinal % 4]}})
        if not jobs:
            break
        accepted, failures = construct_batch(jobs, plans, source_map, stage, index)
        review.extend(failures)
        for group in accepted:
            a = group['rows'][0]
            bucket = (a['domain'], a['input']['questions']['decision']['type'])
            hashes = {fingerprint(r['input']) for r in group['rows']}
            if used_inputs & hashes:
                review.append({'group_id': group['group_id'], 'stage': 'selection', 'reason': 'duplicate_input'})
                continue
            rows.extend(group['rows'])
            used_inputs.update(hashes)
            selected[bucket] += 1
            selected_sources[a['provenance']['source_id']] += 1
            selected_families[a['source_family']] += 1
        rows.sort(key=lambda r: r['id'])
        completed_rounds = index + 1
        audit = audit_release(rows, sources)
        progress = {'accepted_examples': len(rows), 'target': args.target, 'completed_rounds': completed_rounds,
                    'attempted_groups': sum(attempted.values()), 'training': audit,
                    'remaining_groups': {d + '/' + t: n - selected[(d, t)] for (d, t), n in limits.items()},
                    'review_entries': len(review)}
        if not args.offline:
            write_jsonl(args.output / 'accepted_progress.jsonl', rows)
            write_jsonl(args.output / 'review_queue.jsonl', review)
            atomic_json(args.output / 'progress.json', progress)
        print('PROGRESS ' + json.dumps(progress), flush=True)
    if len(rows) != args.target:
        raise ValueError(f'Bounded run accepted {len(rows)}/{args.target}; inspect progress/review before extending')
    audit = audit_release(rows, sources)
    manifest = {**config, 'training': audit, 'train_sha256': fingerprint(rows),
                'audited_plans': len(plans), 'attempted_groups': sum(attempted.values()),
                'completed_rounds': completed_rounds, 'review_entries': len(review),
                'held_out_overlap': False, 'reference_human_reviewed': False,
                'notes': ['500 complete base/counterfactual groups; preserve source_family grouping in any split.',
                          'Deletion variants check necessity but are not exported as labels.',
                          'Each isolated sentence, pair and full context checked in separate API calls.',
                          'Rules audited before construction; context policy and coherence audited afterward.',
                          'All sources and generated contexts synthetic; same model generates and verifies.',
                          'No measured training improvement or human ground-truth claim.']}
    if args.offline:
        if rows != read_jsonl(args.output / 'train.jsonl'):
            raise ValueError('Offline training reconstruction mismatch')
        if scoring_exports(rows) != read_jsonl(args.output / 'train_scoring.jsonl'):
            raise ValueError('Offline scoring reconstruction mismatch')
        if manifest != json.loads((args.output / 'manifest.json').read_text()):
            raise ValueError('Offline manifest reconstruction mismatch')
    else:
        write_jsonl(args.output / 'train.jsonl', rows)
        write_jsonl(args.output / 'train_scoring.jsonl', scoring_exports(rows))
        atomic_json(args.output / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
