"""One bounded redesign for unproductive plans in substantially underfilled buckets."""
import json

from nimble.datasets.create_diverse_dataset import read_jsonl
from nimble.datasets.dataset_io import canonical, write_jsonl
from nimble.datasets.scaled_evidence import normalize_plan, validate_plan, plan_passes
from nimble.datasets.scaled_evidence_stages import PlanGenerator, RulePlan, PlanReviewer, PlanAudit


def select_refinements(plans, source_map, attempted, selected_sources, selected, limits):
    eligible = []
    for sid in sorted(plans):
        source = source_map[sid]
        if source['split'] != 'train':
            raise ValueError('Held-out source in plan refinement')
        bucket = (source['domain'], source['input']['questions']['decision']['type'])
        remaining = limits[bucket] - selected[bucket]
        if (attempted[sid] >= 2 and selected_sources[sid] == 0
                and remaining >= 0.7 * limits[bucket]):
            eligible.append(sid)
    return eligible


def meaning(plan):
    return {'atoms': plan['atoms'], 'focus_atom': plan['focus_atom'], 'policy_evidence': plan['policy_evidence'],
            'rules': [{'when': r['when'], 'target': r['target']} for r in plan['rules']],
            'base_states': plan['base_states'], 'counter_states': plan['counter_states']}


def refine_plans(plans, source_map, attempted, selected_sources, selected, limits, review, stage, round_index):
    candidates = select_refinements(plans, source_map, attempted, selected_sources, selected, limits)
    failures = {sid: [] for sid in candidates}
    for item in review:
        sid = item['group_id'].removeprefix('scale-').rsplit('-', 1)[0]
        if sid in failures and item['stage'] != 'document_structure':
            failures[sid].append({'stage': item['stage'], 'reason': item['reason']})
    evidence = {sid: [] for sid in candidates}
    for index in range(round_index):
        path = stage.output / f'round_{index:02d}_full_checks.jsonl'
        if not path.exists():
            continue
        for item in read_jsonl(path):
            gid, case = item['job_id'].rsplit(':', 1)
            sid = gid.removeprefix('scale-').rsplit('-', 1)[0]
            if sid not in evidence:
                continue
            plan = plans[sid]['plan']
            expected = ({x['atom_id']: x['state'] for x in plan['base_states' if case == 'base' else 'counter_states']}
                        if case in ('base', 'counterfactual') else {plan['focus_atom']: 'unknown'})
            wrong = [j for j in item['response']['judgments'] if j['state'] != expected[j['atom_id']]]
            if wrong:
                evidence[sid].append({'case': case, 'unexpected_judgments': wrong})
    requests = [{'job_id': sid, 'payload_json': canonical({
        'original_input': source_map[sid]['input'], 'previous_proposal': plans[sid]['plan'],
        'reviewer_feedback': {'construction_failures': failures[sid][-4:], 'fact_checks': evidence[sid][-6:],
            'redesign_instruction': 'This rule plan passed review but at least two independent context '
                'constructions failed, and no training example from it was accepted. Redesign the '
                'contrast with a DIFFERENT factual focus or a substantively corrected decomposition. '
                'Prefer a record/identity join between observations. The focus must not already follow '
                'from governing policy, the unchanged question, or other required facts. Do not copy '
                'original case observations as policy. Keep only actual governing state policy in '
                'policy_evidence; the entire question is always retained automatically. Remove '
                'logically dependent or redundant atoms. Preserve all original schema meaning. '
                'Do not merely rename atoms or rewrite justifications.'}})} for sid in candidates]
    proposals = stage(PlanGenerator, RulePlan, requests, 'plan_refresh_generation')
    valid, excluded = {}, []
    for sid, proposal in proposals.items():
        try:
            proposal = normalize_plan(proposal, source_map[sid])
            validate_plan(proposal, source_map[sid])
            if meaning(proposal) == meaning(plans[sid]['plan']):
                raise ValueError('Refinement did not change the proposed contrast')
            valid[sid] = proposal
        except (ValueError, KeyError, TypeError, IndexError) as error:
            excluded.append({'source_id': sid, 'reason': str(error)})
    audits = stage(PlanReviewer, PlanAudit, [{'job_id': sid, 'payload_json': canonical({
        'original_input': source_map[sid]['input'], 'plan': proposal})} for sid, proposal in sorted(valid.items())],
        'plan_refresh_audits')
    result = dict(plans)
    changed = []
    for sid, audit in audits.items():
        if plan_passes(audit, valid[sid]):
            result[sid] = {'source_id': sid, 'plan': valid[sid], 'audit': audit,
                           'attempt': 'construction_refinement', 'active_from_round': round_index}
            changed.append(result[sid])
        else:
            excluded.append({'source_id': sid, 'reason': 'refined_rule_audit_failed', 'audit': audit})
    if not stage.offline:
        write_jsonl(stage.output / 'refined_plans.jsonl', changed)
        write_jsonl(stage.output / 'refinement_review.jsonl', excluded)
    print(f'Bounded plan refinement: {len(changed)}/{len(candidates)} revised plans passed fresh audits.', flush=True)
    return result
