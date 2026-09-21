"""Evidence-level curation: checked premises, explicit rules, abstention on gaps."""
import copy
import itertools
import json
from collections import Counter, defaultdict

from nimble.datasets.dataset_io import canonical
from nimble.datasets.contrastive_data import (
    apply_edit, decode_target, fingerprint, scoring_record, text_at,
)

STATES = ('supported', 'refuted', 'unknown')
VARIANTS = ('base', 'remove_left', 'remove_right')
VERSION = 'evidence-curation-v2'


def select_sources(sources, limit=60):
    buckets = defaultdict(list)
    for row in sorted(sources, key=lambda r: r['id']):
        if row['split'] == 'train':
            buckets[(row['domain'], row['input']['questions']['decision']['type'])].append(row)
    if limit <= 0 or limit > sum(map(len, buckets.values())):
        raise ValueError('Limit must select a nonempty subset of training seeds')
    jobs = []
    for round_index in range(max(map(len, buckets.values()))):
        for key, rows in sorted(buckets.items()):
            if round_index < len(rows):
                source = rows[round_index]
                jobs.append({'id': 'ev2-' + source['id'], 'source': source,
                             'method': 'c2d' if round_index % 2 == 0 else 'source_preserving'})
                if len(jobs) == limit:
                    return jobs
    return jobs


def compile_rules(spec, question):
    atoms = {a['id']: a['statement'] for a in spec['atoms']}
    if not 2 <= len(atoms) <= 6 or len(atoms) != len(spec['atoms']):
        raise ValueError('Expected 2-6 uniquely named atomic propositions')
    if any(not v.strip() for v in atoms.values()) or spec['focus_atom'] not in atoms:
        raise ValueError('Invalid atom statement or focus atom')
    if not 1 <= len(spec['rules']) <= 16:
        raise ValueError('Expected 1-16 sufficient-condition rules')
    for rule in spec['rules']:
        conditions = rule['when']
        if not conditions or len({c['atom_id'] for c in conditions}) != len(conditions):
            raise ValueError('Each rule must have unique nonempty conditions')
        if any(c['atom_id'] not in atoms or c['state'] not in STATES for c in conditions):
            raise ValueError('Invalid rule atom/state')
        decode_target(rule['target'], question)
    # Rules are unordered sufficient conditions. Ambiguous overlaps NEVER use
    # list position as a hidden priority; make exclusions explicit in the rule.
    for states in itertools.product(STATES, repeat=len(atoms)):
        mapping = dict(zip(atoms, states))
        decide(spec, mapping, question)


def decide(spec, statuses, question):
    matches = [rule for rule in spec['rules']
               if all(statuses[c['atom_id']] == c['state'] for c in rule['when'])]
    targets = {rule['target'] for rule in matches}
    if len(targets) > 1:
        raise ValueError('Conflicting sufficient rules; priority must be explicit')
    return decode_target(next(iter(targets)), question) if targets else None


def build_group(job, spec):
    source = job['source']
    if source['split'] != 'train':
        raise ValueError('Held-out source')
    question = source['input']['questions']['decision']
    compile_rules(spec, question)
    base = (json.loads(spec['base_state_json']) if job['method'] == 'c2d'
            else copy.deepcopy(source['input']['state']))
    if not isinstance(base, (str, dict, list)) or len(canonical(base).split()) < 30:
        raise ValueError('A substantive natural-language context is required')
    if job['method'] == 'c2d' and canonical(base) == canonical(source['input']['state']):
        raise ValueError('C2D must create a new context')
    pair = spec['focus_evidence']
    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError('Focus must have two different evidence sentences')
    states = {'base': base}
    for name, span in zip(VARIANTS[1:], pair):
        if len(span['text'].split()) < 4:
            raise ValueError('Evidence must be a substantive sentence, not a token cue')
        states[name] = apply_edit(base, {'path': span['path'], 'old': span['text'], 'new': ''})
    # Both edits must be independent, nonoverlapping spans.
    both_removed = base
    for span in pair:
        both_removed = apply_edit(both_removed, {'path': span['path'], 'old': span['text'], 'new': ''})
    if len({canonical(s) for s in states.values()}) != 3:
        raise ValueError('Duplicate variant')
    for span in spec['policy_evidence']:
        if not span['text'].strip() or span['text'] not in text_at(source['input']['state'], span['path']):
            raise ValueError('Policy quotation must come verbatim from the original source state')
        if any(span['text'] not in canonical(s) and span['text'] not in flatten(s) for s in states.values()):
            raise ValueError('Policy evidence must survive all variants verbatim')
    return {'id': job['id'], 'method': job['method'], 'source_id': source['id'],
            'source_family': source['family'], 'domain': source['domain'], 'spec': copy.deepcopy(spec),
            'question': copy.deepcopy(question), 'states': states,
            'source_sha256': fingerprint(source)}


def flatten(state):
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return '\n'.join(flatten(v) for v in state)
    if isinstance(state, dict):
        return '\n'.join(flatten(v) for v in state.values())
    return str(state)


def fact_jobs(group):
    atoms = group['spec']['atoms']
    focus = [a for a in atoms if a['id'] == group['spec']['focus_atom']]
    pair = group['spec']['focus_evidence']
    cases = [(variant, state, atoms) for variant, state in group['states'].items()]
    cases += [('left_only', pair[0]['text'], focus), ('right_only', pair[1]['text'], focus),
              ('pair_only', pair[0]['text'] + '\n' + pair[1]['text'], focus)]
    # Each context is a separate API call. Neither rules, schema labels, other
    # contexts, source reference, nor expected fact states enter this payload.
    return [{'job_id': group['id'] + ':' + name,
             'payload_json': canonical({'context': context, 'propositions': propositions})}
            for name, context, propositions in cases]


def rule_job(group, source):
    return {'job_id': group['id'], 'payload_json': canonical({
        'original_input': source['input'], 'constructed_context': group['states']['base'],
        'atoms': group['spec']['atoms'], 'rules': group['spec']['rules'],
        'focus_atom': group['spec']['focus_atom'], 'focus_evidence': group['spec']['focus_evidence'],
        'policy_evidence': group['spec']['policy_evidence']})}


def assess_facts(job, response):
    payload = json.loads(job['payload_json'])
    expected = {a['id'] for a in payload['propositions']}
    judgments = response['judgments']
    if len(judgments) != len(expected) or {j['atom_id'] for j in judgments} != expected:
        raise ValueError('Missing or duplicate fact judgment')
    for judgment in judgments:
        if judgment['state'] not in (*STATES, 'inconsistent'):
            raise ValueError('Invalid fact state')
        if judgment['state'] in ('supported', 'refuted') and not judgment['quotes']:
            raise ValueError('Supported/refuted judgments need literal evidence')
        if any(not q.strip() or q not in flatten(payload['context']) for q in judgment['quotes']):
            raise ValueError('Verifier evidence quote absent from its context')
        if not judgment['reason'].strip():
            raise ValueError('Missing judgment rationale')
    return {j['atom_id']: j['state'] for j in judgments}


def evaluate_group(group, audit, fact_results):
    """Return derived rows and exclusions, never ask a model for final gold labels."""
    issues, excluded, rows = [], [], []
    checks = audit['rule_checks']
    if len(checks) != len(group['spec']['rules']) or {r['rule_index'] for r in checks} != set(range(len(checks))):
        raise ValueError('Missing/duplicate rule audit')
    for key in ('atoms_are_atomic', 'policy_preserved', 'focus_is_relevant', 'evidence_is_two_sentences'):
        if not audit[key]:
            issues.append(key)
    for check in checks:
        if not check['sound']:
            issues.append('unsound_rule_' + str(check['rule_index']))
    matrices = {}
    for job in fact_jobs(group):
        case = job['job_id'].split(':')[-1]
        matrices[case] = assess_facts(job, fact_results[job['job_id']])
    focus = group['spec']['focus_atom']
    for case, expected in [('left_only', 'unknown'), ('right_only', 'unknown'),
                           ('pair_only', 'supported'), ('base', 'supported'),
                           ('remove_left', 'unknown'), ('remove_right', 'unknown')]:
        if matrices[case][focus] != expected:
            issues.append(f'{case}:focus_{matrices[case][focus]}_expected_{expected}')
    if any('inconsistent' in matrix.values() for matrix in matrices.values()):
        issues.append('inconsistent_evidence')
    if issues:
        return [], [{'group_id': group['id'], 'reasons': issues, 'audit': audit, 'fact_states': matrices}]
    for variant, state in group['states'].items():
        target = decide(group['spec'], matrices[variant], group['question'])
        if target is None:
            excluded.append({'group_id': group['id'], 'variant': variant,
                             'reasons': ['no_audited_rule_covers_fact_states'], 'fact_states': matrices[variant]})
            continue
        row = {'id': group['id'] + '-' + variant, 'family': group['id'],
               'source_family': group['source_family'], 'split': 'train', 'domain': group['domain'],
               'method': group['method'], 'variant': variant,
               'input': {'state': state, 'questions': {'decision': group['question']}},
               'reference': {'target': target, 'source': 'audited_rule_over_verified_facts',
                             'human_reviewed': False},
               'provenance': {'source_id': group['source_id'], 'source_split': 'train',
                              'source_sha256': group['source_sha256'], 'source_is_synthetic': True},
               'quality_status': 'evidence_model_checked',
               'evidence_certificate': {'rule_audit': audit, 'fact_states': matrices[variant],
                                        'spec': group['spec'], 'necessity_checks_passed': True,
                                        'verifier_independent_model': False}}
        rows.append(row)
    # Preserve the base and at least one genuine schema decision contrast. Merely
    # proving a fact changes support is insufficient if the schema answer never changes.
    if (not any(r['variant'] == 'base' for r in rows)
            or len({canonical(r['reference']['target']) for r in rows}) < 2):
        excluded.append({'group_id': group['id'], 'reasons': ['no_determinate_label_contrast'],
                         'fact_states': matrices})
        return [], excluded
    return rows, excluded


def audit_training(rows, sources):
    source_map = {r['id']: r for r in sources}
    heldout = [r for r in sources if r['split'] != 'train']
    heldout_states = {fingerprint(r['input']['state']) for r in heldout}
    heldout_families = {r['family'] for r in heldout}
    seen, groups = set(), defaultdict(list)
    for row in rows:
        source = source_map[row['provenance']['source_id']]
        if (source['split'] != 'train' or row['split'] != 'train'
                or row['source_family'] in heldout_families
                or fingerprint(row['input']['state']) in heldout_states):
            raise ValueError('Held-out leakage')
        if row['source_family'] != source['family'] or row['provenance']['source_sha256'] != fingerprint(source):
            raise ValueError('Invalid source provenance')
        if row['input']['questions'] != source['input']['questions']:
            raise ValueError('Schema changed')
        sig = fingerprint(row['input'])
        if sig in seen:
            raise ValueError('Duplicate training input')
        seen.add(sig)
        certificate = row['evidence_certificate']
        compile_rules(certificate['spec'], row['input']['questions']['decision'])
        actual = decide(certificate['spec'], certificate['fact_states'], row['input']['questions']['decision'])
        if actual != row['reference']['target'] or not certificate['necessity_checks_passed']:
            raise ValueError('Invalid evidence certificate')
        scoring_record(row)
        groups[row['family']].append(row)
    for group in groups.values():
        variants = [r['variant'] for r in group]
        if len(variants) != len(set(variants)) or 'base' not in variants or not set(variants) <= set(VARIANTS):
            raise ValueError('Invalid group composition')
        if len({canonical(r['reference']['target']) for r in group}) < 2:
            raise ValueError('No schema decision contrast')
    return {'examples': len(rows), 'groups': len(groups),
            'domains': dict(Counter(r['domain'] for r in rows)),
            'primitives': dict(Counter(r['input']['questions']['decision']['type'] for r in rows)),
            'methods': dict(Counter(r['method'] for r in rows)),
            'variants': dict(Counter(r['variant'] for r in rows)),
            'held_out_overlap': False}
