"""Deterministic validation and release selection for scaled evidence curation."""
import copy
from collections import Counter, defaultdict

from nimble.datasets.dataset_io import canonical
from nimble.datasets.evidence_curation import STATES, decide, flatten
from nimble.datasets.contrastive_data import apply_edit, decode_target, fingerprint, text_at


def normalize_plan(plan, source):
    """Canonicalize literal citation coordinates; never repair quote text or rules.

    Some model outputs use full-input paths despite the state-relative contract.
    Verified question citations need no copying: that question is always retained.
    Preserve the raw proposal separately in the API stage file.
    """
    normalized = copy.deepcopy(plan)
    spans = []
    for span in plan['policy_evidence']:
        path, quote = list(span['path']), span['text']
        if path[:1] in (['input'], ['original_input']):
            path = path[1:]
        if path[:1] == ['questions']:
            if not quote.strip() or quote not in text_at(source['input'], path):
                raise ValueError('Question policy quote is not literal')
            continue
        # A state key can also be a real key inside a structured state. Prefer
        # an already-valid relative path before stripping the input-state root.
        relative_ok = False
        try:
            relative_ok = quote in text_at(source['input']['state'], path)
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        if not relative_ok and path[:1] == ['state']:
            path = path[1:]
        if not quote.strip() or quote not in text_at(source['input']['state'], path):
            raise ValueError('State policy quote is not literal')
        canonical_span = {'path': path, 'text': quote}
        if canonical_span not in spans:
            spans.append(canonical_span)
    normalized['policy_evidence'] = spans
    return normalized


def validate_plan(plan, source):
    if source['split'] != 'train':
        raise ValueError('Held-out source')
    atoms = {a['id']: a['statement'] for a in plan['atoms']}
    if not 2 <= len(atoms) <= 20 or len(atoms) != len(plan['atoms']):
        raise ValueError('Need 2-20 uniquely named atomic facts')
    if plan['focus_atom'] not in atoms or any(not v.strip() for v in atoms.values()):
        raise ValueError('Invalid focus or empty proposition')
    if not 2 <= len(plan['rules']) <= 8:
        raise ValueError('Need 2-8 rules')
    question = source['input']['questions']['decision']
    conditions = []
    for rule in plan['rules']:
        cond = {c['atom_id']: c['state'] for c in rule['when']}
        if not cond or len(cond) != len(rule['when']):
            raise ValueError('Empty or duplicate rule condition')
        if any(k not in atoms or v not in STATES for k, v in cond.items()):
            raise ValueError('Invalid rule condition')
        decode_target(rule['target'], question)
        conditions.append(cond)
    # Two conjunctions overlap iff none of their shared variables conflict. This
    # exact check replaces exponential enumeration and scales to 20 atomic facts.
    for i, lhs in enumerate(conditions):
        for j, rhs in enumerate(conditions[:i]):
            if plan['rules'][i]['target'] != plan['rules'][j]['target']:
                if not any(lhs[k] != rhs[k] for k in lhs.keys() & rhs.keys()):
                    raise ValueError('Conflicting sufficient-condition rules')
    assignments = []
    for key in ('base_states', 'counter_states'):
        assignment = {s['atom_id']: s['state'] for s in plan[key]}
        if set(assignment) != set(atoms) or len(assignment) != len(plan[key]):
            raise ValueError('Assignment must cover every atom exactly once')
        if any(v not in STATES for v in assignment.values()):
            raise ValueError('Invalid assignment state')
        if decide(plan, assignment, question) is None:
            raise ValueError('Proposed assignment is not covered')
        assignments.append(assignment)
    base, counter = assignments
    focus = plan['focus_atom']
    if base[focus] != 'supported' or counter[focus] != 'refuted':
        raise ValueError('Focus must flip supported to refuted')
    if {a for a in atoms if base[a] != counter[a]} != {focus}:
        raise ValueError('Only focus may change between proposed assignments')
    if decide(plan, base, question) == decide(plan, counter, question):
        raise ValueError('Proposed assignments do not change the schema label')
    for span in plan['policy_evidence']:
        if not span['text'].strip() or span['text'] not in text_at(source['input']['state'], span['path']):
            raise ValueError('Policy must quote an original string leaf verbatim')


def plan_passes(audit, plan):
    keys = ('atoms_are_atomic', 'policy_complete', 'focus_is_factual', 'assignments_are_realizable')
    checks = audit['rule_checks']
    return (all(audit[k] for k in keys) and len(checks) == len(plan['rules'])
            and {c['rule_index'] for c in checks} == set(range(len(checks)))
            and all(c['sound'] for c in checks))


def build_group(gid, source, plan, document, pair):
    import json
    validate_plan(plan, source)
    base = json.loads(document['base_state_json'])
    if not isinstance(base, (str, dict, list)) or len(flatten(base).split()) < 30:
        raise ValueError('Context is not substantive')
    if type(base) is not type(source['input']['state']):
        raise ValueError('Context kind differs from original')
    if canonical(base) == canonical(source['input']['state']):
        raise ValueError('Generated context must be new')
    spans = document['focus_evidence']
    if len(spans) != 2 or [s['text'] for s in spans] != [pair['left'], pair['right']]:
        raise ValueError('Document must embed both checked pair sentences verbatim in order')
    differences = [name for name in ('left', 'right') if pair[name] != pair['negative_' + name]]
    if len(differences) != 1:
        raise ValueError('Counterfactual must change exactly one sentence')
    states = {'base': base}
    both_removed = base
    for name, span in zip(('remove_left', 'remove_right'), spans):
        if len(span['text'].split()) < 4:
            raise ValueError('Evidence sentence too short')
        edit = {'path': span['path'], 'old': span['text'], 'new': ''}
        states[name] = apply_edit(base, edit)
        both_removed = apply_edit(both_removed, edit)
    changed = differences[0]
    span = spans[('left', 'right').index(changed)]
    states['counterfactual'] = apply_edit(base, {'path': span['path'], 'old': span['text'],
                                               'new': pair['negative_' + changed]})
    if len({canonical(s) for s in states.values()}) != 4:
        raise ValueError('Duplicate variants')
    for quote in plan['policy_evidence']:
        if any(quote['text'] not in flatten(s) for s in states.values()):
            raise ValueError('Governing policy missing or altered by an edit')
    return {'id': gid, 'source_id': source['id'], 'source_family': source['family'],
            'source_sha256': fingerprint(source), 'domain': source['domain'], 'method': 'c2d',
            'question': copy.deepcopy(source['input']['questions']['decision']),
            'spec': {**copy.deepcopy(plan), **document}, 'states': states}


def full_requests(group):
    focus = [a for a in group['spec']['atoms'] if a['id'] == group['spec']['focus_atom']]
    return [{'job_id': group['id'] + ':' + variant, 'payload_json': canonical({
        'context': state, 'propositions': group['spec']['atoms'] if variant in ('base', 'counterfactual') else focus})}
        for variant, state in group['states'].items()]


def make_rows(group, plan_audit, pair, pair_states, context_audit, full_states):
    required = ('policy_preserved', 'question_bindings_preserved', 'evidence_is_two_factual_sentences',
                'counterfactual_is_coherent', 'no_answer_leakage')
    if not all(context_audit[k] for k in required):
        raise ValueError('Context audit failed')
    if not plan_passes(plan_audit, group['spec']):
        raise ValueError('Rule audit failed')
    focus = group['spec']['focus_atom']
    expected = {'left': 'unknown', 'right': 'unknown', 'positive_pair': 'supported',
                'negative_sentence': 'unknown', 'negative_pair': 'refuted'}
    if any(pair_states[k][focus] != v for k, v in expected.items()):
        raise ValueError('Isolated pair evidence check failed')
    expected = {'base': 'supported', 'remove_left': 'unknown', 'remove_right': 'unknown', 'counterfactual': 'refuted'}
    if any(full_states[k][focus] != v for k, v in expected.items()):
        raise ValueError('Full-context focus/necessity check failed')
    if any('inconsistent' in v.values() for v in full_states.values()):
        raise ValueError('Inconsistent evidence')
    # This also ensures that a supposedly local edit did not change other facts.
    for variant, planned in (('base', 'base_states'), ('counterfactual', 'counter_states')):
        intended = {s['atom_id']: s['state'] for s in group['spec'][planned]}
        if full_states[variant] != intended:
            raise ValueError(variant + ': verified facts differ from audited contrast assignment')
    rows = []
    for variant in ('base', 'counterfactual'):
        target = decide(group['spec'], full_states[variant], group['question'])
        if target is None:
            raise ValueError('Uncovered fact assignment')
        rows.append({'id': group['id'] + '-' + variant, 'family': group['id'],
                     'source_family': group['source_family'], 'split': 'train', 'domain': group['domain'],
                     'method': 'c2d', 'variant': variant,
                     'input': {'state': group['states'][variant], 'questions': {'decision': group['question']}},
                     'reference': {'target': target, 'source': 'audited_rule_over_verified_facts', 'human_reviewed': False},
                     'provenance': {'source_id': group['source_id'], 'source_split': 'train',
                                    'source_sha256': group['source_sha256'], 'source_is_synthetic': True},
                     'quality_status': 'evidence_model_checked',
                     'evidence_certificate': {'pipeline_version': 'evidence-curation-v3',
                         'spec': group['spec'], 'rule_audit': plan_audit, 'context_audit': context_audit,
                         'verified_pair': pair, 'pair_fact_states': pair_states,
                         'full_context_fact_states': full_states, 'fact_states': full_states[variant],
                         'necessity_checks_passed': True, 'verifier_independent_model': False}})
    if rows[0]['reference']['target'] == rows[1]['reference']['target']:
        raise ValueError('No schema label contrast')
    return rows


def quotas(sources, total):
    if total <= 0 or total % 2:
        raise ValueError('Target must be a positive even number: complete two-example contrasts')
    buckets = sorted({(r['domain'], r['input']['questions']['decision']['type'])
                      for r in sources if r['split'] == 'train'})
    domains = sorted({d for d, _ in buckets})
    counts = {b: 0 for b in buckets}
    # Round-robin gives exactly 100 examples/domain for 1,000 and spreads the
    # remainder across types, rather than favoring alphabetically early domains.
    for i in range(total // 2):
        domain_index = i % len(domains)
        domain = domains[domain_index]
        kinds = sorted(t for d, t in buckets if d == domain)
        kind = kinds[(i // len(domains) + domain_index) % len(kinds)]
        counts[(domain, kind)] += 1
    return counts


def audit_release(rows, sources):
    lookup = {r['id']: r for r in sources}
    heldout = [r for r in sources if r['split'] != 'train']
    held_families = {r['family'] for r in heldout}
    held_states = {fingerprint(r['input']['state']) for r in heldout}
    seen, ids, groups = set(), set(), defaultdict(list)
    for row in rows:
        source = lookup[row['provenance']['source_id']]
        if source['split'] != 'train' or row['split'] != 'train' or row['source_family'] in held_families:
            raise ValueError('Held-out family leakage')
        if fingerprint(row['input']['state']) in held_states:
            raise ValueError('Held-out context overlap')
        if row['source_family'] != source['family'] or row['domain'] != source['domain']:
            raise ValueError('Source metadata changed')
        if row['provenance']['source_sha256'] != fingerprint(source):
            raise ValueError('Source fingerprint mismatch')
        if row['input']['questions'] != source['input']['questions']:
            raise ValueError('Original schema changed')
        key = fingerprint(row['input'])
        if key in seen or row['id'] in ids:
            raise ValueError('Duplicate training input or ID')
        seen.add(key)
        ids.add(row['id'])
        certificate = row['evidence_certificate']
        validate_plan(certificate['spec'], source)
        if not plan_passes(certificate['rule_audit'], certificate['spec']):
            raise ValueError('Missing passing rule audit')
        if row['reference']['target'] != decide(certificate['spec'], certificate['fact_states'], source['input']['questions']['decision']):
            raise ValueError('Target does not follow audited rules')
        groups[row['family']].append(row)
    for group in groups.values():
        if len(group) != 2 or {r['variant'] for r in group} != {'base', 'counterfactual'}:
            raise ValueError('Incomplete contrast group')
        if len({canonical(r['reference']['target']) for r in group}) != 2:
            raise ValueError('Group does not change label')
    return {'examples': len(rows), 'groups': len(groups),
            'source_families': len({r['source_family'] for r in rows}),
            'source_seeds': len({r['provenance']['source_id'] for r in rows}),
            'domains': dict(Counter(r['domain'] for r in rows)),
            'primitives': dict(Counter(r['input']['questions']['decision']['type'] for r in rows)),
            'variants': dict(Counter(r['variant'] for r in rows)),
            'domain_primitives': dict(Counter(r['domain'] + '/' + r['input']['questions']['decision']['type'] for r in rows)),
            'families': dict(Counter(r['source_family'] for r in rows))}
