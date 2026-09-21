"""Generate a fresh, held-out 1,000-example Curator dataset with Jev labels."""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from nimble.paths import PROJECT_ROOT
from nimble.datasets.create_diverse_dataset import (
    annotate, configure, export, generate_sources, read_jsonl,
)
from nimble.datasets.curation_profiles import EVALUATION_MODEL as GENERATOR_MODEL
from nimble.datasets.dataset_io import canonical
from nimble.datasets.diversity_plan import (
    DOMAINS, FORMATS, MECHANISMS, normalize_state, state_tokens, validate_row,
)

COUNT = 1000


def eval_plans(subjects, seed):
    if len(subjects) != 50 or Counter(r['domain'] for r in subjects) != {d: 5 for d, _ in DOMAINS}:
        raise ValueError("Expected five new subsubjects per domain")
    result = []
    for group, row in enumerate(sorted(subjects, key=lambda r: (r['domain_index'], r['subtopic_index']))):
        slots = []
        for slot in range(20):
            n = group * 20 + slot
            kind = ['choice', 'noul', 'score'][n % 3]
            levels = 3 + (n // 3) % 3
            slots.append({'slot': slot, 'type': kind, 'format': FORMATS[(n // 3) % 3],
                          'mechanism': MECHANISMS[n % 10],
                          'difficulty': ['straightforward', 'contextual', 'boundary_case'][(n // 9) % 3],
                          'target': ((n // 9) % levels if kind == 'score' else
                                     bool((n // 3) % 2) if kind == 'noul' else None),
                          'score_levels': levels if kind == 'score' else None,
                          'choice_options': 3 + (n // 3) % 4 if kind == 'choice' else None,
                          'include_no_match': (n // 3) % 5 == 0 if kind == 'choice' else False})
        result.append({**row, 'group_id': f"fresh-eval-{seed}-{row['group_id']}", 'split': 'eval',
                       'slots_json': canonical(slots), 'seed': seed, 'plan_version': 'fresh-eval-v1'})
    return result


def audit_eval(rows, previous=()):
    if len(rows) != COUNT or len({r['id'] for r in rows}) != COUNT:
        raise ValueError('Expected exactly 1,000 unique examples')
    if Counter(r['split'] for r in rows) != {'eval': COUNT}:
        raise ValueError('Fresh examples must all be reserved for evaluation')
    kinds = Counter(r['input']['questions']['decision']['type'] for r in rows)
    if kinds != {'choice': 334, 'noul': 333, 'score': 333}:
        raise ValueError('Primitive coverage differs from the saved plan')
    groups = Counter(r['family'] for r in rows)
    if len(groups) != 50 or set(groups.values()) != {20}:
        raise ValueError('Expected 50 groups of twenty')
    states = set()
    for row in rows:
        validate_row(row)
        state = normalize_state(row['input']['state'])
        if state in states:
            raise ValueError('Duplicate normalized state')
        states.add(state)
    previous_states = {normalize_state(r['input']['state']) for r in previous}
    if states & previous_states or {r['id'] for r in rows} & {r['id'] for r in previous}:
        raise ValueError('Expanded evaluation overlaps the prior dataset')
    tokens = [state_tokens(r['input']['state']) for r in rows]
    old_tokens = [state_tokens(r['input']['state']) for r in previous]
    near = []
    for i, left in enumerate(tokens):
        for j in range(i + 1, len(tokens)):
            similarity = len(left & tokens[j]) / len(left | tokens[j])
            if similarity >= .8:
                near.append({'left': rows[i]['id'], 'right': rows[j]['id'], 'token_jaccard': similarity})
        for j, right in enumerate(old_tokens):
            if len(left & right) / len(left | right) >= .8:
                raise ValueError(f"Near duplicate of prior data: {rows[i]['id']} and {previous[j]['id']}")
    return {'examples': COUNT, 'groups': len(groups), 'unique_states': len(states),
            'split_counts': {'eval': COUNT}, 'primitive_counts': dict(kinds),
            'domain_counts': dict(Counter(r['domain'] for r in rows)),
            'format_counts': dict(Counter(r['diversity']['format'] for r in rows)),
            'mechanism_counts': dict(Counter(r['diversity']['mechanism'] for r in rows)),
            'difficulty_counts': dict(Counter(r['diversity']['difficulty'] for r in rows)),
            'noul_targets': dict(Counter(str(r['reference']['target']).lower() for r in rows
                                        if r['input']['questions']['decision']['type'] == 'noul')),
            'score_targets': dict(Counter(str(r['reference']['target']) for r in rows
                                         if r['input']['questions']['decision']['type'] == 'score')),
            'choice_target_positions': dict(Counter(
                str(list(r['input']['questions']['decision']['criteria']).index(r['reference']['target']) + 1)
                for r in rows if r['input']['questions']['decision']['type'] == 'choice')),
            'near_duplicate_pairs': near, 'prior_examples_checked': len(previous),
            'cross_dataset_near_duplicates': 0}


def generator_usage(cache):
    # Explicit published GPT-5.6 Sol rates, including reasoning output tokens.
    # Count uncached input rates conservatively; deduplicate saved response IDs.
    seen, inputs, outputs = set(), 0, 0
    for path in cache.rglob('responses_*.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            response_id = (row.get('raw_response') or {}).get('id')
            if response_id and response_id in seen:
                continue
            seen.add(response_id or str(path) + str(len(seen)))
            usage = row.get('token_usage') or {}
            inputs += usage.get('input', 0) or 0
            outputs += usage.get('output', 0) or 0
    return {'responses': len(seen), 'input_tokens': inputs, 'output_tokens': outputs,
            'estimated_usd': (inputs * 4 + outputs * 20) / 1_000_000,
            'input_usd_per_million': 4, 'output_usd_per_million': 20}


class BudgetedCurator:
    def __init__(self, cache, output, cap=60):
        self.cache, self.output, self.cap = cache, output, cap

    def __call__(self, generator, rows, working_dir):
        combined = []
        for start in range(0, len(rows), 25):
            batch = rows[start:start + 25]
            usage = generator_usage(self.cache)
            # Worst-case 8192 completion tokens, 12000 input tokens, 3 attempts.
            # The prompt+schema byte check is a conservative bound on input tokens.
            schema = json.dumps(generator.response_format.model_json_schema())
            if any(len((generator.prompt(row) + schema).encode()) > 11500 for row in batch):
                raise ValueError('Input exceeds the generation budget reservation')
            reserved = len(batch) * 3 * (8192 * 20 + 12000 * 4) / 1_000_000
            if usage['estimated_usd'] + reserved + 1 > self.cap:
                raise RuntimeError('Generation cost cap reached; saved progress is resumable')
            result = generator(batch, working_dir=working_dir)
            combined.extend(dict(row) for row in result.dataset)
            usage = generator_usage(self.cache)
            (self.output / 'generation_usage.json').write_text(json.dumps(usage, indent=2) + '\n')
            print(f"Generation batch {start + len(batch)}/{len(rows)}; estimated API cost ${usage['estimated_usd']:.2f}", flush=True)
        return SimpleNamespace(dataset=combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/typesafe_eval_1000_gpt56')
    parser.add_argument('--previous', type=Path, default=PROJECT_ROOT / 'data/typesafe_diverse_300_gpt56/all.jsonl')
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--generate-only', action='store_true')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env', override=False)
    configure()
    args.output.mkdir(parents=True, exist_ok=True)
    previous = read_jsonl(args.previous)
    settings = {'count': COUNT, 'seed': args.seed, 'model': GENERATOR_MODEL,
                'previous_sha256': hashlib.sha256(args.previous.read_bytes()).hexdigest(),
                'plan_version': 'fresh-eval-v1', 'generator_cap_usd': 60}
    settings_path = args.output / 'generation_settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Existing run settings differ')
    settings_path.write_text(json.dumps(settings, indent=2) + '\n')
    auditor = lambda rows: audit_eval(rows, previous)
    cache = PROJECT_ROOT / '.cache/curator-eval-1000'
    source_path = args.output / 'sources.jsonl'
    if source_path.exists():
        records = read_jsonl(source_path)
        metadata = json.loads((args.output / 'source_manifest.json').read_text())
        if hashlib.sha256(canonical(records).encode()).hexdigest() != metadata['source_sha256']:
            raise ValueError('Source checksum mismatch')
        auditor(records)
    else:
        exclusions = defaultdict(set)
        for row in previous:
            exclusions[row['domain']].add(row['subtopic'])
        records = generate_sources(args.output, GENERATOR_MODEL, args.seed,
                                   plan_builder=eval_plans, auditor=auditor,
                                   id_prefix=f'fresh-eval-{args.seed}', cache=cache,
                                   runner=BudgetedCurator(cache, args.output),
                                   backend_overrides={'max_concurrent_requests': 25},
                                   exclusions={k: sorted(v) for k, v in exclusions.items()})
    if not args.generate_only:
        records = annotate(records, 'jev-1.13.0', os.environ['TYPESAFE_API_KEY'])
        manifest = export(records, args.output, GENERATOR_MODEL, 'jev-1.13.0', args.seed, auditor=auditor)
        manifest['notes'] = [n for n in manifest['notes'] if 'six examples' not in n]
        manifest['notes'] += ['All 1,000 fresh examples are reserved for evaluation; no training split.',
                              'Fifty fresh subsubjects; twenty examples per group.',
                              'Prior 300 examples checked for exact and Jaccard >=0.8 state overlap.']
        manifest['generator_usage'] = generator_usage(cache)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
