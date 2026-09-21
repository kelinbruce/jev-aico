"""Stream smaller-model training contrasts while retaining the v3 acceptance gates.

Reuses previously audited TRAINING rule plans. Evaluation generation remains a
separate Sol workflow. No previous generated examples become evaluation data.
"""
import argparse
import asyncio
import difflib
import json
import os
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pydantic import ValidationError

from nimble.paths import PROJECT_ROOT
from nimble.datasets.create_diverse_dataset import configure, read_jsonl
from nimble.datasets.dataset_io import canonical, write_jsonl
from nimble.datasets.curation_profiles import (
    TRAINING_MODEL, TRAINING_DRAFT_REASONING, TRAINING_CHECK_REASONING, EVALUATION_MODEL,
)
from nimble.datasets.contrastive_data import fingerprint
from nimble.datasets.scaled_evidence import validate_plan, plan_passes, quotas, audit_release
from nimble.datasets.scale_evidence_dataset import construct_batch, scoring_exports, atomic_json
from nimble.datasets.curation_serialization import wrap_paragraph_for_object


class DraftRejected(ValueError):
    """A mechanical failure is a rejected candidate, not a semantic retry."""
    def __init__(self, message, stage='minimal_edit'):
        super().__init__(message)
        self.stage = stage


def curation_executor(group_concurrency):
    # Each synchronous group blocks on async API work. DNS resolution also uses
    # the event loop's default executor, so leave workers free for network setup.
    return ThreadPoolExecutor(max_workers=group_concurrency + 4)


def resume_policy(args):
    path = getattr(args, 'resume_policy', None)
    if path is None:
        return {}
    policy = json.loads(path.read_text())
    if policy.get('version') != 1:
        raise ValueError('Unsupported curation resume policy')
    for name in ('bucket_start_ordinals', 'source_guidance_start_variations'):
        if not isinstance(policy.get(name), dict) or any(
                not isinstance(k, str) or type(v) is not int or v < 1
                for k, v in policy[name].items()):
            raise ValueError('Invalid curation resume thresholds')
    return policy


def select_source(pool, bucket, ordinal, attempted, selected, families, source_map, policy):
    def original(s):
        return (attempted[s], families[source_map[s]['family']], selected[s], s)
    start = policy.get('bucket_start_ordinals', {}).get('/'.join(bucket))
    # Preserve the old sequence up to the recorded checkpoint. Subsequently use
    # a smoothed acceptance rate, with every fifth proposal reserved for exploration.
    if start is None or ordinal < start or (ordinal - start) % 5 == 4:
        return min(pool, key=original)
    window = policy.get('acceptance_rate_windows', {})
    window_start = window.get('start_ordinals', {}).get('/'.join(bucket))
    def rate(s):
        offset = (window.get('sources', {}).get(s, {})
                  if window_start is not None and ordinal >= window_start else {})
        trials = attempted[s] - offset.get('attempted_before', 0)
        successes = selected[s] - offset.get('accepted_before', 0)
        if trials < 0 or successes < 0 or successes > trials:
            raise ValueError('Acceptance-rate window is inconsistent with the recorded history')
        return (successes + 1) / (trials + 4)
    return min(pool, key=lambda s: (
        -rate(s),
        families[source_map[s]['family']], selected[s], attempted[s], s))


def normalize_document_serialization(document, request):
    """Encode a literal paragraph without changing its contents or evidence paths."""
    original = json.loads(request['payload_json'])['original_input']['state']
    raw = document['base_state_json']
    if not isinstance(original, str):
        return document
    try:
        json.loads(raw)
        return document
    except json.JSONDecodeError:
        pass
    # Do not guess at malformed objects, arrays, quoted JSON strings or fences.
    if not raw.strip() or raw.lstrip().startswith(('{', '[', '"', '```')):
        return document
    spans = document['focus_evidence']
    if (len(spans) != 2 or any(s['path'] != [] or not s['text']
                              or raw.count(s['text']) != 1 for s in spans)):
        return document
    return {**document, 'base_state_json': json.dumps(raw, ensure_ascii=False)}


def changed_words(pair):
    sides = [k for k in ('left', 'right') if pair[k] != pair['negative_' + k]]
    if len(sides) != 1:
        raise DraftRejected('Exactly one sentence must change')
    a, b = pair[sides[0]].split(), pair['negative_' + sides[0]].split()
    edits = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != 'equal']
    return max(sum(j-i for _, i, j, _, _ in edits), sum(l-k for _, _, _, k, l in edits))


def load_training_plans(directory, sources):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['source_sha256'] != fingerprint(sources):
        raise ValueError('Audited plan source fingerprint differs')
    source_map = {r['id']: r for r in sources}
    rows = read_jsonl(directory / 'accepted_plans.jsonl')
    if (directory / 'refined_plans.jsonl').exists():
        rows += read_jsonl(directory / 'refined_plans.jsonl')
    plans = {}
    for row in rows:
        source = source_map[row['source_id']]
        validate_plan(row['plan'], source)  # includes the train-only guard
        if not plan_passes(row['audit'], row['plan']):
            raise ValueError('Unaccepted rule plan')
        plans[row['source_id']] = row
    return plans, {'directory': str(directory.resolve()), 'model': manifest['model'],
                   'source_sha256': manifest['source_sha256'], 'plans_sha256': fingerprint(plans)}


class RequestBudget:
    """Rolling RPM and conservative token reservations shared across all stages."""
    def __init__(self, rpm, tpm):
        if min(rpm, tpm) <= 0:
            raise ValueError('Request/token limits must be positive')
        self.rpm, self.tpm = rpm, tpm
        self.events = deque()
        self.lock = asyncio.Lock()

    async def reserve(self, tokens):
        if tokens > self.tpm:
            raise ValueError('A single request exceeds the configured token budget')
        while True:
            async with self.lock:
                now = time.monotonic()
                while self.events and self.events[0][0] <= now - 60:
                    self.events.popleft()
                if len(self.events) < self.rpm and sum(e[1] for e in self.events) + tokens <= self.tpm:
                    event = [now, tokens]
                    self.events.append(event)
                    return event
                delay = max(.01, 60 - (now - self.events[0][0]))
            await asyncio.sleep(min(delay, 1))


class AsyncStages:
    def __init__(self, output, args, client=None):
        self.output, self.args, self.client = output, args, client
        self.cache = output / 'requests'
        self.semaphore = asyncio.Semaphore(args.concurrency)
        self.budget = RequestBudget(args.requests_per_minute, args.tokens_per_minute)
        self.inflight = {}
        self.api_calls = 0
        self.cache_hits = 0
        self.usage = Counter()
        self.resume_policy = resume_policy(args)

    def request_spec(self, cls, schema, row):
        drafting = cls.__name__ in ('PairGenerator', 'DocumentGenerator')
        # These existing prompt methods only read the supplied row. Do not start
        # another Curator client or its event loop just to render a prompt.
        prompt = object.__new__(cls).prompt(row)
        if cls.__name__ == 'PairGenerator':
            prompt += (f'\nKeep the factual edit subtle: change at most {self.args.max_edit_words} '
                       'whitespace-separated words; preserve the rest of the sentence exactly.')
        if not drafting:
            prompt += '\nKeep each reason to one short sentence; retain all required exact evidence quotes.'
        threshold = getattr(self.args, 'guidance_from_variation', 0)
        gid = row.get('job_id', '')
        prefix = f'fast-{self.args.seed}-'
        sid = None
        if gid.startswith(prefix):
            sid = gid.removeprefix(prefix).rsplit('-', 1)[0]
            source_threshold = self.resume_policy.get('source_guidance_start_variations', {}).get(sid)
            if source_threshold is not None:
                threshold = min(threshold, source_threshold) if threshold else source_threshold
        if drafting and threshold:
            payload = json.loads(row['payload_json'])
            if payload.get('variation', {}).get('ordinal', 0) >= threshold:
                if cls.__name__ == 'PairGenerator':
                    prompt += (
                        '\nFINAL CONSTRUCTION CHECK: Prefer a concrete identity/record join: one sentence '
                        'identifies which record belongs to the subject, and the other reports the relevant '
                        'observation for that record. Change only the observation or its binding. Test the '
                        'changed sentence IN ISOLATION: without the other sentence it must neither prove '
                        'nor disprove the proposition. A direct denial, absence statement, or zero value '
                        'about the named subject often fails this requirement. Use a different construction '
                        'if necessary; never change the proposition, policy, or background assignments.')
                else:
                    prompt += (
                        '\nFINAL OUTPUT CHECK: base_state_json must contain the actual new case narrative '
                        'or observed records, NOT the supplied atom-to-state assignment table. Write at '
                        'least 100 words of natural context; supported/refuted/unknown and atom IDs are '
                        'planning metadata, never substitute them for observations. Embed both verified '
                        'sentences exactly once. Encode the entire context as JSON matching the original '
                        'state kind, and point each evidence path to the string that actually contains it. '
                        'For a paragraph use root paths [] and a JSON-encoded string. Check this on the '
                        'actual context you return, preserving all policies and fact assignments.')
        exemplar = self.resume_policy.get('construction_examples', {}).get(sid)
        if drafting and exemplar:
            payload = json.loads(row['payload_json'])
            if payload.get('variation', {}).get('ordinal', 0) >= exemplar['start_variation']:
                example = {'verified_pair': exemplar['verified_pair']}
                if cls.__name__ == 'DocumentGenerator':
                    example['base_context'] = exemplar['base_context']
                prompt += ('\nVERIFIED TRAINING CONSTRUCTION EXAMPLE (illustration only, not evidence '
                           'for the new case):\n' + canonical(example) +
                           '\nCreate a NEW case with different incidental records, observations, '
                           'quantities where permitted, and natural wording. Preserve entities bound '
                           'by the proposition/question, the exact policy, and required fact assignments. '
                           'Use the example to understand how the two-sentence dependency and all '
                           'other facts coexist; do not copy its sentences or merely rename record IDs. '
                           'The new case must independently pass every check. For a document, the '
                           'newly supplied verified_pair takes precedence over the example pair.')
        return {'model': self.args.generator_model if drafting else self.args.verifier_model,
                'reasoning_effort': self.args.draft_reasoning if drafting else self.args.check_reasoning,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_completion_tokens': 8192,
                'response_format': {'type': 'json_schema', 'json_schema': {
                    'name': schema.__name__, 'strict': True, 'schema': schema.model_json_schema()}}}

    async def one(self, cls, schema, row):
        spec = self.request_spec(cls, schema, row)
        key = fingerprint(spec)
        path = self.cache / f'{key}.json'
        rejected_path = self.output / 'rejected_requests' / f'{key}.json'
        if rejected_path.exists():
            saved = json.loads(rejected_path.read_text())
            if (path.exists() or saved['request_sha256'] != key
                    or saved['rejection_sha256'] != fingerprint(saved['rejection'])):
                raise ValueError('Cached rejection fingerprint differs or conflicts with a response')
            self.cache_hits += 1
            raise DraftRejected(saved['rejection']['reason'], stage='model_response')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['request_sha256'] != key or saved['response_sha256'] != fingerprint(saved['response']):
                raise ValueError('Cached request/response fingerprint differs')
            schema.model_validate(saved['response'])
            self.cache_hits += 1
            return saved['response']
        if self.args.offline:
            raise ValueError('Offline replay is missing a response: ' + key)
        if key not in self.inflight:
            self.inflight[key] = asyncio.create_task(self.fetch(spec, schema, path, key))
        return await self.inflight[key]

    async def fetch(self, spec, schema, path, key):
        async with self.semaphore:
            # UTF-8 bytes conservatively bound input tokens, including the schema.
            reservation = await self.budget.reserve(len(canonical(spec).encode()) + spec['max_completion_tokens'])
            start = time.monotonic()
            result = await self.client.chat.completions.create(**spec)
            usage = result.usage.model_dump() if result.usage else {}
            reservation[1] = usage.get('total_tokens', reservation[1])
            self.api_calls += 1
            self.usage.update({k: usage.get(k, 0) for k in ('prompt_tokens', 'completion_tokens')})
            choice = result.choices[0]
            reason = None
            if choice.finish_reason != 'stop' or choice.message.refusal:
                reason = f'Incomplete/refused response ({choice.finish_reason}); candidate rejected'
            else:
                try:
                    response = schema.model_validate_json(choice.message.content).model_dump()
                except ValidationError:
                    reason = 'Structured response does not match ' + schema.__name__
            if reason:
                rejection = {'reason': reason, 'model_returned': result.model,
                             'finish_reason': choice.finish_reason,
                             'refused': bool(choice.message.refusal)}
                rejected_path = self.output / 'rejected_requests' / f'{key}.json'
                rejected_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(rejected_path, {'request_sha256': key, 'rejection': rejection,
                    'rejection_sha256': fingerprint(rejection), 'model_requested': spec['model'],
                    'usage': usage, 'latency_seconds': time.monotonic()-start})
                raise DraftRejected(reason, stage='model_response')
            saved = {'request_sha256': key, 'response_sha256': fingerprint(response),
                     'model_requested': spec['model'], 'model_returned': result.model,
                     'reasoning_effort': spec['reasoning_effort'], 'response': response,
                     'usage': usage, 'latency_seconds': time.monotonic() - start}
            self.cache.mkdir(parents=True, exist_ok=True)
            atomic_json(path, saved)
            return response

    async def run(self, cls, schema, requests):
        # Drain sibling requests before rejecting the group, and choose failures
        # in request order so offline reconstruction remains deterministic.
        responses = await asyncio.gather(*(self.one(cls, schema, r) for r in requests), return_exceptions=True)
        for response in responses:
            if isinstance(response, BaseException):
                raise response
        if cls.__name__ == 'DocumentGenerator':
            responses = [normalize_document_serialization(response, request)
                         for response, request in zip(responses, requests)]
            recovered = []
            for response, request in zip(responses, requests):
                gid = request.get('job_id', '')
                prefix = f'fast-{self.args.seed}-'
                sid = gid.removeprefix(prefix).rsplit('-', 1)[0] if gid.startswith(prefix) else None
                start = self.resume_policy.get('object_wrapper_start_variations', {}).get(sid)
                ordinal = json.loads(request['payload_json']).get('variation', {}).get('ordinal', 0)
                if start is not None and ordinal >= start:
                    response = wrap_paragraph_for_object(response, request)
                recovered.append(response)
            responses = recovered
        if cls.__name__ == 'PairGenerator':
            for pair in responses:
                count = changed_words(pair)
                if not 1 <= count <= self.args.max_edit_words:
                    raise DraftRejected(f'Edit changes {count} words; limit is {self.args.max_edit_words}')
        return {r['job_id']: response for r, response in zip(requests, responses)}

    def bridge(self, loop):
        # Each group advances independently; a slow request only holds its group.
        def stage(cls, schema, requests, name):
            return asyncio.run_coroutine_threadsafe(self.run(cls, schema, requests), loop).result()
        return stage


async def curate(args, sources, plans, config, stage):
    limits = quotas(sources, args.target)
    source_map = {r['id']: r for r in sources}
    pools = defaultdict(list)
    for sid in sorted(plans):
        s = source_map[sid]
        pools[(s['domain'], s['input']['questions']['decision']['type'])].append(sid)
    if any(n and not pools[b] for b, n in limits.items()):
        raise ValueError('A required category has no audited training plans')
    completed = {}
    failures = {}
    attempts = Counter()
    loop = asyncio.get_running_loop()
    bridge = stage.bridge(loop)
    policy = resume_policy(args)
    workers = asyncio.Semaphore(args.group_concurrency)

    async def bucket_run(bucket, quota):
        if not quota:
            return
        accepted, review = [], []
        used_inputs = set()
        attempted, selected, families = Counter(), Counter(), Counter()
        for ordinal in range(1, quota * args.attempt_multiplier + 1):
            if len(accepted) == quota * 2:
                break
            sid = select_source(pools[bucket], bucket, ordinal, attempted, selected, families, source_map, policy)
            attempted[sid] += 1
            gid = f'fast-{args.seed}-{sid}-{attempted[sid]:03d}'
            job = {'id': gid, 'source_id': sid, 'variation': {'ordinal': attempted[sid],
                   'seed': args.seed + ordinal * 997, 'style': 'concise natural case note'}}
            async with workers:
                try:
                    # Index 5 selects the complete-input policy reviewer. All v3
                    # rule, pair, context, full-fact and deletion gates stay intact.
                    groups, rejected = await asyncio.to_thread(construct_batch, [job], plans, source_map, bridge, 5)
                except DraftRejected as error:
                    groups, rejected = [], [{'group_id': gid, 'stage': error.stage, 'reason': str(error)}]
            review.extend(rejected)
            for group in groups:
                hashes = {fingerprint(r['input']) for r in group['rows']}
                if hashes & used_inputs:
                    review.append({'group_id': gid, 'stage': 'selection', 'reason': 'duplicate_input'})
                    continue
                accepted.extend(group['rows'])
                used_inputs.update(hashes)
                selected[sid] += 1
                families[source_map[sid]['family']] += 1
            completed[bucket], failures[bucket], attempts[bucket] = accepted, review, ordinal
            if not args.offline:
                rows = sorted((r for group in completed.values() for r in group), key=lambda r: r['id'])
                write_jsonl(args.output / 'accepted_progress.jsonl', rows)
                write_jsonl(args.output / 'review_queue.jsonl', [r for b in sorted(failures) for r in failures[b]])
                atomic_json(args.output / 'progress.json', {'accepted_examples': len(rows), 'target': args.target,
                            'attempted_groups': sum(attempts.values()), 'api_calls_this_run': stage.api_calls})
            print(f'{bucket[0]}/{bucket[1]}: {len(accepted)//2}/{quota} pairs; '
                  f'total {sum(len(v) for v in completed.values())}/{args.target} rows', flush=True)
        if len(accepted) != quota * 2:
            raise ValueError(f'Bounded shortfall in {bucket}: {len(accepted)}/{quota*2}; inspect review queue')

    await asyncio.gather(*(bucket_run(b, n) for b, n in limits.items()))
    rows = sorted((r for group in completed.values() for r in group), key=lambda r: r['id'])
    manifest = {**config, 'pipeline_version': 'evidence-curation-v3',
                'training': audit_release(rows, sources), 'train_sha256': fingerprint(rows),
                'attempted_groups': sum(attempts.values()), 'held_out_overlap': False,
                'reference_human_reviewed': False,
                'notes': ['Training-only sources; previously audited rules reused with recorded model provenance.',
                          'All v3 semantic gates retained; small model generates and verifies in separate calls.',
                          'Deletion variants test necessity; they are not automatically labeled false.',
                          'No measured quality equivalence or speedup claim; human review remains pending.']}
    scoring = scoring_exports(rows)
    if args.offline:
        if (rows != read_jsonl(args.output / 'train.jsonl') or scoring != read_jsonl(args.output / 'train_scoring.jsonl')
                or manifest != json.loads((args.output / 'manifest.json').read_text())):
            raise ValueError('Offline release reconstruction differs')
    else:
        write_jsonl(args.output / 'train.jsonl', rows)
        write_jsonl(args.output / 'train_scoring.jsonl', scoring)
        atomic_json(args.output / 'manifest.json', manifest)
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=PROJECT_ROOT/'data/typesafe_diverse_300_gpt56/all.jsonl')
    p.add_argument('--plans', type=Path, default=PROJECT_ROOT/'data/evidence_curated_1000_gpt56')
    p.add_argument('--output', type=Path, default=PROJECT_ROOT/'data/contrastive_training_terra')
    p.add_argument('--target', type=int, default=1000)
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--generator-model', default=TRAINING_MODEL)
    p.add_argument('--verifier-model', default=TRAINING_MODEL)
    p.add_argument('--draft-reasoning', default=TRAINING_DRAFT_REASONING, choices=['none','low','medium','high'])
    p.add_argument('--check-reasoning', default=TRAINING_CHECK_REASONING, choices=['none','low','medium','high'])
    p.add_argument('--max-edit-words', type=int, default=8)
    p.add_argument('--attempt-multiplier', type=int, default=8)
    p.add_argument('--guidance-from-variation', type=int, default=0,
                   help='Add construction reminders from this per-source variation onward; 0 disables')
    p.add_argument('--resume-policy', type=Path,
                   help='Recorded thresholds for adaptive source selection and future generation guidance')
    p.add_argument('--concurrency', type=int, default=64)
    p.add_argument('--group-concurrency', type=int, default=16)
    p.add_argument('--requests-per-minute', type=int, default=240)
    p.add_argument('--tokens-per-minute', type=int, default=1_000_000)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if min(args.max_edit_words, args.attempt_multiplier, args.concurrency, args.group_concurrency,
           args.requests_per_minute, args.tokens_per_minute) <= 0:
        raise ValueError('All limits must be positive')
    sources = read_jsonl(args.source)
    quotas(sources, args.target)
    plans, lineage = load_training_plans(args.plans, sources)
    code = [Path(__file__), Path(__file__).with_name('curation_profiles.py'),
            Path(__file__).with_name('curation_serialization.py')]
    code += [Path(__file__).with_name(n) for n in ('scaled_evidence.py','scale_evidence_dataset.py',
             'scaled_evidence_stages.py','evidence_stages.py','curate_paired_evidence.py','evidence_curation.py','contrastive_data.py')]
    config = {'runner_version': 'streaming-training-v1', 'source': str(args.source.resolve()),
              'source_sha256': fingerprint(sources), 'rule_plan_lineage': lineage,
              'generator_model': args.generator_model, 'verifier_model': args.verifier_model,
              'draft_reasoning': args.draft_reasoning, 'check_reasoning': args.check_reasoning,
              'target': args.target, 'seed': args.seed, 'max_edit_words': args.max_edit_words,
              'attempt_multiplier': args.attempt_multiplier,
              'implementation_sha256': fingerprint({p.name: p.read_text() for p in code})}
    if args.guidance_from_variation:
        config['guidance_from_variation'] = args.guidance_from_variation
    if args.resume_policy:
        config['resume_policy'] = resume_policy(args)
    if args.dry_run:
        print(json.dumps({**config, 'audited_training_plans': len(plans),
              'evaluation_generator_unchanged': EVALUATION_MODEL, 'api_calls': 0}, indent=2))
        return
    config_path = args.output/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing run configuration differs; use a new output directory')
    if not args.offline:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, config)
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT/'.env', override=False)
        configure()
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY is required')

    async def run():
        from openai import AsyncOpenAI
        client = None if args.offline else AsyncOpenAI(base_url='https://api.openai.com/v1', max_retries=2, timeout=180)
        stage = AsyncStages(args.output, args, client)
        start = time.monotonic()
        try:
            return await curate(args, sources, plans, config, stage)
        finally:
            if not args.offline:
                atomic_json(args.output/'performance.json', {'elapsed_seconds': time.monotonic()-start,
                    'api_calls_this_run': stage.api_calls, 'cache_hits_this_run': stage.cache_hits,
                    'usage_this_run': dict(stage.usage), 'concurrency': args.concurrency,
                    'group_concurrency': args.group_concurrency})
            if client:
                await client.close()

    with asyncio.Runner() as runner:
        runner.get_loop().set_default_executor(curation_executor(args.group_concurrency))
        result = runner.run(run())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
