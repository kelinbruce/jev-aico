#!/usr/bin/env python3
"""Extract recorded executor requests and benchmark Jev skill routing (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'data/4_0921/qwen3.6_modify_system_prompt_no_lora_delet_router.jsonl'
API_URL = 'https://api.typesafe.ai/v1/systemone'
NO_SKILL = '__no_skill__'


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, values):
    with Path(path).open('w', encoding='utf-8') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def extract_question(messages):
    for index in range(len(messages) - 1, -1, -1):
        content = messages[index].get('content')
        if messages[index].get('role') != 'user' or not isinstance(content, str):
            continue
        match = re.search(r'<task>(.*?)</task>', content, re.S)
        if match:
            question = match.group(1).strip()
            try:
                decoded = json.loads(question)
                if isinstance(decoded, str):
                    question = decoded
            except json.JSONDecodeError:
                pass
            marker = '当前需要执行的计划步骤：'
            plan = content.split(marker, 1)[1].strip() if marker in content else ''
            return question, plan, index
    raise ValueError('No user <task> found; refusing to silently substitute an entire prompt')


def baseline_label(row):
    response = row.get('response_body')
    if isinstance(response, str):
        response = json.loads(response)
    if row.get('status_code') != 200 or not isinstance(response, dict):
        return None
    choices = response.get('choices') or []
    if not choices:
        return None
    message = choices[0].get('message') or {}
    calls = message.get('tool_calls') or []
    if not calls and message.get('function_call'):
        calls = [{'function': message['function_call']}]
    names = {call['function']['name'] for call in calls}
    if len(names) > 1:
        raise ValueError('Multiple different skills in one response: single-choice benchmark unsupported')
    if names:
        return names.pop()
    return NO_SKILL if message.get('content') else None


def convert(row, line, mode='compact', model='jev-latest'):
    body = row['request_body']
    messages = body['messages']
    question, plan, index = extract_question(messages)
    criteria = {tool['function']['name']: tool['function'].get('description', '')
                for tool in body['tools'] if tool.get('type') == 'function'}
    if not criteria or NO_SKILL in criteria:
        raise ValueError('Missing tools or reserved skill name')
    criteria[NO_SKILL] = '任务已完成、工具结果已足够、无可执行的后续步骤，或需要向用户说明失败/澄清时，不调用工具。'
    if mode == 'full':
        state = {'messages': messages}
    elif mode == 'question':
        state = {'question': question}
    else:
        state = {'question': question, 'current_plan': plan,
                 'history': messages[index + 1:]}
    instructions = ('根据当前输入选择下一步需要执行的一个 skill。历史中 assistant 的调用是已经执行过的动作，'
                    '不是本轮答案；结合 tool 返回结果判断是否还需要调用。任务完成或无法继续时选 __no_skill__。'
                    '只选择工具，不生成参数，不执行工具。')
    payload = {'model': model, 'state': state, 'questions': {
        'skill': {'type': 'choice', 'instructions': instructions, 'criteria': criteria}}}
    baseline = baseline_label(row)
    if baseline is not None and baseline not in criteria:
        raise ValueError(f'Unknown baseline skill: {baseline}')
    return {'id': row.get('proxy_request_id') or f'line-{line}', 'source_line': line,
            'question': question, 'question_group': digest(question),
            'phase': 'followup' if messages[index + 1:] else 'initial',
            'mode': mode, 'baseline_skill': baseline,
            'baseline_model': body.get('model'),
            'baseline_latency_ms': row.get('total_duration_ms'),
            'payload': payload, 'payload_sha256': digest(payload)}


def load_key(path):
    key = os.environ.get('JEV_API_KEY') or os.environ.get('TYPESAFE_API_KEY')
    if not key:
        key = Path(path).read_text(encoding='utf-8').strip()
    if not key:
        raise ValueError('Empty API key')
    return key


def request_jev(payload, key, url=API_URL, timeout=30, retries=2):
    start = time.perf_counter()
    for attempt in range(retries + 1):
        req = Request(url, data=json.dumps(payload, ensure_ascii=False).encode(),
                      headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}, method='POST')
        try:
            with urlopen(req, timeout=timeout) as response:
                raw = json.load(response)
            answer = raw['answers']['skill']
            label = answer['choice']
            if label not in payload['questions']['skill']['criteria']:
                raise ValueError(f'Unknown returned choice: {label}')
            return {'prediction': label, 'confidence': answer.get('confidence'),
                    'raw_response': raw, 'latency_ms': (time.perf_counter()-start)*1000,
                    'attempts': attempt+1, 'error': None}
        except HTTPError as exc:
            error = f'HTTP {exc.code}'  # Never persist headers, credentials or echoed request bodies.
            retryable = exc.code in (429, 500, 502, 503, 504, 529)
            try:
                delay = min(30, max(0, float(exc.headers.get('Retry-After', 2**attempt))))
            except (ValueError, TypeError):
                delay = 2**attempt
        except (URLError, TimeoutError, OSError) as exc:
            error, retryable, delay = type(exc).__name__, True, 2**attempt
        except (KeyError, TypeError, ValueError) as exc:
            error, retryable, delay = f'Invalid response ({type(exc).__name__})', False, 0
        if not retryable or attempt == retries:
            return {'prediction': None, 'error': error, 'attempts': attempt+1,
                    'latency_ms': (time.perf_counter()-start)*1000}
        time.sleep(delay)


def latency(values):
    values = sorted(v for v in values if isinstance(v, (float, int)))
    if not values:
        return None
    return {'n': len(values), 'mean_ms': statistics.mean(values),
            'p50_ms': statistics.median(values),
            'p95_ms': values[max(0, math.ceil(len(values)*.95)-1)]}


def metrics(rows, results, gold):
    success = [r for r in rows if results.get(r['id'], {}).get('prediction') is not None]
    comparable = [r for r in rows if r['baseline_skill'] is not None]
    paired = [r for r in success if r['baseline_skill'] is not None]
    labeled = [r for r in rows if r['id'] in gold]
    def pred(r):
        return results.get(r['id'], {}).get('prediction')
    def rate(items, check):
        return sum(bool(check(r)) for r in items)/len(items) if items else None
    confusion = {}
    for r in comparable:
        bucket = confusion.setdefault(r['baseline_skill'], {})
        p = pred(r) or '__error_or_missing__'
        bucket[p] = bucket.get(p, 0)+1
    return {'n': len(rows), 'successful': len(success), 'failed_or_missing': len(rows)-len(success),
            'baseline_distribution': {k: sum(r['baseline_skill'] == k for r in rows)
                                      for k in sorted({r['baseline_skill'] for r in rows if r['baseline_skill']})},
            'agreement_denominator': len(comparable),
            'agreement_all': rate(comparable, lambda r: pred(r) == r['baseline_skill']),
            'agreement_success_only': rate(paired, lambda r: pred(r) == r['baseline_skill']),
            'gold_n': len(labeled),
            'jev_accuracy': rate(labeled, lambda r: pred(r) == gold[r['id']]),
            'baseline_accuracy': rate(labeled, lambda r: r['baseline_skill'] == gold[r['id']]),
            'baseline_to_jev_confusion': confusion,
            'jev_success_latency': latency([results[r['id']]['latency_ms'] for r in success]),
            'baseline_paired_latency': latency([r['baseline_latency_ms'] for r in success])}


def report(rows, results, gold):
    return {'note': 'agreement compares recorded predictions, not ground truth. Accuracy requires gold labels. '
                    'Historical LLM latency includes argument/text generation; Jev only selects a skill.',
            'overall': metrics(rows, results, gold),
            'by_phase': {phase: metrics([r for r in rows if r['phase'] == phase], results, gold)
                         for phase in ('initial', 'followup')}}


def load_gold(path, rows):
    if not path:
        return {}
    known = {r['id']: r for r in rows}
    gold = {}
    seen = set()
    for item in read_jsonl(path):
        rid = item['id']
        if rid in seen:
            raise ValueError(f'Duplicate gold id: {rid}')
        seen.add(rid)
        # A full annotation file can be used with a selected benchmark subset.
        if rid not in known or item.get('gold_skill') is None:
            continue
        if item['gold_skill'] not in known[rid]['payload']['questions']['skill']['criteria']:
            raise ValueError(f'Invalid gold label for {rid}')
        gold[rid] = item['gold_skill']
    return gold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--out', type=Path, default=ROOT/'benchmark/output/question')
    parser.add_argument('--mode', choices=['compact', 'full', 'question'], default='question')
    parser.add_argument('--phase', choices=['all', 'initial', 'followup'], default='initial')
    parser.add_argument('--model', default='jev-latest')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--api-url', default=API_URL)
    parser.add_argument('--api-key-file', type=Path, default=ROOT/'jev_api_key.txt')
    parser.add_argument('--gold', type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0 or (args.limit is not None and args.limit < 1):
        parser.error('workers/timeout/limit must be positive; retries must be nonnegative')
    if args.mode == 'question' and args.phase != 'initial':
        parser.error('question mode requires --phase initial (no history to identify follow-up turns)')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'report':
        rows = read_jsonl(args.out/'dataset.jsonl')
    else:
        rows = [convert(r, i, args.mode, args.model) for i, r in enumerate(read_jsonl(args.input), 1)]
        rows = [r for r in rows if args.phase == 'all' or r['phase'] == args.phase]
        if args.limit and args.limit < len(rows):
            rows = sorted(random.Random(args.seed).sample(rows, args.limit), key=lambda r:r['source_line'])
        if not rows or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Empty dataset or duplicate ids')
        dataset = args.out/'dataset.jsonl'
        if dataset.exists() and read_jsonl(dataset) != rows:
            raise ValueError('Output directory belongs to a different dataset/config; use a different --out')
        write_jsonl(dataset, rows)
        template = args.out/'gold.template.jsonl'
        if not template.exists():
            write_jsonl(template, [{'id':r['id'], 'question':r['question'], 'phase':r['phase'],
                                   'gold_skill':None, 'note':''} for r in rows])
    gold = load_gold(args.gold, rows)
    results_path = args.out/'results.jsonl'
    results = {}
    known = {r['id']:r for r in rows}
    if results_path.exists():
        for result in read_jsonl(results_path):
            if result['id'] not in known or result['payload_sha256'] != known[result['id']]['payload_sha256']:
                raise ValueError('Stale result payload/id; use a new --out')
            if args.command == 'run' and result.get('api_url') != args.api_url:
                raise ValueError('Different API endpoint; use a new --out')
            results[result['id']] = result
    if args.command == 'run':
        key = load_key(args.api_key_file)
        pending = [r for r in rows if not results.get(r['id'], {}).get('prediction')]
        def execute(row):
            return dict(request_jev(row['payload'], key, args.api_url, args.timeout, args.retries),
                        id=row['id'], payload_sha256=row['payload_sha256'], api_url=args.api_url)
        with results_path.open('a', encoding='utf-8') as f, ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(execute, row) for row in pending]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results[result['id']] = result
                f.write(json.dumps(result, ensure_ascii=False)+'\n')
                f.flush()
                print(f"[{i}/{len(pending)}] {result['id']}: {result['prediction'] or result['error']}", flush=True)
    summary = report(rows, results, gold)
    dump_json(args.out/'summary.json', summary)
    write_jsonl(args.out/'disagreements.jsonl', [dict(r, result=results.get(r['id'])) for r in rows
                if r['id'] in results and r['baseline_skill'] != results[r['id']].get('prediction')])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(args.command == 'run' and summary['overall']['failed_or_missing'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
