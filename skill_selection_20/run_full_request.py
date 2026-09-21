#!/usr/bin/env python3
"""Build and send KEV skill requests from recorded messages and tools."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import math
import statistics
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT.parent / 'data/4_0921/qwen3.6_modify_system_prompt_no_lora_delet_router.jsonl'
INSTRUCTIONS = (
    '根据 state.request_body 中完整的 messages、系统规则、工具定义和历史，'
    '选择当前这一次模型响应下一步需要调用的一个工具。'
    '历史 assistant 的工具调用是已发生的动作，不是本轮答案；结合工具结果判断下一步。'
    '只返回候选工具名称，不生成参数、不执行工具。'
    '无需调用工具、任务已完成或需要向用户澄清时选择 __no_skill__。'
)


def build_payload(record, model):
    body = record.get('request_body')
    if isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list) or not body['messages']:
        raise ValueError('Expected a real request_body with nonempty messages')
    criteria = {}
    for tool in body.get('tools', []):
        if tool.get('type') != 'function':
            raise ValueError('Only function tools are supported')
        function = tool['function']
        name = function['name']
        if not isinstance(name, str) or not name or name in criteria or name == '__no_skill__':
            raise ValueError('Invalid, duplicate or reserved tool name')
        criteria[name] = function.get('description') or ''
    if not criteria:
        raise ValueError('request_body.tools has no function candidates')
    criteria['__no_skill__'] = '本轮无需调用工具、任务已完成或需要澄清。'
    # Preserve full message content, role boundaries and tool schemas only.
    # The recorded response_body is deliberately excluded: it contains the answer.
    return {'model': model, 'state': {'request_body': {name: copy.deepcopy(body[name])
                                                 for name in ('messages', 'tools')}},
            'questions': {'skill': {'type': 'choice', 'instructions': INSTRUCTIONS,
                                    'criteria': criteria}}}


def is_initial_request(record):
    """Exclude recorded follow-ups that already contain assistant/tool turns."""
    body = record.get('request_body')
    if isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list) or not body['messages']:
        raise ValueError('Expected a real request_body with nonempty messages')
    return not any(message.get('role') in ('assistant', 'tool', 'function')
                   for message in body['messages'])


def redact(text, key):
    return text.replace(key, '[REDACTED]') if key else text


def call_service(payload, base_url, key, timeout):
    start = time.perf_counter()
    req = Request(base_url.rstrip('/') + '/v1/systemone',
                  data=json.dumps(payload, ensure_ascii=False).encode(),
                  headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key},
                  method='POST')
    result = {'prediction': None, 'http_status': None, 'error': None}
    try:
        with urlopen(req, timeout=timeout) as response:
            result['http_status'] = response.status
            result['request_id'] = response.headers.get('x-typesafe-request-id')
            raw = json.load(response)
        result['raw_response'] = raw
        answer = raw['answers']['skill']
        label = answer['choice']
        if label not in payload['questions']['skill']['criteria']:
            raise ValueError('Response returned an unknown skill')
        result.update(prediction=label, confidence=answer.get('confidence'))
    except HTTPError as exc:
        result.update(error='HTTPError', http_status=exc.code,
                      request_id=exc.headers.get('x-typesafe-request-id'),
                      error_detail=redact(exc.read(8192).decode('utf-8', errors='replace'), key))
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        result.update(error=type(exc).__name__, error_detail=redact(str(exc), key))
    result['latency_ms'] = (time.perf_counter() - start) * 1000
    return result



def baseline_skill(record):
    response = record.get('response_body')
    if isinstance(response, str):
        response = json.loads(response)
    if record.get('status_code') != 200 or not isinstance(response, dict):
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
        return None  # A multi-tool answer is not a single-choice reference.
    return next(iter(names)) if names else ('__no_skill__' if message.get('content') else None)


def latency_stats(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0)
    if not values:
        return None
    return {'n': len(values), 'mean_ms': statistics.mean(values),
            'p50_ms': statistics.median(values),
            'p95_ms': values[math.ceil(len(values) * .95) - 1]}


def read_jsonl(path):
    with path.open(encoding='utf-8') as src:
        return [json.loads(line) for line in src if line.strip()]


def make_report(out, source_path):
    """Report saved batches, including older outputs without baseline metadata."""
    sources = read_jsonl(out/'sources.jsonl')
    payloads = read_jsonl(out/'requests.jsonl')
    if not sources or len(sources) != len(payloads):
        raise ValueError('Saved sources and requests do not match')
    # Old batches did not retain labels; recover only after verifying source identity/payload.
    originals = None
    if any('baseline_skill' not in meta for meta in sources):
        with source_path.open(encoding='utf-8') as src:
            originals = {n: json.loads(line) for n, line in enumerate(src, 1) if line.strip()}
    known = {}
    for meta, payload in zip(sources, payloads):
        encoded = json.dumps(payload, ensure_ascii=False)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        if fingerprint != meta['payload_sha256'] or meta['source_line'] in known:
            raise ValueError('Saved payload mismatch or duplicate source line')
        if 'baseline_skill' not in meta:
            original = originals[meta['source_line']]
            if original.get('proxy_request_id') != meta['id'] or build_payload(original, payload['model']) != payload:
                raise ValueError('Original log does not match saved batch')
            meta['baseline_skill'] = baseline_skill(original)
        known[meta['source_line']] = (meta, payload)
    results = {}
    path = out/'results.jsonl'
    for result in read_jsonl(path) if path.exists() else []:
        key = result['source_line']
        if key not in known or result['payload_sha256'] != known[key][0]['payload_sha256'] or result['id'] != known[key][0]['id']:
            raise ValueError('Result does not match saved request')
        results[key] = result
    successful = matched = comparable = matched_success = comparable_success = 0
    confusion, errors, differences, success_times, attempt_times = {}, {}, [], [], []
    for line, (meta, payload) in known.items():
        result = results.get(line)
        prediction = result.get('prediction') if result else None
        ok = bool(result and result.get('error') is None and prediction in payload['questions']['skill']['criteria'])
        successful += ok
        if result:
            attempt_times.append(result.get('latency_ms'))
        if ok:
            success_times.append(result.get('latency_ms'))
        else:
            error = (result.get('error') or 'InvalidPrediction') if result else 'MissingResult'
            errors[error] = errors.get(error, 0) + 1
        baseline = meta['baseline_skill']
        if baseline is not None:
            comparable += 1
            comparable_success += ok
            match = ok and prediction == baseline
            matched += match
            matched_success += match
            bucket = confusion.setdefault(baseline, {})
            label = prediction if ok else '__error_or_missing__'
            bucket[label] = bucket.get(label, 0) + 1
            if not match:
                differences.append(dict(meta, result=result))
        elif not ok:
            differences.append(dict(meta, result=result))
    summary = {
        'total': len(sources), 'attempted': len(results), 'successful': successful,
        'failed': len(results) - successful, 'missing': len(sources) - len(results),
        'success_rate': successful / len(sources),
        'baseline_comparable': comparable, 'matched_baseline': matched,
        'agreement_all': matched / comparable if comparable else None,
        'agreement_success_only': matched_success / comparable_success if comparable_success else None,
        'accuracy': None, 'baseline_to_local_confusion': confusion, 'errors': errors,
        'success_latency': latency_stats(success_times), 'attempt_latency': latency_stats(attempt_times),
        'note': 'Agreement is not accuracy. agreement_all includes failed/missing requests with known baseline; '
                'agreement_success_only includes successful requests with known baseline. Latency is HTTP call wall time.'}
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/'disagreements.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in differences), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--out', type=Path, default=ROOT/'output_full_request')
    parser.add_argument('--model', default='kev-latest')
    parser.add_argument('--base-url', default='http://71.77.153.223:55332')
    parser.add_argument('--api-key-env', default='LOCAL_MODEL_API_KEY')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--start-line', type=int, default=1, help='1-based source line')
    parser.add_argument('--limit', type=int, default=0, help='0 selects all remaining initial skill-selection requests')
    args = parser.parse_args(argv)
    if args.start_line < 1 or args.limit < 0 or args.timeout <= 0:
        parser.error('start-line/timeout must be positive; limit must be nonnegative')
    if args.base_url.rstrip('/').endswith('/v1/systemone'):
        parser.error('--base-url must be the service root, without /v1/systemone')
    if args.command == 'report':
        make_report(args.out, args.input)
        return 0
    items = []
    with args.input.open(encoding='utf-8') as src:
        for line_number, line in enumerate(src, 1):
            if line_number < args.start_line or not line.strip():
                continue
            record = json.loads(line)
            if not is_initial_request(record):
                continue
            payload = build_payload(record, args.model)
            encoded = json.dumps(payload, ensure_ascii=False)
            items.append(({'source_line': line_number, 'id': record.get('proxy_request_id'),
                           'payload_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
                           'payload_bytes': len(encoded.encode()),
                           'baseline_skill': baseline_skill(record)}, payload))
            if args.limit and len(items) >= args.limit:
                break
    if not items:
        parser.error('No records selected')
    args.out.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting results or a prepared batch; use a fresh output directory.
    if any((args.out/name).exists() for name in ('requests.jsonl', 'results.jsonl')):
        parser.error('Output already contains a batch; choose a new --out directory')
    def write(name, rows):
        (args.out/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    write('requests.jsonl', [payload for _, payload in items])
    write('sources.jsonl', [metadata for metadata, _ in items])
    (args.out/'manifest.json').write_text(json.dumps({
        'input': str(args.input.resolve()), 'base_url': args.base_url, 'model': args.model,
        'selection': 'initial requests only (no assistant/tool/function history)',
        'count': len(items), 'workers': 1, 'retries': 0,
        'state': 'original request_body messages and tools only',
    }, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Prepared {len(items)} complete requests: {args.out / "requests.jsonl"}', flush=True)
    if args.command == 'prepare':
        return 0
    key = os.environ.get(args.api_key_env) or 'local'
    failures = 0
    with (args.out/'results.jsonl').open('w', encoding='utf-8') as dest:
        for index, (metadata, payload) in enumerate(items, 1):
            result = dict(metadata, **call_service(payload, args.base_url, key, args.timeout))
            failures += result['error'] is not None
            dest.write(json.dumps(result, ensure_ascii=False)+'\n')
            dest.flush()
            print(f'[{index}/{len(items)}] line={metadata["source_line"]} '
                  f'HTTP={result["http_status"]} {result["prediction"] or result["error"]}', flush=True)
    make_report(args.out, args.input)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
