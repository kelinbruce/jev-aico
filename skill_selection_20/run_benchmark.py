#!/usr/bin/env python3
"""Portable 20-question skill selection benchmark; Python 3.10+, TypeSafe SDK."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent
API_URL = "http://71.77.153.223:55332"

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

def request_jev(payload, key, url=API_URL):
    # Import lazily: prepare/report and offline tests do not require the SDK.
    from typesafe_sdk import Choice, TypeSafeClient
    start = time.perf_counter()
    try:
        question = payload['questions']['skill']
        with TypeSafeClient(api_key=key, base_url=url, model=payload['model']) as client:
            response = client.system_one(
                state=payload['state'],
                questions={'skill': Choice(instructions=question['instructions'],
                                           criteria=question['criteria'])})
        answer = response.choices['skill']
        label = answer.choice
        if label not in question['criteria']:
            raise ValueError('Unknown skill returned')
        # Normalized fields also support SDK versions without model_dump().
        raw = {'model': getattr(response, 'model', payload['model']),
               'choice': label, 'confidence': getattr(answer, 'confidence', None)}
        if callable(getattr(response, 'model_dump', None)):
            raw = response.model_dump(mode='json')
        return {'prediction': label, 'confidence': getattr(answer, 'confidence', None),
                'raw_response': raw, 'latency_ms': (time.perf_counter()-start)*1000,
                'error': None}
    except Exception as exc:
        # Do not persist exception bodies that could echo credentials or headers.
        return {'prediction': None, 'error': type(exc).__name__,
                'http_status': getattr(exc, 'status_code', None),
                'latency_ms': (time.perf_counter()-start)*1000}


def latency(values):
    values = sorted(v for v in values if isinstance(v, (float, int)))
    if not values:
        return None
    return {'n': len(values), 'mean_ms': statistics.mean(values),
            'p50_ms': statistics.median(values),
            'p95_ms': values[max(0, math.ceil(len(values)*.95)-1)]}

def summarize(rows, results, gold):
    successful = [r for r in rows if results.get(r['id'], {}).get('prediction') is not None]
    labeled = [r for r in rows if r['id'] in gold]
    def prediction(row):
        return results.get(row['id'], {}).get('prediction')
    matched = sum(prediction(r) == r['baseline_skill'] for r in rows)
    confusion = {}
    for r in rows:
        bucket = confusion.setdefault(r['baseline_skill'], {})
        label = prediction(r) or '__error_or_missing__'
        bucket[label] = bucket.get(label, 0) + 1
    return {
        'total': len(rows), 'successful': len(successful),
        'failed_or_missing': len(rows)-len(successful),
        'matched_baseline': matched, 'agreement_all': matched/len(rows),
        'agreement_success_only': matched/len(successful) if successful else None,
        'gold_n': len(labeled),
        'local_accuracy': sum(prediction(r) == gold[r['id']] for r in labeled)/len(labeled) if labeled else None,
        'baseline_accuracy': sum(r['baseline_skill'] == gold[r['id']] for r in labeled)/len(labeled) if labeled else None,
        'baseline_to_local_confusion': confusion,
        'local_success_latency': latency([results[r['id']]['latency_ms'] for r in successful]),
        'historical_baseline_paired_latency': latency([r['baseline_latency_ms'] for r in successful]),
        'note': 'Agreement is not accuracy. Historical baseline latency includes parameter generation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('--data', type=Path, default=ROOT/'questions.jsonl')
    parser.add_argument('--out', type=Path, default=ROOT/'output')
    parser.add_argument('--base-url', dest='api_url', default=API_URL, help='SDK base URL, without /v1/systemone')
    parser.add_argument('--model', default='kev-latest', help='Local model name')
    parser.add_argument('--api-key-env', default='LOCAL_MODEL_API_KEY')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--gold', type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    if args.command == 'run':
        try:
            import typesafe_sdk  # Fail before starting the batch if not installed.
        except ImportError:
            parser.error('Install SDK first: python3 -m pip install -r requirements.txt')
    rows = read_jsonl(args.data)
    if len(rows) != 20 or len({r['id'] for r in rows}) != 20:
        raise ValueError('This benchmark requires exactly 20 unique samples')
    payloads = {}
    for row in rows:
        payload = copy.deepcopy(row['payload'])
        if set(payload['state']) != {'question'} or payload['state']['question'] != row['question']:
            raise ValueError('Only the original question is allowed in state')
        if row['baseline_skill'] not in payload['questions']['skill']['criteria']:
            raise ValueError('Invalid baseline skill')
        if args.model:
            payload['model'] = args.model
        payloads[row['id']] = payload
    gold = {}
    known = {r['id']:r for r in rows}
    if args.gold:
        seen = set()
        for item in read_jsonl(args.gold):
            rid = item['id']
            if rid not in known or rid in seen:
                raise ValueError('Unknown or duplicate gold id')
            seen.add(rid)
            if item.get('gold_skill') is not None:
                if item['gold_skill'] not in payloads[rid]['questions']['skill']['criteria']:
                    raise ValueError('Invalid gold skill')
                gold[rid] = item['gold_skill']
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset_sha256':digest(rows), 'model_override':args.model,
                'api_url':args.api_url}
    manifest_path = args.out/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['dataset_sha256'] != manifest['dataset_sha256']:
            raise ValueError('Different dataset: use a new --out directory')
        if args.command == 'run' and previous != manifest:
            raise ValueError('Different endpoint/model: use a new --out directory')
        if args.command != 'run':
            # Reports can be regenerated without repeating endpoint/model flags.
            for payload in payloads.values():
                if previous['model_override']:
                    payload['model'] = previous['model_override']
    results = {}
    path = args.out/'results.jsonl'
    if path.exists():
        for result in read_jsonl(path):
            if result['id'] not in known or result['payload_sha256'] != digest(payloads[result['id']]):
                raise ValueError('Stale result id/payload: use a new --out directory')
            results[result['id']] = result
    if args.command == 'prepare':
        write_jsonl(args.out/'requests.jsonl', [{'id':r['id'], 'payload':payloads[r['id']]} for r in rows])
        print(f"Prepared 20 requests: {args.out/'requests.jsonl'} (no network calls)")
        return 0
    if args.command == 'run':
        dump_json(manifest_path, manifest)
        key = os.environ.get(args.api_key_env) or 'local'
        pending = [r for r in rows if not results.get(r['id'], {}).get('prediction')]
        def execute(row):
            payload = payloads[row['id']]
            return dict(request_jev(payload, key, args.api_url),
                        id=row['id'], payload_sha256=digest(payload))
        with path.open('a', encoding='utf-8') as f, ThreadPoolExecutor(args.workers) as pool:
            for i, future in enumerate(as_completed([pool.submit(execute,r) for r in pending]), 1):
                result = future.result()
                results[result['id']] = result
                f.write(json.dumps(result, ensure_ascii=False)+'\n')
                f.flush()
                print(f"[{i}/{len(pending)}] {result['id']}: {result['prediction'] or result['error']}", flush=True)
    summary = summarize(rows, results, gold)
    dump_json(args.out/'summary.json', summary)
    write_jsonl(args.out/'disagreements.jsonl', [dict(r, result=results.get(r['id'])) for r in rows
                if r['baseline_skill'] != results.get(r['id'], {}).get('prediction')])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(args.command == 'run' and summary['failed_or_missing'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
