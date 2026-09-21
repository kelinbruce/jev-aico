"""Evaluate decoded OpenRouter answers on saved examples; never fabricate probabilities."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import string
import time
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv

from nimble.paths import PROJECT_ROOT
from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.evaluation.evaluate_models import teacher_assessment
from nimble.scoring.parallel_schema import SYSTEM_PROMPT, choice_key, safe_json

MODELS = {
    'deepseek_v41_flash': {'id': 'deepseek/deepseek-v4.1-flash', 'provider': 'fireworks',
                          'input_per_million': .30, 'output_per_million': 1.20},
    'qwen38_24t': {'id': 'qwen/qwen3.8-2.4t-a95b', 'provider': 'modal',
                   'input_per_million': 2, 'output_per_million': 6},
}


def make_messages(input_data):
    context, schema = adapt_input(input_data)
    if list(schema) != ['decision']:
        raise ValueError('Exactly one decision required')
    field = schema['decision']
    values = field['choices']
    choices = [{'code': code, 'value': value,
                **({'description': field['choice_descriptions'][choice_key(value)]}
                   if choice_key(value) in field.get('choice_descriptions', {}) else {})}
               for code, value in zip(string.ascii_uppercase, values)]
    content = safe_json({'context': context, 'schema': [
        {'name': 'decision', 'description': field['description'], 'choices': choices}]})
    content += '\n\nRequested field: "decision"'
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}], dict(
        zip(string.ascii_uppercase, values))


def parse_answer(content, finish_reason, mapping, kind):
    # No semantic repair, answer extraction from reasoning, or reference-aware retry.
    if finish_reason != 'stop':
        raise ValueError('Non-stop completion: ' + str(finish_reason))
    if not isinstance(content, str) or not re.fullmatch('[A-Z]', content.strip()):
        raise ValueError('Expected exactly one uppercase answer code')
    code = content.strip()
    if code not in mapping:
        raise ValueError('Answer code outside candidate set')
    value = mapping[code]
    return int(value) if kind == 'score' else value


def summarize(rows):
    out = {}
    for kind in ('all', 'choice', 'noul', 'score'):
        selected = [r for r in rows if kind == 'all' or r['type'] == kind]
        if not selected:
            continue
        valid = [r for r in selected if r['valid']]
        out[kind] = {'count': len(selected), 'valid': len(valid),
                     'correct': sum(r['correct'] for r in selected),
                     'accuracy': sum(r['correct'] for r in selected) / len(selected),
                     'teacher_agreement': sum(r['teacher_agreement'] for r in selected)}
        if kind == 'score':
            out[kind]['hard_label_mae_on_valid'] = statistics.mean(
                abs(r['prediction'] - r['reference']['target']) for r in valid) if valid else None
        if kind == 'noul':
            out[kind]['confusion_on_valid'] = dict(Counter(
                ('TP' if r['prediction'] else 'FN') if r['reference']['target'] else
                ('FP' if r['prediction'] else 'TN') for r in valid))
    return out


def run_one(row, config, output, key, settings, reservation):
    messages, mapping = make_messages(row['input'])
    payload = {'model': config['id'], 'messages': messages, 'temperature': 0,
               'max_tokens': settings['max_tokens'], 'reasoning': {'effort': 'medium'},
               'provider': {'only': [config['provider']], 'allow_fallbacks': False,
                            'require_parameters': True}, 'stream': False}
    rid = row['id']
    path = output / 'responses' / (rid + '.json')
    # Persist intent before sending; interrupted/ambiguous requests consume full reservation.
    intent = {'id': rid, 'request_sha256': hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
              'reservation_usd': reservation, 'started_epoch': time.time()}
    (output / 'requests' / (rid + '.json')).write_text(json.dumps(intent, indent=2))
    started = time.perf_counter()
    try:
        response = requests.post('https://openrouter.ai/api/v1/chat/completions',
                                 headers={'Authorization': 'Bearer ' + key}, json=payload, timeout=(20,240))
        try:
            body = response.json()
        except ValueError:
            body = {'error': {'message': 'Non-JSON response', 'status': response.status_code}}
        result = {**intent, 'http_status': response.status_code, 'response': body,
                  'elapsed_seconds': time.perf_counter() - started,
                  'retry_after': response.headers.get('Retry-After')}
    except requests.RequestException as exc:
        result = {**intent, 'http_status': None, 'transport_error': type(exc).__name__,
                  'elapsed_seconds': time.perf_counter() - started}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def assessment(row, saved):
    kind = row['input']['questions']['decision']['type']
    body = saved.get('response', {})
    result = {'id': row['id'], 'type': kind, 'domain': row['domain'], 'reference': row['reference'],
              'valid': False, 'prediction': None, 'correct': False, 'teacher_agreement': False,
              'elapsed_seconds': saved['elapsed_seconds'], 'response_id': body.get('id'),
              'provider': body.get('provider'), 'returned_model': body.get('model'), 'usage': body.get('usage')}
    try:
        if saved.get('http_status') != 200 or body.get('error'):
            raise ValueError('API/transport error')
        message = body['choices'][0]
        result['prediction'] = parse_answer(message['message'].get('content'), message.get('finish_reason'),
                                            make_messages(row['input'])[1], kind)
        result['valid'] = True
        result['correct'] = result['prediction'] == row['reference']['target']
        result['teacher_agreement'] = result['prediction'] == teacher_assessment(row,kind)['prediction']
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        result['error'] = str(exc)
    return result


def charged_or_reserved(saved):
    cost = (saved.get('response', {}).get('usage') or {}).get('cost')
    return float(cost) if isinstance(cost, (float,int)) and math.isfinite(cost) and cost >= 0 else saved['reservation_usd']


def evaluate(args):
    load_dotenv(PROJECT_ROOT / '.env')
    key = os.environ['OPENROUTER_API_KEY']
    config = MODELS[args.model]
    raw = args.data.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line]
    assert len({r['id'] for r in records}) == len(records)
    output = args.output_dir / args.model
    for name in ('responses', 'requests'):
        (output / name).mkdir(parents=True, exist_ok=True)
    settings = {'model': config['id'], 'provider': config['provider'], 'dataset_sha256': hashlib.sha256(raw).hexdigest(),
                'temperature': 0, 'reasoning_effort': 'medium', 'max_tokens': 4096,
                'prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                'mode': 'regular decoding; strict single-letter final answer', 'budget_usd': args.budget,
                'probability_metrics': 'unavailable; no fabricated or self-reported probabilities',
                'parse_failures': 'counted incorrect; no answer-dependent retries',
                'pricing_reservation': config, 'dataset_count': len(records)}
    settings_path = output / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Run settings differ; use a new output directory')
    settings_path.write_text(json.dumps(settings,indent=2))
    # Upper bound uses UTF-8 bytes as a conservative token count plus chat overhead.
    def reserve(row):
        messages,_ = make_messages(row['input'])
        n = len(json.dumps(messages,ensure_ascii=False).encode()) + 128
        return (n*config['input_per_million'] + settings['max_tokens']*config['output_per_million'])/1e6
    chosen = records[:args.limit] if args.limit else records
    if args.retry_transport_errors:
        archive = output / 'transport_attempts'
        archive.mkdir(exist_ok=True)
        for path in (output / 'responses').glob('*.json'):
            saved = json.loads(path.read_text())
            if saved.get('http_status') in (429, 500, 502, 503, 504):
                stem = path.stem
                path.rename(archive / (stem + '-' + str(time.time_ns()) + '.json'))
                (output / 'requests' / (stem + '.json')).unlink()
    cached = {p.stem:json.loads(p.read_text()) for p in (output/'responses').glob('*.json')}
    intents = {p.stem:json.loads(p.read_text()) for p in (output/'requests').glob('*.json')}
    ambiguous = set(intents)-set(cached)
    if ambiguous:
        raise ValueError(f'{len(ambiguous)} interrupted requests require reconciliation before resume')
    prior_transport_cost = sum(charged_or_reserved(json.loads(p.read_text())) for p in (output/'transport_attempts').glob('*.json'))
    spent = prior_transport_cost + sum(charged_or_reserved(s) for s in cached.values())
    pending = [r for r in chosen if r['id'] not in cached]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for start in range(0,len(pending),args.concurrency):
            if start and args.batch_pause:
                time.sleep(args.batch_pause)
            batch = pending[start:start+args.concurrency]
            if spent + sum(reserve(r) for r in batch) > args.budget:
                raise RuntimeError('Cost guard reached before starting next batch')
            futures = {pool.submit(run_one,r,config,output,key,settings,reserve(r)):r for r in batch}
            for future in concurrent.futures.as_completed(futures):
                saved = future.result(); cached[saved['id']] = saved; spent += charged_or_reserved(saved)
            current = [assessment(r,cached[r['id']]) for r in chosen if r['id'] in cached]
            errors = sum(not r['valid'] for r in current)
            print(f"{args.model}: {len(current)}/{len(chosen)}, invalid={errors}, cost/reserve=${spent:.4f}",flush=True)
            fatal_errors = sum(not r['valid'] and cached[r['id']].get('http_status') not in (429,500,502,503,504) for r in current)
            if fatal_errors >= 10 and fatal_errors/len(current) > .05:
                raise RuntimeError('More than 5% non-retryable failures; inspect API results before continuing')
    rows = [assessment(r,cached[r['id']]) for r in chosen]
    result = {**settings, 'count':len(rows), 'complete':len(rows)==len(records), 'summary':summarize(rows),
              'by_domain':{d:summarize([r for r in rows if r['domain']==d]) for d in sorted({r['domain'] for r in rows})},
              'cost_or_reservation_usd':spent, 'prior_transport_attempt_reserve_usd':prior_transport_cost, 'provider_counts':dict(Counter(r['provider'] for r in rows)),
              'returned_model_counts':dict(Counter(r['returned_model'] for r in rows))}
    (output/'rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (output/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['summary'],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'evaluations/large_models_20260917')
    p.add_argument('--budget',type=float,required=True)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--concurrency',type=int,default=20)
    p.add_argument('--batch-pause',type=float,default=0)
    p.add_argument('--retry-transport-errors',action='store_true',help='Archive and retry only HTTP 429/5xx, never model answers or truncations')
    p.add_argument('--transport-retry-rounds',type=int,default=0,
                   help='Bounded extra rounds for HTTP 429/5xx only, pausing 45 seconds between rounds')
    args=p.parse_args()
    for retry_round in range(args.transport_retry_rounds+1):
        evaluate(args)
        output=args.output_dir/args.model
        failures=[path for path in (output/'responses').glob('*.json')
                  if json.loads(path.read_text()).get('http_status') in (429,500,502,503,504)]
        if not failures or retry_round==args.transport_retry_rounds:
            break
        print(f'{len(failures)} transport failures remain; cooling down before retry round {retry_round+1}',flush=True)
        time.sleep(45)
        args.retry_transport_errors=True


if __name__=='__main__':
    main()
