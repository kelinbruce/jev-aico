"""Local comparison UI: private Jev credentials, input-only inference, saved results."""
import argparse
import concurrent.futures
import hashlib
import html
import json
import math
import os
import random
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from nimble.paths import PROJECT_ROOT
from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.datasets.dataset_io import validate_teacher


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_input(value):
    if not isinstance(value, dict) or set(value) != {'state', 'questions'}:
        raise ValueError('Provide state and questions only; labels are never sent to either model.')
    questions = value['questions']
    if not isinstance(questions, dict) or set(questions) != {'decision'}:
        raise ValueError('This comparison supports one question, named decision, per example.')
    question = questions['decision']
    if question.get('type') not in ('choice', 'noul', 'score'):
        raise ValueError('Use Choice, Boolean, or Score.')
    criteria = question.get('criteria')
    if question['type'] == 'score' and (not isinstance(criteria, list) or not 2 <= len(criteria) <= 26):
        raise ValueError('Scores need 2–26 ordered descriptions.')
    if question['type'] == 'choice' and not isinstance(criteria, dict):
        raise ValueError('Choices must map label names to descriptions.')
    if question['type'] == 'noul' and (not isinstance(criteria, dict) or set(criteria) != {'false', 'true'}):
        raise ValueError('Boolean criteria must describe false and true.')
    adapt_input(value)
    return value


def sample_examples(path, count=30, seed=17):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    selected = []
    for kind in ('choice', 'noul', 'score'):
        group = [r for r in rows if r['input']['questions']['decision']['type'] == kind]
        selected.extend(rng.sample(group, count//3))
    rng.shuffle(selected)
    return [{'id': r['id'], 'input': r['input'], 'reference': r['reference']['target'],
             'domain': r.get('domain', r.get('source_family', 'Held-out example')),
             'kind': r['input']['questions']['decision']['type']} for r in selected]


def normalized_answer(probabilities, kind, prediction=None, reported_score=None):
    if not probabilities or any(not isinstance(p, (float, int)) or not math.isfinite(p) or p < 0 for p in probabilities.values()):
        raise ValueError('Invalid candidate probabilities')
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError('Probability total is zero')
    probs = {k: v/total for k, v in probabilities.items()}
    best = max(probs, key=probs.get)
    if prediction is None:
        prediction = best == 'true' if kind == 'noul' else int(best) if kind == 'score' else best
    result = {'prediction': prediction, 'probabilities': probs}
    if kind == 'score':
        result['expected_score'] = sum(int(k)*p for k,p in probs.items())
        if reported_score is not None:
            result['reported_score'] = reported_score
    return result


class Playground:
    providers = ('nimble', 'mac', 'jev')

    def __init__(self, examples, gpu_url, state_dir, run_file, mac_url='http://127.0.0.1:18767',
                 providers=None, adapter_dir=None):
        import requests
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT/'.env')
        self.examples = examples
        self.lookup = {r['id']: r for r in examples}
        self.gpu_url = gpu_url
        self.mac_url = mac_url
        self.providers = tuple(providers or self.providers)
        self.adapter_dir = adapter_dir or PROJECT_ROOT/'.cache/adapters/openjeff-diverse9b-v2'
        self.adapter_sha256 = hashlib.sha256((self.adapter_dir/'adapter_model.safetensors').read_bytes()).hexdigest()
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.run_file = run_file
        self.lock = threading.Lock()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.gpu = requests.Session()
        self.mac = requests.Session()
        self.jev = requests.Session()
        self.jev.headers['Authorization'] = 'Bearer '+os.environ['TYPESAFE_API_KEY']
        self.history = self.state_dir/'comparisons.jsonl'
        self.saved = []
        if self.history.exists():
            self.saved = [json.loads(line) for line in self.history.read_text().splitlines() if line.strip()]
        self.metadata = None

    def status(self):
        import requests
        run = json.loads(self.run_file.read_text()) if self.run_file and self.run_file.exists() else {}
        try:
            if 'nimble' not in self.providers:
                raise ValueError('H100 disabled')
            response = requests.get(self.gpu_url+'/health', timeout=3)
            response.raise_for_status()
            metadata = response.json()
            expected = json.loads((self.adapter_dir/'schema_config.json').read_text())
            if metadata['training_examples'] != expected['data_audit']['training_rows'] or metadata['adapter_sha256'] != self.adapter_sha256:
                raise ValueError('Unexpected checkpoint')
            self.metadata = metadata
            ready = True
        except Exception:
            ready = False
        mac_metadata = None
        try:
            if 'mac' not in self.providers:
                raise ValueError('Mac disabled')
            response = requests.get(self.mac_url+'/health', timeout=3)
            response.raise_for_status()
            mac_metadata = response.json()
            if mac_metadata['adapter_sha256'] != self.adapter_sha256 or mac_metadata['training_examples'] != 2826:
                raise ValueError('Unexpected Mac checkpoint')
            mac_ready = True
        except Exception:
            mac_ready = False
        return {'ready': ready or mac_ready, 'gpu_ready': ready, 'mac_ready': mac_ready,
                'mac_metadata': mac_metadata, 'model': 'Nimble 9B', 'training_examples': 2826,
                'hardware': 'RunPod H100', 'deadline_unix': run.get('deadline_unix'),
                'jev_model': 'jev-1.13.0', 'busy': self.lock.locked(), 'metadata': self.metadata}

    def invoke(self, provider, value):
        kind = value['questions']['decision']['type']
        context, schema = adapt_input(value)
        payload = ({'context': context, 'schema': schema, 'score_fields': ['decision'] if kind == 'score' else []}
                   if provider in ('nimble', 'mac') else {'model': 'jev-1.13.0', **value})
        url = {'nimble': self.gpu_url+'/score', 'mac': self.mac_url+'/score',
               'jev': 'https://api.typesafe.ai/v1/systemone'}[provider]
        session = {'nimble': self.gpu, 'mac': self.mac, 'jev': self.jev}[provider]
        started = time.perf_counter()
        try:
            response = session.post(url, json=payload, timeout=(3, 90))
            elapsed = (time.perf_counter()-started)*1000
            if response.status_code != 200:
                error = (response.json().get('error', 'Model rejected the request') if provider in ('nimble', 'mac')
                         else f'Jev returned HTTP {response.status_code}')
                return {'error': str(error), 'request_ms': elapsed}
            body = response.json()
            elapsed = (time.perf_counter()-started)*1000
            if provider in ('nimble', 'mac'):
                if body['metadata']['adapter_sha256'] != self.adapter_sha256:
                    raise ValueError('Checkpoint differs from the selected adapter')
                field = body['result']['fields']['decision']
                answer = normalized_answer(field['probabilities'], kind, field['prediction'])
                answer.update(scoring_ms=body['scoring_ms'], metadata=body['metadata'])
            else:
                validate_teacher({'input': value}, body, 'jev-1.13.0')
                field = body['answers']['decision']
                probs = {'false': 1-field['noul'], 'true': field['noul']} if kind == 'noul' else field['probabilities']
                # Preserve input order when rounded Jev probabilities tie.
                if kind == 'choice':
                    probs = {k: probs[k] for k in value['questions']['decision']['criteria']}
                elif kind == 'score':
                    probs = {str(i): probs[str(i)] for i in range(len(value['questions']['decision']['criteria']))}
                prediction = field['choice'] if kind == 'choice' else field['noul'] > .5 if kind == 'noul' else None
                answer = normalized_answer(probs, kind, prediction, field.get('score'))
                answer['usage'] = body['usage']
            return {**answer, 'request_ms': elapsed}
        except Exception as exc:
            # Do not expose credential-bearing request objects or provider bodies.
            return {'error': f'{provider.title()} request failed ({type(exc).__name__}). Check the connection and try again.',
                    'request_ms': (time.perf_counter()-started)*1000}

    def compare(self, example_id, value, on_event=None):
        validate_input(value)
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('A comparison is already running.')
        try:
            futures = {self.pool.submit(self.invoke, name, value): name for name in self.providers}
            result = {'run_id': uuid.uuid4().hex, 'id': example_id, 'input': value,
                      'input_fingerprint': digest(value), 'kind': value['questions']['decision']['type'],
                      'created_utc': datetime.now(timezone.utc).isoformat(),
                      'results': {}, 'agreement': None}
            original = self.lookup.get(example_id)
            result['reference_available'] = original is not None and digest(original['input']) == digest(value)
            if result['reference_available']:
                result['reference'] = original['reference']
            if on_event:
                on_event({'type': 'start', 'record': result})
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                answer = future.result()
                if result['reference_available'] and 'error' not in answer:
                    answer['correct'] = answer['prediction'] == original['reference']
                result['results'][name] = answer
                if on_event:
                    on_event({'type': 'result', 'provider': name, 'answer': answer})
            result['agreement'] = (all(a['prediction'] == result['results'][self.providers[0]]['prediction'] for a in result['results'].values())
                                   if all('error' not in a for a in result['results'].values()) else None)
            with self.history.open('a') as stream:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            self.saved.append(result)
            if on_event:
                on_event({'type': 'complete', 'record': result})
            return result
        finally:
            self.lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8891)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--public-origin', help='Exact HTTPS origin when hosted behind a TLS proxy')
    parser.add_argument('--proxy-host', help='Exact internal Host used by the HTTPS proxy, if rewritten')
    parser.add_argument('--password-file', type=Path, help='Private file containing the hosted site password')
    parser.add_argument('--providers', nargs='+', choices=['nimble', 'mac', 'jev'], default=['nimble', 'mac', 'jev'])
    parser.add_argument('--adapter-dir', type=Path, default=PROJECT_ROOT/'.cache/adapters/openjeff-diverse9b-v2')
    parser.add_argument('--data', type=Path, default=PROJECT_ROOT/'data/eval.jsonl')
    parser.add_argument('--gpu-url', default='http://127.0.0.1:18766')
    parser.add_argument('--mac-url', default='http://127.0.0.1:18767')
    parser.add_argument('--state-dir', type=Path, default=PROJECT_ROOT/'.cache/comparison-app')
    parser.add_argument('--run-file', type=Path, default=PROJECT_ROOT/'.cache/comparison-gpu/run.json')
    args = parser.parse_args()
    if len(set(args.providers)) != len(args.providers) or len(args.providers) < 2:
        parser.error('Choose at least two distinct providers')
    if args.bind not in ('127.0.0.1', 'localhost') and not (args.public_origin and args.password_file):
        parser.error('Public binding requires an HTTPS origin and password file')
    if args.public_origin and (urlsplit(args.public_origin).scheme != 'https' or urlsplit(args.public_origin).path):
        parser.error('Public origin must be an HTTPS origin without a path')
    password = args.password_file.read_text().strip() if args.password_file else None
    if password is not None and len(password) < 24:
        parser.error('Use a randomly generated site password of at least 24 characters')
    app = Playground(sample_examples(args.data), args.gpu_url, args.state_dir, args.run_file, args.mac_url,
                     args.providers, args.adapter_dir)
    nonce = secrets.token_urlsafe(32)
    root = Path(__file__).parent
    origin = args.public_origin or f'http://127.0.0.1:{args.port}'
    session_cookie = secrets.token_urlsafe(48)
    origins = {origin} if args.public_origin else {origin, f'http://localhost:{args.port}'}
    hosts = {urlsplit(o).netloc for o in origins}
    login_page = b'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nimble comparison \xe2\x80\x94 Sign in</title><link rel="stylesheet" href="/style.css"><main class="login"><div class="eyebrow">BESPOKE / MODEL LAB</div><h1>Nimble \xc3\x97 Jev</h1><p>Private RunPod comparison</p><form action="/login" method="post"><label for="password">Access password</label><input id="password" type="password" name="password" required autocomplete="current-password"><button class="primary" type="submit">Open comparison</button></form></main></html>'''

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, data, mime='application/json'):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode() if mime == 'application/json' else data
            self.send_response(status)
            self.send_header('Content-Type', mime+'; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def allowed_host(self):
            if self.headers.get('Host') in hosts:
                return True
            return bool(args.public_origin and args.proxy_host
                        and self.headers.get('Host') == args.proxy_host
                        and self.headers.get('X-Forwarded-Host') == urlsplit(origin).netloc
                        and self.headers.get('X-Forwarded-Proto') == 'https')

        def authenticated(self):
            if password is None:
                return True
            try:
                cookies = SimpleCookie(self.headers.get('Cookie', ''))
                return secrets.compare_digest(cookies['nimble_session'].value, session_cookie)
            except (KeyError, ValueError, CookieError):
                return False

        def do_GET(self):
            if not self.allowed_host():
                return self.send(403, {'error': 'Invalid host'})
            if not self.authenticated() and self.path != '/style.css':
                return self.send(200, login_page, 'text/html') if self.path == '/' else self.send(401, {'error': 'Sign in to the comparison site.'})
            if self.path == '/':
                page = (root/'index.html').read_text().replace('__SESSION_TOKEN__', nonce)
                page = page.replace('__CONFIG__', html.escape(json.dumps({'providers': app.providers, 'hosted': bool(args.public_origin)}), quote=True))
                return self.send(200, page.encode(), 'text/html')
            if self.path in ('/app.js', '/style.css'):
                return self.send(200, (root/self.path[1:]).read_bytes(), 'text/javascript' if self.path.endswith('.js') else 'text/css')
            if self.path == '/api/status':
                return self.send(200, app.status())
            if self.path == '/api/examples':
                return self.send(200, {'examples': app.examples, 'seed': 17, 'selection': '10 random held-out examples per question type'})
            if self.path == '/api/results':
                return self.send(200, {'results': list(app.saved)})
            return self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.allowed_host() or self.headers.get('Origin', origin) not in origins:
                return self.send(403, {'error': 'Invalid origin'})
            if self.path == '/login' and password is not None:
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 4096:
                        raise ValueError('Invalid login')
                    supplied = parse_qs(self.rfile.read(size).decode()).get('password', [''])[0]
                    if not secrets.compare_digest(supplied.encode(), password.encode()):
                        return self.send(401, {'error': 'Incorrect access password. Go back and try again.'})
                    self.send_response(303)
                    self.send_header('Location', '/')
                    self.send_header('Set-Cookie', f'nimble_session={session_cookie}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=7200')
                    self.send_header('Content-Length', '0')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    return
                except (ValueError, UnicodeDecodeError):
                    return self.send(400, {'error': 'Invalid login'})
            if (not self.authenticated()
                    or not secrets.compare_digest(self.headers.get('X-Playground-Token', ''), nonce)):
                return self.send(403, {'error': 'Sign in and reload the app to run comparisons.'})
            if self.path != '/api/compare':
                return self.send(404, {'error': 'Not found'})
            streaming = self.headers.get('Accept') == 'application/x-ndjson'
            started = False
            disconnected = False

            def emit(event):
                nonlocal started, disconnected
                if disconnected:
                    return
                try:
                    if not started:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                        self.send_header('Cache-Control', 'no-store')
                        self.send_header('X-Accel-Buffering', 'no')
                        self.send_header('X-Content-Type-Options', 'nosniff')
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        self.close_connection = True
                        started = True
                    self.wfile.write((json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n').encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    # Finish and save inference even if the browser navigates away.
                    disconnected = True

            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 100_000:
                    raise ValueError('Request must be under 100 KB.')
                payload = json.loads(self.rfile.read(size))
                result = app.compare(payload.get('id', 'custom'), payload['input'], emit if streaming else None)
                if not streaming:
                    self.send(200, result)
            except BlockingIOError as exc:
                self.send(409, {'error': str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                if started:
                    emit({'type': 'error', 'error': 'Comparison interrupted. Completed cards are retained; try again.'})
                else:
                    self.send(400, {'error': str(exc)})

    print(f'Comparison app: {origin}', flush=True)
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
