"""Resident MLX inference worker for the local three-way comparison app."""
import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def normalized_field(field, schema, score_fields):
    """Adapt the native MLX result to the portable CUDA worker's wire format."""
    prediction = field['value']
    if 'decision' in score_fields:
        prediction = int(prediction)
    probabilities = {str(k).lower() if isinstance(k, bool) else str(k): v
                     for k, v in field['scores'].items()}
    return {'prediction': prediction, 'probabilities': probabilities}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('.cache/models/nimble-diverse9b-v2-bf16'))
    parser.add_argument('--warmups', type=Path, default=Path('.cache/comparison-gpu/warmups.json'))
    parser.add_argument('--port', type=int, default=18767)
    args = parser.parse_args()
    import mlx.core as mx
    from nimble.scoring.parallel_scorer import ParallelScorer

    contract = json.loads((args.model_dir/'schema_config.json').read_text())
    merge = json.loads((args.model_dir/'READY.json').read_text())
    prompt_hash = hashlib.sha256(Path('nimble/scoring/parallel_schema.py').read_bytes()).hexdigest()
    if contract['prompt_code_sha256'] != prompt_hash or merge['training_examples'] != 2826:
        raise ValueError('Checkpoint contract does not match this comparison')

    # Keep all Metal work on one thread, including initialization and warmup.
    pool = ThreadPoolExecutor(max_workers=1)

    def load():
        model = ParallelScorer(model_path=args.model_dir, max_input_tokens=contract['max_length'],
                               model_id='nimble-diverse9b-v2', revision=merge['adapter_sha256'])
        mx.eval(model.model.parameters())
        mx.synchronize()
        for row in json.loads(args.warmups.read_text()):
            model.score(row['context'], row['schema'], mode='independent')
        mx.synchronize()
        return model

    model = pool.submit(load).result()
    metadata = {'model': 'nimble-diverse9b-v2', 'training_examples': merge['training_examples'],
                'hardware': mx.device_info()['device_name'], 'backend': 'MLX, merged BF16, FP32 candidate head',
                'adapter_sha256': merge['adapter_sha256'], 'base_revision': merge['base_revision'],
                'max_input_tokens': contract['max_length'], 'forward_passes_per_example': 1,
                'autoregressive_tokens': 0}
    lock = threading.Lock()

    def score(body):
        mx.synchronize()
        started = time.perf_counter()
        result = model.score(body['context'], body['schema'], mode='independent')
        mx.synchronize()
        elapsed = (time.perf_counter()-started)*1000
        field = normalized_field(result['fields']['decision'], body['schema'], body['score_fields'])
        return {'result': {'fields': {'decision': field}}, 'scoring_ms': elapsed, 'metadata': metadata}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def respond(self, status, data):
            raw = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def allowed(self):
            return self.headers.get('Host') in (f'127.0.0.1:{args.port}', f'localhost:{args.port}')

        def do_GET(self):
            if not self.allowed():
                return self.respond(403, {})
            self.respond(200 if self.path == '/health' else 404, metadata if self.path == '/health' else {})

        def do_POST(self):
            if not self.allowed() or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.respond(403, {})
            if self.path != '/score':
                return self.respond(404, {})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 100_000:
                    raise ValueError('Request must be under 100 KB')
                body = json.loads(self.rfile.read(size))
                if set(body) != {'context', 'schema', 'score_fields'}:
                    raise ValueError('Expected context, schema, and score_fields only')
                if not lock.acquire(blocking=False):
                    return self.respond(429, {'error': 'MacBook is busy. Try again shortly.'})
                try:
                    result = pool.submit(score, body).result()
                finally:
                    lock.release()
                self.respond(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception:
                self.respond(500, {'error': 'Mac scoring failed; inspect the worker log.'})
                raise

    print(json.dumps({'ready': True, **metadata}), flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
