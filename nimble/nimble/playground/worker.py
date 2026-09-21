"""Loopback-only CUDA worker for the comparison playground; no provider keys."""
import argparse
import hashlib
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--warmups', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    import torch
    sys.path.insert(0, str(args.model_dir.resolve()))
    import inference
    cls = getattr(inference, 'NimbleModel', None) or getattr(inference, 'OpenJeffModel')
    model = cls(args.model_dir)
    contract = model.contract
    metadata = {'model': 'nimble-diverse9b-v2', 'training_examples': contract['data_audit']['training_rows'],
                'hardware': torch.cuda.get_device_name(), 'backend': 'CUDA, BF16, unmerged LoRA',
                'adapter_sha256': hashlib.sha256((args.model_dir/'adapter_model.safetensors').read_bytes()).hexdigest(),
                'base_revision': contract['revision'], 'max_input_tokens': contract['max_length']}
    assert metadata['training_examples'] == 2826
    for row in json.loads(args.warmups.read_text()):
        model.score(row['context'], row['schema'], row['score_fields'])
    torch.cuda.synchronize()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, *args):
            pass

        def respond(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond(200 if self.path == '/health' else 404, metadata if self.path == '/health' else {})

        def do_POST(self):
            if self.path != '/score':
                return self.respond(404, {})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 100_000:
                    raise ValueError('Request is too large')
                body = json.loads(self.rfile.read(size))
                if set(body) != {'context', 'schema', 'score_fields'}:
                    raise ValueError('Expected context, schema, and score_fields')
                if not lock.acquire(blocking=False):
                    return self.respond(429, {'error': 'GPU is busy. Try again shortly.'})
                try:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = model.score(body['context'], body['schema'], body['score_fields'])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-started
                finally:
                    lock.release()
                self.respond(200, {'result': result, 'scoring_ms': elapsed*1000, 'metadata': metadata})
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception:
                self.respond(500, {'error': 'GPU scoring failed; inspect the worker log.'})
                raise

    print(json.dumps({'ready': True, **metadata}), flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
