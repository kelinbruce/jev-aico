"""Loopback-only protocol adapter for Curator's public OpenAI-compatible backend.

TypeSafe uses /v1/systemone, not chat completions. This adapter transports a
JSON-encoded TypeSafe request/response without changing the teacher's outputs.
The real API key stays here and never enters Curator's request cache.
"""

import hmac
import json
import secrets
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def call_typesafe(payload, api_key):
    request = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def completion_envelope(result):
    usage = result["usage"]
    return {
        "id": "typesafe-" + secrets.token_hex(8),
        "object": "chat.completion", "created": int(time.time()),
        "model": result["model"],
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(result, allow_nan=False),
        }}],
        "usage": {"prompt_tokens": usage["input_tokens"],
                  "completion_tokens": usage["output_tokens"],
                  "total_tokens": usage["input_tokens"] + usage["output_tokens"]},
    }


@contextmanager
def bridge(api_key, request_fn=call_typesafe):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, body):
            encoded = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.respond(404, {"error": "Unknown endpoint"})
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                return self.respond(401, {"error": "Invalid local adapter token"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    return self.respond(400, {"error": "Invalid request size"})
                body = json.loads(self.rfile.read(size))
                # Curator probes rate-limit headers with an empty messages list.
                # Answer locally; the probe must never consume a teacher request.
                if body.get("messages") == []:
                    return self.respond(200, {})
                messages = body["messages"]
                if len(messages) != 1 or messages[0]["role"] != "user":
                    raise ValueError("Expected one user message containing a TypeSafe request")
                payload = json.loads(messages[0]["content"])
                if set(payload) != {"model", "state", "questions"}:
                    raise ValueError("Unexpected TypeSafe request fields")
                self.respond(200, completion_envelope(request_fn(payload, api_key)))
            except urllib.error.HTTPError as exc:
                # Do not expose provider bodies/headers or credentials in logs.
                message = "TypeSafe rate limit" if exc.code in (429, 529) else f"TypeSafe HTTP {exc.code}"
                self.respond(429 if exc.code == 529 else exc.code, {"error": message})
            except (ValueError, KeyError, TypeError):
                self.respond(400, {"error": "Invalid adapter request or TypeSafe response"})
            except (urllib.error.URLError, TimeoutError, OSError):
                self.respond(502, {"error": "TypeSafe connection failed"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", token
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
