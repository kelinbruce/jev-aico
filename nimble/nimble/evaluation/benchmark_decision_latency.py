"""Measure native Nimble scoring and Jev decisions from one client.

Serve a frozen input catalog through an SSH tunnel; API credentials stay on the
client. Both providers receive the same semantic questions at concurrency one.
"""
import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def timing_summary(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Expected positive finite timings")
    values = sorted(v*1000 for v in values)
    pos = (len(values)-1)*.95
    lo, hi = math.floor(pos), math.ceil(pos)
    return {"count": len(values), "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "p95_ms": values[lo]+(values[hi]-values[lo])*(pos-lo),
            "min_ms": values[0], "max_ms": values[-1]}


def load_inputs(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 324 or len({r["id"] for r in rows}) != 324:
        raise ValueError("Expected the complete 324-example holdout")
    if any(set(r) != {"id", "input", "context", "schema", "kind"} for r in rows):
        raise ValueError("Inference inputs must exclude gold labels and training metadata")
    return rows


def serve(args):
    import sys
    import socket
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from importlib.metadata import version
    import torch
    sys.path.insert(0, str(args.model_dir.resolve()))
    import inference
    from nimble.compat import published_model_class

    rows = load_inputs(args.inputs)
    lookup = {r["id"]: r for r in rows}
    model = published_model_class(inference)(args.model_dir)
    audit = model.contract["data_audit"]
    assert audit["training_rows"] == 2826 and audit["validation_rows"] == 324
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = {"model": "nimble-diverse9b-v2", "training_examples": 2826,
               "heldout_examples": 324, "base_model": model.contract["model"],
               "base_revision": model.contract["revision"],
               "adapter_sha256": hashlib.sha256((args.model_dir/"adapter_model.safetensors").read_bytes()).hexdigest(),
               "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
               "gpu": torch.cuda.get_device_name(), "adapter_merged": False,
               "versions": {p: version(p) for p in ("torch", "transformers", "peft", "accelerate")},
               "concurrency": 1, "batch_size": 1, "forward_passes_per_request": 1,
               "autoregressive_tokens": 0, "temperature": 1,
               "scoring_boundary": "Synchronized model.score including tokenization and probability formatting; excludes loading, transport and HTTP response serialization"}
    (args.output_dir/"runtime.json").write_text(json.dumps(runtime, indent=2)+"\n")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, *unused):
            pass

        def respond(self, status, obj):
            body = json.dumps(obj, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def do_GET(self):
            self.respond(200 if self.path == "/health" else 404, runtime if self.path == "/health" else {})

        def do_POST(self):
            if self.path != "/score":
                return self.respond(404, {})
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 1_000_000:
                return self.respond(400, {"error": "Invalid request size"})
            payload = json.loads(self.rfile.read(length))
            if set(payload) != {"id", "input_fingerprint", "input"} or payload["id"] not in lookup:
                return self.respond(400, {"error": "Unknown input"})
            row = lookup[payload["id"]]
            if payload["input_fingerprint"] != fingerprint(row["input"]) or payload["input"] != row["input"]:
                return self.respond(400, {"error": "Input changed"})
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = model.score(row["context"], row["schema"], score_fields=["decision"] if row["kind"] == "score" else [])
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            record = {"id": row["id"], "input_fingerprint": payload["input_fingerprint"],
                      "scoring_seconds": elapsed, "result": result}
            with (args.output_dir/"server_responses.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False)+"\n")
            self.respond(200, record)

    print("READY: frozen checkpoint loaded; loopback-only server", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


def compare(args):
    import os
    import requests
    from dotenv import load_dotenv
    from nimble.paths import PROJECT_ROOT
    from nimble.datasets.dataset_io import request_for, validate_teacher

    rows = load_inputs(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = args.output_dir/"responses.jsonl"
    if progress.exists():
        raise FileExistsError("Use a fresh output directory; runs cannot be mixed")
    load_dotenv(PROJECT_ROOT/".env")
    # Distinct sessions prevent credentials ever reaching the Nimble server.
    jev = requests.Session()
    jev.headers["Authorization"] = "Bearer " + os.environ["TYPESAFE_API_KEY"]
    nimble = requests.Session()
    response = nimble.get(args.nimble_url+"/health", timeout=15)
    response.raise_for_status()
    runtime = response.json()
    assert runtime["inputs_sha256"] == hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    settings = {"runtime": runtime, "inputs_sha256": runtime["inputs_sha256"],
                "jev_model": "jev-1.13.0", "concurrency": 1, "requests_per_model": len(rows),
                "repetitions_per_example": 1, "warmups_per_model": 3,
                "transport": "Same local client: Nimble via persistent SSH tunnel to H100; Jev via HTTPS. Separate persistent HTTP sessions.",
                "ordering": "Alternate which provider runs first for successive examples; no concurrent requests",
                "client_boundary": "HTTP request through JSON response parsing; excludes post-response validation, provider errors have separate timings",
                "retry_policy": "No automatic retries", "autoregressive_generation": False}
    (args.output_dir/"settings.json").write_text(json.dumps(settings, indent=2)+"\n")

    def call(name, row, warmup=False):
        digest = fingerprint(row["input"])
        payload = ({"id": row["id"], "input_fingerprint": digest, "input": row["input"]} if name == "nimble"
                   else request_for(row, "jev-1.13.0"))
        session = nimble if name == "nimble" else jev
        url = args.nimble_url+"/score" if name == "nimble" else "https://api.typesafe.ai/v1/systemone"
        started = time.perf_counter()
        try:
            response = session.post(url, json=payload, timeout=90)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            body = response.json()
            elapsed = time.perf_counter()-started
            if name == "jev":
                validate_teacher(row, body, "jev-1.13.0")
            else:
                assert body["id"] == row["id"] and body["input_fingerprint"] == digest
            return {"model": name, "id": row["id"], "kind": row["kind"],
                    "warmup": warmup, "input_fingerprint": digest, "elapsed_seconds": elapsed,
                    "response": body}
        except (requests.RequestException, ValueError, RuntimeError, AssertionError, KeyError) as exc:
            # Never include provider bodies, authorization headers, or request objects.
            return {"model": name, "id": row["id"], "kind": row["kind"], "warmup": warmup,
                    "input_fingerprint": digest, "elapsed_seconds": time.perf_counter()-started,
                    "error": type(exc).__name__ + (": "+str(exc) if isinstance(exc, RuntimeError) else "")}

    warmups = [next(r for r in rows if r["kind"] == kind) for kind in ("choice", "noul", "score")]
    with (args.output_dir/"warmups.jsonl").open("w") as stream:
        for row in warmups:
            for name in ("nimble", "jev"):
                result = call(name, row, True)
                stream.write(json.dumps(result, allow_nan=False)+"\n")
                if "error" in result:
                    raise RuntimeError(f"{name} warmup failed: {result['error']}")
    saved = []
    with progress.open("w") as stream:
        for i, row in enumerate(rows):
            for name in (("nimble", "jev") if i % 2 == 0 else ("jev", "nimble")):
                result = call(name, row)
                saved.append(result)
                stream.write(json.dumps(result, allow_nan=False)+"\n")
                stream.flush()
            if (i+1) % 20 == 0 or i+1 == len(rows):
                print(f"Latency pairs: {i+1}/{len(rows)}", flush=True)
    summary = {"models": {}, "settings": settings}
    for name in ("nimble", "jev"):
        records = [r for r in saved if r["model"] == name]
        valid = [r for r in records if "error" not in r]
        item = {"attempted": len(records), "failures": len(records)-len(valid),
                "client": timing_summary([r["elapsed_seconds"] for r in valid]) if valid else None,
                "by_kind": {kind: timing_summary([r["elapsed_seconds"] for r in valid if r["kind"] == kind])
                            for kind in ("choice", "noul", "score") if any(r["kind"] == kind for r in valid)}}
        if name == "nimble" and valid:
            item["scoring"] = timing_summary([r["response"]["scoring_seconds"] for r in valid])
        summary["models"][name] = item
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary["models"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--model-dir", type=Path, required=True)
    server.add_argument("--port", type=int, default=8765)
    client = sub.add_parser("compare")
    client.add_argument("--nimble-url", default="http://127.0.0.1:18765")
    for command in (server, client):
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (serve if args.mode == "serve" else compare)(args)


if __name__ == "__main__":
    main()
