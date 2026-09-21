"""Benchmark a saved request against the Modal endpoint."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import httpx

from modal_cli import ROOT


def stats(values):
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return {key: round(value, 2) for key, value in {
        "min": min(values), "median": statistics.median(values),
        "mean": statistics.mean(values), "p95": p95, "max": max(values),
    }.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-label", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--private", action="store_true", help="Use local proxy credentials")
    parser.add_argument("--url", default="https://bespokelabs--nimble-sglang-nimble.us-west.modal.direct")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    payload = json.loads(args.request.read_text())
    headers = json.loads((ROOT / ".cache/modal/proxy-token.json").read_text()) if args.private else {}
    started = time.monotonic()
    runs = []
    with httpx.Client(headers=headers, timeout=30) as client:
        while True:
            try:
                response = client.get(args.url + "/health")
                if response.status_code == 200:
                    break
                if response.status_code not in (429, 502, 503, 504):
                    response.raise_for_status()
            except httpx.TimeoutException:
                pass
            if time.monotonic() - started > 1200:
                raise TimeoutError("Service not ready within 20 minutes")
            time.sleep(5)
        readiness = time.monotonic() - started
        health = response.json()
        print(f"Readiness wait: {readiness:.2f} s", flush=True)
        for index in range(args.count + 1):
            begin = time.perf_counter()
            response = client.post(args.url + "/v1/systemone", json=payload, timeout=150)
            elapsed = (time.perf_counter() - begin) * 1000
            response.raise_for_status()
            body = response.json()
            if set(body["answers"]) != set(payload["questions"]):
                raise ValueError("Response fields differ from the request")
            for answer in body["answers"].values():
                if "probabilities" in answer:
                    values = list(answer["probabilities"].values())
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or not math.isclose(sum(values), 1, abs_tol=1e-6):
                        raise ValueError("Invalid probability distribution")
            timing = {}
            for item in response.headers.get("server-timing", "").split(","):
                if ";dur=" in item:
                    name, value = item.strip().split(";dur=", 1)
                    timing[name] = float(value)
            if set(timing) != {"prepare", "prefill", "branches"}:
                raise ValueError("Missing server timing measurements")
            runs.append({"run": index, "phase": "warmup" if index == 0 else "measured",
                         "end_to_end_ms": round(elapsed, 3), "server_timing_ms": timing,
                         "response": body})
            label = "Warmup" if index == 0 else f"Run {index:02d}"
            print(f"{label}: {elapsed:.2f} ms", flush=True)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_label": args.gpu_label, "url": args.url, "request": payload,
        "method": "One warmup then sequential identical requests using a persistent HTTP client; no intentional pauses; p95 uses linear interpolation. Warm results include repeated-prompt cache reuse.",
        "readiness_wait_seconds": round(readiness, 2), "health": health,
        "sample_size": args.count,
        "end_to_end_ms": stats([r["end_to_end_ms"] for r in runs[1:]]),
        "server_total_ms": stats([sum(r["server_timing_ms"].values()) for r in runs[1:]]),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("SUMMARY " + json.dumps({k: v for k, v in result.items() if k not in ("request", "runs")}), flush=True)


if __name__ == "__main__":
    main()
