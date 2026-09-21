"""Verify the deployed endpoint; credentials are never printed."""

import json
import argparse
import math
import time

import httpx
from modal_cli import ROOT

PAYLOAD = {
    "model": "nimble-latest",
    "state": "The customer was charged twice and explicitly requests a refund. The service still works normally.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
        "department": {"type": "choice", "instructions": "Which department should handle this request?",
                       "criteria": {"billing": "Charges, payments and refunds", "technical": "Software bugs and outages"}},
        "urgency": {"type": "score", "instructions": "Assess operational urgency.",
                    "criteria": ["Routine billing request; service works", "Some functionality unavailable", "Complete service outage"]},
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://bespokelabs--nimble-sglang-nimble.us-west.modal.direct")
    parser.add_argument("--private", action="store_true", help="Use local proxy credentials and verify authentication")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    headers = json.loads((ROOT / ".cache/modal/proxy-token.json").read_text()) if args.private else {}
    with httpx.Client(timeout=30) as client:
        unauthorized_status = None
        if args.private:
            unauthorized = client.get(url + "/health")
            unauthorized_status = unauthorized.status_code
            assert unauthorized_status == 401, f"Expected auth protection; got {unauthorized_status}"
        started = time.monotonic()
        deadline = started + 1200
        while True:
            try:
                response = client.get(url + "/health", headers=headers)
                if response.status_code == 200:
                    break
                if response.status_code not in (429, 502, 503, 504):
                    response.raise_for_status()
            except httpx.TimeoutException:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Endpoint did not become healthy within 20 minutes")
            time.sleep(5)
        health = response.json()
        readiness = time.monotonic() - started
        runs = []
        for _ in range(2):
            started = time.monotonic()
            response = client.post(url + "/v1/systemone", headers=headers, json=PAYLOAD, timeout=150)
            response.raise_for_status()
            body = response.json()
            answers = body["answers"]
            assert answers["department"]["choice"] == "billing"
            assert answers["refund"]["noul"] > 0.5
            assert 0 <= answers["urgency"]["score"] <= 2
            for field in ("department", "urgency"):
                assert math.isclose(sum(answers[field]["probabilities"].values()), 1, abs_tol=1e-6)
            assert body["usage"]["output_tokens"] == 4
            runs.append({"elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                         "timing": response.headers.get("server-timing"),
                         "prefix_tokens": response.headers.get("x-openjev-prefix-tokens"),
                         "cached_tokens": response.headers.get("x-openjev-cached-tokens"), "response": body})
        result = {"url": url, "access": "private" if args.private else "public",
                  "unauthorized_status": unauthorized_status,
                  "health": health, "readiness_seconds": round(readiness, 2), "runs": runs}
        (ROOT / ".cache/modal/smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
