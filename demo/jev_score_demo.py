#!/usr/bin/env python3
"""Jev Score demo: rate an input against an ordered set of levels."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.typesafe.ai/v1/systemone"
def _find_api_key_file() -> Path:
    for path in (
        Path(__file__).resolve().parent / "jev_api_key.txt",
        Path(__file__).resolve().parent.parent / "jev_api_key.txt",
    ):
        if path.is_file():
            return path
    return Path(__file__).resolve().parent.parent / "jev_api_key.txt"


API_KEY_FILE = _find_api_key_file()


def load_api_key() -> str:
    try:
        api_key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"API key file not found: {API_KEY_FILE}") from exc

    if not api_key:
        raise RuntimeError(f"API key file is empty: {API_KEY_FILE}")
    return api_key


def request_score(state: str) -> dict:
    payload = {
        "model": "jev-latest",
        "state": state,
        "questions": {
            "severity": {
                "type": "score",
                "instructions": "How severe is this issue?",
                "criteria": [
                    "Minor inconvenience with an easy workaround",
                    "Significant impact, but work can continue",
                    "Critical outage that blocks all users",
                ],
            }
        },
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {load_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            return json.load(response)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Jev API returned HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the Jev API: {exc.reason}") from exc


def main() -> int:
    state = " ".join(sys.argv[1:]).strip() or (
        "The service is unavailable and every customer is blocked."
    )
    try:
        result = request_score(state)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
