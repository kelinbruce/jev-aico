"""Run Modal with the project-local credentials without logging or copying them."""

import os
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def configure():
    path = ROOT / ".modal.toml"
    if not path.exists():
        path = ROOT / ".modal.tomlf"
    if path.exists():
        config = tomllib.loads(path.read_text())
        profile = config.get("default") or next(iter(config.values()))
        for key in ("token_id", "token_secret"):
            value = profile.get(key) or profile.get(key.replace("_", "-"))
            if value:
                os.environ.setdefault("MODAL_" + key.upper(), value)


if __name__ == "__main__":
    configure()
    os.chdir(ROOT)
    os.execv(sys.executable, [sys.executable, "-m", "modal", *sys.argv[1:]])
