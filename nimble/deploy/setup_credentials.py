"""Create the dedicated download secret and save a proxy token privately."""

import json
import os

from dotenv import dotenv_values
import modal
from modal.workspace import Workspace
from modal_cli import ROOT, configure


if __name__ == "__main__":
    configure()
    values = dotenv_values(ROOT / ".env")
    token = os.getenv("HF_TOKEN") or values.get("HF_TOKEN") or values.get("HF_API_KEY")
    if not token:
        raise SystemExit("Set HF_TOKEN or HF_API_KEY in .env first")
    modal.Secret.objects.create("openjeff-huggingface", {"HF_TOKEN": token}, allow_existing=True)
    path = ROOT / ".cache/modal/proxy-token.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        credentials = Workspace.from_context().proxy_tokens.create()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as file:
            json.dump({"Modal-Key": credentials.token_id, "Modal-Secret": credentials.token_secret}, file)
    saved = json.loads(path.read_text())
    Workspace.from_context().proxy_tokens.allow(saved["Modal-Key"], os.getenv("MODAL_ENVIRONMENT", "main"))
    print("Download secret configured; proxy credentials saved privately in .cache/modal/proxy-token.json")
