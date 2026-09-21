"""Build a credential-free source/data archive for the remote CUDA evaluation."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from nimble.paths import PROJECT_ROOT


def build_archive(data, output):
    data = data.resolve()
    relative_data = data.relative_to(PROJECT_ROOT)
    if relative_data.parts[0] != "data" or data.suffix != ".jsonl":
        raise ValueError("Only an explicit JSONL file under data/ can be included")
    files = sorted((PROJECT_ROOT / "nimble").rglob("*.py"))
    files += [PROJECT_ROOT / "requirements/cuda-eval.txt", data]
    files += [PROJECT_ROOT / name for name in (
        "tests/__init__.py", "tests/test_cuda_scorer.py", "tests/test_evaluate_pilot.py",
        "data/typesafe_diverse_300_gpt56/all.jsonl",
    )]
    files = list(dict.fromkeys(files))
    manifest = {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Materialize contents, never filesystem symlinks. No .env, caches, or SSH keys.
    import io
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            content = path.read_bytes()
            member = tarfile.TarInfo(str(path.relative_to(PROJECT_ROOT)))
            member.size, member.mode = len(content), 0o644
            archive.addfile(member, io.BytesIO(content))
        content = (json.dumps(manifest, indent=2) + "\n").encode()
        member = tarfile.TarInfo("transfer_manifest.json")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_archive(args.data, args.output)
    print(f"Packaged {len(manifest)} source/data files into {args.output}")


if __name__ == "__main__":
    main()
