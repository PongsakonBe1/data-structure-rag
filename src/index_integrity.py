"""
Utilities for index integrity verification using SHA-256 manifests.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

INDEX_MANIFEST_NAME = "index_manifest.json"
DEFAULT_INDEX_FILES = [
    "bm25_index.pkl",
    "faiss_index/index.faiss",
    "faiss_index/index.pkl",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_index_manifest(
    index_dir: Path,
    manifest_path: Path | None = None,
    files: list[str] | None = None,
) -> dict:
    files = files or DEFAULT_INDEX_FILES
    manifest_path = manifest_path or (index_dir / INDEX_MANIFEST_NAME)

    payload = {
        "version": 1,
        "generated_at": datetime.utcnow().isoformat(),
        "files": {},
    }

    for rel in files:
        path = index_dir / rel
        if not path.exists():
            continue
        payload["files"][rel] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_index_manifest(
    index_dir: Path,
    manifest_path: Path | None = None,
) -> tuple[bool, list[str]]:
    manifest_path = manifest_path or (index_dir / INDEX_MANIFEST_NAME)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Index manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    if not files:
        raise ValueError("Index manifest is empty.")

    errors: list[str] = []
    for rel, meta in files.items():
        path = index_dir / rel
        if not path.exists():
            errors.append(f"missing:{rel}")
            continue

        expected_sha = str(meta.get("sha256", "")).strip().lower()
        expected_size = int(meta.get("size", -1))

        current_sha = sha256_file(path)
        current_size = path.stat().st_size

        if expected_sha and current_sha != expected_sha:
            errors.append(f"sha_mismatch:{rel}")
        if expected_size >= 0 and current_size != expected_size:
            errors.append(f"size_mismatch:{rel}")

    return len(errors) == 0, errors
