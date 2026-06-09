"""
Security harness for index checksum verification.

What it validates:
1) Clean copied index set passes manifest verification.
2) Tampered manifest fails verification.
3) RAGSystem with strict verification blocks startup.
4) RAGSystem with non-strict verification starts with integrity warnings.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from index_integrity import INDEX_MANIFEST_NAME, verify_index_manifest  # noqa: E402


def run_python(code: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def tamper_manifest(manifest_path: Path) -> str:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    if not files:
        raise RuntimeError("Manifest has no files to tamper")
    target = next(iter(files))
    payload["files"][target]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def main() -> None:
    source_indexes = ROOT / "indexes"
    if not source_indexes.exists():
        raise SystemExit("indexes directory not found")

    with tempfile.TemporaryDirectory(prefix="rag-index-tamper-") as tmp_dir:
        tmp_indexes = Path(tmp_dir) / "indexes"
        shutil.copytree(source_indexes, tmp_indexes)

        manifest = tmp_indexes / INDEX_MANIFEST_NAME
        ok, errors = verify_index_manifest(tmp_indexes, manifest)
        if not ok:
            raise SystemExit(f"Unexpected initial verification failure: {errors}")
        print("Step 1 OK: copied indexes verify successfully.")

        target = tamper_manifest(manifest)
        ok_after, errors_after = verify_index_manifest(tmp_indexes, manifest)
        if ok_after:
            raise SystemExit("Tampered manifest should fail verification but passed.")
        print(f"Step 2 OK: tampered manifest detected ({target}) -> {errors_after}")

        base_env = os.environ.copy()
        base_env.update(
            {
                "RAG_INDEX_DIR": str(tmp_indexes),
                "INDEX_VERIFY_CHECKSUM": "1",
                "DENSE_RETRIEVAL_ENABLED": "0",
                "RERANKER_ENABLED": "0",
            }
        )

        strict_env = base_env | {"INDEX_VERIFY_STRICT": "1"}
        strict_proc = run_python("from src.retriever import RAGSystem; RAGSystem()", strict_env)
        if strict_proc.returncode == 0:
            raise SystemExit("Strict mode should fail on tampered manifest, but startup succeeded.")
        print("Step 3 OK: strict mode blocks startup on integrity mismatch.")

        relaxed_env = base_env | {"INDEX_VERIFY_STRICT": "0"}
        relaxed_code = (
            "import json; "
            "from src.retriever import RAGSystem; "
            "s=RAGSystem().get_runtime_status(); "
            "print(json.dumps(s)); "
            "assert s.get('index_integrity_ok') is False"
        )
        relaxed_proc = run_python(relaxed_code, relaxed_env)
        if relaxed_proc.returncode != 0:
            msg = relaxed_proc.stderr.strip() or relaxed_proc.stdout.strip()
            raise SystemExit(f"Non-strict mode failed unexpectedly: {msg}")
        print("Step 4 OK: non-strict mode starts with integrity warning state.")

    print("Tamper test PASSED.")


if __name__ == "__main__":
    main()
