"""
Verify SHA-256 manifest for index files under ./indexes.
Exit code 0 when valid, 1 when invalid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from index_integrity import verify_index_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify index checksum manifest.")
    parser.add_argument("--index-dir", default="indexes", help="Path to index directory.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional custom manifest path. Default: <index-dir>/index_manifest.json",
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    ok, errors = verify_index_manifest(index_dir=index_dir, manifest_path=manifest_path)
    if ok:
        print("Manifest verification: OK")
        raise SystemExit(0)

    print("Manifest verification: FAILED")
    for e in errors:
        print(f"- {e}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
