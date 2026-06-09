"""
Generate SHA-256 manifest for index files under ./indexes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from index_integrity import build_index_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build index checksum manifest.")
    parser.add_argument("--index-dir", default="indexes", help="Path to index directory.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional custom manifest output path. Default: <index-dir>/index_manifest.json",
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else None

    manifest = build_index_manifest(index_dir=index_dir, manifest_path=manifest_path)
    out_path = manifest_path or (index_dir / "index_manifest.json")
    print(f"Wrote manifest: {out_path}")
    print(f"Files covered: {len(manifest.get('files', {}))}")


if __name__ == "__main__":
    main()
