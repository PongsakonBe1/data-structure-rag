"""
Batch/manual toggling of manual_verified flag in GT JSONL.

Examples:
  python scripts/mark_gt_manual_verified.py --input eval/ocr_gt.jsonl --all --set true --in-place
  python scripts/mark_gt_manual_verified.py --input eval/ocr_gt.jsonl --pages 1,2,5-8 --set true --in-place
  python scripts/mark_gt_manual_verified.py --input eval/ocr_gt.jsonl --splits middle,last --set false --in-place
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_page_selector(spec: str) -> set[int]:
    out: set[int] = set()
    raw = (spec or "").strip()
    if not raw:
        return out
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        if "-" in t:
            a, b = t.split("-", 1)
            try:
                start = int(a.strip())
                end = int(b.strip())
            except Exception:
                continue
            lo, hi = (start, end) if start <= end else (end, start)
            for p in range(lo, hi + 1):
                out.add(p)
        else:
            try:
                out.add(int(t))
            except Exception:
                continue
    return out


def parse_splits(spec: str) -> set[str]:
    out: set[str] = set()
    raw = (spec or "").strip()
    if not raw:
        return out
    for token in raw.split(","):
        t = token.strip().lower()
        if t:
            out.add(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Set manual_verified in GT JSONL by page/split/batch.")
    ap.add_argument("--input", default="eval/ocr_gt_first_mid_last_v3_structured_full.jsonl")
    ap.add_argument("--output", default="")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--all", action="store_true", help="Apply to all rows.")
    ap.add_argument("--pages", default="", help="Comma list/range: 1,2,5-8")
    ap.add_argument("--splits", default="", help="Comma list: first,middle,last")
    ap.add_argument("--set", choices=["true", "false"], default="true")
    ap.add_argument("--only-unverified", action="store_true", help="Only update rows where manual_verified=false.")
    ap.add_argument("--annotator", default="human_reviewer")
    ap.add_argument("--report", default="logs/manual_verify_update_latest.json")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"missing input: {in_path}")

    selected_pages = parse_page_selector(args.pages)
    selected_splits = parse_splits(args.splits)
    use_all = bool(args.all)
    if (not use_all) and (not selected_pages) and (not selected_splits):
        raise ValueError("no selector provided. Use --all or --pages or --splits.")

    set_value = args.set == "true"
    now_iso = datetime.now(timezone.utc).isoformat()

    rows = []
    total = 0
    matched = 0
    changed = 0
    skipped_only_unverified = 0

    for ln in in_path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        total += 1
        obj = json.loads(s)
        page = int(obj.get("page", 0) or 0)
        split = str(obj.get("split", "") or "").strip().lower()

        is_match = (
            use_all
            or (page in selected_pages)
            or (split in selected_splits if selected_splits else False)
        )
        if is_match:
            matched += 1
            if args.only_unverified and bool(obj.get("manual_verified", False)):
                skipped_only_unverified += 1
            else:
                prev = bool(obj.get("manual_verified", False))
                if prev != set_value:
                    changed += 1
                obj["manual_verified"] = set_value
                obj["manual_verified_by"] = str(args.annotator)
                obj["manual_verified_at"] = now_iso
        rows.append(obj)

    if args.in_place:
        out_path = in_path
    else:
        out_path = Path(args.output) if str(args.output).strip() else in_path.with_name(
            in_path.stem + "_manual_marked.jsonl"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "input": str(in_path.as_posix()),
        "output": str(out_path.as_posix()),
        "set_value": set_value,
        "selector": {
            "all": use_all,
            "pages": sorted(selected_pages),
            "splits": sorted(selected_splits),
            "only_unverified": bool(args.only_unverified),
        },
        "counts": {
            "total_rows": total,
            "matched_rows": matched,
            "changed_rows": changed,
            "skipped_only_unverified_rows": skipped_only_unverified,
        },
        "annotator": str(args.annotator),
        "timestamp_utc": now_iso,
    }

    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={rep}")


if __name__ == "__main__":
    main()

