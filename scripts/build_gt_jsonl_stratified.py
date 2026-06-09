"""
Build a stratified OCR GT JSONL from markdown pages (first/middle/last slices).

Default output is a bootstrap GT file that should be human-verified later.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
ANCHOR_BLOCK_RE = re.compile(r"\n### Visual Anchors \(Auto\)[\s\S]*$", re.MULTILINE)


def parse_pages(md_text: str) -> list[dict]:
    rows = []
    headers = list(PAGE_HEADER_RE.finditer(md_text))
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        rows.append(
            {
                "source": m.group(1).strip(),
                "page": int(m.group(2)),
                "body": md_text[start:end].strip(),
            }
        )
    return rows


def clean_page_text(body: str) -> str:
    s = ANCHOR_BLOCK_RE.sub("", body or "").strip()
    lines = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            continue
        if t.startswith("- [VA-"):
            continue
        t = t.lstrip("#").strip()
        lines.append(t)
    out = " ".join(lines)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def choose_pages(total_pages: int, per_slice: int) -> list[int]:
    k = max(1, int(per_slice))
    if total_pages <= (k * 3):
        return list(range(1, total_pages + 1))

    first = list(range(1, k + 1))
    last = list(range(total_pages - k + 1, total_pages + 1))

    mid_center = (total_pages + 1) // 2
    mid_start = max(1, mid_center - (k // 2))
    mid_end = mid_start + k - 1
    if mid_end > total_pages:
        mid_end = total_pages
        mid_start = max(1, mid_end - k + 1)
    middle = list(range(mid_start, mid_end + 1))

    out = sorted(set(first + middle + last))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build stratified OCR GT JSONL.")
    ap.add_argument("--markdown", default="final_extracted_text_only.md")
    ap.add_argument("--pages-per-slice", type=int, default=6)
    ap.add_argument("--output", default="eval/ocr_gt_first_mid_last_v1.jsonl")
    ap.add_argument("--tag", default="bootstrap_assistant_v1")
    args = ap.parse_args()

    md_path = Path(args.markdown)
    if not md_path.exists():
        raise FileNotFoundError(f"missing markdown: {md_path}")

    pages = parse_pages(md_path.read_text(encoding="utf-8"))
    if not pages:
        raise ValueError("no pages found in markdown")

    total = max(p["page"] for p in pages)
    selected = set(choose_pages(total_pages=total, per_slice=args.pages_per_slice))

    page_map = {int(p["page"]): p for p in pages}
    rows = []
    for p in sorted(selected):
        row = page_map.get(p)
        if not row:
            continue
        gt_text = clean_page_text(str(row.get("body", "")))
        if not gt_text:
            continue
        rows.append(
            {
                "page": int(p),
                "source": str(row.get("source", "")),
                "split": (
                    "first" if p <= args.pages_per_slice else
                    ("last" if p > (total - args.pages_per_slice) else "middle")
                ),
                "text": gt_text,
                "char_len": len(gt_text),
                "token_len": len(gt_text.split()),
                "manual_verified": False,
                "gt_tag": args.tag,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"pages_total={total}")
    print(f"selected={len(rows)}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()

