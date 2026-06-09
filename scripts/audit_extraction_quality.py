"""
Audit extracted markdown quality per PDF page.

Input format expects:
# Source: <pdf>
## Page <num>
<markdown body...>
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
FIGURE_REF_RE = re.compile(r"(ภาพที่|ตารางที่|figure|table)\s*\d+(?:\.\d+)?", re.IGNORECASE)
STRUCTURE_BLOCK_RE = re.compile(
    r"(?ms)^\s*>\s*\[Structure:\s*([^\]\n]+)\]\s*\n(?:\s*>\s*-\s*[^\n]*\n?)*"
)
CAPTION_STRUCTURE_HINT_RE = re.compile(
    r"(ลิงค์ลิสต์|linked list|ไบนารีทรี|binary tree|ต้นไม้|tree|กราฟ|graph|"
    r"สแตก|stack|คิว|queue|ผังงาน|flowchart|selection|decision|sequence|"
    r"อาร์เรย์|array|โหนด|node|ตาราง|table)",
    re.IGNORECASE,
)


def parse_pages(text: str) -> list[dict]:
    pages = []
    headers = list(PAGE_HEADER_RE.finditer(text))
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end]
        pages.append({"source": m.group(1).strip(), "page": int(m.group(2)), "body": body})
    return pages


def audit_page(page: dict) -> dict:
    body = page["body"]
    figure_refs = len(FIGURE_REF_RE.findall(body))
    caption_hint_count = len(CAPTION_STRUCTURE_HINT_RE.findall(body))
    blocks = list(STRUCTURE_BLOCK_RE.finditer(body))
    labels = [b.group(1).strip() for b in blocks]
    unknown_count = sum(1 for x in labels if x.lower() == "unknown")
    unique_blocks = len({b.group(0).strip() for b in blocks})
    repeated_count = max(0, len(blocks) - unique_blocks)
    max_allowed = 0 if figure_refs == 0 else min(12, figure_refs * 2 + 1)

    issues = []
    if figure_refs == 0 and len(blocks) > 0:
        issues.append("structure_without_figure")
    if len(blocks) > max_allowed:
        issues.append("excessive_structure_blocks")
    if unknown_count > 0:
        issues.append("contains_unknown_structure")
    if repeated_count > 0:
        issues.append("repeated_structure_blocks")
    if figure_refs > 0 and caption_hint_count > 0 and len(blocks) == 0:
        issues.append("caption_hints_without_structure")

    return {
        "source": page["source"],
        "page": page["page"],
        "figure_refs": figure_refs,
        "caption_hint_count": caption_hint_count,
        "structure_count": len(blocks),
        "unknown_count": unknown_count,
        "repeated_count": repeated_count,
        "max_allowed": max_allowed,
        "issues": "|".join(issues),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit extracted markdown quality.")
    ap.add_argument("--input", required=True, help="Path to extracted markdown.")
    ap.add_argument("--output", required=True, help="CSV output path.")
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    pages = parse_pages(text)
    rows = [audit_page(p) for p in pages]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "page",
                "figure_refs",
                "caption_hint_count",
                "structure_count",
                "unknown_count",
                "repeated_count",
                "max_allowed",
                "issues",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    bad = sum(1 for r in rows if r["issues"])
    print(f"pages={total}, pages_with_issues={bad}")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
