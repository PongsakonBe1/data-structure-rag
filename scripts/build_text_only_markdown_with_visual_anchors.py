"""
Build a text-only markdown corpus with deterministic visual anchors.

Goal:
- Keep OCR narrative text
- Remove markdown image tags
- Attach region/page anchors per page with explicit fallback reason codes
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
STRUCTURE_BLOCK_BODY_RE = re.compile(
    r"(?ms)^\s*>\s*\[Structure:\s*([^\]\n]+)\]\s*\n(?:\s*>\s*-\s*[^\n]*\n?)*"
)
IMAGE_TAG_RE = re.compile(r"!\[[^\]]*?\]\([^)]*?\)")
FIGURE_REF_RE = re.compile(r"(ภาพที่|ตารางที่|figure|table)\s*\d+(?:\.\d+)?", re.IGNORECASE)


def parse_pages(md_text: str) -> list[dict[str, Any]]:
    headers = list(PAGE_HEADER_RE.finditer(md_text))
    pages: list[dict[str, Any]] = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        body = md_text[start:end].strip()
        pages.append(
            {
                "source": m.group(1).strip(),
                "page": int(m.group(2)),
                "body": body,
            }
        )
    return pages


def clean_ocr_text(text: str) -> tuple[str, dict[str, int]]:
    raw = str(text or "")
    s = raw
    before_len = len(s)
    structure_blocks = len(STRUCTURE_BLOCK_BODY_RE.findall(s))
    image_tags = len(IMAGE_TAG_RE.findall(s))
    s = STRUCTURE_BLOCK_BODY_RE.sub("", s)
    s = IMAGE_TAG_RE.sub("", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    stats = {
        "chars_before": before_len,
        "chars_after": len(s),
        "structure_blocks_removed": structure_blocks,
        "image_tags_removed": image_tags,
    }
    return s, stats


def load_page_figure_map(path: Path) -> dict[tuple[str, int], list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_name = Path(str(payload.get("pdf", ""))).name
    out: dict[tuple[str, int], list[str]] = {}
    for item in payload.get("images", []):
        if not isinstance(item, dict):
            continue
        page = int(item.get("page", 0) or 0)
        img_path = str(item.get("path", "")).strip()
        if page < 1 or not img_path:
            continue
        out.setdefault((source_name, page), []).append(img_path)
    return out


def load_page_region_manifest(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for page_item in payload.get("pages", []):
        if not isinstance(page_item, dict):
            continue
        source = str(page_item.get("source", "")).strip()
        page = int(page_item.get("page", 0) or 0)
        if not source or page < 1:
            continue
        out[(source, page)] = page_item
    return out


def _sorted_regions(region_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(r: dict[str, Any]):
        bbox = r.get("bbox", {}) if isinstance(r.get("bbox"), dict) else {}
        y0 = int(bbox.get("y0", 10**9) or 10**9)
        x0 = int(bbox.get("x0", 10**9) or 10**9)
        ridx = int(r.get("region_index", 10**9) or 10**9)
        score = float(r.get("score", 0.0) or 0.0)
        return (y0, x0, ridx, -score)

    return sorted(region_rows, key=_key)


def build_visual_anchor_block(
    *,
    source: str,
    page: int,
    target_page: bool,
    target_reason: list[str],
    region_rows: list[dict[str, Any]],
    figure_rows: list[str],
    max_items: int,
) -> tuple[str, str]:
    anchors: list[str] = []
    mode = "none"

    sorted_regions = _sorted_regions(region_rows)[:max_items]
    for i, r in enumerate(sorted_regions, start=1):
        bbox = r.get("bbox", {}) if isinstance(r.get("bbox", {}), dict) else {}
        method = str(r.get("region_detection_method", "")).strip()
        nearest_text_span = str(r.get("nearest_text_span", "")).strip()
        box_text = ""
        if bbox:
            box_text = f" bbox=({bbox.get('x0')},{bbox.get('y0')},{bbox.get('x1')},{bbox.get('y1')})"
        tail = f" method={method}" if method else ""
        if nearest_text_span:
            tail += f" nearest_text_span={json.dumps(nearest_text_span, ensure_ascii=False)}"
        anchors.append(f"- [VA-R{i}] level=region path={str(r.get('path', '')).strip()}{box_text}{tail}")
    if anchors:
        mode = "region"

    if not anchors:
        reason = "target_no_region" if target_page else "no_region_manifest"
        for i, p in enumerate(figure_rows[:max_items], start=1):
            anchors.append(f"- [VA-P{i}] level=page path={str(p).strip()} reason={reason}")
        if anchors:
            mode = "page"

    if not anchors:
        reason = "target_no_visual_asset" if target_page else "no_visual_asset"
        anchors.append(f"- [VA-N1] level=none reason={reason}")
        mode = "none"

    reason_text = ",".join([str(x).strip() for x in target_reason if str(x).strip()])
    header = [
        "### Visual Anchors (Auto)",
        f"source={source}, page={page}",
        f"target_page={str(bool(target_page)).lower()}",
        f"target_reason={reason_text if reason_text else '-'}",
    ]
    return "\n\n" + "\n".join(header + anchors) + "\n", mode


def main() -> None:
    ap = argparse.ArgumentParser(description="Build text-only markdown with visual anchors.")
    ap.add_argument("--input-markdown", default="final_extracted_content.md")
    ap.add_argument("--figure-manifest", default="logs/figure_manifest_latest.json")
    ap.add_argument("--region-manifest", default="logs/figure_regions_manifest_latest.json")
    ap.add_argument("--output-markdown", default="final_extracted_text_only.md")
    ap.add_argument("--report", default="logs/text_only_markdown_report_latest.json")
    ap.add_argument("--max-anchors-per-page", type=int, default=6)
    args = ap.parse_args()

    in_md = Path(args.input_markdown)
    if not in_md.exists():
        raise FileNotFoundError(f"input markdown not found: {in_md}")

    fig_map = load_page_figure_map(Path(args.figure_manifest))
    region_manifest_map = load_page_region_manifest(Path(args.region_manifest))
    pages = parse_pages(in_md.read_text(encoding="utf-8"))

    out_lines: list[str] = []
    stats = {
        "pages_total": 0,
        "pages_with_region_anchors": 0,
        "pages_with_page_anchors": 0,
        "pages_with_none_anchors": 0,
        "pages_with_any_anchor": 0,
        "target_pages": 0,
        "target_pages_with_any_anchor": 0,
        "structure_blocks_removed": 0,
        "image_tags_removed": 0,
        "figure_ref_pages": 0,
    }

    for p in pages:
        src = str(p.get("source", "")).strip()
        page = int(p.get("page", 0) or 0)
        body = str(p.get("body", ""))
        if not src or page < 1:
            continue
        stats["pages_total"] += 1
        if FIGURE_REF_RE.search(body):
            stats["figure_ref_pages"] += 1

        cleaned, row_stats = clean_ocr_text(body)
        stats["structure_blocks_removed"] += int(row_stats.get("structure_blocks_removed", 0) or 0)
        stats["image_tags_removed"] += int(row_stats.get("image_tags_removed", 0) or 0)

        key = (src, page)
        region_page = region_manifest_map.get(key, {})
        region_rows = (
            region_page.get("regions", [])
            if isinstance(region_page, dict) and isinstance(region_page.get("regions", []), list)
            else []
        )
        page_rows = fig_map.get(key, [])
        target_page = bool(region_page.get("target_page", False)) if isinstance(region_page, dict) else False
        target_reason = region_page.get("target_reason", []) if isinstance(region_page, dict) else []
        if not isinstance(target_reason, list):
            target_reason = []
        if target_page:
            stats["target_pages"] += 1

        anchor_block, mode = build_visual_anchor_block(
            source=src,
            page=page,
            target_page=target_page,
            target_reason=target_reason,
            region_rows=region_rows,
            figure_rows=page_rows,
            max_items=max(1, int(args.max_anchors_per_page)),
        )
        if mode == "region":
            stats["pages_with_region_anchors"] += 1
        elif mode == "page":
            stats["pages_with_page_anchors"] += 1
        else:
            stats["pages_with_none_anchors"] += 1
        if mode in {"region", "page", "none"}:
            stats["pages_with_any_anchor"] += 1
        if target_page and mode in {"region", "page", "none"}:
            stats["target_pages_with_any_anchor"] += 1

        out_lines.append(f"# Source: {src}\n## Page {page}\n\n{cleaned}{anchor_block}".rstrip() + "\n")

    out_md = Path(args.output_markdown)
    out_md.write_text("\n".join(out_lines).strip() + "\n", encoding="utf-8")

    report = {
        **stats,
        "anchor_coverage_ratio": round(stats["pages_with_any_anchor"] / max(1, stats["pages_total"]), 6),
        "target_anchor_coverage_ratio": round(
            stats["target_pages_with_any_anchor"] / max(1, stats["target_pages"]), 6
        ),
        "output_markdown": str(out_md.as_posix()),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved_markdown={out_md}")
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    main()
