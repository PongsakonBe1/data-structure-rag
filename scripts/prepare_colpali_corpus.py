"""
Prepare ColPali corpus metadata from extracted markdown + image manifests.

This script outputs retrieval units in JSONL (1 row per image candidate):
- Region images (preferred, if region manifest exists)
- Page image fallback (if no region image exists)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
FIGURE_REF_RE = re.compile(r"(ภาพที่|ตารางที่|figure|table)\s*\d+(?:\.\d+)?", re.IGNORECASE)
STRUCTURE_RE = re.compile(r"\[Structure:\s*([^\]]+)\]", re.IGNORECASE)
EXAMPLE_RE = re.compile(r"(ตัวอย่างที่|ตัวอย่าง|example)", re.IGNORECASE)
VISUAL_CAPTION_BLOCK_RE = re.compile(r"\n### Visual Captions \(Auto\)\n[\s\S]*$", re.MULTILINE)

TAG_HINTS = [
    (re.compile(r"(array|อาร์เรย์|1 มิติ|2 มิติ|3 มิติ)", re.IGNORECASE), "array"),
    (re.compile(r"(stack|สแตก|push|pop)", re.IGNORECASE), "stack"),
    (re.compile(r"(queue|คิว|enqueue|dequeue)", re.IGNORECASE), "queue"),
    (re.compile(r"(linked list|ลิงก์ลิสต์|โหนด|node)", re.IGNORECASE), "linked_list"),
    (re.compile(r"(tree|ไบนารีทรี|binary tree|subtree)", re.IGNORECASE), "tree"),
    (re.compile(r"(graph|กราฟ)", re.IGNORECASE), "graph"),
    (re.compile(r"(flowchart|ผังงาน|decision|selection|sequence)", re.IGNORECASE), "flowchart"),
]


def parse_pages(md_text: str) -> list[dict]:
    headers = list(PAGE_HEADER_RE.finditer(md_text))
    pages = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        body = md_text[start:end].strip()
        pages.append({"source": m.group(1).strip(), "page": int(m.group(2)), "text": body})
    return pages


def load_page_image_map(path: Path) -> dict[tuple[str, int], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_name = Path(str(payload.get("pdf", ""))).name
    by_page: dict[tuple[str, int], list[str]] = {}
    for item in payload.get("images", []):
        if not isinstance(item, dict):
            continue
        page = int(item.get("page", 0) or 0)
        img_path = str(item.get("path", "")).strip()
        if page < 1 or not img_path:
            continue
        by_page.setdefault((source_name, page), []).append(img_path)
    return by_page


def load_region_map(path: Path | None) -> dict[tuple[str, int], list[dict]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_page: dict[tuple[str, int], list[dict]] = {}
    for page_item in payload.get("pages", []):
        if not isinstance(page_item, dict):
            continue
        source = str(page_item.get("source", "")).strip()
        page = int(page_item.get("page", 0) or 0)
        if not source or page < 1:
            continue
        key = (source, page)
        for region in page_item.get("regions", []):
            if not isinstance(region, dict):
                continue
            r_path = str(region.get("path", "")).strip()
            if not r_path:
                continue
            by_page.setdefault(key, []).append(region)
    return by_page


def compact_text(text: str, max_chars: int = 6000) -> str:
    compact = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "\n\n[...truncated for corpus prep]"


def derive_tags(text: str, structure_labels: list[str]) -> list[str]:
    tags = set()
    lower = text.lower()
    for lbl in structure_labels:
        norm = lbl.strip().lower().replace(" ", "_")
        if norm:
            tags.add(norm)
    for pattern, tag in TAG_HINTS:
        if pattern.search(lower):
            tags.add(tag)
    return sorted(tags)


def parse_figure_refs(text: str) -> list[str]:
    refs: list[str] = []
    for line in text.splitlines():
        for m in FIGURE_REF_RE.finditer(line):
            refs.append(m.group(0).strip())

    out = []
    seen = set()
    for r in refs:
        k = r.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def strip_visual_caption_block(text: str) -> str:
    return VISUAL_CAPTION_BLOCK_RE.sub("", str(text or "")).strip()


def load_hierarchy_metadata(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    if not path.exists():
        return {}, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    topics = payload.get("topics", []) if isinstance(payload, dict) else []
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    topic_map: dict[str, dict] = {}
    for t in topics if isinstance(topics, list) else []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("topic_id", "")).strip()
        if not tid:
            continue
        topic_map[tid] = t
    page_map: dict[str, dict] = {}
    for p in pages if isinstance(pages, list) else []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("page_id", "")).strip()
        if not pid:
            continue
        page_map[pid] = p
    return topic_map, page_map


def infer_hierarchy_fields(page_id: str, topic_map: dict[str, dict], page_map: dict[str, dict]) -> dict:
    item = page_map.get(page_id, {}) if isinstance(page_map, dict) else {}
    best_topic_id = str(item.get("best_topic", "")).strip() if isinstance(item, dict) else ""
    best_topic_title = str((topic_map.get(best_topic_id, {}) or {}).get("title", "")).strip() if best_topic_id else ""
    best_topic_path = [str(x).strip() for x in (item.get("best_topic_path", []) if isinstance(item, dict) else []) if str(x).strip()]
    top_topics = item.get("top_topics", []) if isinstance(item, dict) else []
    top_topic_ids = []
    top_topic_titles = []
    for t in top_topics if isinstance(top_topics, list) else []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("topic_id", "")).strip()
        if not tid:
            continue
        top_topic_ids.append(tid)
        title = str((topic_map.get(tid, {}) or {}).get("title", "")).strip()
        if title:
            top_topic_titles.append(title)
    if best_topic_id and best_topic_id not in top_topic_ids:
        top_topic_ids.insert(0, best_topic_id)
    if best_topic_title and best_topic_title not in top_topic_titles:
        top_topic_titles.insert(0, best_topic_title)

    chapter_id = ""
    chapter_title = ""
    section_id = best_topic_id
    section_title = best_topic_title
    if best_topic_id:
        chapter_id = best_topic_id.split(".", 1)[0].strip()
        chapter_title = str((topic_map.get(chapter_id, {}) or {}).get("title", "")).strip()
    elif top_topic_ids:
        chapter_id = top_topic_ids[0].split(".", 1)[0].strip()
        chapter_title = str((topic_map.get(chapter_id, {}) or {}).get("title", "")).strip()

    return {
        "best_topic_id": best_topic_id,
        "best_topic_title": best_topic_title,
        "best_topic_path": best_topic_path,
        "topic_ids": top_topic_ids,
        "topic_titles": top_topic_titles,
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
        "section_id": section_id,
        "section_title": section_title,
    }


def build_metadata_tags(*, fields: dict, tags: list[str], figure_refs: list[str]) -> list[str]:
    out = []
    chapter_id = str(fields.get("chapter_id", "")).strip()
    section_id = str(fields.get("section_id", "")).strip()
    best_topic_id = str(fields.get("best_topic_id", "")).strip()
    if chapter_id:
        out.append(f"chapter:{chapter_id}")
    if section_id:
        out.append(f"section:{section_id}")
    if best_topic_id:
        out.append(f"topic:{best_topic_id}")
    for tid in fields.get("topic_ids", []) or []:
        t = str(tid).strip()
        if t:
            out.append(f"topic_candidate:{t}")
    for tg in tags or []:
        t = str(tg).strip().lower()
        if t:
            out.append(f"tag:{t}")
    for fr in figure_refs or []:
        f = str(fr).strip().lower()
        if f:
            out.append(f"figure:{f}")
    dedup = []
    seen = set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare ColPali corpus metadata.")
    ap.add_argument("--markdown", default="final_extracted_text_only_structured_full.md")
    ap.add_argument("--figure-manifest", default="logs/figure_manifest_latest.json")
    ap.add_argument("--region-manifest", default="logs/figure_regions_manifest_latest.json")
    ap.add_argument("--output", default="indexes/colpali/pages.jsonl")
    ap.add_argument("--report", default="logs/colpali_prep_report_latest.json")
    ap.add_argument("--hierarchy-index", default="indexes/hierarchical/topic_hierarchy.json")
    ap.add_argument("--max-images-per-page", type=int, default=4)
    ap.add_argument("--max-text-chars", type=int, default=6000)
    ap.add_argument(
        "--include-visual-captions",
        action="store_true",
        help="Include `Visual Captions (Auto)` block in index text. Default is off to reduce caption noise.",
    )
    args = ap.parse_args()

    md_path = Path(args.markdown)
    fig_path = Path(args.figure_manifest)
    region_path = Path(args.region_manifest)
    hierarchy_path = Path(args.hierarchy_index)
    out_path = Path(args.output)
    report_path = Path(args.report)

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown not found: {md_path}")
    if not fig_path.exists():
        raise FileNotFoundError(f"Figure manifest not found: {fig_path}")

    pages = parse_pages(md_path.read_text(encoding="utf-8"))
    page_image_map = load_page_image_map(fig_path)
    region_map = load_region_map(region_path if region_path.exists() else None)
    topic_map, page_topic_map = load_hierarchy_metadata(hierarchy_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    pages_with_page_image = 0
    pages_with_regions = 0
    pages_with_structure = 0
    pages_with_figure_ref = 0
    pages_with_example = 0
    records_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            total_pages += 1
            source = p["source"]
            page = int(p["page"])
            text = p["text"]
            if not bool(args.include_visual_captions):
                text = strip_visual_caption_block(text)
            compact = compact_text(text, max_chars=max(500, int(args.max_text_chars)))

            structure_labels = [s.strip() for s in STRUCTURE_RE.findall(text) if s.strip()]
            figure_refs = parse_figure_refs(text)
            has_example = bool(EXAMPLE_RE.search(text))
            tags = derive_tags(text, structure_labels)

            if structure_labels:
                pages_with_structure += 1
            if figure_refs:
                pages_with_figure_ref += 1
            if has_example:
                pages_with_example += 1

            key = (source, page)
            page_id = f"{source}:{page}"
            hierarchy_fields = infer_hierarchy_fields(page_id, topic_map, page_topic_map)
            metadata_tags = build_metadata_tags(fields=hierarchy_fields, tags=tags, figure_refs=figure_refs)
            region_candidates = region_map.get(key, [])
            page_candidates = page_image_map.get(key, [])
            if page_candidates:
                pages_with_page_image += 1
            if region_candidates:
                pages_with_regions += 1

            image_units: list[dict] = []
            for ridx, region in enumerate(region_candidates[: max(1, int(args.max_images_per_page))], start=1):
                path = str(region.get("path", "")).strip()
                if path:
                    image_units.append(
                        {
                            "image_path": path,
                            "image_level": "region",
                            "region_index": ridx,
                            "region_score": float(region.get("score", 0.0) or 0.0),
                            "region_meta": {
                                "bbox": region.get("bbox", {}),
                                "bbox_norm": region.get("bbox_norm", {}),
                                "area_ratio": float(region.get("area_ratio", 0.0) or 0.0),
                                "ink_ratio": float(region.get("ink_ratio", 0.0) or 0.0),
                                "is_fallback": bool(region.get("is_fallback", False)),
                                "quality_flags": region.get("quality_flags", {}) if isinstance(region.get("quality_flags", {}), dict) else {},
                            },
                        }
                    )

            if not image_units and page_candidates:
                image_units.append(
                    {
                        "image_path": str(page_candidates[0]),
                        "image_level": "page",
                        "region_index": 0,
                        "region_score": 0.0,
                        "region_meta": {},
                    }
                )

            if not image_units:
                continue

            for unit in image_units:
                rec = {
                    "id": f"{source}:{page}:{unit['image_level']}:{unit['region_index']}",
                    "page_id": page_id,
                    "source": source,
                    "page": page,
                    "image_path": unit["image_path"],
                    "image_level": unit["image_level"],
                    "region_index": unit["region_index"],
                    "region_score": unit["region_score"],
                    "region_meta": unit["region_meta"],
                    "figure_refs": figure_refs,
                    "figure_refs_count": len(figure_refs),
                    "structure_labels": structure_labels,
                    "has_structure": bool(structure_labels),
                    "has_example": has_example,
                    "tags": tags,
                    "metadata_tags": metadata_tags,
                    "chapter_id": hierarchy_fields.get("chapter_id", ""),
                    "chapter_title": hierarchy_fields.get("chapter_title", ""),
                    "section_id": hierarchy_fields.get("section_id", ""),
                    "section_title": hierarchy_fields.get("section_title", ""),
                    "best_topic_id": hierarchy_fields.get("best_topic_id", ""),
                    "best_topic_title": hierarchy_fields.get("best_topic_title", ""),
                    "best_topic_path": hierarchy_fields.get("best_topic_path", []),
                    "topic_ids": hierarchy_fields.get("topic_ids", []),
                    "topic_titles": hierarchy_fields.get("topic_titles", []),
                    "text": compact,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records_written += 1

    report = {
        "total_pages": total_pages,
        "pages_with_page_image": pages_with_page_image,
        "pages_with_page_image_ratio": round(pages_with_page_image / total_pages if total_pages else 0.0, 6),
        "pages_with_regions": pages_with_regions,
        "pages_with_regions_ratio": round(pages_with_regions / total_pages if total_pages else 0.0, 6),
        "pages_with_structure": pages_with_structure,
        "pages_with_structure_ratio": round(pages_with_structure / total_pages if total_pages else 0.0, 6),
        "pages_with_figure_ref": pages_with_figure_ref,
        "pages_with_example": pages_with_example,
        "records_written": records_written,
        "output": str(out_path.as_posix()),
        "used_region_manifest": region_path.exists(),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved_jsonl={out_path}")
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    main()
