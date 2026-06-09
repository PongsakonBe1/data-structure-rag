#!/usr/bin/env python3
"""
Generate deterministic section_page_overrides.json from topic_hierarchy.json.

Goal:
- keep strict section filtering stable after TOC updates
- reduce noisy topic_to_pages ranges for leaf sections
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MANUAL_TOPIC_PAGE_OVERRIDES = {
    # Chapter 1 foundational definitions are concentrated around pages 1-3.
    "1.1.1": ["data_structure_data_ch1_to_ch5.pdf:1"],
    "1.1.2": ["data_structure_data_ch1_to_ch5.pdf:1", "data_structure_data_ch1_to_ch5.pdf:2"],
    "1.2.1": ["data_structure_data_ch1_to_ch5.pdf:3"],
    "1.2.2": ["data_structure_data_ch1_to_ch5.pdf:3"],
    "1.2.3": ["data_structure_data_ch1_to_ch5.pdf:3"],
    "1.2.4": ["data_structure_data_ch1_to_ch5.pdf:3"],
    "1.2.5": ["data_structure_data_ch1_to_ch5.pdf:3"],
    "1.2.6": ["data_structure_data_ch1_to_ch5.pdf:3"],
    # Complete binary tree content is in the mid-tree chapter, not appendix pages.
    "5.2.3": ["data_structure_data_ch1_to_ch5.pdf:55"],
}


def _split_page_ref(page_ref: str) -> tuple[str, int]:
    ref = str(page_ref or "").strip()
    if not ref:
        return "", 10**9
    if ":" in ref:
        src, pg = ref.rsplit(":", 1)
        try:
            return src.strip(), int(pg)
        except Exception:
            return src.strip(), 10**9
    try:
        return "", int(ref)
    except Exception:
        return "", 10**9


def _sort_page_refs(refs: set[str] | list[str]) -> list[str]:
    uniq = {str(x).strip() for x in refs if str(x).strip()}
    return sorted(uniq, key=lambda x: _split_page_ref(x))


def _expand_neighbors(seed_pages: set[str], candidate_pages: set[str], max_gap: int = 1, cap: int = 4) -> list[str]:
    if not seed_pages:
        return []
    out = set(seed_pages)
    seeds = [_split_page_ref(x) for x in seed_pages]
    for ref in candidate_pages:
        src, pg = _split_page_ref(ref)
        for s_src, s_pg in seeds:
            if src == s_src and abs(pg - s_pg) <= int(max_gap):
                out.add(ref)
                break
    return _sort_page_refs(out)[: max(1, int(cap))]


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync section page overrides from hierarchy metadata.")
    ap.add_argument("--hierarchy", default="indexes/hierarchical/topic_hierarchy.json")
    ap.add_argument("--output", default="indexes/hierarchical/section_page_overrides.json")
    ap.add_argument("--max-pages-per-topic", type=int, default=4)
    args = ap.parse_args()

    hierarchy_path = Path(args.hierarchy)
    if not hierarchy_path.exists():
        raise FileNotFoundError(f"missing hierarchy: {hierarchy_path}")

    payload = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    topics = payload.get("topics", []) if isinstance(payload, dict) else []
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    topic_to_pages = payload.get("topic_to_pages", {}) if isinstance(payload, dict) else {}
    all_page_refs = {
        str(row.get("page_id", "")).strip()
        for row in pages
        if isinstance(row, dict) and str(row.get("page_id", "")).strip()
    }

    parent_ids = {str(t.get("parent_id", "")).strip() for t in topics if isinstance(t, dict)}
    topic_ids = [str(t.get("topic_id", "")).strip() for t in topics if isinstance(t, dict) and str(t.get("topic_id", "")).strip()]
    leaf_topics = [tid for tid in topic_ids if tid not in parent_ids and "." in tid]

    heading_map: dict[str, set[str]] = {}
    best_map: dict[str, set[str]] = {}
    top_map: dict[str, set[str]] = {}

    for row in pages:
        if not isinstance(row, dict):
            continue
        page_id = str(row.get("page_id", "")).strip()
        if not page_id:
            continue

        for tid in (row.get("heading_hints", []) or []):
            t = str(tid).strip()
            if t:
                heading_map.setdefault(t, set()).add(page_id)

        best = str(row.get("best_topic", "")).strip()
        if best:
            best_map.setdefault(best, set()).add(page_id)

        for node in (row.get("top_topics", []) or []):
            tid = str((node or {}).get("topic_id", "")).strip()
            if tid:
                top_map.setdefault(tid, set()).add(page_id)

    operation_heavy = {"2.4.2", "3.3.2", "3.3.3", "4.3.1", "4.3.2"}
    out_map: dict[str, list[str]] = {}

    for tid in leaf_topics:
        heading_pages = set(heading_map.get(tid, set()))
        best_pages = set(best_map.get(tid, set()))
        top_pages = set(top_map.get(tid, set()))
        fallback_pages = {str(x).strip() for x in (topic_to_pages.get(tid, []) or []) if str(x).strip()}

        if heading_pages:
            selected = _expand_neighbors(
                heading_pages,
                candidate_pages=(best_pages | top_pages | fallback_pages),
                max_gap=1,
                cap=max(2, int(args.max_pages_per_topic)),
            )
        elif best_pages:
            selected = _sort_page_refs(best_pages)[: max(1, int(args.max_pages_per_topic))]
        elif top_pages:
            selected = _sort_page_refs(top_pages)[: max(1, int(args.max_pages_per_topic))]
        else:
            selected = _sort_page_refs(fallback_pages)[: max(1, int(args.max_pages_per_topic))]

        if tid in operation_heavy and selected:
            expanded = set(selected)
            anchors = [_split_page_ref(x) for x in selected]
            if anchors:
                # Expand forward to capture adjacent operation steps in sequential pages.
                anchor_src, anchor_pg = sorted(anchors, key=lambda x: (x[0], x[1]))[0]
                for k in range(0, 4):
                    ref = f"{anchor_src}:{anchor_pg + k}"
                    if ref in all_page_refs:
                        expanded.add(ref)
            selected = _sort_page_refs(expanded)[: max(4, int(args.max_pages_per_topic))]

        if not selected and "." in tid:
            parent = tid.rsplit(".", 1)[0]
            parent_pages = {str(x).strip() for x in (topic_to_pages.get(parent, []) or []) if str(x).strip()}
            if parent_pages:
                selected = _sort_page_refs(parent_pages)[: max(1, int(args.max_pages_per_topic))]

        if selected:
            out_map[tid] = selected

    # Hard pin curated overrides last so auto-heuristics cannot drift these topics.
    for tid, refs in MANUAL_TOPIC_PAGE_OVERRIDES.items():
        cleaned = _sort_page_refs([str(x).strip() for x in refs if str(x).strip()])
        if cleaned:
            out_map[str(tid).strip()] = cleaned

    out_payload = {
        "topic_to_pages": out_map,
        "summary": {
            "leaf_topics_total": len(leaf_topics),
            "topics_with_overrides": len(out_map),
            "max_pages_per_topic": int(args.max_pages_per_topic),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(out_payload["summary"], ensure_ascii=False, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
