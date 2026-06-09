"""
Generate chapter-specific calibration for all topics from hierarchical index.

Why:
- Avoid sparse manual calibration that only covers a few chapters.
- Keep per-topic thresholds explicit and versioned.
- Add chapter-2 specialization for image-heavy linked-list operations.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


OPERATION_KEYWORDS = {
    "ดำเนินการ",
    "การทำงาน",
    "เพิ่ม",
    "ลบ",
    "แทรก",
    "เรียง",
    "ค้นหา",
    "enqueue",
    "dequeue",
    "push",
    "pop",
    "insert",
    "delete",
    "remove",
    "sort",
    "search",
}

STRUCTURE_KEYWORDS = {
    "คิว",
    "สแตก",
    "ลิงค์ลิสต์",
    "ลิงก์ลิสต์",
    "อาร์เรย์",
    "ต้นไม้",
    "กราฟ",
    "queue",
    "stack",
    "linked",
    "list",
    "array",
    "tree",
    "graph",
}


def _default_by_level(level: int) -> dict:
    lv = int(level or 1)
    if lv <= 1:
        return {
            "top_k": 8,
            "candidate_k": 20,
            "topic_threshold": 0.16,
            "visual_score_threshold_scale": 1.0,
            "min_doc_count": 2,
            "min_grounded_docs": 1,
            "min_grounded_facts": 2,
            "min_step_coverage": 0.40,
            "selection_min_step_coverage": 0.70,
            "min_grounding_agreement": 0.20,
            "ground_top_n": 2,
        }
    if lv == 2:
        return {
            "top_k": 10,
            "candidate_k": 24,
            "topic_threshold": 0.14,
            "visual_score_threshold_scale": 0.92,
            "min_doc_count": 2,
            "min_grounded_docs": 1,
            "min_grounded_facts": 2,
            "min_step_coverage": 0.35,
            "selection_min_step_coverage": 0.65,
            "min_grounding_agreement": 0.18,
            "ground_top_n": 2,
        }
    return {
        "top_k": 12,
        "candidate_k": 28,
        "topic_threshold": 0.12,
        "visual_score_threshold_scale": 0.82,
        "min_doc_count": 1,
        "min_grounded_docs": 1,
        "min_grounded_facts": 2,
        "min_step_coverage": 0.30,
        "selection_min_step_coverage": 0.58,
        "min_grounding_agreement": 0.15,
        "ground_top_n": 3,
    }


def _contains_any(text: str, keywords: set[str]) -> bool:
    t = str(text or "").lower()
    return any(k in t for k in keywords)


def _apply_operation_profile(base: dict) -> dict:
    row = dict(base)
    row["top_k"] = max(int(row.get("top_k", 12)), 14)
    row["candidate_k"] = max(int(row.get("candidate_k", 28)), 32)
    row["topic_threshold"] = min(float(row.get("topic_threshold", 0.12)), 0.10)
    row["visual_score_threshold_scale"] = min(float(row.get("visual_score_threshold_scale", 0.82)), 0.72)
    row["min_doc_count"] = 1
    row["min_grounded_docs"] = 1
    row["min_grounded_facts"] = 2
    row["min_step_coverage"] = min(float(row.get("min_step_coverage", 0.30)), 0.25)
    row["selection_min_step_coverage"] = min(float(row.get("selection_min_step_coverage", 0.58)), 0.52)
    row["min_grounding_agreement"] = min(float(row.get("min_grounding_agreement", 0.15)), 0.14)
    row["ground_top_n"] = max(int(row.get("ground_top_n", 3)), 3)
    return row


def _apply_structure_profile(base: dict) -> dict:
    row = dict(base)
    row["top_k"] = max(int(row.get("top_k", 10)), 12)
    row["candidate_k"] = max(int(row.get("candidate_k", 24)), 28)
    row["visual_score_threshold_scale"] = min(float(row.get("visual_score_threshold_scale", 0.92)), 0.80)
    row["min_doc_count"] = min(int(row.get("min_doc_count", 2)), 1)
    row["ground_top_n"] = max(int(row.get("ground_top_n", 2)), 2)
    return row


def _apply_chapter2_profile(topic_id: str, topic_title: str, base: dict) -> dict:
    row = dict(base)
    tid = str(topic_id or "").strip()
    ttl = str(topic_title or "").lower()
    if not tid.startswith("2"):
        return row

    row["top_k"] = max(int(row.get("top_k", 12)), 14)
    row["candidate_k"] = max(int(row.get("candidate_k", 28)), 34)
    row["topic_threshold"] = min(float(row.get("topic_threshold", 0.12)), 0.10)
    row["visual_score_threshold_scale"] = min(float(row.get("visual_score_threshold_scale", 0.82)), 0.74)
    row["ground_top_n"] = max(int(row.get("ground_top_n", 3)), 3)
    row["min_doc_count"] = min(int(row.get("min_doc_count", 1)), 1)
    row["min_grounded_docs"] = min(int(row.get("min_grounded_docs", 1)), 1)

    is_linked_operation = tid.startswith("2.4.2") or ("ลิง" in ttl and "ทำงาน" in ttl)
    is_linked_structure = tid.startswith("2.4.1") or ("ลิง" in ttl and "โครงสร้าง" in ttl)
    if is_linked_operation:
        row["top_k"] = max(int(row.get("top_k", 14)), 16)
        row["candidate_k"] = max(int(row.get("candidate_k", 34)), 40)
        row["visual_score_threshold_scale"] = min(float(row.get("visual_score_threshold_scale", 0.74)), 0.68)
        row["min_step_coverage"] = min(float(row.get("min_step_coverage", 0.30)), 0.22)
        row["selection_min_step_coverage"] = min(float(row.get("selection_min_step_coverage", 0.58)), 0.48)
        row["min_grounding_agreement"] = min(float(row.get("min_grounding_agreement", 0.15)), 0.12)
        row["ground_top_n"] = max(int(row.get("ground_top_n", 3)), 4)
    elif is_linked_structure:
        row["visual_score_threshold_scale"] = min(float(row.get("visual_score_threshold_scale", 0.74)), 0.72)
        row["min_step_coverage"] = min(float(row.get("min_step_coverage", 0.30)), 0.26)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand chapter calibration to all topics.")
    ap.add_argument("--hierarchy-index", default="indexes/hierarchical/topic_hierarchy.json")
    ap.add_argument("--input-calibration", default="indexes/hierarchical/chapter_calibration.json")
    ap.add_argument("--output", default="indexes/hierarchical/chapter_calibration.json")
    args = ap.parse_args()

    hierarchy_path = Path(args.hierarchy_index)
    if not hierarchy_path.exists():
        raise FileNotFoundError(f"hierarchy not found: {hierarchy_path}")
    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    topics = hierarchy.get("topics", []) if isinstance(hierarchy, dict) else []

    existing = {}
    in_path = Path(args.input_calibration)
    if in_path.exists():
        try:
            payload = json.loads(in_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("topics"), dict):
                existing = payload.get("topics", {})
        except Exception:
            existing = {}

    out_topics = {}
    for t in topics:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("topic_id", "")).strip()
        if not tid:
            continue
        title = str(t.get("title", "")).strip()
        level = int(t.get("level", 1) or 1)
        row = _default_by_level(level)

        if _contains_any(title, OPERATION_KEYWORDS):
            row = _apply_operation_profile(row)
        elif _contains_any(title, STRUCTURE_KEYWORDS):
            row = _apply_structure_profile(row)
        row = _apply_chapter2_profile(tid, title, row)

        # Preserve manual overrides.
        if isinstance(existing.get(tid), dict):
            row.update(existing.get(tid, {}))
        out_topics[tid] = row

    out_payload = {
        "version": "1.2",
        "updated_at": datetime.now(timezone.utc).date().isoformat(),
        "policy": "chapter_specific_visual_calibration",
        "generated_from": {
            "hierarchy_index": str(hierarchy_path.as_posix()),
            "topic_count": len(out_topics),
        },
        "topics": out_topics,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(out_path.as_posix()),
                "topic_count": len(out_topics),
                "overrides_preserved": len([k for k in out_topics.keys() if k in existing]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

