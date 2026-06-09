"""
Mine hard-negative pairs for visual retrieval evaluation/training.

Definition used:
- Similar lexical/tag profile (high overlap)
- Different topic assignment (from hierarchical index)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalize(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[\[\]\(\)\{\},.:;!?/\\|\"'`~@#$%^&*+=<>]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(text: str) -> set[str]:
    return {t for t in normalize(text).split(" ") if t}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine hard-negative pairs for visual retrieval.")
    ap.add_argument("--pages-jsonl", default="indexes/colpali/pages.jsonl")
    ap.add_argument("--hierarchy-index", default="indexes/hierarchical/topic_hierarchy.json")
    ap.add_argument("--output", default="eval/visual_hard_negatives_latest.jsonl")
    ap.add_argument("--max-pairs", type=int, default=1200)
    ap.add_argument("--min-overlap", type=float, default=0.28)
    args = ap.parse_args()

    pages = load_jsonl(Path(args.pages_jsonl))
    hierarchy = json.loads(Path(args.hierarchy_index).read_text(encoding="utf-8"))

    page_to_topic = {}
    for p in hierarchy.get("pages", []):
        if not isinstance(p, dict):
            continue
        pid = str(p.get("page_id", "")).strip()
        topic = str(p.get("best_topic", "")).strip()
        if pid and topic:
            page_to_topic[pid] = topic

    enriched = []
    for r in pages:
        pid = str(r.get("page_id", "")).strip()
        txt = str(r.get("text", ""))
        tag_blob = " ".join(str(x) for x in (r.get("tags", []) or []))
        fig_blob = " ".join(str(x) for x in (r.get("figure_refs", []) or []))
        tk = tokens(f"{txt[:1200]} {tag_blob} {fig_blob}")
        if not tk:
            continue
        enriched.append(
            {
                "id": str(r.get("id", "")).strip(),
                "page_id": pid,
                "topic_id": page_to_topic.get(pid, ""),
                "source": str(r.get("source", "")).strip(),
                "page": int(r.get("page", 0) or 0),
                "tokens": tk,
                "text_preview": str(r.get("text", ""))[:220],
            }
        )

    pairs = []
    n = len(enriched)
    for i in range(n):
        a = enriched[i]
        if not a.get("topic_id"):
            continue
        for j in range(i + 1, n):
            b = enriched[j]
            if not b.get("topic_id"):
                continue
            if a["topic_id"] == b["topic_id"]:
                continue
            inter = len(a["tokens"].intersection(b["tokens"]))
            if inter <= 0:
                continue
            union = len(a["tokens"].union(b["tokens"]))
            ov = inter / max(1, union)
            if ov < float(args.min_overlap):
                continue
            pairs.append(
                {
                    "query_like_from": a["id"],
                    "positive_topic": a["topic_id"],
                    "hard_negative_id": b["id"],
                    "hard_negative_topic": b["topic_id"],
                    "overlap_jaccard": round(float(ov), 4),
                    "reason": "lexical_overlap_high_topic_mismatch",
                }
            )
            pairs.append(
                {
                    "query_like_from": b["id"],
                    "positive_topic": b["topic_id"],
                    "hard_negative_id": a["id"],
                    "hard_negative_topic": a["topic_id"],
                    "overlap_jaccard": round(float(ov), 4),
                    "reason": "lexical_overlap_high_topic_mismatch",
                }
            )

    pairs = sorted(pairs, key=lambda x: float(x.get("overlap_jaccard", 0.0)), reverse=True)
    pairs = pairs[: max(1, int(args.max_pairs))]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "pairs": len(pairs),
                "min_overlap": float(args.min_overlap),
                "output": str(out_path.as_posix()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()

