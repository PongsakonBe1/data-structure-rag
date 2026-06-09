"""
Update/append chapter-2 hard-negative rules for linked-list confusion pairs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


CHAPTER2_RULES = [
    {
        "name": "linkedlist_operation_not_definition_figures",
        "query_any": [
            "การทำงานของลิงค์ลิสต์แบบทิศทางเดียว",
            "การเพิ่มโหนด",
            "การลบโหนด",
            "การแทรกโหนด",
            "linked list operation",
        ],
        "negative_any": [
            "ความหมายของลิงค์ลิสต์",
            "โครงสร้างของลิงค์ลิสต์",
            "ภาพที่ 2.11",
            "ภาพที่ 2.12",
            "ภาพที่ 2.13",
            "ภาพที่ 2.14",
        ],
        "penalty": 0.16,
    },
    {
        "name": "linkedlist_structure_not_operation_steps",
        "query_any": [
            "โครงสร้างลิงค์ลิสต์แบบทิศทางเดียว",
            "head node structure",
            "โครงสร้างของลิงค์ลิสต์",
        ],
        "negative_any": [
            "ภาพที่ 2.21",
            "ภาพที่ 2.22",
            "ภาพที่ 2.23",
            "การแทรกโหนด",
            "การลบโหนด",
            "การเพิ่มโหนด",
        ],
        "penalty": 0.12,
    },
    {
        "name": "linkedlist_operation_not_queue_or_stack",
        "query_all": ["ลิงค์ลิสต์", "ทำงาน"],
        "negative_any": ["คิว", "queue", "สแตก", "stack", "ภาพที่ 3.", "ภาพที่ 4."],
        "penalty": 0.14,
    },
    {
        "name": "linkedlist_structure_not_queue_chapter",
        "query_all": ["ลิงค์ลิสต์", "โครงสร้าง"],
        "negative_any": ["คิว", "queue", "ภาพที่ 3.", "enqueue", "dequeue"],
        "penalty": 0.18,
    },
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Append chapter-2 hard-negative rules.")
    ap.add_argument("--input", default="indexes/hierarchical/hard_negative_rules.json")
    ap.add_argument("--output", default="indexes/hierarchical/hard_negative_rules.json")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    payload = {"version": "1.0", "updated_at": "", "rules": []}
    if in_path.exists():
        try:
            got = json.loads(in_path.read_text(encoding="utf-8"))
            if isinstance(got, dict):
                payload.update(got)
        except Exception:
            pass
    rules = payload.get("rules", [])
    if not isinstance(rules, list):
        rules = []

    existing = {str(r.get("name", "")).strip() for r in rules if isinstance(r, dict)}
    added = 0
    for rule in CHAPTER2_RULES:
        name = str(rule.get("name", "")).strip()
        if not name or name in existing:
            continue
        rules.append(rule)
        existing.add(name)
        added += 1

    payload["rules"] = rules
    payload["updated_at"] = datetime.now(timezone.utc).date().isoformat()
    payload["version"] = str(payload.get("version", "1.0"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "output": str(out_path.as_posix()),
                "total_rules": len(rules),
                "added_rules": added,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
