"""
Evaluate data readiness for ColPali-based retrieval.

Checks:
- ingest error budget pass
- region coverage on targeted pages
- corpus records count
- optional smoke query hit quality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def top_has_structure_hits(search_payload: dict, top_n: int = 3) -> float:
    hits = list((search_payload or {}).get("hits", []))[: max(1, int(top_n))]
    if not hits:
        return 0.0
    ok = sum(1 for h in hits if bool(h.get("has_structure", False)))
    return ok / len(hits)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ColPali data readiness")
    ap.add_argument("--ingest-assessment", default="logs/ingest_error_assessment_latest.json")
    ap.add_argument("--region-manifest", default="logs/figure_regions_manifest_latest.json")
    ap.add_argument("--prep-report", default="logs/colpali_prep_report_latest.json")
    ap.add_argument("--search-linked", default="logs/colpali_search_linked_list.json")
    ap.add_argument("--search-queue-array", default="logs/colpali_search_queue_array.json")
    ap.add_argument("--search-circular-queue", default="logs/colpali_search_circular_queue.json")
    ap.add_argument("--output", default="logs/colpali_data_readiness_latest.json")
    args = ap.parse_args()

    ingest = load_json(Path(args.ingest_assessment))
    region = load_json(Path(args.region_manifest))
    prep = load_json(Path(args.prep_report))

    q_linked = load_json(Path(args.search_linked))
    q_queue = load_json(Path(args.search_queue_array))
    q_circular = load_json(Path(args.search_circular_queue))

    ingest_pass = bool(ingest.get("overall_pass", False))

    region_summary = region.get("summary", {}) if isinstance(region, dict) else {}
    targeted = int(region_summary.get("targeted_pages", 0) or 0)
    pages_with_regions = int(region_summary.get("pages_with_regions", 0) or 0)
    fallback_pages = int(region_summary.get("fallback_pages", 0) or 0)
    region_coverage = pages_with_regions / max(1, targeted)
    fallback_ratio = fallback_pages / max(1, targeted)

    records_written = int(prep.get("records_written", 0) or 0)
    total_pages = int(prep.get("total_pages", 0) or 0)
    rows_per_page = records_written / max(1, total_pages)

    smoke = {
        "linked_list_top3_structure_ratio": round(top_has_structure_hits(q_linked, top_n=3), 6),
        "queue_array_top3_structure_ratio": round(top_has_structure_hits(q_queue, top_n=3), 6),
        "circular_queue_top3_structure_ratio": round(top_has_structure_hits(q_circular, top_n=3), 6),
    }

    checks = {
        "ingest_error_budget_pass": {
            "value": ingest_pass,
            "pass": ingest_pass,
        },
        "region_coverage": {
            "value": round(region_coverage, 6),
            "threshold": 0.8,
            "pass": region_coverage >= 0.8,
        },
        "fallback_ratio": {
            "value": round(fallback_ratio, 6),
            "threshold": 0.2,
            "pass": fallback_ratio <= 0.2,
        },
        "rows_per_page": {
            "value": round(rows_per_page, 6),
            "threshold": 1.2,
            "pass": rows_per_page >= 1.2,
        },
        "smoke_structure_precision": {
            "value": round(
                (smoke["linked_list_top3_structure_ratio"]
                 + smoke["queue_array_top3_structure_ratio"]
                 + smoke["circular_queue_top3_structure_ratio"]) / 3.0,
                6,
            ),
            "threshold": 0.75,
            "pass": (
                (smoke["linked_list_top3_structure_ratio"]
                 + smoke["queue_array_top3_structure_ratio"]
                 + smoke["circular_queue_top3_structure_ratio"]) / 3.0
            ) >= 0.75,
        },
    }

    overall_pass = all(bool(item.get("pass", False)) for item in checks.values())

    payload = {
        "overall_pass": overall_pass,
        "summary": {
            "targeted_pages": targeted,
            "pages_with_regions": pages_with_regions,
            "fallback_pages": fallback_pages,
            "records_written": records_written,
            "total_pages": total_pages,
        },
        "smoke": smoke,
        "checks": checks,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
