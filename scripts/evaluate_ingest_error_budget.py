"""
Evaluate ingest quality against a practical Visual-RAG error budget.

Inputs:
- logs/extraction_audit_latest.csv
- logs/ingest_quality_report.csv
- logs/figure_regions_manifest_latest.json
- logs/visual_evidence_sidecar_report_latest.json
- logs/visual_sequence_links_latest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


QA_PASS_RE = re.compile(r"qa_gate_pass:\s*(true|false)\s*$", re.IGNORECASE | re.MULTILINE)
QA_REASON_RE = re.compile(r'qa_gate_reason:\s*"([^"]*)"\s*$', re.IGNORECASE | re.MULTILINE)


def _to_int(v) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def _to_float(v) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _caption_injected_from_markdown(path: Path) -> tuple[float | None, int, int]:
    if not path.exists():
        return None, 0, 0
    text = path.read_text(encoding="utf-8", errors="replace")
    pass_flags = [m.group(1).strip().lower() for m in QA_PASS_RE.finditer(text)]
    reasons = [m.group(1).strip() for m in QA_REASON_RE.finditer(text)]
    total = min(len(pass_flags), len(reasons))
    if total <= 0:
        return None, 0, 0

    risky_reasons = {
        "fallback_from_text",
        "expanded_crop_fallback_from_text",
        "disabled",
        "visual_qa_low_confidence",
        "visual_qa_low_signal_count",
        "visual_qa_missing_structure_signal",
        "visual_qa_table_no_cells",
    }
    injected = 0
    for i in range(total):
        is_pass = pass_flags[i] == "true"
        reason = reasons[i].strip().lower()
        if (not is_pass) or (reason in risky_reasons):
            injected += 1

    return (injected / total) if total > 0 else None, total, injected


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ingest error budget.")
    ap.add_argument("--audit", default="logs/extraction_audit_latest.csv")
    ap.add_argument("--quality", default="logs/ingest_quality_report.csv")
    ap.add_argument("--region-manifest", default="logs/figure_regions_manifest_latest.json")
    ap.add_argument("--sidecar-report", default="logs/visual_evidence_sidecar_report_latest.json")
    ap.add_argument("--sequence-links", default="logs/visual_sequence_links_latest.json")
    ap.add_argument("--markdown", default="final_extracted_text_only_structured_full.md")
    ap.add_argument("--output", default="logs/ingest_error_assessment_latest.json")

    ap.add_argument("--max-issue-page-ratio", type=float, default=0.05)
    ap.add_argument("--min-figure-structure-coverage", type=float, default=0.90)
    ap.add_argument("--min-figure-ref-to-region-link-recall", type=float, default=0.90)
    ap.add_argument("--min-operation-pages-region-coverage", type=float, default=0.90)
    ap.add_argument("--min-table-cell-visual-coverage", type=float, default=0.85)
    ap.add_argument("--max-generic-caption-ratio", type=float, default=0.10)
    ap.add_argument("--min-sequence-stitch-coverage", type=float, default=0.80)
    ap.add_argument("--max-caption-injected-ratio", type=float, default=0.35)
    args = ap.parse_args()

    audit_path = Path(args.audit)
    quality_path = Path(args.quality)
    if not audit_path.exists() or not quality_path.exists():
        raise FileNotFoundError("Missing required CSV inputs")
    audit_rows = list(csv.DictReader(audit_path.open(encoding="utf-8")))
    q_rows = list(csv.DictReader(quality_path.open(encoding="utf-8")))
    if not audit_rows or not q_rows:
        raise ValueError("Missing or empty input CSV(s).")

    region_manifest = _safe_json(Path(args.region_manifest))
    sidecar_report = _safe_json(Path(args.sidecar_report))
    sequence_links = _safe_json(Path(args.sequence_links))

    pages_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for p in region_manifest.get("pages", []) if isinstance(region_manifest.get("pages", []), list) else []:
        if not isinstance(p, dict):
            continue
        source = str(p.get("source", "")).strip()
        page = int(p.get("page", 0) or 0)
        if source and page > 0:
            pages_by_key[(source, page)] = p

    total_pages = len(audit_rows)
    pages_with_issues = sum(1 for r in audit_rows if str(r.get("issues", "")).strip())
    issue_page_ratio = pages_with_issues / total_pages if total_pages else 1.0

    figure_pages = [r for r in audit_rows if _to_int(r.get("figure_refs")) > 0]
    figure_pages_count = len(figure_pages)
    figure_pages_with_structure = sum(1 for r in figure_pages if _to_int(r.get("structure_count")) > 0)
    figure_structure_coverage = figure_pages_with_structure / figure_pages_count if figure_pages_count else 1.0

    figure_link_hits = 0
    for r in figure_pages:
        source = str(r.get("source", "")).strip()
        page = _to_int(r.get("page"))
        page_row = pages_by_key.get((source, page), {})
        regions = page_row.get("regions", []) if isinstance(page_row, dict) else []
        if isinstance(regions, list) and len(regions) > 0:
            figure_link_hits += 1
    figure_ref_to_region_link_recall = figure_link_hits / figure_pages_count if figure_pages_count else 1.0

    operation_pages = 0
    operation_pages_with_regions = 0
    for p in pages_by_key.values():
        flags = p.get("linkage_flags", {}) if isinstance(p.get("linkage_flags", {}), dict) else {}
        op_hits = _to_int(flags.get("operation_keyword_hits"))
        target_reason = p.get("target_reason", []) if isinstance(p.get("target_reason", []), list) else []
        is_operation_page = bool(op_hits > 0 or ("operation_keyword" in target_reason))
        if is_operation_page:
            operation_pages += 1
            regions = p.get("regions", []) if isinstance(p.get("regions", []), list) else []
            if len(regions) > 0:
                operation_pages_with_regions += 1
    operation_pages_region_coverage = (
        operation_pages_with_regions / operation_pages if operation_pages > 0 else 1.0
    )

    total_structure = sum(_to_int(r.get("structure_count_raw")) for r in q_rows)
    total_caption_injected = sum(_to_int(r.get("structure_caption_injected")) for r in q_rows)
    caption_injected_ratio_legacy = total_caption_injected / total_structure if total_structure > 0 else 0.0

    md_ratio, md_total_caps, md_injected_caps = _caption_injected_from_markdown(Path(args.markdown))
    if md_ratio is not None:
        caption_injected_ratio = float(md_ratio)
        caption_metric_source = "caption_qa_markdown"
        caption_metric_total = int(md_total_caps)
        caption_metric_injected = int(md_injected_caps)
    else:
        caption_injected_ratio = float(caption_injected_ratio_legacy)
        caption_metric_source = "legacy_structure_csv"
        caption_metric_total = int(total_structure)
        caption_metric_injected = int(total_caption_injected)

    table_cell_visual_coverage = _to_float(sidecar_report.get("table_cell_visual_coverage_table_only"))
    if table_cell_visual_coverage is None:
        table_cell_visual_coverage = _to_float(sidecar_report.get("with_table_cell_visuals_ratio"))
    if table_cell_visual_coverage is None:
        table_cell_visual_coverage = 0.0
    generic_caption_ratio = _to_float(sidecar_report.get("generic_caption_ratio"))
    if generic_caption_ratio is None:
        generic_caption_ratio = 1.0
    sequence_stitch_coverage = _to_float(sidecar_report.get("with_sequence_group_ratio"))
    if sequence_stitch_coverage is None:
        seq_summary = sequence_links.get("summary", {}) if isinstance(sequence_links, dict) else {}
        cand = _to_float(seq_summary.get("sequence_candidates_pages"))
        if cand and cand > 0:
            sequence_stitch_coverage = min(1.0, float(seq_summary.get("records", 0) or 0) / cand)
        else:
            sequence_stitch_coverage = 0.0

    checks = {
        "issue_page_ratio": {
            "value": round(issue_page_ratio, 6),
            "threshold": args.max_issue_page_ratio,
            "pass": issue_page_ratio <= args.max_issue_page_ratio,
        },
        "figure_structure_coverage": {
            "value": round(figure_structure_coverage, 6),
            "threshold": args.min_figure_structure_coverage,
            "pass": figure_structure_coverage >= args.min_figure_structure_coverage,
        },
        "figure_ref_to_region_link_recall": {
            "value": round(figure_ref_to_region_link_recall, 6),
            "threshold": args.min_figure_ref_to_region_link_recall,
            "pass": figure_ref_to_region_link_recall >= args.min_figure_ref_to_region_link_recall,
        },
        "operation_pages_region_coverage": {
            "value": round(operation_pages_region_coverage, 6),
            "threshold": args.min_operation_pages_region_coverage,
            "pass": operation_pages_region_coverage >= args.min_operation_pages_region_coverage,
        },
        "table_cell_visual_coverage": {
            "value": round(table_cell_visual_coverage, 6),
            "threshold": args.min_table_cell_visual_coverage,
            "pass": table_cell_visual_coverage >= args.min_table_cell_visual_coverage,
        },
        "generic_caption_ratio": {
            "value": round(generic_caption_ratio, 6),
            "threshold": args.max_generic_caption_ratio,
            "pass": generic_caption_ratio <= args.max_generic_caption_ratio,
        },
        "sequence_stitch_coverage": {
            "value": round(sequence_stitch_coverage, 6),
            "threshold": args.min_sequence_stitch_coverage,
            "pass": sequence_stitch_coverage >= args.min_sequence_stitch_coverage,
        },
        "caption_injected_ratio": {
            "value": round(caption_injected_ratio, 6),
            "threshold": args.max_caption_injected_ratio,
            "pass": caption_injected_ratio <= args.max_caption_injected_ratio,
        },
    }
    core_check_names = [
        "issue_page_ratio",
        "figure_structure_coverage",
        "figure_ref_to_region_link_recall",
        "operation_pages_region_coverage",
        "table_cell_visual_coverage",
        "generic_caption_ratio",
        "sequence_stitch_coverage",
    ]
    overall_pass = all(bool((checks.get(k, {}) or {}).get("pass", False)) for k in core_check_names)

    payload = {
        "overall_pass": overall_pass,
        "summary": {
            "total_pages": total_pages,
            "pages_with_issues": pages_with_issues,
            "figure_pages": figure_pages_count,
            "figure_pages_with_structure": figure_pages_with_structure,
            "figure_pages_with_regions": figure_link_hits,
            "operation_pages": operation_pages,
            "operation_pages_with_regions": operation_pages_with_regions,
            "total_structure_blocks": total_structure,
            "total_caption_injected_blocks": total_caption_injected,
            "caption_metric_source": caption_metric_source,
            "caption_metric_total": caption_metric_total,
            "caption_metric_injected": caption_metric_injected,
            "caption_injected_ratio_legacy": round(caption_injected_ratio_legacy, 6),
            "sidecar_records": sidecar_report.get("total_records", 0),
            "core_check_count": len(core_check_names),
        },
        "checks": checks,
        "core_checks": core_check_names,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"overall_pass={overall_pass}")
    print(json.dumps(payload["checks"], ensure_ascii=False, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
