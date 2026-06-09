#!/usr/bin/env python3
"""
Aggregate research-grade gates from existing benchmark reports.

This script is intentionally strict: it reports PASS only when all selected
gates satisfy thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _as_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def main() -> int:
    ap = argparse.ArgumentParser(description="Run strict research gate summary.")
    ap.add_argument("--ocr-report", default="logs/ocr_research_grade_structured_full_with_gt_v3_latest.json")
    ap.add_argument("--retrieval-report", default="logs/visual_topic_benchmark_hotfix_latest.json")
    ap.add_argument("--task-report", default="logs/visual_task_metrics_hotfix_latest.json")
    ap.add_argument("--output-json", default="logs/research_gate_latest.json")
    ap.add_argument("--output-md", default="docs/research_gate_latest.md")
    ap.add_argument("--min-topic-f1", type=float, default=0.95)
    ap.add_argument("--min-hit-at-k", type=float, default=0.95)
    ap.add_argument("--min-figure-hit-at-k", type=float, default=0.90)
    ap.add_argument("--min-endpoint-nonempty-rate", type=float, default=0.95)
    ap.add_argument("--min-candidate-image-coverage", type=float, default=0.90)
    ap.add_argument("--min-filter-survival-rate", type=float, default=0.01)
    ap.add_argument("--min-page-recall-at-k", type=float, default=0.95)
    ap.add_argument("--min-operation-coverage", type=float, default=0.40)
    args = ap.parse_args()

    ocr = _load_json(Path(args.ocr_report))
    ret = _load_json(Path(args.retrieval_report))
    task = _load_json(Path(args.task_report))

    checks = []

    ocr_overall = ocr.get("overall", {}) if isinstance(ocr.get("overall", {}), dict) else {}
    ocr_strict = ocr.get("strict_research_pass", ocr_overall.get("strict_research_pass"))
    ocr_oper = ocr.get("operational_pass", ocr_overall.get("operational_pass"))
    ocr_pass = _as_bool(ocr_strict, False) or _as_bool(ocr_oper, False)
    checks.append(
        {
            "name": "ocr_research_gate",
            "pass": bool(ocr_pass),
            "value": {
                "strict_research_pass": ocr_strict,
                "operational_pass": ocr_oper,
            },
            "threshold": "strict_research_pass=true (or operational_pass=true)",
        }
    )

    ret_summary = ret.get("summary", {}) if isinstance(ret.get("summary", {}), dict) else {}
    topic_f1 = _as_float(ret_summary.get("topic_f1_strict_mean"), 0.0)
    hit_k = _as_float(ret_summary.get("retrieval_hit_at_k_mean"), 0.0)
    fig_hit = _as_float(ret_summary.get("figure_hit_at_k_mean"), 0.0)
    endpoint_nonempty = _as_float(ret_summary.get("endpoint_nonempty_rate"), 0.0)
    candidate_cov = _as_float(ret_summary.get("candidate_image_coverage_mean"), 0.0)
    filter_survival = _as_float(ret_summary.get("filter_survival_rate"), 0.0)
    checks.append(
        {
            "name": "retrieval_topic_f1",
            "pass": bool(topic_f1 >= float(args.min_topic_f1)),
            "value": round(topic_f1, 6),
            "threshold": f">={float(args.min_topic_f1):.2f}",
        }
    )
    checks.append(
        {
            "name": "retrieval_hit_at_k",
            "pass": bool(hit_k >= float(args.min_hit_at_k)),
            "value": round(hit_k, 6),
            "threshold": f">={float(args.min_hit_at_k):.2f}",
        }
    )
    checks.append(
        {
            "name": "figure_hit_at_k",
            "pass": bool(fig_hit >= float(args.min_figure_hit_at_k)),
            "value": round(fig_hit, 6),
            "threshold": f">={float(args.min_figure_hit_at_k):.2f}",
        }
    )
    checks.append(
        {
            "name": "endpoint_nonempty_rate",
            "pass": bool(endpoint_nonempty >= float(args.min_endpoint_nonempty_rate)),
            "value": round(endpoint_nonempty, 6),
            "threshold": f">={float(args.min_endpoint_nonempty_rate):.2f}",
        }
    )
    checks.append(
        {
            "name": "candidate_image_coverage_mean",
            "pass": bool(candidate_cov >= float(args.min_candidate_image_coverage)),
            "value": round(candidate_cov, 6),
            "threshold": f">={float(args.min_candidate_image_coverage):.2f}",
        }
    )
    checks.append(
        {
            "name": "filter_survival_rate",
            "pass": bool(filter_survival >= float(args.min_filter_survival_rate)),
            "value": round(filter_survival, 6),
            "threshold": f">={float(args.min_filter_survival_rate):.2f}",
        }
    )

    task_summary = task.get("summary", {}) if isinstance(task.get("summary", {}), dict) else {}
    page_recall = _as_float(task_summary.get("page_recall_at_k_mean"), 0.0)
    op_cov = _as_float(task_summary.get("operation_coverage_ratio_at_k_mean"), 0.0)
    checks.append(
        {
            "name": "task_page_recall_at_k",
            "pass": bool(page_recall >= float(args.min_page_recall_at_k)),
            "value": round(page_recall, 6),
            "threshold": f">={float(args.min_page_recall_at_k):.2f}",
        }
    )
    checks.append(
        {
            "name": "task_operation_coverage",
            "pass": bool(op_cov >= float(args.min_operation_coverage)),
            "value": round(op_cov, 6),
            "threshold": f">={float(args.min_operation_coverage):.2f}",
        }
    )

    failed = [c["name"] for c in checks if not bool(c.get("pass", False))]
    result = {
        "ok": len(failed) == 0,
        "failed_checks": failed,
        "checks": checks,
        "inputs": {
            "ocr_report": args.ocr_report,
            "retrieval_report": args.retrieval_report,
            "task_report": args.task_report,
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Research Gate Summary",
        "",
        f"- Overall: {'PASS' if result['ok'] else 'FAIL'}",
        f"- Failed checks: {', '.join(failed) if failed else '-'}",
        "",
        "| Check | Pass | Value | Threshold |",
        "| --- | --- | --- | --- |",
    ]
    for c in checks:
        md_lines.append(f"| {c['name']} | {c['pass']} | {c['value']} | {c['threshold']} |")
    md = "\n".join(md_lines) + "\n"

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved_json={out_json}")
    print(f"saved_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
