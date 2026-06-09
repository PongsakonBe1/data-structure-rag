#!/usr/bin/env python3
"""
One-command baseline validation for Visual RAG (BM25 + CoPali endpoint).

This script runs the full validation chain with stable defaults and writes
artifacts to the `*_latest` reports used by downstream gates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run(cmd: list[str], *, timeout: int = 0) -> dict:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout if timeout > 0 else None,
    )
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one-command baseline validation.")
    ap.add_argument("--endpoint", default="", help="HF Space endpoint (owner/space)")
    ap.add_argument("--sparse-strategy", choices=["bm25", "splade"], default="bm25")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--topic-dataset", default="eval/visual_task_dataset_top5.jsonl")
    ap.add_argument("--human-dataset", default="eval/visual_grounding_human_labels_ch2_ch3_v2.jsonl")
    ap.add_argument("--markdown", default="final_extracted_text_only_structured_full.md")
    ap.add_argument("--gt-jsonl", default="eval/ocr_gt_first_mid_last_v3_structured_full.jsonl")
    ap.add_argument("--pdf", default="data/data_structure_data_ch1_to_ch5.pdf")
    ap.add_argument("--toc", default="list_hitachi.txt")
    ap.add_argument("--strict-mode", choices=["internal", "external_publish_grade"], default="external_publish_grade")
    ap.add_argument("--skip-topic-regression", action="store_true")
    ap.add_argument("--min-topic-routing-accuracy", type=float, default=1.0)
    ap.add_argument("--min-topic-ambiguous-pass-rate", type=float, default=1.0)
    ap.add_argument("--output-json", default="logs/baseline_validate_latest.json")
    ap.add_argument("--output-md", default="docs/baseline_validate_latest.md")
    args = ap.parse_args()

    endpoint = str(args.endpoint or "").strip() or os.getenv("COLPALI_ENDPOINT_URL", "").strip()
    if not endpoint:
        raise SystemExit("missing endpoint: set --endpoint or COLPALI_ENDPOINT_URL")

    steps: list[dict] = []

    steps.append(
        _run(
            [
                PY,
                "scripts/check_visual_endpoint.py",
                "--endpoint",
                endpoint,
                "--require-phase2",
                "--benchmark-runs",
                "2",
            ],
            timeout=1800,
        )
    )

    steps.append(
        _run(
            [
                PY,
                "scripts/benchmark_visual_topic_retrieval.py",
                "--dataset",
                str(args.topic_dataset),
                "--endpoint",
                endpoint,
                "--pages-jsonl",
                "indexes/colpali/pages.jsonl",
                "--hierarchy-index",
                "indexes/hierarchical/topic_hierarchy.json",
                "--hard-negative-rules",
                "indexes/hierarchical/hard_negative_rules.json",
                "--top-k",
                str(int(args.top_k)),
                "--candidate-k",
                str(int(args.candidate_k)),
                "--sparse-strategy",
                str(args.sparse_strategy),
                "--output-json",
                "logs/visual_topic_benchmark_latest.json",
                "--output-csv",
                "logs/visual_topic_benchmark_latest.csv",
            ],
            timeout=2400,
        )
    )

    steps.append(
        _run(
            [
                PY,
                "scripts/benchmark_visual_task_metrics.py",
                "--dataset",
                str(args.topic_dataset),
                "--endpoint",
                endpoint,
                "--top-k",
                str(int(args.top_k)),
                "--candidate-k",
                str(int(args.candidate_k)),
                "--sparse-strategy",
                str(args.sparse_strategy),
                "--output-json",
                "logs/visual_task_metrics_latest.json",
                "--output-csv",
                "logs/visual_task_metrics_latest.csv",
            ],
            timeout=2400,
        )
    )

    steps.append(
        _run(
            [
                PY,
                "scripts/benchmark_visual_grounding_human.py",
                "--dataset",
                str(args.human_dataset),
                "--endpoint",
                endpoint,
                "--top-k",
                str(int(args.top_k)),
                "--candidate-k",
                str(int(args.candidate_k)),
                "--output-json",
                "logs/visual_grounding_human_benchmark_latest.json",
                "--output-csv",
                "logs/visual_grounding_human_benchmark_latest.csv",
            ],
            timeout=3600,
        )
    )

    steps.append(
        _run(
            [
                PY,
                "scripts/evaluate_ingest_error_budget.py",
                "--markdown",
                str(args.markdown),
                "--region-manifest",
                "logs/figure_regions_manifest_latest.json",
                "--output",
                "logs/ingest_error_assessment_latest.json",
            ],
            timeout=600,
        )
    )

    steps.append(
        _run(
            [
                PY,
                "scripts/evaluate_ocr_research_grade.py",
                "--pdf",
                str(args.pdf),
                "--markdown",
                str(args.markdown),
                "--toc",
                str(args.toc),
                "--labels",
                str(args.human_dataset),
                "--gt-jsonl",
                str(args.gt_jsonl),
                "--strict-mode",
                str(args.strict_mode),
                "--output",
                "logs/ocr_research_grade_latest.json",
            ],
            timeout=1800,
        )
    )

    steps.append(
        _run(
            [
                PY,
                "scripts/run_research_gate.py",
                "--ocr-report",
                "logs/ocr_research_grade_latest.json",
                "--retrieval-report",
                "logs/visual_topic_benchmark_latest.json",
                "--task-report",
                "logs/visual_task_metrics_latest.json",
                "--output-json",
                "logs/research_gate_latest.json",
                "--output-md",
                "docs/research_gate_latest.md",
            ],
            timeout=600,
        )
    )

    if not bool(args.skip_topic_regression):
        steps.append(
            _run(
                [
                    PY,
                    "scripts/regression_topic_routing.py",
                    "--from-timestamp",
                    "",
                    "--enforce",
                    "--min-new-accuracy",
                    str(float(args.min_topic_routing_accuracy)),
                    "--min-ambiguous-pass-rate",
                    str(float(args.min_topic_ambiguous_pass_rate)),
                ],
                timeout=600,
            )
        )

    failed_steps = [i for i, s in enumerate(steps, start=1) if int(s.get("returncode", 1)) != 0]
    ok = len(failed_steps) == 0

    topic = _load_json(Path("logs/visual_topic_benchmark_latest.json")).get("summary", {})
    human = _load_json(Path("logs/visual_grounding_human_benchmark_latest.json")).get("summary", {})
    ocr = _load_json(Path("logs/ocr_research_grade_latest.json")).get("overall", {})
    gate = _load_json(Path("logs/research_gate_latest.json"))
    topic_routing = _load_json(Path("logs/topic_routing_regression_summary_latest.json"))
    topic_routing_new_acc = topic_routing.get("new_accuracy")
    topic_routing_amb_rate = topic_routing.get("ambiguous_pass_rate")
    topic_routing_gate_ok = True
    if not bool(args.skip_topic_regression):
        topic_routing_gate_ok = bool(topic_routing) and (topic_routing_new_acc is not None) and (topic_routing_amb_rate is not None)
        if topic_routing_gate_ok:
            topic_routing_gate_ok = (
                float(topic_routing_new_acc) >= float(args.min_topic_routing_accuracy)
                and float(topic_routing_amb_rate) >= float(args.min_topic_ambiguous_pass_rate)
            )

    payload = {
        "ok": ok and bool(gate.get("ok", False)) and bool(topic_routing_gate_ok),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "endpoint": endpoint,
            "sparse_strategy": str(args.sparse_strategy),
            "top_k": int(args.top_k),
            "candidate_k": int(args.candidate_k),
            "strict_mode": str(args.strict_mode),
            "skip_topic_regression": bool(args.skip_topic_regression),
            "min_topic_routing_accuracy": float(args.min_topic_routing_accuracy),
            "min_topic_ambiguous_pass_rate": float(args.min_topic_ambiguous_pass_rate),
        },
        "failed_step_indexes": failed_steps,
        "summary": {
            "retrieval_hit_at_k_mean": topic.get("retrieval_hit_at_k_mean"),
            "figure_hit_at_k_mean": topic.get("figure_hit_at_k_mean"),
            "endpoint_nonempty_rate": topic.get("endpoint_nonempty_rate"),
            "human_pass_rate": human.get("pass_rate"),
            "ocr_operational_pass": ocr.get("operational_pass"),
            "ocr_strict_research_pass": ocr.get("strict_research_pass"),
            "research_gate_ok": gate.get("ok"),
            "topic_routing_new_accuracy": topic_routing_new_acc,
            "topic_routing_ambiguous_pass_rate": topic_routing_amb_rate,
            "topic_routing_gate_ok": topic_routing_gate_ok,
        },
        "artifacts": {
            "endpoint_healthcheck": "logs/visual_endpoint_healthcheck_latest.json",
            "topic_benchmark": "logs/visual_topic_benchmark_latest.json",
            "task_metrics": "logs/visual_task_metrics_latest.json",
            "human_benchmark": "logs/visual_grounding_human_benchmark_latest.json",
            "ingest_assessment": "logs/ingest_error_assessment_latest.json",
            "ocr_research_grade": "logs/ocr_research_grade_latest.json",
            "research_gate": "logs/research_gate_latest.json",
            "topic_routing_summary": "logs/topic_routing_regression_summary_latest.json",
            "topic_routing_dashboard": "logs/topic_routing_dashboard_latest.html",
            "topic_routing_expert_review_csv": "logs/topic_routing_expert_review_latest.csv",
        },
        "steps": steps,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Baseline Validate",
        "",
        f"- overall_ok: `{payload['ok']}`",
        f"- retrieval_hit_at_k_mean: `{payload['summary']['retrieval_hit_at_k_mean']}`",
        f"- figure_hit_at_k_mean: `{payload['summary']['figure_hit_at_k_mean']}`",
        f"- endpoint_nonempty_rate: `{payload['summary']['endpoint_nonempty_rate']}`",
        f"- human_pass_rate: `{payload['summary']['human_pass_rate']}`",
        f"- ocr_operational_pass: `{payload['summary']['ocr_operational_pass']}`",
        f"- ocr_strict_research_pass: `{payload['summary']['ocr_strict_research_pass']}`",
        f"- research_gate_ok: `{payload['summary']['research_gate_ok']}`",
        f"- topic_routing_new_accuracy: `{payload['summary']['topic_routing_new_accuracy']}`",
        f"- topic_routing_ambiguous_pass_rate: `{payload['summary']['topic_routing_ambiguous_pass_rate']}`",
        f"- topic_routing_gate_ok: `{payload['summary']['topic_routing_gate_ok']}`",
        "",
        "## Artifacts",
    ]
    for k, v in payload["artifacts"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Step Status")
    for i, s in enumerate(steps, start=1):
        lines.append(f"- step_{i}: returncode={s['returncode']}")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"saved_json={out_json}")
    print(f"saved_md={out_md}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
