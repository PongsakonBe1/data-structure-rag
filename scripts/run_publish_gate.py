"""
Run external publish-grade gate and emit a short PASS/FAIL report with reasons.

This script always executes evaluate_ocr_research_grade.py first, then summarizes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _failed_checks(checks: dict) -> list[dict]:
    out = []
    for k, v in (checks or {}).items():
        if not isinstance(v, dict):
            continue
        if bool(v.get("pass", False)):
            continue
        out.append(
            {
                "check": k,
                "value": v.get("value"),
                "threshold": v.get("threshold"),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run publish gate and generate short pass/fail report.")
    ap.add_argument("--markdown", default="final_extracted_text_only_structured_full.md")
    ap.add_argument("--gt-jsonl", default="eval/ocr_gt_first_mid_last_v3_structured_full.jsonl")
    ap.add_argument("--pdf", default="data/data_structure_data_ch1_to_ch5.pdf")
    ap.add_argument("--toc", default="list_hitachi.txt")
    ap.add_argument("--labels", default="eval/visual_grounding_human_labels_ch2_ch3_v2.jsonl")
    ap.add_argument("--strict-mode", choices=["internal", "external_publish_grade"], default="external_publish_grade")
    ap.add_argument("--min-manual-verified-samples", type=int, default=10)
    ap.add_argument("--cer-threshold", type=float, default=0.05)
    ap.add_argument("--wer-threshold", type=float, default=0.2)
    ap.add_argument("--eval-output", default="logs/ocr_research_grade_external_publish_latest.json")
    ap.add_argument("--output-json", default="logs/publish_gate_latest.json")
    ap.add_argument("--output-md", default="logs/publish_gate_latest.md")
    args = ap.parse_args()

    eval_cmd = [
        sys.executable,
        "scripts/evaluate_ocr_research_grade.py",
        "--pdf", str(args.pdf),
        "--markdown", str(args.markdown),
        "--toc", str(args.toc),
        "--labels", str(args.labels),
        "--gt-jsonl", str(args.gt_jsonl),
        "--strict-mode", str(args.strict_mode),
        "--min-manual-verified-samples", str(int(args.min_manual_verified_samples)),
        "--cer-threshold", str(float(args.cer_threshold)),
        "--wer-threshold", str(float(args.wer_threshold)),
        "--output", str(args.eval_output),
    ]
    _run(eval_cmd)

    eval_json = Path(args.eval_output)
    if not eval_json.exists():
        raise FileNotFoundError(f"missing evaluator output: {eval_json}")
    payload = json.loads(eval_json.read_text(encoding="utf-8"))

    overall = payload.get("overall", {})
    strict_policy = payload.get("strict_policy", {})
    checks = payload.get("checks", {})
    ocr = payload.get("ocr_intrinsic", {})

    operational_pass = bool(overall.get("operational_pass", False))
    strict_pass = bool(overall.get("strict_research_pass", False))
    gate_pass = operational_pass and strict_pass

    reasons: list[str] = []
    if not operational_pass:
        reasons.append("operational_checks_failed")
    if not strict_pass:
        reasons.append("strict_research_failed")
    if strict_policy.get("manual_verified_required", False) and (not strict_policy.get("gt_metrics_available", False)):
        reasons.append("missing_verified_gt_samples")
    if isinstance(ocr, dict) and ocr.get("reason"):
        reasons.append(str(ocr.get("reason")))

    failed = _failed_checks(checks)
    if failed:
        reasons.append("failed_checks:" + ",".join(str(x["check"]) for x in failed))

    result = {
        "status": "PASS" if gate_pass else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gate_pass": gate_pass,
        "operational_pass": operational_pass,
        "strict_research_pass": strict_pass,
        "strict_mode": overall.get("strict_mode", args.strict_mode),
        "manual_verified_required": bool(strict_policy.get("manual_verified_required", False)),
        "min_manual_verified_samples": strict_policy.get("min_manual_verified_samples"),
        "gt_metrics_available": bool(strict_policy.get("gt_metrics_available", False)),
        "ocr_intrinsic": {
            "available": bool(ocr.get("available", False)),
            "samples": ocr.get("samples"),
            "samples_used_rows": ocr.get("samples_used_rows"),
            "samples_total_rows": ocr.get("samples_total_rows"),
            "cer_mean": ocr.get("cer_mean"),
            "wer_mean": ocr.get("wer_mean"),
            "reason": ocr.get("reason"),
        },
        "reasons": reasons,
        "failed_checks": failed,
        "artifacts": {
            "evaluator_report": str(eval_json.as_posix()),
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Publish Gate: {result['status']}",
        "",
        f"- gate_pass: `{result['gate_pass']}`",
        f"- operational_pass: `{result['operational_pass']}`",
        f"- strict_research_pass: `{result['strict_research_pass']}`",
        f"- strict_mode: `{result['strict_mode']}`",
        f"- manual_verified_required: `{result['manual_verified_required']}`",
        f"- gt_metrics_available: `{result['gt_metrics_available']}`",
        f"- samples_used/total: `{result['ocr_intrinsic']['samples_used_rows']}/{result['ocr_intrinsic']['samples_total_rows']}`",
        f"- cer_mean: `{result['ocr_intrinsic']['cer_mean']}`",
        f"- wer_mean: `{result['ocr_intrinsic']['wer_mean']}`",
        "",
        "## Reasons",
    ]
    if reasons:
        lines.extend([f"- {r}" for r in reasons])
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Failed Checks")
    if failed:
        for row in failed:
            lines.append(f"- `{row['check']}`: value={row['value']} threshold={row['threshold']}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append(f"- evaluator_report: `{result['artifacts']['evaluator_report']}`")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved_json={out_json}")
    print(f"saved_md={out_md}")


if __name__ == "__main__":
    main()

