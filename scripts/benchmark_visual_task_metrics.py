"""
Benchmark visual retrieval with task-specific metrics for image-heavy documents.

Dataset format (JSONL, one object per line):
{
  "query": "...",
  "require_structure": true,
  "expected_pages": [31, 32],
  "expected_figure_refs": ["3.4", "3.5"],
  "expected_structure_labels": ["Queue"],
  "expected_tags": ["queue", "array"]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def normalize_figure_ref(text: str) -> str:
    t = str(text or "").strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if m:
        return m.group(1)
    t = re.sub(r"(ภาพที่|figure|table|ตารางที่)\s*", "", t, flags=re.IGNORECASE).strip()
    return t


def load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            q = str(item.get("query", "")).strip()
            if not q:
                raise ValueError(f"Line {line_no}: missing query")
            rows.append(item)
    if not rows:
        raise ValueError("Empty dataset")
    return rows


def run_query(
    *,
    query: str,
    require_structure: bool,
    endpoint_url: str,
    top_k: int,
    candidate_k: int,
    sparse_strategy: str,
    use_vlm_rerank: bool,
    use_grounding: bool,
    tmp_output: Path,
) -> dict:
    cmd = [
        PY,
        "scripts/retrieve_visual_hybrid.py",
        "--query",
        query,
        "--backend",
        "colpali_endpoint",
        "--colpali-endpoint-url",
        endpoint_url,
        "--top-k",
        str(max(1, int(top_k))),
        "--candidate-k",
        str(max(2, int(candidate_k))),
        "--sparse-strategy",
        str(sparse_strategy),
        "--output",
        str(tmp_output),
    ]
    if require_structure:
        cmd.append("--require-structure")
    if str(sparse_strategy).strip().lower() == "splade":
        cmd.append("--use-splade")
    if use_vlm_rerank:
        cmd.append("--use-vlm-rerank")
    if use_grounding:
        cmd.append("--use-visual-grounding")

    env = dict(**__import__("os").environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1200,
    )
    if p.returncode != 0:
        return {"ok": False, "error": f"subprocess_nonzero:{p.returncode}", "stderr": (p.stderr or "")[-1000:]}
    if not tmp_output.exists():
        return {"ok": False, "error": "missing_output_json"}
    try:
        payload = json.loads(tmp_output.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"parse_failed:{exc}"}
    return {"ok": True, "payload": payload}


def eval_row(row: dict, payload: dict) -> dict:
    hits = payload.get("hits", []) if isinstance(payload, dict) else []
    filter_trace = payload.get("filter_trace", {}) if isinstance(payload, dict) else {}
    candidate_quality = payload.get("candidate_quality", {}) if isinstance(payload, dict) else {}

    expected_pages = {int(x) for x in row.get("expected_pages", []) if str(x).strip().isdigit()}
    expected_figs = {normalize_figure_ref(x) for x in row.get("expected_figure_refs", []) if str(x).strip()}
    expected_struct = {str(x).strip().lower() for x in row.get("expected_structure_labels", []) if str(x).strip()}
    expected_tags = {str(x).strip().lower() for x in row.get("expected_tags", []) if str(x).strip()}

    hit_pages = {int(h.get("page", 0) or 0) for h in hits if str(h.get("page", "")).strip().isdigit()}
    hit_figs = set()
    for h in hits:
        for fr in h.get("figure_refs", []) or []:
            hit_figs.add(normalize_figure_ref(fr))

    page_recall = (1.0 if expected_pages.intersection(hit_pages) else 0.0) if expected_pages else None
    fig_recall = (1.0 if expected_figs.intersection(hit_figs) else 0.0) if expected_figs else None

    struct_hit = None
    if expected_struct:
        got = set()
        for h in hits:
            got.update(str(x).strip().lower() for x in h.get("structure_labels", []) if str(x).strip())
        struct_hit = 1.0 if expected_struct.intersection(got) else 0.0

    tag_hit = None
    if expected_tags:
        got_tags = set()
        for h in hits:
            got_tags.update(str(x).strip().lower() for x in h.get("tags", []) if str(x).strip())
        tag_hit = 1.0 if expected_tags.intersection(got_tags) else 0.0

    task_metrics = payload.get("task_metrics", {}) if isinstance(payload, dict) else {}
    initial_records = int(filter_trace.get("initial_records", 0) or 0)
    final_records = int(filter_trace.get("final_records", 0) or 0)
    filter_survival_ratio = (float(final_records) / float(initial_records)) if initial_records > 0 else None
    col_status = str(payload.get("colpali_status", "") or "").strip()
    endpoint_nonempty = 1.0 if col_status in {"ok", "ok_partial"} else 0.0
    return {
        "query": row.get("query", ""),
        "backend": payload.get("backend", ""),
        "colpali_status": col_status,
        "colpali_status_detail": payload.get("colpali_status_detail", ""),
        "endpoint_nonempty": endpoint_nonempty,
        "filter_survival_ratio": round(float(filter_survival_ratio), 4) if isinstance(filter_survival_ratio, float) else None,
        "candidate_image_coverage": candidate_quality.get("image_base64_coverage"),
        "count": int(payload.get("count", 0) or 0),
        "page_recall_at_k": page_recall,
        "figure_recall_at_k": fig_recall,
        "structure_hit_at_k": struct_hit,
        "tag_hit_at_k": tag_hit,
        "region_hit_ratio_at_k": task_metrics.get("region_hit_ratio_at_k"),
        "figure_ref_hit_ratio_at_k": task_metrics.get("figure_ref_hit_ratio_at_k"),
        "operation_coverage_ratio_at_k": task_metrics.get("operation_coverage_ratio_at_k"),
        "crop_completeness_proxy": task_metrics.get("crop_completeness_proxy"),
        "small_region_ratio_at_k": task_metrics.get("small_region_ratio_at_k"),
        "region_quality_mean": task_metrics.get("region_quality_mean"),
        "unique_figure_refs_at_k": task_metrics.get("unique_figure_refs_at_k"),
        "page_diversity_at_k": task_metrics.get("page_diversity_at_k"),
        "procedure_step_coverage_proxy": task_metrics.get("procedure_step_coverage_proxy"),
        "grounding_success_rate": task_metrics.get("grounding_success_rate"),
        "grounding_consistency_mean": task_metrics.get("grounding_consistency_mean"),
        "mean_final_score": task_metrics.get("mean_final_score"),
    }


def safe_mean(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark visual retrieval task metrics.")
    ap.add_argument("--dataset", required=True, help="JSONL dataset path")
    ap.add_argument("--endpoint", default="", help="ColPali HF Space endpoint (owner/space)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--candidate-k", type=int, default=16)
    ap.add_argument("--sparse-strategy", choices=["splade", "bm25"], default="splade")
    ap.add_argument("--use-vlm-rerank", action="store_true")
    ap.add_argument("--use-grounding", action="store_true")
    ap.add_argument("--output-json", default="logs/visual_task_metrics_latest.json")
    ap.add_argument("--output-csv", default="logs/visual_task_metrics_latest.csv")
    args = ap.parse_args()

    endpoint = str(args.endpoint or "").strip() or __import__("os").getenv("COLPALI_ENDPOINT_URL", "").strip()
    if not endpoint:
        raise SystemExit("missing --endpoint and COLPALI_ENDPOINT_URL")

    rows = load_dataset(Path(args.dataset))
    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, row in enumerate(rows, start=1):
        tmp = Path("logs") / f"__visual_task_eval_tmp_{i}.json"
        run = run_query(
            query=str(row.get("query", "")),
            require_structure=bool(row.get("require_structure", False)),
            endpoint_url=endpoint,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            sparse_strategy=str(args.sparse_strategy),
            use_vlm_rerank=bool(args.use_vlm_rerank),
            use_grounding=bool(args.use_grounding),
            tmp_output=tmp,
        )
        if not run.get("ok", False):
            results.append({"query": row.get("query", ""), "error": run.get("error", "unknown")})
            continue
        payload = run.get("payload", {}) or {}
        results.append(eval_row(row, payload))

    summary = {
        "queries": len(rows),
        "evaluated": len([r for r in results if "error" not in r]),
        "endpoint_nonempty_rate": safe_mean([r.get("endpoint_nonempty") for r in results]),
        "filter_survival_rate": safe_mean([r.get("filter_survival_ratio") for r in results]),
        "candidate_image_coverage_mean": safe_mean([r.get("candidate_image_coverage") for r in results]),
        "page_recall_at_k_mean": safe_mean([r.get("page_recall_at_k") for r in results]),
        "figure_recall_at_k_mean": safe_mean([r.get("figure_recall_at_k") for r in results]),
        "structure_hit_at_k_mean": safe_mean([r.get("structure_hit_at_k") for r in results]),
        "tag_hit_at_k_mean": safe_mean([r.get("tag_hit_at_k") for r in results]),
        "region_hit_ratio_at_k_mean": safe_mean([r.get("region_hit_ratio_at_k") for r in results]),
        "figure_ref_hit_ratio_at_k_mean": safe_mean([r.get("figure_ref_hit_ratio_at_k") for r in results]),
        "operation_coverage_ratio_at_k_mean": safe_mean([r.get("operation_coverage_ratio_at_k") for r in results]),
        "crop_completeness_proxy_mean": safe_mean([r.get("crop_completeness_proxy") for r in results]),
        "small_region_ratio_at_k_mean": safe_mean([r.get("small_region_ratio_at_k") for r in results]),
        "region_quality_mean": safe_mean([r.get("region_quality_mean") for r in results]),
        "unique_figure_refs_at_k_mean": safe_mean([r.get("unique_figure_refs_at_k") for r in results]),
        "page_diversity_at_k_mean": safe_mean([r.get("page_diversity_at_k") for r in results]),
        "procedure_step_coverage_proxy_mean": safe_mean([r.get("procedure_step_coverage_proxy") for r in results]),
        "grounding_success_rate_mean": safe_mean([r.get("grounding_success_rate") for r in results]),
        "grounding_consistency_mean": safe_mean([r.get("grounding_consistency_mean") for r in results]),
        "mean_final_score_mean": safe_mean([r.get("mean_final_score") for r in results]),
    }

    out_payload = {
        "config": {
            "endpoint": endpoint,
            "top_k": int(args.top_k),
            "candidate_k": int(args.candidate_k),
            "sparse_strategy": str(args.sparse_strategy),
            "use_vlm_rerank": bool(args.use_vlm_rerank),
            "use_grounding": bool(args.use_grounding),
        },
        "summary": summary,
        "results": results,
    }
    out_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = sorted({k for r in results for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(json.dumps(out_payload, ensure_ascii=False, indent=2))
    print(f"saved_json={out_json}")
    print(f"saved_csv={out_csv}")


if __name__ == "__main__":
    main()
