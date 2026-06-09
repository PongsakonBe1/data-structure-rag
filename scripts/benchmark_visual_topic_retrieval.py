"""
Benchmark visual retrieval with topic-F1 + retrieval hit@k for image-heavy queries.

Dataset format (JSONL):
{
  "query": "...",
  "require_structure": true,
  "expected_topic_id": "2.4.2",
  "expected_topic_ids": ["2.4.2"],  # optional
  "expected_pages": [25, 26],
  "expected_figure_refs": ["2.18", "2.21"]  # optional
}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _mean(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if isinstance(v, (int, float))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def normalize_fig_ref(text: str) -> str:
    t = str(text or "").strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    return m.group(1) if m else t


def topic_relaxed_match(pred: str, gold_ids: set[str]) -> bool:
    p = str(pred or "").strip()
    if not p or not gold_ids:
        return False
    if p in gold_ids:
        return True
    for g in gold_ids:
        if p.startswith(g + ".") or g.startswith(p + "."):
            return True
    return False


def run_query(
    *,
    query: str,
    require_structure: bool,
    endpoint: str,
    pages_jsonl: str,
    hierarchy_index: str,
    hard_negative_rules: str,
    top_k: int,
    candidate_k: int,
    use_vlm_rerank: bool,
    use_visual_grounding: bool,
    sparse_strategy: str,
    tmp_out: Path,
) -> dict:
    cmd = [
        PY,
        "scripts/retrieve_visual_hybrid.py",
        "--query",
        query,
        "--backend",
        "colpali_endpoint",
        "--colpali-endpoint-url",
        endpoint,
        "--pages-jsonl",
        pages_jsonl,
        "--hierarchy-index",
        hierarchy_index,
        "--hard-negative-rules",
        hard_negative_rules,
        "--top-k",
        str(max(1, int(top_k))),
        "--candidate-k",
        str(max(2, int(candidate_k))),
        "--sparse-strategy",
        str(sparse_strategy),
        "--output",
        str(tmp_out),
    ]
    if require_structure:
        cmd.append("--require-structure")
    if str(sparse_strategy).strip().lower() == "splade":
        cmd.append("--use-splade")
    if use_vlm_rerank:
        cmd.append("--use-vlm-rerank")
    if use_visual_grounding:
        cmd.append("--use-visual-grounding")

    env = dict(os.environ)
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
        return {"ok": False, "error": f"nonzero:{p.returncode}", "stderr": (p.stderr or "")[-1000:]}
    if not tmp_out.exists():
        return {"ok": False, "error": "missing_output"}
    try:
        payload = json.loads(tmp_out.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"invalid_json:{exc}"}
    return {"ok": True, "payload": payload}


def load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = str(obj.get("query", "")).strip()
            if not q:
                raise ValueError(f"line {i}: missing query")
            rows.append(obj)
    if not rows:
        raise ValueError("empty dataset")
    return rows


def eval_row(row: dict, payload: dict, tiny_area_threshold: float) -> dict:
    query = str(row.get("query", "")).strip()
    hits = payload.get("hits", []) if isinstance(payload, dict) else []
    task = payload.get("task_metrics", {}) if isinstance(payload, dict) else {}
    filter_trace = payload.get("filter_trace", {}) if isinstance(payload, dict) else {}
    candidate_quality = payload.get("candidate_quality", {}) if isinstance(payload, dict) else {}

    pred_topic = str((payload.get("topic_prediction", {}) or {}).get("topic_id", "")).strip()
    gold_ids = set()
    if isinstance(row.get("expected_topic_ids"), list):
        gold_ids.update(str(x).strip() for x in row.get("expected_topic_ids", []) if str(x).strip())
    if str(row.get("expected_topic_id", "")).strip():
        gold_ids.add(str(row.get("expected_topic_id", "")).strip())

    topic_match_strict = 1.0 if (pred_topic in gold_ids and pred_topic) else 0.0
    topic_match_relaxed = 1.0 if topic_relaxed_match(pred_topic, gold_ids) else 0.0

    expected_pages = {int(x) for x in (row.get("expected_pages", []) or []) if str(x).strip().isdigit()}
    hit_pages = {int(h.get("page", 0) or 0) for h in hits if str(h.get("page", "")).strip().isdigit()}
    retrieval_hit_at_k = 1.0 if (expected_pages and expected_pages.intersection(hit_pages)) else 0.0

    expected_figs = {normalize_fig_ref(x) for x in (row.get("expected_figure_refs", []) or []) if str(x).strip()}
    hit_figs = set()
    for h in hits:
        for fr in (h.get("figure_refs", []) or []):
            hit_figs.add(normalize_fig_ref(fr))
    figure_hit_at_k = 1.0 if (expected_figs and expected_figs.intersection(hit_figs)) else (None if not expected_figs else 0.0)

    region_hits = [h for h in hits if str(h.get("image_level", "")).strip().lower() == "region"]
    tiny_count = 0
    area_vals: list[float] = []
    for h in region_hits:
        area = (h.get("region_meta", {}) or {}).get("area_ratio")
        if isinstance(area, (int, float)):
            area_f = float(area)
            area_vals.append(area_f)
            if area_f < float(tiny_area_threshold):
                tiny_count += 1
    tiny_ratio = (tiny_count / len(region_hits)) if region_hits else 0.0
    avg_area_ratio = _mean(area_vals)
    fragmented_crop_risk = 1.0 if tiny_ratio >= 0.4 else 0.0
    initial_records = int(filter_trace.get("initial_records", 0) or 0)
    final_records = int(filter_trace.get("final_records", 0) or 0)
    filter_survival_ratio = (float(final_records) / float(initial_records)) if initial_records > 0 else None
    col_status = str(payload.get("colpali_status", "") or "").strip()
    col_detail = str(payload.get("colpali_status_detail", "") or "").strip()
    endpoint_nonempty = 1.0 if col_status in {"ok", "ok_partial"} else 0.0

    return {
        "query": query,
        "colpali_status": col_status,
        "colpali_status_detail": col_detail,
        "endpoint_nonempty": endpoint_nonempty,
        "filter_survival_ratio": round(float(filter_survival_ratio), 4) if isinstance(filter_survival_ratio, float) else None,
        "candidate_image_coverage": candidate_quality.get("image_base64_coverage"),
        "pred_topic_id": pred_topic,
        "gold_topic_ids": ",".join(sorted(gold_ids)),
        "topic_f1_strict": topic_match_strict,
        "topic_f1_relaxed": topic_match_relaxed,
        "retrieval_hit_at_k": retrieval_hit_at_k,
        "figure_hit_at_k": figure_hit_at_k,
        "region_hit_ratio_at_k": task.get("region_hit_ratio_at_k"),
        "figure_ref_hit_ratio_at_k": task.get("figure_ref_hit_ratio_at_k"),
        "operation_coverage_ratio_at_k": task.get("operation_coverage_ratio_at_k"),
        "small_region_ratio_at_k": task.get("small_region_ratio_at_k"),
        "crop_completeness_proxy": task.get("crop_completeness_proxy"),
        "region_quality_mean": task.get("region_quality_mean"),
        "tiny_region_ratio_local": round(float(tiny_ratio), 4),
        "avg_region_area_ratio_local": round(float(avg_area_ratio), 6) if isinstance(avg_area_ratio, float) else None,
        "fragmented_crop_risk": fragmented_crop_risk,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark visual retrieval topic-F1 and hit@k.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--pages-jsonl", default="indexes/colpali/pages.jsonl")
    ap.add_argument("--hierarchy-index", default="indexes/hierarchical/topic_hierarchy.json")
    ap.add_argument("--hard-negative-rules", default="indexes/hierarchical/hard_negative_rules.json")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--sparse-strategy", choices=["splade", "bm25"], default="splade")
    ap.add_argument("--use-vlm-rerank", action="store_true")
    ap.add_argument("--use-visual-grounding", action="store_true")
    ap.add_argument("--tiny-area-threshold", type=float, default=0.012)
    ap.add_argument("--output-json", default="logs/visual_topic_benchmark_latest.json")
    ap.add_argument("--output-csv", default="logs/visual_topic_benchmark_latest.csv")
    args = ap.parse_args()

    endpoint = str(args.endpoint or "").strip() or os.getenv("COLPALI_ENDPOINT_URL", "").strip()
    if not endpoint:
        raise SystemExit("missing endpoint")

    rows = load_dataset(Path(args.dataset))
    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict] = []
    for i, row in enumerate(rows, start=1):
        tmp = ROOT / "logs" / f"__visual_topic_bench_{i}.json"
        run = run_query(
            query=str(row.get("query", "")),
            require_structure=bool(row.get("require_structure", False)),
            endpoint=endpoint,
            pages_jsonl=str(args.pages_jsonl),
            hierarchy_index=str(args.hierarchy_index),
            hard_negative_rules=str(args.hard_negative_rules),
            top_k=int(args.top_k),
            candidate_k=int(args.candidate_k),
            use_vlm_rerank=bool(args.use_vlm_rerank),
            use_visual_grounding=bool(args.use_visual_grounding),
            sparse_strategy=str(args.sparse_strategy),
            tmp_out=tmp,
        )
        if not run.get("ok", False):
            result_rows.append({"query": row.get("query", ""), "error": run.get("error", "unknown")})
            continue
        result_rows.append(eval_row(row, run.get("payload", {}) or {}, float(args.tiny_area_threshold)))

    summary = {
        "queries": len(rows),
        "evaluated": len([r for r in result_rows if "error" not in r]),
        "endpoint_nonempty_rate": _mean([r.get("endpoint_nonempty") for r in result_rows]),
        "filter_survival_rate": _mean([r.get("filter_survival_ratio") for r in result_rows]),
        "candidate_image_coverage_mean": _mean([r.get("candidate_image_coverage") for r in result_rows]),
        "topic_f1_strict_mean": _mean([r.get("topic_f1_strict") for r in result_rows]),
        "topic_f1_relaxed_mean": _mean([r.get("topic_f1_relaxed") for r in result_rows]),
        "retrieval_hit_at_k_mean": _mean([r.get("retrieval_hit_at_k") for r in result_rows]),
        "figure_hit_at_k_mean": _mean([r.get("figure_hit_at_k") for r in result_rows]),
        "operation_coverage_ratio_at_k_mean": _mean([r.get("operation_coverage_ratio_at_k") for r in result_rows]),
        "small_region_ratio_at_k_mean": _mean([r.get("small_region_ratio_at_k") for r in result_rows]),
        "tiny_region_ratio_local_mean": _mean([r.get("tiny_region_ratio_local") for r in result_rows]),
        "fragmented_crop_risk_rate": _mean([r.get("fragmented_crop_risk") for r in result_rows]),
        "crop_completeness_proxy_mean": _mean([r.get("crop_completeness_proxy") for r in result_rows]),
    }

    payload = {
        "config": {
            "dataset": str(args.dataset),
            "endpoint": endpoint,
            "pages_jsonl": str(args.pages_jsonl),
            "hierarchy_index": str(args.hierarchy_index),
            "hard_negative_rules": str(args.hard_negative_rules),
            "top_k": int(args.top_k),
            "candidate_k": int(args.candidate_k),
            "sparse_strategy": str(args.sparse_strategy),
            "use_vlm_rerank": bool(args.use_vlm_rerank),
            "use_visual_grounding": bool(args.use_visual_grounding),
            "tiny_area_threshold": float(args.tiny_area_threshold),
        },
        "summary": summary,
        "results": result_rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = sorted({k for r in result_rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in result_rows:
            w.writerow(r)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved_json={out_json}")
    print(f"saved_csv={out_csv}")


if __name__ == "__main__":
    main()
