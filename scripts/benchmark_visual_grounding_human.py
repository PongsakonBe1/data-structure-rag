"""
Benchmark visual grounding against human-verified labels.

Dataset format (JSONL):
{
  "query": "...",
  "require_structure": true,
  "expected_pages": [24,25],
  "expected_figure_refs": ["2.12","2.13"],
  "expected_grounding_keywords_any": ["head","node","link"],
  "expected_grounding_keywords_all": ["head"],
  "min_grounded_docs": 1,
  "min_grounded_facts": 2,
  "min_step_coverage": 0.5,
  "annotator": "human_id",
  "note": "..."
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


def normalize(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_fig(text: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)", str(text or ""))
    return m.group(1) if m else normalize(text)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
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


def run_query(
    *,
    query: str,
    require_structure: bool,
    endpoint: str,
    top_k: int,
    candidate_k: int,
    use_vlm_rerank: bool,
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
        endpoint,
        "--top-k",
        str(max(1, int(top_k))),
        "--candidate-k",
        str(max(2, int(candidate_k))),
        "--use-visual-grounding",
        "--ground-top-n",
        "3",
        "--output",
        str(tmp_output),
    ]
    if require_structure:
        cmd.append("--require-structure")
    if use_vlm_rerank:
        cmd.append("--use-vlm-rerank")

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
    payload = json.loads(tmp_output.read_text(encoding="utf-8"))
    return {"ok": True, "payload": payload}


def eval_row(label: dict, payload: dict) -> dict:
    hits = payload.get("hits", []) if isinstance(payload, dict) else []
    grounding = payload.get("grounding", []) if isinstance(payload, dict) else []
    tm = payload.get("task_metrics", {}) if isinstance(payload, dict) else {}

    exp_pages = {int(x) for x in (label.get("expected_pages", []) or []) if str(x).strip().isdigit()}
    exp_figs = {norm_fig(x) for x in (label.get("expected_figure_refs", []) or []) if str(x).strip()}
    exp_any = [normalize(x) for x in (label.get("expected_grounding_keywords_any", []) or []) if str(x).strip()]
    exp_all = [normalize(x) for x in (label.get("expected_grounding_keywords_all", []) or []) if str(x).strip()]

    hit_pages = {int(h.get("page", 0) or 0) for h in hits if str(h.get("page", "")).strip().isdigit()}
    hit_figs = set()
    for h in hits:
        for fr in h.get("figure_refs", []) or []:
            hit_figs.add(norm_fig(fr))

    grounded_facts = []
    normalized_terms = []
    chapter_operation_tags = []
    for g in grounding:
        gp = g.get("grounding", {}) if isinstance(g, dict) else {}
        parsed = gp.get("parsed", {}) if isinstance(gp, dict) else {}
        facts = parsed.get("grounded_facts", []) if isinstance(parsed, dict) else []
        if isinstance(facts, list):
            grounded_facts.extend(normalize(f) for f in facts if str(f).strip())
        nterms = parsed.get("normalized_terms", []) if isinstance(parsed, dict) else []
        if isinstance(nterms, list):
            normalized_terms.extend(normalize(x) for x in nterms if str(x).strip())
        op_tags = parsed.get("chapter_operation_tags", []) if isinstance(parsed, dict) else []
        if isinstance(op_tags, list):
            chapter_operation_tags.extend(normalize(x) for x in op_tags if str(x).strip())
    grounded_text = " ".join(grounded_facts + normalized_terms + chapter_operation_tags)

    any_hit = sum(1 for kw in exp_any if kw and kw in grounded_text)
    all_ok = all((kw in grounded_text) for kw in exp_all) if exp_all else True
    any_recall = (any_hit / max(1, len(exp_any))) if exp_any else None

    grounded_doc_count = sum(
        1
        for g in grounding
        if isinstance(g, dict)
        and isinstance((g.get("grounding", {}) or {}).get("parsed"), dict)
        and len(((g.get("grounding", {}) or {}).get("parsed", {}) or {}).get("grounded_facts", []) or []) > 0
    )
    grounded_fact_count = len(grounded_facts)

    min_docs = int(label.get("min_grounded_docs", 1) or 1)
    min_facts = int(label.get("min_grounded_facts", 1) or 1)
    min_step_cov = float(label.get("min_step_coverage", 0.0) or 0.0)
    step_cov = tm.get("procedure_step_coverage_proxy")
    step_cov_ok = bool((not isinstance(step_cov, (int, float))) or float(step_cov) >= min_step_cov)

    page_hit = bool((not exp_pages) or bool(exp_pages.intersection(hit_pages)))
    fig_hit = bool((not exp_figs) or bool(exp_figs.intersection(hit_figs)))
    human_pass = bool(
        page_hit
        and fig_hit
        and grounded_doc_count >= min_docs
        and grounded_fact_count >= min_facts
        and all_ok
        and (any_recall is None or any_recall >= 0.5)
        and step_cov_ok
    )
    return {
        "query": str(label.get("query", "")),
        "annotator": str(label.get("annotator", "")),
        "label_id": str(label.get("label_id", "")),
        "chapter": infer_chapter(label),
        "topic_hint": str(label.get("topic_hint", "")),
        "page_hit": page_hit,
        "figure_hit": fig_hit,
        "grounded_doc_count": grounded_doc_count,
        "grounded_fact_count": grounded_fact_count,
        "normalized_term_count": len([x for x in normalized_terms if x]),
        "chapter_operation_tag_count": len([x for x in chapter_operation_tags if x]),
        "keyword_any_recall": round(float(any_recall), 4) if isinstance(any_recall, (int, float)) else None,
        "keyword_all_ok": all_ok,
        "step_coverage": step_cov,
        "human_verified_pass": human_pass,
        "backend": payload.get("backend", ""),
        "colpali_status": payload.get("colpali_status", ""),
    }


def safe_mean(vals: list[float | None]) -> float | None:
    nums = [float(v) for v in vals if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def infer_chapter(label: dict) -> str:
    chapter = str((label or {}).get("chapter", "") or "").strip()
    if chapter:
        return chapter
    hint = str((label or {}).get("topic_hint", "") or "").strip()
    m = re.match(r"^(\d+)", hint)
    return m.group(1) if m else "unknown"


def summarize_by_chapter(results: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for row in results:
        ch = str(row.get("chapter", "unknown") or "unknown")
        buckets.setdefault(ch, []).append(row)
    out = {}
    for ch, rows in sorted(buckets.items(), key=lambda kv: kv[0]):
        pass_count = sum(1 for r in rows if bool(r.get("human_verified_pass", False)))
        out[ch] = {
            "total": len(rows),
            "pass_count": pass_count,
            "pass_rate": round(pass_count / max(1, len(rows)), 4),
            "keyword_any_recall_mean": safe_mean([r.get("keyword_any_recall") for r in rows]),
            "step_coverage_mean": safe_mean([r.get("step_coverage") for r in rows]),
            "grounded_doc_count_mean": safe_mean([r.get("grounded_doc_count") for r in rows]),
            "grounded_fact_count_mean": safe_mean([r.get("grounded_fact_count") for r in rows]),
            "normalized_term_count_mean": safe_mean([r.get("normalized_term_count") for r in rows]),
            "chapter_operation_tag_count_mean": safe_mean([r.get("chapter_operation_tag_count") for r in rows]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark visual grounding with human-verified labels.")
    ap.add_argument("--dataset", default="eval/visual_grounding_human_labels_ch2_ch3_v2.jsonl")
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--candidate-k", type=int, default=28)
    ap.add_argument("--use-vlm-rerank", action="store_true")
    ap.add_argument("--output-json", default="logs/visual_grounding_human_benchmark_latest.json")
    ap.add_argument("--output-csv", default="logs/visual_grounding_human_benchmark_latest.csv")
    args = ap.parse_args()

    endpoint = str(args.endpoint or "").strip() or __import__("os").getenv("COLPALI_ENDPOINT_URL", "").strip()
    if not endpoint:
        raise SystemExit("missing --endpoint and COLPALI_ENDPOINT_URL")

    labels = load_jsonl(Path(args.dataset))
    results = []
    for i, row in enumerate(labels, start=1):
        tmp = Path("logs") / f"__visual_grounding_human_tmp_{i}.json"
        run = run_query(
            query=str(row.get("query", "")),
            require_structure=bool(row.get("require_structure", True)),
            endpoint=endpoint,
            top_k=int(args.top_k),
            candidate_k=int(args.candidate_k),
            use_vlm_rerank=bool(args.use_vlm_rerank),
            tmp_output=tmp,
        )
        if not run.get("ok", False):
            results.append({"query": row.get("query", ""), "human_verified_pass": False, "error": run.get("error", "unknown")})
            continue
        results.append(eval_row(row, run.get("payload", {}) or {}))

    pass_count = sum(1 for r in results if bool(r.get("human_verified_pass", False)))
    summary = {
        "total": len(labels),
        "evaluated": len([r for r in results if "error" not in r]),
        "pass_count": pass_count,
        "pass_rate": round(pass_count / max(1, len(labels)), 4),
        "keyword_any_recall_mean": safe_mean([r.get("keyword_any_recall") for r in results]),
        "step_coverage_mean": safe_mean([r.get("step_coverage") for r in results]),
        "grounded_doc_count_mean": safe_mean([r.get("grounded_doc_count") for r in results]),
        "grounded_fact_count_mean": safe_mean([r.get("grounded_fact_count") for r in results]),
        "normalized_term_count_mean": safe_mean([r.get("normalized_term_count") for r in results]),
        "chapter_operation_tag_count_mean": safe_mean([r.get("chapter_operation_tag_count") for r in results]),
        "by_chapter": summarize_by_chapter([r for r in results if "error" not in r]),
    }

    out_payload = {
        "config": {
            "endpoint": endpoint,
            "dataset": args.dataset,
            "top_k": int(args.top_k),
            "candidate_k": int(args.candidate_k),
            "use_vlm_rerank": bool(args.use_vlm_rerank),
        },
        "summary": summary,
        "results": results,
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = sorted({k for r in results for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(json.dumps(out_payload, ensure_ascii=False, indent=2))
    print(f"saved_json={out_json}")
    print(f"saved_csv={out_csv}")


if __name__ == "__main__":
    main()
