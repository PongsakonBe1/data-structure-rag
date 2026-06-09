"""
Run OCR re-ingest + visual caption/table/diagram extraction end-to-end,
then compare before/after metrics and execute retrieval smoke checks.

This script is the fastest way to run the full data-first stabilization loop.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(
    cmd: list[str],
    *,
    timeout: int = 0,
    allow_fail: bool = False,
    env_extra: dict[str, str] | None = None,
) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout if timeout > 0 else None,
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0 and not allow_fail:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return p.returncode, out


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summarize_metrics(
    *,
    ocr_report: dict[str, Any],
    topic_report: dict[str, Any],
    task_report: dict[str, Any],
) -> dict[str, Any]:
    topic_summary = topic_report.get("summary", {}) if isinstance(topic_report, dict) else {}
    task_summary = task_report.get("summary", {}) if isinstance(task_report, dict) else {}
    ocr_intrinsic = ocr_report.get("ocr_intrinsic", {}) if isinstance(ocr_report, dict) else {}
    overall = ocr_report.get("overall", {}) if isinstance(ocr_report, dict) else {}
    return {
        "strict_research_pass": bool(overall.get("strict_research_pass", False)),
        "operational_pass": bool(overall.get("operational_pass", False)),
        "cer_mean": ocr_intrinsic.get("cer_mean"),
        "wer_mean": ocr_intrinsic.get("wer_mean"),
        "figure_ref_recall": (ocr_report.get("figure_reference_eval", {}) or {}).get("recall"),
        "heading_soft_recall_effective": (ocr_report.get("toc_heading_effective", {}) or {}).get("recall"),
        "retrieval_hit_at_k_mean": topic_summary.get("retrieval_hit_at_k_mean"),
        "figure_hit_at_k_mean": topic_summary.get("figure_hit_at_k_mean"),
        "endpoint_nonempty_rate": topic_summary.get("endpoint_nonempty_rate"),
        "filter_survival_rate": topic_summary.get("filter_survival_rate"),
        "page_recall_at_k_mean": task_summary.get("page_recall_at_k_mean"),
        "figure_recall_at_k_mean": task_summary.get("figure_recall_at_k_mean"),
        "operation_coverage_ratio_at_k_mean": task_summary.get("operation_coverage_ratio_at_k_mean"),
    }


def diff_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = sorted(set(before.keys()) | set(after.keys()))
    for k in keys:
        b = before.get(k)
        a = after.get(k)
        delta = None
        try:
            if b is not None and a is not None:
                delta = round(float(a) - float(b), 6)
        except Exception:
            delta = None
        out[k] = {"before": b, "after": a, "delta_after_minus_before": delta}
    return out


def tokenize_query(text: str) -> list[str]:
    tokens = re.split(r"\s+", re.sub(r"[^\w\u0E00-\u0E7F]+", " ", (text or "").lower()))
    return [t for t in tokens if t and len(t) >= 2]


def smoke_map(payload: dict[str, Any], query: str) -> dict[str, Any]:
    hits = payload.get("hits", []) if isinstance(payload, dict) else []
    hits = hits if isinstance(hits, list) else []
    top_hits = hits[:3]
    q_tokens = tokenize_query(query)
    evidence_text = " ".join(
        str(h.get("preview", "") or "")
        + " "
        + str(h.get("best_topic_title", "") or "")
        + " "
        + " ".join(str(x) for x in (h.get("figure_refs", []) or []) if str(x).strip())
        for h in top_hits
        if isinstance(h, dict)
    ).lower()
    covered = sum(1 for t in q_tokens if t in evidence_text)
    coverage = round(covered / max(1, len(q_tokens)), 6)
    pred_topic = str((payload.get("topic_prediction", {}) or {}).get("topic_id", "")).strip()
    top_topics = [str(h.get("best_topic_id", "")).strip() for h in top_hits if isinstance(h, dict)]
    top_topics = [x for x in top_topics if x]
    pred_ch = pred_topic.split(".", 1)[0] if pred_topic else ""
    chapter_alignment = any(tt.startswith(pred_ch + ".") or tt == pred_ch for tt in top_topics) if pred_ch else False
    return {
        "query": query,
        "colpali_status": payload.get("colpali_status"),
        "colpali_status_detail": payload.get("colpali_status_detail"),
        "topic_id": pred_topic,
        "top_topic_ids": top_topics,
        "chapter_alignment": bool(chapter_alignment),
        "query_term_coverage_in_top_hits": coverage,
        "top_pages": [str(h.get("page_id", "")) for h in top_hits if isinstance(h, dict)],
        "top_fig_refs": [
            list(dict.fromkeys(str(x) for x in (h.get("figure_refs", []) or []) if str(x).strip()))
            for h in top_hits
            if isinstance(h, dict)
        ],
    }


def write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    before = report.get("before", {})
    after = report.get("after", {})
    diff = report.get("delta", {})
    smoke = report.get("smoke", [])
    lines: list[str] = []
    lines.append("# OCR Re-ingest Before/After Summary")
    lines.append("")
    lines.append("## Key Status")
    lines.append("")
    lines.append(f"- Before strict research pass: `{before.get('strict_research_pass')}`")
    lines.append(f"- After strict research pass: `{after.get('strict_research_pass')}`")
    lines.append(f"- Before retrieval_hit_at_k_mean: `{before.get('retrieval_hit_at_k_mean')}`")
    lines.append(f"- After retrieval_hit_at_k_mean: `{after.get('retrieval_hit_at_k_mean')}`")
    lines.append(f"- Before figure_hit_at_k_mean: `{before.get('figure_hit_at_k_mean')}`")
    lines.append(f"- After figure_hit_at_k_mean: `{after.get('figure_hit_at_k_mean')}`")
    lines.append("")
    lines.append("## Metric Delta")
    lines.append("")
    for k, row in sorted(diff.items()):
        lines.append(
            f"- `{k}`: before=`{row.get('before')}` | after=`{row.get('after')}` | delta=`{row.get('delta_after_minus_before')}`"
        )
    lines.append("")
    lines.append("## Smoke Retrieval Mapping")
    lines.append("")
    for item in smoke:
        lines.append(f"- Query: `{item.get('query')}`")
        lines.append(f"  - colpali_status: `{item.get('colpali_status')}` / `{item.get('colpali_status_detail')}`")
        lines.append(f"  - topic_id: `{item.get('topic_id')}`")
        lines.append(f"  - chapter_alignment: `{item.get('chapter_alignment')}`")
        lines.append(f"  - query_term_coverage_in_top_hits: `{item.get('query_term_coverage_in_top_hits')}`")
        lines.append(f"  - top_pages: `{item.get('top_pages')}`")
        lines.append(f"  - top_topic_ids: `{item.get('top_topic_ids')}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv(ROOT / ".env")

    ap = argparse.ArgumentParser(description="OCR re-ingest and full before/after validation.")
    ap.add_argument("--pdf", default="data/data_structure_data_ch1_to_ch5.pdf")
    ap.add_argument("--toc", default="list_hitachi.txt")
    ap.add_argument("--gt-jsonl", default="eval/ocr_gt_first_mid_last_v3_structured_full.jsonl")
    ap.add_argument("--dataset", default="eval/visual_task_dataset_top5.jsonl")
    ap.add_argument("--endpoint", default=(os.getenv("COLPALI_ENDPOINT_URL", "") or "").strip())
    ap.add_argument("--vision-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--caption-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--caption-workers", type=int, default=2)
    ap.add_argument("--caption-max-pages", type=int, default=120)
    ap.add_argument("--caption-max-images-per-page", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--candidate-k", type=int, default=16)
    ap.add_argument("--skip-rescan", action="store_true")
    ap.add_argument("--preprocess-only", action="store_true", help="Skip retrieval/topic/task benchmarks and smoke retrieval.")
    ap.add_argument("--output-json", default="logs/ocr_reingest_before_after_latest.json")
    ap.add_argument("--output-md", default="docs/ocr_reingest_before_after_latest.md")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    baseline_md = Path("final_extracted_text_only_structured_full.md")
    if not baseline_md.exists():
        baseline_md = Path("final_extracted_text_only.md")
    if not baseline_md.exists():
        baseline_md = Path("final_extracted_content.md")

    logs: list[dict[str, Any]] = []

    before_ocr = Path(f"logs/ocr_research_grade_before_reingest_{ts}.json")
    before_topic = Path(f"logs/visual_topic_benchmark_before_reingest_{ts}.json")
    before_task = Path(f"logs/visual_task_metrics_before_reingest_{ts}.json")

    code, out = run(
        [
            PY,
            "scripts/evaluate_ocr_research_grade.py",
            "--pdf",
            str(args.pdf),
            "--markdown",
            str(baseline_md),
            "--toc",
            str(args.toc),
            "--gt-jsonl",
            str(args.gt_jsonl),
            "--output",
            str(before_ocr),
        ],
        timeout=1200,
    )
    logs.append({"step": "before_ocr_eval", "code": code, "tail": out[-1200:]})

    if args.endpoint and (not bool(args.preprocess_only)):
        code, out = run(
            [
                PY,
                "scripts/benchmark_visual_topic_retrieval.py",
                "--dataset",
                str(args.dataset),
                "--endpoint",
                str(args.endpoint),
                "--sparse-strategy",
                "bm25",
                "--top-k",
                str(max(1, int(args.top_k))),
                "--candidate-k",
                str(max(1, int(args.candidate_k))),
                "--output-json",
                str(before_topic),
                "--output-csv",
                str(before_topic.with_suffix(".csv")),
            ],
            timeout=3600,
            allow_fail=True,
        )
        logs.append({"step": "before_topic_bench", "code": code, "tail": out[-1500:]})

        code, out = run(
            [
                PY,
                "scripts/benchmark_visual_task_metrics.py",
                "--dataset",
                str(args.dataset),
                "--endpoint",
                str(args.endpoint),
                "--sparse-strategy",
                "bm25",
                "--top-k",
                str(max(1, int(args.top_k))),
                "--candidate-k",
                str(max(1, int(args.candidate_k))),
                "--output-json",
                str(before_task),
                "--output-csv",
                str(before_task.with_suffix(".csv")),
            ],
            timeout=3600,
            allow_fail=True,
        )
        logs.append({"step": "before_task_bench", "code": code, "tail": out[-1500:]})
    elif bool(args.preprocess_only):
        logs.append({"step": "before_retrieval_bench", "code": -1, "tail": "skipped: --preprocess-only"})
    else:
        logs.append({"step": "before_retrieval_bench", "code": -1, "tail": "skipped: missing endpoint"})

    # Backup source markdown before re-ingest.
    backup_path = Path(f"logs/final_extracted_backup_before_reingest_{ts}.md")
    if Path("final_extracted_content.md").exists():
        shutil.copyfile("final_extracted_content.md", backup_path)
        logs.append({"step": "backup_final_extracted_content", "code": 0, "tail": str(backup_path)})

    if not bool(args.skip_rescan):
        code, out = run(
            [PY, "src/ingest.py"],
            timeout=10800,
            env_extra={"VISION_MODEL_ID": str(args.vision_model)},
        )
        logs.append({"step": "rescan_ingest", "code": code, "tail": out[-2000:]})
    else:
        logs.append({"step": "rescan_ingest", "code": -1, "tail": "skipped by --skip-rescan"})

    code, out = run(
        [
            PY,
            "scripts/extract_pdf_figures.py",
            "--pdf",
            str(args.pdf),
            "--output-dir",
            "assets/figures",
            "--manifest",
            "logs/figure_manifest_latest.json",
        ],
        timeout=1800,
    )
    logs.append({"step": "extract_pdf_figures", "code": code, "tail": out[-1200:]})

    code, out = run(
        [
            PY,
            "scripts/extract_structure_regions.py",
            "--markdown",
            "final_extracted_content.md",
            "--figure-manifest",
            "logs/figure_manifest_latest.json",
            "--output-dir",
            "assets/figure_regions",
            "--output-manifest",
            "logs/figure_regions_manifest_latest.json",
            "--target-only",
            "--fallback-when-empty",
        ],
        timeout=3600,
    )
    logs.append({"step": "extract_structure_regions", "code": code, "tail": out[-1500:]})

    code, out = run(
        [
            PY,
            "scripts/build_text_only_markdown_with_visual_anchors.py",
            "--input-markdown",
            "final_extracted_content.md",
            "--figure-manifest",
            "logs/figure_manifest_latest.json",
            "--region-manifest",
            "logs/figure_regions_manifest_latest.json",
            "--output-markdown",
            "final_extracted_text_only.md",
            "--report",
            "logs/text_only_markdown_report_latest.json",
        ],
        timeout=1800,
    )
    logs.append({"step": "build_text_only_markdown", "code": code, "tail": out[-1200:]})

    code, out = run(
        [
            PY,
            "scripts/enrich_visual_captions_markdown.py",
            "--input-markdown",
            "final_extracted_text_only.md",
            "--region-manifest",
            "logs/figure_regions_manifest_latest.json",
            "--output-markdown",
            "final_extracted_text_only_structured_full.md",
            "--cache-path",
            "logs/visual_caption_cache_latest.json",
            "--report",
            "logs/visual_caption_enrich_report_latest.json",
            "--model",
            str(args.caption_model),
            "--workers",
            str(max(1, int(args.caption_workers))),
            "--max-pages",
            str(max(1, int(args.caption_max_pages))),
            "--max-images-per-page",
            str(max(1, int(args.caption_max_images_per_page))),
        ],
        timeout=7200,
        allow_fail=True,
    )
    logs.append({"step": "enrich_visual_captions", "code": code, "tail": out[-1800:]})

    code, out = run(
        [
            PY,
            "scripts/link_visual_sequences.py",
            "--markdown",
            "final_extracted_text_only_structured_full.md",
            "--region-manifest",
            "logs/figure_regions_manifest_latest.json",
            "--output-manifest",
            "logs/figure_regions_manifest_latest.json",
            "--output-links",
            "logs/visual_sequence_links_latest.json",
        ],
        timeout=1200,
        allow_fail=True,
    )
    logs.append({"step": "link_visual_sequences", "code": code, "tail": out[-1600:]})

    code, out = run(
        [
            PY,
            "scripts/build_visual_evidence_sidecar.py",
            "--markdown",
            "final_extracted_text_only_structured_full.md",
            "--region-manifest",
            "logs/figure_regions_manifest_latest.json",
            "--sequence-links",
            "logs/visual_sequence_links_latest.json",
            "--output-jsonl",
            "indexes/visual/region_evidence.jsonl",
            "--report",
            "logs/visual_evidence_sidecar_report_latest.json",
        ],
        timeout=1200,
        allow_fail=True,
    )
    logs.append({"step": "build_visual_evidence_sidecar", "code": code, "tail": out[-1600:]})

    code, out = run(
        [
            PY,
            "scripts/audit_extraction_quality.py",
            "--input",
            "final_extracted_content.md",
            "--output",
            "logs/extraction_audit_latest.csv",
        ],
        timeout=1200,
    )
    logs.append({"step": "audit_extraction_quality", "code": code, "tail": out[-1200:]})

    code, out = run(
        [
            PY,
            "scripts/evaluate_ingest_error_budget.py",
            "--audit",
            "logs/extraction_audit_latest.csv",
                "--quality",
                "logs/ingest_quality_report.csv",
                "--region-manifest",
                "logs/figure_regions_manifest_latest.json",
                "--sidecar-report",
                "logs/visual_evidence_sidecar_report_latest.json",
                "--sequence-links",
                "logs/visual_sequence_links_latest.json",
                "--output",
                "logs/ingest_error_assessment_latest.json",
            ],
        timeout=600,
    )
    logs.append({"step": "evaluate_ingest_error_budget", "code": code, "tail": out[-1200:]})

    code, out = run(
        [
            PY,
            "scripts/build_document_hierarchy.py",
            "--toc",
            str(args.toc),
            "--markdown",
            "final_extracted_text_only_structured_full.md",
            "--output",
            "indexes/hierarchical/topic_hierarchy.json",
            "--allow-legacy-encodings",
        ],
        timeout=1800,
    )
    logs.append({"step": "build_document_hierarchy", "code": code, "tail": out[-1400:]})

    code, out = run(
        [
            PY,
            "scripts/sync_section_page_overrides.py",
            "--hierarchy",
            "indexes/hierarchical/topic_hierarchy.json",
            "--output",
            "indexes/hierarchical/section_page_overrides.json",
            "--max-pages-per-topic",
            "4",
        ],
        timeout=600,
    )
    logs.append({"step": "sync_section_page_overrides", "code": code, "tail": out[-1200:]})

    code, out = run(
        [
            PY,
            "scripts/prepare_colpali_corpus.py",
            "--markdown",
            "final_extracted_text_only_structured_full.md",
            "--figure-manifest",
            "logs/figure_manifest_latest.json",
            "--region-manifest",
            "logs/figure_regions_manifest_latest.json",
            "--hierarchy-index",
            "indexes/hierarchical/topic_hierarchy.json",
            "--include-visual-captions",
            "--output",
            "indexes/colpali/pages.jsonl",
            "--report",
            "logs/colpali_prep_report_latest.json",
        ],
        timeout=1800,
    )
    logs.append({"step": "prepare_colpali_corpus", "code": code, "tail": out[-1400:]})

    after_ocr = Path(f"logs/ocr_research_grade_after_reingest_{ts}.json")
    after_topic = Path(f"logs/visual_topic_benchmark_after_reingest_{ts}.json")
    after_task = Path(f"logs/visual_task_metrics_after_reingest_{ts}.json")

    code, out = run(
        [
            PY,
            "scripts/evaluate_ocr_research_grade.py",
            "--pdf",
            str(args.pdf),
            "--markdown",
            "final_extracted_text_only_structured_full.md",
            "--toc",
            str(args.toc),
            "--gt-jsonl",
            str(args.gt_jsonl),
            "--output",
            str(after_ocr),
        ],
        timeout=1200,
    )
    logs.append({"step": "after_ocr_eval", "code": code, "tail": out[-1200:]})

    if args.endpoint and (not bool(args.preprocess_only)):
        code, out = run(
            [
                PY,
                "scripts/benchmark_visual_topic_retrieval.py",
                "--dataset",
                str(args.dataset),
                "--endpoint",
                str(args.endpoint),
                "--sparse-strategy",
                "bm25",
                "--top-k",
                str(max(1, int(args.top_k))),
                "--candidate-k",
                str(max(1, int(args.candidate_k))),
                "--output-json",
                str(after_topic),
                "--output-csv",
                str(after_topic.with_suffix(".csv")),
            ],
            timeout=3600,
            allow_fail=True,
        )
        logs.append({"step": "after_topic_bench", "code": code, "tail": out[-1500:]})

        code, out = run(
            [
                PY,
                "scripts/benchmark_visual_task_metrics.py",
                "--dataset",
                str(args.dataset),
                "--endpoint",
                str(args.endpoint),
                "--sparse-strategy",
                "bm25",
                "--top-k",
                str(max(1, int(args.top_k))),
                "--candidate-k",
                str(max(1, int(args.candidate_k))),
                "--output-json",
                str(after_task),
                "--output-csv",
                str(after_task.with_suffix(".csv")),
            ],
            timeout=3600,
            allow_fail=True,
        )
        logs.append({"step": "after_task_bench", "code": code, "tail": out[-1500:]})
    elif bool(args.preprocess_only):
        logs.append({"step": "after_retrieval_bench", "code": -1, "tail": "skipped: --preprocess-only"})
    else:
        logs.append({"step": "after_retrieval_bench", "code": -1, "tail": "skipped: missing endpoint"})

    smoke: list[dict[str, Any]] = []
    if not bool(args.preprocess_only):
        smoke_queries = [
            "การดำเนินการแทนคิวด้วยอาร์เรย์",
            "การแทนคิวด้วยวงกลม",
            "การทำงานของลิงค์ลิสต์แบบทิศทางเดียว",
        ]
        for i, q in enumerate(smoke_queries, start=1):
            out_path = Path(f"logs/reingest_smoke_query_{i}_{ts}.json")
            cmd = [
                PY,
                "scripts/retrieve_visual_hybrid.py",
                "--query",
                q,
                "--backend",
                "auto",
                "--sparse-strategy",
                "bm25",
                "--metadata-filter-strict",
                "--top-k",
                str(max(1, int(args.top_k))),
                "--candidate-k",
                str(max(1, int(args.candidate_k))),
                "--prefilter-min-records",
                "8",
                "--prefilter-rescue-topn",
                "24",
                "--output",
                str(out_path),
            ]
            if args.endpoint:
                cmd.extend(["--colpali-endpoint-url", str(args.endpoint)])
            code, out = run(cmd, timeout=2400, allow_fail=True)
            logs.append({"step": f"smoke_retrieval_{i}", "code": code, "tail": out[-1400:]})
            payload = load_json(out_path)
            smoke.append(smoke_map(payload, q))
    else:
        logs.append({"step": "smoke_retrieval", "code": -1, "tail": "skipped: --preprocess-only"})
    before_ocr_report = load_json(before_ocr)
    before_topic_report = load_json(before_topic)
    before_task_report = load_json(before_task)
    after_ocr_report = load_json(after_ocr)
    after_topic_report = load_json(after_topic)
    after_task_report = load_json(after_task)
    before_summary = summarize_metrics(
        ocr_report=before_ocr_report,
        topic_report=before_topic_report,
        task_report=before_task_report,
    )
    after_summary = summarize_metrics(
        ocr_report=after_ocr_report,
        topic_report=after_topic_report,
        task_report=after_task_report,
    )
    delta = diff_metrics(before_summary, after_summary)

    report = {
        "timestamp": ts,
        "config": {
            "pdf": str(args.pdf),
            "toc": str(args.toc),
            "gt_jsonl": str(args.gt_jsonl),
            "dataset": str(args.dataset),
            "endpoint": str(args.endpoint),
            "vision_model": str(args.vision_model),
            "caption_model": str(args.caption_model),
            "caption_workers": int(args.caption_workers),
            "caption_max_pages": int(args.caption_max_pages),
            "caption_max_images_per_page": int(args.caption_max_images_per_page),
            "top_k": int(args.top_k),
            "candidate_k": int(args.candidate_k),
            "skip_rescan": bool(args.skip_rescan),
            "preprocess_only": bool(args.preprocess_only),
        },
        "before": before_summary,
        "after": after_summary,
        "delta": delta,
        "artifacts": {
            "before_ocr_json": str(before_ocr.as_posix()),
            "after_ocr_json": str(after_ocr.as_posix()),
            "before_topic_json": str(before_topic.as_posix()),
            "after_topic_json": str(after_topic.as_posix()),
            "before_task_json": str(before_task.as_posix()),
            "after_task_json": str(after_task.as_posix()),
            "ingest_error_json": "logs/ingest_error_assessment_latest.json",
            "caption_report_json": "logs/visual_caption_enrich_report_latest.json",
            "text_only_report_json": "logs/text_only_markdown_report_latest.json",
            "sequence_links_json": "logs/visual_sequence_links_latest.json",
            "visual_sidecar_jsonl": "indexes/visual/region_evidence.jsonl",
            "visual_sidecar_report_json": "logs/visual_evidence_sidecar_report_latest.json",
            "hierarchy_json": "indexes/hierarchical/topic_hierarchy.json",
            "colpali_pages_jsonl": "indexes/colpali/pages.jsonl",
        },
        "smoke": smoke,
        "logs": logs,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_summary(output_md, report)
    print(json.dumps({"before": before_summary, "after": after_summary}, ensure_ascii=False, indent=2))
    print(f"saved_json={output_json}")
    print(f"saved_md={output_md}")


if __name__ == "__main__":
    main()

