"""
Orchestrate multimodal retrieval pipeline setup.

Steps:
1) (optional) rescan PDF via src/ingest.py
2) extraction audit + ingest error budget
3) build text-only markdown + extract structure regions
4) enrich visual captions into structured-full markdown (cached/parallel)
5) build hierarchy index from list_hitachi
6) prepare ColPali corpus JSONL
7) optional build ColPali index
8) chapter-window weak-vote relabel + chapter-2 hard-negative update
9) smoke retrieval queries
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def pick_primary_markdown() -> str:
    """
    Prefer the fully-structured markdown if available.
    Fallback order keeps backward compatibility for older runs.
    """
    candidates = [
        ROOT / "final_extracted_text_only_structured_full.md",
        ROOT / "final_extracted_text_only.md",
        ROOT / "final_extracted_content.md",
    ]
    for p in candidates:
        if p.exists():
            return str(p.name)
    return "final_extracted_content.md"


def run(cmd: list[str], *, timeout: int = 0, allow_fail: bool = False) -> tuple[int, str]:
    env = os.environ.copy()
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
        timeout=timeout if timeout > 0 else None,
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0 and not allow_fail:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return p.returncode, out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multimodal retrieval pipeline setup.")
    ap.add_argument("--rescan", action="store_true", help="Run full PDF rescan via src/ingest.py")
    ap.add_argument("--build-colpali", action="store_true", help="Attempt local ColPali index build")
    ap.add_argument("--colpali-model", default="vidore/colpali-v1.3-hf")
    ap.add_argument("--smoke-output", default="logs/visual_pipeline_smoke_latest.json")
    args = ap.parse_args()

    logs = []
    working_markdown = pick_primary_markdown()

    if args.rescan:
        code, out = run([PY, "src/ingest.py"], timeout=7200, allow_fail=False)
        logs.append({"step": "rescan", "code": code, "tail": out[-2000:]})

    code, out = run([PY, "scripts/audit_extraction_quality.py", "--input", "final_extracted_content.md", "--output", "logs/extraction_audit_latest.csv"], timeout=300)
    logs.append({"step": "audit_extraction", "code": code, "tail": out[-1200:]})

    code, out = run(
        [
            PY,
            "scripts/extract_structure_regions.py",
            "--markdown",
            "final_extracted_content.md",
            "--target-only",
            "--fallback-when-empty",
        ],
        timeout=1800,
    )
    logs.append({"step": "extract_regions", "code": code, "tail": out[-1200:]})

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
        timeout=600,
    )
    logs.append({"step": "build_text_only_markdown", "code": code, "tail": out[-1200:]})
    # Prefer the structured-full markdown if present; otherwise use text-only output.
    structured_full = ROOT / "final_extracted_text_only_structured_full.md"
    if structured_full.exists():
        working_markdown = str(structured_full.name)
    else:
        working_markdown = "final_extracted_text_only.md"

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
            "--workers",
            "2",
            "--max-images-per-page",
            "2",
        ],
        timeout=2400,
        allow_fail=True,
    )
    logs.append({"step": "enrich_visual_captions", "code": code, "tail": out[-1200:]})
    structured_full_after = ROOT / "final_extracted_text_only_structured_full.md"
    if code == 0 and structured_full_after.exists():
        working_markdown = str(structured_full_after.name)
    else:
        working_markdown = "final_extracted_text_only.md"

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
        timeout=900,
        allow_fail=True,
    )
    logs.append({"step": "link_visual_sequences", "code": code, "tail": out[-1200:]})

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
        timeout=900,
        allow_fail=True,
    )
    logs.append({"step": "build_visual_evidence_sidecar", "code": code, "tail": out[-1200:]})

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
        timeout=300,
    )
    logs.append({"step": "ingest_error_budget", "code": code, "tail": out[-1200:]})

    code, out = run(
        [
            PY,
            "scripts/build_document_hierarchy.py",
            "--toc",
            "list_hitachi.txt",
            "--markdown",
            working_markdown,
            "--output",
            "indexes/hierarchical/topic_hierarchy.json",
        ],
        timeout=300,
    )
    logs.append({"step": "build_hierarchy", "code": code, "tail": out[-1200:]})

    code, out = run(
        [
            PY,
            "scripts/relabel_chapter_window_weak_vote.py",
            "--hierarchy-index",
            "indexes/hierarchical/topic_hierarchy.json",
            "--markdown",
            working_markdown,
            "--source",
            "data_structure_data_ch1_to_ch5.pdf",
            "--page-start",
            "24",
            "--page-end",
            "28",
            "--allowed-topics",
            "2.4,2.4.1,2.4.2",
            "--sequence-decoder",
            "hmm",
            "--output",
            "indexes/hierarchical/topic_hierarchy.json",
            "--report",
            "logs/chapter_window_relabel_report_latest.json",
        ],
        timeout=300,
    )
    logs.append({"step": "relabel_chapter_window_weak_vote", "code": code, "tail": out[-1200:]})

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
        timeout=300,
    )
    logs.append({"step": "sync_section_page_overrides", "code": code, "tail": out[-1200:]})

    code, out = run(
        [
            PY,
            "scripts/expand_chapter_calibration.py",
            "--hierarchy-index",
            "indexes/hierarchical/topic_hierarchy.json",
            "--input-calibration",
            "indexes/hierarchical/chapter_calibration.json",
            "--output",
            "indexes/hierarchical/chapter_calibration.json",
        ],
        timeout=300,
    )
    logs.append({"step": "expand_chapter_calibration", "code": code, "tail": out[-1200:]})

    code, out = run([PY, "scripts/prepare_colpali_corpus.py", "--markdown", working_markdown], timeout=300)
    logs.append({"step": "prepare_colpali_corpus", "code": code, "tail": out[-1200:]})

    code, out = run(
        [
            PY,
            "scripts/update_chapter2_hard_negatives.py",
            "--input",
            "indexes/hierarchical/hard_negative_rules.json",
            "--output",
            "indexes/hierarchical/hard_negative_rules.json",
        ],
        timeout=300,
    )
    logs.append({"step": "update_chapter2_hard_negatives", "code": code, "tail": out[-1200:]})

    code, out = run(
        [
            PY,
            "scripts/mine_visual_hard_negatives.py",
            "--pages-jsonl",
            "indexes/colpali/pages.jsonl",
            "--hierarchy-index",
            "indexes/hierarchical/topic_hierarchy.json",
            "--output",
            "eval/visual_hard_negatives_latest.jsonl",
        ],
        timeout=600,
        allow_fail=True,
    )
    logs.append({"step": "mine_hard_negatives", "code": code, "tail": out[-1200:]})

    if args.build_colpali:
        code, out = run([
            PY,
            "scripts/build_colpali_index.py",
            "--model",
            args.colpali_model,
            "--output",
            "indexes/colpali/colpali_index.pt",
            "--report",
            "logs/colpali_index_report_latest.json",
        ], timeout=7200, allow_fail=True)
        logs.append({"step": "build_colpali", "code": code, "tail": out[-2000:]})

    smoke_queries = [
        "โครงสร้างลิงค์ลิสต์แบบทิศทางเดียว",
        "โครงสร้างของการแทนคิวด้วยอาร์เรย์",
        "การดำเนินการแทนคิวด้วยวงกลม",
    ]
    smoke_results = []
    for i, q in enumerate(smoke_queries, start=1):
        out_path = f"logs/visual_retrieval_smoke_{i}.json"
        code, out = run(
            [
                PY,
                "scripts/retrieve_visual_hybrid.py",
                "--query",
                q,
                "--backend",
                "auto",
                "--require-structure",
                "--use-vlm-rerank",
                "--use-visual-grounding",
                "--top-k",
                "3",
                "--candidate-k",
                "10",
                "--output",
                out_path,
            ],
            timeout=1200,
            allow_fail=True,
        )
        smoke_results.append({"query": q, "code": code, "output": out_path, "tail": out[-1000:]})

    endpoint = (os.getenv("COLPALI_ENDPOINT_URL", "") or "").strip()
    if endpoint:
        code, out = run(
            [
                PY,
                "scripts/benchmark_visual_grounding_human.py",
                "--dataset",
                "eval/visual_grounding_human_labels_ch2_ch3_v2.jsonl",
                "--endpoint",
                endpoint,
                "--top-k",
                "12",
                "--candidate-k",
                "30",
                "--use-vlm-rerank",
                "--output-json",
                "logs/visual_grounding_human_benchmark_latest.json",
                "--output-csv",
                "logs/visual_grounding_human_benchmark_latest.csv",
            ],
            timeout=1800,
            allow_fail=True,
        )
        logs.append({"step": "benchmark_visual_grounding_human", "code": code, "tail": out[-1500:]})
    else:
        logs.append({"step": "benchmark_visual_grounding_human", "code": -1, "tail": "skipped: missing COLPALI_ENDPOINT_URL"})

    payload = {
        "logs": logs,
        "smoke": smoke_results,
    }
    out = Path(args.smoke_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
