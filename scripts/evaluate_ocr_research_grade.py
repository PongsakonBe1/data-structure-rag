"""
Research-grade OCR evaluation report for this project.

This script combines:
1) Intrinsic OCR metrics (CER/WER) when ground-truth text is available.
2) OCR/data-quality proxies that can be measured from current artifacts.
3) Downstream signal checks (visual retrieval + human grounding summaries).

Output:
- logs/ocr_research_grade_report_latest.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
FIGURE_REF_RE = re.compile(r"(ภาพที่|figure|ตารางที่|table)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
TOC_LINE_RE = re.compile(r"[^\d]*([0-9]+(?:\.[0-9]+)*)\s+(.*)")
ANCHOR_BLOCK_RE = re.compile(r"\n### Visual Anchors \(Auto\)[\s\S]*$", re.MULTILINE)
VISUAL_CAPTION_BLOCK_RE = re.compile(r"\n### Visual Captions \(Auto\)\n[\s\S]*$", re.MULTILINE)


@dataclass
class PageRow:
    source: str
    page: int
    body: str


def _safe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_load_first(paths: list[Path]) -> dict:
    for p in paths:
        obj = _safe_load_json(p)
        if obj:
            return obj
    return {}


def _parse_markdown_pages(md_text: str) -> list[PageRow]:
    rows: list[PageRow] = []
    headers = list(PAGE_HEADER_RE.finditer(md_text))
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        rows.append(
            PageRow(
                source=m.group(1).strip(),
                page=int(m.group(2)),
                body=md_text[start:end].strip(),
            )
        )
    return rows


def _normalize_heading(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("•", " ").replace("·", " ").replace("โ€ข", " ").replace("เนเธ", " ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", s)
    canonical_map = {
        "แท่นคิว": "แทนคิว",
        "โพสออเดอร์": "โพสต์ออเดอร์",
        "อินออร์เดอร์": "อินออเดอร์",
        "ไบนารีทรี": "นารีทรี",
    }
    for bad, good in canonical_map.items():
        s = s.replace(bad, good)
    cleaned = []
    for ch in s:
        if ch.isspace():
            cleaned.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    s = "".join(cleaned)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_md_headings(md_text: str) -> list[str]:
    out: list[str] = []
    for line in md_text.splitlines():
        if not line.startswith("#"):
            continue
        h = line.lstrip("#").strip()
        if not h:
            continue
        h_low = h.lower()
        if h_low.startswith("source:") or h_low.startswith("page "):
            continue
        if h_low.startswith("visual captions (auto)"):
            continue
        out.append(_normalize_heading(h))
    return out


def _extract_toc_headings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        s = s.replace("โข", " ").strip()
        m = TOC_LINE_RE.search(s)
        if not m:
            continue
        topic_id = m.group(1).strip()
        title = _normalize_heading(m.group(2).strip())
        if topic_id and title:
            rows.append({"topic_id": topic_id, "title": title})
    return rows


def _heading_soft_recall(md_headings: list[str], toc_headings: list[dict], threshold: float) -> dict:
    if not toc_headings:
        return {
            "count_toc": 0,
            "matched": 0,
            "recall": 0.0,
            "threshold": threshold,
            "examples_low_score": [],
        }
    matched = 0
    lows = []
    for row in toc_headings:
        t = str(row.get("title", "")).strip()
        if not t:
            continue
        best = 0.0
        for h in md_headings:
            sc = SequenceMatcher(None, t, h).ratio()
            if sc > best:
                best = sc
        if best >= threshold:
            matched += 1
        else:
            lows.append({"topic_id": row.get("topic_id", ""), "score": round(best, 4), "title": t})
    lows = sorted(lows, key=lambda x: x["score"])[:10]
    total = len(toc_headings)
    return {
        "count_toc": total,
        "matched": matched,
        "recall": round(matched / max(1, total), 6),
        "threshold": threshold,
        "examples_low_score": lows,
    }


def _char_word_error_rates(pred: str, gt: str) -> tuple[float, float]:
    """Approximate CER/WER from SequenceMatcher opcodes."""

    def _edit_distance(seq_a: list[str], seq_b: list[str]) -> int:
        n = len(seq_a)
        m = len(seq_b)
        if n == 0:
            return m
        if m == 0:
            return n
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            ai = seq_a[i - 1]
            for j in range(1, m + 1):
                cur = dp[j]
                cost = 0 if ai == seq_b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = cur
        return dp[m]

    pred_chars = list(pred)
    gt_chars = list(gt)
    char_ed = _edit_distance(pred_chars, gt_chars)
    cer = char_ed / max(1, len(gt_chars))

    pred_words = [w for w in re.split(r"\s+", pred.strip()) if w]
    gt_words = [w for w in re.split(r"\s+", gt.strip()) if w]
    word_ed = _edit_distance(pred_words, gt_words)
    wer = word_ed / max(1, len(gt_words))
    return float(cer), float(wer)


def _compute_gt_metrics(
    gt_jsonl: Path | None,
    md_pages: list[PageRow],
    *,
    manual_verified_only: bool = False,
    min_samples: int = 1,
) -> dict:
    def _norm_page_text(s: str) -> str:
        s = ANCHOR_BLOCK_RE.sub("", str(s or "")).strip()
        s = VISUAL_CAPTION_BLOCK_RE.sub("", s).strip()
        lines = []
        for ln in s.splitlines():
            t = ln.strip()
            if not t:
                continue
            t = t.lstrip("#").strip()
            if t.startswith("- [VA-"):
                continue
            lines.append(t)
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    page_map = {int(r.page): _norm_page_text(r.body) for r in md_pages}
    if gt_jsonl is None or (not gt_jsonl.exists()):
        return {
            "available": False,
            "reason": "missing_gt_jsonl",
            "samples": 0,
            "samples_total_rows": 0,
            "samples_used_rows": 0,
            "manual_verified_only": bool(manual_verified_only),
        }

    rows = []
    total_rows = 0
    skipped_unverified = 0
    for ln in gt_jsonl.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        total_rows += 1
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if manual_verified_only and (not bool(obj.get("manual_verified", False))):
            skipped_unverified += 1
            continue
        page = int(obj.get("page", 0) or 0)
        gt_text = _norm_page_text(str(obj.get("text", "") or ""))
        if page <= 0 or not gt_text:
            continue
        pred = _norm_page_text(str(page_map.get(page, "") or ""))
        cer, wer = _char_word_error_rates(pred=pred, gt=gt_text)
        rows.append({"page": page, "cer": cer, "wer": wer})

    if len(rows) < max(1, int(min_samples)):
        return {
            "available": False,
            "reason": "insufficient_verified_gt_samples" if manual_verified_only else "empty_or_invalid_gt_rows",
            "samples": len(rows),
            "samples_total_rows": total_rows,
            "samples_used_rows": len(rows),
            "skipped_unverified_rows": skipped_unverified,
            "manual_verified_only": bool(manual_verified_only),
            "min_samples_required": max(1, int(min_samples)),
        }

    cer_mean = sum(r["cer"] for r in rows) / len(rows)
    wer_mean = sum(r["wer"] for r in rows) / len(rows)
    return {
        "available": True,
        "samples": len(rows),
        "samples_total_rows": total_rows,
        "samples_used_rows": len(rows),
        "skipped_unverified_rows": skipped_unverified,
        "manual_verified_only": bool(manual_verified_only),
        "min_samples_required": max(1, int(min_samples)),
        "cer_mean": round(cer_mean, 6),
        "wer_mean": round(wer_mean, 6),
        "cer_p95": round(sorted(r["cer"] for r in rows)[int(0.95 * (len(rows) - 1))], 6),
        "wer_p95": round(sorted(r["wer"] for r in rows)[int(0.95 * (len(rows) - 1))], 6),
    }


def _pdf_page_count(pdf_path: Path) -> int | None:
    if fitz is None or not pdf_path.exists():
        return None
    try:
        with fitz.open(str(pdf_path)) as doc:
            return int(doc.page_count)
    except Exception:
        return None


def _pdf_extractable_text_pages(pdf_path: Path) -> int | None:
    if fitz is None or not pdf_path.exists():
        return None
    try:
        n = 0
        with fitz.open(str(pdf_path)) as doc:
            for p in doc:
                t = p.get_text("text")
                if str(t or "").strip():
                    n += 1
        return n
    except Exception:
        return None


def _figure_ref_recall_from_labels(md_pages: list[PageRow], labels_jsonl: Path) -> dict:
    if not labels_jsonl.exists():
        return {"available": False, "reason": "missing_labels_jsonl", "recall": 0.0, "total_refs": 0}

    page_map: dict[int, str] = {}
    for r in md_pages:
        page_map[int(r.page)] = page_map.get(int(r.page), "") + "\n" + str(r.body)

    total_refs = 0
    hit_refs = 0
    missing_examples = []
    for ln in labels_jsonl.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        expected_pages = [int(x) for x in (obj.get("expected_pages", []) or []) if str(x).strip().isdigit()]
        expected_refs = [str(x).strip() for x in (obj.get("expected_figure_refs", []) or []) if str(x).strip()]
        if not expected_pages or not expected_refs:
            continue
        page_text = "\n".join(page_map.get(p, "") for p in expected_pages)
        for ref in expected_refs:
            total_refs += 1
            if re.search(rf"\b{re.escape(ref)}\b", page_text):
                hit_refs += 1
            elif len(missing_examples) < 10:
                missing_examples.append(
                    {"label_id": obj.get("label_id", ""), "ref": ref, "expected_pages": expected_pages}
                )

    return {
        "available": True,
        "total_refs": total_refs,
        "hit_refs": hit_refs,
        "recall": round(hit_refs / max(1, total_refs), 6),
        "missing_examples": missing_examples,
    }


def _encoding_health(md_text: str) -> dict:
    total = max(1, len(md_text))
    replacement = md_text.count("\ufffd")
    c0 = sum(1 for ch in md_text if ord(ch) < 32 and ch not in ("\n", "\r", "\t"))
    c1 = sum(1 for ch in md_text if 127 <= ord(ch) <= 159)
    return {
        "replacement_char_count": replacement,
        "replacement_char_ratio": round(replacement / total, 8),
        "control_char_count": c0 + c1,
        "control_char_ratio": round((c0 + c1) / total, 8),
    }


def _non_empty_page_ratio(md_pages: list[PageRow], min_chars: int) -> float:
    if not md_pages:
        return 0.0
    ok = sum(1 for r in md_pages if len((r.body or "").strip()) >= max(1, min_chars))
    return round(ok / len(md_pages), 6)


def _check_leq(value: float | None, threshold: float) -> dict:
    if value is None:
        return {"value": None, "threshold": threshold, "pass": False}
    return {"value": round(float(value), 6), "threshold": threshold, "pass": float(value) <= threshold}


def _check_geq(value: float | None, threshold: float) -> dict:
    if value is None:
        return {"value": None, "threshold": threshold, "pass": False}
    return {"value": round(float(value), 6), "threshold": threshold, "pass": float(value) >= threshold}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate OCR quality with research-grade criteria.")
    ap.add_argument("--pdf", default="data/data_structure_data_ch1_to_ch5.pdf")
    ap.add_argument("--markdown", default="final_extracted_text_only.md")
    ap.add_argument("--toc", default="list_hitachi.txt")
    ap.add_argument("--labels", default="eval/visual_grounding_human_labels_ch2_ch3_v2.jsonl")
    ap.add_argument("--gt-jsonl", default="")
    ap.add_argument("--heading-sim-threshold", type=float, default=0.65)
    ap.add_argument("--min-page-chars", type=int, default=80)
    ap.add_argument("--cer-threshold", type=float, default=0.05)
    ap.add_argument("--wer-threshold", type=float, default=0.2)
    ap.add_argument(
        "--strict-mode",
        choices=["internal", "external_publish_grade"],
        default="external_publish_grade",
        help="external_publish_grade: strict pass requires CER/WER from manual_verified=true GT rows only.",
    )
    ap.add_argument(
        "--min-manual-verified-samples",
        type=int,
        default=10,
        help="Minimum manual_verified rows required when strict-mode=external_publish_grade.",
    )
    ap.add_argument("--output", default="logs/ocr_research_grade_report_latest.json")
    args = ap.parse_args()

    markdown_path = Path(args.markdown)
    if not markdown_path.exists():
        raise FileNotFoundError(f"missing markdown: {markdown_path}")

    md_text = markdown_path.read_text(encoding="utf-8")
    md_pages = _parse_markdown_pages(md_text)
    toc_rows = _extract_toc_headings(Path(args.toc))
    md_headings = _extract_md_headings(md_text)

    ingest_assessment = _safe_load_json(Path("logs/ingest_error_assessment_latest.json"))
    text_only_report = _safe_load_json(Path("logs/text_only_markdown_report_latest.json"))
    region_manifest = _safe_load_json(Path("logs/figure_regions_manifest_latest.json"))
    topic_bench = _safe_load_first(
        [
            Path("logs/visual_topic_benchmark_current_latest.json"),
            Path("logs/visual_topic_benchmark_latest.json"),
            Path("logs/visual_topic_benchmark_after_reingest_latest.json"),
            Path("logs/visual_topic_benchmark_after_norm_latest.json"),
        ]
    )
    human_bench = _safe_load_first(
        [
            Path("logs/visual_grounding_human_benchmark_latest.json"),
            Path("logs/visual_grounding_human_benchmark_ch2_ch3_v2_latest.json"),
            Path("logs/visual_grounding_human_benchmark_ch2_ch3_v2_after_norm_noleak_latest.json"),
        ]
    )
    hierarchy_index = _safe_load_json(Path("indexes/hierarchical/topic_hierarchy.json"))

    pdf_path = Path(args.pdf)
    pdf_pages = _pdf_page_count(pdf_path)
    pdf_extractable_pages = _pdf_extractable_text_pages(pdf_path)
    page_coverage_ratio = (len(md_pages) / max(1, pdf_pages)) if pdf_pages else None

    heading_report = _heading_soft_recall(
        md_headings=md_headings,
        toc_headings=toc_rows,
        threshold=float(args.heading_sim_threshold),
    )
    topic_to_pages = hierarchy_index.get("topic_to_pages", {}) if isinstance(hierarchy_index, dict) else {}
    toc_effective = [
        r for r in toc_rows
        if isinstance(topic_to_pages.get(str(r.get("topic_id", "")), []), list)
        and len(topic_to_pages.get(str(r.get("topic_id", "")), [])) > 0
    ]
    heading_report_effective = _heading_soft_recall(
        md_headings=md_headings,
        toc_headings=toc_effective if toc_effective else toc_rows,
        threshold=float(args.heading_sim_threshold),
    )
    figure_ref_report = _figure_ref_recall_from_labels(md_pages=md_pages, labels_jsonl=Path(args.labels))
    enc_report = _encoding_health(md_text)
    non_empty_ratio = _non_empty_page_ratio(md_pages, min_chars=int(args.min_page_chars))

    strict_mode = str(args.strict_mode or "external_publish_grade").strip()
    manual_only = strict_mode == "external_publish_grade"
    gt_metrics = _compute_gt_metrics(
        gt_jsonl=Path(args.gt_jsonl) if str(args.gt_jsonl).strip() else None,
        md_pages=md_pages,
        manual_verified_only=manual_only,
        min_samples=(int(args.min_manual_verified_samples) if manual_only else 1),
    )

    ingest_checks = (ingest_assessment.get("checks", {}) if isinstance(ingest_assessment, dict) else {}) or {}
    issue_page_ratio = ((ingest_checks.get("issue_page_ratio", {}) or {}).get("value"))
    figure_structure_coverage = ((ingest_checks.get("figure_structure_coverage", {}) or {}).get("value"))
    figure_ref_to_region_link_recall = ((ingest_checks.get("figure_ref_to_region_link_recall", {}) or {}).get("value"))
    operation_pages_region_coverage = ((ingest_checks.get("operation_pages_region_coverage", {}) or {}).get("value"))
    table_cell_visual_coverage = ((ingest_checks.get("table_cell_visual_coverage", {}) or {}).get("value"))
    generic_caption_ratio = ((ingest_checks.get("generic_caption_ratio", {}) or {}).get("value"))
    sequence_stitch_coverage = ((ingest_checks.get("sequence_stitch_coverage", {}) or {}).get("value"))
    caption_injected_ratio = ((ingest_checks.get("caption_injected_ratio", {}) or {}).get("value"))

    anchor_coverage_ratio = text_only_report.get("anchor_coverage_ratio")

    region_summary = region_manifest.get("summary", {}) if isinstance(region_manifest, dict) else {}
    tiny_before = float(region_summary.get("tiny_regions_before", 0) or 0)
    tiny_after = float(region_summary.get("tiny_regions_after", 0) or 0)
    tiny_after_ratio = (tiny_after / max(1.0, tiny_before)) if tiny_before > 0 else 0.0

    topic_summary = topic_bench.get("summary", {}) if isinstance(topic_bench, dict) else {}
    crop_completeness_mean = topic_summary.get("crop_completeness_proxy_mean")
    retrieval_hit_mean = topic_summary.get("retrieval_hit_at_k_mean")

    human_summary = human_bench.get("summary", {}) if isinstance(human_bench, dict) else {}
    human_pass_rate = human_summary.get("pass_rate")

    thresholds = {
        "page_coverage_ratio_min": 0.99,
        "non_empty_page_ratio_min": 0.99,
        "heading_soft_recall_min": 0.9,
        "heading_soft_recall_effective_min": 0.82,
        "expected_figure_ref_recall_min": 0.85,
        "replacement_char_ratio_max": 0.0001,
        "control_char_ratio_max": 0.0001,
        "issue_page_ratio_max": 0.05,
        "figure_structure_coverage_min": 0.90,
        "figure_ref_to_region_link_recall_min": 0.90,
        "operation_pages_region_coverage_min": 0.90,
        "table_cell_visual_coverage_min": 0.85,
        "generic_caption_ratio_max": 0.10,
        "sequence_stitch_coverage_min": 0.80,
        "caption_injected_ratio_max": 0.35,
        "anchor_coverage_ratio_min": 0.95,
        "tiny_region_after_ratio_max": 0.1,
        "crop_completeness_proxy_mean_min": 0.75,
        "retrieval_hit_at_k_mean_min": 0.9,
        "human_grounding_pass_rate_min": 0.9,
        "cer_max": float(args.cer_threshold),
        "wer_max": float(args.wer_threshold),
    }

    checks = {
        "page_coverage_ratio": _check_geq(page_coverage_ratio, thresholds["page_coverage_ratio_min"]),
        "non_empty_page_ratio": _check_geq(non_empty_ratio, thresholds["non_empty_page_ratio_min"]),
        "heading_soft_recall": _check_geq(heading_report.get("recall"), thresholds["heading_soft_recall_min"]),
        "heading_soft_recall_effective": _check_geq(
            heading_report_effective.get("recall"),
            thresholds["heading_soft_recall_effective_min"],
        ),
        "expected_figure_ref_recall": _check_geq(
            figure_ref_report.get("recall"), thresholds["expected_figure_ref_recall_min"]
        ),
        "replacement_char_ratio": _check_leq(
            enc_report.get("replacement_char_ratio"), thresholds["replacement_char_ratio_max"]
        ),
        "control_char_ratio": _check_leq(enc_report.get("control_char_ratio"), thresholds["control_char_ratio_max"]),
        "issue_page_ratio": _check_leq(issue_page_ratio, thresholds["issue_page_ratio_max"]),
        "figure_structure_coverage": _check_geq(
            figure_structure_coverage, thresholds["figure_structure_coverage_min"]
        ),
        "figure_ref_to_region_link_recall": _check_geq(
            figure_ref_to_region_link_recall, thresholds["figure_ref_to_region_link_recall_min"]
        ),
        "operation_pages_region_coverage": _check_geq(
            operation_pages_region_coverage, thresholds["operation_pages_region_coverage_min"]
        ),
        "table_cell_visual_coverage": _check_geq(
            table_cell_visual_coverage, thresholds["table_cell_visual_coverage_min"]
        ),
        "generic_caption_ratio": _check_leq(
            generic_caption_ratio, thresholds["generic_caption_ratio_max"]
        ),
        "sequence_stitch_coverage": _check_geq(
            sequence_stitch_coverage, thresholds["sequence_stitch_coverage_min"]
        ),
        "caption_injected_ratio": _check_leq(caption_injected_ratio, thresholds["caption_injected_ratio_max"]),
        "anchor_coverage_ratio": _check_geq(anchor_coverage_ratio, thresholds["anchor_coverage_ratio_min"]),
        "tiny_region_after_ratio": _check_leq(tiny_after_ratio, thresholds["tiny_region_after_ratio_max"]),
        "crop_completeness_proxy_mean": _check_geq(
            crop_completeness_mean, thresholds["crop_completeness_proxy_mean_min"]
        ),
        "retrieval_hit_at_k_mean": _check_geq(retrieval_hit_mean, thresholds["retrieval_hit_at_k_mean_min"]),
        "human_grounding_pass_rate": _check_geq(human_pass_rate, thresholds["human_grounding_pass_rate_min"]),
    }

    if gt_metrics.get("available"):
        checks["cer_mean"] = _check_leq(gt_metrics.get("cer_mean"), thresholds["cer_max"])
        checks["wer_mean"] = _check_leq(gt_metrics.get("wer_mean"), thresholds["wer_max"])
    else:
        checks["cer_mean"] = {"value": None, "threshold": thresholds["cer_max"], "pass": False}
        checks["wer_mean"] = {"value": None, "threshold": thresholds["wer_max"], "pass": False}

    operational_keys = [
        k for k in checks.keys()
        if k not in {"cer_mean", "wer_mean", "heading_soft_recall", "caption_injected_ratio"}
    ]
    operational_pass = all(bool(checks[k]["pass"]) for k in operational_keys)
    strict_research_pass = operational_pass and bool(checks["cer_mean"]["pass"]) and bool(checks["wer_mean"]["pass"])

    report = {
        "overall": {
            "operational_pass": operational_pass,
            "strict_research_pass": strict_research_pass,
            "strict_research_note": (
                "strict_research requires GT-based CER/WER from manual_verified=true rows"
                if manual_only and (not gt_metrics.get("available"))
                else (
                    "strict_research requires GT-based CER/WER"
                    if not gt_metrics.get("available")
                    else "strict_research uses GT-based CER/WER"
                )
            ),
            "strict_mode": strict_mode,
        },
        "strict_policy": {
            "strict_mode": strict_mode,
            "manual_verified_required": bool(manual_only),
            "min_manual_verified_samples": int(args.min_manual_verified_samples) if manual_only else 1,
            "gt_metrics_available": bool(gt_metrics.get("available", False)),
            "gt_manual_verified_only": bool(gt_metrics.get("manual_verified_only", False)),
        },
        "inputs": {
            "pdf": str(pdf_path.as_posix()),
            "markdown": str(markdown_path.as_posix()),
            "toc": str(Path(args.toc).as_posix()),
            "labels": str(Path(args.labels).as_posix()),
            "gt_jsonl": str(Path(args.gt_jsonl).as_posix()) if str(args.gt_jsonl).strip() else "",
        },
        "dataset": {
            "pdf_pages": pdf_pages,
            "pdf_extractable_text_pages": pdf_extractable_pages,
            "markdown_pages": len(md_pages),
            "page_coverage_ratio": round(page_coverage_ratio, 6) if page_coverage_ratio is not None else None,
            "non_empty_page_ratio": non_empty_ratio,
        },
        "ocr_intrinsic": gt_metrics,
        "encoding_health": enc_report,
        "toc_heading": heading_report,
        "toc_heading_effective": heading_report_effective,
        "figure_reference_eval": figure_ref_report,
        "derived": {
            "tiny_region_after_ratio": round(tiny_after_ratio, 6),
            "crop_completeness_proxy_mean": crop_completeness_mean,
            "retrieval_hit_at_k_mean": retrieval_hit_mean,
            "human_grounding_pass_rate": human_pass_rate,
            "figure_ref_to_region_link_recall": figure_ref_to_region_link_recall,
            "operation_pages_region_coverage": operation_pages_region_coverage,
            "table_cell_visual_coverage": table_cell_visual_coverage,
            "generic_caption_ratio": generic_caption_ratio,
            "sequence_stitch_coverage": sequence_stitch_coverage,
        },
        "thresholds": thresholds,
        "checks": checks,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(report["dataset"], ensure_ascii=False, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
