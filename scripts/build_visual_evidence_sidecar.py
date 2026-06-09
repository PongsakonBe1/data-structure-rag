"""
Build structured visual evidence sidecar JSONL from markdown + region manifest.

Output:
- indexes/visual/region_evidence.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
VISUAL_BLOCK_RE = re.compile(r"### Visual Captions \(Auto\)\n([\s\S]*)$", re.MULTILINE)
IMAGE_LINE_RE = re.compile(r"^\s*>\s*\[Image\s+\d+\]\s*(.+?)\s*$")
YAML_FENCE_RE = re.compile(r"```yaml\s*([\s\S]*?)```", re.IGNORECASE)
GENERIC_CAPTION_RE = re.compile(r"(สรุปภาษาไทย\s*1-3\s*ประโยค|บรรยายภาพทั่วไป)", re.IGNORECASE)


def parse_pages(md_text: str) -> list[dict[str, Any]]:
    headers = list(PAGE_HEADER_RE.finditer(md_text))
    pages: list[dict[str, Any]] = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        body = md_text[start:end].strip()
        pages.append({"source": m.group(1).strip(), "page": int(m.group(2)), "body": body})
    return pages


def _parse_value(raw: str) -> Any:
    s = str(raw or "").strip()
    if s in {"true", "false"}:
        return s == "true"
    if s == "null":
        return None
    if re.match(r"^-?\d+(?:\.\d+)?$", s):
        try:
            return float(s) if "." in s else int(s)
        except Exception:
            pass
    if s.startswith('"') or s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def _parse_yaml_block(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ln in str(text or "").splitlines():
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        key = k.strip()
        if not key:
            continue
        out[key] = _parse_value(v.strip())
    return out


def _norm_img(path: str) -> str:
    return str(path or "").strip().replace("\\", "/")


def parse_visual_captions_from_page(page_body: str) -> list[dict[str, Any]]:
    m = VISUAL_BLOCK_RE.search(str(page_body or ""))
    if not m:
        return []
    block = m.group(1)
    lines = block.splitlines()
    image_positions: list[tuple[int, str]] = []
    for idx, ln in enumerate(lines):
        mm = IMAGE_LINE_RE.match(ln)
        if mm:
            image_positions.append((idx, _norm_img(mm.group(1).strip())))
    if not image_positions:
        return []

    captions: list[dict[str, Any]] = []
    for i, (line_idx, image_path) in enumerate(image_positions):
        start = line_idx
        end = image_positions[i + 1][0] if i + 1 < len(image_positions) else len(lines)
        chunk = "\n".join(lines[start:end])
        yaml_match = YAML_FENCE_RE.search(chunk)
        obj: dict[str, Any] = {"image_path": image_path}
        if yaml_match:
            obj.update(_parse_yaml_block(yaml_match.group(1)))
        obj["image_path"] = _norm_img(str(obj.get("image_path", image_path)))
        captions.append(obj)
    return captions


def build_caption_map(markdown_path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    pages = parse_pages(markdown_path.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for p in pages:
        key = (str(p.get("source", "")).strip(), int(p.get("page", 0) or 0))
        caps = parse_visual_captions_from_page(str(p.get("body", "")))
        if caps:
            out[key] = caps
    return out


def _pick_caption_for_region(region_path: str, caps: list[dict[str, Any]]) -> dict[str, Any]:
    rp = _norm_img(region_path)
    for c in caps:
        if _norm_img(str(c.get("image_path", ""))) == rp:
            return c
    for c in caps:
        if _norm_img(str(c.get("image_path", ""))) == _norm_img(rp.split("/")[-1]):
            return c
    return caps[0] if caps else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build visual evidence sidecar JSONL.")
    ap.add_argument("--markdown", default="final_extracted_text_only_structured_full.md")
    ap.add_argument("--region-manifest", default="logs/figure_regions_manifest_latest.json")
    ap.add_argument("--sequence-links", default="logs/visual_sequence_links_latest.json")
    ap.add_argument(
        "--corpus-jsonl",
        default="indexes/colpali/pages.jsonl",
        help="Optional retrieval corpus JSONL; records missing from region manifest will be added as page-level fallback.",
    )
    ap.add_argument("--output-jsonl", default="indexes/visual/region_evidence.jsonl")
    ap.add_argument("--report", default="logs/visual_evidence_sidecar_report_latest.json")
    args = ap.parse_args()

    md_path = Path(args.markdown)
    manifest_path = Path(args.region_manifest)
    if not md_path.exists():
        raise FileNotFoundError(f"missing markdown: {md_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing region manifest: {manifest_path}")

    caption_map = build_caption_map(md_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = manifest.get("pages", []) if isinstance(manifest, dict) else []
    if not isinstance(pages, list):
        raise ValueError("invalid pages in region manifest")

    seq_links_map: dict[str, dict[str, Any]] = {}
    seq_path = Path(args.sequence_links)
    if seq_path.exists():
        try:
            seq_payload = json.loads(seq_path.read_text(encoding="utf-8"))
            for row in seq_payload.get("links", []) if isinstance(seq_payload, dict) else []:
                if not isinstance(row, dict):
                    continue
                rid = str(row.get("id", "")).strip()
                if rid:
                    seq_links_map[rid] = row
        except Exception:
            seq_links_map = {}

    manifest_page_map: dict[tuple[str, int], dict[str, Any]] = {}
    for page_row in pages:
        if not isinstance(page_row, dict):
            continue
        source = str(page_row.get("source", "")).strip()
        page = int(page_row.get("page", 0) or 0)
        if source and page > 0:
            manifest_page_map[(source, page)] = page_row

    corpus_rows: list[dict[str, Any]] = []
    corpus_path = Path(args.corpus_jsonl)
    if corpus_path.exists():
        try:
            with corpus_path.open("r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        corpus_rows.append(obj)
        except Exception:
            corpus_rows = []

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with_caption = 0
    with_steps = 0
    with_table_cell_visuals = 0
    table_records = 0
    table_records_with_cell_visuals = 0
    with_sequence = 0
    generic_caption_count = 0
    uncertainty_count = 0
    fallback_from_corpus = 0
    written_ids: set[str] = set()

    with out_path.open("w", encoding="utf-8") as f:
        for page_row in pages:
            source = str(page_row.get("source", "")).strip()
            page = int(page_row.get("page", 0) or 0)
            key = (source, page)
            caps = caption_map.get(key, [])
            regions = page_row.get("regions", []) if isinstance(page_row.get("regions", []), list) else []
            for r in regions:
                if not isinstance(r, dict):
                    continue
                ridx = int(r.get("region_index", 0) or 0)
                region_path = _norm_img(str(r.get("path", "")))
                rec_id = f"{source}:{page}:region:{ridx if ridx > 0 else 1}"
                cap = _pick_caption_for_region(region_path, caps)
                seq_row = seq_links_map.get(rec_id, {})

                caption_th = str(cap.get("caption_th", "")).strip()
                table_cell_visuals = cap.get("table_cell_visuals", []) if isinstance(cap.get("table_cell_visuals", []), list) else []
                diagram_steps = cap.get("diagram_steps", []) if isinstance(cap.get("diagram_steps", []), list) else []
                entities = cap.get("entities", []) if isinstance(cap.get("entities", []), list) else []
                figure_refs_seen = cap.get("figure_refs_seen", []) if isinstance(cap.get("figure_refs_seen", []), list) else []

                seq_group_id = str(r.get("sequence_group_id", "") or seq_row.get("sequence_group_id", "")).strip()
                seq_step_idx = int(r.get("sequence_step_index", 0) or seq_row.get("sequence_step_index", 0) or 0)
                nearest_text_span = str(r.get("nearest_text_span", "") or seq_row.get("nearest_text_span", "")).strip()
                op_terms = (
                    r.get("operation_context_terms", [])
                    if isinstance(r.get("operation_context_terms", []), list)
                    else seq_row.get("operation_context_terms", [])
                )
                if not isinstance(op_terms, list):
                    op_terms = []
                if (not diagram_steps) and nearest_text_span and op_terms:
                    diagram_steps = [nearest_text_span[:240]]

                row = {
                    "id": rec_id,
                    "source": source,
                    "page": page,
                    "page_id": f"{source}:{page}",
                    "region_path": region_path,
                    "bbox": r.get("bbox", {}),
                    "visual_type": str(cap.get("visual_type", "other")).strip().lower() or "other",
                    "caption_th": caption_th,
                    "diagram_steps": diagram_steps,
                    "entities": entities,
                    "table_markdown": str(cap.get("table_markdown", "")).strip(),
                    "table_cell_visuals": table_cell_visuals,
                    "figure_refs_seen": figure_refs_seen,
                    "confidence": float(cap.get("confidence", 0.0) or 0.0),
                    "uncertainty_flag": bool(cap.get("uncertainty_flag", False)),
                    "evidence_span_hint": str(cap.get("evidence_span_hint", "")).strip(),
                    "linkage": {
                        "nearest_text_span": nearest_text_span,
                        "operation_context_terms": op_terms,
                        "sequence_group_id": seq_group_id,
                        "sequence_step_index": seq_step_idx,
                    },
                    "region_detection_method": str(r.get("region_detection_method", "")).strip(),
                    "target_reason": page_row.get("target_reason", []),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_ids.add(str(row.get("id", "")).strip())

                total += 1
                if caption_th:
                    with_caption += 1
                if diagram_steps:
                    with_steps += 1
                if table_cell_visuals:
                    with_table_cell_visuals += 1
                if str(row.get("visual_type", "")).strip().lower() == "table":
                    table_records += 1
                    if table_cell_visuals:
                        table_records_with_cell_visuals += 1
                if seq_group_id:
                    with_sequence += 1
                if GENERIC_CAPTION_RE.search(caption_th):
                    generic_caption_count += 1
                if bool(row.get("uncertainty_flag", False)):
                    uncertainty_count += 1

        for rec in corpus_rows:
            rec_id = str(rec.get("id", "")).strip()
            if not rec_id or rec_id in written_ids:
                continue
            source = str(rec.get("source", "")).strip()
            page = int(rec.get("page", 0) or 0)
            page_id = str(rec.get("page_id", "")).strip() or f"{source}:{page}"
            region_path = _norm_img(str(rec.get("image_path", "")).strip())
            if not source or page <= 0:
                continue

            caps = caption_map.get((source, page), [])
            cap = _pick_caption_for_region(region_path, caps) if caps else {}
            seq_row = seq_links_map.get(rec_id, {})
            page_manifest = manifest_page_map.get((source, page), {})
            target_reason = page_manifest.get("target_reason", []) if isinstance(page_manifest, dict) else []
            if not isinstance(target_reason, list):
                target_reason = []
            nearest_text_span = str(seq_row.get("nearest_text_span", "")).strip()
            op_terms = seq_row.get("operation_context_terms", [])
            if not isinstance(op_terms, list):
                op_terms = []
            seq_group_id = str(seq_row.get("sequence_group_id", "")).strip()
            seq_step_idx = int(seq_row.get("sequence_step_index", 0) or 0)
            table_cell_visuals = cap.get("table_cell_visuals", []) if isinstance(cap.get("table_cell_visuals", []), list) else []
            diagram_steps = cap.get("diagram_steps", []) if isinstance(cap.get("diagram_steps", []), list) else []
            entities = cap.get("entities", []) if isinstance(cap.get("entities", []), list) else []
            figure_refs_seen = cap.get("figure_refs_seen", []) if isinstance(cap.get("figure_refs_seen", []), list) else []
            caption_th = str(cap.get("caption_th", "")).strip()

            row = {
                "id": rec_id,
                "source": source,
                "page": page,
                "page_id": page_id,
                "region_path": region_path,
                "bbox": {},
                "visual_type": str(cap.get("visual_type", "other")).strip().lower() or "other",
                "caption_th": caption_th,
                "diagram_steps": diagram_steps,
                "entities": entities,
                "table_markdown": str(cap.get("table_markdown", "")).strip(),
                "table_cell_visuals": table_cell_visuals,
                "figure_refs_seen": figure_refs_seen,
                "confidence": float(cap.get("confidence", 0.0) or 0.0),
                "uncertainty_flag": bool(cap.get("uncertainty_flag", True)),
                "evidence_span_hint": str(cap.get("evidence_span_hint", "")).strip(),
                "linkage": {
                    "nearest_text_span": nearest_text_span,
                    "operation_context_terms": op_terms,
                    "sequence_group_id": seq_group_id,
                    "sequence_step_index": seq_step_idx,
                },
                "region_detection_method": "corpus_page_fallback",
                "target_reason": target_reason,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written_ids.add(rec_id)
            fallback_from_corpus += 1
            total += 1
            if caption_th:
                with_caption += 1
            if diagram_steps:
                with_steps += 1
            if table_cell_visuals:
                with_table_cell_visuals += 1
            if str(row.get("visual_type", "")).strip().lower() == "table":
                table_records += 1
                if table_cell_visuals:
                    table_records_with_cell_visuals += 1
            if seq_group_id:
                with_sequence += 1
            if GENERIC_CAPTION_RE.search(caption_th):
                generic_caption_count += 1
            if bool(row.get("uncertainty_flag", False)):
                uncertainty_count += 1

    report = {
        "total_records": total,
        "with_caption": with_caption,
        "with_caption_ratio": round(with_caption / max(1, total), 6),
        "with_diagram_steps": with_steps,
        "with_diagram_steps_ratio": round(with_steps / max(1, total), 6),
        "with_table_cell_visuals": with_table_cell_visuals,
        "with_table_cell_visuals_ratio": round(with_table_cell_visuals / max(1, total), 6),
        "table_records": table_records,
        "table_records_with_cell_visuals": table_records_with_cell_visuals,
        "table_cell_visual_coverage_table_only": round(
            table_records_with_cell_visuals / max(1, table_records), 6
        ),
        "with_sequence_group": with_sequence,
        "with_sequence_group_ratio": round(with_sequence / max(1, total), 6),
        "generic_caption_ratio": round(generic_caption_count / max(1, total), 6),
        "uncertainty_ratio": round(uncertainty_count / max(1, total), 6),
        "fallback_from_corpus_records": int(fallback_from_corpus),
        "output_jsonl": str(out_path.as_posix()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved_jsonl={out_path}")
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    main()
