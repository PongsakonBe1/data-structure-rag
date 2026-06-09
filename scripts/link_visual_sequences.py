"""
Link visual regions into operation sequences across adjacent pages.

Outputs:
- updated region manifest with sequence linkage fields
- logs/visual_sequence_links_latest.json (flat linkage records)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
FIGURE_REF_RE = re.compile(r"(ภาพที่|ตารางที่|figure|table)\s*\d+(?:\.\d+)?", re.IGNORECASE)
OP_RE = re.compile(
    r"(insert|remove|enqueue|dequeue|push|pop|front|rear|"
    r"เพิ่ม|ลบ|แทรก|ดึงข้อมูลออก|นำเข้าข้อมูล|นำข้อมูลออก|ขั้นตอน|การดำเนินการ)",
    re.IGNORECASE,
)


def parse_pages(md_text: str) -> dict[tuple[str, int], str]:
    headers = list(PAGE_HEADER_RE.finditer(md_text))
    out: dict[tuple[str, int], str] = {}
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        body = md_text[start:end].strip()
        out[(m.group(1).strip(), int(m.group(2)))] = body
    return out


def topic_hint_from_terms(terms: list[str], page_text: str) -> str:
    joined = " ".join(terms).lower() + " " + str(page_text or "").lower()
    if any(k in joined for k in ["enqueue", "dequeue", "rear", "front", "คิว"]):
        return "queue"
    if any(k in joined for k in ["linked", "list", "ลิงค์", "โหนด"]):
        return "linked_list"
    if any(k in joined for k in ["tree", "ทรี", "binary"]):
        return "tree"
    if any(k in joined for k in ["array", "อาร์เรย์", "[1]"]):
        return "array"
    if any(k in joined for k in ["stack", "push", "pop", "สแตก"]):
        return "stack"
    return "generic"


def extract_operation_spans(page_text: str, max_spans: int = 8) -> tuple[list[str], list[str], list[str]]:
    lines = [ln.strip() for ln in str(page_text or "").splitlines() if ln.strip()]
    terms: list[str] = []
    spans: list[str] = []
    figure_refs: list[str] = []
    for ln in lines:
        for m in OP_RE.finditer(ln):
            term = m.group(1).strip().lower()
            if term and term not in terms:
                terms.append(term)
        if (OP_RE.search(ln) or FIGURE_REF_RE.search(ln)) and len(spans) < max(1, int(max_spans)):
            spans.append(ln[:240])
        for m in FIGURE_REF_RE.finditer(ln):
            ref = m.group(0).strip()
            if ref and ref not in figure_refs:
                figure_refs.append(ref)
    return terms[:16], spans[: max(1, int(max_spans))], figure_refs[:16]


def assign_sequence_groups(pages: list[dict[str, Any]], page_text_map: dict[tuple[str, int], str]) -> dict[tuple[str, int], dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for p in pages:
        source = str(p.get("source", "")).strip()
        page = int(p.get("page", 0) or 0)
        if not source or page <= 0:
            continue
        by_source.setdefault(source, []).append(p)

    linkage: dict[tuple[str, int], dict[str, Any]] = {}
    for source, rows in by_source.items():
        rows = sorted(rows, key=lambda x: int(x.get("page", 0) or 0))
        seq_idx = 0
        current_group = ""
        current_hint = ""
        current_last_page = -99999
        for row in rows:
            page = int(row.get("page", 0) or 0)
            key = (source, page)
            text = page_text_map.get(key, "")
            terms, spans, fig_refs = extract_operation_spans(text)
            sequence_candidate = bool(row.get("sequence_candidate", False) or len(terms) > 0 or len(spans) >= 2)
            hint = topic_hint_from_terms(terms, text)

            start_new = False
            if not sequence_candidate:
                current_group = ""
                current_hint = ""
                current_last_page = page
                linkage[key] = {
                    "sequence_group_id": "",
                    "topic_hint": hint,
                    "sequence_candidate": False,
                    "operation_context_terms": terms,
                    "operation_spans": spans,
                    "figure_refs_seen": fig_refs,
                }
                continue

            if not current_group:
                start_new = True
            elif page - current_last_page > 1:
                start_new = True
            elif current_hint and hint != current_hint:
                start_new = True

            if start_new:
                seq_idx += 1
                current_group = f"{source}:seq:{seq_idx:03d}:{hint}:{page}"
                current_hint = hint
            current_last_page = page
            linkage[key] = {
                "sequence_group_id": current_group,
                "topic_hint": hint,
                "sequence_candidate": True,
                "operation_context_terms": terms,
                "operation_spans": spans,
                "figure_refs_seen": fig_refs,
            }
    return linkage


def main() -> None:
    ap = argparse.ArgumentParser(description="Link visual step sequences across adjacent pages.")
    ap.add_argument("--markdown", default="final_extracted_text_only_structured_full.md")
    ap.add_argument("--region-manifest", default="logs/figure_regions_manifest_latest.json")
    ap.add_argument("--output-manifest", default="logs/figure_regions_manifest_latest.json")
    ap.add_argument("--output-links", default="logs/visual_sequence_links_latest.json")
    args = ap.parse_args()

    markdown_path = Path(args.markdown)
    manifest_path = Path(args.region_manifest)
    if not markdown_path.exists():
        raise FileNotFoundError(f"missing markdown: {markdown_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing region manifest: {manifest_path}")

    page_text_map = parse_pages(markdown_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = manifest.get("pages", []) if isinstance(manifest, dict) else []
    if not isinstance(pages, list):
        raise ValueError("invalid pages in manifest")

    linkage = assign_sequence_groups(pages, page_text_map)

    link_rows: list[dict[str, Any]] = []
    seq_counts: dict[str, int] = {}
    for page_row in pages:
        source = str(page_row.get("source", "")).strip()
        page = int(page_row.get("page", 0) or 0)
        key = (source, page)
        info = linkage.get(key, {})
        seq_id = str(info.get("sequence_group_id", "")).strip()
        spans = info.get("operation_spans", []) if isinstance(info.get("operation_spans", []), list) else []
        terms = info.get("operation_context_terms", []) if isinstance(info.get("operation_context_terms", []), list) else []
        page_row["sequence_candidate"] = bool(info.get("sequence_candidate", page_row.get("sequence_candidate", False)))
        if seq_id:
            page_row["sequence_group_id"] = seq_id

        flags = page_row.get("linkage_flags", {}) if isinstance(page_row.get("linkage_flags", {}), dict) else {}
        flags["operation_context_terms"] = terms
        flags["operation_spans_count"] = len(spans)
        flags["sequence_group_id"] = seq_id
        page_row["linkage_flags"] = flags

        regions = page_row.get("regions", []) if isinstance(page_row.get("regions", []), list) else []
        regions_sorted = sorted(
            regions,
            key=lambda r: (
                int((r.get("bbox", {}) or {}).get("y0", 10**9) or 10**9),
                int((r.get("bbox", {}) or {}).get("x0", 10**9) or 10**9),
                int(r.get("region_index", 10**9) or 10**9),
            ),
        )
        step_base = int(seq_counts.get(seq_id, 0)) if seq_id else 0
        for idx, r in enumerate(regions_sorted, start=1):
            step_idx = step_base + idx if seq_id else idx
            nearest_text_span = ""
            if spans:
                span_idx = min(len(spans) - 1, max(0, idx - 1))
                nearest_text_span = str(spans[span_idx]).strip()
            r["sequence_group_id"] = seq_id
            r["sequence_step_index"] = int(step_idx)
            r["operation_context_terms"] = terms
            r["nearest_text_span"] = nearest_text_span
            link_rows.append(
                {
                    "id": f"{source}:{page}:region:{int(r.get('region_index', idx) or idx)}",
                    "source": source,
                    "page": page,
                    "region_index": int(r.get("region_index", idx) or idx),
                    "sequence_group_id": seq_id,
                    "sequence_step_index": int(step_idx),
                    "operation_context_terms": terms,
                    "nearest_text_span": nearest_text_span,
                }
            )
        if seq_id:
            seq_counts[seq_id] = step_base + len(regions_sorted)

    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    groups = sorted({str(r.get("sequence_group_id", "")).strip() for r in link_rows if str(r.get("sequence_group_id", "")).strip()})
    payload = {
        "summary": {
            "records": len(link_rows),
            "sequence_groups": len(groups),
            "sequence_candidates_pages": sum(1 for p in pages if bool(p.get("sequence_candidate", False))),
        },
        "sequence_groups": groups,
        "links": link_rows,
    }
    out_links = Path(args.output_links)
    out_links.parent.mkdir(parents=True, exist_ok=True)
    out_links.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"saved_manifest={out_manifest}")
    print(f"saved_links={out_links}")


if __name__ == "__main__":
    main()
