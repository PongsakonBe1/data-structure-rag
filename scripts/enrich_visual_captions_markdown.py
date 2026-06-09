"""
Enrich markdown pages with structured VLM visual captions.

Adds deterministic "Visual Captions (Auto)" blocks with fields designed for:
- unlabeled step-by-step images
- table-cell visuals (symbols/diagrams inside table cells)
- downstream sidecar evidence building
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


ROOT = Path(__file__).resolve().parent.parent
PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
FIGURE_REF_RE = re.compile(r"(ภาพที่|ตารางที่|figure|table)\s*\d+(?:\.\d+)?", re.IGNORECASE)
VISUAL_BLOCK_RE = re.compile(r"\n### Visual Captions \(Auto\)\n[\s\S]*$", re.MULTILINE)
VISUAL_ANCHOR_BLOCK_RE = re.compile(
    r"\n### Visual Anchors \(Auto\)\n[\s\S]*?(?=\n### Visual Captions \(Auto\)\n|\Z)",
    re.MULTILINE,
)
OPERATION_HINT_RE = re.compile(
    r"(insert|remove|enqueue|dequeue|push|pop|front|rear|เพิ่ม|ลบ|แทรก|ขั้นตอน|การดำเนินการ)",
    re.IGNORECASE,
)
TABLE_HINT_RE = re.compile(r"(ตาราง|table|\|.+\|.+\|)", re.IGNORECASE)
GENERIC_PATTERNS = [
    re.compile(r"^(สรุปภาษาไทย\s*1-3\s*ประโยค|บรรยายภาพทั่วไป)$", re.IGNORECASE),
    re.compile(r"^(ไม่สามารถ|ไม่ชัดเจน)$", re.IGNORECASE),
]
PLACEHOLDER_TEXT_RE = re.compile(
    r"^(caption_th|summary|evidence_span_hint|key_elements|diagram_steps|entities|n/?a|none|null|unknown)$",
    re.IGNORECASE,
)
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
GENERIC_EVIDENCE_HINT_RE = re.compile(r"^(ภาพ|รูป|ข้อมูลในภาพ|ไม่มี|none|n/?a)$", re.IGNORECASE)
GENERIC_CAPTION_EXTRA_RE = re.compile(r"^ภาพที่\s*\d+(?:\.\d+)?$", re.IGNORECASE)
META_ARTIFACT_LINE_RE = re.compile(
    r"^(figure_refs_seen:|caption_th:|table_markdown:|diagram_steps:|entities:|evidence_span_hint:|> \[|```)",
    re.IGNORECASE,
)

CN_TERM_MAP = {
    "根节点": "โหนดราก",
    "子节点": "โหนดลูก",
    "左子节点": "โหนดลูกซ้าย",
    "右子节点": "โหนดลูกขวา",
}


def parse_pages(md_text: str) -> list[dict[str, Any]]:
    headers = list(PAGE_HEADER_RE.finditer(md_text))
    pages: list[dict[str, Any]] = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        body = md_text[start:end].strip()
        pages.append({"source": m.group(1).strip(), "page": int(m.group(2)), "content": body})
    return pages


def strip_auto_visual_blocks(content: str) -> str:
    s = str(content or "")
    s = VISUAL_BLOCK_RE.sub("", s)
    s = VISUAL_ANCHOR_BLOCK_RE.sub("", s)
    return s.strip()


def rebuild_markdown(pages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for p in pages:
        chunks.append(f"# Source: {p['source']}\n## Page {p['page']}\n\n{str(p['content']).strip()}\n")
    return "\n".join(chunks).strip() + "\n"


def _abs_image(path_str: str) -> Path:
    p = Path(str(path_str or "").strip())
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_region_candidates(
    region_manifest_path: Path,
    max_images_per_page: int,
    min_region_score: float,
) -> dict[str, list[dict[str, Any]]]:
    if not region_manifest_path.exists():
        return {}
    payload = json.loads(region_manifest_path.read_text(encoding="utf-8"))
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    out: dict[str, list[dict[str, Any]]] = {}
    for row in pages:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "")).strip()
        page = int(row.get("page", 0) or 0)
        if not source or page <= 0:
            continue
        page_id = f"{source}:{page}"
        regions = row.get("regions", []) if isinstance(row.get("regions"), list) else []
        ranked: list[tuple[float, float, dict[str, Any]]] = []
        for r in regions:
            if not isinstance(r, dict):
                continue
            path = str(r.get("path", "")).strip()
            score = float(r.get("score", 0.0) or 0.0)
            area = float(r.get("area_ratio", 0.0) or 0.0)
            if not path:
                continue
            if score < float(min_region_score):
                continue
            ranked.append((score, area, r))
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = [x[2] for x in ranked[: max(1, int(max_images_per_page))]]
        if not selected:
            page_img = str(row.get("image_path", "")).strip()
            if page_img:
                selected = [{"path": page_img, "score": 0.0, "area_ratio": 1.0, "is_fallback": True}]
        if selected:
            page_img = str(row.get("image_path", "")).strip()
            for s in selected:
                if not isinstance(s, dict):
                    continue
                s.setdefault("_page_image_path", page_img)
                s.setdefault("_source", source)
                s.setdefault("_page", page)
        if selected:
            out[page_id] = selected
    return out


def load_page_image_fallback(figure_manifest_path: Path) -> dict[str, list[dict[str, Any]]]:
    if not figure_manifest_path.exists():
        return {}
    try:
        payload = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    source_name = Path(str(payload.get("pdf", "") or "")).name
    if not source_name:
        return {}

    out: dict[str, list[dict[str, Any]]] = {}
    images = payload.get("images", []) if isinstance(payload, dict) else []
    for item in images if isinstance(images, list) else []:
        if not isinstance(item, dict):
            continue
        page = int(item.get("page", 0) or 0)
        path = str(item.get("path", "")).strip()
        if page <= 0 or not path:
            continue
        page_id = f"{source_name}:{page}"
        out.setdefault(page_id, []).append(
            {
                "path": path,
                "score": 0.0,
                "area_ratio": 1.0,
                "is_fallback": True,
                "region_detection_method": "figure_manifest_page_fallback",
            }
        )
    return out


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json_like(text: str) -> dict[str, Any]:
    s = str(text or "").strip()
    if not s:
        return {}
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def image_cache_key(img_path: Path, model: str, prompt_version: str) -> str:
    h = hashlib.sha1()
    h.update(str(model).encode("utf-8"))
    h.update(str(prompt_version).encode("utf-8"))
    h.update(img_path.read_bytes())
    return h.hexdigest()


def should_caption(page_text: str, page_images: list[dict[str, Any]], *, require_hints: bool) -> bool:
    if not page_images:
        return False
    # Default policy: caption any page that has usable visual asset.
    # Enable `--caption-require-hints` when needing narrower, hint-driven runs.
    if not require_hints:
        return True
    text = str(page_text or "")
    if FIGURE_REF_RE.search(text):
        return True
    if TABLE_HINT_RE.search(text):
        return True
    if OPERATION_HINT_RE.search(text):
        return True
    hints = ("[Structure:", "โครงสร้าง", "diagram", "table", "flowchart")
    return any(h.lower() in text.lower() for h in hints)


def _normalize_list(v: Any, max_items: int = 8) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        s = str(x).strip()
        if s:
            out.append(s)
        if len(out) >= max(1, int(max_items)):
            break
    return out


def _normalize_table_cell_visuals(v: Any, max_items: int = 16) -> list[dict[str, Any]]:
    if not isinstance(v, list):
        return []
    out: list[dict[str, Any]] = []
    for item in v:
        if not isinstance(item, dict):
            continue
        row = item.get("row")
        col = item.get("col")
        cell_text = str(item.get("cell_text", "")).strip()
        cap = str(item.get("cell_visual_caption", "")).strip()
        if PLACEHOLDER_TEXT_RE.match(cap):
            cap = ""
        try:
            row_i = int(row)
        except Exception:
            row_i = 0
        try:
            col_i = int(col)
        except Exception:
            col_i = 0
        if row_i <= 0 and col_i <= 0 and not cap and not cell_text:
            continue
        out.append(
            {
                "row": row_i if row_i > 0 else None,
                "col": col_i if col_i > 0 else None,
                "cell_text": cell_text,
                "cell_visual_caption": cap,
            }
        )
        if len(out) >= max(1, int(max_items)):
            break
    return out


def _extract_markdown_table_block(page_text: str) -> str:
    lines = str(page_text or "").splitlines()
    blocks: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("|") and "|" in s[1:]:
            cur.append(s)
        else:
            if len(cur) >= 2:
                blocks.append(cur)
            cur = []
    if len(cur) >= 2:
        blocks.append(cur)
    if not blocks:
        return ""
    best = max(blocks, key=len)
    return "\n".join(best).strip()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for x in items:
        s = str(x).strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _clean_figure_refs(refs: list[str], page_text: str) -> list[str]:
    page_refs = _dedupe_keep_order([m.group(0).strip() for m in FIGURE_REF_RE.finditer(str(page_text or ""))])
    if not page_refs:
        return []
    page_refs_norm = {r.lower() for r in page_refs}
    out = _dedupe_keep_order([str(r).strip() for r in refs])
    if not out:
        return page_refs
    matched = [x for x in out if x.lower() in page_refs_norm]
    return matched if matched else page_refs


def _contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(str(text or "")))


def _normalize_multilingual_text(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    for k, v in CN_TERM_MAP.items():
        s = s.replace(k, v)
    # If CJK still exists after replacement, drop to avoid cross-language leakage.
    if _contains_cjk(s):
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_table_markdown(table_md: str, page_text: str) -> str:
    md = str(table_md or "").strip()
    page_refs = _dedupe_keep_order([m.group(0).strip() for m in FIGURE_REF_RE.finditer(str(page_text or ""))])
    table_refs = _dedupe_keep_order([m.group(0).strip() for m in FIGURE_REF_RE.finditer(md)])
    if not md:
        return ""
    if not table_refs:
        return md
    page_ref_set = {x.lower() for x in page_refs}
    # If table contains ref IDs not in this page context, prefer deterministic table from page text.
    has_cross_page_ref = (not page_ref_set) or any(r.lower() not in page_ref_set for r in table_refs)
    if has_cross_page_ref:
        extracted = _extract_markdown_table_block(page_text)
        if extracted:
            return extracted
    return md


def _extract_figure_refs_from_text(text: str) -> list[str]:
    return _dedupe_keep_order([m.group(0).strip() for m in FIGURE_REF_RE.finditer(str(text or ""))])


def _clean_context_span(text: str) -> str:
    lines = str(text or "").splitlines()
    kept: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if META_ARTIFACT_LINE_RE.match(s):
            continue
        if s.startswith("- [VA-") or s.startswith("source=") or s.startswith("target_page=") or s.startswith("target_reason="):
            continue
        if ("path=" in s and "bbox=" in s) or "method=" in s:
            continue
        kept.append(s)
    if not kept:
        return ""
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def _extract_operation_steps(text: str, max_items: int = 4) -> list[str]:
    lines = str(text or "").splitlines()
    out: list[str] = []
    for ln in lines:
        s = ln.strip().lstrip("-").strip()
        if not s:
            continue
        if META_ARTIFACT_LINE_RE.match(s):
            continue
        if not OPERATION_HINT_RE.search(s):
            continue
        out.append(s[:220])
        if len(out) >= max(1, int(max_items)):
            break
    return _dedupe_keep_order(out)


def _is_operation_visual_context(page_text: str, region_context: dict[str, Any] | None) -> bool:
    page_text = strip_auto_visual_blocks(page_text)
    ctx = region_context or {}
    op_terms = ctx.get("operation_context_terms", [])
    if isinstance(op_terms, list) and any(str(t).strip() for t in op_terms):
        return True
    nearest = _clean_context_span(str(ctx.get("nearest_text_span", "") or ""))
    if OPERATION_HINT_RE.search(nearest):
        return True
    return bool(OPERATION_HINT_RE.search(str(page_text or "")))


def _looks_like_state_snapshot_table(table_md: str) -> bool:
    md = str(table_md or "").strip()
    if not md:
        return False
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    if not lines:
        return False
    if len(lines) > 4:
        return False
    has_header_words = bool(re.search(r"(รูปแบบ|นิพจน์|เปรียบเทียบ|ตารางที่|หัวข้อ|ชนิด)", md, re.IGNORECASE))
    if has_header_words:
        return False
    # queue/stack state snapshots usually contain index/cell tokens.
    return bool(re.search(r"(\[\d+\]|front|rear|enqueue|dequeue|insert|remove|[A-Z])", md, re.IGNORECASE))


def _state_snapshot_signal_score(
    *,
    table_md: str,
    key_elements: list[str],
    nearest_span: str,
    caption: str,
) -> int:
    score = 0
    haystack = " ".join(
        [
            str(table_md or ""),
            " ".join(str(x) for x in key_elements),
            str(nearest_span or ""),
            str(caption or ""),
        ]
    )
    if re.search(r"\[\d{1,2}\]", haystack):
        score += 1
    if re.search(r"\b(front|rear|top|head|tail)\b", haystack, re.IGNORECASE):
        score += 1
    if re.search(r"\b(insert|remove|enqueue|dequeue|push|pop)\b", haystack, re.IGNORECASE):
        score += 1
    if len(re.findall(r"\b[A-Z]\b", haystack)) >= 2:
        score += 1
    return score


def build_visual_qa_prompt(*, context_hint: str, page_refs: list[str]) -> str:
    refs = ", ".join(page_refs[:6]) if page_refs else "-"
    hint = str(context_hint or "").strip()[:260]
    return (
        "คุณเป็นระบบตรวจคุณภาพหลักฐานเชิงภาพ (visual QA) สำหรับเอกสารโครงสร้างข้อมูล\n"
        "ตอบเป็น JSON object เท่านั้น ห้ามมีข้อความอื่น\n"
        "ให้ระบุเฉพาะสิ่งที่เห็นได้จริงในภาพ ไม่เดา\n"
        "schema:\n"
        "{"
        '"image_kind":"state_array|table|diagram|chart|other",'
        '"has_index_labels":false,'
        '"index_labels":["[1]","[2]"],'
        '"has_arrows":false,'
        '"arrow_labels":["front","rear"],'
        '"cell_tokens":["A","B","C"],'
        '"operation_terms":["enqueue","dequeue","insert","remove","push","pop"],'
        '"confidence":0.0'
        "}\n"
        f"context_hint: {hint}\n"
        f"page_local_refs: {refs}\n"
    )


def _normalize_visual_qa_obj(obj: dict[str, Any]) -> dict[str, Any]:
    kind = str(obj.get("image_kind", "other")).strip().lower() or "other"
    if kind not in {"state_array", "table", "diagram", "chart", "other"}:
        kind = "other"
    idx = _dedupe_keep_order(_normalize_list(obj.get("index_labels", []), max_items=20))
    arrows = _dedupe_keep_order(_normalize_list(obj.get("arrow_labels", []), max_items=12))
    cells = _dedupe_keep_order(_normalize_list(obj.get("cell_tokens", []), max_items=24))
    ops = _dedupe_keep_order(_normalize_list(obj.get("operation_terms", []), max_items=12))
    raw_conf = obj.get("confidence", 0.0)
    try:
        conf = max(0.0, min(1.0, float(raw_conf or 0.0)))
    except Exception:
        conf = 0.0
    has_idx = bool(obj.get("has_index_labels", False)) or bool(idx)
    has_arrows = bool(obj.get("has_arrows", False)) or bool(arrows)
    signal_count = int(has_idx) + int(has_arrows) + int(bool(cells)) + int(bool(ops))
    # Some model outputs omit/zero confidence despite strong visual signals.
    if conf <= 0.0 and signal_count >= 3 and (has_idx or has_arrows or len(cells) >= 2):
        conf = 0.7
    elif conf <= 0.0 and signal_count >= 2:
        conf = 0.55
    return {
        "image_kind": kind,
        "has_index_labels": bool(has_idx),
        "index_labels": idx,
        "has_arrows": bool(has_arrows),
        "arrow_labels": arrows,
        "cell_tokens": cells,
        "operation_terms": ops,
        "confidence": round(conf, 3),
        "signal_count": int(signal_count),
    }


def _derive_visual_qa_from_text(raw_text: str) -> dict[str, Any]:
    txt = str(raw_text or "").strip()
    low = txt.lower()
    kind = "other"
    if "state array" in low or "queue" in low or "คิว" in txt:
        kind = "state_array"
    elif "table" in low or "ตาราง" in txt:
        kind = "table"
    elif "diagram" in low or "แผนภาพ" in txt:
        kind = "diagram"
    elif "chart" in low:
        kind = "chart"

    index_labels = _dedupe_keep_order(re.findall(r"\[\d{1,2}\]", txt))
    has_index = bool(index_labels)
    arrow_labels = []
    for t in ["front", "rear", "top", "head", "tail"]:
        if re.search(rf"\b{re.escape(t)}\b", low):
            arrow_labels.append(t)
    if "หน้า" in txt and "คิว" in txt:
        arrow_labels.append("front")
    if "ท้าย" in txt and "คิว" in txt:
        arrow_labels.append("rear")
    arrow_labels = _dedupe_keep_order(arrow_labels)
    has_arrows = bool(arrow_labels)

    cell_tokens = _dedupe_keep_order(re.findall(r"\b[A-Z]\b", txt))[:24]
    op_terms = []
    for t in ["enqueue", "dequeue", "insert", "remove", "push", "pop"]:
        if re.search(rf"\b{re.escape(t)}\b", low):
            op_terms.append(t)
    op_terms = _dedupe_keep_order(op_terms)

    signal_count = int(has_index) + int(has_arrows) + int(bool(cell_tokens)) + int(bool(op_terms))
    conf = 0.0
    if signal_count >= 3:
        conf = 0.7
    elif signal_count >= 2:
        conf = 0.55
    elif signal_count >= 1:
        conf = 0.4

    return {
        "image_kind": kind,
        "has_index_labels": has_index,
        "index_labels": index_labels,
        "has_arrows": has_arrows,
        "arrow_labels": arrow_labels,
        "cell_tokens": cell_tokens,
        "operation_terms": op_terms,
        "confidence": conf,
    }


def visual_qa_one_image(
    img_path: Path,
    *,
    model: str,
    hf_token: str,
    timeout_sec: int,
    page_text: str,
    region_context: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    region_context = region_context or {}
    page_text = strip_auto_visual_blocks(page_text)
    nearest = _clean_context_span(str(region_context.get("nearest_text_span", "") or ""))
    ctx_hint = nearest if nearest else _best_evidence_hint(page_text)
    page_refs = _extract_figure_refs_from_text(page_text)

    raw = img_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    client = InferenceClient(api_key=hf_token, timeout=max(10, int(timeout_sec)))
    prompt = build_visual_qa_prompt(context_hint=ctx_hint, page_refs=page_refs)
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        res = client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=420,
            temperature=0.0,
            top_p=0.9,
        )
        raw_text = str(res.choices[0].message.content or "").strip()
        obj = parse_json_like(raw_text)
        if not obj:
            return _normalize_visual_qa_obj(_derive_visual_qa_from_text(raw_text)), "fallback_from_text"
        has_expected = any(k in obj for k in ["image_kind", "has_index_labels", "has_arrows", "cell_tokens", "operation_terms"])
        if not has_expected:
            merged = dict(_derive_visual_qa_from_text(raw_text))
            merged.update({k: v for k, v in obj.items() if k in merged})
            return _normalize_visual_qa_obj(merged), "fallback_from_text"
        return _normalize_visual_qa_obj(obj), "ok"
    except Exception as exc:
        return _normalize_visual_qa_obj({}), f"qa_error:{exc}"


def validate_visual_qa_gate(
    cap: dict[str, Any],
    qa: dict[str, Any],
    *,
    page_text: str,
    region_context: dict[str, Any] | None,
    min_confidence: float,
    min_signal_count: int,
) -> tuple[bool, str]:
    vtype = str(cap.get("visual_type", "other")).strip().lower() or "other"
    op_context = _is_operation_visual_context(page_text, region_context or {})
    qa_conf = float(qa.get("confidence", 0.0) or 0.0)
    signal_count = int(qa.get("signal_count", 0) or 0)
    has_shape_signal = bool(qa.get("has_index_labels", False) or qa.get("has_arrows", False) or qa.get("cell_tokens"))

    if not op_context and vtype not in {"table", "diagram", "chart"}:
        return True, "skip_non_visual"
    if qa_conf < float(min_confidence):
        if not (signal_count >= max(2, int(min_signal_count) + 1) and has_shape_signal):
            return False, "low_confidence"
    if signal_count < max(1, int(min_signal_count)):
        return False, "low_signal_count"
    if vtype in {"diagram", "chart"} and not has_shape_signal and not cap.get("diagram_steps"):
        return False, "missing_structure_signal"
    if vtype == "table" and not (qa.get("cell_tokens") or cap.get("table_markdown")):
        return False, "table_no_cells"
    return True, "ok"


def _build_expanded_crop_from_region(
    *,
    region_context: dict[str, Any] | None,
    expand_factor: float,
) -> Path | None:
    ctx = region_context or {}
    page_img = str(ctx.get("_page_image_path", "")).strip()
    bbox_obj = ctx.get("bbox", {})
    if not page_img or not isinstance(bbox_obj, dict):
        return None
    abs_page = _abs_image(page_img)
    if not abs_page.exists():
        return None
    try:
        from PIL import Image

        with Image.open(abs_page) as im:
            w, h = im.size
            x0 = int(bbox_obj.get("x0", 0) or 0)
            y0 = int(bbox_obj.get("y0", 0) or 0)
            x1 = int(bbox_obj.get("x1", 0) or 0)
            y1 = int(bbox_obj.get("y1", 0) or 0)
            if x1 <= x0 or y1 <= y0:
                return None

            bw = x1 - x0
            bh = y1 - y0
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            nw = int(math.ceil(bw * max(1.0, float(expand_factor))))
            nh = int(math.ceil(bh * max(1.0, float(expand_factor))))
            nx0 = max(0, int(round(cx - nw / 2.0)))
            ny0 = max(0, int(round(cy - nh / 2.0)))
            nx1 = min(w, int(round(cx + nw / 2.0)))
            ny1 = min(h, int(round(cy + nh / 2.0)))
            if nx1 <= nx0 or ny1 <= ny0:
                return None

            crop = im.crop((nx0, ny0, nx1, ny1))
            out_dir = ROOT / "tmp" / "visual_qa_expanded"
            out_dir.mkdir(parents=True, exist_ok=True)
            page_no = int(ctx.get("_page", 0) or 0)
            region_idx = int(ctx.get("region_index", 0) or 0)
            out_path = out_dir / f"page_{page_no:03d}_region_{region_idx:02d}_exp_{int(float(expand_factor)*100):03d}.png"
            crop.save(out_path)
            return out_path
    except Exception:
        return None


def _has_offpage_ref(text: str, page_refs: list[str]) -> bool:
    refs = _extract_figure_refs_from_text(text)
    if not refs:
        return False
    page_ref_set = {x.lower() for x in (page_refs or [])}
    if not page_ref_set:
        return True
    return any(r.lower() not in page_ref_set for r in refs)


def _best_evidence_hint(page_text: str) -> str:
    text = str(page_text or "")
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if META_ARTIFACT_LINE_RE.match(s):
            continue
        if FIGURE_REF_RE.search(s):
            return s[:240]
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if META_ARTIFACT_LINE_RE.match(s):
            continue
        if OPERATION_HINT_RE.search(s) or TABLE_HINT_RE.search(s):
            return s[:240]
    merged = re.sub(r"\s+", " ", text).strip()
    return merged[:240]


def _is_weak_evidence_hint(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return True
    if GENERIC_EVIDENCE_HINT_RE.match(s):
        return True
    if len(s) < 6:
        return True
    return False


def _is_generic_caption(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return True
    if PLACEHOLDER_TEXT_RE.match(s):
        return True
    if GENERIC_CAPTION_EXTRA_RE.match(s):
        return True
    if len(s) < 10:
        return True
    for p in GENERIC_PATTERNS:
        if p.search(s):
            return True
    return False


def _caption_out_of_page_context(caption: str, page_text: str) -> bool:
    c = str(caption or "").strip()
    p = str(page_text or "")
    if not c:
        return False
    # direct mention in page text => trusted
    if c in p:
        return False
    # If caption is very short and non-generic, allow.
    if len(c) <= 6 and not _is_generic_caption(c):
        return False
    # Fallback condition: caption not present and looks model-invented for this page.
    return True


def validate_caption_obj(obj: dict[str, Any]) -> tuple[bool, str]:
    caption = str(obj.get("caption_th", "")).strip()
    vtype = str(obj.get("visual_type", "other")).strip().lower() or "other"
    table_md = str(obj.get("table_markdown", "")).strip()
    table_cells = _normalize_table_cell_visuals(obj.get("table_cell_visuals", []))
    steps = _normalize_list(obj.get("diagram_steps", []), max_items=12)
    entities = _normalize_list(obj.get("entities", []), max_items=12)
    evidence_hint = str(obj.get("evidence_span_hint", "")).strip()

    if _is_generic_caption(caption):
        return False, "generic_caption"
    if PLACEHOLDER_TEXT_RE.match(evidence_hint):
        return False, "placeholder_evidence_hint"

    if vtype == "table" and (not table_md) and (not table_cells):
        return False, "table_missing_structure"
    if vtype == "table" and table_cells:
        bad_cells = 0
        for c in table_cells:
            cap = str(c.get("cell_visual_caption", "")).strip()
            if cap and PLACEHOLDER_TEXT_RE.match(cap):
                bad_cells += 1
        if bad_cells > 0:
            return False, "table_cell_placeholder"

    if vtype == "diagram" and (not steps) and (not entities):
        return False, "diagram_missing_steps_entities"

    return True, "ok"


def normalize_caption_obj(
    obj: dict[str, Any],
    *,
    image_path: str,
    prompt_version: str,
    model: str,
    page_text: str,
    region_context: dict[str, Any] | None = None,
    balanced_uncertainty_threshold: float,
) -> dict[str, Any]:
    region_context = region_context or {}
    page_text_clean = strip_auto_visual_blocks(page_text)
    vtype = str(obj.get("visual_type", "other")).strip().lower() or "other"
    if vtype not in {"diagram", "table", "chart", "other"}:
        vtype = "other"
    caption = str(obj.get("caption_th", "")).strip()
    if PLACEHOLDER_TEXT_RE.match(caption):
        caption = ""
    key_elements = _normalize_list(obj.get("key_elements", []), max_items=8)
    table_md = str(obj.get("table_markdown", "")).strip()
    if PLACEHOLDER_TEXT_RE.match(table_md):
        table_md = ""
    if not table_md:
        table_md = _extract_markdown_table_block(page_text_clean)
    table_md = _sanitize_table_markdown(table_md, page_text_clean)
    page_refs_local = _extract_figure_refs_from_text(page_text_clean)
    figure_refs_seen = _normalize_list(obj.get("figure_refs_seen", []), max_items=8)
    figure_refs_seen = _clean_figure_refs(figure_refs_seen, page_text_clean)
    diagram_steps = [_normalize_multilingual_text(x) for x in _normalize_list(obj.get("diagram_steps", []), max_items=12)]
    entities = [_normalize_multilingual_text(x) for x in _normalize_list(obj.get("entities", []), max_items=12)]
    diagram_steps = [x for x in diagram_steps if x]
    entities = [x for x in entities if x]
    table_cell_visuals = _normalize_table_cell_visuals(obj.get("table_cell_visuals", []), max_items=20)
    evidence_span_hint = str(obj.get("evidence_span_hint", "")).strip()
    if PLACEHOLDER_TEXT_RE.match(evidence_span_hint):
        evidence_span_hint = ""
    nearest_span = _clean_context_span(str(region_context.get("nearest_text_span", "") or ""))
    nearest_refs = _extract_figure_refs_from_text(nearest_span)
    if nearest_refs:
        figure_refs_seen = _clean_figure_refs(nearest_refs, page_text)
    if not evidence_span_hint and nearest_span:
        evidence_span_hint = nearest_span[:240]
    if _is_weak_evidence_hint(evidence_span_hint) or _has_offpage_ref(evidence_span_hint, page_refs_local):
        evidence_span_hint = nearest_span[:240] if nearest_span else _best_evidence_hint(page_text_clean)
    if nearest_refs:
        hint_refs = _extract_figure_refs_from_text(evidence_span_hint)
        if not hint_refs or hint_refs[0].lower() != nearest_refs[0].lower():
            evidence_span_hint = nearest_span[:240]

    op_context = _is_operation_visual_context(page_text_clean, region_context)
    if op_context and nearest_span:
        evidence_span_hint = nearest_span[:240]
    snapshot_score = _state_snapshot_signal_score(
        table_md=table_md,
        key_elements=key_elements,
        nearest_span=nearest_span,
        caption=caption,
    )
    if vtype in {"other", "table"} and op_context and (_looks_like_state_snapshot_table(table_md) or snapshot_score >= 2 or not table_md):
        vtype = "diagram"
        table_md = ""
        table_cell_visuals = []
        if not diagram_steps:
            derived_steps = _extract_operation_steps(nearest_span) or _extract_operation_steps(page_text_clean)
            diagram_steps = derived_steps[:4]
        if not entities:
            entities = _dedupe_keep_order(
                [str(x) for x in re.findall(r"\b[A-Z]\b|front|rear|enqueue|dequeue|insert|remove", nearest_span, flags=re.IGNORECASE)]
            )[:10]
    if op_context and vtype in {"diagram", "chart"} and not diagram_steps:
        diagram_steps = (_extract_operation_steps(nearest_span) or _extract_operation_steps(page_text_clean))[:4]
    if vtype in {"diagram", "chart"} and not diagram_steps and nearest_span:
        if FIGURE_REF_RE.search(nearest_span) or OPERATION_HINT_RE.search(nearest_span):
            diagram_steps = [nearest_span[:220]]

    if vtype == "table":
        # Keep table-specific signal clean; step/entity in table row often becomes hallucinated refs.
        diagram_steps = []
        entities = []
    else:
        # Diagram/chart should not inherit markdown table from nearby text.
        table_md = ""

    if vtype == "table" and table_md and not caption:
        caption = "ตารางสรุปข้อมูลจากเอกสาร"
    if vtype == "diagram" and not caption and (diagram_steps or entities):
        caption = "แผนภาพขั้นตอนการทำงานจากเอกสาร"
    if _caption_out_of_page_context(caption, page_text_clean):
        if vtype == "table":
            caption = "ตารางสรุปข้อมูลจากเอกสาร"
        elif figure_refs_seen:
            caption = f"แผนภาพจาก {figure_refs_seen[0]}"
        elif op_context:
            caption = "แผนภาพขั้นตอนการดำเนินการจากเอกสาร"
        else:
            caption = "แผนภาพโครงสร้างข้อมูล"

    conf_raw = obj.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(conf_raw)))
    except Exception:
        confidence = 0.0
    if confidence <= 0.0:
        if vtype == "table" and table_md:
            confidence = 0.72
        elif vtype == "diagram" and (diagram_steps or entities):
            confidence = 0.68

    uncertain_from_model = bool(obj.get("uncertainty_flag", False))
    uncertainty_flag = bool(
        uncertain_from_model
        or confidence < float(balanced_uncertainty_threshold)
        or (vtype == "diagram" and not diagram_steps and not entities)
        or (vtype == "table" and not table_md and not table_cell_visuals)
    )

    return {
        "image_path": str(image_path).replace("\\", "/"),
        "visual_type": vtype,
        "caption_th": caption,
        "key_elements": key_elements,
        "table_markdown": table_md,
        "table_cell_visuals": table_cell_visuals,
        "diagram_steps": diagram_steps,
        "entities": entities,
        "figure_refs_seen": figure_refs_seen,
        "evidence_span_hint": evidence_span_hint,
        "confidence": round(float(confidence), 3),
        "uncertainty_flag": bool(uncertainty_flag),
        "prompt_version": prompt_version,
        "model": model,
    }


def build_prompt(*, retry_reason: str = "") -> str:
    strict_line = ""
    if retry_reason:
        strict_line = (
            f"Previous output failed validation ({retry_reason}). "
            "Return concrete visible evidence only and strictly follow schema.\n"
        )
    return (
        "คุณเป็นโมดูลสกัดข้อมูลจากภาพเอกสารภาษาไทย\n"
        + strict_line
        + "ให้ตอบ JSON object เพียงอย่างเดียว ห้ามมีข้อความอื่น\n"
        "หลักการ:\n"
        "- ห้ามแต่งข้อมูลนอกภาพ\n"
        "- ตอบภาษาไทยเป็นหลัก (คำเทคนิคอังกฤษได้) ห้ามภาษาจีน/ญี่ปุ่น/เกาหลี\n"
        "- ถ้าไม่มั่นใจให้ตั้ง uncertainty_flag=true\n"
        "- ถ้าเป็นตารางที่มีรูปในเซลล์ ให้ใส่ table_cell_visuals\n"
        "- ถ้าเป็นภาพขั้นตอนการทำงาน ให้ใส่ diagram_steps ตามลำดับที่เห็น\n"
        "- figure_refs_seen ต้องอ้างอิงเฉพาะที่เห็นในหน้าเดียวกันเท่านั้น\n"
        "schema:\n"
        "{"
        '"visual_type":"diagram|table|chart|other",'
        '"caption_th":"...",'
        '"key_elements":["..."],'
        '"table_markdown":"...",'
        '"table_cell_visuals":[{"row":1,"col":1,"cell_text":"...","cell_visual_caption":"..."}],'
        '"diagram_steps":["..."],'
        '"entities":["..."],'
        '"figure_refs_seen":["ภาพที่ x.x"],'
        '"evidence_span_hint":"...",'
        '"confidence":0.0,'
        '"uncertainty_flag":false'
        "}"
    )


def caption_one_image(
    img_path: Path,
    *,
    model: str,
    visual_qa_model: str,
    hf_token: str,
    timeout_sec: int,
    prompt_version: str,
    page_text: str,
    region_context: dict[str, Any] | None,
    enable_visual_qa: bool,
    visual_qa_min_confidence: float,
    visual_qa_min_signal_count: int,
    max_retries: int,
    balanced_uncertainty_threshold: float,
) -> tuple[dict[str, Any], bool, str]:
    raw = img_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    client = InferenceClient(api_key=hf_token, timeout=max(10, int(timeout_sec)))
    last_reason = "invalid_schema"
    last_obj: dict[str, Any] = {}
    for attempt in range(0, max(1, int(max_retries)) + 1):
        prompt = build_prompt(retry_reason=last_reason if attempt > 0 else "")
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        res = client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=700,
            temperature=0.0,
            top_p=0.9,
        )
        raw_text = str(res.choices[0].message.content or "").strip()
        obj = parse_json_like(raw_text)
        if not obj:
            last_reason = "not_json"
            last_obj = {
                "visual_type": "other",
                "caption_th": raw_text[:480],
            }
            continue
        normalized = normalize_caption_obj(
            obj,
            image_path=str(img_path),
            prompt_version=prompt_version,
            model=model,
            page_text=page_text,
            region_context=region_context,
            balanced_uncertainty_threshold=float(balanced_uncertainty_threshold),
        )
        ok, reason = validate_caption_obj(normalized)
        if ok:
            if enable_visual_qa:
                qa_obj, qa_status = visual_qa_one_image(
                    img_path,
                    model=visual_qa_model,
                    hf_token=hf_token,
                    timeout_sec=timeout_sec,
                    page_text=page_text,
                    region_context=region_context,
                )
                qa_pass, qa_reason = validate_visual_qa_gate(
                    normalized,
                    qa_obj,
                    page_text=page_text,
                    region_context=region_context,
                    min_confidence=float(visual_qa_min_confidence),
                    min_signal_count=int(visual_qa_min_signal_count),
                )
                normalized["visual_qa"] = qa_obj
                normalized["qa_gate_pass"] = bool(qa_pass)
                normalized["qa_gate_reason"] = qa_reason if qa_status == "ok" else qa_status
                if not qa_pass:
                    last_reason = f"visual_qa_{qa_reason}"
                    last_obj = normalized
                    continue
            else:
                normalized["visual_qa"] = {}
                normalized["qa_gate_pass"] = True
                normalized["qa_gate_reason"] = "disabled"
            return normalized, True, "ok"
        last_reason = reason
        last_obj = normalized

    fallback = normalize_caption_obj(
        last_obj if last_obj else {"visual_type": "other", "caption_th": ""},
        image_path=str(img_path),
        prompt_version=prompt_version,
        model=model,
        page_text=page_text,
        region_context=region_context,
        balanced_uncertainty_threshold=float(balanced_uncertainty_threshold),
    )
    fallback["uncertainty_flag"] = True
    if enable_visual_qa:
        qa_obj, qa_status = visual_qa_one_image(
            img_path,
            model=visual_qa_model,
            hf_token=hf_token,
            timeout_sec=timeout_sec,
            page_text=page_text,
            region_context=region_context,
        )
        qa_pass, qa_reason = validate_visual_qa_gate(
            fallback,
            qa_obj,
            page_text=page_text,
            region_context=region_context,
            min_confidence=float(visual_qa_min_confidence),
            min_signal_count=int(visual_qa_min_signal_count),
        )
        fallback["visual_qa"] = qa_obj
        fallback["qa_gate_pass"] = bool(qa_pass)
        # preserve root cause and QA status for traceability
        prefix = f"{last_reason}+" if last_reason else ""
        fallback["qa_gate_reason"] = f"{prefix}{qa_status if qa_status != 'ok' else qa_reason}"
        if qa_pass:
            conf_val = 0.0
            try:
                conf_val = float(fallback.get("confidence", 0.0) or 0.0)
            except Exception:
                conf_val = 0.0
            # If QA passes and confidence is usable, do not force uncertainty.
            if conf_val >= float(balanced_uncertainty_threshold):
                has_struct_signal = bool(
                    fallback.get("diagram_steps")
                    or fallback.get("entities")
                    or fallback.get("table_markdown")
                    or fallback.get("table_cell_visuals")
                )
                if has_struct_signal:
                    fallback["uncertainty_flag"] = False
            fallback.pop("validation_error", None)
            return fallback, True, "ok_fallback_caption_not_json"
        fallback["validation_error"] = f"{last_reason}|visual_qa_{qa_reason}"
        return fallback, False, f"{last_reason}|visual_qa_{qa_reason}"

    if "visual_qa" not in fallback:
        fallback["visual_qa"] = {}
    if "qa_gate_pass" not in fallback:
        fallback["qa_gate_pass"] = False
    if "qa_gate_reason" not in fallback:
        fallback["qa_gate_reason"] = last_reason
    fallback["validation_error"] = last_reason
    return fallback, False, last_reason


def build_caption_block(model: str, page_id: str, captions: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("### Visual Captions (Auto)")
    lines.append(f"> [CaptionModel] {model}")
    lines.append(f"> [PageEvidence] {page_id}")
    for i, cap in enumerate(captions, start=1):
        img = str(cap.get("image_path", "")).strip()
        vtype = str(cap.get("visual_type", "other")).strip().lower() or "other"
        summary = str(cap.get("caption_th", "")).strip()
        conf = cap.get("confidence", 0.0)
        try:
            conf_val = round(float(conf), 3)
        except Exception:
            conf_val = 0.0
        lines.append(f"> [Image {i}] {img}")
        lines.append(f"> [VisualType] {vtype}")
        lines.append(f"> [Confidence] {conf_val}")
        if summary:
            lines.append(f"> [SummaryTH] {summary}")
        lines.append("```yaml")
        lines.append(f"image_path: {json.dumps(img, ensure_ascii=False)}")
        lines.append(f"visual_type: {vtype}")
        lines.append(f"caption_th: {json.dumps(summary, ensure_ascii=False)}")
        lines.append(f"key_elements: {json.dumps(cap.get('key_elements', []), ensure_ascii=False)}")
        lines.append(f"table_markdown: {json.dumps(str(cap.get('table_markdown', '')).strip(), ensure_ascii=False)}")
        lines.append(f"table_cell_visuals: {json.dumps(cap.get('table_cell_visuals', []), ensure_ascii=False)}")
        lines.append(f"diagram_steps: {json.dumps(cap.get('diagram_steps', []), ensure_ascii=False)}")
        lines.append(f"entities: {json.dumps(cap.get('entities', []), ensure_ascii=False)}")
        lines.append(f"figure_refs_seen: {json.dumps(cap.get('figure_refs_seen', []), ensure_ascii=False)}")
        lines.append(f"evidence_span_hint: {json.dumps(str(cap.get('evidence_span_hint', '')).strip(), ensure_ascii=False)}")
        lines.append(f"confidence: {conf_val}")
        lines.append(f"uncertainty_flag: {str(bool(cap.get('uncertainty_flag', False))).lower()}")
        lines.append(f"qa_gate_pass: {str(bool(cap.get('qa_gate_pass', False))).lower()}")
        lines.append(f"qa_gate_reason: {json.dumps(str(cap.get('qa_gate_reason', '')).strip(), ensure_ascii=False)}")
        lines.append(f"visual_qa: {json.dumps(cap.get('visual_qa', {}), ensure_ascii=False)}")
        lines.append("```")
    return "\n".join(lines).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich markdown with cached VLM visual captions.")
    ap.add_argument("--input-markdown", default="final_extracted_text_only.md")
    ap.add_argument("--region-manifest", default="logs/figure_regions_manifest_latest.json")
    ap.add_argument("--page-image-manifest", default="logs/figure_manifest_latest.json")
    ap.add_argument("--output-markdown", default="final_extracted_text_only_structured_full.md")
    ap.add_argument("--cache-path", default="logs/visual_caption_cache_latest.json")
    ap.add_argument("--report", default="logs/visual_caption_enrich_report_latest.json")
    ap.add_argument("--model", default=os.getenv("VISUAL_CAPTION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--visual-qa-model", default=os.getenv("VISUAL_QA_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--max-pages", type=int, default=120)
    ap.add_argument("--max-images-per-page", type=int, default=2)
    ap.add_argument("--min-region-score", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout-sec", type=int, default=90)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--balanced-uncertainty-threshold", type=float, default=0.55)
    ap.add_argument("--enable-visual-qa", action="store_true")
    ap.add_argument("--visual-qa-min-confidence", type=float, default=0.55)
    ap.add_argument("--visual-qa-min-signal-count", type=int, default=1)
    ap.add_argument("--visual-qa-expand-factor", type=float, default=1.35)
    ap.add_argument("--page-start", type=int, default=1)
    ap.add_argument("--page-end", type=int, default=9999)
    ap.add_argument(
        "--caption-require-hints",
        action="store_true",
        help="Caption only when text hints are detected (legacy behavior). Default captions all pages with usable images.",
    )
    args = ap.parse_args()

    dotenv_path = ROOT / ".env"
    load_dotenv(dotenv_path)
    hf_token = (os.getenv("HUGGINGFACE_READ_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
    if not hf_token:
        raise SystemExit("missing HF token in .env")

    in_md = Path(args.input_markdown)
    if not in_md.exists():
        raise FileNotFoundError(f"input markdown not found: {in_md}")

    region_manifest = Path(args.region_manifest)
    page_image_manifest = Path(args.page_image_manifest)
    cache_path = Path(args.cache_path)
    out_md = Path(args.output_markdown)
    report_path = Path(args.report)

    page_images = load_region_candidates(
        region_manifest,
        max_images_per_page=max(1, int(args.max_images_per_page)),
        min_region_score=float(args.min_region_score),
    )
    page_image_fallback = load_page_image_fallback(page_image_manifest)
    raw_md = in_md.read_text(encoding="utf-8")
    pages = parse_pages(raw_md)
    cache = load_cache(cache_path)
    prompt_version = "v4_structured_visual_caption_balanced_evidence_strict_localrefs_thai"

    tasks: list[tuple[int, str, Path, str, str, dict[str, Any]]] = []
    selected_pages = 0
    for idx, p in enumerate(pages):
        page_id = f"{p['source']}:{int(p['page'])}"
        page_no = int(p["page"])
        if page_no < int(args.page_start) or page_no > int(args.page_end):
            continue
        page_text_clean = strip_auto_visual_blocks(str(p.get("content", "") or ""))
        imgs = list(page_images.get(page_id, []))
        if not imgs:
            imgs = list(page_image_fallback.get(page_id, []))
        if not should_caption(page_text_clean, imgs, require_hints=bool(args.caption_require_hints)):
            continue
        selected_pages += 1
        if selected_pages > max(1, int(args.max_pages)):
            break
        for img_meta in imgs[: max(1, int(args.max_images_per_page))]:
            img = str(img_meta.get("path", "")).strip()
            abs_img = _abs_image(img)
            if not abs_img.exists():
                continue
            tasks.append(
                (
                    idx,
                    page_id,
                    abs_img,
                    str(img),
                    page_text_clean,
                    dict(img_meta),
                )
            )

    started = time.time()
    cache_hits = 0
    cache_miss = 0
    failures = 0
    generic_rejected = 0
    nonfatal_validation_failures = 0
    qa_gate_rejected = 0
    failure_samples: list[dict[str, str]] = []
    page_caps: dict[int, list[dict[str, Any]]] = {}

    enable_visual_qa = bool(args.enable_visual_qa)

    def _worker(item: tuple[int, str, Path, str, str, dict[str, Any]]) -> tuple[int, str, str, dict[str, Any], bool, str, bool]:
        page_idx, page_id, abs_img, rel_img, page_text, region_ctx = item

        def _try_expanded_qa(cap_obj: dict[str, Any]) -> tuple[dict[str, Any], bool]:
            if not enable_visual_qa:
                return cap_obj, False
            if bool(cap_obj.get("qa_gate_pass", False)):
                return cap_obj, False
            gate_reason = str(cap_obj.get("qa_gate_reason", "") or cap_obj.get("validation_error", "")).strip().lower()
            if not any(x in gate_reason for x in ["low_confidence", "low_signal_count", "missing_structure_signal", "table_no_cells"]):
                return cap_obj, False
            exp_img = _build_expanded_crop_from_region(
                region_context=region_ctx,
                expand_factor=float(args.visual_qa_expand_factor),
            )
            if exp_img is None or not exp_img.exists():
                return cap_obj, False
            qa_obj, qa_status = visual_qa_one_image(
                exp_img,
                model=str(args.visual_qa_model),
                hf_token=hf_token,
                timeout_sec=max(10, int(args.timeout_sec)),
                page_text=page_text,
                region_context=region_ctx,
            )
            qa_pass, qa_reason = validate_visual_qa_gate(
                cap_obj,
                qa_obj,
                page_text=page_text,
                region_context=region_ctx,
                min_confidence=float(args.visual_qa_min_confidence),
                min_signal_count=int(args.visual_qa_min_signal_count),
            )
            cap_obj["visual_qa"] = qa_obj
            cap_obj["qa_gate_pass"] = bool(qa_pass)
            cap_obj["qa_gate_reason"] = (
                f"expanded_crop_{qa_reason}" if qa_status == "ok" else f"expanded_crop_{qa_status}"
            )
            if qa_pass and "validation_error" in cap_obj:
                cap_obj.pop("validation_error", None)
            if not qa_pass:
                cap_obj["validation_error"] = f"visual_qa_{qa_reason}"
                cap_obj["uncertainty_flag"] = True
            return cap_obj, bool(qa_pass)

        key = image_cache_key(abs_img, str(args.model), prompt_version)
        cached = cache.get(key)
        if isinstance(cached, dict):
            cap = normalize_caption_obj(
                dict(cached),
                image_path=str(abs_img),
                prompt_version=prompt_version,
                model=str(args.model),
                page_text=page_text,
                region_context=region_ctx,
                balanced_uncertainty_threshold=float(args.balanced_uncertainty_threshold),
            )
            if isinstance(cached.get("visual_qa"), dict):
                cap["visual_qa"] = dict(cached.get("visual_qa", {}))
            cap["qa_gate_pass"] = bool(cached.get("qa_gate_pass", False))
            cap["qa_gate_reason"] = str(cached.get("qa_gate_reason", "")).strip()

            if enable_visual_qa and (not cap.get("qa_gate_pass", False)):
                qa_obj, qa_status = visual_qa_one_image(
                    abs_img,
                    model=str(args.visual_qa_model),
                    hf_token=hf_token,
                    timeout_sec=max(10, int(args.timeout_sec)),
                    page_text=page_text,
                    region_context=region_ctx,
                )
                qa_pass, qa_reason = validate_visual_qa_gate(
                    cap,
                    qa_obj,
                    page_text=page_text,
                    region_context=region_ctx,
                    min_confidence=float(args.visual_qa_min_confidence),
                    min_signal_count=int(args.visual_qa_min_signal_count),
                )
                cap["visual_qa"] = qa_obj
                cap["qa_gate_pass"] = bool(qa_pass)
                cap["qa_gate_reason"] = qa_reason if qa_status == "ok" else qa_status
                if not qa_pass:
                    cap["uncertainty_flag"] = True
                    cap["validation_error"] = f"visual_qa_{qa_reason}"

            ok_cached, reason_cached = validate_caption_obj(cap)
            if not ok_cached:
                cap["uncertainty_flag"] = True
                cap["validation_error"] = reason_cached
            cap, expanded_ok = _try_expanded_qa(cap)
            err_cached = ""
            if enable_visual_qa and not bool(cap.get("qa_gate_pass", False)) and not expanded_ok:
                err_cached = str(cap.get("validation_error", "") or cap.get("qa_gate_reason", "") or "visual_qa_reject")
            cap["image_path"] = rel_img.replace("\\", "/")
            cache[key] = dict(cap)
            return page_idx, page_id, key, cap, True, err_cached, False
        try:
            cap, valid_ok, reason = caption_one_image(
                abs_img,
                model=str(args.model),
                visual_qa_model=str(args.visual_qa_model),
                hf_token=hf_token,
                timeout_sec=max(10, int(args.timeout_sec)),
                prompt_version=prompt_version,
                page_text=page_text,
                region_context=region_ctx,
                enable_visual_qa=enable_visual_qa,
                visual_qa_min_confidence=float(args.visual_qa_min_confidence),
                visual_qa_min_signal_count=int(args.visual_qa_min_signal_count),
                max_retries=max(0, int(args.max_retries)),
                balanced_uncertainty_threshold=float(args.balanced_uncertainty_threshold),
            )
            cap["image_path"] = rel_img.replace("\\", "/")
            cap, expanded_ok = _try_expanded_qa(cap)
            if expanded_ok and str(reason).startswith("visual_qa_"):
                reason = "ok"
            return page_idx, page_id, key, cap, False, reason, (not valid_ok and reason == "generic_caption")
        except Exception as exc:
            return page_idx, page_id, key, {"image_path": rel_img.replace("\\", "/")}, False, str(exc), False

    max_workers = max(1, int(args.workers))
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for fut in cf.as_completed(futures):
            page_idx, page_id, key, cap, from_cache, err, generic_flag = fut.result()
            if from_cache:
                cache_hits += 1
                if err:
                    nonfatal_validation_failures += 1
                    if "visual_qa_" in err or "low_signal" in err or "low_confidence" in err:
                        qa_gate_rejected += 1
            else:
                allowed_nonfatal = {
                    "ok",
                    "ok_fallback_caption_not_json",
                    "generic_caption",
                    "table_missing_structure",
                    "diagram_missing_steps_entities",
                    "not_json",
                    "invalid_schema",
                    "visual_qa_low_confidence",
                    "visual_qa_low_signal_count",
                    "visual_qa_missing_structure_signal",
                    "visual_qa_table_no_cells",
                    "visual_qa_qa_error",
                    "visual_qa_not_json",
                }
                if str(err).startswith("visual_qa_"):
                    qa_gate_rejected += 1
                if err and err not in {
                    *allowed_nonfatal,
                } and not str(err).startswith("visual_qa_") and not str(err).startswith("ok_"):
                    failures += 1
                    if len(failure_samples) < 20:
                        failure_samples.append(
                            {
                                "page_id": page_id,
                                "image_path": str(cap.get("image_path", "") or ""),
                                "error": str(err)[:240],
                            }
                        )
                    cap.update(
                        {
                            "visual_type": "other",
                            "caption_th": "",
                            "key_elements": [],
                            "table_markdown": "",
                            "table_cell_visuals": [],
                            "diagram_steps": [],
                            "entities": [],
                            "figure_refs_seen": [],
                            "evidence_span_hint": "",
                            "confidence": 0.0,
                            "uncertainty_flag": True,
                            "error": err[:240],
                            "model": str(args.model),
                            "prompt_version": prompt_version,
                        }
                    )
                else:
                    cache[key] = dict(cap)
                    cache_miss += 1
                    if generic_flag:
                        generic_rejected += 1
                    if err and err not in {"ok"}:
                        nonfatal_validation_failures += 1
            page_caps.setdefault(page_idx, []).append(cap)

    updated_pages: list[dict[str, Any]] = []
    pages_enriched = 0
    for idx, p in enumerate(pages):
        original_content = str(p.get("content", "") or "")
        caps = page_caps.get(idx)
        if caps is None:
            content = original_content.strip()
        else:
            content = VISUAL_BLOCK_RE.sub("", original_content).strip()
            page_id = f"{p['source']}:{int(p['page'])}"
            block = build_caption_block(str(args.model), page_id, caps)
            if block:
                content = (content + "\n\n" + block).strip()
                pages_enriched += 1
        updated_pages.append({"source": p["source"], "page": p["page"], "content": content})

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(rebuild_markdown(updated_pages), encoding="utf-8")
    save_cache(cache_path, cache)

    elapsed = round(time.time() - started, 3)
    report = {
        "input_markdown": str(in_md.as_posix()),
        "output_markdown": str(out_md.as_posix()),
        "region_manifest": str(region_manifest.as_posix()),
        "model": str(args.model),
        "workers": max_workers,
        "selected_pages": selected_pages,
        "pages_enriched": pages_enriched,
        "tasks_total": len(tasks),
        "cache_hits": cache_hits,
        "cache_miss": cache_miss,
        "failures": failures,
        "generic_caption_rejected": generic_rejected,
        "nonfatal_validation_failures": nonfatal_validation_failures,
        "qa_gate_rejected": qa_gate_rejected,
        "failure_samples": failure_samples,
        "elapsed_sec": elapsed,
        "caption_uncertainty_policy": "balanced_evidence",
        "prompt_version": prompt_version,
        "caption_require_hints": bool(args.caption_require_hints),
        "visual_qa_enabled": bool(enable_visual_qa),
        "visual_qa_model": str(args.visual_qa_model),
        "visual_qa_min_confidence": float(args.visual_qa_min_confidence),
        "visual_qa_min_signal_count": int(args.visual_qa_min_signal_count),
        "visual_qa_expand_factor": float(args.visual_qa_expand_factor),
        "page_start": int(args.page_start),
        "page_end": int(args.page_end),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved_markdown={out_md}")
    print(f"saved_cache={cache_path}")
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    main()
