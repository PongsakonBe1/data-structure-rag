"""
Regression report for topic routing (old vs new) with confusion matrices.

Outputs:
- logs/topic_routing_regression_latest.csv
- logs/topic_confusion_old_latest.csv
- logs/topic_confusion_new_latest.csv
- logs/topic_routing_ambiguous_latest.csv
- logs/topic_routing_direct_answers_latest.csv
- logs/topic_routing_ambiguous_answers_latest.csv
- logs/topic_routing_regression_summary_latest.json
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _read_text_auto(path: Path) -> str:
    encodings = ("utf-8-sig", "utf-8", "cp874", "tis-620", "latin-1")
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"cannot decode: {path}")


def _load_csv_auto(path: Path) -> list[dict]:
    text = _read_text_auto(path)
    rows = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        rows.append({str(k or "").strip(): str(v or "").strip() for k, v in row.items()})
    return rows


def _load_topic_titles(topic_hierarchy_path: Path) -> dict[str, str]:
    text = _read_text_auto(topic_hierarchy_path)
    payload = json.loads(text)
    out: dict[str, str] = {}
    for t in (payload.get("topics", []) or []):
        if not isinstance(t, dict):
            continue
        tid = str(t.get("topic_id", "")).strip()
        title = str(t.get("title", "")).strip()
        if tid:
            out[tid] = title
    return out


def _load_answer_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return _load_csv_auto(path)


def _build_answer_map(answer_rows: list[dict]) -> tuple[dict[int, str], dict[str, str]]:
    by_order: dict[int, str] = {}
    by_question: dict[str, str] = {}
    for r in answer_rows:
        answer = str(
            r.get("assistant_answer")
            or r.get("answer")
            or r.get("response")
            or ""
        ).strip()
        if not answer:
            continue

        order_raw = str(r.get("order", "")).strip()
        if order_raw.isdigit():
            by_order[int(order_raw)] = answer

        question = str(r.get("question", "")).strip()
        if question:
            by_question[question] = answer
    return by_order, by_question


def _merge_existing_expert_labels(new_rows: list[dict], existing_rows: list[dict]) -> list[dict]:
    if not existing_rows:
        return new_rows
    by_id: dict[str, dict] = {}
    for r in existing_rows:
        rid = str(r.get("review_id", "")).strip()
        if rid:
            by_id[rid] = r

    merged = []
    for r in new_rows:
        rid = str(r.get("review_id", "")).strip()
        old = by_id.get(rid, {})
        row = dict(r)
        if old:
            ans_mark = str(old.get("answer_correct(Y/N)", "")).strip().upper()
            if ans_mark in {"Y", "N"}:
                row["answer_correct(Y/N)"] = ans_mark
        merged.append(row)
    return merged


def _normalize_topic_match_text(text: str) -> str:
    txt = str(text or "").strip().lower()
    txt = re.sub(r"[^\w\s\u0E00-\u0E7F]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _select_most_specific_section_id(text: str | None) -> str:
    raw = str(text or "")
    matches = re.findall(r"\b\d+(?:\.\d+)+\b", raw)
    if not matches:
        m = re.search(r"(?:หัวข้อที่|บทที่|chapter)\s*(\d+)\b", raw, re.IGNORECASE)
        if m:
            return str(m.group(1)).strip()
        return ""
    matches.sort(key=lambda x: (x.count("."), len(x)), reverse=True)
    return matches[0]


def _infer_section_id_from_question(question: str | None) -> str:
    q = _normalize_topic_match_text(question or "")
    if not q:
        return ""

    has = lambda *terms: any(t in q for t in terms)
    has_all = lambda *terms: all(t in q for t in terms)

    explicit_sid = _select_most_specific_section_id(q)
    if explicit_sid:
        return explicit_sid

    # list_hitachi-driven deterministic section targeting for high-noise intents.
    # Keep high-specificity entity definitions first.
    if has("บิต", "bit"):
        return "1.2.1"
    if has("ไบต์", "byte"):
        return "1.2.2"
    if has("ฟิลด์", "field"):
        return "1.2.3"
    if has("เรคอร์ด", "record"):
        return "1.2.4"
    if has("ฐานข้อมูล", "database"):
        return "1.2.6"
    if has("ไฟล์", "file"):
        return "1.2.5"

    if has("ความหมายของโครงสร้างข้อมูล", "โครงสร้างข้อมูลคือ", "data structure"):
        return "1.1.1"
    if has("ความหมายของอัลกอริทึม", "อัลกอริทึมคือ", "algorithm"):
        return "1.1.2"
    if has("ความสำคัญของการเลือกโครงสร้างข้อมูล", "เลือกโครงสร้างข้อมูล", "เหมาะสมกับอัลกอริทึม"):
        return "1.1.3"
    if has_all("ประเภท", "โครงสร้างข้อมูล"):
        return "1.3.1"
    if has_all("ประเภท", "อัลกอริทึม"):
        return "1.3.2"
    if has("เครื่องมือพัฒนาอัลกอริทึม", "เครื่องมือพัฒนา", "ผังงาน", "flowchart"):
        return "1.4.1"
    if has("รหัสเทียม", "pseudocode"):
        return "1.4.2"
    if has("โครงสร้างแบบเรียงลำดับ", "sequence"):
        return "1.5.1"
    if has("โครงสร้างแบบเลือกการทำงาน", "selection"):
        return "1.5.2"
    if has("โครงสร้างแบบทำซ้ำ", "repetition", "loop"):
        return "1.5.3"

    if has("อาร์เรย์", "array") and (not has("คิว", "queue", "แทนคิว")):
        if has("โครงสร้าง", "structure"):
            return "2.1.2"
        if has("2 มิติ", "สองมิติ", "2d"):
            return "2.2.2"
        if has("1 มิติ", "หนึ่งมิติ", "1d"):
            return "2.2.1"
        if has("ความหมาย", "ลักษณะ", "คือ"):
            return "2.1.1"

    if has("ลิงค์ลิสต์", "ลิงก์ลิสต์", "linked list"):
        if has("แบบทิศทางเดียว", "single linked"):
            if has("การทำงาน", "ดำเนินการ", "operation", "แทรกโหนด", "เพิ่มโหนด", "ลบโหนด"):
                return "2.4.2"
            if has("โครงสร้าง", "structure"):
                return "2.4.1"
            return "2.4.1"
        if has("โครงสร้าง", "structure"):
            return "2.3.2"
        if has("ความหมาย", "คือ"):
            return "2.3.1"
        return "2.3"

    # Disambiguate stack vs queue for "นำข้อมูลเข้า/ออก"
    if has("push") and (not has("enqueue", "queue", "คิว")):
        return "4.2.1"
    if has("pop") and (not has("queue", "คิว")):
        return "4.2.2"
    if has("ตำแหน่งบนสุด", "top"):
        return "4.2.3"

    if has("สแตก", "stack"):
        if has("นำข้อมูลเข้า", "push"):
            return "4.2.1"
        if has("ดึงข้อมูลออก", "นำข้อมูลออก", "pop"):
            return "4.2.2"
        if has("ตำแหน่งบนสุด", "top"):
            return "4.2.3"
        if has("แทนสแตก", "แทนสแตกด้วยอาร์เรย์", "array"):
            return "4.3.1"
        if has("แปลงรูปนิพจน์", "นิพจน์ทางคณิตศาสตร์", "infix", "postfix", "prefix"):
            return "4.3.2"
        if has("โครงสร้าง", "structure"):
            return "4.1"
        return "4.2"

    if has("คิว", "queue"):
        if has("นำข้อมูลเข้า", "enqueue"):
            return "3.2.1"
        if has("นำข้อมูลออก", "dequeue", "remove"):
            return "3.2.2"
        if has("วงกลม", "circular"):
            return "3.3.3"
        if has("อาร์เรย์", "อาร์เร", "array", "แทนคิว"):
            if has("การดำเนินการ", "ดำเนินการ", "insert", "remove"):
                return "3.3.2"
            if has("โครงสร้าง", "structure"):
                return "3.3.1"
            return "3.3.2"
        if has("โครงสร้าง", "structure"):
            return "3.1"
        return "3.2"

    if has("ทรีทั่วไป", "general tree"):
        return "5.2.1"
    if has("expression tree", "ทรีกับการดำเนินการด้านนิพจน์"):
        return "5.4"
    if has("แปลงทรี", "convert tree") and has("ไบนารีทรี", "binary tree"):
        return "5.6"
    if has("ไบนารีทรีแบบสมบูรณ์", "complete binary tree"):
        return "5.2.3"
    if has("ไบนารีทรี", "binary tree"):
        if has("พรีออร์เดอร์", "preorder", "nlr"):
            return "5.3.1"
        if has("อินออร์เดอร์", "inorder", "lnr"):
            return "5.3.2"
        if has("โพสต์ออร์เดอร์", "postorder", "lrn"):
            return "5.3.3"
        if has("ซีเควนเชียล", "sequential", "แทนโครงสร้าง"):
            return "5.5"
        return "5.2.2"
    if has("โครงสร้างทรี", "โครงสร้างของทรี", "tree structure"):
        return "5.1"
    if has("การเข้าถึงข้อมูลในไบนารีทรี", "binary tree traversal"):
        return "5.3"
    if has("พรีออร์เดอร์", "preorder", "nlr"):
        return "5.3.1"
    if has("อินออร์เดอร์", "inorder", "lnr"):
        return "5.3.2"
    if has("โพสต์ออร์เดอร์", "postorder", "lrn"):
        return "5.3.3"

    # In-domain ambiguous intents (not exact TOC phrasing)
    if has("fifo", "เข้าก่อนออกก่อน"):
        return "3.1"
    if has("lifo", "เข้าทีหลังออกก่อน"):
        return "4.1"
    if has("traverse", "traversal", "ท่องทรี", "การท่องทรี") and has("ทรี", "tree"):
        return "5.3"
    if has("infix", "postfix", "prefix", "อินฟิกซ์", "โพสต์ฟิกซ์", "พรีฟิกซ์"):
        return "4.3.2"
    if has("โหนดลูก", "ลูกซ้าย", "ลูกขวา"):
        return "5.2.2"
    if has("ลำดับชั้น") and has("โครงสร้าง", "structure"):
        return "5.1"
    if has("ค้นหา", "search") and has("โครงสร้างข้อมูล", "data structure"):
        return "1.1.1"
    if has("อัลกอริทึม"):
        return "1.1.2"

    return ""


def _topic_label(topic_id: str, topic_titles: dict[str, str]) -> str:
    tid = str(topic_id or "").strip()
    if not tid:
        return ""
    title = str(topic_titles.get(tid, "")).strip()
    return f"{tid} {title}".strip()


def _to_bool_text(val: bool | None) -> str:
    if val is None:
        return ""
    return "pass" if val else "fail"


_TOPIC_ID_ALIASES = {
    # Backward compatibility for legacy dataset only.
    "1.3.3": "1.4.1",
}


def _canonical_topic_id(topic_id: str | None) -> str:
    tid = str(topic_id or "").strip()
    if not tid:
        return ""
    # IMPORTANT: apply alias once only (no chaining), otherwise
    # 1.3.3 -> 1.4.1 could be incorrectly remapped again to 1.5.1.
    return _TOPIC_ID_ALIASES.get(tid, tid)


def _sort_topic_ids(ids: list[str]) -> list[str]:
    def _key(x: str):
        parts = []
        for p in str(x).split("."):
            if p.isdigit():
                parts.append((0, int(p)))
            else:
                parts.append((1, p))
        return parts

    return sorted(ids, key=_key)


def _parse_runtime_events(path: Path) -> list[dict]:
    text = _read_text_auto(path)
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _extract_old_topics(
    events: list[dict],
    needed: int,
    from_timestamp: str | None,
) -> list[str]:
    deterministic = [e for e in events if str(e.get("event", "")).strip() == "topic_classified_deterministic"]
    deterministic = sorted(deterministic, key=lambda x: str(x.get("timestamp", "")))

    scoped = deterministic
    ts = str(from_timestamp or "").strip()
    if ts:
        scoped = [e for e in deterministic if str(e.get("timestamp", "")) >= ts]

    if len(scoped) >= needed:
        picked = scoped[:needed]
    else:
        picked = deterministic[-needed:] if len(deterministic) >= needed else deterministic

    out = [str(e.get("predicted_topic", "")).strip() for e in picked]
    if len(out) < needed:
        out.extend([""] * (needed - len(out)))
    return out[:needed]


def _save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


_SECTION_PAGE_MAP_CACHE: dict[str, list[str]] | None = None
_OPERATION_HEAVY_SIDS = {"2.4.2", "3.3.2", "3.3.3", "4.3.1", "4.3.2"}


def _load_section_page_map() -> dict[str, list[str]]:
    global _SECTION_PAGE_MAP_CACHE
    if _SECTION_PAGE_MAP_CACHE is not None:
        return _SECTION_PAGE_MAP_CACHE

    merged: dict[str, list[str]] = {}
    hierarchy_path = ROOT / "indexes" / "hierarchical" / "topic_hierarchy.json"
    override_path = ROOT / "indexes" / "hierarchical" / "section_page_overrides.json"

    if hierarchy_path.exists():
        try:
            payload = json.loads(_read_text_auto(hierarchy_path))
            base = payload.get("topic_to_pages", {}) if isinstance(payload, dict) else {}
            if isinstance(base, dict):
                for sid, pages in base.items():
                    sid_txt = str(sid).strip()
                    if not sid_txt:
                        continue
                    if isinstance(pages, list):
                        merged[sid_txt] = [str(x).strip() for x in pages if str(x).strip()]
        except Exception:
            pass

    if override_path.exists():
        try:
            payload = json.loads(_read_text_auto(override_path))
            over = payload.get("topic_to_pages", payload if isinstance(payload, dict) else {})
            if isinstance(over, dict):
                for sid, pages in over.items():
                    sid_txt = str(sid).strip()
                    if not sid_txt:
                        continue
                    if isinstance(pages, list):
                        vals = [str(x).strip() for x in pages if str(x).strip()]
                        if vals:
                            merged[sid_txt] = vals
        except Exception:
            pass

    _SECTION_PAGE_MAP_CACHE = merged
    return _SECTION_PAGE_MAP_CACHE


def _dynamic_min_section_page_diversity(section_id: str) -> int:
    sid = str(section_id or "").strip()
    if not sid:
        return 1
    page_map = _load_section_page_map()
    pages = [str(x).strip() for x in (page_map.get(sid, []) or []) if str(x).strip()]
    page_nums = {x.split(":")[-1] for x in pages}
    page_count = len(page_nums)
    if page_count <= 1:
        return 1
    if sid in _OPERATION_HEAVY_SIDS:
        return min(3, page_count)
    return min(2, page_count)


def _build_rag_filters(question: str, topic_hint: str, target_section_id: str) -> dict:
    target = str(target_section_id or "").strip()
    hint = str(topic_hint or "").strip()
    return {
        "topic_hint": hint,
        "target_section_id": target,
        "query_text": str(question or "").strip(),
        "require_structure": False,
        "strict_topic": bool(hint),
        "strict_structure": False,
        "strict_section_only": bool(target),
        "allow_offtopic_docs": 0,
        "min_section_page_diversity": _dynamic_min_section_page_diversity(target),
    }


def _safe_preview(text: str, limit: int = 1200) -> str:
    txt = re.sub(r"\s+", " ", str(text or "")).strip()
    return txt[: max(80, int(limit))]


def _build_context_from_docs(docs: list, context_doc_limit: int = 5) -> tuple[str, list[dict]]:
    context_lines = []
    sources_data = []
    for i, d in enumerate(docs[: max(1, int(context_doc_limit))]):
        meta = getattr(d, "metadata", {}) or {}
        src = str(meta.get("source", "reference_document")).strip()
        page = str(meta.get("page", "")).strip()
        chunk_id = str(meta.get("chunk_id", f"chunk-{i+1}")).strip()
        citation = f"{src}" + (f":{page}" if page else "")
        preview = _safe_preview(getattr(d, "page_content", ""))
        context_lines.append(f"Context {i+1} [{citation} | {chunk_id}]: {preview}")
        sources_data.append(
            {
                "source": src,
                "page": page,
                "chunk_id": chunk_id,
                "citation": citation,
            }
        )
    return "\n".join(context_lines).strip(), sources_data


def _has_thai_text(text: str) -> bool:
    return bool(re.search(r"[\u0E00-\u0E7F]", str(text or "")))


def _translate_english_topic_fragment_to_thai(fragment: str) -> str:
    txt = str(fragment or "").strip()
    if not txt:
        return txt
    replacements = [
        ("array 1d index", "ดัชนีของอาร์เรย์ 1 มิติ"),
        ("single linked list", "ลิงก์ลิสต์แบบทิศทางเดียว"),
        ("head node in linked list", "โหนดหัวในลิงก์ลิสต์"),
        ("insert node in single linked list", "การแทรกโหนดในลิงก์ลิสต์แบบทิศทางเดียว"),
        ("insert node in ลิงก์ลิสต์แบบทิศทางเดียว", "การแทรกโหนดในลิงก์ลิสต์แบบทิศทางเดียว"),
        ("insert node in", "การแทรกโหนดใน"),
        ("front rear in queue", "ตำแหน่งหน้าและท้ายของคิว"),
        ("front rear in คิว", "ตำแหน่งหน้าและท้ายของคิว"),
        ("tree structure", "โครงสร้างทรี"),
        ("ทรี structure", "โครงสร้างทรี"),
        ("push in stack", "การเพิ่มข้อมูลในสแตก"),
        ("pop in stack", "การนำข้อมูลออกจากสแตก"),
        ("infix to postfix with stack", "การแปลงอินฟิกซ์เป็นโพสต์ฟิกซ์ด้วยสแตก"),
        ("enqueue in queue", "การนำข้อมูลเข้าคิว"),
        ("dequeue in queue", "การนำข้อมูลออกจากคิว"),
        ("file and database", "ไฟล์และฐานข้อมูล"),
        ("data structure choice", "การเลือกโครงสร้างข้อมูล"),
        ("sequence flow", "โครงสร้างแบบเรียงลำดับ"),
        ("selection logic", "โครงสร้างแบบเลือกการทำงาน"),
        ("loop control", "โครงสร้างแบบทำซ้ำ"),
        ("array 1d", "อาร์เรย์ 1 มิติ"),
        ("array 2d", "อาร์เรย์ 2 มิติ"),
        ("traverse tree", "การท่องทรี"),
        ("tree traversal", "การท่องทรี"),
        ("first in first out", "เข้าก่อนออกก่อน"),
        ("last in first out", "เข้าทีหลังออกก่อน"),
        ("fifo", "เข้าก่อนออกก่อน"),
        ("lifo", "เข้าทีหลังออกก่อน"),
        ("infix", "อินฟิกซ์"),
        ("postfix", "โพสต์ฟิกซ์"),
        ("prefix", "พรีฟิกซ์"),
        ("bit and byte", "บิตและไบต์"),
        ("field and record", "ฟิลด์และเรคอร์ด"),
        ("linear data structure", "โครงสร้างข้อมูลเชิงเส้น"),
        ("non-linear data structure", "โครงสร้างข้อมูลไม่เชิงเส้น"),
        ("data structure", "โครงสร้างข้อมูล"),
        ("complete binary tree", "ไบนารีทรีแบบสมบูรณ์"),
        ("binary tree", "ไบนารีทรี"),
        ("general tree", "ทรีทั่วไป"),
        ("preorder traversal", "การท่องแบบพรีออร์เดอร์"),
        ("inorder traversal", "การท่องแบบอินออร์เดอร์"),
        ("postorder traversal", "การท่องแบบโพสต์ออร์เดอร์"),
        ("sequential representation", "การแทนแบบซีเควนเชียล"),
        ("left child right child", "ลูกซ้าย-ลูกขวา"),
        ("linked list", "ลิงก์ลิสต์"),
        ("circular queue", "คิววงกลม"),
        ("queue", "คิว"),
        ("stack", "สแตก"),
        ("array", "อาร์เรย์"),
        ("algorithm", "อัลกอริทึม"),
        ("tree", "ทรี"),
        ("record", "เรคอร์ด"),
        ("field", "ฟิลด์"),
        ("database", "ฐานข้อมูล"),
        ("file", "ไฟล์"),
        ("byte", "ไบต์"),
        ("bit", "บิต"),
        ("choice", "การเลือก"),
        ("and", "และ"),
    ]
    for en, th in replacements:
        txt = re.sub(rf"\b{re.escape(en)}\b", th, txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s+", " ", txt).strip(" .?")
    return txt


def _normalize_ambiguous_question_for_display(question: str, thai_only: bool = True) -> str:
    q = str(question or "").strip()
    if not thai_only or not q:
        return q

    m = re.match(r"(?is)^\s*can you explain\s+(.+?)\s+briefly\??\s*$", q)
    if m:
        topic = _translate_english_topic_fragment_to_thai(m.group(1))
        return f"ช่วยอธิบาย{topic}แบบสั้นๆ"

    m = re.match(r"(?is)^\s*what should i check before using\s+(.+?)\??\s*$", q)
    if m:
        topic = _translate_english_topic_fragment_to_thai(m.group(1))
        return f"ก่อนใช้{topic}ควรตรวจสอบอะไรบ้าง"

    m = re.match(r"(?is)^\s*give me one practical example of\s+(.+?)\.?\s*$", q)
    if m:
        topic = _translate_english_topic_fragment_to_thai(m.group(1))
        return f"ยกตัวอย่างการใช้งานจริงของ{topic} 1 ตัวอย่าง"

    normalized = _translate_english_topic_fragment_to_thai(q)
    # Never return blank display text; fall back to original question.
    return normalized or q


def _load_hf_client():
    token = (os.getenv("HUGGINGFACE_READ_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
    if not token:
        raise RuntimeError("Missing HUGGINGFACE_READ_TOKEN or HUGGINGFACE_API_KEY in environment.")
    from huggingface_hub import InferenceClient

    return InferenceClient(api_key=token)


def _load_rag_system():
    src_dir = ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from retriever import RAGSystem

    return RAGSystem()


def _retrieve_docs_for_live_answer(rag, question: str, filters: dict, top_k: int = 20, top_n: int = 8) -> list:
    try:
        return rag.retrieve(question, top_k, top_n, filters=filters)
    except TypeError as exc:
        err_text = str(exc)
        if ("unexpected keyword argument" not in err_text) or ("filters" not in err_text):
            raise
        base_docs = rag.retrieve(question, top_k, top_n)
        if hasattr(rag, "_apply_filters"):
            return rag._apply_filters(base_docs, filters)
        return base_docs


def _response_needs_continuation(text: str, finish_reason: str | None) -> bool:
    if not text:
        return False
    tail = str(text).rstrip()
    if not tail:
        return False
    if str(finish_reason or "").strip().lower() == "length":
        return True
    if re.search(r"\([^\)]*$", tail):
        return True
    return tail.endswith((",", ":", ";", "-", "–", "—", "/", "…"))


def _merge_with_overlap(base: str, suffix: str) -> str:
    if not suffix:
        return base
    a = str(base or "")
    b = str(suffix or "")
    max_overlap = min(len(a), len(b), 200)
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if a[-size:] == b[:size]:
            overlap = size
            break
    b = b[overlap:]
    if not b.strip():
        return a
    joiner = "" if (a.endswith("\n") or b.startswith("\n")) else " "
    return f"{a}{joiner}{b}"


def _continue_live_answer(
    hf_client,
    model_id: str,
    question: str,
    context_text: str,
    partial_answer: str,
    max_tokens: int = 220,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are continuing an unfinished Thai answer. "
                "Continue from the exact tail, do not repeat old text, and end with a complete sentence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"คำถาม:\n{question}\n\n"
                f"บริบท:\n{str(context_text or '')[:1800]}\n\n"
                f"คำตอบที่ถูกตัดท้าย:\n{str(partial_answer or '')[-1800:]}\n\n"
                "โปรดเขียนต่ออีกสั้นๆ 1-3 ประโยคให้จบประเด็น"
            ),
        },
    ]
    try:
        res = hf_client.chat.completions.create(
            model=str(model_id).strip(),
            messages=messages,
            max_tokens=max(80, int(max_tokens)),
            temperature=0.1,
            stream=False,
        )
        return str(res.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _generate_live_answer(
    hf_client,
    model_id: str,
    question: str,
    context_text: str,
    max_tokens: int = 420,
    temperature: float = 0.1,
    retry_attempts: int = 3,
    retry_sleep_sec: float = 1.2,
) -> str:
    if not str(context_text or "").strip():
        return "ไม่พบข้อมูลเพียงพอในบริบทที่ให้มา"

    system_prompt = (
        "คุณเป็นผู้ช่วยสอนวิชาโครงสร้างข้อมูล "
        "ตอบโดยอ้างอิงเฉพาะบริบทที่ให้เท่านั้น ห้ามเติมความรู้ภายนอก "
        "หากบริบทมีเพียงบางส่วน ให้สรุปเฉพาะเท่าที่บริบทมีและบอกข้อจำกัดสั้นๆ "
        "ให้ตอบว่า 'ไม่พบข้อมูลเพียงพอในบริบทที่ให้มา' เฉพาะกรณีที่บริบทไม่มีข้อมูลเกี่ยวข้องจริงๆ เท่านั้น "
        "ตอบเป็นภาษาไทย กระชับ แต่ครบใจความสำคัญ"
    )
    user_prompt = (
        f"บริบท:\n{context_text}\n\n"
        f"คำถาม: {question}\n"
        "ตอบให้ชัดเจนและครบประเด็นจากบริบท"
    )
    res = None
    last_exc = None
    for attempt in range(1, max(1, int(retry_attempts)) + 1):
        try:
            res = hf_client.chat.completions.create(
                model=str(model_id).strip(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max(80, int(max_tokens)),
                temperature=float(temperature),
                stream=False,
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= max(1, int(retry_attempts)):
                raise
            time.sleep(max(0.2, float(retry_sleep_sec)) * attempt)
    if res is None and last_exc is not None:
        raise last_exc
    text = ""
    finish_reason = None
    try:
        text = str(res.choices[0].message.content or "").strip()
        finish_reason = str(getattr(res.choices[0], "finish_reason", "") or "").strip()
    except Exception:
        text = ""
        finish_reason = None

    if text and _response_needs_continuation(text, finish_reason):
        continuation = _continue_live_answer(
            hf_client=hf_client,
            model_id=model_id,
            question=question,
            context_text=context_text,
            partial_answer=text,
            max_tokens=max(120, min(260, int(max_tokens) // 2)),
        )
        if continuation:
            text = _merge_with_overlap(text, continuation)
    return text or "ไม่พบข้อมูลเพียงพอในบริบทที่ให้มา"


def _attach_ambiguous_answers(
    amb_rows: list[dict],
    *,
    topic_titles: dict[str, str],
    cache_csv_path: Path,
    resolve_live: bool,
    refresh_live: bool,
    hf_model_id: str,
    max_tokens: int,
    temperature: float,
    context_doc_limit: int,
) -> tuple[list[dict], dict]:
    def _is_error_answer(v: str) -> bool:
        txt = str(v or "").strip()
        return txt.startswith("ERROR:")

    cache_rows = _load_answer_rows(cache_csv_path)
    by_order, by_question = _build_answer_map(cache_rows)

    for row in amb_rows:
        order = int(str(row.get("order", "")).strip() or 0)
        q = str(row.get("question", "")).strip()
        cached = by_order.get(order) or by_question.get(q) or ""
        row["assistant_answer"] = str(cached or "").strip()

    stats = {
        "cache_hits": sum(
            1
            for r in amb_rows
            if str(r.get("assistant_answer", "")).strip()
            and (not _is_error_answer(r.get("assistant_answer", "")))
        ),
        "live_generated": 0,
        "live_errors": 0,
        "live_enabled": bool(resolve_live),
        "hf_model_id": str(hf_model_id).strip(),
    }
    if not resolve_live:
        return amb_rows, stats

    try:
        rag = _load_rag_system()
        hf_client = _load_hf_client()
    except Exception as exc:
        stats["live_errors"] = len(amb_rows)
        stats["error"] = str(exc)
        return amb_rows, stats

    total = len(amb_rows)
    for idx, row in enumerate(amb_rows, start=1):
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        if (not refresh_live) and str(row.get("assistant_answer", "")).strip() and (
            not _is_error_answer(row.get("assistant_answer", ""))
        ):
            continue

        pred_id = str(row.get("pred_topic_id", "")).strip()
        topic_hint = _topic_label(pred_id, topic_titles) if pred_id else ""
        filters = _build_rag_filters(question, topic_hint, pred_id)
        try:
            docs = _retrieve_docs_for_live_answer(rag, question, filters, top_k=20, top_n=8)
            context_text, _ = _build_context_from_docs(docs, context_doc_limit=context_doc_limit)
            answer = _generate_live_answer(
                hf_client=hf_client,
                model_id=hf_model_id,
                question=question,
                context_text=context_text,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            row["assistant_answer"] = str(answer).strip()
            stats["live_generated"] += 1
        except Exception as exc:
            row["assistant_answer"] = f"ERROR: {str(exc)}"
            stats["live_errors"] += 1
        if idx % 10 == 0 or idx == total:
            print(f"[AMB_LIVE] {idx}/{total} done")

    cache_write_rows = []
    for row in amb_rows:
        cache_write_rows.append(
            {
                "order": row.get("order", ""),
                "question": row.get("question", ""),
                "pred_topic_id": row.get("pred_topic_id", ""),
                "assistant_answer": row.get("assistant_answer", ""),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
    _save_csv(
        cache_csv_path,
        cache_write_rows,
        ["order", "question", "pred_topic_id", "assistant_answer", "updated_at"],
    )
    return amb_rows, stats


def _attach_direct_answers(
    rows: list[dict],
    *,
    topic_titles: dict[str, str],
    cache_csv_path: Path,
    resolve_live: bool,
    refresh_live: bool,
    hf_model_id: str,
    max_tokens: int,
    temperature: float,
    context_doc_limit: int,
    force_thai_answers: bool = True,
) -> tuple[list[dict], dict]:
    def _is_error_answer(v: str) -> bool:
        txt = str(v or "").strip()
        return txt.startswith("ERROR:")

    cache_rows = _load_answer_rows(cache_csv_path)
    by_order, by_question = _build_answer_map(cache_rows)

    for row in rows:
        order = int(str(row.get("order", "")).strip() or 0)
        q = str(row.get("question", "")).strip()
        cached = by_order.get(order) or by_question.get(q) or ""
        if cached:
            row["assistant_answer"] = str(cached).strip()

    stats = {
        "cache_hits": sum(
            1
            for r in rows
            if str(r.get("assistant_answer", "")).strip()
            and (not _is_error_answer(r.get("assistant_answer", "")))
        ),
        "live_generated": 0,
        "live_errors": 0,
        "live_enabled": bool(resolve_live),
        "hf_model_id": str(hf_model_id).strip(),
        "force_thai_answers": bool(force_thai_answers),
    }
    if not resolve_live:
        return rows, stats

    try:
        rag = _load_rag_system()
        hf_client = _load_hf_client()
    except Exception as exc:
        stats["live_errors"] = len(rows)
        stats["error"] = str(exc)
        return rows, stats

    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        existing = str(row.get("assistant_answer", "")).strip()
        if (not refresh_live) and existing and (not _is_error_answer(existing)):
            if (not force_thai_answers) or _has_thai_text(existing):
                continue

        target_id = (
            str(row.get("expected_topic_id", "")).strip()
            or str(row.get("new_topic_id", "")).strip()
        )
        topic_hint = _topic_label(target_id, topic_titles) if target_id else ""
        filters = _build_rag_filters(question, topic_hint, target_id)
        try:
            docs = _retrieve_docs_for_live_answer(rag, question, filters, top_k=20, top_n=8)
            context_text, _ = _build_context_from_docs(docs, context_doc_limit=context_doc_limit)
            answer = _generate_live_answer(
                hf_client=hf_client,
                model_id=hf_model_id,
                question=question,
                context_text=context_text,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            row["assistant_answer"] = str(answer).strip()
            stats["live_generated"] += 1
        except Exception as exc:
            row["assistant_answer"] = f"ERROR: {str(exc)}"
            stats["live_errors"] += 1
        if idx % 10 == 0 or idx == total:
            print(f"[DIRECT_LIVE] {idx}/{total} done")

    cache_write_rows = []
    for row in rows:
        cache_write_rows.append(
            {
                "order": row.get("order", ""),
                "question": row.get("question", ""),
                "expected_topic_id": row.get("expected_topic_id", ""),
                "assistant_answer": row.get("assistant_answer", ""),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
    _save_csv(
        cache_csv_path,
        cache_write_rows,
        ["order", "question", "expected_topic_id", "assistant_answer", "updated_at"],
    )
    return rows, stats


def _build_confusion(rows: list[dict], pred_key: str) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    expected_ids = []
    predicted_ids = []
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in rows:
        exp = str(r.get("expected_topic_id", "")).strip()
        pred = str(r.get(pred_key, "")).strip()
        if not exp:
            continue
        if not pred:
            pred = "(none)"
        table[exp][pred] += 1
        expected_ids.append(exp)
        predicted_ids.append(pred)

    exp_sorted = _sort_topic_ids(sorted(set(expected_ids)))
    pred_sorted = _sort_topic_ids(sorted(set(predicted_ids)))
    return exp_sorted, pred_sorted, table


def _save_confusion_csv(path: Path, exp_ids: list[str], pred_ids: list[str], table: dict[str, dict[str, int]]) -> None:
    rows = []
    fields = ["expected_topic_id"] + pred_ids + ["row_total"]
    for exp in exp_ids:
        row_total = 0
        row = {"expected_topic_id": exp}
        for pred in pred_ids:
            v = int(table.get(exp, {}).get(pred, 0))
            row[pred] = v
            row_total += v
        row["row_total"] = row_total
        rows.append(row)
    _save_csv(path, rows, fields)


def _confusion_payload(exp_ids: list[str], pred_ids: list[str], table: dict[str, dict[str, int]]) -> dict:
    matrix = [[int(table.get(exp, {}).get(pred, 0)) for pred in pred_ids] for exp in exp_ids]
    max_value = 0
    for row in matrix:
        for v in row:
            if v > max_value:
                max_value = v
    return {
        "rows": exp_ids,
        "cols": pred_ids,
        "matrix": matrix,
        "max_value": max_value,
    }


def _build_expert_review_rows(rows: list[dict], amb_rows: list[dict], topic_titles: dict[str, str]) -> list[dict]:
    out = []
    for r in rows:
        expected_id = str(r.get("expected_topic_id", "")).strip()
        predicted_id = str(r.get("new_topic_id", "")).strip()
        expected_topic = _topic_label(expected_id, topic_titles) if expected_id else ""
        predicted_topic = _topic_label(predicted_id, topic_titles) if predicted_id else ""
        out.append(
            {
                "review_id": str(r.get("order", "")).strip(),
                "category": str(r.get("category", "")).strip(),
                "question": str(r.get("question", "")).strip(),
                "expected_topic": expected_topic,
                "predicted_topic": predicted_topic,
                "assistant_answer": str(r.get("assistant_answer", "")).strip(),
                "answer_correct(Y/N)": "",
            }
        )
    for r in amb_rows:
        pred_id = str(r.get("pred_topic_id", "")).strip()
        predicted_topic = _topic_label(pred_id, topic_titles) if pred_id else "(none)"
        amb_expected_id = str(r.get("expected_topic_id", "")).strip()
        if amb_expected_id:
            expected_topic = _topic_label(amb_expected_id, topic_titles)
        else:
            expected_topic = f"should_attempt={str(r.get('should_attempt', '')).strip()}"
        out.append(
            {
                "review_id": f"A{str(r.get('order', '')).strip()}",
                "category": "ambiguous_in_domain",
                "question": str(r.get("question", "")).strip(),
                "expected_topic": expected_topic,
                "predicted_topic": predicted_topic,
                "assistant_answer": str(r.get("assistant_answer", "")).strip(),
                "answer_correct(Y/N)": "",
            }
        )
    return out


def _render_dashboard_html(
    out_path: Path,
    rows: list[dict],
    amb_rows: list[dict],
    summary: dict,
    conf_old_payload: dict,
    conf_new_payload: dict,
    expert_review_csv_path: str,
) -> None:
    payload = {
        "summary": summary,
        "rows": rows,
        "ambiguousRows": amb_rows,
        "confOld": conf_old_payload,
        "confNew": conf_new_payload,
        "expertReviewCsv": expert_review_csv_path,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    html_text = """<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Topic Routing Regression Dashboard</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --surface: #ffffff;
      --text: #1e293b;
      --muted: #64748b;
      --accent: #0f766e;
      --ok: #15803d;
      --bad: #b91c1c;
      --border: #e2e8f0;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 16px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
    }}
    .card .k {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .card .v {{
      font-size: 22px;
      font-weight: 700;
    }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 14px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    input, select {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
      background: #fff;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 980px;
      background: #fff;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      text-align: left;
      padding: 8px;
      font-size: 13px;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f8fafc;
      z-index: 1;
    }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 600;
    }}
    .ok {{ background: #dcfce7; color: var(--ok); }}
    .bad {{ background: #fee2e2; color: var(--bad); }}
    .same {{ background: #e2e8f0; color: #334155; }}
    .heat-grid {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .heat-grid table {{
      min-width: 760px;
    }}
    .small {{
      color: var(--muted);
      font-size: 12px;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Topic Routing Regression Dashboard</h1>
    <div class="sub">Regression + confusion matrix + expert review (รวมคำถามและคำตอบที่แชทตอบ)</div>

    <div class="cards">
      <div class="card"><div class="k">Queries</div><div class="v" id="kQueries"></div></div>
      <div class="card"><div class="k">Old Accuracy</div><div class="v" id="kOldAcc"></div></div>
      <div class="card"><div class="k">New Accuracy</div><div class="v" id="kNewAcc"></div></div>
      <div class="card"><div class="k">Improved</div><div class="v" id="kImproved"></div></div>
      <div class="card"><div class="k">Regressed</div><div class="v" id="kRegressed"></div></div>
      <div class="card"><div class="k">Ambiguous Pass Rate</div><div class="v" id="kAmbRate"></div></div>
      <div class="card"><div class="k">Direct Queries</div><div class="v" id="kDirect"></div></div>
      <div class="card"><div class="k">Ambiguous Queries</div><div class="v" id="kAmbTotal"></div></div>
    </div>

    <div class="panel">
      <h2>Regression Table</h2>
      <div class="toolbar">
        <select id="categoryFilter">
          <option value="all">all</option>
          <option value="direct">direct</option>
          <option value="ambiguous_in_domain">ambiguous_in_domain</option>
        </select>
        <select id="passFilter">
          <option value="all">all results</option>
          <option value="pass">pass only</option>
          <option value="fail">fail only</option>
        </select>
        <input id="searchInput" type="text" placeholder="ค้นหาคำถาม..." />
      </div>
      <div class="table-wrap">
        <table id="regTable">
          <thead>
            <tr>
              <th>order</th>
              <th>category</th>
              <th>question</th>
              <th>assistant_answer</th>
              <th>expected</th>
              <th>old</th>
              <th>new</th>
              <th>old pass/fail</th>
              <th>new pass/fail</th>
              <th>delta</th>
            </tr>
          </thead>
          <tbody id="regBody"></tbody>
        </table>
      </div>
    </div>

    <div class="panel">
      <h2>Confusion Matrix (Old)</h2>
      <div class="small">expected_topic_id เป็นแถว, predicted_topic_id เป็นคอลัมน์</div>
      <div id="heatOld" class="heat-grid"></div>
    </div>

    <div class="panel">
      <h2>Confusion Matrix (New)</h2>
      <div class="small">expected_topic_id เป็นแถว, predicted_topic_id เป็นคอลัมน์</div>
      <div id="heatNew" class="heat-grid"></div>
    </div>

    <div class="panel">
      <h2>Ambiguous In-domain Check</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>order</th>
              <th>question</th>
              <th>should_attempt</th>
              <th>attempted</th>
              <th>pred_topic_id</th>
              <th>assistant_answer</th>
              <th>pass/fail</th>
            </tr>
          </thead>
          <tbody id="ambBody"></tbody>
        </table>
      </div>
    </div>

    <div class="panel">
      <h2>Expert Review CSV</h2>
      <div class="small">ไฟล์สำหรับผู้เชี่ยวชาญตรวจเนื้อหา: <a href="../{html.escape(expert_review_csv_path)}">{html.escape(expert_review_csv_path)}</a></div>
      <div class="small">คอลัมน์หลัก: question, expected_topic, predicted_topic, assistant_answer, answer_correct(Y/N)</div>
    </div>
  </div>

  <script>
    const DATA = {payload_json};

    const $ = (id) => document.getElementById(id);
    const pct = (x) => (x === null || x === undefined || Number.isNaN(x)) ? "-" : `${{(Number(x) * 100).toFixed(2)}}%`;
    const esc = (s) => String(s ?? "").replace(/[&<>]/g, (m) => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[m]));
    const pill = (v) => {{
      if (v === "pass") return `<span class="pill ok">pass</span>`;
      if (v === "fail") return `<span class="pill bad">fail</span>`;
      if (v === "improved") return `<span class="pill ok">improved</span>`;
      if (v === "regressed") return `<span class="pill bad">regressed</span>`;
      if (v === "same") return `<span class="pill same">same</span>`;
      return esc(v);
    }};

    function renderSummary() {{
      const s = DATA.summary || {{}};
      $("kQueries").textContent = s.queries_total ?? "-";
      $("kOldAcc").textContent = pct(s.old_accuracy);
      $("kNewAcc").textContent = pct(s.new_accuracy);
      $("kImproved").textContent = s.improved_count ?? "-";
      $("kRegressed").textContent = s.regressed_count ?? "-";
      $("kAmbRate").textContent = pct(s.ambiguous_pass_rate);
      $("kDirect").textContent = s.direct_total ?? "-";
      $("kAmbTotal").textContent = s.ambiguous_total ?? "-";
    }}

    function renderRegressionTable() {{
      const tbody = $("regBody");
      const rows = DATA.rows || [];
      const category = $("categoryFilter").value;
      const passFilter = $("passFilter").value;
      const q = $("searchInput").value.trim().toLowerCase();
      let htmlRows = "";
      for (const r of rows) {{
        if (category !== "all" && r.category !== category) continue;
        if (passFilter !== "all" && r.new_pass_fail !== passFilter) continue;
        const hay = `${{r.question || ""}} ${{r.assistant_answer || ""}} ${{r.expected_topic_id || ""}} ${{r.new_topic_id || ""}}`.toLowerCase();
        if (q && !hay.includes(q)) continue;
        htmlRows += `<tr>
          <td>${{esc(r.order)}}</td>
          <td>${{esc(r.category)}}</td>
          <td>${{esc(r.question)}}</td>
          <td>${{esc(r.assistant_answer)}}</td>
          <td>${{esc(r.expected_topic_id)}}</td>
          <td>${{esc(r.old_topic_id)}}</td>
          <td>${{esc(r.new_topic_id)}}</td>
          <td>${{pill(r.old_pass_fail)}}</td>
          <td>${{pill(r.new_pass_fail)}}</td>
          <td>${{pill(r.delta)}}</td>
        </tr>`;
      }}
      tbody.innerHTML = htmlRows || '<tr><td colspan="10">no rows</td></tr>';
    }}

    function renderAmbiguousTable() {{
      const tbody = $("ambBody");
      const rows = DATA.ambiguousRows || [];
      let htmlRows = "";
      for (const r of rows) {{
        htmlRows += `<tr>
          <td>${{esc(r.order)}}</td>
          <td>${{esc(r.question)}}</td>
          <td>${{esc(r.should_attempt)}}</td>
          <td>${{esc(r.attempted)}}</td>
          <td>${{esc(r.pred_topic_id)}}</td>
          <td>${{esc(r.assistant_answer)}}</td>
          <td>${{pill(r.pass_fail)}}</td>
        </tr>`;
      }}
      tbody.innerHTML = htmlRows || '<tr><td colspan="7">no rows</td></tr>';
    }}

    function renderHeatmap(targetId, conf) {{
      const host = $(targetId);
      if (!conf || !conf.rows || !conf.cols) {{
        host.innerHTML = "<div class='small'>no data</div>";
        return;
      }}
      const maxV = Math.max(1, Number(conf.max_value || 0));
      let html = "<table><thead><tr><th>expected \\\\ predicted</th>";
      for (const c of conf.cols) html += `<th>${{esc(c)}}</th>`;
      html += "</tr></thead><tbody>";
      for (let i = 0; i < conf.rows.length; i++) {{
        html += `<tr><th>${{esc(conf.rows[i])}}</th>`;
        const mRow = conf.matrix[i] || [];
        for (let j = 0; j < conf.cols.length; j++) {{
          const v = Number(mRow[j] || 0);
          const alpha = 0.08 + (v / maxV) * 0.92;
          const bg = `rgba(15, 118, 110, ${{alpha.toFixed(3)}})`;
          const fg = alpha > 0.55 ? "#ffffff" : "#0f172a";
          html += `<td style="background:${{bg}};color:${{fg}};text-align:right;font-weight:600;">${{v}}</td>`;
        }}
        html += "</tr>";
      }}
      html += "</tbody></table>";
      host.innerHTML = html;
    }}

    renderSummary();
    renderRegressionTable();
    renderAmbiguousTable();
    renderHeatmap("heatOld", DATA.confOld);
    renderHeatmap("heatNew", DATA.confNew);
    $("categoryFilter").addEventListener("change", renderRegressionTable);
    $("passFilter").addEventListener("change", renderRegressionTable);
    $("searchInput").addEventListener("input", renderRegressionTable);
  </script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")


def _render_expert_review_dashboard_html(out_path: Path, expert_rows: list[dict]) -> None:
    payload_json = json.dumps(expert_rows, ensure_ascii=False)
    html_text = """<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Topic Routing Expert Review</title>
  <style>
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: #f8fafc;
      color: #1e293b;
    }
    .container {
      max-width: 1600px;
      margin: 0 auto;
      padding: 18px;
    }
    .panel {
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 12px;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .card {
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 10px;
      background: #f8fafc;
    }
    .card .k {
      color: #64748b;
      font-size: 12px;
      margin-bottom: 4px;
    }
    .card .v {
      font-size: 20px;
      font-weight: 700;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }
    button, input {
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
      background: #fff;
    }
    button {
      background: #0f766e;
      color: #fff;
      border-color: #0f766e;
      cursor: pointer;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 1450px;
    }
    th, td {
      border-bottom: 1px solid #e2e8f0;
      padding: 8px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }
    th {
      background: #f8fafc;
      position: sticky;
      top: 0;
    }
    .small {
      color: #64748b;
      font-size: 12px;
      margin-bottom: 8px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Expert Review Dashboard</h1>
    <div class="small">Direct: mark whether answer is content-correct (Y/N). Ambiguous: mark whether handling is appropriate (Y/N).</div>

    <div class="panel">
      <h2>Confusion-style Summary</h2>
      <div class="cards">
        <div class="card"><div class="k">Direct: Y</div><div class="v" id="kDirectY">0</div></div>
        <div class="card"><div class="k">Direct: N</div><div class="v" id="kDirectN">0</div></div>
        <div class="card"><div class="k">Direct: Pending</div><div class="v" id="kDirectP">0</div></div>
        <div class="card"><div class="k">Ambiguous: Y</div><div class="v" id="kAmbY">0</div></div>
        <div class="card"><div class="k">Ambiguous: N</div><div class="v" id="kAmbN">0</div></div>
        <div class="card"><div class="k">Ambiguous: Pending</div><div class="v" id="kAmbP">0</div></div>
      </div>
      <div class="table-wrap" id="matrixHost"></div>
    </div>

    <div class="panel">
      <div class="toolbar">
        <input id="searchInput" type="text" placeholder="search question/topic/answer..." />
        <button id="exportBtn" type="button">Export Reviewed CSV</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>review_id</th>
              <th>category</th>
              <th>question</th>
              <th>expected_topic</th>
              <th>predicted_topic</th>
              <th>assistant_answer</th>
              <th>answer_correct(Y/N)</th>
            </tr>
          </thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const ROWS = __PAYLOAD_JSON__;
    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s ?? "").replace(/[&<>]/g, (m) => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[m]));

    function computeCounts() {
      const counters = {
        directY: 0, directN: 0, directP: 0,
        ambY: 0, ambN: 0, ambP: 0
      };
      for (const r of ROWS) {
        const cat = String(r.category || "").trim();
        const mark = String(r["answer_correct(Y/N)"] || "").trim().toUpperCase();
        const isDirect = cat === "direct";
        if (isDirect) {
          if (mark === "Y") counters.directY += 1;
          else if (mark === "N") counters.directN += 1;
          else counters.directP += 1;
        } else {
          if (mark === "Y") counters.ambY += 1;
          else if (mark === "N") counters.ambN += 1;
          else counters.ambP += 1;
        }
      }
      return counters;
    }

    function renderMatrix() {
      const c = computeCounts();
      $("kDirectY").textContent = c.directY;
      $("kDirectN").textContent = c.directN;
      $("kDirectP").textContent = c.directP;
      $("kAmbY").textContent = c.ambY;
      $("kAmbN").textContent = c.ambN;
      $("kAmbP").textContent = c.ambP;
      $("matrixHost").innerHTML = `
        <table>
          <thead>
            <tr>
              <th>question_type</th>
              <th>Y</th>
              <th>N</th>
              <th>Pending</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>direct</td><td>${c.directY}</td><td>${c.directN}</td><td>${c.directP}</td></tr>
            <tr><td>ambiguous</td><td>${c.ambY}</td><td>${c.ambN}</td><td>${c.ambP}</td></tr>
          </tbody>
        </table>
      `;
    }

    function render() {
      const q = $("searchInput").value.trim().toLowerCase();
      const tbody = $("tbody");
      let html = "";
      for (let i = 0; i < ROWS.length; i++) {
        const r = ROWS[i];
        const hay = `${r.question || ""} ${r.expected_topic || ""} ${r.predicted_topic || ""} ${r.assistant_answer || ""}`.toLowerCase();
        if (q && !hay.includes(q)) continue;
        html += `<tr data-row="${i}">
          <td>${esc(r.review_id)}</td>
          <td>${esc(r.category)}</td>
          <td>${esc(r.question)}</td>
          <td>${esc(r.expected_topic)}</td>
          <td>${esc(r.predicted_topic)}</td>
          <td>${esc(r.assistant_answer)}</td>
          <td>
            <select data-key="answer_correct(Y/N)">
              <option value="" ${!r["answer_correct(Y/N)"] ? "selected" : ""}></option>
              <option value="Y" ${r["answer_correct(Y/N)"] === "Y" ? "selected" : ""}>Y</option>
              <option value="N" ${r["answer_correct(Y/N)"] === "N" ? "selected" : ""}>N</option>
            </select>
          </td>
        </tr>`;
      }
      tbody.innerHTML = html || '<tr><td colspan="7">no rows</td></tr>';

      tbody.querySelectorAll("tr[data-row]").forEach((tr) => {
        const idx = Number(tr.getAttribute("data-row"));
        tr.querySelectorAll("[data-key]").forEach((el) => {
          el.addEventListener("input", () => {
            const key = el.getAttribute("data-key");
            ROWS[idx][key] = el.value;
            renderMatrix();
          });
          el.addEventListener("change", () => {
            const key = el.getAttribute("data-key");
            ROWS[idx][key] = el.value;
            renderMatrix();
          });
        });
      });
    }

    function toCsv(rows) {
      const headers = [
        "review_id","category","question","expected_topic","predicted_topic","assistant_answer","answer_correct(Y/N)"
      ];
      const escCsv = (v) => {
        const s = String(v ?? "");
        const needsQuote = s.includes('"') || s.includes(",") || s.includes(String.fromCharCode(10));
        return needsQuote ? ('"' + s.replace(/"/g, '""') + '"') : s;
      };
      const lines = [headers.join(",")];
      for (const r of rows) {
        lines.push(headers.map((h) => escCsv(r[h])).join(","));
      }
      return lines.join("\\n");
    }

    function exportCsv() {
      const csv = toCsv(ROWS);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "topic_routing_expert_review_filled.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    $("searchInput").addEventListener("input", render);
    $("exportBtn").addEventListener("click", exportCsv);
    render();
    renderMatrix();
  </script>
</body>
</html>
"""
    html_text = html_text.replace("__PAYLOAD_JSON__", payload_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build old-vs-new topic routing regression report.")
    ap.add_argument("--queries", default="eval/topic_routing_regression_queries.csv")
    ap.add_argument("--ambiguous-queries", default="eval/topic_routing_ambiguous_queries.csv")
    ap.add_argument("--answers-csv", default="eval/topic_routing_answers_seed.csv")
    ap.add_argument("--out-ambiguous-answers-csv", default="logs/topic_routing_ambiguous_answers_latest.csv")
    ap.add_argument("--resolve-ambiguous-answers-live", action="store_true")
    ap.add_argument("--refresh-ambiguous-answers-live", action="store_true")
    ap.add_argument("--out-direct-answers-csv", default="logs/topic_routing_direct_answers_latest.csv")
    ap.add_argument("--resolve-direct-answers-live", action="store_true")
    ap.add_argument("--refresh-direct-answers-live", action="store_true")
    ap.add_argument("--hf-model-id", default=os.getenv("CHAT_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507"))
    ap.add_argument("--ambiguous-answer-max-tokens", type=int, default=520)
    ap.add_argument("--ambiguous-answer-temperature", type=float, default=0.1)
    ap.add_argument("--ambiguous-context-doc-limit", type=int, default=5)
    ap.add_argument("--direct-answer-max-tokens", type=int, default=420)
    ap.add_argument("--direct-answer-temperature", type=float, default=0.1)
    ap.add_argument("--direct-context-doc-limit", type=int, default=5)
    ap.add_argument(
        "--thai-only-ambiguous-questions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--runtime-events", default="logs/runtime_events.jsonl")
    ap.add_argument("--topic-hierarchy", default="indexes/hierarchical/topic_hierarchy.json")
    ap.add_argument("--from-timestamp", default="2026-02-26T10:24:48")
    ap.add_argument("--out-regression-csv", default="logs/topic_routing_regression_latest.csv")
    ap.add_argument("--out-confusion-old-csv", default="logs/topic_confusion_old_latest.csv")
    ap.add_argument("--out-confusion-new-csv", default="logs/topic_confusion_new_latest.csv")
    ap.add_argument("--out-ambiguous-csv", default="logs/topic_routing_ambiguous_latest.csv")
    ap.add_argument("--out-expert-review-csv", default="logs/topic_routing_expert_review_latest.csv")
    ap.add_argument("--out-dashboard-html", default="logs/topic_routing_dashboard_latest.html")
    ap.add_argument("--out-expert-dashboard-html", default="logs/topic_routing_expert_review_dashboard_latest.html")
    ap.add_argument("--out-summary-json", default="logs/topic_routing_regression_summary_latest.json")
    ap.add_argument("--min-new-accuracy", type=float, default=1.0)
    ap.add_argument("--min-ambiguous-pass-rate", type=float, default=1.0)
    ap.add_argument("--enforce", action="store_true")
    args = ap.parse_args()

    queries = _load_csv_auto(Path(args.queries))
    ambiguous = _load_csv_auto(Path(args.ambiguous_queries))
    events = _parse_runtime_events(Path(args.runtime_events))
    topic_titles = _load_topic_titles(Path(args.topic_hierarchy))
    answer_rows = _load_answer_rows(Path(args.answers_csv))
    ans_by_order, ans_by_question = _build_answer_map(answer_rows)
    # Keep existing answer text if this run does not provide a replacement.
    existing_regression_rows = _load_answer_rows(Path(args.out_regression_csv))
    old_ans_by_order, old_ans_by_question = _build_answer_map(existing_regression_rows)

    old_topics = _extract_old_topics(
        events=events,
        needed=len(queries),
        from_timestamp=args.from_timestamp,
    )

    rows = []
    old_pass_ct = 0
    new_pass_ct = 0
    expected_ct = 0
    for i, q in enumerate(queries):
        order = int(str(q.get("order", i + 1)).strip() or (i + 1))
        question = str(q.get("question", "")).strip()
        category = str(q.get("category", "")).strip()
        expected_raw = str(q.get("expected_topic_id", "")).strip()
        expected = _canonical_topic_id(expected_raw)

        old_label = old_topics[i] if i < len(old_topics) else ""
        old_id_raw = _select_most_specific_section_id(old_label)
        old_id = _canonical_topic_id(old_id_raw)
        new_id_raw = _infer_section_id_from_question(question)
        new_id = new_id_raw
        old_ok = None
        new_ok = None
        if expected:
            expected_ct += 1
            old_ok = old_id == expected
            new_ok = new_id == expected
            old_pass_ct += 1 if old_ok else 0
            new_pass_ct += 1 if new_ok else 0

        rows.append(
            {
                "order": order,
                "category": category,
                "question": question,
                "assistant_answer": (
                    ans_by_order.get(order)
                    or ans_by_question.get(question)
                    or old_ans_by_order.get(order)
                    or old_ans_by_question.get(question)
                    or ""
                ),
                "expected_topic_id_raw": expected_raw,
                "expected_topic_id": expected,
                "old_topic_label": old_label,
                "old_topic_id_raw": old_id_raw,
                "old_topic_id": old_id,
                "new_topic_label": _topic_label(new_id, topic_titles),
                "new_topic_id_raw": new_id_raw,
                "new_topic_id": new_id,
                "old_pass_fail": _to_bool_text(old_ok),
                "new_pass_fail": _to_bool_text(new_ok),
                "delta": "" if (old_ok is None or new_ok is None) else ("improved" if (not old_ok and new_ok) else ("regressed" if (old_ok and not new_ok) else "same")),
            }
        )

    rows = sorted(rows, key=lambda x: int(x["order"]))
    rows, direct_answer_stats = _attach_direct_answers(
        rows,
        topic_titles=topic_titles,
        cache_csv_path=Path(args.out_direct_answers_csv),
        resolve_live=bool(args.resolve_direct_answers_live),
        refresh_live=bool(args.refresh_direct_answers_live),
        hf_model_id=args.hf_model_id,
        max_tokens=int(args.direct_answer_max_tokens),
        temperature=float(args.direct_answer_temperature),
        context_doc_limit=int(args.direct_context_doc_limit),
        force_thai_answers=True,
    )
    _save_csv(
        Path(args.out_regression_csv),
        rows,
        [
            "order",
            "category",
            "question",
            "assistant_answer",
            "expected_topic_id_raw",
            "expected_topic_id",
            "old_topic_label",
            "old_topic_id_raw",
            "old_topic_id",
            "new_topic_label",
            "new_topic_id_raw",
            "new_topic_id",
            "old_pass_fail",
            "new_pass_fail",
            "delta",
        ],
    )

    exp_old, pred_old, conf_old = _build_confusion(rows, pred_key="old_topic_id")
    exp_new, pred_new, conf_new = _build_confusion(rows, pred_key="new_topic_id")
    _save_confusion_csv(Path(args.out_confusion_old_csv), exp_old, pred_old, conf_old)
    _save_confusion_csv(Path(args.out_confusion_new_csv), exp_new, pred_new, conf_new)

    amb_rows = []
    for i, q in enumerate(ambiguous, start=1):
        question_raw = str(q.get("question", "")).strip()
        question = _normalize_ambiguous_question_for_display(
            question_raw,
            thai_only=bool(args.thai_only_ambiguous_questions),
        )
        should_attempt = str(q.get("should_attempt", "1")).strip() in {"1", "true", "yes", "on"}
        expected_amb_raw = str(q.get("expected_topic_id", "")).strip()
        expected_amb = _canonical_topic_id(expected_amb_raw)
        pred_id = _infer_section_id_from_question(question_raw)
        attempted = bool(pred_id)
        if should_attempt:
            if expected_amb:
                passed = bool(pred_id) and (pred_id == expected_amb)
            else:
                passed = attempted
        else:
            passed = not attempted
        amb_rows.append(
            {
                "order": int(str(q.get("order", i)).strip() or i),
                "question": question,
                "should_attempt": int(should_attempt),
                "expected_topic_id": expected_amb,
                "attempted": int(attempted),
                "pred_topic_id": pred_id,
                "pred_topic_label": _topic_label(pred_id, topic_titles),
                "assistant_answer": "",
                "pass_fail": _to_bool_text(passed),
            }
        )
    amb_rows = sorted(amb_rows, key=lambda x: int(x["order"]))
    amb_rows, amb_answer_stats = _attach_ambiguous_answers(
        amb_rows,
        topic_titles=topic_titles,
        cache_csv_path=Path(args.out_ambiguous_answers_csv),
        resolve_live=bool(args.resolve_ambiguous_answers_live),
        refresh_live=bool(args.refresh_ambiguous_answers_live),
        hf_model_id=args.hf_model_id,
        max_tokens=int(args.ambiguous_answer_max_tokens),
        temperature=float(args.ambiguous_answer_temperature),
        context_doc_limit=int(args.ambiguous_context_doc_limit),
    )
    _save_csv(
        Path(args.out_ambiguous_csv),
        amb_rows,
        [
            "order",
            "question",
            "should_attempt",
            "expected_topic_id",
            "attempted",
            "pred_topic_id",
            "pred_topic_label",
            "assistant_answer",
            "pass_fail",
        ],
    )

    expert_rows = _build_expert_review_rows(rows, amb_rows, topic_titles)
    existing_expert_rows = _load_answer_rows(Path(args.out_expert_review_csv))
    expert_rows = _merge_existing_expert_labels(expert_rows, existing_expert_rows)
    _save_csv(
        Path(args.out_expert_review_csv),
        expert_rows,
        [
            "review_id",
            "category",
            "question",
            "expected_topic",
            "predicted_topic",
            "assistant_answer",
            "answer_correct(Y/N)",
        ],
    )

    old_accuracy = round(old_pass_ct / expected_ct, 4) if expected_ct else None
    new_accuracy = round(new_pass_ct / expected_ct, 4) if expected_ct else None
    ambiguous_pass = sum(1 for r in amb_rows if r.get("pass_fail") == "pass")
    ambiguous_pass_rate = round(ambiguous_pass / len(amb_rows), 4) if amb_rows else None

    conf_old_payload = _confusion_payload(exp_old, pred_old, conf_old)
    conf_new_payload = _confusion_payload(exp_new, pred_new, conf_new)

    summary = {
        "queries_total": len(rows),
        "direct_total": sum(1 for r in rows if str(r.get("category", "")).strip() == "direct"),
        "queries_with_expected": expected_ct,
        "old_pass": old_pass_ct,
        "new_pass": new_pass_ct,
        "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy,
        "improved_count": sum(1 for r in rows if r.get("delta") == "improved"),
        "regressed_count": sum(1 for r in rows if r.get("delta") == "regressed"),
        "ambiguous_total": len(amb_rows),
        "ambiguous_pass": ambiguous_pass,
        "ambiguous_pass_rate": ambiguous_pass_rate,
        "direct_answer_stats": direct_answer_stats,
        "ambiguous_answer_stats": amb_answer_stats,
        "thresholds": {
            "min_new_accuracy": args.min_new_accuracy,
            "min_ambiguous_pass_rate": args.min_ambiguous_pass_rate,
            "enforce": bool(args.enforce),
        },
        "files": {
            "regression_csv": args.out_regression_csv,
            "confusion_old_csv": args.out_confusion_old_csv,
            "confusion_new_csv": args.out_confusion_new_csv,
            "ambiguous_csv": args.out_ambiguous_csv,
            "direct_answers_csv": args.out_direct_answers_csv,
            "ambiguous_answers_csv": args.out_ambiguous_answers_csv,
            "expert_review_csv": args.out_expert_review_csv,
            "dashboard_html": args.out_dashboard_html,
            "expert_dashboard_html": args.out_expert_dashboard_html,
            "answers_csv": args.answers_csv,
        },
    }

    _render_dashboard_html(
        out_path=Path(args.out_dashboard_html),
        rows=rows,
        amb_rows=amb_rows,
        summary=summary,
        conf_old_payload=conf_old_payload,
        conf_new_payload=conf_new_payload,
        expert_review_csv_path=args.out_expert_review_csv,
    )
    _render_expert_review_dashboard_html(Path(args.out_expert_dashboard_html), expert_rows)

    Path(args.out_summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.enforce:
        failures = []
        if new_accuracy is None:
            failures.append("new_accuracy is missing")
        elif float(new_accuracy) < float(args.min_new_accuracy):
            failures.append(f"new_accuracy={new_accuracy} < min_new_accuracy={args.min_new_accuracy}")

        if amb_rows and ambiguous_pass_rate is not None and float(ambiguous_pass_rate) < float(args.min_ambiguous_pass_rate):
            failures.append(
                f"ambiguous_pass_rate={ambiguous_pass_rate} < min_ambiguous_pass_rate={args.min_ambiguous_pass_rate}"
            )

        if failures:
            for msg in failures:
                print(f"[ENFORCE_FAIL] {msg}")
            sys.exit(2)


if __name__ == "__main__":
    main()

