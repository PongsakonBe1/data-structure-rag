"""
Build document hierarchical index from list_hitachi (TOC) and extracted markdown pages.

Outputs:
- indexes/hierarchical/topic_hierarchy.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from pythainlp.tokenize import word_tokenize
except Exception:
    word_tokenize = None

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
HEADING_TOPIC_ID_RE = re.compile(r"^\s{0,3}#{1,6}\s*(\d+\.\d+(?:\.\d+)*)\b", re.MULTILINE)
VISUAL_CAPTION_BLOCK_RE = re.compile(r"\n### Visual Captions \(Auto\)\n[\s\S]*$", re.MULTILINE)


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[\[\]\(\)\{\},.:;!?/\\|\"'`~@#$%^&*+=<>]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_text_robust(path: Path, *, allow_legacy_encodings: bool = False) -> str:
    raw = path.read_bytes()
    # TOC and markdown in this project are expected UTF-8.
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc).replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue
    # Optional fallback only for one-time recovery of legacy files.
    if allow_legacy_encodings:
        for enc in ("cp874", "tis-620", "latin-1"):
            try:
                return raw.decode(enc).replace("\r\n", "\n").replace("\r", "\n")
            except UnicodeDecodeError:
                continue
    raise UnicodeDecodeError("utf-8", raw, 0, 1, f"unable to decode {path}")


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if word_tokenize is not None:
        try:
            tokens = [t.strip().lower() for t in word_tokenize(text, engine="newmm") if t.strip()]
            if tokens:
                return tokens
        except Exception:
            pass
    return [t for t in re.split(r"\s+", text) if t]


def parse_pages(md_text: str) -> list[dict]:
    headers = list(PAGE_HEADER_RE.finditer(md_text))
    pages = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        body = md_text[start:end].strip()
        pages.append({"source": m.group(1).strip(), "page": int(m.group(2)), "text": body})
    return pages


def strip_visual_caption_block(text: str) -> str:
    return VISUAL_CAPTION_BLOCK_RE.sub("", str(text or "")).strip()


@dataclass
class TocEntry:
    topic_id: str
    title: str
    level: int
    parent_id: str | None


def parse_toc(path: Path, *, allow_legacy_encodings: bool = False) -> list[TocEntry]:
    text = read_text_robust(path, allow_legacy_encodings=allow_legacy_encodings)
    entries: list[TocEntry] = []

    current_main: str | None = None
    current_sub: str | None = None

    for raw in text.splitlines():
        line = raw.rstrip()
        s = line.strip()
        if not s:
            continue

        # Main section: "1: ..."
        m_main = re.match(r"^(\d+)\s*:\s*(.+)$", s)
        if not m_main:
            m_main = re.match(r"^(?:หัวข้อที่\s*)?(\d+)\s*:\s*(.+)$", s)
        if not m_main:
            # Fallback for occasional TOC lines without ':'.
            m_main2 = re.match(r"^(?:หัวข้อที่\s*)?(\d+)\s+(.+)$", s)
            if m_main2 and ("." not in m_main2.group(1)):
                m_main = m_main2
        if m_main:
            tid = m_main.group(1)
            title = m_main.group(2).strip()
            entries.append(TocEntry(topic_id=tid, title=title, level=1, parent_id=None))
            current_main = tid
            current_sub = None
            continue

        # Bullet sub-section: "• 3.3 ..."
        m_sub = re.match(r"^[•\-]\s*(\d+(?:\.\d+)*)\s+(.+)$", s)
        if not m_sub:
            # Generic section line fallback: allow malformed bullet prefix.
            m_sub = re.match(r"^(?:[^\d]{0,6})?(\d+(?:\.\d+)*)\s+(.+)$", s)
        if m_sub:
            tid = m_sub.group(1)
            title = m_sub.group(2).strip()
            parent = tid.rsplit(".", 1)[0] if "." in tid else current_main
            level = tid.count(".") + 1
            entries.append(TocEntry(topic_id=tid, title=title, level=level, parent_id=parent))
            if level == 2:
                current_sub = tid
            continue

        # Indented children (tab/space) or numbered child line
        if raw.startswith("\t") or raw.startswith("    "):
            m_child = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", s)
            if m_child:
                tid = m_child.group(1)
                title = m_child.group(2).strip()
                parent = tid.rsplit(".", 1)[0] if "." in tid else (current_sub or current_main)
                level = tid.count(".") + 1
            else:
                parent = current_sub or current_main
                if parent:
                    idx = len([e for e in entries if e.parent_id == parent]) + 1
                    tid = f"{parent}.{idx}"
                else:
                    tid = f"misc.{len(entries)+1}"
                title = s
                level = tid.count(".") + 1
            entries.append(TocEntry(topic_id=tid, title=title, level=level, parent_id=parent))
            continue

    # Stable unique by topic_id (keep first occurrence)
    out = []
    seen = set()
    for e in entries:
        if e.topic_id in seen:
            continue
        seen.add(e.topic_id)
        out.append(e)
    return out


def topic_score(topic: TocEntry, page_text: str) -> float:
    t_tokens = tokenize(topic.title)
    if not t_tokens:
        return 0.0

    p_norm = normalize_text(page_text)
    p_tokens = set(tokenize(page_text))

    overlap = sum(1 for t in t_tokens if t in p_tokens)
    overlap_ratio = overlap / max(1, len(set(t_tokens)))

    # Numeric heading bonus: if "3.3" appears in the page.
    code_bonus = 0.0
    if topic.topic_id and topic.topic_id in p_norm:
        code_bonus = 0.25

    # Substring bonus for exact phrase hits.
    phrase_bonus = 0.0
    t_norm = normalize_text(topic.title)
    if t_norm and t_norm in p_norm:
        phrase_bonus = 0.25

    return float(min(1.0, overlap_ratio + code_bonus + phrase_bonus))


def chapter_id_from_topic(topic_id: str) -> str:
    tid = str(topic_id or "").strip()
    return tid.split(".", 1)[0].strip() if tid else ""


def extract_heading_topic_hints(page_text: str) -> set[str]:
    hints: set[str] = set()
    for m in HEADING_TOPIC_ID_RE.finditer(page_text or ""):
        tid = str(m.group(1) or "").strip()
        if tid:
            hints.add(tid)
            # Include parent levels (e.g., 2.4.2 -> 2.4 -> 2).
            parts = tid.split(".")
            for i in range(1, len(parts)):
                hints.add(".".join(parts[:i]))
    return hints


def heading_boost(topic_id: str, heading_hints: set[str]) -> float:
    if not heading_hints:
        return 0.0
    tid = str(topic_id or "").strip()
    if not tid:
        return 0.0
    bonus = 0.0
    for h in heading_hints:
        if tid == h:
            bonus = max(bonus, 0.45)
        elif tid.startswith(h + ".") or h.startswith(tid + "."):
            bonus = max(bonus, 0.26)
        elif chapter_id_from_topic(tid) == chapter_id_from_topic(h):
            bonus = max(bonus, 0.08)
    return bonus


def transition_bonus(prev_topic: str, cur_topic: str, topic_map: dict[str, TocEntry]) -> float:
    if not prev_topic or not cur_topic:
        return 0.0
    if prev_topic == cur_topic:
        return 0.16

    prev = topic_map.get(prev_topic)
    cur = topic_map.get(cur_topic)
    if prev and cur:
        if prev.parent_id and prev.parent_id == cur.parent_id:
            return 0.11
        if prev.topic_id == cur.parent_id or cur.topic_id == prev.parent_id:
            return 0.12

    if chapter_id_from_topic(prev_topic) == chapter_id_from_topic(cur_topic):
        return 0.07
    return -0.03


def viterbi_best_topics(
    page_candidates: list[dict[str, Any]],
    topic_map: dict[str, TocEntry],
) -> list[str]:
    if not page_candidates:
        return []
    dp: list[dict[str, tuple[float, str]]] = []

    for i, row in enumerate(page_candidates):
        cand = row.get("candidates", {})
        hints = row.get("heading_hints", set()) or set()
        cur: dict[str, tuple[float, str]] = {}
        if not cand:
            dp.append(cur)
            continue
        for tid, base_score in cand.items():
            emit = float(base_score) + heading_boost(str(tid), hints)
            if i == 0:
                cur[str(tid)] = (emit, "")
                continue
            best_score = None
            best_prev = ""
            prev_layer = dp[i - 1]
            if not prev_layer:
                cur[str(tid)] = (emit, "")
                continue
            for prev_tid, (prev_score, _) in prev_layer.items():
                score = float(prev_score) + transition_bonus(prev_tid, str(tid), topic_map) + emit
                if best_score is None or score > best_score:
                    best_score = score
                    best_prev = prev_tid
            cur[str(tid)] = (float(best_score if best_score is not None else emit), best_prev)
        dp.append(cur)

    last_idx = len(dp) - 1
    if not dp[last_idx]:
        return ["" for _ in page_candidates]
    last_topic = max(dp[last_idx].items(), key=lambda kv: kv[1][0])[0]

    seq = ["" for _ in page_candidates]
    seq[last_idx] = last_topic
    for i in range(last_idx, 0, -1):
        layer = dp[i]
        cur_tid = seq[i]
        prev_tid = layer.get(cur_tid, (0.0, ""))[1]
        seq[i - 1] = prev_tid
    return seq


def topic_path(topic_id: str, topic_map: dict[str, TocEntry]) -> list[str]:
    out = []
    cur = topic_id
    guard = 0
    while cur and cur in topic_map and guard < 10:
        e = topic_map[cur]
        out.append(f"{e.topic_id} {e.title}")
        cur = e.parent_id or ""
        guard += 1
    out.reverse()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build hierarchical topic index for document pages.")
    ap.add_argument("--toc", default="list_hitachi.txt")
    ap.add_argument("--markdown", default="final_extracted_text_only_structured_full.md")
    ap.add_argument("--output", default="indexes/hierarchical/topic_hierarchy.json")
    ap.add_argument("--min-score", type=float, default=0.15)
    ap.add_argument(
        "--allow-legacy-encodings",
        action="store_true",
        help="Allow cp874/tis-620/latin-1 fallback decoding for one-time legacy recovery.",
    )
    args = ap.parse_args()

    toc_path = Path(args.toc)
    md_path = Path(args.markdown)
    out_path = Path(args.output)

    if not toc_path.exists():
        raise FileNotFoundError(f"TOC file not found: {toc_path}")
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    topics = parse_toc(toc_path, allow_legacy_encodings=bool(args.allow_legacy_encodings))
    pages = parse_pages(read_text_robust(md_path, allow_legacy_encodings=bool(args.allow_legacy_encodings)))

    topic_map = {t.topic_id: t for t in topics}
    topic_to_pages: dict[str, list[str]] = {t.topic_id: [] for t in topics}
    page_assignments = []
    page_candidates: list[dict[str, Any]] = []

    for p in pages:
        source = p["source"]
        page = int(p["page"])
        page_id = f"{source}:{page}"
        text = strip_visual_caption_block(p["text"])
        heading_hints = extract_heading_topic_hints(text)

        scored: list[tuple[float, str]] = []
        for t in topics:
            s = topic_score(t, text)
            if s >= float(args.min_score):
                scored.append((s, t.topic_id))

        scored.sort(reverse=True)
        top = scored[:8]

        if not top and topics:
            # Guaranteed fallback to keep every page mappable for downstream metadata filtering.
            fallback = max(((topic_score(t, text), t.topic_id) for t in topics), key=lambda x: x[0], default=(0.0, ""))
            if fallback[1]:
                top = [fallback]

        cand_map = {tid: float(s) for s, tid in top}
        page_candidates.append(
            {
                "page_id": page_id,
                "source": source,
                "page": page,
                "text": text,
                "heading_hints": heading_hints,
                "candidates": cand_map,
                "top_raw": top,
            }
        )

    best_topic_sequence = viterbi_best_topics(page_candidates, topic_map)

    for i, row in enumerate(page_candidates):
        page_id = str(row.get("page_id", ""))
        source = str(row.get("source", ""))
        page = int(row.get("page", 0))
        heading_hints = row.get("heading_hints", set()) or set()
        top_raw = row.get("top_raw", []) or []
        cand_map = row.get("candidates", {}) or {}

        # Re-rank top topics with heading prior for transparent per-page diagnostics.
        top_reweighted = []
        for s, tid in top_raw:
            w = float(s) + heading_boost(str(tid), heading_hints)
            top_reweighted.append((w, str(tid)))
        top_reweighted.sort(reverse=True)
        top = top_reweighted[:3]

        best_topic = str(best_topic_sequence[i]).strip() if i < len(best_topic_sequence) else ""
        if not best_topic and top:
            best_topic = top[0][1]
        if not best_topic and cand_map:
            best_topic = max(cand_map.items(), key=lambda kv: kv[1])[0]

        if best_topic:
            topic_to_pages.setdefault(best_topic, []).append(page_id)
        for _, tid in top:
            if tid and tid != best_topic:
                topic_to_pages.setdefault(tid, []).append(page_id)

        page_assignments.append(
            {
                "page_id": page_id,
                "source": source,
                "page": page,
                "best_topic": best_topic,
                "best_topic_path": topic_path(best_topic, topic_map) if best_topic else [],
                "top_topics": [
                    {
                        "topic_id": tid,
                        "score": round(float(s), 6),
                        "path": topic_path(tid, topic_map),
                    }
                    for s, tid in top
                ],
                "heading_hints": sorted(list(heading_hints)),
            }
        )

    # De-duplicate page lists.
    for k, v in list(topic_to_pages.items()):
        seen = set()
        out = []
        for pid in v:
            if pid in seen:
                continue
            seen.add(pid)
            out.append(pid)
        topic_to_pages[k] = out

    payload = {
        "topics": [
            {
                "topic_id": t.topic_id,
                "title": t.title,
                "level": t.level,
                "parent_id": t.parent_id,
                "path": topic_path(t.topic_id, topic_map),
                "tokens": tokenize(t.title),
            }
            for t in topics
        ],
        "topic_to_pages": topic_to_pages,
        "page_to_best_topic": {
            str(x.get("page_id", "")): str(x.get("best_topic", ""))
            for x in page_assignments
            if str(x.get("page_id", "")).strip()
        },
        "page_to_top_topics": {
            str(x.get("page_id", "")): [str(y.get("topic_id", "")) for y in (x.get("top_topics", []) or []) if str(y.get("topic_id", "")).strip()]
            for x in page_assignments
            if str(x.get("page_id", "")).strip()
        },
        "pages": page_assignments,
        "summary": {
            "topic_count": len(topics),
            "page_count": len(pages),
            "assigned_pages": sum(1 for x in page_assignments if x.get("best_topic")),
            "min_score": float(args.min_score),
            "encoding_policy": "utf8_strict" if not bool(args.allow_legacy_encodings) else "utf8_with_legacy_fallback",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
