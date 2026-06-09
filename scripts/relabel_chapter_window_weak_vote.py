"""
Apply TOC-constrained relabel with weak-vote rules for a page window.

Primary use:
- Fix operation-heavy, image-heavy ranges where OCR text is noisy.
- Keep labels within allowed TOC topic scope.
- Optional sequence smoothing (HMM/CRF-style linear-chain Viterbi).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
HEADING_TOPIC_RE = re.compile(r"^\s{0,3}#{1,6}\s*(\d+\.\d+(?:\.\d+)*)\b", re.MULTILINE)
FIG_RE = re.compile(r"(?:ภาพที่|figure)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_pages(md_text: str) -> dict[str, str]:
    headers = list(PAGE_HEADER_RE.finditer(md_text))
    out: dict[str, str] = {}
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        source = m.group(1).strip()
        page = int(m.group(2))
        page_id = f"{source}:{page}"
        out[page_id] = md_text[start:end].strip()
    return out


def topic_path(topic_id: str, topic_map: dict[str, dict]) -> list[str]:
    out = []
    cur = str(topic_id or "").strip()
    for _ in range(10):
        if not cur or cur not in topic_map:
            break
        row = topic_map[cur]
        out.append(f"{cur} {str(row.get('title', '')).strip()}".strip())
        cur = str(row.get("parent_id", "") or "").strip()
    out.reverse()
    return out


def heading_votes(text: str, allowed: set[str]) -> dict[str, float]:
    votes: dict[str, float] = {}
    for m in HEADING_TOPIC_RE.finditer(text or ""):
        tid = str(m.group(1)).strip()
        if tid in allowed:
            votes[tid] = max(votes.get(tid, 0.0), 0.90)
        for a in allowed:
            if tid.startswith(a + ".") and a not in votes:
                votes[a] = max(votes.get(a, 0.0), 0.55)
    return votes


def figure_votes(text: str) -> dict[str, float]:
    votes: dict[str, float] = {}
    nums = []
    for m in FIG_RE.finditer(text or ""):
        try:
            nums.append(float(m.group(1)))
        except Exception:
            continue
    for n in nums:
        if 2.12 <= n <= 2.14:
            votes["2.4.1"] = max(votes.get("2.4.1", 0.0), 0.78)
            votes["2.4"] = max(votes.get("2.4", 0.0), 0.50)
        if 2.15 <= n <= 2.23:
            votes["2.4.2"] = max(votes.get("2.4.2", 0.0), 0.82)
            votes["2.4"] = max(votes.get("2.4", 0.0), 0.52)
    return votes


def keyword_votes(text: str) -> dict[str, float]:
    t = str(text or "").lower()
    votes: dict[str, float] = {}
    has_linked = any(x in t for x in ["ลิงค์ลิสต์", "ลิงก์ลิสต์", "linked list", "node", "โหนด"])
    has_operation = any(x in t for x in ["การทำงาน", "เพิ่มโหนด", "ลบโหนด", "แทรกโหนด", "insert", "delete", "remove"])
    has_structure = any(x in t for x in ["โครงสร้างลิงค์ลิสต์", "head node", "head structure"])

    if has_linked and has_operation:
        votes["2.4.2"] = max(votes.get("2.4.2", 0.0), 0.74)
        votes["2.4"] = max(votes.get("2.4", 0.0), 0.35)
    if has_linked and has_structure:
        votes["2.4.1"] = max(votes.get("2.4.1", 0.0), 0.66)
        votes["2.4"] = max(votes.get("2.4", 0.0), 0.32)
    if has_linked and not has_operation and not has_structure:
        votes["2.4"] = max(votes.get("2.4", 0.0), 0.28)
    return votes


def topic_order_key(topic_id: str) -> tuple[int, ...]:
    out = []
    for part in str(topic_id or "").split("."):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            out.append(9999)
    return tuple(out) if out else (9999,)


def transition_score(prev_tid: str, cur_tid: str) -> float:
    if prev_tid == cur_tid:
        return 0.55

    prev_key = topic_order_key(prev_tid)
    cur_key = topic_order_key(cur_tid)
    if prev_key[:-1] == cur_key[:-1]:
        # sibling transitions: prefer short forward progression.
        if cur_key > prev_key:
            return 0.25
        return -0.25

    if cur_tid.startswith(prev_tid + ".") or prev_tid.startswith(cur_tid + "."):
        return 0.18

    return -0.35


def softmax_distribution(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    max_v = max(vals)
    exps = {k: math.exp(float(v) - float(max_v)) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    return {k: float(v) / float(z) for k, v in exps.items()}


def viterbi_decode(
    *,
    page_ids: list[str],
    emissions: dict[str, dict[str, float]],
    states: list[str],
) -> tuple[dict[str, str], dict[str, float]]:
    if not page_ids or not states:
        return {}, {}

    logp: dict[str, dict[str, float]] = {}
    backptr: dict[str, dict[str, str]] = {}
    eps = 1e-9
    first = page_ids[0]
    first_em = emissions.get(first, {})
    logp[first] = {}
    backptr[first] = {}
    for s in states:
        e = float(first_em.get(s, 0.0))
        logp[first][s] = math.log(max(eps, e))
        backptr[first][s] = ""

    for i in range(1, len(page_ids)):
        pid = page_ids[i]
        prev_pid = page_ids[i - 1]
        cur_em = emissions.get(pid, {})
        logp[pid] = {}
        backptr[pid] = {}
        for s in states:
            e = float(cur_em.get(s, 0.0))
            e_log = math.log(max(eps, e))
            best_prev = None
            best_val = -1e18
            for p in states:
                cand = float(logp[prev_pid].get(p, -1e18)) + float(transition_score(p, s)) + e_log
                if cand > best_val:
                    best_val = cand
                    best_prev = p
            logp[pid][s] = best_val
            backptr[pid][s] = str(best_prev or "")

    last = page_ids[-1]
    last_state = max(states, key=lambda s: logp[last].get(s, -1e18))
    seq: dict[str, str] = {last: last_state}
    for i in range(len(page_ids) - 1, 0, -1):
        pid = page_ids[i]
        prev_pid = page_ids[i - 1]
        prev_state = backptr[pid].get(seq[pid], "")
        seq[prev_pid] = prev_state if prev_state in states else states[0]

    confidence = {}
    for pid in page_ids:
        vals = sorted([float(v) for v in logp[pid].values()], reverse=True)
        if len(vals) >= 2:
            confidence[pid] = round(float(vals[0] - vals[1]), 6)
        elif vals:
            confidence[pid] = round(float(vals[0]), 6)
        else:
            confidence[pid] = 0.0
    return seq, confidence


def apply_relabel(
    *,
    hierarchy: dict,
    page_text_map: dict[str, str],
    source_name: str,
    page_start: int,
    page_end: int,
    allowed_topics: set[str],
    min_conf: float,
    sequence_decoder: str,
) -> tuple[dict, dict]:
    topics = hierarchy.get("topics", []) if isinstance(hierarchy, dict) else []
    pages = hierarchy.get("pages", []) if isinstance(hierarchy, dict) else []
    if not isinstance(topics, list) or not isinstance(pages, list):
        raise ValueError("invalid hierarchy format")

    topic_map = {
        str(t.get("topic_id", "")).strip(): t
        for t in topics
        if isinstance(t, dict) and str(t.get("topic_id", "")).strip()
    }
    allowed_topics = {x for x in allowed_topics if x in topic_map}
    if not allowed_topics:
        raise ValueError("no valid allowed_topics found in hierarchy")

    changes = []
    sequence_candidates = []

    for p in pages:
        if not isinstance(p, dict):
            continue
        src = str(p.get("source", "")).strip()
        page = int(p.get("page", 0) or 0)
        if src != source_name or page < int(page_start) or page > int(page_end):
            continue

        pid = str(p.get("page_id", "")).strip()
        text = page_text_map.get(pid, "")
        old_best = str(p.get("best_topic", "")).strip()
        votes = {tid: 0.0 for tid in allowed_topics}

        hv = heading_votes(text, allowed_topics)
        fv = figure_votes(text)
        kv = keyword_votes(text)
        for tid, s in hv.items():
            if tid in votes:
                votes[tid] += float(s)
        for tid, s in fv.items():
            if tid in votes:
                votes[tid] += float(s)
        for tid, s in kv.items():
            if tid in votes:
                votes[tid] += float(s)

        top_topics = p.get("top_topics", []) if isinstance(p.get("top_topics", []), list) else []
        for rank, t in enumerate(top_topics[:3], start=1):
            tid = str((t or {}).get("topic_id", "")).strip()
            if tid in votes:
                votes[tid] += 0.22 / float(rank)
        if old_best in votes:
            votes[old_best] += 0.10

        local_best_tid = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        local_conf = float(votes.get(local_best_tid, 0.0))
        sequence_candidates.append(
            {
                "page": page,
                "page_id": pid,
                "old_best": old_best,
                "votes": votes,
                "local_best": local_best_tid,
                "local_conf": local_conf,
                "record": p,
            }
        )

    selected_map: dict[str, str] = {}
    sequence_conf_map: dict[str, float] = {}
    if str(sequence_decoder or "").strip().lower() in {"hmm", "crf"} and sequence_candidates:
        sequence_candidates = sorted(sequence_candidates, key=lambda x: int(x["page"]))
        page_ids = [str(x["page_id"]) for x in sequence_candidates]
        emissions = {
            str(x["page_id"]): softmax_distribution({k: float(v) for k, v in dict(x["votes"]).items()})
            for x in sequence_candidates
        }
        states = sorted(list(allowed_topics), key=topic_order_key)
        selected_map, sequence_conf_map = viterbi_decode(page_ids=page_ids, emissions=emissions, states=states)

    for row in sequence_candidates:
        p = row["record"]
        pid = str(row["page_id"])
        old_best = str(row["old_best"])
        local_best_tid = str(row["local_best"])
        local_conf = float(row["local_conf"])
        votes = dict(row["votes"])

        best_tid = str(selected_map.get(pid, local_best_tid))
        conf = float(local_conf if pid not in sequence_conf_map else sequence_conf_map.get(pid, local_conf))

        if (best_tid not in allowed_topics) or (conf < float(min_conf) and best_tid != local_best_tid):
            best_tid = local_best_tid
            conf = local_conf

        if conf < float(min_conf):
            # Conservative fallback: keep within allowed scope via nearest parent.
            if old_best in allowed_topics:
                best_tid = old_best
            elif old_best.startswith("2.4"):
                best_tid = "2.4"
            else:
                best_tid = "2.4.2"

        if best_tid != old_best:
            changes.append(
                {
                    "page_id": pid,
                    "old": old_best,
                    "new": best_tid,
                    "local_best": local_best_tid,
                    "votes": votes,
                    "confidence": round(conf, 4),
                }
            )

        p["best_topic"] = best_tid
        p["best_topic_path"] = topic_path(best_tid, topic_map)
        p["relabel_info"] = {
            "method": "toc_constrained_weak_vote_sequence" if selected_map else "toc_constrained_weak_vote",
            "decoder": str(sequence_decoder or "none"),
            "allowed_topics": sorted(list(allowed_topics)),
            "confidence": round(conf, 4),
            "local_best": local_best_tid,
            "local_confidence": round(local_conf, 4),
            "votes": {k: round(float(v), 4) for k, v in sorted(votes.items())},
        }

        top_topics = p.get("top_topics", []) if isinstance(p.get("top_topics", []), list) else []
        new_top = []
        new_top.append({"topic_id": best_tid, "score": round(max(conf, 1e-6), 6), "path": topic_path(best_tid, topic_map)})
        seen = {best_tid}
        for t in top_topics:
            tid = str((t or {}).get("topic_id", "")).strip()
            if not tid or tid in seen:
                continue
            seen.add(tid)
            new_top.append(t)
            if len(new_top) >= 3:
                break
        p["top_topics"] = new_top

    # Rebuild page maps and topic_to_pages.
    page_to_best = {}
    page_to_top = {}
    topic_to_pages = {tid: [] for tid in topic_map.keys()}
    for p in pages:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("page_id", "")).strip()
        if not pid:
            continue
        best = str(p.get("best_topic", "")).strip()
        if best:
            page_to_best[pid] = best
            topic_to_pages.setdefault(best, []).append(pid)
        tlist = []
        for t in p.get("top_topics", []) if isinstance(p.get("top_topics", []), list) else []:
            tid = str((t or {}).get("topic_id", "")).strip()
            if tid:
                tlist.append(tid)
                topic_to_pages.setdefault(tid, []).append(pid)
        if tlist:
            page_to_top[pid] = tlist

    # De-dup topic_to_pages
    for tid, ids in list(topic_to_pages.items()):
        seen = set()
        dedup = []
        for pid in ids:
            if pid in seen:
                continue
            seen.add(pid)
            dedup.append(pid)
        topic_to_pages[tid] = dedup

    hierarchy["pages"] = pages
    hierarchy["page_to_best_topic"] = page_to_best
    hierarchy["page_to_top_topics"] = page_to_top
    hierarchy["topic_to_pages"] = topic_to_pages

    report = {
        "source": source_name,
        "page_start": int(page_start),
        "page_end": int(page_end),
        "sequence_decoder": str(sequence_decoder or "none"),
        "allowed_topics": sorted(list(allowed_topics)),
        "changed_pages": len(changes),
        "changes": changes,
    }
    return hierarchy, report


def main() -> None:
    ap = argparse.ArgumentParser(description="TOC-constrained weak-vote relabel for a chapter window.")
    ap.add_argument("--hierarchy-index", default="indexes/hierarchical/topic_hierarchy.json")
    ap.add_argument("--markdown", default="final_extracted_text_only.md")
    ap.add_argument("--source", default="data_structure_data_ch1_to_ch5.pdf")
    ap.add_argument("--page-start", type=int, default=24)
    ap.add_argument("--page-end", type=int, default=28)
    ap.add_argument("--allowed-topics", default="2.4,2.4.1,2.4.2")
    ap.add_argument("--min-confidence", type=float, default=0.60)
    ap.add_argument("--sequence-decoder", default="hmm", choices=["none", "hmm", "crf"])
    ap.add_argument("--output", default="indexes/hierarchical/topic_hierarchy.json")
    ap.add_argument("--report", default="logs/chapter_window_relabel_report_latest.json")
    args = ap.parse_args()

    h_path = Path(args.hierarchy_index)
    md_path = Path(args.markdown)
    out_path = Path(args.output)
    report_path = Path(args.report)
    if not h_path.exists():
        raise FileNotFoundError(f"missing hierarchy index: {h_path}")
    if not md_path.exists():
        raise FileNotFoundError(f"missing markdown: {md_path}")

    hierarchy = json.loads(h_path.read_text(encoding="utf-8"))
    page_text_map = parse_pages(md_path.read_text(encoding="utf-8"))
    allowed = {x.strip() for x in str(args.allowed_topics).split(",") if x.strip()}

    updated, report = apply_relabel(
        hierarchy=hierarchy,
        page_text_map=page_text_map,
        source_name=str(args.source),
        page_start=int(args.page_start),
        page_end=int(args.page_end),
        allowed_topics=allowed,
        min_conf=float(args.min_confidence),
        sequence_decoder=str(args.sequence_decoder),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={out_path}")
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    main()
