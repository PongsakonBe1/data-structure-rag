"""
Repair markdown structure by injecting missing TOC headings as anchors.

This is a post-OCR structural normalization step:
- does not alter core OCR paragraph text
- only adds missing markdown headings from list_hitachi/topic_hierarchy
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
TOC_LINE_RE = re.compile(r"[^\d]*([0-9]+(?:\.[0-9]+)*)\s+(.*)")


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^0-9a-zA-Z\u0E00-\u0E7F\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_pages(md: str) -> list[dict]:
    rows = []
    headers = list(PAGE_HEADER_RE.finditer(md))
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md)
        rows.append(
            {
                "source": m.group(1).strip(),
                "page": int(m.group(2)),
                "body": md[start:end].rstrip(),
            }
        )
    return rows


def parse_toc(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        s = s.replace("โข", " ").strip()
        m = TOC_LINE_RE.search(s)
        if not m:
            continue
        tid = m.group(1).strip()
        title = m.group(2).strip()
        if not tid or not title:
            continue
        rows.append({"topic_id": tid, "title": title})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Inject missing TOC headings into markdown pages.")
    ap.add_argument("--input-markdown", default="final_extracted_text_only.md")
    ap.add_argument("--toc", default="list_hitachi.txt")
    ap.add_argument("--hierarchy", default="indexes/hierarchical/topic_hierarchy.json")
    ap.add_argument("--output-markdown", default="final_extracted_text_only_structured.md")
    ap.add_argument("--report", default="logs/markdown_toc_anchor_repair_latest.json")
    ap.add_argument("--force-anchor-all", action="store_true")
    args = ap.parse_args()

    in_md = Path(args.input_markdown)
    toc_path = Path(args.toc)
    hierarchy_path = Path(args.hierarchy)
    if not in_md.exists():
        raise FileNotFoundError(f"missing markdown: {in_md}")
    if not toc_path.exists():
        raise FileNotFoundError(f"missing toc: {toc_path}")
    if not hierarchy_path.exists():
        raise FileNotFoundError(f"missing hierarchy: {hierarchy_path}")

    pages = parse_pages(in_md.read_text(encoding="utf-8"))
    toc_rows = parse_toc(toc_path)
    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    topic_to_pages = hierarchy.get("topic_to_pages", {}) if isinstance(hierarchy, dict) else {}

    whole_text_norm = normalize_text("\n".join(p["body"] for p in pages))
    page_idx = {(str(p["source"]).strip(), int(p["page"])): i for i, p in enumerate(pages)}
    # primary source in this project
    source_name = str(pages[0]["source"]).strip() if pages else ""

    injected = []
    for t in toc_rows:
        topic_id = str(t["topic_id"]).strip()
        title = str(t["title"]).strip()
        title_norm = normalize_text(title)
        if not title_norm:
            continue
        if (not args.force_anchor_all) and (title_norm in whole_text_norm):
            continue

        page_refs = topic_to_pages.get(topic_id, [])
        page_num = None
        if isinstance(page_refs, list) and page_refs:
            nums = []
            for ref in page_refs:
                m = re.search(r":(\d+)\s*$", str(ref))
                if m:
                    nums.append(int(m.group(1)))
            if nums:
                page_num = min(nums)

        # fallback: parent topic
        if page_num is None and "." in topic_id:
            parent = topic_id.rsplit(".", 1)[0]
            pref = topic_to_pages.get(parent, [])
            if isinstance(pref, list) and pref:
                nums = []
                for ref in pref:
                    m = re.search(r":(\d+)\s*$", str(ref))
                    if m:
                        nums.append(int(m.group(1)))
                if nums:
                    page_num = min(nums)

        if page_num is None:
            continue
        key = (source_name, int(page_num))
        i = page_idx.get(key)
        if i is None:
            continue
        depth = topic_id.count(".") + 1
        heading_level = min(6, max(3, depth + 2))
        heading = f'{"#" * heading_level} {topic_id} {title}'
        body = str(pages[i]["body"] or "").strip()
        if heading in body:
            continue
        pages[i]["body"] = f"{heading}\n\n{body}".strip()
        whole_text_norm += "\n" + title_norm
        injected.append({"topic_id": topic_id, "title": title, "page": int(page_num)})

    out_lines = []
    for p in pages:
        out_lines.append(f"# Source: {p['source']}\n## Page {p['page']}\n\n{p['body']}\n")
    out_md = Path(args.output_markdown)
    out_md.write_text("\n".join(out_lines).strip() + "\n", encoding="utf-8")

    report = {
        "input_markdown": str(in_md.as_posix()),
        "output_markdown": str(out_md.as_posix()),
        "injected_count": len(injected),
        "injected_examples": injected[:20],
    }
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={rep}")


if __name__ == "__main__":
    main()
