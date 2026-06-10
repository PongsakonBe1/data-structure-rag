"""
ingest.py โ€” Vision Pipeline (The "No-Hallucination" Edition)
Improvements:
1. PROMPT: Explicitly forbids markdown image tags with examples of what NOT to do.
2. PROMPT: Forces 'Structural Description' for diagrams.
3. CODE: Python regex guardrail to strip any remaining image tags.
"""

import os
import time
import pickle
import base64
import re  # Import Regex
import sys
import csv
import fitz  # PyMuPDF
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
from huggingface_hub import InferenceClient
try:
    import ftfy
except Exception:
    ftfy = None
try:
    from .index_integrity import build_index_manifest
except ImportError:
    from index_integrity import build_index_manifest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

# Make console output robust on Windows terminals with legacy encodings.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"
DEBUG_FILE = BASE_DIR / "final_extracted_content.md"
QUALITY_REPORT_FILE = BASE_DIR / "logs" / "ingest_quality_report.csv"

HF_API_KEY = (os.getenv("HUGGINGFACE_READ_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
EMBEDDING_MODEL = "BAAI/bge-m3"
VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "Qwen/Qwen2.5-VL-72B-Instruct").strip()
INGEST_PAGE_RANGE = os.getenv("INGEST_PAGE_RANGE", "").strip()
INGEST_CONTEXTUAL_CHUNKING_ENABLED = os.getenv("INGEST_CONTEXTUAL_CHUNKING_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
INGEST_CONTEXTUAL_CHUNK_MAX_PREFIX_CHARS = max(80, int(os.getenv("INGEST_CONTEXTUAL_CHUNK_MAX_PREFIX_CHARS", "260")))
INGEST_CONTEXTUAL_INCLUDE_SOURCE_PAGE = os.getenv("INGEST_CONTEXTUAL_INCLUDE_SOURCE_PAGE", "1").strip().lower() in {"1", "true", "yes", "on"}
INGEST_CONTEXTUAL_INCLUDE_HEADINGS = os.getenv("INGEST_CONTEXTUAL_INCLUDE_HEADINGS", "1").strip().lower() in {"1", "true", "yes", "on"}
INGEST_CONTEXTUAL_INCLUDE_STRUCTURE = os.getenv("INGEST_CONTEXTUAL_INCLUDE_STRUCTURE", "1").strip().lower() in {"1", "true", "yes", "on"}

STRUCTURE_BLOCK_RE = re.compile(r"\[Structure:\s*([^\]]+)\]", re.IGNORECASE)
STRUCTURE_ARROW_RE = re.compile(r"->|→|-->|=>")
STRUCTURE_BLOCK_BODY_RE = re.compile(
    r"(?ms)^\s*>\s*\[Structure:\s*([^\]\n]+)\]\s*\n(?:\s*>\s*-\s*[^\n]*\n?)*"
)
FIGURE_REF_RE = re.compile(r"(ภาพที่|ตารางที่)\s*\d+(?:\.\d+)?", re.IGNORECASE)
VALID_STRUCTURE_TYPES = {
    "binary tree", "tree", "linked list", "graph", "stack", "queue",
    "flowchart", "sequence", "selection", "decision", "array",
    "node", "subtree", "1 มิติ", "2 มิติ", "2d array", "ตาราง",
}
CAPTION_STRUCTURE_HINTS = [
    (re.compile(r"(ลิงค์ลิสต์|linked list)", re.IGNORECASE), "Linked List"),
    (re.compile(r"(ไบนารีทรี|binary tree)", re.IGNORECASE), "Binary Tree"),
    (re.compile(r"(ต้นไม้|tree)", re.IGNORECASE), "Tree"),
    (re.compile(r"(กราฟ|graph)", re.IGNORECASE), "Graph"),
    (re.compile(r"(สแตก|stack)", re.IGNORECASE), "Stack"),
    (re.compile(r"(คิว|queue)", re.IGNORECASE), "Queue"),
    (re.compile(r"(ผังงาน|flowchart)", re.IGNORECASE), "Flowchart"),
    (re.compile(r"(selection|เลือก)", re.IGNORECASE), "Selection"),
    (re.compile(r"(decision|เงื่อนไข)", re.IGNORECASE), "Decision"),
    (re.compile(r"(sequence|เรียงลำดับ)", re.IGNORECASE), "Sequence"),
    (re.compile(r"(อาร์เรย์|array|1 มิติ|2 มิติ|3 มิติ)", re.IGNORECASE), "Array"),
    (re.compile(r"(โหนด|node)", re.IGNORECASE), "Node"),
    (re.compile(r"(ตาราง|table)", re.IGNORECASE), "ตาราง"),
]

# ---------------------------------------------------------------------------
# AI Vision Function
# ---------------------------------------------------------------------------
def image_to_base64(pix_map):
    img_bytes = pix_map.tobytes("png")
    return base64.b64encode(img_bytes).decode('utf-8')

def clean_hallucinated_images(text: str) -> str:
    """
    Guardrail Function:
    เธ•เธฃเธงเธเธเธฑเธ Markdown Image Tag เธ—เธตเนเนเธกเน€เธ”เธฅเน€เธเธฅเธญเธชเธฃเนเธฒเธเธกเธฒ เนเธฅเนเธงเธฅเธเธ—เธดเนเธเธซเธฃเธทเธญเน€เธเธฅเธตเนเธขเธเน€เธเนเธ Text
    Ex: ![Binary Tree](tree.png) -> [AI เธฅเธทเธกเธเธฃเธฃเธขเธฒเธขเธฃเธนเธเธ เธฒเธเธเธตเน]
    """
    # Pattern: ![alt](url)
    pattern = r"!\[(.*?)\]\(.*?\)"
    
    # เธ–เนเธฒเน€เธเธญ เนเธซเนเน€เธเธฅเธตเนเธขเธเน€เธเนเธเธเนเธญเธเธงเธฒเธกเนเธเนเธเน€เธ•เธทเธญเธ (เธซเธฃเธทเธญเธฅเธเธ—เธดเนเธเนเธเน€เธฅเธขเธเนเนเธ”เน)
    # เนเธ•เนเน€เธฃเธฒเธเธฐเน€เธเธฅเธตเนเธขเธเน€เธเนเธ Blockquote เน€เธเธทเนเธญเนเธซเนเธฃเธนเนเธงเนเธฒเธ•เธฃเธเธเธตเนเน€เธเธขเธกเธตเธฃเธนเธ
    def replacement(match):
        alt_text = match.group(1)
        return f"> [Note: Diagram detected but AI failed to describe. Alt: {alt_text}]"

    return re.sub(pattern, replacement, text)


def repair_text_encoding(text: str) -> str:
    """
    Repair common OCR/VLM encoding artifacts while preserving valid Thai text.
    """
    s = str(text or "")
    if not s:
        return ""
    if ftfy is None:
        return s
    try:
        repaired = ftfy.fix_text(s, normalization="NFC")
    except Exception:
        return s

    # Keep the candidate that looks more like natural Thai.
    thai_re = re.compile(r"[\u0E00-\u0E7F]")
    orig_score = len(thai_re.findall(s))
    rep_score = len(thai_re.findall(repaired))
    if rep_score >= orig_score:
        return repaired
    return s


def strip_markdown_fences(text: str) -> str:
    """Remove wrapping ```markdown fences that models often add."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^\s*```(?:markdown|md)?\s*\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n\s*```\s*$", "", cleaned)
    return cleaned.strip()


def infer_structure_type(text: str) -> str:
    lower = text.lower()
    if "binary tree" in lower or "ไบนารีทรี" in lower or "ต้นไม้" in lower:
        return "Binary Tree"
    if "linked list" in lower or "ลิงก์ลิสต์" in lower:
        return "Linked List"
    if "graph" in lower or "กราฟ" in lower:
        return "Graph"
    if "stack" in lower or "สแตก" in lower:
        return "Stack"
    if "queue" in lower or "คิว" in lower:
        return "Queue"
    return "Unknown"


def contains_structure_markers(text: str) -> bool:
    if not text:
        return False
    if STRUCTURE_BLOCK_RE.search(text):
        return True
    if STRUCTURE_ARROW_RE.search(text):
        # Arrow alone is a weak signal, but acceptable when used in visual relation text.
        return True
    # Avoid broad keyword triggers like "โครงสร้าง" that over-fire on narrative text pages.
    lowered = text.lower()
    keywords = ("binary tree", "linked list", "graph", "stack", "queue", "flowchart", "ไบนารีทรี", "ลิงก์ลิสต์")
    return any(k in lowered for k in keywords)


def extract_structure_types(text: str) -> list[str]:
    types = []
    for match in STRUCTURE_BLOCK_RE.findall(text or ""):
        candidate = str(match).strip()
        if candidate:
            types.append(candidate)
    if not types and contains_structure_markers(text):
        types.append(infer_structure_type(text))
    # preserve order, unique
    seen = set()
    out = []
    for t in types:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def enforce_structure_markdown(text: str) -> str:
    """
    Ensure structure diagrams are represented as text blocks in markdown.
    If model mentions structure but omits [Structure: ...], append a normalized placeholder.
    """
    cleaned = strip_markdown_fences(clean_hallucinated_images(text or ""))
    cleaned = repair_text_encoding(cleaned)
    if not cleaned.strip():
        return ""
    return cleaned


def _normalize_structure_label(label: str) -> str:
    return str(label or "").strip().lower()


def _drop_unknown_or_invalid_structure_blocks(text: str) -> str:
    out = text
    for m in list(STRUCTURE_BLOCK_BODY_RE.finditer(text)):
        label = _normalize_structure_label(m.group(1))
        if label == "unknown" or label not in VALID_STRUCTURE_TYPES:
            out = out.replace(m.group(0), "")
    return out


def _trim_structure_blocks_by_figure_refs(text: str) -> tuple[str, dict]:
    """
    Heuristic quality gate:
    - If no figure/table references on page, remove all structure blocks.
    - If too many structure blocks, cap by figure/table count.
    """
    figure_refs = len(FIGURE_REF_RE.findall(text or ""))
    matches = list(STRUCTURE_BLOCK_BODY_RE.finditer(text or ""))
    structure_count = len(matches)
    max_allowed = 0 if figure_refs == 0 else min(12, figure_refs * 2 + 1)
    removed = 0
    if structure_count <= max_allowed:
        return text, {
            "figure_refs": figure_refs,
            "structure_count": structure_count,
            "max_allowed": max_allowed,
            "removed_structure_blocks": removed,
        }

    rebuilt = []
    last = 0
    for i, m in enumerate(matches):
        rebuilt.append(text[last:m.start()])
        if i < max_allowed:
            rebuilt.append(m.group(0))
        else:
            removed += 1
        last = m.end()
    rebuilt.append(text[last:])
    cleaned = "".join(rebuilt)
    return cleaned, {
        "figure_refs": figure_refs,
        "structure_count": structure_count,
        "max_allowed": max_allowed,
        "removed_structure_blocks": removed,
    }


def _dedupe_structure_blocks(text: str) -> tuple[str, int]:
    matches = list(STRUCTURE_BLOCK_BODY_RE.finditer(text or ""))
    if len(matches) <= 1:
        return text, 0

    rebuilt = []
    last = 0
    seen = set()
    removed = 0
    for m in matches:
        rebuilt.append(text[last:m.start()])
        block = m.group(0)
        key = re.sub(r"\s+", " ", block.strip().lower())
        if key in seen:
            removed += 1
        else:
            seen.add(key)
            rebuilt.append(block)
        last = m.end()
    rebuilt.append(text[last:])
    return "".join(rebuilt), removed


def _append_caption_structure_blocks_if_missing(text: str) -> tuple[str, int]:
    """
    Backfill [Structure: ...] blocks from figure/table captions when missing.
    This targets false negatives without creating blocks on pages that have no references.
    """
    if not text:
        return text, 0
    if STRUCTURE_BLOCK_RE.search(text):
        return text, 0

    figure_lines = [line.strip() for line in text.splitlines() if FIGURE_REF_RE.search(line)]
    if not figure_lines:
        return text, 0

    labels: list[str] = []
    for line in figure_lines:
        for pattern, label in CAPTION_STRUCTURE_HINTS:
            if pattern.search(line):
                labels.append(label)

    if not labels:
        # Fallback: allow page-level hinting only when caption refs exist.
        for pattern, label in CAPTION_STRUCTURE_HINTS:
            if pattern.search(text):
                labels.append(label)

    unique_labels = []
    seen = set()
    for label in labels:
        key = str(label).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_labels.append(str(label).strip())

    if not unique_labels:
        return text, 0

    max_new = min(3, len(figure_lines))
    blocks = []
    for label in unique_labels[:max_new]:
        blocks.append(
            f"> [Structure: {label}]\n"
            "> - อ้างอิงจากคำบรรยายภาพ/ตารางในหน้าเดียวกัน"
        )

    suffix = "\n\n" + "\n\n".join(blocks)
    return text.rstrip() + suffix + "\n", len(blocks)


def sanitize_page_markdown(text: str) -> tuple[str, dict]:
    cleaned = enforce_structure_markdown(text)
    cleaned = _drop_unknown_or_invalid_structure_blocks(cleaned)
    cleaned, caption_injected = _append_caption_structure_blocks_if_missing(cleaned)
    cleaned, dedup_removed = _dedupe_structure_blocks(cleaned)
    cleaned, stats = _trim_structure_blocks_by_figure_refs(cleaned)
    # Collapse excessive empty lines after structure block removals.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    stats["dedup_removed"] = dedup_removed
    stats["caption_injected"] = caption_injected
    return cleaned, stats

def analyze_page_with_qwen(base64_image: str, page_num: int) -> str:
    if not HF_API_KEY:
        print("Error: missing HUGGINGFACE_READ_TOKEN or HUGGINGFACE_API_KEY")
        return ""

    client = InferenceClient(api_key=HF_API_KEY)
    
    # Prompt engineering for OCR+table+diagram extraction with low hallucination.
    prompt_text = (
        "You are an OCR + document-structure extraction engine for Thai CS textbook pages.\n"
        "Your task is to extract ALL visible text EXACTLY as it appears. Be thorough and complete.\n\n"
        "CRITICAL RULES (DO NOT SKIP ANY TEXT):\n"
        "1) Extract ALL section numbers like '1.2.1', '1.2.2' EXACTLY as shown - these are IMPORTANT.\n"
        "2) Extract ALL headings including sub-headings and sub-sub-headings (## 1.2.1, ### 1.2.1.1).\n"
        "3) Extract ALL small text, metadata, and captions - do not skip anything.\n"
        "4) Preserve the exact hierarchy: # for main, ## for 1.x, ### for 1.x.x sections.\n"
        "5) Never output image links (`![...](...)`).\n"
        "6) Do not invent text that is not visible in the page.\n"
        "7) If text is unclear, mark `[ไม่ชัดเจน]` but still try to extract it.\n\n"
        "Extraction rules:\n"
        "- OCR ALL visible Thai/English text including small fonts and section numbers.\n"
        "- If a table is visible, convert it to Markdown table (`| col | ... |`).\n"
        "- If a figure/diagram is visible (nodes/arrows/flow), add a structure block:\n"
        "  > [Structure: <Type>]\n"
        "  > - <relation 1>\n"
        "  > - <relation 2>\n\n"
        "Output format:\n"
        "- Use # for main chapter headings\n"
        "- Use ## for section headings (1.1, 1.2, etc.)\n"
        "- Use ### for subsection headings (1.2.1, 1.2.2, etc.)\n"
        "- Keep figure/table captions in-line (e.g., ภาพที่ 3.4 ...).\n"
        "- Return complete Markdown with ALL text extracted."
    )

    print(f"    -> OCR page {page_num} with {VISION_MODEL_ID} ...")
    
    # Retry Logic
    for attempt in range(3):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]

            response = client.chat.completions.create(
                model=VISION_MODEL_ID,
                messages=messages,
                max_tokens=4600,
                temperature=0.0,
                top_p=0.9,
            )

            raw_content = str(response.choices[0].message.content or "")
            
            # --- POST-PROCESSING GUARDRAIL ---
            # เน€เธฃเธตเธขเธเนเธเนเธเธฑเธเธเนเธเธฑเธเธฅเนเธฒเธ Image Tag เธ—เธฑเธเธ—เธต
            clean_content = enforce_structure_markdown(raw_content)
            
            return clean_content.strip()

        except Exception as e:
            error_msg = str(e)
            if "504" in error_msg:
                print("       timeout 504; retry in 10s")
                time.sleep(10)
            elif "503" in error_msg:
                print("       model warming 503; retry in 10s")
                time.sleep(10)
            elif "429" in error_msg:
                print("       rate-limited 429; backoff 60s")
                time.sleep(60)
            elif "402" in error_msg:
                print("       credit depleted (402)")
                return ""
            else:
                print(f"       retryable error: {error_msg}")
                time.sleep(5)
    
    print(f"       failed page {page_num}")
    return ""

# ---------------------------------------------------------------------------
# Process Steps
# ---------------------------------------------------------------------------
def process_pdf_vision(data_dir: Path) -> list[dict]:
    # เนเธเนเนเธ glob เนเธซเนเธฃเธญเธเธฃเธฑเธ path
    pdf_paths = list(data_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"โ เนเธกเนเธเธ PDF เนเธ {data_dir}")
        return []

    all_pages = []
    selected_pages: set[int] | None = None
    if INGEST_PAGE_RANGE:
        selected_pages = set()
        for part in INGEST_PAGE_RANGE.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                try:
                    lo, hi = int(a), int(b)
                    if lo > hi:
                        lo, hi = hi, lo
                    selected_pages.update(range(lo, hi + 1))
                except ValueError:
                    continue
            else:
                try:
                    selected_pages.add(int(part))
                except ValueError:
                    continue
    
    for pdf_path in pdf_paths:
        print(f"๐“ เธเธฃเธฐเธกเธงเธฅเธเธฅ: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        for i, page in enumerate(doc):
            page_num = i + 1
            if selected_pages is not None and page_num not in selected_pages:
                continue
            print(f"  [{page_num}/{total_pages}] เธญเนเธฒเธเธซเธเนเธฒเธเธฃเธฐเธ”เธฒเธฉ...")
            
            # Zoom 3.0x 
            pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
            b64_img = image_to_base64(pix)
            
            extracted_text = analyze_page_with_qwen(b64_img, page_num)
            
            if extracted_text:
                sanitized_text, stats = sanitize_page_markdown(extracted_text)
                if not sanitized_text:
                    print(f"     โ ๏ธ เธซเธเนเธฒ {page_num} เธซเธฅเธฑเธ sanitize เนเธฅเนเธงเธงเนเธฒเธ")
                else:
                    all_pages.append(
                        {
                            "source": pdf_path.name,
                            "page": str(page_num),
                            "content": sanitized_text,
                            "figure_refs": stats.get("figure_refs", 0),
                            "structure_count_raw": stats.get("structure_count", 0),
                            "structure_max_allowed": stats.get("max_allowed", 0),
                            "structure_removed": stats.get("removed_structure_blocks", 0),
                            "structure_dedup_removed": stats.get("dedup_removed", 0),
                            "structure_caption_injected": stats.get("caption_injected", 0),
                        }
                    )
                    print(
                        "     โ… เธชเธณเน€เธฃเนเธ "
                        f"({len(sanitized_text)} chars, figures={stats.get('figure_refs', 0)}, "
                        f"structures={stats.get('structure_count', 0)}, "
                        f"caption_injected={stats.get('caption_injected', 0)}, "
                        f"dedup_removed={stats.get('dedup_removed', 0)}, "
                        f"trim_removed={stats.get('removed_structure_blocks', 0)})"
                    )
            else:
                print(f"     โ ๏ธ เธซเธเนเธฒ {page_num} เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅ")
            
            time.sleep(1)

    return all_pages

def save_debug_markdown(pages: list[dict]):
    print(f"\n๐’พ เธเธฑเธเธ—เธถเธเนเธเธฅเนเธ•เธฃเธงเธเธชเธญเธ: {DEBUG_FILE}")
    with open(DEBUG_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Verification Document\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for item in pages:
            f.write(f"# Source: {item['source']}\n")
            f.write(f"## Page {item['page']}\n\n")
            f.write(item["content"].strip() + "\n\n")
    print(f"โ… เธเธฑเธเธ—เธถเธเน€เธชเธฃเนเธเธชเธดเนเธ! เน€เธเนเธเนเธเธฅเนเนเธ”เนเธ—เธตเน {DEBUG_FILE}")

def write_quality_report(pages: list[dict]):
    QUALITY_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QUALITY_REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "page",
                "figure_refs",
                "structure_count_raw",
                "structure_max_allowed",
                "structure_dedup_removed",
                "structure_removed",
                "structure_caption_injected",
            ],
        )
        writer.writeheader()
        for item in pages:
            writer.writerow(
                {
                    "source": item.get("source", ""),
                    "page": item.get("page", ""),
                    "figure_refs": item.get("figure_refs", 0),
                    "structure_count_raw": item.get("structure_count_raw", 0),
                    "structure_max_allowed": item.get("structure_max_allowed", 0),
                    "structure_dedup_removed": item.get("structure_dedup_removed", 0),
                    "structure_removed": item.get("structure_removed", 0),
                    "structure_caption_injected": item.get("structure_caption_injected", 0),
                }
            )
    print(f"   Wrote quality report: {QUALITY_REPORT_FILE}")


def build_figure_evidence_docs(pages: list[dict]) -> list[Document]:
    """
    Build compact evidence chunks around figure/table references and structure blocks.
    This improves recall for theory/diagram questions where key facts are near captions.
    """
    docs: list[Document] = []
    seen = set()
    for item in pages:
        content = str(item.get("content", "") or "")
        if not content.strip():
            continue
        source = str(item.get("source", "reference_document"))
        page = str(item.get("page", ""))
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if not (FIGURE_REF_RE.search(line) or "[Structure:" in line):
                continue
            lo = max(0, i - 4)
            hi = min(len(lines), i + 9)
            excerpt = "\n".join(lines[lo:hi]).strip()
            if len(excerpt) < 40:
                continue
            key = re.sub(r"\s+", " ", excerpt.lower())
            if key in seen:
                continue
            seen.add(key)
            docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        "source": source,
                        "page": page,
                        "evidence_type": "figure_excerpt",
                        "anchor_line": int(i + 1),
                    },
                )
            )
    return docs


def _compact_text_snippet(text: str, max_chars: int) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(raw) <= max_chars:
        return raw
    return raw[: max(0, max_chars - 1)].rstrip() + "…"


def _build_chunk_context_prefix(meta: dict) -> str:
    parts: list[str] = []
    source = str(meta.get("source", "")).strip()
    page = str(meta.get("page", "")).strip()
    chapter = str(meta.get("chapter", "")).strip()
    section = str(meta.get("section", "")).strip()
    topic_path = str(meta.get("topic_path", "")).strip()
    structure_types = [str(x).strip() for x in (meta.get("structure_types", []) or []) if str(x).strip()]
    evidence_type = str(meta.get("evidence_type", "")).strip()

    if INGEST_CONTEXTUAL_INCLUDE_SOURCE_PAGE and source:
        source_page = source + (f":{page}" if page else "")
        parts.append(f"เอกสาร={source_page}")
    if INGEST_CONTEXTUAL_INCLUDE_HEADINGS:
        if chapter:
            parts.append(f"บท={chapter}")
        if section:
            parts.append(f"หัวข้อย่อย={section}")
        if topic_path:
            parts.append(f"เส้นทางหัวข้อ={topic_path}")
    if INGEST_CONTEXTUAL_INCLUDE_STRUCTURE and structure_types:
        parts.append(f"โครงสร้างที่พบ={', '.join(structure_types[:4])}")
    if evidence_type:
        parts.append(f"ชนิดหลักฐาน={evidence_type}")

    if not parts:
        return ""
    prefix = " | ".join(parts)
    prefix = _compact_text_snippet(prefix, INGEST_CONTEXTUAL_CHUNK_MAX_PREFIX_CHARS)
    return f"[บริบทของชิ้นข้อมูล: {prefix}]"


def apply_contextual_chunking(chunks: list[Document]) -> int:
    if not INGEST_CONTEXTUAL_CHUNKING_ENABLED:
        return 0
    applied = 0
    for doc in chunks:
        meta = getattr(doc, "metadata", {}) or {}
        prefix = _build_chunk_context_prefix(meta)
        if not prefix:
            continue
        body = str(doc.page_content or "").strip()
        if not body:
            continue
        if body.startswith(prefix):
            continue
        doc.metadata["context_prefix"] = prefix
        doc.metadata["contextual_chunking"] = True
        doc.page_content = f"{prefix}\n{body}"
        applied += 1
    return applied


def build_indexes(pages: list[dict]):
    if not pages:
        return

    print("\n๐”จ เธชเธฃเนเธฒเธ Index...")
    source_docs = []
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")])
    for item in pages:
        content = item.get("content", "")
        source = item.get("source", "reference_document")
        page = item.get("page", "")
        base_meta = {"source": source, "page": str(page)}
        per_page_docs = splitter.split_text(content)
        if not per_page_docs:
            per_page_docs = [Document(page_content=content, metadata={})]
        for d in per_page_docs:
            meta = getattr(d, "metadata", {}) or {}
            meta.update(base_meta)
            d.metadata = meta
            source_docs.append(d)

    figure_docs = build_figure_evidence_docs(pages)
    if figure_docs:
        source_docs.extend(figure_docs)
        print(f"   Added figure evidence docs: {len(figure_docs)}")
    
    # Fallback
    rec_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_chunks = rec_splitter.split_documents(source_docs)

    # Enrich metadata for provenance/citation tracing and topic/structure filtering.
    for idx, doc in enumerate(final_chunks, start=1):
        meta = getattr(doc, "metadata", {}) or {}
        source = str(meta.get("source", "")).strip()
        page = str(meta.get("page", "")).strip()
        h1 = str(meta.get("H1", "")).strip()
        h2 = str(meta.get("H2", "")).strip()
        h3 = str(meta.get("H3", "")).strip()

        # Topic path/tags: keep heading lineage for topic-aware retrieval.
        topic_tags = [v for v in [h1, h2, h3] if v]
        normalized_tags = []
        for t in topic_tags:
            low = t.lower()
            if low.startswith("source:") or low.startswith("page"):
                continue
            normalized_tags.append(t)

        topic_path = " > ".join(normalized_tags)
        chapter = ""
        section = h3 or h2
        chapter_match = re.search(r"(หัวข้อที่\s*\d+[^>\n]*|บทที่\s*\d+[^>\n]*|chapter\s*\d+[^>\n]*)", topic_path, re.IGNORECASE)
        if chapter_match:
            chapter = chapter_match.group(1).strip()

        page_content = doc.page_content or ""
        has_structure = contains_structure_markers(page_content)
        structure_types = extract_structure_types(page_content)

        doc.metadata["source"] = source or "reference_document"
        doc.metadata["page"] = page
        doc.metadata["chunk_id"] = f"chunk-{idx:05d}"
        doc.metadata["chapter"] = chapter
        doc.metadata["section"] = section
        doc.metadata["topic_path"] = topic_path
        doc.metadata["topic_tags"] = normalized_tags
        doc.metadata["has_structure"] = bool(has_structure)
        doc.metadata["structure_types"] = structure_types

    contextual_applied = apply_contextual_chunking(final_chunks)
    if contextual_applied:
        print(f"   Applied contextual chunking: {contextual_applied}/{len(final_chunks)} chunks")
    
    print(f"   เนเธ”เนเธ—เธฑเนเธเธซเธกเธ” {len(final_chunks)} Chunks")

    print("   Building FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(final_chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR / "faiss_index"))

    print("   Building BM25...")
    tokenized = [word_tokenize(doc.page_content, engine="newmm") for doc in final_chunks]
    bm25 = BM25Okapi(tokenized)
    
    with open(INDEX_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": final_chunks}, f)

    manifest = build_index_manifest(INDEX_DIR)
    print(f"   Wrote index manifest with {len(manifest.get('files', {}))} files")

    print("\nIngestion Complete!")

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    pages = process_pdf_vision(DATA_DIR)
    if pages:
        save_debug_markdown(pages)
        write_quality_report(pages)
        build_indexes(pages)

if __name__ == "__main__":
    main()

