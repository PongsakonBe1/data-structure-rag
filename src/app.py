"""
app.py — ระบบแชทบอทช่วยสอนวิชาโครงสร้างข้อมูล (Interactive UI & Dropdown Edition)
โมเดล: Qwen3-4B (Responder) และ Llama-3.2-3B (Judge) ผ่าน Hugging Face
การแก้ไข: ปรับปรุงการแสดงผล Sidebar ให้ถูกต้อง, ใช้ Emoji แทนไอคอนที่พัง, และจัดการ Token
"""

import os
import sys
import time
import csv
import json
import re
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import httpx
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
try:
    import fitz  # PyMuPDF for extracting full page images
except Exception:
    fitz = None
try:
    from pythainlp.tokenize import word_tokenize as thai_word_tokenize
except Exception:
    thai_word_tokenize = None

# ตั้งค่า Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Ensure src is in path for imports
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from retriever import RAGSystem
except ImportError as e:
    st.error(f"❌ ไม่พบไฟล์ retriever.py: {e}")
    st.error(f"Project root: {PROJECT_ROOT}")
    st.error(f"SRC dir: {SRC_DIR}")
    st.error(f"sys.path: {sys.path[:3]}")
    st.stop()

try:
    from index_integrity import verify_index_manifest
except ImportError:
    verify_index_manifest = None

# ---------------------------------------------------------------------------
# Config & Setup
# ---------------------------------------------------------------------------
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(DOTENV_PATH)

# อ่าน HF Token จาก Streamlit Secrets ก่อน แล้ว fallback ไป .env (สำหรับ local dev)
try:
    HF_TOKEN = (
        st.secrets.get("huggingface", {}).get("token") 
        or st.secrets.get("huggingface", {}).get("api_key")
        or st.secrets.get("HUGGINGFACE_READ_TOKEN")
        or st.secrets.get("HUGGINGFACE_API_KEY")
        or os.getenv("HUGGINGFACE_READ_TOKEN")
        or os.getenv("HUGGINGFACE_API_KEY")
        or ""
    ).strip()
except Exception:
    # Fallback to .env only when secrets.toml is not present (local dev)
    HF_TOKEN = (
        os.getenv("HUGGINGFACE_READ_TOKEN")
        or os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("HF_TOKEN")
        or ""
    ).strip()

if not HF_TOKEN:
    st.error("❌ ไม่พบ HUGGINGFACE_READ_TOKEN หรือ HUGGINGFACE_API_KEY")
    st.error("กรุณาเพิ่ม Secrets ใน Streamlit Cloud (Settings → Secrets) หรือสร้างไฟล์ .env สำหรับ local")
    st.stop()

CHAT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507" 
JUDGE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "1400"))
TEXT_SECTION_MAX_TOKENS = int(os.getenv("TEXT_SECTION_MAX_TOKENS", "1200"))
SECTION_CONCISE_TRIGGER_CHARS = int(os.getenv("SECTION_CONCISE_TRIGGER_CHARS", "900"))
SECTION_CONCISE_MAX_TOKENS = int(os.getenv("SECTION_CONCISE_MAX_TOKENS", "320"))
OPERATION_SECTION_MAX_TOKENS = int(os.getenv("OPERATION_SECTION_MAX_TOKENS", "1800"))
OPERATION_SECTION_ENABLE_COMPACTION = os.getenv("OPERATION_SECTION_ENABLE_COMPACTION", "0").strip().lower() in {"1", "true", "yes", "on"}
CHAT_AUTO_CONTINUE = os.getenv("CHAT_AUTO_CONTINUE", "1").strip().lower() in {"1", "true", "yes", "on"}
CHAT_CONTINUE_MAX_TOKENS = int(os.getenv("CHAT_CONTINUE_MAX_TOKENS", "220"))
SECTION_AUTO_CONTINUE = os.getenv("SECTION_AUTO_CONTINUE", "1").strip().lower() in {"1", "true", "yes", "on"}
SECTION_CONTINUE_MAX_TOKENS = int(os.getenv("SECTION_CONTINUE_MAX_TOKENS", "420"))
OPERATION_HEAVY_SECTION_IDS = {"3.3.2", "3.3.3", "2.4.2"}
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "20"))
RETRIEVE_TOP_N = int(os.getenv("RETRIEVE_TOP_N", "8"))
CONTEXT_DOC_LIMIT = int(os.getenv("CONTEXT_DOC_LIMIT", "12"))
# Relaxed evidence gate to reduce false abstains
RETRIEVE_STRICT_TOPIC = os.getenv("RETRIEVE_STRICT_TOPIC", "0").strip().lower() in {"1", "true", "yes", "on"}
RETRIEVE_STRICT_STRUCTURE = os.getenv("RETRIEVE_STRICT_STRUCTURE", "0").strip().lower() in {"1", "true", "yes", "on"}
RETRIEVE_ALLOW_OFFTOPIC_DOCS = 0
RETRIEVE_MIN_SECTION_PAGE_DIVERSITY = int(os.getenv("RETRIEVE_MIN_SECTION_PAGE_DIVERSITY", "2"))
RETRIEVE_STRICT_SECTION_ONLY = os.getenv("RETRIEVE_STRICT_SECTION_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
EVIDENCE_MIN_DOCS = int(os.getenv("EVIDENCE_MIN_DOCS", "1"))
EVIDENCE_MIN_TOPIC_MATCH_RATIO = float(os.getenv("EVIDENCE_MIN_TOPIC_MATCH_RATIO", "0.25"))
EVIDENCE_MIN_AVG_RRF = float(os.getenv("EVIDENCE_MIN_AVG_RRF", "0.001"))
# FAST_RETRIEVAL_MODE = os.getenv("FAST_RETRIEVAL_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
# Full mode: ColPali active for structure/operation queries
FAST_RETRIEVAL_MODE = False
STRICT_CITATION_ENFORCEMENT = os.getenv(
    "STRICT_CITATION_ENFORCEMENT",
    "0",  # Disabled by default for better UX
).strip().lower() in {"1", "true", "yes", "on"}
MIN_CITED_CLAIM_RATIO = float(os.getenv("MIN_CITED_CLAIM_RATIO", "0.40"))
VISUAL_CITATION_MIN_CLAIM_RATIO = max(0.0, min(1.0, float(os.getenv("VISUAL_CITATION_MIN_CLAIM_RATIO", "0.40"))))
CITATION_REPAIR_ENABLED = os.getenv(
    "CITATION_REPAIR_ENABLED",
    "0" if FAST_RETRIEVAL_MODE else "1",
).strip().lower() in {"1", "true", "yes", "on"}
CITATION_REPAIR_MAX_TOKENS = int(os.getenv("CITATION_REPAIR_MAX_TOKENS", "260"))
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
QUERY_REWRITE_USE_LLM = os.getenv("QUERY_REWRITE_USE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}
QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", CHAT_MODEL_ID).strip()
QUERY_REWRITE_MAX_TOKENS = max(64, int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "96")))
QUERY_REWRITE_MAX_KEYWORDS = max(2, int(os.getenv("QUERY_REWRITE_MAX_KEYWORDS", "6")))
SELF_CHECK_ENABLED = os.getenv("SELF_CHECK_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
SELF_CHECK_MAX_TOKENS = max(180, int(os.getenv("SELF_CHECK_MAX_TOKENS", "420")))
SELF_CHECK_MIN_ANSWER_CHARS = max(40, int(os.getenv("SELF_CHECK_MIN_ANSWER_CHARS", "90")))
SELF_CHECK_CONTEXT_MAX_CHARS = max(800, int(os.getenv("SELF_CHECK_CONTEXT_MAX_CHARS", "2600")))
VISUAL_RETRIEVAL_ENABLED = os.getenv("VISUAL_RETRIEVAL_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_RETRIEVAL_DEFAULT_MODE = os.getenv("VISUAL_RETRIEVAL_DEFAULT_MODE", "text").strip().lower()
VISUAL_RETRIEVAL_BACKEND = os.getenv("VISUAL_RETRIEVAL_BACKEND", "auto").strip().lower()
VISUAL_RETRIEVAL_TOP_K = int(os.getenv("VISUAL_RETRIEVAL_TOP_K", "6"))
VISUAL_RETRIEVAL_CANDIDATE_K = int(os.getenv("VISUAL_RETRIEVAL_CANDIDATE_K", "16"))
VISUAL_RETRIEVAL_TOPIC_THRESHOLD = float(os.getenv("VISUAL_RETRIEVAL_TOPIC_THRESHOLD", "0.18"))
VISUAL_RETRIEVAL_USE_VLM_RERANK = os.getenv("VISUAL_RETRIEVAL_USE_VLM_RERANK", "0").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_RETRIEVAL_USE_GROUNDING = os.getenv("VISUAL_RETRIEVAL_USE_GROUNDING", "0").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_RETRIEVAL_USE_SPLADE = os.getenv("VISUAL_RETRIEVAL_USE_SPLADE", "0").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_SPARSE_STRATEGY = "bm25"
VISUAL_RETRIEVAL_SPLADE_MODE = os.getenv("VISUAL_RETRIEVAL_SPLADE_MODE", "auto").strip().lower()
VISUAL_RETRIEVAL_SPLADE_PROVIDER = os.getenv("VISUAL_RETRIEVAL_SPLADE_PROVIDER", "hf-inference").strip()
VISUAL_RETRIEVAL_SPLADE_MODEL = os.getenv("VISUAL_RETRIEVAL_SPLADE_MODEL", "naver/splade-v3").strip()
VISUAL_METADATA_FILTER_STRICT = True
VISUAL_INTENT_USE_LLM = os.getenv("VISUAL_INTENT_USE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_INTENT_CANDIDATE_TOP = max(3, int(os.getenv("VISUAL_INTENT_CANDIDATE_TOP", "12")))
VISUAL_INTENT_MIN_CONFIDENCE = max(0.0, min(1.0, float(os.getenv("VISUAL_INTENT_MIN_CONFIDENCE", "0.55"))))
VISUAL_INTENT_MAX_TOPIC_IDS = max(1, int(os.getenv("VISUAL_INTENT_MAX_TOPIC_IDS", "4")))
VISUAL_QUERY_EXPANSION_USE_LLM = os.getenv("VISUAL_QUERY_EXPANSION_USE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_QUERY_EXPANSION_MODEL = os.getenv("VISUAL_QUERY_EXPANSION_MODEL", "Qwen/Qwen3-4B-Instruct-2507").strip()
VISUAL_QUERY_EXPANSION_MAX_TERMS = max(0, int(os.getenv("VISUAL_QUERY_EXPANSION_MAX_TERMS", "8")))
VISUAL_PREFILTER_MIN_RECORDS = max(6, int(os.getenv("VISUAL_PREFILTER_MIN_RECORDS", "8")))
VISUAL_PREFILTER_RESCUE_TOPN = max(VISUAL_PREFILTER_MIN_RECORDS, int(os.getenv("VISUAL_PREFILTER_RESCUE_TOPN", "24")))
VISUAL_RETRIEVAL_SPLADE_DEVICE = os.getenv("VISUAL_RETRIEVAL_SPLADE_DEVICE", "auto").strip()
VISUAL_RETRIEVAL_SPLADE_MAX_LENGTH = max(32, int(os.getenv("VISUAL_RETRIEVAL_SPLADE_MAX_LENGTH", "256")))
VISUAL_RETRIEVAL_SPLADE_BATCH_SIZE = max(1, int(os.getenv("VISUAL_RETRIEVAL_SPLADE_BATCH_SIZE", "8")))
VISUAL_VLM_RERANK_TOP_M = max(4, int(os.getenv("VISUAL_VLM_RERANK_TOP_M", "12")))
VISUAL_GROUND_TOP_N = max(1, int(os.getenv("VISUAL_GROUND_TOP_N", "2")))
VISUAL_GROUNDING_ENSEMBLE_RUNS = max(1, int(os.getenv("VISUAL_GROUNDING_ENSEMBLE_RUNS", "2")))
VISUAL_GROUNDING_CONSENSUS_MIN_VOTES = max(1, int(os.getenv("VISUAL_GROUNDING_CONSENSUS_MIN_VOTES", "2")))
VISUAL_GROUNDING_TEMPERATURE = float(os.getenv("VISUAL_GROUNDING_TEMPERATURE", "0.0"))
VISUAL_GROUNDING_TOP_P = float(os.getenv("VISUAL_GROUNDING_TOP_P", "0.9"))
# อ่าน ColPali Endpoint จาก Streamlit Secrets ก่อน แล้ว fallback ไป .env หรือใช้ default
_DEFAULT_COLPALI_ENDPOINT = "https://macza5546-copali-endpoint.hf.space/"
try:
    VISUAL_RETRIEVAL_ENDPOINT_URL = (
        st.secrets.get("colpali", {}).get("endpoint_url")
        or st.secrets.get("COLPALI_ENDPOINT_URL")
        or os.getenv("COLPALI_ENDPOINT_URL", "")
        or _DEFAULT_COLPALI_ENDPOINT
    ).strip()
except Exception:
    VISUAL_RETRIEVAL_ENDPOINT_URL = (
        os.getenv("COLPALI_ENDPOINT_URL", "")
        or _DEFAULT_COLPALI_ENDPOINT
    ).strip()
VISUAL_RETRIEVAL_ENDPOINT_TIMEOUT_SEC = int(os.getenv("VISUAL_RETRIEVAL_ENDPOINT_TIMEOUT_SEC", "30"))
VISUAL_ENDPOINT_AUTOWARM_ON_STARTUP = os.getenv("VISUAL_ENDPOINT_AUTOWARM_ON_STARTUP", "1").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_ENDPOINT_AUTOWARM_TIMEOUT_SEC = max(5, int(os.getenv("VISUAL_ENDPOINT_AUTOWARM_TIMEOUT_SEC", "20")))
VISUAL_ENDPOINT_MAX_RETRIES = max(0, int(os.getenv("VISUAL_ENDPOINT_MAX_RETRIES", "2")))
VISUAL_ENDPOINT_RETRY_BACKOFF_MS = max(0, int(os.getenv("VISUAL_ENDPOINT_RETRY_BACKOFF_MS", "700")))
VISUAL_ENDPOINT_LOCAL_CACHE_SIZE = max(16, int(os.getenv("VISUAL_ENDPOINT_LOCAL_CACHE_SIZE", "256")))
VISUAL_ENDPOINT_IMAGE_MAX_SIDE = max(256, int(os.getenv("VISUAL_ENDPOINT_IMAGE_MAX_SIDE", "896")))
VISUAL_ENDPOINT_JPEG_QUALITY = max(40, min(95, int(os.getenv("VISUAL_ENDPOINT_JPEG_QUALITY", "80"))))
VISUAL_RETRIEVAL_ALLOW_CPU_ENGINE = os.getenv("VISUAL_RETRIEVAL_ALLOW_CPU_ENGINE", "0").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_RETRIEVAL_VLM_MODEL = os.getenv("VISUAL_RETRIEVAL_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct").strip()
VISUAL_HARD_NEGATIVE_RULES = os.getenv(
    "VISUAL_HARD_NEGATIVE_RULES",
    str((PROJECT_ROOT / "indexes" / "hierarchical" / "hard_negative_rules.json").as_posix()),
).strip()
VISUAL_HARD_NEGATIVE_PENALTY_MAX = float(os.getenv("VISUAL_HARD_NEGATIVE_PENALTY_MAX", "0.22"))
FORCED_VISUAL_BACKEND = os.getenv("FORCED_VISUAL_BACKEND", "metadata")
VISUAL_FORCE_ON_REQUIRE_STRUCTURE = os.getenv("VISUAL_FORCE_ON_REQUIRE_STRUCTURE", "0").strip().lower() in {"1", "true", "yes", "on"}
TEXT_MODE_PROMOTE_VISUAL_FOR_STRUCTURE = os.getenv(
    "TEXT_MODE_PROMOTE_VISUAL_FOR_STRUCTURE",
    "0" if FAST_RETRIEVAL_MODE else "1",
).strip().lower() in {"1", "true", "yes", "on"}
VISUAL_REGION_PAD_RATIO = max(0.0, min(0.5, float(os.getenv("VISUAL_REGION_PAD_RATIO", "0.30"))))
VISUAL_CONTEXT_DOC_LIMIT_OPERATION = int(os.getenv("VISUAL_CONTEXT_DOC_LIMIT_OPERATION", "8"))
VISUAL_STEP_STITCHING_ENABLED = os.getenv("VISUAL_STEP_STITCHING_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_STEP_STITCHING_MAX_PAGE_GAP = max(1, int(os.getenv("VISUAL_STEP_STITCHING_MAX_PAGE_GAP", "2")))
VISUAL_STEP_STITCHING_MIN_EXTRA_HITS = max(0, int(os.getenv("VISUAL_STEP_STITCHING_MIN_EXTRA_HITS", "2")))
VISUAL_LINKEDLIST_OPERATION_TOPIC_PREFIX = os.getenv("VISUAL_LINKEDLIST_OPERATION_TOPIC_PREFIX", "2.4.2").strip()
VISUAL_TOPIC_ADJACENT_PAGE_GAP = max(0, int(os.getenv("VISUAL_TOPIC_ADJACENT_PAGE_GAP", "3")))
VISUAL_TOPIC_ADJACENT_MAX_EXTRA_PAGES = max(0, int(os.getenv("VISUAL_TOPIC_ADJACENT_MAX_EXTRA_PAGES", "8")))
VISUAL_DYNAMIC_TOKEN_ENABLED = os.getenv("VISUAL_DYNAMIC_TOKEN_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_DYNAMIC_TOKEN_BASE = int(os.getenv("VISUAL_DYNAMIC_TOKEN_BASE", "800"))
VISUAL_DYNAMIC_TOKEN_PER_SOURCE = int(os.getenv("VISUAL_DYNAMIC_TOKEN_PER_SOURCE", "110"))
VISUAL_DYNAMIC_TOKEN_PER_FIGURE = int(os.getenv("VISUAL_DYNAMIC_TOKEN_PER_FIGURE", "80"))
VISUAL_DYNAMIC_TOKEN_OPERATION_BONUS = int(os.getenv("VISUAL_DYNAMIC_TOKEN_OPERATION_BONUS", "180"))
VISUAL_DYNAMIC_TOKEN_CAP = int(os.getenv("VISUAL_DYNAMIC_TOKEN_CAP", "2400"))
VISUAL_CORRECTIVE_RETRIEVAL_ENABLED = os.getenv("VISUAL_CORRECTIVE_RETRIEVAL_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_CORRECTIVE_MAX_RETRIES = max(0, int(os.getenv("VISUAL_CORRECTIVE_MAX_RETRIES", "2")))
VISUAL_CORRECTIVE_TOPK_MULTIPLIER = max(1.0, float(os.getenv("VISUAL_CORRECTIVE_TOPK_MULTIPLIER", "2.0")))
VISUAL_CORRECTIVE_CANDIDATE_MULTIPLIER = max(1.0, float(os.getenv("VISUAL_CORRECTIVE_CANDIDATE_MULTIPLIER", "2.2")))
VISUAL_CORRECTIVE_TOPIC_THRESHOLD_RELAX = float(os.getenv("VISUAL_CORRECTIVE_TOPIC_THRESHOLD_RELAX", "0.12"))
VISUAL_COVERAGE_RERANK_ENABLED = os.getenv("VISUAL_COVERAGE_RERANK_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_JUDGE_EVALUATION = os.getenv("ENABLE_JUDGE_EVALUATION", "0").strip().lower() in {"1", "true", "yes", "on"}

VISUAL_ADAPTIVE_EXPAND_ENABLED = os.getenv("VISUAL_ADAPTIVE_EXPAND_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_ADAPTIVE_PAD_MIN = max(0.0, min(0.5, float(os.getenv("VISUAL_ADAPTIVE_PAD_MIN", "0.26"))))
VISUAL_ADAPTIVE_PAD_MAX = max(VISUAL_ADAPTIVE_PAD_MIN, min(0.6, float(os.getenv("VISUAL_ADAPTIVE_PAD_MAX", "0.50"))))
VISUAL_ADAPTIVE_LOW_QUALITY = max(0.0, min(1.0, float(os.getenv("VISUAL_ADAPTIVE_LOW_QUALITY", "0.80"))))
VISUAL_ADAPTIVE_MID_QUALITY = max(0.0, min(1.0, float(os.getenv("VISUAL_ADAPTIVE_MID_QUALITY", "0.90"))))
VISUAL_GROUNDING_GATE_ENABLED = os.getenv("VISUAL_GROUNDING_GATE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_GROUNDING_GATE_OPERATION_ONLY = os.getenv("VISUAL_GROUNDING_GATE_OPERATION_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_GROUNDING_MIN_DOCS = max(1, int(os.getenv("VISUAL_GROUNDING_MIN_DOCS", "1")))
VISUAL_GROUNDING_MIN_FACTS = max(0, int(os.getenv("VISUAL_GROUNDING_MIN_FACTS", "0")))
VISUAL_GROUNDING_MIN_OPERATION_FACTS = max(1, int(os.getenv("VISUAL_GROUNDING_MIN_OPERATION_FACTS", "2")))
VISUAL_GROUNDING_MIN_STEP_COVERAGE = max(0.0, min(1.0, float(os.getenv("VISUAL_GROUNDING_MIN_STEP_COVERAGE", "0.10"))))
VISUAL_GROUNDING_RELAX_FOR_IMAGE_HEAVY = os.getenv("VISUAL_GROUNDING_RELAX_FOR_IMAGE_HEAVY", "1").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_GROUNDING_RELAX_MIN_UNIQUE_FIGURES = max(1, int(os.getenv("VISUAL_GROUNDING_RELAX_MIN_UNIQUE_FIGURES", "2")))
VISUAL_GROUNDING_RELAX_FACT_REDUCTION = max(0, int(os.getenv("VISUAL_GROUNDING_RELAX_FACT_REDUCTION", "1")))
VISUAL_CHAPTER_CALIBRATION_ENABLED = os.getenv("VISUAL_CHAPTER_CALIBRATION_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
VISUAL_CHAPTER_CALIBRATION_FILE = Path(
    os.getenv(
        "VISUAL_CHAPTER_CALIBRATION_FILE",
        str((PROJECT_ROOT / "indexes" / "hierarchical" / "chapter_calibration.json").as_posix()),
    )
)

MIN_CITED_CLAIM_RATIO = max(0.0, min(1.0, MIN_CITED_CLAIM_RATIO))

hf_client = InferenceClient(api_key=HF_TOKEN)

# Logging Paths
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
RESEARCH_LOG_FILE = LOG_DIR / "research_log_v4.csv"
EVENT_LOG_FILE = LOG_DIR / "runtime_events.jsonl"
FIGURE_MANIFEST_FILE = LOG_DIR / "figure_manifest_latest.json"
TOPIC_FILE = PROJECT_ROOT / "list_list_hierarchy.txt"
VISUAL_RETRIEVAL_SCRIPT = PROJECT_ROOT / "scripts" / "retrieve_visual_hybrid.py"
VISUAL_RETRIEVAL_RUNTIME_OUTPUT = LOG_DIR / "visual_retrieval_runtime_latest.json"
VISUAL_ENDPOINT_HEALTHCHECK_LOG = LOG_DIR / "visual_endpoint_healthcheck_latest.json"
VISUAL_STANDARD_QUERY_LOG = LOG_DIR / "visual_standard_query_test_latest.json"
FIGURE_REGION_MANIFEST_FILE = LOG_DIR / "figure_regions_manifest_latest.json"
EXPANDED_REGION_DIR = LOG_DIR / "expanded_regions"
EXPANDED_REGION_DIR.mkdir(exist_ok=True)
IOC_EVAL_FILE = LOG_DIR / "expert_ioc_eval.csv"
VISUAL_STANDARD_QUERY_CASES = [
    {"query": "โครงสร้างลิงค์ลิสต์แบบทิศทางเดียว", "require_structure": True},
    {"query": "โครงสร้างของการแทนคิวด้วยอาร์เรย์", "require_structure": True},
    {"query": "การดำเนินการแทนคิวด้วยวงกลม", "require_structure": True},
]
# Disabled by default: page-level evidence images were too coarse for the target UX.
INLINE_EVIDENCE_IMAGE_MAX = int(os.getenv("INLINE_EVIDENCE_IMAGE_MAX", "4"))
IMAGE_ASSISTED_FALLBACK = os.getenv("IMAGE_ASSISTED_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
RESEARCH_FIELDNAMES = [
    "Timestamp",
    "Question",
    "Predicted_Topic",
    "Answer_Length",
    "Faithfulness",
    "Relevance",
    "Context_Precision",
    "Index_Dir",
    "Index_Integrity_OK",
    "Index_Integrity_Error",
    "Dense_Ready",
    "Dense_Error",
    "Reranker_Ready",
    "Reranker_Error",
]

APP_LOGGER = logging.getLogger("typhoon_rag.app")
if not APP_LOGGER.handlers:
    APP_LOGGER.setLevel(logging.INFO)
    APP_LOGGER.addHandler(logging.StreamHandler())


def log_event(event: str, **fields):
    """Write structured runtime events as JSONL."""
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        **fields,
    }
    try:
        with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        APP_LOGGER.warning("Failed to write event log: %s", exc)


def ensure_research_log():
    if not RESEARCH_LOG_FILE.exists():
        with open(RESEARCH_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESEARCH_FIELDNAMES)
            writer.writeheader()
        return

    with open(RESEARCH_LOG_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames or []
        if old_fieldnames == RESEARCH_FIELDNAMES:
            return
        old_rows = list(reader)

    with open(RESEARCH_LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESEARCH_FIELDNAMES)
        writer.writeheader()
        for row in old_rows:
            writer.writerow({k: row.get(k, "") for k in RESEARCH_FIELDNAMES})


def append_research_log(question: str, topic: str, answer: str, judge_result: dict, rag_status: dict):
    ensure_research_log()
    scores = judge_result.get("scores", {}) if isinstance(judge_result, dict) else {}
    with open(RESEARCH_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESEARCH_FIELDNAMES)
        writer.writerow(
            {
                "Timestamp": datetime.utcnow().isoformat(),
                "Question": question,
                "Predicted_Topic": topic,
                "Answer_Length": len(answer or ""),
                "Faithfulness": scores.get("faithfulness", ""),
                "Relevance": scores.get("relevance", ""),
                "Context_Precision": scores.get("context_precision", ""),
                "Index_Dir": rag_status.get("index_dir", "") if rag_status else "",
                "Index_Integrity_OK": rag_status.get("index_integrity_ok", "") if rag_status else "",
                "Index_Integrity_Error": rag_status.get("index_integrity_error", "") if rag_status else "",
                "Dense_Ready": rag_status.get("dense_ready", "") if rag_status else "",
                "Dense_Error": rag_status.get("dense_error", "") if rag_status else "",
                "Reranker_Ready": rag_status.get("reranker_ready", "") if rag_status else "",
                "Reranker_Error": rag_status.get("reranker_error", "") if rag_status else "",
            }
        )


@st.cache_data(show_spinner=False)
def load_figure_index(manifest_path: str) -> dict:
    """
    Build {(source_filename, page): [image_path,...]} for showing document evidence images in chat.
    """
    path = Path(manifest_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    source_name = Path(str(payload.get("pdf", ""))).name
    images = payload.get("images", []) if isinstance(payload, dict) else []
    index: dict[tuple[str, str], list[str]] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        page = str(item.get("page", "")).strip()
        img_path = str(item.get("path", "")).strip()
        if not page or not img_path:
            continue
        key = (source_name, page)
        index.setdefault(key, []).append(img_path)
    return index


def _norm_path(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lower()


@st.cache_data(show_spinner=False)
def load_region_manifest_index(manifest_path: str) -> dict:
    """
    Build {region_path -> {page_image_path,bbox,source,page}} for padded rendering.
    """
    path = Path(manifest_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    out = {}
    for p in pages:
        if not isinstance(p, dict):
            continue
        page_img = str(p.get("image_path", "")).strip()
        source = str(p.get("source", "")).strip()
        page = str(p.get("page", "")).strip()
        for r in p.get("regions", []) or []:
            if not isinstance(r, dict):
                continue
            region_path = str(r.get("path", "")).strip()
            bbox = r.get("bbox", {}) if isinstance(r.get("bbox", {}), dict) else {}
            key = _norm_path(region_path)
            if not key or key in out:
                continue
            out[key] = {
                "page_image_path": page_img,
                "bbox": bbox,
                "source": source,
                "page": page,
            }
    return out


@st.cache_data(show_spinner=False)
def load_chapter_calibration(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"topics": {}}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"topics": {}}
    if not isinstance(payload, dict):
        return {"topics": {}}
    topics = payload.get("topics", {})
    if not isinstance(topics, dict):
        topics = {}
    payload["topics"] = topics
    return payload


def resolve_chapter_calibration(topic_id: str) -> dict:
    if not VISUAL_CHAPTER_CALIBRATION_ENABLED:
        return {}
    tid = str(topic_id or "").strip()
    if not tid:
        return {}
    payload = load_chapter_calibration(str(VISUAL_CHAPTER_CALIBRATION_FILE))
    topics = payload.get("topics", {}) if isinstance(payload, dict) else {}
    if not isinstance(topics, dict) or not topics:
        return {}

    best_key = ""
    best_val = {}
    for key, val in topics.items():
        k = str(key or "").strip()
        if not k or not isinstance(val, dict):
            continue
        if tid == k or tid.startswith(k + "."):
            if len(k) > len(best_key):
                best_key = k
                best_val = val
    if not best_key:
        return {}
    out = dict(best_val)
    out["matched_topic_id"] = best_key
    out["resolved_topic_id"] = tid
    return out


def resolve_expanded_region_path(region_path: str, pad_ratio: float = VISUAL_REGION_PAD_RATIO) -> str:
    """
    Expand a pre-cropped region using page bbox to keep labels/arrows (front/rear) in view.
    """
    rp = str(region_path or "").strip()
    if not rp:
        return rp
    key = _norm_path(rp)
    meta = region_manifest_index.get(key) or {}
    page_img = str(meta.get("page_image_path", "")).strip()
    bbox = meta.get("bbox", {}) if isinstance(meta.get("bbox", {}), dict) else {}
    if not page_img or not Path(page_img).exists():
        return rp
    try:
        x0 = int(bbox.get("x0", 0))
        y0 = int(bbox.get("y0", 0))
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        if x1 <= x0 or y1 <= y0:
            return rp
    except Exception:
        return rp

    out_name = f"{Path(rp).stem}_pad{int(pad_ratio * 100)}.png"
    out_path = EXPANDED_REGION_DIR / out_name
    if out_path.exists():
        return str(out_path)

    try:
        with Image.open(page_img) as im:
            w, h = im.size
            pad_x = int((x1 - x0) * pad_ratio)
            pad_y = int((y1 - y0) * pad_ratio)
            cx0 = max(0, x0 - pad_x)
            cy0 = max(0, y0 - pad_y)
            cx1 = min(w, x1 + pad_x)
            cy1 = min(h, y1 + pad_y)
            crop = im.crop((cx0, cy0, cx1, cy1))
            crop.save(out_path)
            return str(out_path)
    except Exception:
        return rp


def _to_float_or_none(value):
    try:
        return float(value)
    except Exception:
        return None


def compute_adaptive_region_pad_ratio(source_row: dict) -> float:
    """
    Adaptive crop expansion by retrieval quality:
    - lower region_quality_score => larger padding
    - tiny area_ratio => extra padding to recover arrows/labels
    """
    base = float(VISUAL_REGION_PAD_RATIO)
    if not VISUAL_ADAPTIVE_EXPAND_ENABLED:
        return round(max(0.0, min(0.6, base)), 3)
    if not isinstance(source_row, dict):
        return round(max(VISUAL_ADAPTIVE_PAD_MIN, min(VISUAL_ADAPTIVE_PAD_MAX, base)), 3)

    pad = base
    quality = _to_float_or_none(source_row.get("region_quality_score"))
    region_meta = source_row.get("region_meta", {}) if isinstance(source_row.get("region_meta", {}), dict) else {}
    area_ratio = _to_float_or_none(region_meta.get("area_ratio"))

    if quality is not None:
        if quality < VISUAL_ADAPTIVE_LOW_QUALITY:
            pad = max(pad, 0.46)
        elif quality < VISUAL_ADAPTIVE_MID_QUALITY:
            pad = max(pad, 0.38)
        else:
            pad = max(pad, VISUAL_ADAPTIVE_PAD_MIN)

    if area_ratio is not None:
        if area_ratio < 0.008:
            pad = max(pad, 0.50)
        elif area_ratio < 0.015:
            pad = max(pad, 0.42)
        elif area_ratio < 0.03:
            pad = max(pad, 0.34)

    pad = max(VISUAL_ADAPTIVE_PAD_MIN, min(VISUAL_ADAPTIVE_PAD_MAX, pad))
    return round(float(pad), 3)


def sanitize_visual_preview(text: str) -> str:
    """
    Remove synthetic structure blocks and image tags from OCR/markdown preview.
    """
    if not text:
        return ""
    out = []
    for raw in text.splitlines():
        line = raw.rstrip()
        s = line.strip()
        if s.startswith("> [Structure:") or s.startswith("[Structure:"):
            continue
        if s.startswith("> - ") and ("connects to" in s.lower() or "ข้อมูล" in s):
            continue
        if re.search(r"!\[.*?\]\(.*?\)", s):
            continue
        if "<image" in s.lower():
            continue
        out.append(line)
    return "\n".join(out).strip()


def sanitize_source_excerpt(text: str) -> str:
    cleaned = sanitize_visual_preview(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:300]


def _text_tokens(text: str) -> set[str]:
    toks = set()
    for t in re.findall(r"[A-Za-z0-9ก-๙_]+", str(text or "").lower()):
        if len(t) >= 2:
            toks.add(t)
    return toks


def _normalize_for_match(text: str) -> str:
    s = str(text or "").lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^a-z0-9ก-๙]", "", s)
    return s


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    s = _normalize_for_match(text)
    if not s:
        return set()
    if len(s) <= n:
        return {s}
    return {s[i : i + n] for i in range(0, len(s) - n + 1)}


def filter_grounded_facts_by_preview(grounded_facts: list[str], preview_text: str) -> list[str]:
    """
    Keep grounding facts only when they overlap with OCR preview tokens,
    to reduce hallucinated visual grounding facts leaking into generation context.
    """
    if not grounded_facts:
        return []
    preview_tokens = _text_tokens(preview_text)
    preview_norm = _normalize_for_match(preview_text)
    preview_ngrams = _char_ngrams(preview_text, n=3)
    if not preview_tokens and not preview_ngrams:
        return []
    kept = []
    for fact in grounded_facts:
        f = str(fact).strip()
        if not f:
            continue
        fact_tokens = _text_tokens(f)
        token_overlap = len(fact_tokens.intersection(preview_tokens)) if (fact_tokens and preview_tokens) else 0
        token_overlap_ratio = (token_overlap / max(1, len(fact_tokens))) if fact_tokens else 0.0

        fact_norm = _normalize_for_match(f)
        substring_hit = bool(fact_norm and preview_norm and (fact_norm in preview_norm or preview_norm in fact_norm))
        fact_ngrams = _char_ngrams(f, n=3)
        ngram_overlap_ratio = (
            len(fact_ngrams.intersection(preview_ngrams)) / max(1, len(fact_ngrams))
            if (fact_ngrams and preview_ngrams) else 0.0
        )

        if substring_hit or token_overlap >= 1 or token_overlap_ratio >= 0.15 or ngram_overlap_ratio >= 0.18:
            kept.append(f)
    return kept


def _first_figure_ref_num(row: dict) -> float:
    refs = row.get("figure_refs", []) if isinstance(row, dict) else []
    nums = []
    for r in refs or []:
        for m in re.findall(r"(\d+(?:\.\d+)?)", str(r)):
            try:
                nums.append(float(m))
            except Exception:
                continue
    return min(nums) if nums else 9999.0


def stitch_cross_page_operation_hits(
    ordered_hits: list[dict],
    *,
    query_profile: dict,
    topic_id: str,
    doc_limit: int,
) -> list[dict]:
    """
    Cross-page step stitching for operation-heavy questions.
    Goal: increase process coverage across adjacent pages/figures while
    preserving high-confidence anchor hits at the top.
    """
    if not VISUAL_STEP_STITCHING_ENABLED:
        return ordered_hits
    if not isinstance(query_profile, dict) or not bool(query_profile.get("operation_intent")):
        return ordered_hits
    if not ordered_hits:
        return ordered_hits

    qn = str(query_profile.get("query_norm", "")).lower()
    tid = str(topic_id or "").strip()
    linkedlist_mode = bool(
        (VISUAL_LINKEDLIST_OPERATION_TOPIC_PREFIX and tid.startswith(VISUAL_LINKEDLIST_OPERATION_TOPIC_PREFIX))
        or ("ลิงค์ลิสต์" in qn)
        or ("linked list" in qn)
        or ("linked" in qn and "list" in qn)
    )

    def _id(h: dict) -> str:
        return str(h.get("id", "")).strip()

    def _page(h: dict) -> int:
        try:
            return int(h.get("page", 0) or 0)
        except Exception:
            return 0

    def _rank(h: dict) -> int:
        try:
            return int(h.get("rank", 9999) or 9999)
        except Exception:
            return 9999

    anchors = []
    for h in ordered_hits:
        if str(h.get("image_level", "")).strip().lower() != "region":
            continue
        if not bool(h.get("figure_refs", [])):
            continue
        anchors.append(h)
        if len(anchors) >= 3:
            break
    if not anchors:
        anchors = ordered_hits[:2]

    anchor_by_source: dict[str, list[int]] = {}
    for a in anchors:
        src = str(a.get("source", "")).strip().lower()
        pg = _page(a)
        if src and pg > 0:
            anchor_by_source.setdefault(src, []).append(pg)

    picked_ids = {_id(a) for a in anchors if _id(a)}
    extras = []
    for h in ordered_hits:
        hid = _id(h)
        if not hid or hid in picked_ids:
            continue
        src = str(h.get("source", "")).strip().lower()
        pg = _page(h)
        if not src or pg <= 0:
            continue
        anchor_pages = anchor_by_source.get(src, [])
        if not anchor_pages:
            continue
        if min(abs(pg - ap) for ap in anchor_pages) > int(VISUAL_STEP_STITCHING_MAX_PAGE_GAP):
            continue
        sem = float(h.get("operation_score", 0.0) or 0.0) + float(h.get("figure_score", 0.0) or 0.0)
        if linkedlist_mode:
            sem += float(h.get("structure_score", 0.0) or 0.0)
        if sem >= 0.20 or bool(h.get("figure_refs", [])):
            extras.append(h)
            picked_ids.add(hid)

    extras = sorted(
        extras,
        key=lambda h: (
            _page(h),
            _first_figure_ref_num(h),
            _rank(h),
            -float(h.get("operation_score", 0.0) or 0.0),
        ),
    )
    # Keep enough stitched evidence, but avoid flooding context.
    extras = extras[: max(0, int(VISUAL_STEP_STITCHING_MIN_EXTRA_HITS) * 2)]

    remainder = [h for h in ordered_hits if _id(h) not in picked_ids]
    stitched = anchors + extras + remainder

    if linkedlist_mode:
        # For linked-list operation chapters, keep nearby pages in natural order.
        stitched = sorted(
            stitched,
            key=lambda h: (
                0 if str(h.get("source", "")).strip().lower() in anchor_by_source else 1,
                _page(h),
                _first_figure_ref_num(h),
                _rank(h),
            ),
        )

    out = []
    seen = set()
    for h in stitched:
        hid = _id(h)
        if not hid or hid in seen:
            continue
        seen.add(hid)
        out.append(h)
        if len(out) >= max(int(doc_limit) * 2, int(doc_limit) + int(VISUAL_STEP_STITCHING_MIN_EXTRA_HITS)):
            break
    return out


def get_full_page_image_from_pdf(source: str, page: int, output_dir: Path = None, dpi: int = 150) -> str | None:
    """
    Extract a full page image from PDF and save to cache.
    Returns path to the generated image file.
    """
    if not fitz:
        return None
    
    pdf_path = PROJECT_ROOT / "data" / source
    if not pdf_path.exists():
        # Try to find the file in other locations
        pdf_path = PROJECT_ROOT / source
    if not pdf_path.exists():
        return None
    
    # Create cache directory
    if output_dir is None:
        output_dir = PROJECT_ROOT / "assets" / "page_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    safe_source = Path(source).stem
    output_path = output_dir / f"{safe_source}_page_{page:03d}.png"
    
    # Return cached image if exists
    if output_path.exists():
        return str(output_path)
    
    try:
        doc = fitz.open(pdf_path)
        if page < 1 or page > len(doc):
            doc.close()
            return None
        
        # Get page (0-indexed)
        page_obj = doc[page - 1]
        
        # Render page to image
        mat = fitz.Matrix(dpi/72, dpi/72)  # Scale by DPI
        pix = page_obj.get_pixmap(matrix=mat)
        
        # Save image
        pix.save(output_path)
        doc.close()
        
        return str(output_path)
    except Exception as e:
        logging.warning(f"Failed to extract page {page} from {source}: {e}")
        return None


def collect_evidence_images_from_sources(
    sources: list[dict],
    max_images: int,
    preferred_page_keys: list[str] | None = None,
) -> list[dict]:
    if not sources or not figure_index:
        # visual retrieval sources may already carry per-hit image_path.
        if not sources:
            return []
    picked = []
    seen = set()
    limit = max(0, int(max_images))
    if limit <= 0:
        return []

    def _int_or(value, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    # Keep retrieval ordering semantics while biasing image display to
    # the same page cluster as top-ranked evidence (prevents cross-topic second image).
    anchor = None
    for s in sources:
        src = str(s.get("source", "")).strip()
        page = _int_or(s.get("page", 0), 0)
        rank = _int_or(s.get("retrieval_rank", 9999), 9999)
        if src and page > 0:
            if anchor is None or rank < anchor[2]:
                anchor = (src, page, rank)

    def _cluster_bucket(s: dict) -> int:
        if anchor is None:
            return 3
        src = str(s.get("source", "")).strip()
        page = _int_or(s.get("page", 0), 0)
        if not src or page <= 0:
            return 3
        if src != anchor[0]:
            return 3
        dist = abs(page - anchor[1])
        if dist == 0:
            return 0
        if dist <= 1:
            return 1
        if dist <= 2:
            return 2
        return 3

    has_near_region = any(
        str(s.get("image_level", "")).strip().lower() == "region" and _cluster_bucket(s) <= 1
        for s in sources
    )

    def _effective_bucket(s: dict) -> int:
        b = _cluster_bucket(s)
        level = str(s.get("image_level", "")).strip().lower()
        # If near-page region evidence exists, deprioritize full-page anchor images.
        if has_near_region and level != "region" and b <= 1:
            return b + 2
        return b

    preferred_set = {str(x).strip() for x in (preferred_page_keys or []) if str(x).strip()}

    def _is_preferred_source(s: dict) -> int:
        if not preferred_set:
            return 1
        src = str(s.get("source", "")).strip()
        page = str(s.get("page", "")).strip()
        key = f"{src}:{page}" if src and page else ""
        return 0 if key and key in preferred_set else 1

    ordered_sources = sorted(
        sources,
        key=lambda s: (
            _is_preferred_source(s),
            _effective_bucket(s),
            0 if str(s.get("image_level", "")).strip().lower() == "region" else 1,
            0 if bool(s.get("figure_refs", [])) else 1,
            -float(s.get("query_align_score", 0.0) or 0.0),
            _int_or(s.get("retrieval_rank", 9999), 9999),
        ),
    )
    seen_pages = set()  # deduplicate by page number
    for s in ordered_sources:
        if len(picked) >= limit:
            break
        direct_img_path = str(s.get("image_path", "")).strip()
        src = str(s.get("source", "")).strip()
        page = str(s.get("page", "")).strip()
        
        # PRIORITY: Always try full page image first for better UX
        page_num = int(page) if page.isdigit() else 0
        if page_num > 0 and src:
            # Deduplicate by page number — same page = same image
            if page in seen_pages:
                continue
            full_page_path = get_full_page_image_from_pdf(src, page_num)
            if full_page_path and Path(full_page_path).exists():
                picked.append(
                    {
                        "path": full_page_path,
                        "citation": str(s.get("citation", "")).strip() or f"{src}:{page}",
                        "source": src,
                        "page": page,
                        "figure_text": "",
                        "image_type": "full_page"
                    }
                )
                seen_pages.add(page)
                seen.add((full_page_path, page))
                if len(picked) >= limit:
                    break
                continue
        
        # FALLBACK: Use provided image path if full page not available
        if direct_img_path and Path(direct_img_path).exists():
            key_direct = (direct_img_path, str(s.get("citation", "")).strip())
            if key_direct in seen:
                continue
            img_level = str(s.get("image_level", "")).strip().lower()
            render_path = direct_img_path
            adaptive_pad = None
            # Skip region expansion - use full page instead for better UX
            figure_refs = s.get("figure_refs", [])
            figure_text = ""
            if isinstance(figure_refs, list) and figure_refs:
                figure_text = ", ".join(str(x).strip() for x in figure_refs[:2] if str(x).strip())
            picked.append(
                {
                    "path": render_path,
                    "citation": str(s.get("citation", "")).strip() or str(s.get("source", "")),
                    "source": str(s.get("source", "")).strip(),
                    "page": str(s.get("page", "")).strip(),
                    "figure_text": figure_text,
                    "adaptive_pad_ratio": adaptive_pad,
                    "image_type": "region" if img_level == "region" else "direct"
                }
            )
            seen.add(key_direct)
            continue

        if not figure_index:
            continue
        src = str(s.get("source", "")).strip()
        page = str(s.get("page", "")).strip()
        if not src or not page:
            continue
        key = (src, page)
        if key in seen:
            continue
        
        # PRIORITY: Try to get full page image from PDF first
        page_num = int(page) if page.isdigit() else 0
        full_page_path = None
        if page_num > 0:
            full_page_path = get_full_page_image_from_pdf(src, page_num)
        
        if full_page_path and Path(full_page_path).exists():
            cite = s.get("citation", f"{src}:{page}")
            picked.append({
                "path": full_page_path,
                "citation": str(cite),
                "source": src,
                "page": page,
                "figure_text": "",
                "image_type": "full_page"
            })
            seen.add(key)
            continue
        
        # FALLBACK: Use cropped regions if full page not available
        paths = figure_index.get(key) or []
        if not paths:
            continue
        img_path = paths[0]
        if not Path(img_path).exists():
            continue
        cite = s.get("citation", f"{src}:{page}")
        picked.append({"path": img_path, "citation": str(cite), "source": src, "page": page, "figure_text": "", "image_type": "region"})
        seen.add(key)
    return picked


def extract_cited_source_page_keys(answer: str, sources: list[dict]) -> list[str]:
    """
    Parse inline citations in the final answer and return preferred source:page keys
    for evidence-image rendering.
    """
    text = str(answer or "")
    if not text:
        return []

    chunk_to_page = {}
    for s in sources or []:
        src = str(s.get("source", "")).strip()
        page = str(s.get("page", "")).strip()
        chunk = str(s.get("chunk_id", "")).strip().lower()
        if src and page and chunk:
            chunk_to_page[chunk] = f"{src}:{page}"

    page_keys: list[str] = []
    seen = set()

    for m in _INLINE_CITATION_RE.finditer(text):
        raw = m.group(1)
        parts = [p.strip() for p in re.split(r"[;,]", raw) if p.strip()]
        if not parts:
            parts = [raw.strip()]
        for p in parts:
            p0 = str(p).strip()
            if not p0:
                continue
            # direct source:page or source:page|chunk
            if _looks_like_source_page_citation(p0):
                base = p0.split("|", 1)[0].strip()
                if base and base not in seen:
                    seen.add(base)
                    page_keys.append(base)
                continue
            # chunk-only citation -> map via sources
            cm = _CHUNK_ID_RE.search(p0)
            if cm:
                c = cm.group(0).lower().strip()
                mapped = chunk_to_page.get(c)
                if mapped and mapped not in seen:
                    seen.add(mapped)
                    page_keys.append(mapped)
    return page_keys


def has_image_evidence_for_docs(docs: list) -> bool:
    if not docs or not figure_index:
        return False
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        src = str(meta.get("source", "")).strip()
        page = str(meta.get("page", "")).strip()
        if not src or not page:
            continue
        key = (src, page)
        if key in figure_index and any(Path(p).exists() for p in (figure_index.get(key) or [])):
            return True
    return False


def render_evidence_images(images: list[dict]) -> None:
    if not images:
        return
    st.caption("📄 ภาพหลักฐานจากเอกสาร (หน้าเต็ม)")
    for img in images:
        citation_raw = str(img.get("citation", "")).strip()
        page = str(img.get("page", "")).strip()
        image_type = str(img.get("image_type", "")).strip()
        
        # Build caption
        if citation_raw:
            caption = _humanize_citation_token(citation_raw)
        elif page:
            caption = f"หน้า {page}"
        else:
            caption = ""
        
        # Add indicator for full page images
        if image_type == "full_page":
            caption = f"📄 {caption} (ภาพหน้าเต็ม)"
        
        figure_text = str(img.get("figure_text", "")).strip()
        if figure_text:
            caption = f"{caption} | {figure_text}"
        st.image(
            img["path"],
            caption=caption,
            use_container_width=True,
        )


figure_index: dict[tuple[str, str], list[str]] = {}
region_manifest_index: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# UI Enhancements (Custom CSS & Mermaid)
# ---------------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
        /* โหลด Material Symbols / Icons font ให้ Streamlit ใช้แสดงไอคอนได้ถูกต้อง */
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
        @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');

        html, body, [class*="st-"] { font-family: 'Sarabun', sans-serif; }

        /* ========== Material Icon text fix ========== */
        /* ซ่อนข้อความ icon ทุกตัวใน Streamlit ด้วย visibility + overflow */
        span[data-testid="stIconMaterial"] {
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
            overflow: hidden !important;
            display: inline-block !important;
            width: 24px !important;
            height: 24px !important;
            font-size: 24px !important;
            line-height: 24px !important;
            color: transparent !important;
            -webkit-text-fill-color: transparent !important;
        }
        /* แสดง icon จริงผ่าน ::after */
        span[data-testid="stIconMaterial"]::after {
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
            -webkit-text-fill-color: currentColor !important;
            color: inherit !important;
            font-size: 24px !important;
            visibility: visible !important;
        }
        /* Sidebar collapse: keyboard_double_arrow_right -> ลูกศร */
        [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
            color: transparent !important;
        }
        [data-testid="collapsedControl"] span[data-testid="stIconMaterial"] {
            color: transparent !important;
        }

        /* ปรับแต่งกล่องแชท — ไม่ใส่ขอบ */
        .stChatMessage { border-radius: 12px; margin-bottom: 8px; border: none !important; }
        [data-testid="stChatMessageAvatar"] { display: none !important; }
        [data-testid^="stChatMessageAvatar"] { display: none !important; }
        .stChatMessageAvatar { display: none !important; }
        [data-testid="stChatMessage"] [aria-label*="avatar"] { display: none !important; }
        [data-testid="stChatMessageContent"] { margin-left: 0 !important; }
        [data-testid="stChatMessage"] > div:first-child { display: none !important; }
        [data-testid="stChatMessage"] { gap: 0 !important; }

        /* ปรับแต่ง Dropdown/Expander ใน Sidebar — ไม่ใส่ขอบขาว */
        .stExpander { border: none !important; border-radius: 8px !important; margin-bottom: 5px !important; }

        /* สไตล์ปุ่มใน Sidebar (ยกเว้นปุ่มล้าง session) */
        .stButton button {
            text-align: left !important;
            width: 100% !important;
            padding: 8px 12px !important;
            font-size: 14px !important;
            transition: 0.2s;
        }

        /* ปุ่มล้าง session — แดง + ตัวอักษรขาว */
        #btn_clear_session,
        [data-testid="stBaseButton-secondary"][id="btn_clear_session"],
        .stButton button:has(#btn_clear_session) {
            background-color: #dc3545 !important;
            color: white !important;
            border: 1px solid #dc3545 !important;
            font-weight: 600 !important;
        }
        #btn_clear_session:hover {
            background-color: #b02a37 !important;
            border-color: #b02a37 !important;
        }

        code { color: #a5d6ff !important; background-color: #1e1e2e !important; padding: 2px 4px; border-radius: 4px; }
        pre code { color: #c9d1d9 !important; background-color: #161b22 !important; }

        /* Code block ปรับให้เข้ากับ dark theme */
        .stCodeBlock, pre {
            background-color: #161b22 !important;
            border-radius: 8px !important;
        }

        /* ปรับปรุงหัวข้อ Sidebar */
        [data-testid="stSidebarNav"] { display: none; }
        h1, h2, h3 { color: #1e293b; }
        .icon-label {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            line-height: 1.2;
        }
        .inline-icon {
            font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif;
            font-size: 1em;
            line-height: 1;
            vertical-align: middle;
            font-variation-settings: 'FILL' 1, 'wght' 500, 'GRAD' 0, 'opsz' 24;
        }
        .icon-label-title {
            font-size: 2rem;
            font-weight: 700;
        }
        .icon-label-section {
            font-size: 1.05rem;
            font-weight: 600;
        }
        .role-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 700;
            border: 1px solid transparent;
            margin-bottom: 0.35rem;
        }
        .role-chip-user {
            background: #fee2e2;
            color: #991b1b;
            border-color: #fca5a5;
        }
        .role-chip-assistant {
            background: #fef3c7;
            color: #92400e;
            border-color: #fcd34d;
        }
    </style>
    """, unsafe_allow_html=True)

# Common Material-Icon text tokens the LLM may emit instead of real icons
_ICON_TEXT_RE = re.compile(
    r'\b(?:keyboard_double_arrow_right|arrow_forward|arrow_right|arrow_drop_down'
    r'|check_circle|radio_button_unchecked|star|grade|info|warning|error'
    r'|lightbulb|tips_and_updates|menu_book|school|science|edit_note'
    r'|task_alt|trending_flat|east|west|north|south|open_in_new'
    r'|chevron_right|expand_more|expand_less|navigate_next|navigate_before)\b'
)

def clean_icon_text(text: str) -> str:
    """Remove Material-Icon text codes that the LLM sometimes outputs."""
    return _ICON_TEXT_RE.sub("", text)


def icon_label(icon_name: str, label: str, variant: str = "section") -> str:
    variant_cls = "icon-label-title" if variant == "title" else "icon-label-section"
    return (
        f'<span class="icon-label {variant_cls}">'
        f'<span class="inline-icon">{icon_name}</span>'
        f'<span>{label}</span>'
        f"</span>"
    )


def role_chip_html(role: str) -> str:
    if role == "user":
        icon_name = "person"
        label = "ผู้ใช้"
        cls = "role-chip role-chip-user"
    else:
        icon_name = "smart_toy"
        label = "ผู้ช่วย AI"
        cls = "role-chip role-chip-assistant"
    return (
        f'<span class="{cls}">'
        f'<span class="inline-icon">{icon_name}</span>'
        f"<span>{label}</span>"
        f"</span>"
    )


def preferred_image_limit(base_limit: int) -> int:
    if base_limit > 0:
        return int(base_limit)
    # Show at least 2 evidence images by default to support image-heavy chapters.
    return 2


_INLINE_CITATION_RE = re.compile(r"\[([^\[\]\n]{3,220})\]")
_CHUNK_ID_RE = re.compile(r"chunk-\d+", re.IGNORECASE)
# Require at least one alphabetic char in source-part (avoid matching array ranges like 1:5 or -1:3).
_SOURCE_PAGE_CITE_RE = re.compile(
    r"^(?=[a-z0-9_\-./]*[a-z])[a-z0-9_\-./]+:\d+(?:\|[a-z0-9_\-./]+)?$",
    re.IGNORECASE,
)
_ARRAY_INDEX_RANGE_RE = re.compile(r"^-?\d+\s*:\s*-?\d+$")
_NOISE_CITE_RE = re.compile(r"^[a-z]:[a-z]$", re.IGNORECASE)
_MERMAID_BLOCK_RE = re.compile(r"```\s*mermaid\s*\n.*?\n\s*```", re.IGNORECASE | re.DOTALL)
_CONTEXT_PLACEHOLDER_RE = re.compile(r"context\s*\d+", re.IGNORECASE)
_SOURCE_PAGE_PARSE_RE = re.compile(r"^(?P<src>[^:\|\]]+):(?P<page>\d+)(?:\|(?P<chunk>[^\]]+))?$", re.IGNORECASE)


def _looks_like_source_page_citation(raw_part: str) -> bool:
    p = str(raw_part or "").strip()
    if not p:
        return False
    if _ARRAY_INDEX_RANGE_RE.match(p):
        return False
    return bool(_SOURCE_PAGE_CITE_RE.match(p))


def _looks_like_citation_token(raw_part: str) -> bool:
    p = str(raw_part or "").strip()
    if not p:
        return False
    if _NOISE_CITE_RE.match(p):
        return False
    if _ARRAY_INDEX_RANGE_RE.match(p):
        return False
    if _CHUNK_ID_RE.search(p):
        return True
    if _CONTEXT_PLACEHOLDER_RE.search(p):
        return True
    if _looks_like_source_page_citation(p):
        return True
    return False


def _normalize_citation_key(citation: str) -> str:
    normalized = str(citation or "").strip()
    normalized = normalized.strip("[](){}")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.lower()


def _allowed_citation_key_map(sources: list[dict]) -> dict[str, str]:
    """
    Build normalized->display citation map with several accepted canonical forms.
    """
    mapping: dict[str, str] = {}
    for s in sources or []:
        source = str(s.get("source", "")).strip()
        page = str(s.get("page", "")).strip()
        chunk_id = str(s.get("chunk_id", "")).strip()
        citation = str(s.get("citation", "")).strip()
        source_page = citation or (f"{source}:{page}" if source and page else source)
        if not source_page:
            continue

        candidates = []
        if chunk_id:
            canonical_chunk = ""
            if source and page:
                canonical_chunk = f"{source}:{page}|{chunk_id}"
            elif source:
                canonical_chunk = f"{source}|{chunk_id}"
            else:
                canonical_chunk = chunk_id
            candidates.append((f"{source_page}|{chunk_id}", f"{source_page}|{chunk_id}"))
            if source and page:
                candidates.append((f"{source}:{page}|{chunk_id}", f"{source}:{page}|{chunk_id}"))
            if source:
                candidates.append((f"{source}|{chunk_id}", f"{source}|{chunk_id}"))
            candidates.append((chunk_id, canonical_chunk))
        if source and page:
            candidates.append((f"{source}:{page}", f"{source}:{page}"))
        if source_page:
            candidates.append((source_page, source_page))

        for raw_key, display in candidates:
            key = _normalize_citation_key(raw_key)
            if key and key not in mapping:
                mapping[key] = display
    return mapping


def _preferred_citation_for_source_row(row: dict) -> str:
    src = str(row.get("source", "")).strip()
    page = str(row.get("page", "")).strip()
    chunk = str(row.get("chunk_id", "")).strip()
    if src and page and chunk:
        return f"{src}:{page}|{chunk}"
    if src and page:
        return f"{src}:{page}"
    return src or chunk


def _normalize_chunk_label(chunk_id: str) -> str:
    raw = str(chunk_id or "").strip()
    if not raw:
        return ""
    return re.sub(r"(?i)^chunk-?", "", raw).strip() or raw


def _humanize_citation_token(token: str) -> str:
    raw = str(token or "").strip().strip("[]")
    if not raw:
        return raw
    m = _SOURCE_PAGE_PARSE_RE.match(raw)
    if m:
        page = str(m.group("page") or "").strip()
        chunk = _normalize_chunk_label(m.group("chunk") or "")
        if page and chunk:
            return f"[หน้า {page}]"
        if page:
            return f"หน้า {page}"
    cm = _CHUNK_ID_RE.search(raw)
    if cm:
        return f"ตอน {_normalize_chunk_label(cm.group(0))}"
    return raw


def humanize_answer_citations_for_user(answer: str, sources: list[dict]) -> str:
    """
    Convert technical inline citations to compact Thai labels for UI readability.
    Example: [data.pdf:35|chunk-00118] -> [หน้า 35]
    """
    if not answer:
        return answer
    key_map = _allowed_citation_key_map(sources)

    def repl(match):
        raw = str(match.group(1) or "").strip()
        if not raw:
            return match.group(0)
        parts = [p.strip() for p in re.split(r"[;,]", raw) if p.strip()]
        if not parts:
            parts = [raw]

        has_known_cite = False
        rendered: list[str] = []
        seen = set()
        for p in parts:
            key = _normalize_citation_key(p)
            canonical = key_map.get(key, p)
            if key in key_map or _looks_like_citation_token(p):
                has_known_cite = True
            human = _humanize_citation_token(canonical)
            hkey = _normalize_citation_key(human)
            if human and hkey not in seen:
                seen.add(hkey)
                rendered.append(human)

        if not has_known_cite or not rendered:
            return match.group(0)
        return "[" + "; ".join(rendered) + "]"

    return _INLINE_CITATION_RE.sub(repl, answer)


def _scope_pages_for_display(scope_pages: list[str]) -> list[str]:
    pages = []
    for p in scope_pages or []:
        t = str(p).strip()
        if not t:
            continue
        if ":" in t:
            t = t.split(":")[-1].strip()
        if t.isdigit():
            pages.append(t)
    uniq = sorted({int(x) for x in pages})
    return [str(x) for x in uniq]


def _friendly_source_badge(row: dict) -> str:
    page = str(row.get("page", "")).strip()
    chunk = _normalize_chunk_label(row.get("chunk_id", ""))
    if page and chunk:
        return f"[หน้า {page}]"
    if page:
        return f"หน้า {page}"
    if chunk:
        return f"ตอน {chunk}"
    return "อ้างอิงเอกสาร"


def _allowed_citation_keys(sources: list[dict]) -> set[str]:
    return set(_allowed_citation_key_map(sources).keys())


def _extract_inline_citations(text: str) -> list[str]:
    found = []
    for m in _INLINE_CITATION_RE.finditer(text or ""):
        raw = m.group(1)
        parts = [p.strip() for p in re.split(r"[;,]", raw) if p.strip()]
        if not parts:
            parts = [raw.strip()]
        for p in parts:
            if not _looks_like_citation_token(p):
                continue
            key = _normalize_citation_key(p)
            if key:
                found.append(key)
    return found


def strip_generated_mermaid_blocks(text: str) -> str:
    if not text:
        return text
    cleaned = _MERMAID_BLOCK_RE.sub("", text)

    # Also remove accidental plain Mermaid bodies (no fenced code block),
    # e.g. lines starting with "graph TD" followed by arrow/node lines.
    lines = cleaned.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip().lower()
        if re.match(r"^(graph|flowchart)\s+[a-z]{2}\b", line):
            j = i + 1
            saw_mermaid_signal = False
            while j < len(lines):
                ln = lines[j].strip()
                ln_l = ln.lower()
                if not ln:
                    if saw_mermaid_signal:
                        j += 1
                    break
                if (
                    "-->" in ln
                    or "---" in ln
                    or "subgraph" in ln_l
                    or "classdef" in ln_l
                    or ":::"
                    in ln
                    or bool(re.match(r"^[a-z0-9_]+\s*\[.*\]\s*$", ln_l))
                ):
                    saw_mermaid_signal = True
                    j += 1
                    continue
                break
            if saw_mermaid_signal:
                i = j
                continue
        out.append(lines[i])
        i += 1

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def normalize_answer_citations(answer: str, sources: list[dict]) -> str:
    """
    Normalize inline citations to known canonical forms to reduce false unknown_citations.
    """
    if not answer:
        return answer
    key_map = _allowed_citation_key_map(sources)
    if not key_map:
        return answer
    context_idx_map = {}
    for i, s in enumerate(sources or [], start=1):
        preferred = _preferred_citation_for_source_row(s)
        if preferred:
            context_idx_map[str(i)] = preferred

    def resolve_unknown_key(key: str) -> str | None:
        if key in key_map:
            return key_map[key]
        if "|" in key:
            left, right = key.split("|", 1)
            left = _normalize_citation_key(left)
            right = _normalize_citation_key(right)
            if right in key_map:
                return key_map[right]
            if left in key_map:
                return key_map[left]
        ctx_match = re.search(r"context\s*(\d+)", key, re.IGNORECASE)
        if ctx_match:
            mapped_ctx = context_idx_map.get(ctx_match.group(1))
            if mapped_ctx:
                return mapped_ctx
        chunk_match = _CHUNK_ID_RE.search(key)
        if chunk_match:
            chunk = _normalize_citation_key(chunk_match.group(0))
            for k, v in key_map.items():
                if chunk in k:
                    return v
        if ":" in key:
            source_page = key.split("|", 1)[0]
            for k, v in key_map.items():
                if k.startswith(source_page):
                    return v
        return None

    def repl(match: re.Match) -> str:
        raw_inside = match.group(1)
        parts = [p.strip() for p in re.split(r"[;,]", raw_inside) if p.strip()]
        if not parts:
            parts = [raw_inside.strip()]
        resolved = []
        for p in parts:
            key = _normalize_citation_key(p)
            mapped = resolve_unknown_key(key)
            if mapped:
                resolved.append(mapped)
            else:
                resolved.append(p.strip())
        # de-dup while preserving order
        uniq = []
        seen = set()
        for x in resolved:
            k = _normalize_citation_key(x)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(x)
        return "[" + "; ".join(uniq) + "]"

    return _INLINE_CITATION_RE.sub(repl, answer)


def strip_context_placeholder_citations(answer: str) -> str:
    if not answer:
        return answer
    # Remove unresolved placeholders like [Context 2 ...] that are not valid source citations.
    cleaned = re.sub(r"\[(?:\s*context\s*\d+[^\]]*)\]", "", answer, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def sanitize_broken_citation_brackets(answer: str) -> str:
    if not answer:
        return answer
    text = str(answer)
    # normalize accidental duplicated brackets from streamed/truncated output
    text = re.sub(r"\[\s*\[+", "[", text)
    text = re.sub(r"\]+\s*\]+", "]", text)
    # drop dangling unfinished citation tail
    if text.count("[") > text.count("]"):
        last_open = text.rfind("[")
        last_close = text.rfind("]")
        if last_open > last_close:
            text = text[:last_open].rstrip()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def trim_cited_claim_lines(answer: str, max_claim_lines: int = 6) -> str:
    """
    Keep response concise by limiting number of citation-bearing claim lines.
    """
    if not answer:
        return answer
    max_keep = max(1, int(max_claim_lines))
    out_lines: list[str] = []
    in_code = False
    claim_seen = 0
    for raw in str(answer).splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code:
            out_lines.append(line)
            continue
        if not stripped:
            out_lines.append(line)
            continue
        if stripped.startswith("#"):
            out_lines.append(line)
            continue
        if stripped.startswith("|"):
            out_lines.append(line)
            continue
        if _INLINE_CITATION_RE.search(stripped):
            claim_seen += 1
            if claim_seen > max_keep:
                continue
            out_lines.append(line)
            continue
        # keep short connectors only before first claim; drop long uncited tails
        if claim_seen == 0 and len(stripped) <= 80:
            out_lines.append(line)
    compact = "\n".join(out_lines)
    return re.sub(r"\n{3,}", "\n\n", compact).strip()


def _extract_cited_sentence_units(answer: str) -> list[str]:
    """
    Extract citation-bearing sentence/claim units from mixed-format markdown text.
    Designed for deterministic low-latency section compaction.
    """
    if not answer:
        return []
    units: list[str] = []
    seen: set[str] = set()
    in_code = False
    for raw in str(answer).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.startswith("#") or line.startswith("|"):
            continue
        # Prefer splitting after citation blocks to keep sentence+citation coupled.
        parts = re.split(r"(?<=\])\s+|(?<=[\.\!\?])\s+", line)
        for part in parts:
            seg = re.sub(r"\s+", " ", (part or "").strip())
            if not seg:
                continue
            if not _INLINE_CITATION_RE.search(seg):
                continue
            if len(re.sub(r"\[[^\]]+\]", " ", seg).strip()) < 10:
                continue
            key = re.sub(r"\s+", "", seg.lower())
            if key in seen:
                continue
            seen.add(key)
            units.append(seg)
    return units


def compact_section_answer_deterministic(answer: str, target_section_id: str = "") -> str:
    """
    Compact long section answers using only citation-bearing units.
    No model call, deterministic, and citation-safe.
    """
    if not answer:
        return answer

    sid = str(target_section_id or "").strip()
    max_units = 4 if sid in {"3.3.2", "3.3.3"} else 5
    # Approximate 1 token ~= 4 chars for Thai/English mixed text budgeting.
    max_chars_from_tokens = int(max(120, int(SECTION_CONCISE_MAX_TOKENS))) * 4
    max_chars = max(420, min(int(SECTION_CONCISE_TRIGGER_CHARS), max_chars_from_tokens))

    units = _extract_cited_sentence_units(answer)
    if not units:
        return trim_cited_claim_lines(answer, max_claim_lines=max_units)

    out_lines: list[str] = []
    total = 0
    for idx, unit in enumerate(units[: max(1, max_units * 2)], start=1):
        clean = re.sub(r"^[-*•\d\.\)\s]+", "", unit).strip()
        if not clean:
            continue
        line = f"{idx}. {clean}"
        if out_lines and (total + len(line) + 1) > max_chars:
            break
        out_lines.append(line)
        total += len(line) + 1
        if len(out_lines) >= max_units:
            break

    if not out_lines:
        return trim_cited_claim_lines(answer, max_claim_lines=max_units)
    return "\n".join(out_lines).strip()


def _effective_citation_ratio_target(sources: list[dict]) -> float:
    base = float(MIN_CITED_CLAIM_RATIO)
    try:
        mode = str(st.session_state.get("retrieval_mode", "text")).strip().lower()
    except Exception:
        mode = "text"
    if mode != "visual":
        return base
    if not sources:
        return base
    fig_refs = set()
    for s in sources or []:
        for fr in s.get("figure_refs", []) or []:
            frs = str(fr).strip()
            if frs:
                fig_refs.add(frs)
    # Visual mode tends to be longer/step-wise; relax coverage target to reduce false abstain
    # while still enforcing unknown-citation=0.
    target = min(base, float(VISUAL_CITATION_MIN_CLAIM_RATIO))
    if len(sources) <= 2 or len(fig_refs) <= 1:
        target = min(target, 0.58)
    return max(0.50, float(target))


def autofill_missing_claim_citations(answer: str, sources: list[dict]) -> str:
    """
    Conservative fallback: append one top citation to claim-like lines that have no citation.
    """
    if not answer or not sources:
        return answer
    key_map = _allowed_citation_key_map(sources)
    if not key_map:
        return answer

    preferred = None
    for s in sources:
        src = str(s.get("source", "")).strip()
        page = str(s.get("page", "")).strip()
        chunk = str(s.get("chunk_id", "")).strip()
        if src and page and chunk:
            preferred = f"{src}:{page}|{chunk}"
            break
    if not preferred:
        preferred = next(iter(key_map.values()))
    cite = f"[{preferred}]"

    out_lines = []
    in_code = False
    for raw in (answer or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code or not stripped:
            out_lines.append(line)
            continue
        if stripped.startswith("|"):
            out_lines.append(line)
            continue
        if _INLINE_CITATION_RE.search(stripped):
            out_lines.append(line)
            continue
        if len(re.sub(r"^[-*•\d\.\)\s]+", "", stripped)) < 12:
            out_lines.append(line)
            continue
        if not re.search(r"[A-Za-zก-๙]", stripped):
            out_lines.append(line)
            continue
        if stripped.lower().startswith(("แหล่งที่มา", "sources", "อ้างอิง")):
            out_lines.append(line)
            continue
        out_lines.append(f"{line} {cite}")
    return "\n".join(out_lines)


def _is_abstain_like_answer(text: str) -> bool:
    t = (text or "").lower()
    markers = (
        "ไม่สามารถตอบ", "ข้อมูลไม่เพียงพอ", "ยังไม่พบหลักฐาน",
        "ไม่ทราบจากเอกสาร", "insufficient evidence", "i don't know",
    )
    return any(m in t for m in markers)


def _extract_claim_units(answer: str) -> list[str]:
    if not answer:
        return []
    units = []
    in_code = False
    for raw in answer.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not line:
            continue
        if line.startswith("#"):
            # Treat markdown headings as structural text, not factual claim units.
            continue
        if line.startswith("|"):
            continue
        line = line.lstrip("#").strip()
        line = re.sub(r"^[-*•\d\.\)\s]+", "", line).strip()
        if len(line) < 8:
            continue
        if line.lower().startswith(("แหล่งที่มา", "sources", "อ้างอิง")):
            continue
        if not re.search(r"[A-Za-zก-๙]", line):
            continue
        units.append(line)
    return units


def validate_claim_citations(answer: str, sources: list[dict], min_ratio: float | None = None) -> dict:
    allowed = _allowed_citation_keys(sources)
    target_ratio = float(MIN_CITED_CLAIM_RATIO if min_ratio is None else min_ratio)
    target_ratio = max(0.0, min(1.0, target_ratio))
    if _is_abstain_like_answer(answer):
        return {
            "citation_ok": True,
            "claim_count": 0,
            "cited_claim_count": 0,
            "cited_claim_ratio": 1.0,
            "target_ratio": target_ratio,
            "unknown_citations": [],
            "missing_claims": [],
            "allowed_citations_count": len(allowed),
        }

    claims = _extract_claim_units(answer)
    if not claims:
        return {
            "citation_ok": False,
            "claim_count": 0,
            "cited_claim_count": 0,
            "cited_claim_ratio": 0.0,
            "target_ratio": target_ratio,
            "unknown_citations": [],
            "missing_claims": ["empty_or_non_claim_answer"],
            "allowed_citations_count": len(allowed),
        }

    cited_claim_count = 0
    unknown_citations = set()
    missing_claims = []
    for claim in claims:
        citations = _extract_inline_citations(claim)
        valid = [c for c in citations if c in allowed]
        unknown_citations.update(c for c in citations if c not in allowed)
        if valid:
            cited_claim_count += 1
        else:
            missing_claims.append(claim[:180])

    ratio = cited_claim_count / len(claims)
    unknown_context_placeholders = sorted([u for u in unknown_citations if _CONTEXT_PLACEHOLDER_RE.search(str(u))])
    unknown_real = sorted(
        [
            u
            for u in unknown_citations
            if u not in set(unknown_context_placeholders)
            and not _NOISE_CITE_RE.match(str(u))
        ]
    )
    citation_ok = (ratio >= target_ratio) and (len(unknown_real) == 0)
    return {
        "citation_ok": bool(citation_ok),
        "claim_count": len(claims),
        "cited_claim_count": int(cited_claim_count),
        "cited_claim_ratio": round(float(ratio), 4),
        "target_ratio": target_ratio,
        "unknown_citations": sorted(unknown_real),
        "unknown_context_placeholders": unknown_context_placeholders,
        "missing_claims": missing_claims[:5],
        "allowed_citations_count": len(allowed),
    }


def _tokenize_overlap_terms(text: str) -> list[str]:
    txt = _normalize_match_text(text or "")
    if not txt:
        return []
    raw_tokens: list[str] = []
    if thai_word_tokenize:
        try:
            raw_tokens = [str(t).strip().lower() for t in thai_word_tokenize(txt, keep_whitespace=False) if str(t).strip()]
        except Exception:
            raw_tokens = []
    if not raw_tokens:
        raw_tokens = [t.lower() for t in re.findall(r"[A-Za-zก-๙0-9_]+", txt)]

    stop = {
        "การ", "ของ", "และ", "หรือ", "ที่", "ใน", "เป็น", "ได้", "ให้", "กับ",
        "โดย", "จาก", "เพื่อ", "มี", "ว่า", "this", "that", "with", "from", "the",
    }
    out = []
    for tok in raw_tokens:
        tok = tok.strip().lower()
        if not tok or tok in stop:
            continue
        if tok.isdigit():
            continue
        if len(tok) < 2:
            continue
        out.append(tok)
    return out


def _build_context_vocab(context_text: str) -> set[str]:
    return set(_tokenize_overlap_terms(context_text or ""))


def enforce_sentence_level_citation_gate(answer: str, sources: list[dict], context_text: str = "") -> tuple[str, dict]:
    """
    Strict sentence-level gate:
    - keep only claim lines with valid citations
    - reject lines with unknown citations
    - reject lines with too-low lexical overlap vs context
    """
    stats = {
        "input_claim_lines": 0,
        "kept_claim_lines": 0,
        "dropped_no_citation": 0,
        "dropped_unknown_citation": 0,
        "dropped_low_overlap": 0,
    }
    if not answer:
        return answer, stats

    allowed = _allowed_citation_keys(sources)
    if not allowed:
        return answer, stats
    ctx_vocab = _build_context_vocab(context_text)

    out_lines = []
    in_code = False
    for raw in (answer or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code:
            out_lines.append(line)
            continue
        if not stripped:
            out_lines.append(line)
            continue
        if stripped.startswith("|"):
            out_lines.append(line)
            continue
        if stripped.startswith("#"):
            out_lines.append(line)
            continue

        content_only = re.sub(r"\[[^\[\]\n]{3,220}\]", " ", stripped)
        content_only = re.sub(r"^[-*•\d\.\)\s]+", "", content_only).strip()
        if len(content_only) < 10 or not re.search(r"[A-Za-zก-๙]", content_only):
            out_lines.append(line)
            continue

        stats["input_claim_lines"] += 1
        cites = _extract_inline_citations(stripped)
        valid = [c for c in cites if c in allowed]
        unknown = [c for c in cites if c not in allowed and not _NOISE_CITE_RE.match(str(c))]
        if _CONTEXT_PLACEHOLDER_RE.search(stripped):
            unknown.append("context_placeholder")
        if not valid:
            stats["dropped_no_citation"] += 1
            continue
        if unknown:
            stats["dropped_unknown_citation"] += 1
            continue

        if ctx_vocab:
            claim_tokens = set(_tokenize_overlap_terms(content_only))
            overlap = len(claim_tokens & ctx_vocab)
            if overlap < (1 if len(claim_tokens) <= 2 else 2):
                stats["dropped_low_overlap"] += 1
                continue

        out_lines.append(line)
        stats["kept_claim_lines"] += 1

    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, stats


def repair_answer_citations(question: str, answer: str, context_text: str, sources: list[dict]) -> str:
    allowed = sorted(set(_allowed_citation_key_map(sources).values()))
    if not allowed:
        return ""
    allowed_text = "\n".join(f"- [{c}]" for c in allowed[:24])
    prompt = (
        "Rewrite the answer to be strictly grounded in provided context.\n"
        "Rules:\n"
        "1) Keep only claims supported by context.\n"
        "2) Every claim line must include at least one inline citation from the allowed list.\n"
        "3) Do NOT invent citations outside allowed list.\n"
        "4) Never use placeholder citations like context1/context2.\n"
        "5) Keep response concise and in Thai.\n\n"
        f"[Question]\n{question}\n\n"
        f"[Allowed citations]\n{allowed_text}\n\n"
        f"[Context]\n{context_text[:3000]}\n\n"
        f"[Draft answer]\n{answer[:2200]}"
    )
    messages = [
        {"role": "system", "content": "You are a strict citation repair assistant. Output only the repaired Thai answer."},
        {"role": "user", "content": prompt},
    ]
    res = call_hf_api(messages, CHAT_MODEL_ID, stream=False, max_tokens=CITATION_REPAIR_MAX_TOKENS, temperature=0.0)
    if not res or not getattr(res, "choices", None):
        return ""
    return (res.choices[0].message.content or "").strip()


def repair_answer_for_step_coverage(
    question: str,
    draft_answer: str,
    context_text: str,
    sources: list[dict],
    step_gate: dict,
) -> str:
    allowed = sorted(set(_allowed_citation_key_map(sources).values()))
    if not allowed:
        return ""
    missing_groups = step_gate.get("missing_groups", []) if isinstance(step_gate, dict) else []
    missing_text = "; ".join("/".join(str(t) for t in (g or []) if str(t).strip()) for g in missing_groups[:4])
    allowed_text = "\n".join(f"- [{c}]" for c in allowed[:24])
    prompt = (
        "Rewrite the answer in Thai to improve procedural step coverage.\n"
        "Rules:\n"
        "1) Use only facts explicitly present in context.\n"
        "2) Keep concise: 3-6 numbered steps maximum.\n"
        "3) Every step line must include at least one citation from allowed list.\n"
        "4) Do not invent facts; if a requested step is missing in context, state that briefly with citation.\n"
        "5) Do not output Mermaid/code blocks.\n"
        "6) Ensure the final sentence is complete (no truncated tail).\n\n"
        f"[Question]\n{question}\n\n"
        f"[Missing step groups to prioritize]\n{missing_text or 'none'}\n\n"
        f"[Allowed citations]\n{allowed_text}\n\n"
        f"[Context]\n{context_text[:3200]}\n\n"
        f"[Draft answer]\n{draft_answer[:1800]}"
    )
    messages = [
        {"role": "system", "content": "You are a strict Thai procedural answer repair assistant."},
        {"role": "user", "content": prompt},
    ]
    max_tok = min(int(TEXT_SECTION_MAX_TOKENS), 420)
    res = call_hf_api(messages, CHAT_MODEL_ID, stream=False, max_tokens=max_tok, temperature=0.0)
    if not res or not getattr(res, "choices", None):
        return ""
    return (res.choices[0].message.content or "").strip()


def sanitize_mermaid(code: str) -> str:
    """Clean up LLM-generated Mermaid code to avoid syntax errors."""
    code = code.strip()
    # Remove leading/trailing empty lines
    lines = [l for l in code.split("\n") if l.strip()]
    if not lines:
        return ""

    # Ensure first line is a valid diagram type
    first = lines[0].strip().lower()
    valid_starts = ("graph", "flowchart", "sequencediagram", "classdiagram",
                    "statediagram", "erdiagram", "gantt", "pie", "mindmap",
                    "gitgraph", "timeline", "journey", "quadrantchart")
    if not any(first.startswith(v) for v in valid_starts):
        lines.insert(0, "graph TD")

    # Sanitize each line
    cleaned = []
    for line in lines:
        # Replace Thai/smart quotes that break Mermaid
        line = line.replace('"', "'")

        # Fix content inside square-bracket node labels: A[...text...]
        # Parentheses () inside [...] break Mermaid (it thinks they're node shapes)
        # Replace ( ) with fullwidth equivalents inside [...] only
        def _fix_brackets(m):
            inner = m.group(1)
            inner = inner.replace("(", "&#40;").replace(")", "&#41;")
            inner = inner.replace("{", "&#123;").replace("}", "&#125;")
            return "[" + inner + "]"

        line = re.sub(r'\[([^\]]*\([^\]]*)\]', _fix_brackets, line)
        # Also catch cases where {} inside [] (rhombus clash)
        line = re.sub(r'\[([^\]]*\{[^\]]*)\]', _fix_brackets, line)

        cleaned.append(line)

    return "\n".join(cleaned)


def render_mermaid(code: str):
    """เรนเดอร์ Mermaid Code เป็นกราฟจริง (ใช้ dark theme) พร้อม error handling"""
    sanitized = sanitize_mermaid(code)
    if not sanitized:
        return

    # Escape the code for safe HTML embedding
    import html as html_mod
    safe_code = html_mod.escape(sanitized)

    mermaid_html = f"""
    <div id="mermaid-container">
        <pre class="mermaid">
{sanitized}
        </pre>
    </div>
    <div id="mermaid-error" style="display:none; background:#2d2d2d; color:#ff6b6b;
         padding:10px; border-radius:8px; font-size:13px; margin-top:5px;">
        <strong>⚠️ Mermaid Error:</strong> <span id="err-msg"></span>
        <details style="margin-top:5px;">
            <summary style="cursor:pointer; color:#a5d6ff;">ดูโค้ดต้นฉบับ</summary>
            <pre style="color:#c9d1d9; font-size:12px; white-space:pre-wrap;">{safe_code}</pre>
        </details>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
            startOnLoad: false,
            theme: 'dark',
            securityLevel: 'loose',
            flowchart: {{ htmlLabels: true, curve: 'basis' }}
        }});
        try {{
            const container = document.getElementById('mermaid-container');
            const pre = container.querySelector('.mermaid');
            const {{ svg }} = await mermaid.render('mermaid-svg', pre.textContent);
            container.innerHTML = '<div style="display:flex;justify-content:center;">' + svg + '</div>';
        }} catch (e) {{
            document.getElementById('mermaid-error').style.display = 'block';
            document.getElementById('err-msg').textContent = e.message || String(e);
        }}
    </script>
    """
    components.html(mermaid_html, height=500, scrolling=True)

# ---------------------------------------------------------------------------
# Data Loading & Structured Topics
# ---------------------------------------------------------------------------
@st.cache_resource
def load_rag():
    try:
        if not (PROJECT_ROOT / "indexes" / "faiss_index").exists():
            log_event("rag_load_skipped", reason="faiss_index_missing")
            return None
        rag_system = RAGSystem()
        rag_status = rag_system.get_runtime_status() if hasattr(rag_system, "get_runtime_status") else {}
        log_event("rag_loaded", **rag_status)
        return rag_system
    except Exception as exc:
        log_event("rag_load_failed", error=str(exc))
        return None

@st.cache_data
def load_structured_topics():
    """
    Parse list_list_hierarchy.txt into a nested structure.
    Format:
      - "1: โครงสร้างข้อมูลเบื้องต้น..." (Main topic)
      - "• 1.1 ความหมาย..." (Sub-topic with bullet)
      - "\t1.1.1 ความหมาย..." (Sub-sub-topic with tab)
    """
    structured: dict[str, list] = {}
    current_main = None
    current_sub = None

    if not TOPIC_FILE.exists():
        return structured

    with open(TOPIC_FILE, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue

            # --- Main topic ---
            # Match "1: โครงสร้างข้อมูลเบื้องต้น..."
            main_match = re.match(r'^(\d+)\s*:\s*(.+)$', stripped)
            if main_match:
                num = main_match.group(1)
                title = main_match.group(2).strip()
                current_main = f"{num}: {title}"
                structured[current_main] = []
                current_sub = None
                continue

            if current_main is None:
                continue

            # --- Sub-topic ---
            # Match "• 1.1 ความหมายโครงสร้างข้อมูลและอัลกอริทึม"
            sub_match = re.match(r'^•\s*(\d+\.\d+)\s+(.+)$', stripped)
            if sub_match:
                sub_title = f"{sub_match.group(1)} {sub_match.group(2).strip()}"
                current_sub = {"title": sub_title, "children": []}
                structured[current_main].append(current_sub)
                continue

            # --- Sub-sub-topic ---
            # Match lines that start with tab followed by numbering like "1.1.1"
            child_match = re.match(r'^(\d+\.\d+\.\d+)\s+(.+)$', stripped)
            if child_match and (raw_line.startswith("\t") or raw_line.startswith("    ")):
                child_title = f"{child_match.group(1)} {child_match.group(2).strip()}"
                if current_sub is not None:
                    current_sub["children"].append(child_title)
                else:
                    # If no current_sub, create one with the child as title
                    current_sub = {"title": child_title, "children": []}
                    structured[current_main].append(current_sub)

    return structured

rag = load_rag()
rag_runtime_status = rag.get_runtime_status() if (rag and hasattr(rag, "get_runtime_status")) else {}
topic_structure = load_structured_topics()
ensure_research_log()
figure_index = load_figure_index(str(FIGURE_MANIFEST_FILE))
region_manifest_index = load_region_manifest_index(str(FIGURE_REGION_MANIFEST_FILE))


def build_topic_labels(structured_topics: dict) -> list[str]:
    labels = []
    for main_topic, sub_items in structured_topics.items():
        labels.append(main_topic)
        for sub_item in sub_items:
            if isinstance(sub_item, dict):
                title = sub_item.get("title", "").strip()
                if title:
                    labels.append(title)
                for child in sub_item.get("children", []):
                    child = str(child).strip()
                    if child:
                        labels.append(child)
            else:
                sub_text = str(sub_item).strip()
                if sub_text:
                    labels.append(sub_text)
    return labels


def _normalize_topic_match_text(text: str) -> str:
    txt = str(text or "").strip().lower()
    txt = re.sub(r"[^\w\s\u0E00-\u0E7F]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _topic_match_tokens(text: str) -> list[str]:
    stop = {
        "หัวข้อ",
        "ที่",
        "และ",
        "หรือ",
        "ของ",
        "การ",
        "แบบ",
        "topic",
        "other",
        "the",
        "of",
        "and",
    }
    normalized = _normalize_topic_match_text(text)
    raw_tokens: list[str] = []
    if thai_word_tokenize is not None:
        try:
            raw_tokens.extend([str(t) for t in thai_word_tokenize(normalized, engine="newmm")])
        except Exception:
            pass
    raw_tokens.extend(re.split(r"\s+", normalized))

    out = []
    seen = set()
    for token in raw_tokens:
        token = token.strip()
        if not token or len(token) < 2 or token in stop:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)*", token):
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _label_for_matching(label: str) -> tuple[str, str]:
    raw = _normalize_topic_match_text(label)
    compact = re.sub(r"^(หัวข้อที่\s*\d+\s*:)?\s*", "", raw).strip()
    compact = re.sub(r"^\d+(?:\.\d+)*\s*", "", compact).strip()
    return raw, compact or raw


def shortlist_topic_labels_for_question(question: str, structured_topics: dict, max_candidates: int = 60) -> list[str]:
    q_norm = _normalize_topic_match_text(question)
    q_tokens = set(_topic_match_tokens(question))
    if not q_norm:
        return []

    scored: list[tuple[float, str]] = []
    for label in build_topic_labels(structured_topics):
        raw, compact = _label_for_matching(label)
        l_tokens = set(_topic_match_tokens(compact))
        overlap = len(q_tokens.intersection(l_tokens))
        contains = int(compact in q_norm or raw in q_norm)
        weak_contains = int(any(tok in q_norm for tok in l_tokens if len(tok) >= 3))
        if overlap == 0 and contains == 0 and weak_contains == 0:
            continue
        score = (
            1.8 * contains
            + 1.0 * overlap
            + 0.5 * weak_contains
            + min(len(l_tokens), 10) * 0.05
            + min(len(compact), 90) / 90.0 * 0.2
        )
        scored.append((score, label))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [label for _, label in scored[: max(1, int(max_candidates))]]


def best_topic_label_from_question(question: str, structured_topics: dict) -> str | None:
    # High-priority deterministic routing for short/noisy intents.
    sid = _infer_section_id_from_question(question)
    if sid:
        label = _topic_label_for_section_id(sid, structured_topics)
        if label:
            return label
        return sid
    candidates = shortlist_topic_labels_for_question(question, structured_topics, max_candidates=1)
    return candidates[0] if candidates else None


def _topic_label_for_section_id(section_id: str, structured_topics: dict) -> str | None:
    sid = str(section_id or "").strip()
    if not sid:
        return None

    labels = build_topic_labels(structured_topics)
    if not labels:
        return None

    exact_pat = re.compile(rf"^{re.escape(sid)}(?:\b|[\s\-\)])")
    for label in labels:
        raw = str(label).strip()
        compact = re.sub(r"^(หัวข้อที่\s*\d+\s*:)\s*", "", raw).strip()
        if exact_pat.search(compact):
            return raw

    # Chapter-level prompts, e.g. sid == "1" should map to "หัวข้อที่ 1: ..."
    if "." not in sid:
        chapter_pat = re.compile(rf"^หัวข้อที่\s*{re.escape(sid)}\s*:", re.IGNORECASE)
        for label in labels:
            raw = str(label).strip()
            if chapter_pat.search(raw):
                return raw
        sub_pat = re.compile(rf"^{re.escape(sid)}\.\d+")
        for label in labels:
            raw = str(label).strip()
            compact = re.sub(r"^(หัวข้อที่\s*\d+\s*:)\s*", "", raw).strip()
            if sub_pat.search(compact):
                return raw
    return None


def _select_most_specific_section_id(text: str | None) -> str:
    raw = str(text or "")
    matches = re.findall(r"\b\d+(?:\.\d+)+\b", raw)
    if not matches:
        # Fallback for chapter-level prompts, e.g. "หัวข้อที่ 1", "บทที่ 2", "Chapter 3"
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

    if has("อาร์เรย์", "array"):
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
    if has("สแตก", "stack"):
        if has("นำข้อมูลเข้า", "push"):
            return "4.2.1"
        if has("ดึงข้อมูลออก", "นำข้อมูลออก", "pop"):
            return "4.2.2"
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
        return "1.1.3"
    if has("อัลกอริทึม"):
        return "1.1.2"

    return ""


def resolve_target_section_id(topic_hint: str | None, question: str | None) -> str:
    from_hint = _select_most_specific_section_id(topic_hint)
    if from_hint:
        return from_hint
    return _infer_section_id_from_question(question)


_OPERATION_STEP_REQUIREMENTS = {
    # Queue via array: enqueue/dequeue mechanics around front/rear
    "3.3.2": {
        "groups": [
            ["insert", "enqueue", "เพิ่มข้อมูล"],
            ["remove", "dequeue", "ดึงข้อมูลออก"],
            ["front"],
            ["rear"],
        ],
        "min_groups": 3,
    },
    # Circular queue: circular concept + enqueue/dequeue + pointers
    "3.3.3": {
        "groups": [
            ["วงกลม", "circular"],
            ["enqueue", "เพิ่มข้อมูล"],
            ["dequeue", "ดึงข้อมูลออก"],
            ["front"],
            ["rear"],
        ],
        "min_groups": 3,
    },
    # Single linked list operation chapter
    "2.4.2": {
        "groups": [
            ["create list", "การสร้างลิสต์"],
            ["add node", "เพิ่มโหนด"],
            ["delete node", "ลบโหนด"],
            ["insert", "แทรกโหนด"],
        ],
        "min_groups": 3,
    },
}


def evaluate_operation_step_coverage(answer: str, question: str, target_section_id: str | None) -> dict:
    sid = str(target_section_id or "").strip() or _infer_section_id_from_question(question)
    req = _OPERATION_STEP_REQUIREMENTS.get(sid)
    if not req:
        return {"enabled": False, "pass": True, "target_section_id": sid, "matched_groups": 0, "required_groups": 0}
    text = _normalize_topic_match_text(answer or "")
    groups = req.get("groups", []) if isinstance(req, dict) else []
    matched = 0
    missing = []
    for g in groups:
        terms = [str(t).strip().lower() for t in (g or []) if str(t).strip()]
        hit = any(t in text for t in terms)
        if hit:
            matched += 1
        else:
            missing.append(terms[:3])
    min_groups = int(req.get("min_groups", max(1, len(groups) // 2)))
    return {
        "enabled": True,
        "pass": bool(matched >= min_groups),
        "target_section_id": sid,
        "matched_groups": int(matched),
        "required_groups": int(min_groups),
        "total_groups": int(len(groups)),
        "missing_groups": missing[:5],
    }


def topic_exists(topic_name: str, structured_topics: dict) -> bool:
    if not topic_name:
        return False
    for label in build_topic_labels(structured_topics):
        if topic_name in label or label in topic_name:
            return True
    return False


def topic_hint_for_retrieval(topic_name: str, structured_topics: dict) -> str | None:
    """Use predicted topic as retrieval hint only when it maps to known syllabus labels."""
    if not topic_name:
        return None
    if topic_name.strip().lower() == "other":
        return None
    if topic_exists(topic_name, structured_topics):
        return topic_name.strip()
    return None


def rewrite_query_for_retrieval(
    question: str,
    *,
    topic_hint: str | None = None,
    target_section_id: str | None = None,
) -> tuple[str, dict]:
    """
    Query routing/rewriting layer before retrieval.
    - Deterministic expansion by section/topic labels (default).
    - Optional LLM rewrite for noisy/short questions.
    """
    original = str(question or "").strip()
    if not QUERY_REWRITE_ENABLED or not original:
        return original, {"enabled": False, "mode": "disabled", "changed": False}

    sid = str(target_section_id or "").strip()
    topic_anchor = str(topic_hint or "").strip()
    if not topic_anchor and sid:
        topic_anchor = _topic_label_for_section_id(sid, topic_structure) or sid

    topic_compact = re.sub(r"^(หัวข้อที่\s*\d+\s*:)\s*", "", topic_anchor).strip() if topic_anchor else ""
    keyword_terms = _topic_match_tokens(topic_compact)[: max(1, int(QUERY_REWRITE_MAX_KEYWORDS))]
    deterministic = original
    if topic_compact and keyword_terms:
        deterministic = (
            f"{original} | ขอบเขตหัวข้อ: {topic_compact} | คำหลัก: {', '.join(keyword_terms)}"
        )
    elif topic_compact:
        deterministic = f"{original} | ขอบเขตหัวข้อ: {topic_compact}"

    used_mode = "deterministic"
    rewritten = deterministic
    llm_status = "not_used"
    if QUERY_REWRITE_USE_LLM:
        rewrite_prompt = (
            "Rewrite the user question for retrieval in Thai.\n"
            "Rules:\n"
            "1) Keep intent unchanged.\n"
            "2) Keep concise (1 line).\n"
            "3) Add key terms from target topic if relevant.\n"
            "4) Do not answer, do not add new facts.\n\n"
            f"[User Question]\n{original}\n\n"
            f"[Target Topic]\n{topic_compact or sid or 'unknown'}\n\n"
            f"[Keywords]\n{', '.join(keyword_terms) if keyword_terms else 'none'}\n\n"
            "Return only rewritten query text."
        )
        llm_messages = [{"role": "user", "content": rewrite_prompt}]
        res = call_hf_api(
            llm_messages,
            QUERY_REWRITE_MODEL or CHAT_MODEL_ID,
            stream=False,
            max_tokens=int(QUERY_REWRITE_MAX_TOKENS),
            temperature=0.0,
        )
        if res and getattr(res, "choices", None):
            candidate = str(res.choices[0].message.content or "").strip()
            candidate = re.sub(r"\s+", " ", candidate).strip().strip('"').strip("'")
            if len(candidate) >= 4:
                rewritten = candidate
                used_mode = "llm"
                llm_status = "ok"
            else:
                llm_status = "empty"
        else:
            llm_status = "api_failed"

    return rewritten, {
        "enabled": True,
        "mode": used_mode,
        "changed": rewritten != original,
        "target_section_id": sid,
        "topic_anchor": topic_anchor,
        "keyword_terms": keyword_terms,
        "llm_status": llm_status,
    }


def self_check_grounded_answer(
    question: str,
    answer: str,
    context_text: str,
    sources: list[dict],
) -> tuple[str, dict]:
    """
    Self-RAG pass: force answer to stay within retrieved context before final validation.
    """
    raw_answer = str(answer or "").strip()
    if (not SELF_CHECK_ENABLED) or len(raw_answer) < int(SELF_CHECK_MIN_ANSWER_CHARS):
        return raw_answer, {"enabled": False, "applied": False, "reason": "disabled_or_too_short"}

    allowed = sorted(set(_allowed_citation_key_map(sources).values()))
    allowed_text = ", ".join(allowed[:120]) if allowed else "none"
    prompt = (
        "You are a strict grounding verifier for Thai RAG answers.\n"
        "Rewrite the draft answer so that every claim is supported by the provided context only.\n"
        "Rules:\n"
        "1) Remove any unsupported claim.\n"
        "2) Keep all statements in Thai.\n"
        "3) Keep inline citations and only use allowed citations.\n"
        "4) If evidence is insufficient, output exactly: ยังไม่พบหลักฐานเพียงพอในบริบทที่ให้มา\n\n"
        f"[Question]\n{question}\n\n"
        f"[Allowed citations]\n{allowed_text}\n\n"
        f"[Context]\n{str(context_text or '')[: int(SELF_CHECK_CONTEXT_MAX_CHARS)]}\n\n"
        f"[Draft answer]\n{raw_answer}\n\n"
        "Return only the corrected answer."
    )
    messages = [
        {"role": "system", "content": "You are a strict grounding editor. Output only final Thai answer text."},
        {"role": "user", "content": prompt},
    ]
    res = call_hf_api(
        messages,
        CHAT_MODEL_ID,
        stream=False,
        max_tokens=int(SELF_CHECK_MAX_TOKENS),
        temperature=0.0,
    )
    if not res or not getattr(res, "choices", None):
        return raw_answer, {"enabled": True, "applied": False, "reason": "api_failed"}

    candidate = str(res.choices[0].message.content or "").strip()
    candidate = clean_icon_text(candidate)
    candidate = strip_generated_mermaid_blocks(candidate)
    candidate = sanitize_broken_citation_brackets(candidate)
    candidate = normalize_answer_citations(candidate, sources)
    candidate = strip_context_placeholder_citations(candidate)
    if not candidate:
        return raw_answer, {"enabled": True, "applied": False, "reason": "empty_candidate"}

    return candidate, {
        "enabled": True,
        "applied": candidate != raw_answer,
        "reason": "ok",
        "before_len": len(raw_answer),
        "after_len": len(candidate),
    }


def question_requires_structure(question: str) -> bool:
    q = (question or "").lower()
    keywords = (
        "graph", "tree", "binary tree", "structure", "diagram",
        "กราฟ", "ต้นไม้", "ไบนารีทรี", "โครงสร้าง", "แผนภาพ",
        "เชื่อมโยง", "โหนด", "linked list", "ลิงก์ลิสต์",
    )
    return any(k in q for k in keywords)


def compute_evidence_stats(
    docs: list,
    topic_hint: str | None,
    require_structure: bool,
    *,
    target_section_id: str | None = None,
    section_scope_pages: list[str] | None = None,
) -> dict:
    if not docs:
        return {
            "doc_count": 0,
            "topic_match_count": 0,
            "topic_match_ratio": 0.0,
            "structure_match_count": 0,
            "avg_rrf_score": 0.0,
            "rrf_count": 0,
            "evidence_ok": False,
            "reasons": ["no_docs"],
        }

    doc_count = len(docs)
    topic_match_count = 0
    structure_match_count = 0
    rrf_values = []
    sid = str(target_section_id or "").strip()
    sid_norm = _normalize_match_text(sid)
    scope_pages = {str(x).strip() for x in (section_scope_pages or []) if str(x).strip()}
    scope_page_nums = {p.split(":")[-1] if ":" in p else p for p in scope_pages}

    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        section_hit = False
        if sid_norm:
            best_tid = _normalize_match_text(str(meta.get("best_topic_id", "")).strip())
            top_tids = {
                _normalize_match_text(str(x).strip())
                for x in (meta.get("top_topic_ids", []) or [])
                if str(x).strip()
            }
            if best_tid == sid_norm or sid_norm in top_tids:
                section_hit = True
            if not section_hit and scope_pages:
                src = str(meta.get("source", "")).strip()
                page = str(meta.get("page", "")).strip()
                # Accept both styles:
                # - "source:page" keys
                # - plain page numbers (topic_to_pages override style)
                if page and (page in scope_page_nums):
                    section_hit = True
                elif src and page and f"{src}:{page}" in scope_pages:
                    section_hit = True

        if section_hit or bool(meta.get("topic_match", False)):
            topic_match_count += 1
            # Keep downstream behavior consistent in strict-section mode.
            meta["topic_match"] = True
            d.metadata = meta
        if bool(meta.get("structure_match", False)) or bool(meta.get("has_structure", False)):
            structure_match_count += 1
        rrf = meta.get("rrf_score")
        if isinstance(rrf, (int, float)):
            rrf_values.append(float(rrf))

    topic_ratio = topic_match_count / doc_count if doc_count else 0.0
    avg_rrf = sum(rrf_values) / len(rrf_values) if rrf_values else 0.0

    reasons = []
    if doc_count < EVIDENCE_MIN_DOCS:
        reasons.append("low_doc_count")
    if (topic_hint or sid_norm) and topic_ratio < EVIDENCE_MIN_TOPIC_MATCH_RATIO:
        reasons.append("low_topic_match")
    if require_structure and structure_match_count < 1:
        reasons.append("missing_structure_evidence")
    if rrf_values and avg_rrf < EVIDENCE_MIN_AVG_RRF:
        reasons.append("low_rrf_score")

    return {
        "doc_count": doc_count,
        "topic_match_count": topic_match_count,
        "topic_match_ratio": round(topic_ratio, 4),
        "structure_match_count": structure_match_count,
        "avg_rrf_score": round(avg_rrf, 8),
        "rrf_count": len(rrf_values),
        "evidence_ok": len(reasons) == 0,
        "reasons": reasons,
    }


def _doc_unique_key(doc) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    src = str(meta.get("source", "")).strip()
    page = str(meta.get("page", "")).strip()
    chunk = str(meta.get("chunk_id", "")).strip()
    return f"{src}:{page}:{chunk}"


def select_context_docs_diverse(docs: list, limit: int, min_unique_pages: int = 2, *, target_section_id: str | None = None) -> list:
    """
    Pick context docs with page diversity first, then fill by rank.
    This prevents operation answers from seeing only one repeated page.
    If target_section_id provided, prioritize chunks containing that section.
    """
    if not docs:
        return []
    k = max(1, int(limit))
    need_pages = max(1, int(min_unique_pages))
    
    # PRIORITIZE: If target section specified, move section-matching chunks to front
    if target_section_id:
        sid = str(target_section_id).strip()
        def _has_section(d) -> bool:
            meta = getattr(d, "metadata", {}) or {}
            check = " ".join([
                str(meta.get("H1", "")), str(meta.get("H2", "")), str(meta.get("H3", "")),
                str(meta.get("section", "")), str(meta.get("best_topic_id", "")),
                str(getattr(d, "page_content", "")[:200]),
            ])
            return sid in check
        section_docs = [d for d in docs if _has_section(d)]
        other_docs = [d for d in docs if not _has_section(d)]
        docs = section_docs + other_docs

    selected = []
    used_keys = set()
    seen_pages = set()

    for d in docs:
        if len(selected) >= k:
            break
        meta = getattr(d, "metadata", {}) or {}
        page = str(meta.get("page", "")).strip()
        key = _doc_unique_key(d)
        if key in used_keys:
            continue
        if page and page in seen_pages:
            continue
        selected.append(d)
        used_keys.add(key)
        if page:
            seen_pages.add(page)
        if len(seen_pages) >= need_pages and len(selected) >= min(k, need_pages):
            # keep gathering unique pages until first coverage target satisfied
            continue

    for d in docs:
        if len(selected) >= k:
            break
        key = _doc_unique_key(d)
        if key in used_keys:
            continue
        selected.append(d)
        used_keys.add(key)

    return selected[:k]


def _normalize_match_text(text: str) -> str:
    txt = (text or "").strip().lower()
    txt = re.sub(r"\s+", " ", txt)
    return txt


def _doc_has_structure_local(doc) -> bool:
    meta = getattr(doc, "metadata", {}) or {}
    if bool(meta.get("has_structure", False)):
        return True
    content = (getattr(doc, "page_content", "") or "").lower()
    if "[structure:" in content:
        return True
    hints = (
        "binary tree", "linked list", "graph", "queue", "stack",
        "ไบนารีทรี", "ลิงก์ลิสต์", "กราฟ", "โครงสร้าง", "แผนภาพ",
    )
    return any(h in content for h in hints)


def _doc_topic_match_local(doc, topic_hint: str | None) -> bool:
    if not topic_hint:
        return True
    hint = _normalize_match_text(topic_hint)
    if not hint or hint == "other":
        return True

    meta = getattr(doc, "metadata", {}) or {}
    haystack_parts = [
        str(meta.get("chapter", "")),
        str(meta.get("section", "")),
        str(meta.get("topic_path", "")),
        " ".join(str(x) for x in meta.get("topic_tags", []) if str(x).strip()),
        (getattr(doc, "page_content", "") or "")[:700],
    ]
    haystack = _normalize_match_text(" ".join(haystack_parts))
    if not haystack:
        return False
    if hint in haystack:
        return True
    for token in re.split(r"\s+", hint):
        token = token.strip()
        if len(token) >= 2 and token in haystack:
            return True
    return False


def _doc_section_match_local(doc, section_id: str) -> bool:
    sid = _normalize_match_text(section_id or "")
    if not sid:
        return False
    meta = getattr(doc, "metadata", {}) or {}
    candidates = set()
    best_tid = str(meta.get("best_topic_id", "")).strip()
    if best_tid:
        candidates.add(_normalize_match_text(best_tid))
    for tid in (meta.get("top_topic_ids", []) or []):
        t = str(tid).strip()
        if t:
            candidates.add(_normalize_match_text(t))
    if sid in candidates:
        return True

    haystack_parts = [
        str(meta.get("chapter", "")),
        str(meta.get("section", "")),
        str(meta.get("best_topic_id", "")),
        " ".join(str(x) for x in meta.get("top_topic_ids", []) if str(x).strip()),
    ]
    haystack = _normalize_match_text(" ".join(haystack_parts))
    return bool(sid and sid in haystack)


def _doc_query_phrase_match_local(doc, query_text: str | None) -> bool:
    q = _normalize_match_text(query_text or "")
    if not q or len(q) < 8:
        return False
    meta = getattr(doc, "metadata", {}) or {}
    haystack_parts = [
        str(meta.get("chapter", "")),
        str(meta.get("section", "")),
        str(meta.get("topic_path", "")),
        " ".join(str(x) for x in meta.get("topic_tags", []) if str(x).strip()),
        (getattr(doc, "page_content", "") or "")[:1400],
    ]
    haystack = _normalize_match_text(" ".join(haystack_parts))
    return q in haystack


def _build_query_anchor_profile_local(query_text: str | None) -> dict:
    text = _normalize_match_text(query_text or "")
    queue_terms = ["คิว", "queue", "enqueue", "dequeue", "front", "rear"]
    circular_terms = ["วงกลม", "circular"]
    linked_list_terms = ["ลิงค์ลิสต์", "ลิงก์ลิสต์", "linked list", "node", "head", "count", "nil"]
    stack_terms = ["สแตก", "stack", "push", "pop", "top"]
    tree_terms = ["ทรี", "tree", "binary", "traversal", "preorder", "inorder", "postorder"]

    def has_any(terms: list[str]) -> bool:
        return any(t in text for t in terms)

    enabled = any(
        [
            has_any(queue_terms),
            has_any(circular_terms),
            has_any(linked_list_terms),
            has_any(stack_terms),
            has_any(tree_terms),
        ]
    )
    return {
        "enabled": bool(enabled),
        "queue_terms": queue_terms,
        "circular_terms": circular_terms,
        "linked_list_terms": linked_list_terms,
        "stack_terms": stack_terms,
        "tree_terms": tree_terms,
        "require_queue": has_any(queue_terms),
        "require_circular": has_any(circular_terms),
        "require_linked_list": has_any(linked_list_terms),
        "require_stack": has_any(stack_terms),
        "require_tree": has_any(tree_terms),
    }


def _doc_query_anchor_match_local(doc, profile: dict | None) -> bool:
    if not profile or not profile.get("enabled", False):
        return True
    meta = getattr(doc, "metadata", {}) or {}
    haystack_parts = [
        str(meta.get("chapter", "")),
        str(meta.get("section", "")),
        str(meta.get("best_topic_title", "")),
        (getattr(doc, "page_content", "") or "")[:1200],
    ]
    haystack = _normalize_match_text(" ".join(haystack_parts))
    if not haystack:
        return False

    def has_any(terms: list[str]) -> bool:
        return any(t in haystack for t in terms)

    if profile.get("require_queue") and not has_any(profile.get("queue_terms", [])):
        return False
    if profile.get("require_circular") and not has_any(profile.get("circular_terms", [])):
        return False
    if profile.get("require_linked_list") and not has_any(profile.get("linked_list_terms", [])):
        return False
    if profile.get("require_stack") and not has_any(profile.get("stack_terms", [])):
        return False
    if profile.get("require_tree") and not has_any(profile.get("tree_terms", [])):
        return False
    return True


def apply_local_retrieval_filters(docs: list, filters: dict | None) -> list:
    if not docs:
        return []
    if not filters:
        return docs

    topic_hint = (filters.get("topic_hint") or "").strip()
    query_text = (filters.get("query_text") or "").strip()
    require_structure = bool(filters.get("require_structure", False))
    strict_topic = bool(filters.get("strict_topic", False))
    strict_structure = bool(filters.get("strict_structure", False))
    # Strict-only policy (no relaxed/off-topic fallback).
    allow_offtopic_docs = 0
    strict_section_only = bool(filters.get("strict_section_only", True))
    target_section_id = str(filters.get("target_section_id", "")).strip()
    if not target_section_id:
        target_section_id = _select_most_specific_section_id(topic_hint)
    query_profile = _build_query_anchor_profile_local(query_text)

    enriched = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        topic_match = _doc_topic_match_local(doc, topic_hint)
        structure_match = _doc_has_structure_local(doc)
        query_anchor_match = _doc_query_anchor_match_local(doc, query_profile)
        meta["topic_match"] = bool(topic_match)
        meta["structure_match"] = bool(structure_match)
        meta["query_anchor_match"] = bool(query_anchor_match)
        doc.metadata = meta
        enriched.append(doc)

    ordered = list(enriched)
    if topic_hint and topic_hint.lower() != "other":
        topic_docs = [d for d in ordered if d.metadata.get("topic_match")]
        off_topic_docs = [d for d in ordered if not d.metadata.get("topic_match")]
        if topic_docs:
            ordered = topic_docs + off_topic_docs[:max(0, allow_offtopic_docs)]
        elif strict_topic and not target_section_id:
            # If section is known, defer strictness to section/page gate.
            return []

    if target_section_id:
        section_docs = [d for d in ordered if _doc_section_match_local(d, target_section_id)]
        non_section_docs = [d for d in ordered if not _doc_section_match_local(d, target_section_id)]
        if section_docs:
            ordered = section_docs + non_section_docs[:max(0, allow_offtopic_docs)]
        elif strict_section_only:
            return []

    if not target_section_id:
        exact_query_docs = [d for d in ordered if _doc_query_phrase_match_local(d, query_text)]
        if exact_query_docs:
            non_exact_docs = [d for d in ordered if not _doc_query_phrase_match_local(d, query_text)]
            ordered = exact_query_docs + non_exact_docs[:max(0, allow_offtopic_docs)]

    if query_profile.get("enabled", False):
        anchor_docs = [d for d in ordered if d.metadata.get("query_anchor_match")]
        non_anchor_docs = [d for d in ordered if not d.metadata.get("query_anchor_match")]
        if anchor_docs:
            ordered = anchor_docs + non_anchor_docs[:max(0, allow_offtopic_docs)]

    if require_structure:
        structure_docs = [d for d in ordered if d.metadata.get("structure_match")]
        non_structure_docs = [d for d in ordered if not d.metadata.get("structure_match")]
        if structure_docs:
            ordered = structure_docs + non_structure_docs[:max(0, allow_offtopic_docs)]
        elif strict_structure:
            return []

    return ordered


SECURITY_POLICY_COLORS = {
    "strict": "#198754",
    "relaxed": "#fd7e14",
    "disabled": "#6c757d",
    "unknown": "#6c757d",
}


def _security_policy_from_status(status: dict) -> str:
    if not status:
        return "unknown"
    if not status.get("index_verify_enabled", False):
        return "disabled"
    if status.get("index_verify_strict", False):
        return "strict"
    return "relaxed"


def run_startup_security_checks(rag_obj, rag_status: dict) -> dict:
    policy = _security_policy_from_status(rag_status)
    index_dir = rag_status.get("index_dir", "")
    manifest_path = rag_status.get("index_manifest", "")
    manifest_exists = Path(manifest_path).exists() if manifest_path else False
    integrity_ok = rag_status.get("index_integrity_ok")
    integrity_error = rag_status.get("index_integrity_error", "")

    issues = []
    if rag_obj is None:
        issues.append("rag_unavailable")
    if policy != "disabled" and not manifest_exists:
        issues.append("manifest_missing")
    if policy == "disabled":
        issues.append("checksum_disabled")

    # Re-verify checksums at call-time so manual re-run reflects current disk state.
    if policy != "disabled" and manifest_exists and verify_index_manifest and index_dir:
        try:
            verify_ok, verify_errors = verify_index_manifest(Path(index_dir), Path(manifest_path))
            integrity_ok = verify_ok
            integrity_error = ", ".join(verify_errors)
        except Exception as exc:
            integrity_ok = False
            integrity_error = str(exc)

    if integrity_ok is False:
        issues.append("integrity_failed")

    if rag_obj is None or integrity_ok is False:
        result = "fail"
    elif policy in {"relaxed", "disabled"}:
        result = "warn"
    else:
        result = "pass"

    return {
        "policy": policy,
        "result": result,
        "manifest_path": manifest_path,
        "manifest_exists": manifest_exists,
        "integrity_ok": integrity_ok,
        "integrity_error": integrity_error,
        "issues": issues,
    }


def render_security_policy_badge(policy: str) -> str:
    color = SECURITY_POLICY_COLORS.get(policy, SECURITY_POLICY_COLORS["unknown"])
    label = policy.upper()
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:999px;font-size:12px;font-weight:600;">{label}</span>'
    )


def save_env_var_to_dotenv(key: str, value: str, dotenv_path: Path = DOTENV_PATH) -> tuple[bool, str]:
    key = str(key or "").strip()
    if not key:
        return False, "missing_key"
    target = str(value or "").strip()

    try:
        lines = []
        if dotenv_path.exists():
            lines = dotenv_path.read_text(encoding="utf-8").splitlines()

        patt = re.compile(rf"^\s*{re.escape(key)}\s*=")
        updated = False
        new_lines = []
        for line in lines:
            if patt.match(line):
                new_lines.append(f"{key}={target}")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"{key}={target}")

        text = "\n".join(new_lines).rstrip() + "\n"
        dotenv_path.write_text(text, encoding="utf-8")
        os.environ[key] = target
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def read_dotenv_value(key: str, dotenv_path: Path = DOTENV_PATH) -> str:
    k = str(key or "").strip()
    if not k or not dotenv_path.exists():
        return ""
    try:
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            lhs, rhs = raw.split("=", 1)
            if lhs.strip() == k:
                return rhs.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def _convert_hf_space_id_to_url(value: str) -> str:
    """Convert Hugging Face Space ID (user/space) to full URL (user-space.hf.space)"""
    value = value.strip()
    if not value:
        return ""
    # If already a full URL, return as-is
    if value.startswith("http://") or value.startswith("https://"):
        return value
    # If it's a Space ID format (user/space-name), convert to URL
    # Hugging Face Spaces convert both '/' and '_' to '-' in the URL
    if "/" in value and "." not in value:
        space_path = value.replace('/', '-').replace('_', '-')
        return f"https://{space_path}.hf.space"
    return value

def resolve_colpali_endpoint_url(preferred: str = "") -> str:
    # ลำดับ priority: preferred > st.secrets > .env > global var
    # ใช้ try-except เพื่อรองรับ local dev ที่ไม่มี secrets.toml
    try:
        secrets_endpoint = (
            st.secrets.get("colpali", {}).get("endpoint_url")
            or st.secrets.get("colpali", {}).get("space_id")
            or st.secrets.get("COLPALI_ENDPOINT_URL")
        )
    except Exception:
        secrets_endpoint = None
    
    raw = (
        str(preferred or "").strip()
        or secrets_endpoint
        or read_dotenv_value("COLPALI_ENDPOINT_URL", DOTENV_PATH)
        or os.getenv("COLPALI_ENDPOINT_URL", "").strip()
        or VISUAL_RETRIEVAL_ENDPOINT_URL
    )
    return _convert_hf_space_id_to_url(raw)


def build_visual_endpoint_contract_payload(query: str = "โครงสร้างของการแทนคิวด้วยอาร์เรย์") -> dict:
    sample_path = PROJECT_ROOT / "assets" / "figure_regions" / "page_031_region_01.png"
    if sample_path.exists():
        try:
            import base64
            img_b64 = base64.b64encode(sample_path.read_bytes()).decode("ascii")
        except Exception:
            img_b64 = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7f0l8AAAAASUVORK5CYII="
            )
    else:
        img_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7f0l8AAAAASUVORK5CYII="
        )
    candidate = {
        "id": "doc.pdf:31:region:1",
        "page_id": "doc.pdf:31",
        "source": "doc.pdf",
        "page": 31,
        "image_path": "assets/figure_regions/page_031_region_01.png",
        "image_base64": img_b64,
        "text_preview": "ภาพที่ 3.4 การนำเข้าและนำออกข้อมูลในโครงสร้างคิว",
    }
    return {
        "corpus_id": "healthcheck-corpus-v1",
        "query": query,
        "candidate_ids": [candidate["id"]],
        "candidates": [candidate],
    }


def validate_visual_endpoint_contract_response(data: dict, expected_candidate_ids: list[str] | None = None) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "response_not_object"
    endpoint_error = str(data.get("error", "") or "").strip()
    if endpoint_error:
        return False, f"endpoint_error:{endpoint_error[:240]}"
    scores = data.get("scores")
    if not isinstance(scores, list):
        return False, "missing_scores_list"
    if len(scores) == 0:
        return False, "empty_scores"
    expected = {str(x).strip() for x in (expected_candidate_ids or []) if str(x).strip()}
    seen = set()
    for i, row in enumerate(scores):
        if not isinstance(row, dict):
            return False, f"scores[{i}]_not_object"
        if "id" not in row or "score" not in row:
            return False, f"scores[{i}]_missing_id_or_score"
        rid = str(row.get("id", "")).strip()
        if not rid:
            return False, f"scores[{i}]_id_empty"
        seen.add(rid)
        try:
            _ = float(row.get("score"))
        except Exception:
            return False, f"scores[{i}]_score_not_numeric"
    if expected and not seen.intersection(expected):
        return False, "id_mismatch_no_expected_candidates"
    return True, "ok"


def run_visual_endpoint_healthcheck(endpoint_url: str, api_key: str, timeout_sec: int = 30) -> dict:
    endpoint = (endpoint_url or "").strip()
    payload = build_visual_endpoint_contract_payload()
    compact_contract = {
        "query": payload.get("query", ""),
        "candidates": [
            {
                **{k: v for k, v in (payload.get("candidates", [{}])[0] or {}).items() if k != "image_base64"},
                "image_base64": "<omitted>" if (payload.get("candidates", [{}])[0] or {}).get("image_base64") else "",
            }
        ],
    }
    if not endpoint:
        return {
            "ok": False,
            "endpoint_url": "",
            "error": "missing_endpoint_url",
            "contract": compact_contract,
        }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _extract_scores_payload(data_obj):
        if isinstance(data_obj, dict):
            return data_obj
        if isinstance(data_obj, str):
            try:
                parsed = json.loads(data_obj)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        if isinstance(data_obj, (list, tuple)) and data_obj:
            for item in data_obj:
                parsed = _extract_scores_payload(item)
                if isinstance(parsed, dict):
                    return parsed
        return None

    started = time.perf_counter()
    data = None
    last_err = ""
    status_code = None
    is_http_endpoint = endpoint.lower().startswith(("http://", "https://"))
    if is_http_endpoint:
        try:
            with httpx.Client(timeout=max(5, int(timeout_sec))) as client:
                res = client.post(endpoint, json=payload, headers=headers)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                body_text = res.text[:2000]
                status_code = int(res.status_code)
                if res.status_code >= 400:
                    last_err = f"http_error:{res.status_code}:{body_text[:240]}"
                else:
                    data = res.json()
        except Exception as exc:
            last_err = f"request_failed:{exc}"

    if data is None:
        try:
            from gradio_client import Client

            # gradio_client 2.x+ uses hf_token parameter directly
            # Convert full HF URL to space name if needed
            space_name = endpoint
            if "huggingface.co/spaces/" in endpoint:
                # Extract space name from URL like "https://huggingface.co/spaces/user/name"
                space_name = endpoint.split("huggingface.co/spaces/")[-1].strip("/")
            elif endpoint.startswith("http"):
                # Direct Gradio app URL - use as is
                space_name = endpoint

            client_kwargs = {"verbose": False}
            if api_key:
                client_kwargs["token"] = api_key

            client = Client(space_name, **client_kwargs)
            for api_name in ("/register_and_score", "/score", "/api_endpoint"):
                try:
                    out = client.predict(payload, api_name=api_name)
                    parsed = _extract_scores_payload(out)
                    if isinstance(parsed, dict):
                        data = parsed
                        break
                except Exception:
                    try:
                        out = client.predict(json.dumps(payload, ensure_ascii=False), api_name=api_name)
                        parsed = _extract_scores_payload(out)
                        if isinstance(parsed, dict):
                            data = parsed
                            break
                    except Exception:
                        continue
        except Exception as exc:
            if last_err:
                last_err = f"{last_err} | gradio_client:{exc}"
            else:
                last_err = f"gradio_client:{exc}"

    if data is None:
        result = {
            "ok": False,
            "endpoint_url": endpoint,
            "status_code": status_code,
            "error": last_err or "healthcheck_failed",
            "contract": compact_contract,
        }
        VISUAL_ENDPOINT_HEALTHCHECK_LOG.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    expected_ids = [str(x).strip() for x in (payload.get("candidate_ids", []) or []) if str(x).strip()]
    ok, reason = validate_visual_endpoint_contract_response(data, expected_candidate_ids=expected_ids)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    result = {
        "ok": bool(ok),
        "endpoint_url": endpoint,
        "status_code": status_code or 200,
        "latency_ms": elapsed_ms,
        "validation": reason,
        "response_preview": data.get("scores", [])[:3] if isinstance(data, dict) else [],
        "response_error": data.get("error", "") if isinstance(data, dict) else "",
        "contract": compact_contract,
    }
    VISUAL_ENDPOINT_HEALTHCHECK_LOG.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_visual_retrieval_query(
    query: str,
    *,
    require_structure: bool,
    backend: str,
    endpoint_url: str,
    use_vlm_rerank: bool,
    use_visual_grounding: bool,
    sparse_strategy_override: str | None = None,
    top_k_override: int | None = None,
    candidate_k_override: int | None = None,
    topic_threshold_override: float | None = None,
    ground_top_n_override: int | None = None,
) -> dict:
    if not VISUAL_RETRIEVAL_SCRIPT.exists():
        return {"ok": False, "error": f"visual_script_not_found:{VISUAL_RETRIEVAL_SCRIPT}"}
    endpoint_effective = resolve_colpali_endpoint_url(endpoint_url)
    backend = FORCED_VISUAL_BACKEND
    # Fast/robust default path: force BM25 sparse channel in this phase.
    sparse_strategy = "bm25"

    cmd = [
        sys.executable,
        str(VISUAL_RETRIEVAL_SCRIPT),
        "--query",
        query,
        "--backend",
        backend,
        "--top-k",
        str(max(1, int(top_k_override if isinstance(top_k_override, int) and top_k_override > 0 else VISUAL_RETRIEVAL_TOP_K))),
        "--candidate-k",
        str(
            max(
                2,
                int(candidate_k_override if isinstance(candidate_k_override, int) and candidate_k_override > 0 else VISUAL_RETRIEVAL_CANDIDATE_K),
            )
        ),
        "--topic-threshold",
        str(
            float(topic_threshold_override)
            if isinstance(topic_threshold_override, (int, float)) else VISUAL_RETRIEVAL_TOPIC_THRESHOLD
        ),
        "--topic-adjacent-page-gap",
        str(max(0, int(VISUAL_TOPIC_ADJACENT_PAGE_GAP))),
        "--topic-adjacent-max-extra-pages",
        str(max(0, int(VISUAL_TOPIC_ADJACENT_MAX_EXTRA_PAGES))),
        "--prefilter-min-records",
        str(max(6, int(VISUAL_PREFILTER_MIN_RECORDS))),
        "--prefilter-rescue-topn",
        str(max(VISUAL_PREFILTER_MIN_RECORDS, int(VISUAL_PREFILTER_RESCUE_TOPN))),
        "--vlm-model",
        VISUAL_RETRIEVAL_VLM_MODEL,
        "--endpoint-timeout-sec",
        str(max(10, int(VISUAL_RETRIEVAL_ENDPOINT_TIMEOUT_SEC))),
        "--endpoint-max-retries",
        str(max(0, int(VISUAL_ENDPOINT_MAX_RETRIES))),
        "--endpoint-retry-backoff-ms",
        str(max(0, int(VISUAL_ENDPOINT_RETRY_BACKOFF_MS))),
        "--endpoint-local-cache-size",
        str(max(16, int(VISUAL_ENDPOINT_LOCAL_CACHE_SIZE))),
        "--endpoint-image-max-side",
        str(max(256, int(VISUAL_ENDPOINT_IMAGE_MAX_SIDE))),
        "--endpoint-jpeg-quality",
        str(max(40, min(95, int(VISUAL_ENDPOINT_JPEG_QUALITY)))),
        "--sparse-strategy",
        sparse_strategy,
        "--output",
        str(VISUAL_RETRIEVAL_RUNTIME_OUTPUT),
    ]
    if sparse_strategy == "splade" and VISUAL_RETRIEVAL_USE_SPLADE:
        cmd.append("--use-splade")
    cmd.extend(["--splade-mode", VISUAL_RETRIEVAL_SPLADE_MODE or "auto"])
    cmd.extend(["--splade-provider", VISUAL_RETRIEVAL_SPLADE_PROVIDER or "hf-inference"])
    cmd.extend(["--splade-model", VISUAL_RETRIEVAL_SPLADE_MODEL])
    cmd.extend(["--splade-device", VISUAL_RETRIEVAL_SPLADE_DEVICE or "auto"])
    cmd.extend(["--splade-max-length", str(max(32, int(VISUAL_RETRIEVAL_SPLADE_MAX_LENGTH)))])
    cmd.extend(["--splade-batch-size", str(max(1, int(VISUAL_RETRIEVAL_SPLADE_BATCH_SIZE)))])
    if VISUAL_INTENT_USE_LLM:
        cmd.append("--use-intent-llm")
    cmd.extend(["--intent-candidate-top", str(max(3, int(VISUAL_INTENT_CANDIDATE_TOP)))])
    cmd.extend(["--intent-min-confidence", str(float(VISUAL_INTENT_MIN_CONFIDENCE))])
    cmd.extend(["--intent-max-topic-ids", str(max(1, int(VISUAL_INTENT_MAX_TOPIC_IDS)))])
    if VISUAL_QUERY_EXPANSION_USE_LLM:
        cmd.append("--use-query-expansion-llm")
    cmd.extend(["--query-expansion-model", VISUAL_QUERY_EXPANSION_MODEL])
    cmd.extend(["--query-expansion-max-terms", str(max(0, int(VISUAL_QUERY_EXPANSION_MAX_TERMS)))])
    cmd.append("--metadata-filter-strict")
    if require_structure:
        cmd.append("--require-structure")
    if use_vlm_rerank:
        cmd.append("--use-vlm-rerank")
        cmd.extend(["--vlm-rerank-top-m", str(max(4, int(VISUAL_VLM_RERANK_TOP_M)))])
    if use_visual_grounding:
        cmd.append("--use-visual-grounding")
        ground_n = max(
            1,
            int(
                ground_top_n_override
                if isinstance(ground_top_n_override, int) and ground_top_n_override > 0
                else VISUAL_GROUND_TOP_N
            ),
        )
        cmd.extend(["--ground-top-n", str(ground_n)])
        cmd.extend(["--grounding-ensemble-runs", str(max(1, int(VISUAL_GROUNDING_ENSEMBLE_RUNS)))])
        cmd.extend(["--grounding-consensus-min-votes", str(max(1, int(VISUAL_GROUNDING_CONSENSUS_MIN_VOTES)))])
        cmd.extend(["--grounding-temperature", str(float(VISUAL_GROUNDING_TEMPERATURE))])
        cmd.extend(["--grounding-top-p", str(float(VISUAL_GROUNDING_TOP_P))])
    if VISUAL_HARD_NEGATIVE_RULES:
        cmd.extend(["--hard-negative-rules", VISUAL_HARD_NEGATIVE_RULES])
        cmd.extend(["--hard-negative-penalty-max", str(float(VISUAL_HARD_NEGATIVE_PENALTY_MAX))])
    if not VISUAL_COVERAGE_RERANK_ENABLED:
        cmd.append("--disable-coverage-rerank")
    if VISUAL_RETRIEVAL_ALLOW_CPU_ENGINE:
        cmd.append("--colpali-engine-allow-cpu")
    if endpoint_effective.strip():
        cmd.extend(["--colpali-endpoint-url", endpoint_effective.strip()])

    env = os.environ.copy()
    env["HUGGINGFACE_API_KEY"] = HF_TOKEN
    env["VISUAL_USE_SPLADE"] = "0"
    env["VISUAL_SPARSE_STRATEGY"] = sparse_strategy
    env["VISUAL_SPLADE_MODE"] = VISUAL_RETRIEVAL_SPLADE_MODE or "auto"
    env["VISUAL_SPLADE_PROVIDER"] = VISUAL_RETRIEVAL_SPLADE_PROVIDER or "hf-inference"
    env["VISUAL_SPLADE_MODEL"] = VISUAL_RETRIEVAL_SPLADE_MODEL
    env["VISUAL_METADATA_FILTER_STRICT"] = "1" if VISUAL_METADATA_FILTER_STRICT else "0"
    env["VISUAL_INTENT_USE_LLM"] = "1" if VISUAL_INTENT_USE_LLM else "0"
    env["VISUAL_INTENT_CANDIDATE_TOP"] = str(max(3, int(VISUAL_INTENT_CANDIDATE_TOP)))
    env["VISUAL_INTENT_MIN_CONFIDENCE"] = str(float(VISUAL_INTENT_MIN_CONFIDENCE))
    env["VISUAL_INTENT_MAX_TOPIC_IDS"] = str(max(1, int(VISUAL_INTENT_MAX_TOPIC_IDS)))
    env["VISUAL_QUERY_EXPANSION_USE_LLM"] = "1" if VISUAL_QUERY_EXPANSION_USE_LLM else "0"
    env["VISUAL_QUERY_EXPANSION_MODEL"] = VISUAL_QUERY_EXPANSION_MODEL
    env["VISUAL_QUERY_EXPANSION_MAX_TERMS"] = str(max(0, int(VISUAL_QUERY_EXPANSION_MAX_TERMS)))
    env["VISUAL_PREFILTER_MIN_RECORDS"] = str(max(6, int(VISUAL_PREFILTER_MIN_RECORDS)))
    env["VISUAL_PREFILTER_RESCUE_TOPN"] = str(max(VISUAL_PREFILTER_MIN_RECORDS, int(VISUAL_PREFILTER_RESCUE_TOPN)))
    env["VISUAL_SPLADE_DEVICE"] = VISUAL_RETRIEVAL_SPLADE_DEVICE or "auto"
    env["VISUAL_SPLADE_MAX_LENGTH"] = str(max(32, int(VISUAL_RETRIEVAL_SPLADE_MAX_LENGTH)))
    env["VISUAL_SPLADE_BATCH_SIZE"] = str(max(1, int(VISUAL_RETRIEVAL_SPLADE_BATCH_SIZE)))
    env["VISUAL_VLM_RERANK_TOP_M"] = str(max(4, int(VISUAL_VLM_RERANK_TOP_M)))
    if endpoint_effective.strip():
        env["COLPALI_ENDPOINT_URL"] = endpoint_effective.strip()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
    except Exception as exc:
        return {"ok": False, "error": f"visual_subprocess_failed:{exc}"}

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"visual_subprocess_nonzero:{proc.returncode}",
            "stderr_tail": (proc.stderr or "")[-2000:],
            "stdout_tail": (proc.stdout or "")[-2000:],
        }
    if not VISUAL_RETRIEVAL_RUNTIME_OUTPUT.exists():
        return {"ok": False, "error": "visual_output_missing"}

    try:
        payload = json.loads(VISUAL_RETRIEVAL_RUNTIME_OUTPUT.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"visual_output_parse_failed:{exc}"}

    return {
        "ok": True,
        "payload": payload,
        "stdout_tail": (proc.stdout or "")[-1200:],
        "stderr_tail": (proc.stderr or "")[-1200:],
    }


def run_visual_standard_query_suite(
    *,
    backend: str,
    endpoint_url: str,
    use_vlm_rerank: bool,
    use_visual_grounding: bool,
    sparse_strategy: str,
) -> dict:
    backend = FORCED_VISUAL_BACKEND
    endpoint_effective = resolve_colpali_endpoint_url(endpoint_url)
    results = []
    def _mean_num(vals):
        nums = [float(v) for v in vals if isinstance(v, (int, float))]
        return round(float(sum(nums) / len(nums)), 4) if nums else None

    for case in VISUAL_STANDARD_QUERY_CASES:
        q = str(case.get("query", "")).strip()
        req_struct = bool(case.get("require_structure", False))
        started = time.perf_counter()
        run_res = run_visual_retrieval_query(
            q,
            require_structure=req_struct,
            backend=backend,
            endpoint_url=endpoint_effective,
            use_vlm_rerank=use_vlm_rerank,
            use_visual_grounding=use_visual_grounding,
            sparse_strategy_override=sparse_strategy,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        if not run_res.get("ok", False):
            results.append(
                {
                    "query": q,
                    "pass": False,
                    "error": str(run_res.get("error", "unknown")),
                    "latency_ms": elapsed_ms,
                }
            )
            continue

        payload = run_res.get("payload", {}) or {}
        task_metrics = payload.get("task_metrics", {}) if isinstance(payload, dict) else {}
        topic_prediction = payload.get("topic_prediction", {}) if isinstance(payload, dict) else {}
        topic_id = str(topic_prediction.get("topic_id", "")).strip()
        chapter_profile = resolve_chapter_calibration(topic_id)
        sources_data, _, evidence = build_visual_context_and_sources(
            payload,
            topic_hint=None,
            require_structure=req_struct,
            calibration=chapter_profile,
        )
        has_hits = len(sources_data) > 0
        evidence_ok = bool(evidence.get("evidence_ok", False))
        endpoint_backend_ok = True
        reasons = list(evidence.get("reasons", []))
        if backend == "colpali_endpoint":
            endpoint_backend_ok = str(payload.get("colpali_status", "")).strip().lower() == "ok"
            if not endpoint_backend_ok:
                reasons.append("endpoint_not_ready")
        case_pass = has_hits and evidence_ok and endpoint_backend_ok
        results.append(
            {
                "query": q,
                "pass": case_pass,
                "latency_ms": elapsed_ms,
                "backend": payload.get("backend"),
                "colpali_status": payload.get("colpali_status"),
                "hits": len(sources_data),
                "evidence_ok": evidence_ok,
                "reasons": reasons,
                "region_hit_ratio_at_k": task_metrics.get("region_hit_ratio_at_k"),
                "crop_completeness_proxy": task_metrics.get("crop_completeness_proxy"),
                "small_region_ratio_at_k": task_metrics.get("small_region_ratio_at_k"),
                "region_quality_mean": task_metrics.get("region_quality_mean"),
                "unique_figure_refs_at_k": task_metrics.get("unique_figure_refs_at_k"),
                "page_diversity_at_k": task_metrics.get("page_diversity_at_k"),
                "procedure_step_coverage_proxy": task_metrics.get("procedure_step_coverage_proxy"),
            }
        )

    total = len(results)
    passed = sum(1 for r in results if bool(r.get("pass", False)))
    report = {
        "ok": total > 0 and passed == total,
        "summary": {
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "pass_rate": round((passed / total), 4) if total else 0.0,
            "region_hit_ratio_at_k_mean": _mean_num([r.get("region_hit_ratio_at_k") for r in results]),
            "crop_completeness_proxy_mean": _mean_num([r.get("crop_completeness_proxy") for r in results]),
            "small_region_ratio_at_k_mean": _mean_num([r.get("small_region_ratio_at_k") for r in results]),
            "region_quality_mean": _mean_num([r.get("region_quality_mean") for r in results]),
            "unique_figure_refs_at_k_mean": _mean_num([r.get("unique_figure_refs_at_k") for r in results]),
            "page_diversity_at_k_mean": _mean_num([r.get("page_diversity_at_k") for r in results]),
            "procedure_step_coverage_proxy_mean": _mean_num([r.get("procedure_step_coverage_proxy") for r in results]),
        },
        "config": {
            "backend": backend,
            "endpoint_url": endpoint_effective,
            "sparse_strategy": str(sparse_strategy),
            "use_vlm_rerank": bool(use_vlm_rerank),
            "use_visual_grounding": bool(use_visual_grounding),
        },
        "results": results,
    }
    VISUAL_STANDARD_QUERY_LOG.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_visual_context_and_sources(
    visual_payload: dict,
    *,
    topic_hint: str | None,
    require_structure: bool,
    calibration: dict | None = None,
) -> tuple[list[dict], str, dict]:
    hits = visual_payload.get("hits", []) if isinstance(visual_payload, dict) else []
    grounding_list = visual_payload.get("grounding", []) if isinstance(visual_payload, dict) else []
    backend_name = str(visual_payload.get("backend", "")).strip().lower() if isinstance(visual_payload, dict) else ""
    colpali_score_ready = bool(visual_payload.get("colpali_score_ready", False)) if isinstance(visual_payload, dict) else False
    grounded_by_id = {}
    for row in grounding_list:
        rid = str(row.get("id", "")).strip()
        if rid:
            grounded_by_id[rid] = row.get("grounding", {})
    query_profile = visual_payload.get("query_profile", {}) if isinstance(visual_payload, dict) else {}
    topic_prediction = visual_payload.get("topic_prediction", {}) if isinstance(visual_payload, dict) else {}
    topic_id = str(topic_prediction.get("topic_id", "")).strip()
    op_intent = bool(query_profile.get("operation_intent"))
    region_fig_hits = [
        h for h in hits
        if str(h.get("image_level", "")).strip().lower() == "region"
        and bool(h.get("figure_refs", []))
    ]
    prefer_region_only = len(region_fig_hits) >= 2

    def _rank_num(hit: dict) -> int:
        try:
            return int(hit.get("rank", 9999) or 9999)
        except Exception:
            return 9999

    def _page_num(hit: dict) -> int:
        try:
            return int(hit.get("page", 0) or 0)
        except Exception:
            return 0

    ordered_hits = sorted(
        hits,
        key=lambda h: (
            0 if str(h.get("image_level", "")).strip().lower() == "region" else 1,
            0 if bool(h.get("figure_refs", [])) else 1,
            _rank_num(h),
            _first_figure_ref_num(h),
            _page_num(h),
        ),
    )

    profile = query_profile if isinstance(query_profile, dict) else {}
    op_terms = [str(x).strip().lower() for x in (profile.get("operation_terms", []) or []) if str(x).strip()]
    struct_terms = [str(x).strip().lower() for x in (profile.get("structure_terms", []) or []) if str(x).strip()]
    generic_terms = {"การ", "ทำงาน", "ดำเนินการ", "โครงสร้าง", "แบบ", "ข้อมูล"}
    op_focus_terms = [t for t in op_terms if len(t) >= 3 and t not in generic_terms]
    struct_focus_terms = [t for t in struct_terms if len(t) >= 3 and t not in generic_terms]
    focus_terms = op_focus_terms + struct_focus_terms
    if focus_terms:
        def _hit_term_aligned(hit: dict) -> bool:
            preview_t = str(hit.get("preview", "")).lower()
            refs_t = " ".join(str(x).lower() for x in (hit.get("figure_refs", []) or []))
            tags_t = " ".join(str(x).lower() for x in (hit.get("tags", []) or []))
            labels_t = " ".join(str(x).lower() for x in (hit.get("structure_labels", []) or []))
            blob = f"{preview_t} {refs_t} {tags_t} {labels_t}"
            if op_focus_terms:
                return any(t in blob for t in op_focus_terms)
            return any(t in blob for t in struct_focus_terms)

        aligned_hits = [h for h in ordered_hits if _hit_term_aligned(h)]
        # Do not hard-drop the tail; keep aligned hits first to avoid collapsing to single-doc context.
        if aligned_hits:
            aligned_ids = {str(h.get("id", "")).strip() for h in aligned_hits}
            tail_hits = [h for h in ordered_hits if str(h.get("id", "")).strip() not in aligned_ids]
            ordered_hits = aligned_hits + tail_hits

    if prefer_region_only:
        region_only = [h for h in ordered_hits if str(h.get("image_level", "")).strip().lower() == "region"]
        if region_only:
            ordered_hits = region_only

    if require_structure:
        region_hits = [h for h in ordered_hits if str(h.get("image_level", "")).strip().lower() == "region"]
        if region_hits:
            ordered_hits = region_hits
    if op_intent:
        # Keep retrieval confidence priority for operation questions; do not globally sort by page number.
        ordered_hits = sorted(
            ordered_hits,
            key=lambda h: (
                0 if str(h.get("image_level", "")).strip().lower() == "region" else 1,
                0 if bool(h.get("figure_refs", [])) else 1,
                _rank_num(h),
            ),
        )
    doc_limit = max(1, int(CONTEXT_DOC_LIMIT))
    if op_intent:
        doc_limit = max(doc_limit, int(VISUAL_CONTEXT_DOC_LIMIT_OPERATION))
        ordered_hits = stitch_cross_page_operation_hits(
            ordered_hits,
            query_profile=query_profile if isinstance(query_profile, dict) else {},
            topic_id=topic_id,
            doc_limit=doc_limit,
        )

    calib = calibration if isinstance(calibration, dict) else {}
    sources_data = []
    context_lines = []
    for i, hit in enumerate(ordered_hits[:doc_limit], start=1):
        source = str(hit.get("source", "reference_document")).strip()
        page = str(hit.get("page", "")).strip()
        citation = source + (f":{page}" if source and page else "")
        hit_id = str(hit.get("id", f"visual-hit-{i}")).strip()
        preview = sanitize_visual_preview(str(hit.get("preview", "")).strip())
        grounding = grounded_by_id.get(hit_id, {})
        grounded = grounding.get("parsed", {}) if isinstance(grounding, dict) else {}
        consensus = grounding.get("consensus", {}) if isinstance(grounding, dict) else {}
        grounded_facts_raw = grounded.get("grounded_facts", [])
        if not isinstance(grounded_facts_raw, list):
            grounded_facts_raw = []
        grounded_facts_raw = [str(x).strip() for x in grounded_facts_raw if str(x).strip()]
        grounded_facts = filter_grounded_facts_by_preview(grounded_facts_raw, preview)
        consensus_agreement = (
            float(consensus.get("agreement_ratio"))
            if isinstance(consensus, dict) and isinstance(consensus.get("agreement_ratio"), (int, float))
            else None
        )
        facts_text = "; ".join(grounded_facts[:3])
        content = preview[:700]
        if facts_text:
            content = f"{content}\nVisual facts: {facts_text}"

        structure_labels = hit.get("structure_labels", [])
        structure_match = bool(structure_labels) or bool(require_structure)
        final_score = hit.get("final_score", hit.get("base_score"))
        base_score = hit.get("base_score")
        score_for_evidence = final_score
        if not colpali_score_ready and isinstance(base_score, (int, float)):
            if isinstance(final_score, (int, float)):
                score_for_evidence = max(float(final_score), float(base_score))
            else:
                score_for_evidence = float(base_score)
        sources_data.append(
            {
                "source": source,
                "page": page,
                "retrieval_rank": _rank_num(hit),
                "chunk_id": hit_id,
                "citation": citation,
                "rrf_score": final_score,
                "score_for_evidence": score_for_evidence,
                "topic_match": True,
                "structure_match": structure_match,
                "query_align_score": float(hit.get("operation_score", 0.0) or 0.0)
                + float(hit.get("structure_score", 0.0) or 0.0)
                + float(hit.get("figure_score", 0.0) or 0.0),
                "content": content,
                "source_excerpt": sanitize_source_excerpt(facts_text or content),
                "image_path": str(hit.get("image_path", "")).strip(),
                "image_level": str(hit.get("image_level", "")).strip(),
                "figure_refs": hit.get("figure_refs", []),
                "region_meta": hit.get("region_meta", {}) if isinstance(hit.get("region_meta", {}), dict) else {},
                "region_quality_score": hit.get("region_quality_score"),
                "grounded_fact_count": len(grounded_facts),
                "grounded_facts": grounded_facts[:6],
                "grounded_fact_count_raw": len(grounded_facts_raw),
                "grounding_agreement": consensus_agreement,
            }
        )
        context_lines.append(f"Context {i} [{citation} | {hit_id}]: {content}")

    doc_count = len(sources_data)
    structure_count = sum(1 for s in sources_data if bool(s.get("structure_match")))
    page_nums = []
    for s in sources_data:
        try:
            pnum = int(s.get("page", 0) or 0)
            if pnum > 0:
                page_nums.append(pnum)
        except Exception:
            continue
    page_span = (max(page_nums) - min(page_nums)) if page_nums else 0
    unique_ref_nums = set()
    for s in sources_data:
        for fr in s.get("figure_refs", []) or []:
            for n in re.findall(r"(\d+(?:\.\d+)?)", str(fr)):
                unique_ref_nums.add(n)
    score_values = []
    for s in sources_data:
        score = s.get("score_for_evidence")
        if isinstance(score, (int, float)):
            score_values.append(float(score))
    avg_score = sum(score_values) / len(score_values) if score_values else 0.0
    grounding_doc_count = sum(1 for s in sources_data if int(s.get("grounded_fact_count", 0) or 0) > 0)
    grounding_fact_count = sum(int(s.get("grounded_fact_count", 0) or 0) for s in sources_data)
    grounding_doc_count_raw = sum(1 for s in sources_data if int(s.get("grounded_fact_count_raw", 0) or 0) > 0)
    grounding_fact_count_raw = sum(int(s.get("grounded_fact_count_raw", 0) or 0) for s in sources_data)
    grounding_agreement_vals = [float(s.get("grounding_agreement")) for s in sources_data if isinstance(s.get("grounding_agreement"), (int, float))]
    grounding_agreement_mean = (sum(grounding_agreement_vals) / len(grounding_agreement_vals)) if grounding_agreement_vals else None

    topic_ratio = 1.0 if topic_prediction.get("topic_filtered") else 0.0
    reasons = []
    min_doc_count = max(1, int(calib.get("min_doc_count", EVIDENCE_MIN_DOCS) or EVIDENCE_MIN_DOCS))
    if doc_count < min_doc_count:
        if not (op_intent and len(unique_ref_nums) >= 2):
            reasons.append("low_doc_count")
    if topic_hint and topic_ratio < EVIDENCE_MIN_TOPIC_MATCH_RATIO:
        reasons.append("low_topic_match")
    if require_structure and structure_count < 1:
        reasons.append("missing_structure_evidence")
    visual_score_threshold = 0.2 if colpali_score_ready else 0.08
    if backend_name == "metadata":
        visual_score_threshold = min(visual_score_threshold, 0.08)
    threshold_scale = float(calib.get("visual_score_threshold_scale", 1.0) or 1.0)
    threshold_scale = max(0.5, min(1.4, threshold_scale))
    visual_score_threshold = visual_score_threshold * threshold_scale
    if score_values and avg_score < visual_score_threshold:
        payload_mean_final = None
        payload_step_cov = None
        if isinstance(visual_payload, dict):
            tm = visual_payload.get("task_metrics", {})
            if isinstance(tm, dict):
                mf = tm.get("mean_final_score")
                if isinstance(mf, (int, float)):
                    payload_mean_final = float(mf)
                psc = tm.get("procedure_step_coverage_proxy")
                if isinstance(psc, (int, float)):
                    payload_step_cov = float(psc)

        strong_structure_signal = bool(require_structure and structure_count >= 1 and len(unique_ref_nums) >= 1)
        strong_grounding_signal = bool(grounding_doc_count_raw >= 1 and grounding_fact_count_raw >= 2)
        op_relax = bool(op_intent and len(unique_ref_nums) >= 2 and avg_score >= (visual_score_threshold * 0.55))
        payload_relax = bool(
            payload_mean_final is not None
            and payload_mean_final >= (visual_score_threshold * 0.75)
            and (strong_structure_signal or strong_grounding_signal or doc_count <= 1)
        )
        step_relax = bool(
            op_intent
            and payload_step_cov is not None
            and payload_step_cov >= 0.60
            and len(unique_ref_nums) >= 2
        )
        if not (op_relax or payload_relax or strong_grounding_signal or step_relax):
            reasons.append("low_visual_score")

    evidence = {
        "doc_count": doc_count,
        "topic_match_count": doc_count if topic_ratio >= 1.0 else 0,
        "topic_match_ratio": round(topic_ratio, 4),
        "structure_match_count": structure_count,
        "avg_rrf_score": round(avg_score, 8),
        "rrf_count": len(score_values),
        "visual_score_threshold": visual_score_threshold,
        "min_doc_count_threshold": min_doc_count,
        "operation_intent": op_intent,
        "stitched_page_count": len(set(page_nums)),
        "stitched_page_span": int(page_span),
        "unique_figure_refs": len(unique_ref_nums),
        "grounding_doc_count": grounding_doc_count,
        "grounding_fact_count": grounding_fact_count,
        "grounding_doc_count_raw": grounding_doc_count_raw,
        "grounding_fact_count_raw": grounding_fact_count_raw,
        "grounding_agreement_mean": round(float(grounding_agreement_mean), 4) if isinstance(grounding_agreement_mean, (int, float)) else None,
        "chapter_calibration": {k: v for k, v in calib.items() if k != "resolved_topic_id"},
        "evidence_ok": doc_count >= 1,
        "reasons": reasons,
    }
    context_text = "\n".join(context_lines).strip()
    return sources_data, context_text, evidence

# ---------------------------------------------------------------------------
# AI Logic (Token Optimized)
# ---------------------------------------------------------------------------
def call_hf_api(messages, model, stream=False, max_tokens=None, temperature=0.2):
    try:
        token_budget = CHAT_MAX_TOKENS if max_tokens is None else int(max_tokens)
        return hf_client.chat_completion(
            model=model,
            messages=messages,
            stream=stream,
            max_tokens=token_budget,
            temperature=temperature,
        )
    except Exception as e:
        log_event("hf_api_error", model=model, stream=stream, error=str(e))
        st.error(f"เกิดข้อผิดพลาดชั่วคราวจาก API: {str(e)}")
        return None


def compute_visual_generation_budget(question: str, payload: dict, sources_data: list[dict]) -> int:
    if not VISUAL_DYNAMIC_TOKEN_ENABLED:
        return max(256, CHAT_MAX_TOKENS)
    query_profile = payload.get("query_profile", {}) if isinstance(payload, dict) else {}
    operation_intent = bool(query_profile.get("operation_intent"))
    unique_refs = set()
    for s in sources_data or []:
        for fr in s.get("figure_refs", []) or []:
            m = re.findall(r"(\d+(?:\.\d+)?)", str(fr))
            for x in m:
                unique_refs.add(x)
    question_norm = str(question or "").strip().lower()
    procedural_hint = any(k in question_norm for k in ["ขั้นตอน", "กระบวนการ", "ดำเนินการ", "ทำงาน"])
    budget = int(VISUAL_DYNAMIC_TOKEN_BASE)
    budget += int(VISUAL_DYNAMIC_TOKEN_PER_SOURCE) * max(1, len(sources_data or []))
    budget += int(VISUAL_DYNAMIC_TOKEN_PER_FIGURE) * len(unique_refs)
    if operation_intent or procedural_hint:
        budget += int(VISUAL_DYNAMIC_TOKEN_OPERATION_BONUS)
    budget = max(320, min(int(VISUAL_DYNAMIC_TOKEN_CAP), budget))
    return int(budget)


def evaluate_visual_grounding_gate(
    evidence: dict,
    payload: dict,
    task_metrics: dict,
    *,
    use_grounding: bool,
    gate_overrides: dict | None = None,
) -> dict:
    op_intent = bool((payload.get("query_profile", {}) or {}).get("operation_intent"))
    if not use_grounding:
        # Fast mode: grounding model is intentionally disabled, so do not block answering here.
        return {
            "enabled": False,
            "pass": True,
            "reasons": ["grounding_disabled"],
            "operation_intent": op_intent,
            "mode": "disabled_bypass",
        }
    if not VISUAL_GROUNDING_GATE_ENABLED:
        return {
            "enabled": False,
            "pass": True,
            "reasons": [],
            "operation_intent": op_intent,
        }
    if VISUAL_GROUNDING_GATE_OPERATION_ONLY and not op_intent:
        return {
            "enabled": True,
            "pass": True,
            "reasons": [],
            "operation_intent": op_intent,
            "mode": "operation_only_bypass",
        }

    overrides = gate_overrides if isinstance(gate_overrides, dict) else {}
    reasons = []
    grounded_docs = int(evidence.get("grounding_doc_count", 0) or 0)
    grounded_facts = int(evidence.get("grounding_fact_count", 0) or 0)
    grounded_docs_raw = int(evidence.get("grounding_doc_count_raw", 0) or 0)
    grounded_facts_raw = int(evidence.get("grounding_fact_count_raw", 0) or 0)
    grounding_agreement_mean = evidence.get("grounding_agreement_mean")
    unique_figures = int(evidence.get("unique_figure_refs", 0) or 0)
    image_heavy = bool(unique_figures >= VISUAL_GROUNDING_RELAX_MIN_UNIQUE_FIGURES)
    min_docs = int(overrides.get("min_grounded_docs", VISUAL_GROUNDING_MIN_DOCS) or VISUAL_GROUNDING_MIN_DOCS)
    min_facts = int(
        overrides.get(
            "min_grounded_facts",
            VISUAL_GROUNDING_MIN_OPERATION_FACTS if op_intent else VISUAL_GROUNDING_MIN_FACTS,
        )
        or (VISUAL_GROUNDING_MIN_OPERATION_FACTS if op_intent else VISUAL_GROUNDING_MIN_FACTS)
    )
    if op_intent and VISUAL_GROUNDING_RELAX_FOR_IMAGE_HEAVY and image_heavy:
        min_facts = max(VISUAL_GROUNDING_MIN_FACTS, min_facts - VISUAL_GROUNDING_RELAX_FACT_REDUCTION)
    step_cov = task_metrics.get("procedure_step_coverage_proxy") if isinstance(task_metrics, dict) else None
    region_quality = task_metrics.get("region_quality_mean") if isinstance(task_metrics, dict) else None
    step_cov_num = float(step_cov) if isinstance(step_cov, (int, float)) else None
    region_quality_num = float(region_quality) if isinstance(region_quality, (int, float)) else None
    min_step_coverage = float(overrides.get("min_step_coverage", VISUAL_GROUNDING_MIN_STEP_COVERAGE) or VISUAL_GROUNDING_MIN_STEP_COVERAGE)
    if op_intent and VISUAL_GROUNDING_RELAX_FOR_IMAGE_HEAVY and image_heavy:
        min_step_coverage = max(0.30, min_step_coverage - 0.10)
    high_region_quality = bool(region_quality_num is not None and region_quality_num >= 0.88)
    min_grounding_agreement = float(overrides.get("min_grounding_agreement", 0.0) or 0.0)

    if grounded_docs < min_docs:
        reasons.append("low_grounded_doc_count")
    if grounded_facts < min_facts:
        reasons.append("low_grounded_fact_count")
    if op_intent and (step_cov_num is None or step_cov_num < min_step_coverage):
        if not (VISUAL_GROUNDING_RELAX_FOR_IMAGE_HEAVY and image_heavy and high_region_quality and grounded_docs >= 1):
            # Relax one condition for image-heavy chapters when crop quality is high and we already have grounded facts.
            reasons.append("low_step_coverage_proxy")
    if min_grounding_agreement > 0 and isinstance(grounding_agreement_mean, (int, float)):
        if float(grounding_agreement_mean) < min_grounding_agreement:
            reasons.append("low_grounding_agreement")

    # If filtered facts are low but raw grounding is present with good visual quality, avoid false abstain.
    if reasons and VISUAL_GROUNDING_RELAX_FOR_IMAGE_HEAVY:
        if grounded_docs_raw >= min_docs and grounded_facts_raw >= max(1, min_facts - 1):
            if high_region_quality and image_heavy:
                reasons = [r for r in reasons if r not in {"low_grounded_doc_count", "low_grounded_fact_count"}]

    return {
        "enabled": True,
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "operation_intent": op_intent,
        "image_heavy": image_heavy,
        "unique_figure_refs": unique_figures,
        "grounded_doc_count": grounded_docs,
        "grounded_fact_count": grounded_facts,
        "grounded_doc_count_raw": grounded_docs_raw,
        "grounded_fact_count_raw": grounded_facts_raw,
        "grounding_agreement_mean": float(grounding_agreement_mean) if isinstance(grounding_agreement_mean, (int, float)) else None,
        "min_grounding_agreement": min_grounding_agreement if min_grounding_agreement > 0 else None,
        "min_docs": min_docs,
        "min_facts": min_facts,
        "step_coverage_proxy": step_cov_num,
        "region_quality_mean": region_quality_num,
        "min_step_coverage": min_step_coverage if op_intent else None,
    }


def build_visual_corrective_attempts(require_structure: bool) -> list[dict]:
    attempts = [
        {
            "name": "primary",
            "require_structure": bool(require_structure),
            "top_k": int(VISUAL_RETRIEVAL_TOP_K),
            "candidate_k": int(VISUAL_RETRIEVAL_CANDIDATE_K),
            "topic_threshold": float(VISUAL_RETRIEVAL_TOPIC_THRESHOLD),
        }
    ]
    if not VISUAL_CORRECTIVE_RETRIEVAL_ENABLED or VISUAL_CORRECTIVE_MAX_RETRIES <= 0:
        return attempts

    attempts.append(
        {
            "name": "corrective_expand",
            "require_structure": bool(require_structure),
            "top_k": max(int(VISUAL_RETRIEVAL_TOP_K), int(round(VISUAL_RETRIEVAL_TOP_K * VISUAL_CORRECTIVE_TOPK_MULTIPLIER))),
            "candidate_k": max(
                int(VISUAL_RETRIEVAL_CANDIDATE_K),
                int(round(VISUAL_RETRIEVAL_CANDIDATE_K * VISUAL_CORRECTIVE_CANDIDATE_MULTIPLIER)),
            ),
            "topic_threshold": min(float(VISUAL_RETRIEVAL_TOPIC_THRESHOLD), float(VISUAL_CORRECTIVE_TOPIC_THRESHOLD_RELAX)),
        }
    )

    if VISUAL_CORRECTIVE_MAX_RETRIES >= 2 and bool(require_structure):
        attempts.append(
            {
                "name": "corrective_relaxed_structure",
                "require_structure": False,
                "top_k": max(
                    int(VISUAL_RETRIEVAL_TOP_K),
                    int(round(VISUAL_RETRIEVAL_TOP_K * VISUAL_CORRECTIVE_TOPK_MULTIPLIER)),
                ),
                "candidate_k": max(
                    int(VISUAL_RETRIEVAL_CANDIDATE_K),
                    int(round(VISUAL_RETRIEVAL_CANDIDATE_K * VISUAL_CORRECTIVE_CANDIDATE_MULTIPLIER)),
                ),
                "topic_threshold": min(float(VISUAL_RETRIEVAL_TOPIC_THRESHOLD), float(VISUAL_CORRECTIVE_TOPIC_THRESHOLD_RELAX)),
            }
        )

    return attempts[: 1 + VISUAL_CORRECTIVE_MAX_RETRIES]


def _response_needs_continuation(text: str, finish_reason: str | None) -> bool:
    """Heuristic guard: continue when model stops due to token cap or obvious dangling tail."""
    if not text:
        return False
    tail = text.rstrip()
    if not tail:
        return False
    if finish_reason == "length":
        return True
    if re.search(r"\([^\)]*$", tail):
        return True
    return tail.endswith((",", ":", ";", "-", "–", "—", "/", "…"))


def _merge_with_overlap(base: str, suffix: str) -> str:
    """Join generated continuation while removing simple duplicated overlap."""
    if not suffix:
        return base
    a = base or ""
    b = suffix or ""
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


def _continue_answer(
    question: str,
    context_text: str,
    partial_answer: str,
    max_tokens: int | None = None,
) -> str:
    """Request a short continuation to complete a truncated answer tail."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are continuing an unfinished assistant answer in Thai. "
                "Continue from the exact tail, do not repeat previous text, and end with a complete sentence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"คำถามผู้ใช้:\n{question}\n\n"
                f"บริบท:\n{context_text[:1800]}\n\n"
                f"คำตอบที่ถูกตัดท้าย:\n{partial_answer[-1800:]}\n\n"
                "โปรดเขียนต่ออีกสั้นๆ 1-3 ประโยคให้จบประเด็น"
            ),
        },
    ]
    token_budget = int(max_tokens) if max_tokens is not None else int(CHAT_CONTINUE_MAX_TOKENS)
    res = call_hf_api(messages, CHAT_MODEL_ID, stream=False, max_tokens=token_budget, temperature=0.1)
    if not res or not getattr(res, "choices", None):
        return ""
    return (res.choices[0].message.content or "").strip()


def classify_topic(question):
    deterministic = best_topic_label_from_question(question, topic_structure)
    if deterministic:
        log_event("topic_classified_deterministic", question_len=len(question), predicted_topic=deterministic)
        return deterministic

    all_names = build_topic_labels(topic_structure)
    shortlist = shortlist_topic_labels_for_question(question, topic_structure, max_candidates=60)
    topic_pool = shortlist if shortlist else all_names[:60]
    prompt = (
        f"Topics: {topic_pool}\n"
        f"Question: '{question}'\n"
        "Return only one best-matching topic name from the list, or 'other'."
    )
    messages = [{"role": "user", "content": prompt}]
    res = call_hf_api(messages, CHAT_MODEL_ID)
    if res and res.choices:
        predicted = res.choices[0].message.content.strip()
        log_event("topic_classified", question_len=len(question), predicted_topic=predicted)
        return predicted
    log_event("topic_classification_fallback", question_len=len(question))
    return "other"


def run_judge(question, answer, context):
    """Evaluate the answer with 3 RAGAS-style metrics + reasoning."""
    judge_prompt = (
        "You are a strict evaluator for a Retrieval-Augmented Generation system.\n"
        "Score the following answer on three criteria (0.0 to 1.0):\n"
        "1. faithfulness: Does the answer only contain claims supported by the context?\n"
        "2. relevance: Does the answer directly address the question?\n"
        "3. context_precision: Does the retrieved context contain the information needed to answer?\n\n"
        f"[Question]: {question}\n"
        f"[Context]: {context[:1200]}\n"
        f"[Answer]: {answer[:800]}\n\n"
        'Return ONLY valid JSON: {"scores": {"faithfulness": 0.8, "relevance": 0.9, "context_precision": 0.7}, '
        '"reasoning": "short explanation"}'
    )
    messages = [
        {"role": "system", "content": "You are a JSON evaluation judge. Output ONLY valid JSON. No extra text."},
        {"role": "user", "content": judge_prompt},
    ]
    try:
        res = hf_client.chat_completion(
            model=JUDGE_MODEL_ID,
            messages=messages,
            max_tokens=300,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = res.choices[0].message.content.strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in judge response")

        if "scores" not in result:
            result = {"scores": result, "reasoning": ""}
        scores = result.get("scores", {})
        for key in ["faithfulness", "relevance", "context_precision"]:
            val = scores.get(key, 0)
            if isinstance(val, (int, float)):
                scores[key] = max(0.0, min(1.0, float(val)))
            else:
                scores[key] = 0.5
        result["scores"] = scores
        if "reasoning" not in result:
            result["reasoning"] = ""
        log_event(
            "judge_success",
            question_len=len(question),
            answer_len=len(answer),
            context_len=len(context),
            scores=result.get("scores", {}),
        )
        return result
    except Exception as e:
        log_event("judge_fallback", error=str(e), question_len=len(question))
        return {
            "scores": {"faithfulness": 0.5, "relevance": 0.5, "context_precision": 0.5},
            "reasoning": f"Judge ใช้ค่า fallback เนื่องจากเกิดข้อผิดพลาด: {str(e)[:120]}",
        }


def generate_response_visual(question, topic_hint: str | None = None, require_structure: bool = False):
    resolved_section_id = resolve_target_section_id(topic_hint, question)
    retrieval_query, rewrite_meta = rewrite_query_for_retrieval(
        question,
        topic_hint=topic_hint,
        target_section_id=resolved_section_id,
    )
    log_event(
        "query_rewrite",
        stage="visual",
        changed=bool(rewrite_meta.get("changed", False)),
        mode=str(rewrite_meta.get("mode", "")),
        llm_status=str(rewrite_meta.get("llm_status", "")),
        target_section_id=str(rewrite_meta.get("target_section_id", "")),
        retrieval_query_len=len(retrieval_query or ""),
        original_query_len=len(str(question or "")),
    )

    backend = FORCED_VISUAL_BACKEND
    endpoint_url = resolve_colpali_endpoint_url(str(st.session_state.get("visual_endpoint_url", "")))
    use_vlm_rerank = bool(st.session_state.get("visual_use_vlm_rerank", VISUAL_RETRIEVAL_USE_VLM_RERANK))
    use_grounding = bool(st.session_state.get("visual_use_grounding", VISUAL_RETRIEVAL_USE_GROUNDING))
    sparse_strategy = str(st.session_state.get("visual_sparse_strategy", VISUAL_SPARSE_STRATEGY))

    attempts = build_visual_corrective_attempts(bool(require_structure))
    attempt_logs = []
    selected = None
    best_failed = None
    calibrated_inserted_topics = set()
    attempt_idx = 0

    def _as_int(value, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    while attempt_idx < len(attempts):
        attempt = attempts[attempt_idx]
        attempt_idx += 1
        visual_result = run_visual_retrieval_query(
            retrieval_query,
            require_structure=bool(attempt.get("require_structure", require_structure)),
            backend=backend,
            endpoint_url=endpoint_url,
            use_vlm_rerank=use_vlm_rerank,
            use_visual_grounding=use_grounding,
            sparse_strategy_override=sparse_strategy,
            top_k_override=int(attempt.get("top_k", VISUAL_RETRIEVAL_TOP_K)),
            candidate_k_override=int(attempt.get("candidate_k", VISUAL_RETRIEVAL_CANDIDATE_K)),
            topic_threshold_override=float(attempt.get("topic_threshold", VISUAL_RETRIEVAL_TOPIC_THRESHOLD)),
            ground_top_n_override=int(attempt.get("ground_top_n", VISUAL_GROUND_TOP_N)),
        )
        if not visual_result.get("ok", False):
            err = str(visual_result.get("error", "visual_retrieval_failed"))
            attempt_logs.append({"name": attempt.get("name", "unknown"), "ok": False, "error": err})
            continue

        payload = visual_result.get("payload", {}) or {}
        topic_prediction = payload.get("topic_prediction", {}) if isinstance(payload, dict) else {}
        topic_id = str(topic_prediction.get("topic_id", "")).strip()
        chapter_profile = resolve_chapter_calibration(topic_id)
        task_metrics = payload.get("task_metrics", {}) if isinstance(payload, dict) else {}
        sources_data, context_text, evidence = build_visual_context_and_sources(
            payload,
            topic_hint=topic_hint,
            require_structure=bool(attempt.get("require_structure", require_structure)),
            calibration=chapter_profile,
        )
        quality = (
            (100.0 if evidence.get("evidence_ok", False) else 0.0)
            + (2.5 * float(evidence.get("doc_count", 0) or 0))
            + (250.0 * float(evidence.get("avg_rrf_score", 0.0) or 0.0))
        )
        attempt_logs.append(
            {
                "name": attempt.get("name", "unknown"),
                "ok": True,
                "retrieved_docs": len(sources_data),
                "evidence_ok": bool(evidence.get("evidence_ok", False)),
                "reasons": evidence.get("reasons", []),
                "avg_rrf_score": evidence.get("avg_rrf_score"),
                "task_metrics": task_metrics,
                "topic_id": topic_id,
                "chapter_calibration": chapter_profile.get("matched_topic_id", "") if isinstance(chapter_profile, dict) else "",
            }
        )

        candidate = {
            "attempt": attempt,
            "payload": payload,
            "task_metrics": task_metrics,
            "sources_data": sources_data,
            "context_text": context_text,
            "evidence": evidence,
            "quality": quality,
            "chapter_profile": chapter_profile,
        }
        proc_cov = task_metrics.get("procedure_step_coverage_proxy")
        op_intent = bool((payload.get("query_profile", {}) or {}).get("operation_intent"))
        target_step_cov = float(chapter_profile.get("selection_min_step_coverage", 0.75) or 0.75) if isinstance(chapter_profile, dict) else 0.75
        needs_more_step_coverage = bool(
            op_intent and isinstance(proc_cov, (int, float)) and float(proc_cov) < target_step_cov
        )
        if (
            topic_id
            and topic_id not in calibrated_inserted_topics
            and isinstance(chapter_profile, dict)
            and chapter_profile
            and attempt.get("name") != "chapter_calibrated"
            and (not evidence.get("evidence_ok", False) or needs_more_step_coverage)
        ):
            calibrated_attempt = {
                "name": "chapter_calibrated",
                "require_structure": bool(attempt.get("require_structure", require_structure)),
                "top_k": _as_int(chapter_profile.get("top_k"), int(attempt.get("top_k", VISUAL_RETRIEVAL_TOP_K))),
                "candidate_k": _as_int(chapter_profile.get("candidate_k"), int(attempt.get("candidate_k", VISUAL_RETRIEVAL_CANDIDATE_K))),
                "topic_threshold": _as_float(
                    chapter_profile.get("topic_threshold"),
                    float(attempt.get("topic_threshold", VISUAL_RETRIEVAL_TOPIC_THRESHOLD)),
                ),
                "ground_top_n": _as_int(chapter_profile.get("ground_top_n"), int(VISUAL_GROUND_TOP_N)),
            }
            attempts.insert(attempt_idx, calibrated_attempt)
            calibrated_inserted_topics.add(topic_id)

        if evidence.get("evidence_ok", False) and sources_data and not needs_more_step_coverage:
            selected = candidate
            break
        if needs_more_step_coverage:
            attempt_logs[-1]["reasons"] = sorted(set(list(attempt_logs[-1].get("reasons", [])) + ["low_step_coverage"]))
        if (best_failed is None) or (quality > best_failed.get("quality", -1e9)):
            best_failed = candidate

    if selected is None:
        selected = best_failed

    log_event(
        "visual_retrieve_attempts",
        attempts=attempt_logs,
        selected_attempt=selected.get("attempt", {}).get("name") if isinstance(selected, dict) else None,
        corrective_enabled=VISUAL_CORRECTIVE_RETRIEVAL_ENABLED,
    )

    if not selected:
        return "ABSTAIN: ไม่พบหลักฐานภาพ/เนื้อหาจากเอกสารที่ตรงคำถาม", [], ""

    payload = selected.get("payload", {}) or {}
    task_metrics = selected.get("task_metrics", {}) if isinstance(selected.get("task_metrics"), dict) else {}
    sources_data = selected.get("sources_data", []) or []
    context_text = selected.get("context_text", "") or ""
    evidence = selected.get("evidence", {}) if isinstance(selected.get("evidence"), dict) else {}
    selected_profile = selected.get("chapter_profile", {}) if isinstance(selected.get("chapter_profile"), dict) else {}

    log_event(
        "visual_retrieve_success",
        backend=payload.get("backend"),
        splade_status=payload.get("splade_status"),
        splade_score_ready=payload.get("splade_score_ready"),
        colpali_status=payload.get("colpali_status"),
        colpali_score_ready=payload.get("colpali_score_ready"),
        retrieved_docs=len(sources_data),
        evidence=evidence,
        query_profile=payload.get("query_profile", {}),
        topic_prediction=payload.get("topic_prediction", {}),
        chapter_calibration=selected_profile,
        hierarchy_quality=payload.get("hierarchy_quality", {}),
        task_metrics=task_metrics,
        scoring_weights=payload.get("scoring_weights", {}),
        selected_attempt=selected.get("attempt", {}).get("name"),
    )

    if not sources_data:
        return "ABSTAIN: ไม่พบหลักฐานภาพ/เนื้อหาจากเอกสารที่ตรงคำถาม", [], ""

    # Relaxed: ไม่ ABSTAIN ถ้ามี context อย่างน้อย 1 รายการ
    if not evidence.get("evidence_ok", False) and not sources_data:
        reasons = ", ".join(evidence.get("reasons") or []) or "insufficient_evidence"
        return (
            "ABSTAIN: หลักฐานจาก visual retrieval ยังไม่เพียงพอสำหรับการตอบแบบอ้างอิงเอกสารเท่านั้น "
            f"(เหตุผล: {reasons})",
            sources_data,
            context_text,
        )

    grounding_gate = evaluate_visual_grounding_gate(
        evidence,
        payload,
        task_metrics,
        use_grounding=use_grounding,
        gate_overrides=selected_profile,
    )
    log_event(
        "visual_grounding_gate",
        gate=grounding_gate,
        selected_attempt=selected.get("attempt", {}).get("name"),
    )
    if grounding_gate.get("enabled") and not grounding_gate.get("pass", False):
        reasons = ", ".join(grounding_gate.get("reasons", [])) or "grounding_gate_failed"
        return (
            "ABSTAIN: หลักฐาน grounding จากภาพยังไม่พอสำหรับตอบแบบยืนยันขั้นตอน "
            f"(เหตุผล: {reasons})",
            sources_data,
            context_text,
        )

    system_instr = (
        "You are an AI tutor for Data Structure. "
        "Respond in Thai only. "
        "Answer using ONLY evidence from the provided context. "
        "The context comes from visual retrieval + grounded document snippets. "
        "Do not introduce facts from outside the context. "
        "Use the original wording from context as much as possible, preserving the author's language. "
        "Do NOT generate Mermaid, flowcharts, code blocks, or ASCII diagrams unless explicitly present in context. "
        "Include inline citation [[หน้า X]] for key claims. "
        "If evidence is insufficient, explicitly abstain."
    )
    query_profile = payload.get("query_profile", {}) if isinstance(payload, dict) else {}
    if bool(query_profile.get("operation_intent")):
        system_instr += (
            " For procedural questions, answer as explicit step-by-step numbered list. "
            "Cover the process flow comprehensively across retrieved figures before concluding. "
            "Every step must be grounded by visual facts from context and include inline citations."
        )
    figure_refs = []
    seen_refs = set()
    for s in sources_data:
        for fr in s.get("figure_refs", []) or []:
            frs = str(fr).strip()
            if frs and frs not in seen_refs:
                seen_refs.add(frs)
                figure_refs.append(frs)
    figure_hint = ", ".join(figure_refs[:12]) if figure_refs else ""
    messages = [
        {"role": "system", "content": system_instr},
        {
            "role": "user",
            "content": (
                f"Provided context:\n{context_text}\n\n"
                f"Student question: {question}\n"
                f"Available figure references: {figure_hint or 'none'}\n"
                "If this is a process question, include all major steps present in context and map each step to citations."
            ),
        },
    ]
    max_tokens = compute_visual_generation_budget(question, payload, sources_data)
    stream = call_hf_api(messages, CHAT_MODEL_ID, stream=True, temperature=0.2, max_tokens=max_tokens)
    log_event(
        "visual_generation_budget",
        max_tokens=max_tokens,
        source_count=len(sources_data),
        unique_fig_refs=task_metrics.get("unique_figure_refs_at_k"),
        procedure_step_coverage_proxy=task_metrics.get("procedure_step_coverage_proxy"),
    )
    return stream, sources_data, context_text


def should_use_visual_path(question: str, require_structure: bool = False) -> bool:
    if bool(require_structure) and VISUAL_FORCE_ON_REQUIRE_STRUCTURE:
        return True
    q = str(question or "").strip().lower()
    if not q:
        return False
    explicit_visual_terms = (
        "ภาพ",
        "รูป",
        "แผนภาพ",
        "ผังงาน",
        "diagram",
        "flowchart",
        "figure",
        "visual",
        "พร้อมภาพ",
    )
    broad_visual_terms = explicit_visual_terms + (
        "โครงสร้าง",
        "ขั้นตอน",
        "กระบวนการ",
        "การทำงาน",
        "คิว",
        "สแตก",
        "ลิงค์ลิสต์",
        "linked list",
        "queue",
        "stack",
        "tree",
        "graph",
        "enqueue",
        "dequeue",
        "insert",
        "delete",
    )
    visual_terms = explicit_visual_terms if FAST_RETRIEVAL_MODE else broad_visual_terms
    return any(t in q for t in visual_terms)


def generate_response(question, topic_hint: str | None = None, require_structure: bool = False):
    # Policy in this phase: text-first for latency/reliability, visual when query clearly needs structure/operation evidence.
    if VISUAL_RETRIEVAL_ENABLED and should_use_visual_path(question, require_structure=require_structure):
        st.session_state.retrieval_mode = "visual"
        return generate_response_visual(question, topic_hint=topic_hint, require_structure=require_structure)
    st.session_state.retrieval_mode = "text"

    retrieval_mode = str(st.session_state.get("retrieval_mode", "text")).strip().lower()
    if retrieval_mode == "visual":
        return generate_response_visual(question, topic_hint=topic_hint, require_structure=require_structure)
    if (
        retrieval_mode == "text"
        and VISUAL_RETRIEVAL_ENABLED
        and not FAST_RETRIEVAL_MODE
        and TEXT_MODE_PROMOTE_VISUAL_FOR_STRUCTURE
        and bool(require_structure)
    ):
        visual_stream, visual_sources, visual_context = generate_response_visual(
            question,
            topic_hint=topic_hint,
            require_structure=True,
        )
        if not (isinstance(visual_stream, str) and (visual_stream.startswith("ERROR:") or visual_stream.startswith("ABSTAIN:"))):
            log_event(
                "text_mode_promoted_to_visual",
                question_len=len(question),
                visual_sources=len(visual_sources),
            )
            return visual_stream, visual_sources, visual_context
        log_event(
            "text_mode_visual_promotion_fallback",
            reason=str(visual_stream)[:200],
            question_len=len(question),
        )

    try:
        if rag is None:
            log_event("retrieve_unavailable", reason="rag_not_loaded")
            return "ERROR: ระบบ RAG ใช้งานไม่ได้ กรุณาตรวจสอบดัชนีและทรัพยากรโมเดล", [], ""

        resolved_section_id = resolve_target_section_id(topic_hint, question)
        retrieval_query, rewrite_meta = rewrite_query_for_retrieval(
            question,
            topic_hint=topic_hint,
            target_section_id=resolved_section_id,
        )
        log_event(
            "query_rewrite",
            stage="text",
            changed=bool(rewrite_meta.get("changed", False)),
            mode=str(rewrite_meta.get("mode", "")),
            llm_status=str(rewrite_meta.get("llm_status", "")),
            target_section_id=str(rewrite_meta.get("target_section_id", "")),
            retrieval_query_len=len(retrieval_query or ""),
            original_query_len=len(str(question or "")),
        )
        min_section_page_diversity = int(RETRIEVE_MIN_SECTION_PAGE_DIVERSITY)
        if resolved_section_id == "3.3.2":
            # Queue array operation spans 32-35 in this corpus.
            min_section_page_diversity = max(min_section_page_diversity, 4)
        elif resolved_section_id == "2.4.2":
            # Linked-list operation chapter is distributed across 25-28.
            min_section_page_diversity = max(min_section_page_diversity, 4)
        elif resolved_section_id in {"3.3.3"}:
            min_section_page_diversity = max(min_section_page_diversity, 3)

        retrieval_filters = {
            "topic_hint": topic_hint or "",
            "target_section_id": resolved_section_id,
            "query_text": question,
            "retrieval_query_text": retrieval_query,
            "require_structure": bool(require_structure),
            "strict_topic": bool(RETRIEVE_STRICT_TOPIC),
            "strict_structure": bool(RETRIEVE_STRICT_STRUCTURE),
            "strict_section_only": bool(RETRIEVE_STRICT_SECTION_ONLY),
            "allow_offtopic_docs": int(RETRIEVE_ALLOW_OFFTOPIC_DOCS),
            "min_section_page_diversity": min_section_page_diversity,
        }
        try:
            docs = rag.retrieve(retrieval_query, RETRIEVE_TOP_K, RETRIEVE_TOP_N, filters=retrieval_filters)
        except TypeError as exc:
            err_text = str(exc)
            if ("unexpected keyword argument" not in err_text) or ("filters" not in err_text):
                raise
            # Backward-compatibility: older retriever API without "filters".
            base_docs = rag.retrieve(retrieval_query, RETRIEVE_TOP_K, RETRIEVE_TOP_N)
            docs = apply_local_retrieval_filters(base_docs, retrieval_filters)
            log_event("retrieve_filters_fallback", reason="legacy_retriever_signature")

        rag_status = rag.get_runtime_status() if hasattr(rag, "get_runtime_status") else {}
        _target_section_id = str(retrieval_filters.get("target_section_id", "")).strip()
        _section_scope_pages = []
        if _target_section_id and hasattr(rag, "topic_to_pages"):
            try:
                _section_scope_pages = [
                    str(x).strip()
                    for x in ((getattr(rag, "topic_to_pages", {}) or {}).get(_target_section_id, []) or [])
                    if str(x).strip()
                ]
            except Exception:
                _section_scope_pages = []
        evidence = compute_evidence_stats(
            docs,
            topic_hint,
            require_structure,
            target_section_id=_target_section_id,
            section_scope_pages=_section_scope_pages,
        )
        log_event(
            "retrieve_success",
            query_len=len(question),
            retrieval_query_len=len(retrieval_query or ""),
            retrieved_docs=len(docs),
            retrieve_top_k=RETRIEVE_TOP_K,
            retrieve_top_n=RETRIEVE_TOP_N,
            topic_hint=topic_hint or "",
            require_structure=bool(require_structure),
            evidence=evidence,
            reranker_ready=rag_status.get("reranker_ready"),
            reranker_error=rag_status.get("reranker_error"),
        )

        context_text = ""
        sources_data = []
        target_section_id = str(retrieval_filters.get("target_section_id", "")).strip()
        section_scope_pages = []
        if target_section_id and hasattr(rag, "topic_to_pages"):
            try:
                raw_scope = (getattr(rag, "topic_to_pages", {}) or {}).get(target_section_id, []) or []
                section_scope_pages = [str(x).strip() for x in raw_scope if str(x).strip()]
            except Exception:
                section_scope_pages = []
        if target_section_id and section_scope_pages:
            allowed_pages = set(_scope_pages_for_display(section_scope_pages))
            if allowed_pages:
                before_scope = len(docs)
                scoped_docs = []
                for d in docs:
                    meta = getattr(d, "metadata", {}) or {}
                    page = str(meta.get("page", "")).strip()
                    if page and page in allowed_pages:
                        scoped_docs.append(d)
                if scoped_docs:
                    docs = scoped_docs
                    log_event(
                        "context_docs_scoped_to_section_pages",
                        target_section_id=target_section_id,
                        before_docs=before_scope,
                        after_docs=len(docs),
                        allowed_pages=sorted(allowed_pages),
                    )
        
        # FILTER: Prioritize chunks that actually contain the section heading in content
        # This fixes the issue where a page has multiple sections and wrong chunks are selected
        if target_section_id and docs:
            section_heading_patterns = {
                "1.1.1": ["1.1.1 ความหมาย", "### 1.1.1", "ความหมายโครงสร้างข้อมูล"],
                "1.1.2": ["1.1.2 อัลกอริทึม", "### 1.1.2"],
                "1.2.1": ["1.2.1 บิต", "### 1.2.1", "บิต (Bit)"],
                "1.2.2": ["1.2.2 ไบต์", "1.2.2 เบต์", "### 1.2.2", "ไบต์ (Byte)", "เบต์ (Byte)"],
            }
            # Auto-generate pattern for any section ID like "X.X.X"
            base_patterns = section_heading_patterns.get(target_section_id, [])
            auto_patterns = [target_section_id, f"### {target_section_id}"]
            patterns = list(dict.fromkeys(base_patterns + auto_patterns))  # Remove duplicates
            section_docs = []
            other_docs = []
            for d in docs:
                content = str(getattr(d, "page_content", ""))
                meta = getattr(d, "metadata", {}) or {}
                # Build text to check from both content and metadata
                # Include H1/H2/H3 and section fields which contain the actual section heading
                meta_text = " ".join([
                    str(meta.get("H1", "")),
                    str(meta.get("H2", "")),
                    str(meta.get("H3", "")),
                    str(meta.get("section", "")),
                    str(meta.get("best_topic_id", "")),
                    str(meta.get("topic_path", "")),
                ])
                check_text = f"{content} {meta_text}"
                # Check if contains section heading
                has_section = any(p in check_text for p in patterns)
                if has_section:
                    section_docs.append(d)
                else:
                    other_docs.append(d)
            # Prioritize docs with section heading, but keep others as fallback
            if section_docs:
                docs = section_docs + other_docs
                log_event(
                    "context_docs_section_content_filtered",
                    target_section_id=target_section_id,
                    section_docs_count=len(section_docs),
                    other_docs_count=len(other_docs),
                )
        context_doc_limit = max(1, CONTEXT_DOC_LIMIT)
        min_unique_pages = 2 if target_section_id else 1
        if target_section_id in OPERATION_HEAVY_SECTION_IDS:
            # Operation chapters are step-wise across adjacent pages.
            # Increase context breadth deterministically to avoid one-page collapse.
            context_doc_limit = max(context_doc_limit, int(VISUAL_CONTEXT_DOC_LIMIT_OPERATION))
            min_unique_pages = max(min_unique_pages, 3)

        docs_for_context = select_context_docs_diverse(
            docs,
            context_doc_limit,
            min_unique_pages=min_unique_pages,
            target_section_id=target_section_id,
        )
        
        # Sort by page number to maintain document reading order (top-to-bottom)
        # This ensures the context flows naturally as in the original textbook
        def _get_page_num(doc):
            meta = getattr(doc, "metadata", {}) or {}
            page_str = str(meta.get("page", "0")).strip()
            try:
                return int(page_str)
            except (ValueError, TypeError):
                return 0
        
        docs_for_context = sorted(docs_for_context, key=_get_page_num)
        
        if docs_for_context:
            for i, d in enumerate(docs_for_context):
                meta = getattr(d, "metadata", {}) or {}
                src = str(meta.get("source", "reference_document"))
                page = str(meta.get("page", "")).strip()
                chunk_id = str(meta.get("chunk_id", f"chunk-{i+1}")).strip()
                citation = f"{src}" + (f":{page}" if page else "")
                sources_data.append(
                    {
                        "source": src,
                        "page": page,
                        "chunk_id": chunk_id,
                        "citation": citation,
                        "rrf_score": meta.get("rrf_score"),
                        "topic_match": bool(meta.get("topic_match", False)),
                        "structure_match": bool(meta.get("structure_match", False)),
                        "content": sanitize_visual_preview(d.page_content),
                        "source_excerpt": sanitize_source_excerpt(d.page_content),
                        "target_section_id": target_section_id,
                        "section_scope_pages": section_scope_pages,
                    }
                )
                context_text += f"Context {i+1} [{citation} | {chunk_id}]: {sanitize_visual_preview(d.page_content)}\n"

        if not docs:
            no_doc_msg = "ABSTAIN: ไม่พบหลักฐานจากเอกสารที่ตรงคำถาม จึงยังไม่สามารถตอบได้อย่างน่าเชื่อถือ"
            return no_doc_msg, sources_data, context_text

        weak_evidence_mode = False
        weak_reasons = []
        # Relaxed: ถ้ามี docs แล้วไม่ ABSTAIN - ให้ LLM ตัดสินใจเอง
        if not evidence.get("evidence_ok", False) and not docs_for_context:
            reason_list = [str(r) for r in (evidence.get("reasons") or [])]
            reasons = ", ".join(reason_list) or "insufficient_evidence"
            abstain_msg = (
                "ABSTAIN: หลักฐานจากเอกสารยังไม่เพียงพอสำหรับการตอบแบบอ้างอิงเอกสารเท่านั้น "
                f"(เหตุผล: {reasons})"
            )
            return abstain_msg, sources_data, context_text

        system_instr = (
            "You are an AI tutor for Data Structure. "
            "Respond in Thai only. "
            "Answer using ONLY evidence from the provided context. "
            "Do not introduce facts from outside the context. "
            "Use the original wording from context as much as possible, preserving the author's language. "
            "Include inline citation [[หน้า X]] for key claims. "
            "Do NOT generate Mermaid, flowcharts, code blocks, or ASCII diagrams unless explicitly present in context. "
            "If evidence is insufficient, say you do not know."
        )
        if weak_evidence_mode:
            system_instr += (
                " Retrieval confidence is weak, so be conservative: "
                "quote only explicit facts in context, avoid speculative claims, "
                "and begin answer with 'หมายเหตุ: ใช้หลักฐานจากหน้าที่ค้นคืนได้พร้อมภาพประกอบจากเอกสาร'."
            )

        messages = [
            {"role": "system", "content": system_instr},
            {
                "role": "user",
                "content": f"Provided context:\n{context_text}\n\nStudent question: {question}",
            },
        ]
        text_max_tokens = int(CHAT_MAX_TOKENS)
        if resolved_section_id:
            if resolved_section_id in OPERATION_HEAVY_SECTION_IDS:
                # Use fixed high budget for procedural operation sections to reduce truncation.
                text_max_tokens = int(OPERATION_SECTION_MAX_TOKENS)
            else:
                text_max_tokens = min(text_max_tokens, int(TEXT_SECTION_MAX_TOKENS))
        log_event(
            "text_generation_budget",
            max_tokens=text_max_tokens,
            target_section_id=resolved_section_id,
            source_count=len(sources_data),
        )
        stream = call_hf_api(messages, CHAT_MODEL_ID, stream=True, max_tokens=text_max_tokens, temperature=0.2)
        return stream, sources_data, context_text
    except Exception as e:
        log_event("generate_response_error", error=str(e), query_len=len(question))
        return f"ERROR: สร้างคำตอบไม่สำเร็จ: {str(e)}", [], ""

# ---------------------------------------------------------------------------
# Streamlit UI Construction
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ติวเตอร์โครงสร้างข้อมูล AI", layout="wide")
inject_custom_css()

# Initialize Session State
if "visited_topics" not in st.session_state: st.session_state.visited_topics = set()
if "messages" not in st.session_state: st.session_state.messages = []
if "current_scores" not in st.session_state: st.session_state.current_scores = None
if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = "auto"
else:
    st.session_state.retrieval_mode = "auto"
if "visual_backend" not in st.session_state:
    st.session_state.visual_backend = FORCED_VISUAL_BACKEND
else:
    st.session_state.visual_backend = FORCED_VISUAL_BACKEND
if "visual_endpoint_url" not in st.session_state:
    st.session_state.visual_endpoint_url = resolve_colpali_endpoint_url("")
if "visual_use_vlm_rerank" not in st.session_state:
    st.session_state.visual_use_vlm_rerank = VISUAL_RETRIEVAL_USE_VLM_RERANK
if "visual_use_grounding" not in st.session_state:
    st.session_state.visual_use_grounding = VISUAL_RETRIEVAL_USE_GROUNDING
if "visual_sparse_strategy" not in st.session_state:
    st.session_state.visual_sparse_strategy = VISUAL_SPARSE_STRATEGY
if "visual_endpoint_health" not in st.session_state:
    st.session_state.visual_endpoint_health = None
if "visual_endpoint_autowarm_done" not in st.session_state:
    st.session_state.visual_endpoint_autowarm_done = False
if "visual_env_save_status" not in st.session_state:
    st.session_state.visual_env_save_status = None
if "visual_standard_query_report" not in st.session_state:
    st.session_state.visual_standard_query_report = None
if "startup_security" not in st.session_state:
    st.session_state.startup_security = run_startup_security_checks(rag, rag_runtime_status)
    sec = st.session_state.startup_security
    log_event(
        "startup_security_checks",
        policy=sec.get("policy"),
        result=sec.get("result"),
        manifest_exists=sec.get("manifest_exists"),
        integrity_ok=sec.get("integrity_ok"),
        issues=sec.get("issues", []),
    )

if (
    VISUAL_RETRIEVAL_ENABLED
    and VISUAL_ENDPOINT_AUTOWARM_ON_STARTUP
    and not bool(st.session_state.get("visual_endpoint_autowarm_done", False))
):
    # Skip endpoint warm-up if using local backend
    if FORCED_VISUAL_BACKEND == "local":
        st.session_state.visual_endpoint_health = {"ok": True, "backend": "local", "note": "Using local backend, endpoint not needed"}
        st.session_state.visual_endpoint_autowarm_done = True
    else:
        endpoint_for_warm = resolve_colpali_endpoint_url(str(st.session_state.get("visual_endpoint_url", "")))
        if endpoint_for_warm:
            warm = run_visual_endpoint_healthcheck(
                endpoint_for_warm,
                HF_TOKEN,
                timeout_sec=VISUAL_ENDPOINT_AUTOWARM_TIMEOUT_SEC,
            )
            st.session_state.visual_endpoint_health = warm
            st.session_state.visual_endpoint_autowarm_done = True
            log_event(
                "visual_endpoint_autowarm_startup",
                ok=warm.get("ok"),
                endpoint=warm.get("endpoint_url", ""),
                validation=warm.get("validation", ""),
                error=warm.get("error", ""),
            )

# --- Sidebar ---
with st.sidebar:
    st.markdown(icon_label("menu_book", "สารบัญเนื้อหา", variant="section"), unsafe_allow_html=True)
    st.caption("คลิกหัวข้อหลักเพื่อดูหัวข้อย่อย และใช้เป็นแนวทางในการตั้งคำถาม")

    for main_topic, sub_list in topic_structure.items():
        with st.expander(f"{main_topic}", expanded=False):
            if not sub_list:
                st.write("_(ไม่มีหัวข้อย่อย)_")
            for sub_item in sub_list:
                sub_title = sub_item["title"] if isinstance(sub_item, dict) else str(sub_item)
                children = sub_item.get("children", []) if isinstance(sub_item, dict) else []

                # Display sub-topic header
                st.markdown(f"**{sub_title}**")

                # Display sub-sub-topics as an indented list (each on its own line)
                if children:
                    for c in children:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- {c}")
    

    st.divider()
    st.markdown(icon_label("monitor_heart", "สถานะระบบ", variant="section"), unsafe_allow_html=True)
    sec = st.session_state.startup_security
    if st.button("ตรวจสอบความปลอดภัยอีกครั้ง", key="btn_rerun_security", use_container_width=True):
        latest_rag_status = rag.get_runtime_status() if (rag and hasattr(rag, "get_runtime_status")) else {}
        st.session_state.startup_security = run_startup_security_checks(rag, latest_rag_status)
        sec = st.session_state.startup_security
        log_event(
            "startup_security_checks_rerun",
            policy=sec.get("policy"),
            result=sec.get("result"),
            manifest_exists=sec.get("manifest_exists"),
            integrity_ok=sec.get("integrity_ok"),
            issues=sec.get("issues", []),
        )
        st.rerun()

    st.markdown(
        "นโยบายความปลอดภัย: " + render_security_policy_badge(sec.get("policy", "unknown")),
        unsafe_allow_html=True,
    )

    if sec.get("result") == "pass":
        st.success("ผลตรวจเริ่มระบบ: ผ่าน")
    elif sec.get("result") == "warn":
        st.warning("ผลตรวจเริ่มระบบ: เตือน")
    else:
        st.error("ผลตรวจเริ่มระบบ: ล้มเหลว")

    # Version indicator for deployment verification
    st.caption("Build: 2025-06-10-v3-section-priority")
    
    st.divider()
    st.markdown(icon_label("image_search", "โหมดค้นคืน", variant="section"), unsafe_allow_html=True)
    if VISUAL_RETRIEVAL_ENABLED:
        st.session_state.retrieval_mode = "auto"
        if FAST_RETRIEVAL_MODE:
            st.caption("Retrieval Mode (เร็ว): `Text-first, Visual only whenถามหา 'ภาพ/แผนภาพ' ชัดเจน`")
        else:
            st.caption("Retrieval Mode (อัตโนมัติ): `Text-first, Visual when structure/operation query`")

        st.session_state.visual_backend = FORCED_VISUAL_BACKEND
        st.caption(f"Visual Backend (บังคับ): `{FORCED_VISUAL_BACKEND}`")
        # Fast path in current phase: force BM25 and disable heavy post-retrieval steps.
        st.session_state.visual_use_vlm_rerank = False
        st.session_state.visual_use_grounding = False
        st.session_state.visual_sparse_strategy = "bm25"
        st.caption("Sparse Fusion Mode (บังคับ): `CoPali + BM25`")
        if st.session_state.visual_use_vlm_rerank:
            st.caption("VLM Re-rank: `on`")
        if st.session_state.visual_use_grounding:
            st.caption("Visual Grounding: `on`")
        st.caption(f"Fast Retrieval Mode: `{'on' if FAST_RETRIEVAL_MODE else 'off'}`")
        st.session_state.visual_endpoint_url = resolve_colpali_endpoint_url("")
        st.caption(f"endpoint (บังคับจาก .env): `{st.session_state.visual_endpoint_url or 'not set'}`")

        if st.button("Run Visual Endpoint Health Check", use_container_width=True):
            endpoint_for_health = resolve_colpali_endpoint_url(str(st.session_state.get("visual_endpoint_url", "")))
            health = run_visual_endpoint_healthcheck(
                endpoint_for_health,
                HF_TOKEN,
                timeout_sec=VISUAL_RETRIEVAL_ENDPOINT_TIMEOUT_SEC,
            )
            st.session_state.visual_endpoint_health = health
            log_event(
                "visual_endpoint_healthcheck",
                ok=health.get("ok"),
                endpoint=health.get("endpoint_url", ""),
                validation=health.get("validation", ""),
                error=health.get("error", ""),
            )

        health = st.session_state.get("visual_endpoint_health")
        if health:
            if health.get("backend") == "local":
                st.success("Using local backend (BM25 + Dense), endpoint not required")
            elif health.get("ok"):
                st.success(
                    f"Endpoint พร้อมใช้งาน (latency={health.get('latency_ms', 'n/a')} ms, validation={health.get('validation', 'ok')})"
                )
            else:
                st.warning(f"Endpoint ยังไม่พร้อม: {health.get('error', health.get('validation', 'unknown'))}")
            st.caption(f"log: {VISUAL_ENDPOINT_HEALTHCHECK_LOG}")

        if st.button("ทดสอบ query มาตรฐาน 3 ข้อ", use_container_width=True):
            with st.spinner("กำลังทดสอบ retrieval มาตรฐาน 3 ข้อ..."):
                report = run_visual_standard_query_suite(
                    backend=str(st.session_state.get("visual_backend", VISUAL_RETRIEVAL_BACKEND)).strip().lower() or "auto",
                    endpoint_url=str(st.session_state.get("visual_endpoint_url", VISUAL_RETRIEVAL_ENDPOINT_URL)).strip(),
                    use_vlm_rerank=bool(st.session_state.get("visual_use_vlm_rerank", VISUAL_RETRIEVAL_USE_VLM_RERANK)),
                    use_visual_grounding=bool(st.session_state.get("visual_use_grounding", VISUAL_RETRIEVAL_USE_GROUNDING)),
                    sparse_strategy=str(st.session_state.get("visual_sparse_strategy", VISUAL_SPARSE_STRATEGY)),
                )
            st.session_state.visual_standard_query_report = report
            log_event(
                "visual_standard_query_test",
                ok=report.get("ok"),
                summary=report.get("summary", {}),
                backend=report.get("config", {}).get("backend", ""),
            )

        suite = st.session_state.get("visual_standard_query_report")
        if suite:
            summary = suite.get("summary", {})
            if suite.get("ok"):
                st.success(
                    f"มาตรฐานผ่านทั้งหมด ({summary.get('passed', 0)}/{summary.get('total', 0)})"
                )
            else:
                st.error(
                    f"มาตรฐานไม่ผ่าน ({summary.get('failed', 0)} fail จาก {summary.get('total', 0)})"
                )
            with st.expander("ผลทดสอบมาตรฐาน", expanded=False):
                for row in suite.get("results", []):
                    mark = "PASS" if row.get("pass") else "FAIL"
                    st.markdown(f"- `{mark}` {row.get('query', '')}")
                    st.caption(
                        f"backend={row.get('backend', '-')}, colpali_status={row.get('colpali_status', '-')}, "
                        f"hits={row.get('hits', 0)}, evidence_ok={row.get('evidence_ok', False)}, "
                        f"latency_ms={row.get('latency_ms', 'n/a')}, error={row.get('error', '-')}, "
                        f"reasons={row.get('reasons', [])}, "
                        f"region_hit_ratio_at_k={row.get('region_hit_ratio_at_k', 'n/a')}, "
                        f"crop_completeness_proxy={row.get('crop_completeness_proxy', 'n/a')}, "
                        f"small_region_ratio_at_k={row.get('small_region_ratio_at_k', 'n/a')}, "
                        f"region_quality_mean={row.get('region_quality_mean', 'n/a')}, "
                        f"unique_figure_refs_at_k={row.get('unique_figure_refs_at_k', 'n/a')}, "
                        f"page_diversity_at_k={row.get('page_diversity_at_k', 'n/a')}, "
                        f"procedure_step_coverage_proxy={row.get('procedure_step_coverage_proxy', 'n/a')}"
                    )
                st.caption(f"log: {VISUAL_STANDARD_QUERY_LOG}")
    else:
        st.caption("โหมด Visual Retrieval ถูกปิดด้วย environment (`VISUAL_RETRIEVAL_ENABLED=0`)")

    dense_ok = bool(rag_runtime_status.get("dense_ready", True))
    reranker_ok = bool(rag_runtime_status.get("reranker_ready", False))
    status_lines = [
        f"- Dense retrieval: {'พร้อม' if dense_ok else 'ไม่พร้อม'}",
        f"- Reranker: {'พร้อม' if reranker_ok else 'ไม่พร้อม'}",
        (
            f"- Retrieval depth: top_k={rag_runtime_status.get('default_top_k', RETRIEVE_TOP_K)}, "
            f"top_n={rag_runtime_status.get('default_top_n', RETRIEVE_TOP_N)}"
        ),
        f"- Retrieval mode: {st.session_state.get('retrieval_mode', 'text')}",
    ]
    if VISUAL_RETRIEVAL_ENABLED:
        status_lines.append(f"- Visual backend: {st.session_state.get('visual_backend', VISUAL_RETRIEVAL_BACKEND)}")
        sparse_strategy_active = str(
            st.session_state.get("visual_sparse_strategy", VISUAL_SPARSE_STRATEGY)
        ).strip().lower()
        status_lines.append(f"- Sparse strategy: {sparse_strategy_active.upper()}")
        if sparse_strategy_active == "splade" and VISUAL_RETRIEVAL_USE_SPLADE:
            status_lines.append(
                (
                    f"- SPLADE runtime: mode={VISUAL_RETRIEVAL_SPLADE_MODE}, "
                    f"provider={VISUAL_RETRIEVAL_SPLADE_PROVIDER}, model={VISUAL_RETRIEVAL_SPLADE_MODEL}"
                )
            )
        if VISUAL_METADATA_FILTER_STRICT:
            status_lines.append("- Metadata pre-filter strict: True")
        if VISUAL_INTENT_USE_LLM:
            status_lines.append(
                (
                    f"- Intent pre-filter: use_llm=True, "
                    f"min_conf={VISUAL_INTENT_MIN_CONFIDENCE:.2f}, max_topic_ids={VISUAL_INTENT_MAX_TOPIC_IDS}"
                )
            )
    status_lines.append(
        (
            f"- Citation policy: strict={STRICT_CITATION_ENFORCEMENT}, "
            f"ratio>={MIN_CITED_CLAIM_RATIO:.2f}"
            + (f" (visual>={VISUAL_CITATION_MIN_CLAIM_RATIO:.2f})" if VISUAL_RETRIEVAL_ENABLED else "")
        )
    )
    st.markdown("\n".join(status_lines))

    with st.expander("รายละเอียดระบบ", expanded=False):
        if rag is None:
            st.error("ระบบ RAG ใช้งานไม่ได้ กรุณาตรวจสอบดัชนีและทรัพยากรโมเดล")
        st.caption(
            "Manifest: "
            + (f"พบ (`{sec.get('manifest_path', '')}`)" if sec.get("manifest_exists") else "ไม่พบ")
        )

        integrity_ok = sec.get("integrity_ok")
        if integrity_ok is True:
            st.caption("Index integrity: ผ่าน")
        elif integrity_ok is False:
            st.caption(f"Index integrity: ไม่ผ่าน ({sec.get('integrity_error', 'unknown')})")
        else:
            st.caption("Index integrity: ยังไม่ได้ประเมิน")

        if not dense_ok:
            st.caption(f"Dense error: {rag_runtime_status.get('dense_error', 'unknown')}")
        if reranker_ok:
            model_name = rag_runtime_status.get("reranker_model", "unknown")
            load_sec = rag_runtime_status.get("reranker_load_seconds")
            if isinstance(load_sec, (int, float)):
                st.caption(f"Reranker: {model_name} ({load_sec:.2f}s)")
            else:
                st.caption(f"Reranker: {model_name}")
        elif rag_runtime_status.get("reranker_error"):
            st.caption(f"Reranker error: {rag_runtime_status.get('reranker_error')}")

        st.caption(
            "Evidence policy: "
            f"strict_topic={RETRIEVE_STRICT_TOPIC}, "
            f"strict_structure={RETRIEVE_STRICT_STRUCTURE}, "
            f"strict_section_only={RETRIEVE_STRICT_SECTION_ONLY}, "
            f"allow_offtopic_docs={RETRIEVE_ALLOW_OFFTOPIC_DOCS}, "
            f"min_docs={EVIDENCE_MIN_DOCS}, "
            f"min_topic_ratio={EVIDENCE_MIN_TOPIC_MATCH_RATIO}, "
            f"min_avg_rrf={EVIDENCE_MIN_AVG_RRF}"
        )
        st.caption(
            "Citation policy: "
            f"strict={STRICT_CITATION_ENFORCEMENT}, "
            f"min_cited_claim_ratio={MIN_CITED_CLAIM_RATIO}, "
            f"min_cited_claim_ratio_visual={VISUAL_CITATION_MIN_CLAIM_RATIO}, "
            f"auto_repair={CITATION_REPAIR_ENABLED}"
        )

    # =====================================================================
    # RAGAS Evaluation Dashboard
    # =====================================================================
    st.divider()
    st.markdown(icon_label("analytics", "การประเมินผล RAGAS", variant="section"), unsafe_allow_html=True)

    if st.session_state.current_scores:
        scores = st.session_state.current_scores.get("scores", {})
        reasoning = st.session_state.current_scores.get("reasoning", "")

        # --- Helper: colored progress bar via HTML ---
        def _progress_html(label: str, value: float, color: str) -> str:
            pct = max(0.0, min(value, 1.0)) * 100
            return (
                f'<div style="margin-bottom:12px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">'
                f'<span>{label}</span><span style="font-weight:600;">{pct:.0f}%</span></div>'
                f'<div style="background:#e9ecef;border-radius:6px;height:10px;overflow:hidden;">'
                f'<div style="width:{pct}%;height:100%;background:{color};border-radius:6px;'
                f'transition:width .4s ease;"></div></div></div>'
            )

        faith = scores.get("faithfulness", 0)
        relev = scores.get("relevance", 0)
        ctx_p = scores.get("context_precision", 0)

        # Pick colors: green >= 0.7, orange >= 0.4, red < 0.4
        def _color(v: float) -> str:
            if v >= 0.7: return "#28a745"
            if v >= 0.4: return "#fd7e14"
            return "#dc3545"

        html_bars = (
            _progress_html("ความซื่อสัตย์ (Faithfulness)", faith, _color(faith))
            + _progress_html("ความตรงประเด็น (Relevance)", relev, _color(relev))
            + _progress_html("ความแม่น Context (Precision)", ctx_p, _color(ctx_p))
        )
        st.markdown(html_bars, unsafe_allow_html=True)

        # Average score badge
        avg = (faith + relev + ctx_p) / 3
        avg_color = _color(avg)
        st.markdown(
            f'<div style="text-align:center;margin:8px 0 4px 8px;">'
            f'<span style="background:{avg_color};color:white;padding:4px 14px;'
            f'border-radius:12px;font-size:14px;font-weight:600;">'
            f'คะแนนเฉลี่ย {avg*100:.0f}%</span></div>',
            unsafe_allow_html=True,
        )

        # Reasoning expander
        if reasoning:
            with st.expander("📝 เหตุผลจาก AI Judge", expanded=False):
                st.info(reasoning)
    else:
        st.caption("ยังไม่มีผลประเมิน — ถามคำถามเพื่อเริ่มต้น")

    # =====================================================================
    # Clear session button (red)
    # =====================================================================
    st.divider()
    if st.button("🗑️ ล้างข้อมูลเซสชัน", use_container_width=True, key="btn_clear_session", type="primary"):
        st.session_state.clear()
        st.rerun()
    st.markdown("""
    <style>
        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: #dc3545 !important;
            color: white !important;
            border: 1px solid #dc3545 !important;
        }
        [data-testid="stSidebar"] button[kind="primary"]:hover {
            background-color: #b02a37 !important;
            border-color: #b02a37 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Main Chat Area ---
st.markdown(icon_label("school", "ติวเตอร์ AI วิชาโครงสร้างข้อมูล", variant="title"), unsafe_allow_html=True)

# Sticky Tabs CSS (Dark Theme Compatible)
st.markdown("""
<style>
    /* Make tabs sticky at top - compatible with dark theme */
    div[data-testid="stTabs"] {
        position: sticky;
        top: 0;
        z-index: 100;
        background-color: #0e1117;
        padding-top: 10px;
        border-bottom: 2px solid #2d3139;
    }
    /* Add some space below tabs */
    div[data-testid="stTabContent"] {
        padding-top: 20px;
    }
    /* Ensure tab labels are visible */
    div[data-testid="stTabs"] button p {
        color: #fafafa !important;
    }
</style>
""", unsafe_allow_html=True)

# Tabs Layout: Chat | OOS Sets (A,B,C) | Expert Evaluation | Per-Evaluator Matrix | Research Summary
tab_chat, tab_oos_a, tab_oos_b, tab_oos_c, tab_ioc, tab_evaluators, tab_research = st.tabs([
    "💬 แชท", "❓ OOS ชุด A", "❓ OOS ชุด B", "❓ OOS ชุด C", "📝 ประเมินผล IOC", "👥 สถิติรายผู้ประเมิน", "📊 สรุปผลวิจัย"
])

with tab_chat:
    inline_image_limit = INLINE_EVIDENCE_IMAGE_MAX
    if st.session_state.get("retrieval_mode") == "visual" and inline_image_limit <= 0:
        inline_image_limit = 2

    # แสดงประวัติ
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(role_chip_html(msg["role"]), unsafe_allow_html=True)
            st.markdown(clean_icon_text(msg["content"]))

            # เรนเดอร์ Mermaid หากมีในข้อความ
            mermaid_blocks = re.findall(r"```\s*mermaid\s*\n(.*?)\n\s*```", msg["content"], re.DOTALL)
            for code in mermaid_blocks:
                render_mermaid(code)

            msg_images = msg.get("evidence_images", [])
            if not msg_images and msg.get("sources"):
                msg_images = collect_evidence_images_from_sources(
                    msg.get("sources", []),
                    preferred_image_limit(inline_image_limit),
                    preferred_page_keys=msg.get("evidence_preferred_pages", []) if isinstance(msg, dict) else [],
                )
            if msg_images:
                render_evidence_images(msg_images)

            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 แหล่งที่มาข้อมูล"):
                    scope_pages = []
                    scope_section = ""
                    if isinstance(msg["sources"], list) and msg["sources"]:
                        first_src = msg["sources"][0] or {}
                        scope_pages = first_src.get("section_scope_pages", []) or []
                        scope_section = str(first_src.get("target_section_id", "")).strip()
                    display_pages = _scope_pages_for_display(scope_pages)
                    if scope_section and display_pages:
                        st.caption(f"📚 หัวข้อ {scope_section} | ช่วงหน้า {', '.join(display_pages)}")

                    filtered_sources = list(msg["sources"])
                    if display_pages:
                        allowed = set(display_pages)
                        in_scope = [s for s in filtered_sources if str((s or {}).get("page", "")).strip() in allowed]
                        if in_scope:
                            filtered_sources = in_scope

                    for s in filtered_sources:
                        badge = _friendly_source_badge(s)
                        score = s.get("rrf_score", None)
                        score_text = f" | rrf={score}" if isinstance(score, (int, float)) else ""
                        snippet = str(s.get("source_excerpt", s.get("content", ""))).strip()
                        figure_refs = s.get("figure_refs", [])
                        refs_text = ""
                        if isinstance(figure_refs, list) and figure_refs:
                            refs_text = " | refs=" + ", ".join(str(x).strip() for x in figure_refs[:2] if str(x).strip())
                        st.caption(f"📌 {badge}{score_text}{refs_text}: {snippet[:300]}...")

    # Input
    if prompt := st.chat_input("ถามเกี่ยวกับโครงสร้างข้อมูล..."):
        log_event("chat_turn_started", prompt_len=len(prompt))
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(role_chip_html("user"), unsafe_allow_html=True)
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.markdown(role_chip_html("assistant"), unsafe_allow_html=True)
            with st.status("กำลังวิเคราะห์คำถาม...", expanded=False) as status:
                topic_name = classify_topic(prompt)
                deterministic_topic = best_topic_label_from_question(prompt, topic_structure)
                if deterministic_topic:
                    topic_name = deterministic_topic
                    log_event("topic_hint_overridden_by_deterministic", topic=topic_name)
                if topic_exists(topic_name, topic_structure):
                    st.session_state.visited_topics.add(topic_name)

                topic_hint = topic_hint_for_retrieval(topic_name, topic_structure)
                require_structure = question_requires_structure(prompt)
                response_stream, sources, context = generate_response(
                    prompt,
                    topic_hint=topic_hint,
                    require_structure=require_structure,
                )
                status.update(label="ประมวลผลเสร็จสิ้น", state="complete")

            if isinstance(response_stream, str) and (
                response_stream.startswith("ERROR:") or response_stream.startswith("ABSTAIN:")
            ):
                is_abstain = response_stream.startswith("ABSTAIN:")
                display_text_raw = response_stream.split(":", 1)[1].strip() if ":" in response_stream else response_stream
                display_text = humanize_answer_citations_for_user(display_text_raw, sources)
                if is_abstain:
                    log_event("assistant_abstain_response", topic=topic_name, reason=display_text)
                    st.warning(display_text)
                else:
                    log_event("assistant_error_response", topic=topic_name, error=response_stream)
                    st.error(response_stream)

                evidence_preferred_pages = extract_cited_source_page_keys(display_text_raw, sources)
                evidence_images = []
                evidence_images = collect_evidence_images_from_sources(
                    sources,
                    preferred_image_limit(inline_image_limit),
                    preferred_page_keys=evidence_preferred_pages,
                )
                if evidence_images:
                    render_evidence_images(evidence_images)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": display_text,
                        "sources": sources,
                        "evidence_images": evidence_images,
                        "evidence_preferred_pages": evidence_preferred_pages,
                    }
                )
                st.rerun()
            else:
                stream_finish_reason = {"value": None}

                def stream_parser(stream):
                    if stream is None:
                        return
                    for chunk in stream:
                        try:
                            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                                choice = chunk.choices[0]
                                finish_reason = getattr(choice, "finish_reason", None)
                                if finish_reason:
                                    stream_finish_reason["value"] = finish_reason

                                delta = getattr(choice, 'delta', None)
                                if delta:
                                    content = getattr(delta, 'content', None)
                                    if content:
                                        yield clean_icon_text(content)
                        except (IndexError, AttributeError):
                            continue

                answer_placeholder = st.empty()
                full_response = answer_placeholder.write_stream(stream_parser(response_stream))
                full_response = full_response or ""

                target_sid_for_turn = str((sources[0] if sources else {}).get("target_section_id", "")).strip()
                auto_continue_enabled = bool(
                    CHAT_AUTO_CONTINUE and (not target_sid_for_turn or SECTION_AUTO_CONTINUE)
                )
                if auto_continue_enabled and _response_needs_continuation(full_response, stream_finish_reason["value"]):
                    cont_max_tokens = int(CHAT_CONTINUE_MAX_TOKENS)
                    if target_sid_for_turn in OPERATION_HEAVY_SECTION_IDS:
                        cont_max_tokens = max(cont_max_tokens, int(SECTION_CONTINUE_MAX_TOKENS))
                    continuation = _continue_answer(
                        prompt,
                        context,
                        full_response,
                        max_tokens=cont_max_tokens,
                    )
                    if continuation:
                        continuation = clean_icon_text(continuation)
                        full_response = _merge_with_overlap(full_response, continuation)
                        answer_placeholder.markdown(full_response)
                        log_event(
                            "chat_auto_continued",
                            finish_reason=stream_finish_reason["value"],
                            continuation_len=len(continuation),
                            max_tokens=cont_max_tokens,
                            target_section_id=target_sid_for_turn,
                        )

                full_response = clean_icon_text(full_response)
                full_response = strip_generated_mermaid_blocks(full_response)
                streamed_response = full_response  # preserve original streaming output
                full_response, self_check_meta = self_check_grounded_answer(
                    prompt,
                    full_response,
                    context,
                    sources,
                )
                log_event("self_check_grounding", **self_check_meta)
                # Guard: never replace with a significantly shorter version
                if len(full_response) < len(streamed_response) * 0.8:
                    log_event("self_check_rollback", before_len=len(streamed_response), after_len=len(full_response))
                    full_response = streamed_response
                answer_placeholder.markdown(full_response)
                full_response = normalize_answer_citations(full_response, sources)
                full_response = strip_context_placeholder_citations(full_response)
                if STRICT_CITATION_ENFORCEMENT:
                    gated_response, sentence_gate_stats = enforce_sentence_level_citation_gate(
                        full_response,
                        sources,
                        context_text=context,
                    )
                    log_event("sentence_citation_gate", **sentence_gate_stats)
                    if gated_response.strip():
                        full_response = gated_response
                else:
                    full_response = autofill_missing_claim_citations(full_response, sources)
                full_response = normalize_answer_citations(full_response, sources)
                full_response = strip_context_placeholder_citations(full_response)
                citation_min_ratio = _effective_citation_ratio_target(sources)
                citation_validation = validate_claim_citations(full_response, sources, min_ratio=citation_min_ratio)
                log_event(
                    "citation_validation",
                    claim_count=citation_validation.get("claim_count", 0),
                    cited_claim_ratio=citation_validation.get("cited_claim_ratio", 0.0),
                    target_ratio=citation_validation.get("target_ratio", citation_min_ratio),
                    citation_ok=citation_validation.get("citation_ok", False),
                    unknown_citations=citation_validation.get("unknown_citations", []),
                    unknown_context_placeholders=citation_validation.get("unknown_context_placeholders", []),
                )

                if STRICT_CITATION_ENFORCEMENT and not citation_validation.get("citation_ok", False):
                    # Second-pass autofill/normalize for tougher malformed outputs.
                    autofilled = autofill_missing_claim_citations(full_response, sources)
                    autofilled = normalize_answer_citations(autofilled, sources)
                    autofilled = strip_context_placeholder_citations(autofilled)
                    autofilled_validation = validate_claim_citations(autofilled, sources, min_ratio=citation_min_ratio)
                    log_event(
                        "citation_autofill_attempt",
                        citation_ok=autofilled_validation.get("citation_ok", False),
                        cited_claim_ratio=autofilled_validation.get("cited_claim_ratio", 0.0),
                        target_ratio=autofilled_validation.get("target_ratio", citation_min_ratio),
                        unknown_citations=autofilled_validation.get("unknown_citations", []),
                        unknown_context_placeholders=autofilled_validation.get("unknown_context_placeholders", []),
                    )
                    if autofilled_validation.get("citation_ok", False):
                        full_response = autofilled
                        citation_validation = autofilled_validation
                        answer_placeholder.markdown(full_response)
                    elif (
                        float(autofilled_validation.get("cited_claim_ratio", 0.0))
                        > float(citation_validation.get("cited_claim_ratio", 0.0))
                    ):
                        full_response = autofilled
                        citation_validation = autofilled_validation
                        answer_placeholder.markdown(full_response)

                if STRICT_CITATION_ENFORCEMENT and not citation_validation.get("citation_ok", False):
                    repaired = ""
                    if CITATION_REPAIR_ENABLED:
                        repaired = repair_answer_citations(prompt, full_response, context, sources)
                        repaired = clean_icon_text(repaired)
                        repaired = strip_generated_mermaid_blocks(repaired)
                        repaired = autofill_missing_claim_citations(repaired, sources)
                        repaired = normalize_answer_citations(repaired, sources)
                        repaired = strip_context_placeholder_citations(repaired)
                        if repaired:
                            repaired_validation = validate_claim_citations(repaired, sources, min_ratio=citation_min_ratio)
                            log_event(
                                "citation_repair_attempt",
                                repaired_len=len(repaired),
                                citation_ok=repaired_validation.get("citation_ok", False),
                                cited_claim_ratio=repaired_validation.get("cited_claim_ratio", 0.0),
                                target_ratio=repaired_validation.get("target_ratio", citation_min_ratio),
                                unknown_citations=repaired_validation.get("unknown_citations", []),
                                unknown_context_placeholders=repaired_validation.get("unknown_context_placeholders", []),
                            )
                            if repaired_validation.get("citation_ok", False):
                                full_response = repaired
                                citation_validation = repaired_validation
                                answer_placeholder.markdown(full_response)
                            elif (
                                float(repaired_validation.get("cited_claim_ratio", 0.0))
                                > float(citation_validation.get("cited_claim_ratio", 0.0))
                                and not repaired_validation.get("unknown_citations")
                            ):
                                full_response = repaired
                                citation_validation = repaired_validation
                                answer_placeholder.markdown(full_response)

                    if not citation_validation.get("citation_ok", False):
                        reasons = []
                        low_coverage = citation_validation.get("cited_claim_ratio", 0.0) < citation_validation.get(
                            "target_ratio",
                            citation_min_ratio,
                        )
                        has_unknown = bool(citation_validation.get("unknown_citations"))
                        if low_coverage:
                            reasons.append("low_citation_coverage")
                        if has_unknown:
                            reasons.append("unknown_citations")
                        # Soft-pass: if only coverage is low but citations are known, keep grounded answer.
                        if low_coverage and not has_unknown:
                            log_event(
                                "citation_enforcement_soft_pass",
                                reasons=reasons,
                                claim_count=citation_validation.get("claim_count", 0),
                                cited_claim_ratio=citation_validation.get("cited_claim_ratio", 0.0),
                                target_ratio=citation_validation.get("target_ratio", citation_min_ratio),
                                unknown_citations=[],
                            )
                        else:
                            full_response = (
                                "ยังไม่สามารถตอบแบบยืนยันหลักฐานได้ เนื่องจากการอ้างอิงไม่ผ่านนโยบายความเข้มงวด "
                                f"(เหตุผล: {', '.join(reasons) or 'citation_validation_failed'}) "
                                "โปรดลองถามใหม่แบบเฉพาะเจาะจงหัวข้อในเอกสาร"
                            )
                            sources = []
                            answer_placeholder.markdown(full_response)
                            log_event(
                                "citation_enforcement_abstain",
                                reasons=reasons,
                                claim_count=citation_validation.get("claim_count", 0),
                                cited_claim_ratio=citation_validation.get("cited_claim_ratio", 0.0),
                                target_ratio=citation_validation.get("target_ratio", citation_min_ratio),
                                unknown_citations=citation_validation.get("unknown_citations", []),
                                unknown_context_placeholders=citation_validation.get("unknown_context_placeholders", []),
                            )

                section_gate = evaluate_operation_step_coverage(
                    full_response,
                    prompt,
                    resolve_target_section_id(topic_hint, prompt),
                )
                log_event("text_step_coverage_gate", gate=section_gate)
                if section_gate.get("enabled") and not section_gate.get("pass", False) and bool(sources):
                    repaired_step = repair_answer_for_step_coverage(
                        prompt,
                        full_response,
                        context,
                        sources,
                        section_gate,
                    )
                    repaired_step = clean_icon_text(repaired_step)
                    repaired_step = strip_generated_mermaid_blocks(repaired_step)
                    repaired_step = sanitize_broken_citation_brackets(repaired_step)
                    repaired_step = normalize_answer_citations(repaired_step, sources)
                    repaired_step = strip_context_placeholder_citations(repaired_step)
                    if repaired_step:
                        repaired_cit = validate_claim_citations(
                            repaired_step,
                            sources,
                            min_ratio=_effective_citation_ratio_target(sources),
                        )
                        repaired_gate = evaluate_operation_step_coverage(
                            repaired_step,
                            prompt,
                            resolve_target_section_id(topic_hint, prompt),
                        )
                        log_event(
                            "step_coverage_repair_attempt",
                            before_matched=section_gate.get("matched_groups", 0),
                            after_matched=repaired_gate.get("matched_groups", 0),
                            required_groups=repaired_gate.get("required_groups", 0),
                            citation_ok=repaired_cit.get("citation_ok", False),
                            target_section_id=repaired_gate.get("target_section_id", ""),
                        )
                        if repaired_cit.get("citation_ok", False) and (
                            int(repaired_gate.get("matched_groups", 0))
                            >= int(section_gate.get("matched_groups", 0))
                        ):
                            full_response = repaired_step
                            section_gate = repaired_gate
                            answer_placeholder.markdown(full_response)

                if STRICT_CITATION_ENFORCEMENT and section_gate.get("enabled") and not section_gate.get("pass", False):
                    full_response = (
                        "ยังไม่สามารถตอบแบบยืนยันหลักฐานได้ เนื่องจากความครอบคลุมขั้นตอนไม่พอ "
                        "(เหตุผล: low_step_coverage_text)"
                    )
                    sources = []
                    answer_placeholder.markdown(full_response)

                # Section-mode compacting: reduce latency spillover and long repetitive outputs.
                answer_sid = resolve_target_section_id(topic_hint, prompt) or str(
                    (sources[0] if sources else {}).get("target_section_id", "")
                ).strip()
                if answer_sid:
                    should_compact = False  # Disabled: compaction degrades answer quality
                    if False:  # was: answer_sid in OPERATION_HEAVY_SECTION_IDS and not OPERATION_SECTION_ENABLE_COMPACTION
                        should_compact = False
                        log_event(
                            "section_answer_compaction_skipped",
                            target_section_id=answer_sid,
                            reason="operation_section_compaction_disabled",
                            answer_len=len(full_response or ""),
                        )

                    if should_compact:
                        max_claim_lines = 6 if answer_sid in {"3.3.2", "3.3.3"} else 7
                        compact = trim_cited_claim_lines(full_response, max_claim_lines=max_claim_lines)
                        if len(compact or "") > int(SECTION_CONCISE_TRIGGER_CHARS):
                            compact2 = compact_section_answer_deterministic(compact, answer_sid)
                            if compact2:
                                compact = compact2
                        compact = sanitize_broken_citation_brackets(compact)
                        compact = normalize_answer_citations(compact, sources)
                        compact = strip_context_placeholder_citations(compact)
                        compact_validation = validate_claim_citations(
                            compact,
                            sources,
                            min_ratio=_effective_citation_ratio_target(sources),
                        )
                        compact_tail_ok = not _response_needs_continuation(compact, None)
                        if compact_validation.get("citation_ok", False) and compact.strip() and compact_tail_ok:
                            log_event(
                                "section_answer_compacted",
                                target_section_id=answer_sid,
                                before_len=len(full_response or ""),
                                after_len=len(compact or ""),
                                claim_count=compact_validation.get("claim_count", 0),
                                cited_claim_ratio=compact_validation.get("cited_claim_ratio", 0.0),
                            )
                            full_response = compact
                            answer_placeholder.markdown(full_response)
                        elif compact_validation.get("citation_ok", False) and compact.strip() and not compact_tail_ok:
                            log_event(
                                "section_answer_compaction_rejected",
                                target_section_id=answer_sid,
                                reason="incomplete_tail_after_compaction",
                                compact_len=len(compact or ""),
                            )

                mermaid_matches = re.findall(r"```\s*mermaid\s*\n(.*?)\n\s*```", full_response, re.DOTALL)
                for m_code in mermaid_matches:
                    render_mermaid(m_code)

                evidence_preferred_pages = extract_cited_source_page_keys(full_response, sources)
                evidence_images = []
                evidence_images = collect_evidence_images_from_sources(
                    sources,
                    preferred_image_limit(inline_image_limit),
                    preferred_page_keys=evidence_preferred_pages,
                )
                display_response = humanize_answer_citations_for_user(full_response, sources)
                answer_placeholder.markdown(display_response)
                if evidence_images:
                    render_evidence_images(evidence_images)

                if ENABLE_JUDGE_EVALUATION:
                    with st.spinner("กำลังประเมินผล..."):
                        judge_res = run_judge(prompt, full_response, context)
                        st.session_state.current_scores = judge_res
                else:
                    judge_res = {
                        "scores": {"faithfulness": 0.0, "relevance": 0.0, "context_precision": 0.0},
                        "reasoning": "ปิด Judge evaluation ชั่วคราวเพื่อลด latency (ENABLE_JUDGE_EVALUATION=0)",
                    }
                    st.session_state.current_scores = None

                rag_status = rag.get_runtime_status() if (rag and hasattr(rag, "get_runtime_status")) else {}
                append_research_log(prompt, topic_name, full_response, judge_res, rag_status)
                log_event(
                    "chat_turn_completed",
                    topic=topic_name,
                    topic_hint=topic_hint or "",
                    require_structure=bool(require_structure),
                    answer_len=len(full_response),
                    source_count=len(sources),
                    judge_scores=judge_res.get("scores", {}),
                    reranker_ready=rag_status.get("reranker_ready"),
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": display_response,
                    "raw_content": full_response,
                    "sources": sources,
                    "evidence_images": evidence_images,
                    "evidence_preferred_pages": evidence_preferred_pages,
                })
                st.rerun()

# Module-level functions for IOC evaluation (moved from inside tab_ioc block)
def load_ioc_evaluations():
    """Load IOC evaluations from CSV file."""
    if not IOC_EVAL_FILE.exists():
        return []
    try:
        with open(IOC_EVAL_FILE, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                # Clean BOM characters from keys and values
                cleaned = {}
                for key, value in row.items():
                    if key:
                        clean_key = key.lstrip('\ufeff').strip()
                        cleaned[clean_key] = value.strip() if value else ""
                rows.append(cleaned)
            return rows
    except Exception as e:
        st.error(f"Error loading evaluations: {e}")
        return []


def save_ioc_evaluation(data: dict):
    """Save IOC evaluation to CSV file."""
    fieldnames = [
        "timestamp", "evaluator_name", "question", "answer", "question_type",
        "system_behavior", "ioc_score", "comments"
    ]
    file_exists = IOC_EVAL_FILE.exists()
    try:
        with open(IOC_EVAL_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        return True
    except Exception as e:
        st.error(f"❌ บันทึกไม่สำเร็จ: {e}")
        return False


def calculate_confusion_matrix(evaluations):
    """Calculate confusion matrix metrics from evaluations.
    
    Standard Confusion Matrix:
    - TP: In-Scope + ตอบ (ถูกต้อง)
    - FN: In-Scope + Abstain (ผิด - ควรตอบแต่ไม่ตอบ)
    - TN: Out-of-Scope + Abstain (ถูกต้อง - ปฏิเสธถูกต้อง)
    - FP: Out-of-Scope + ตอบ (ผิด - ไม่ควรตอบแต่ตอบ)
    """
    tp = tn = fp = fn = 0
    for ev in evaluations:
        q_type = ev.get("question_type", "")
        behavior = ev.get("system_behavior", "")

        if q_type == "In-Scope (มีในเนื้อหา)":
            if behavior == "ตอบคำถามปกติ":
                tp += 1
            else:  # Abstain
                fn += 1
                
        elif q_type == "Out-of-Scope (ไม่มีในเนื้อหา)":
            if behavior == "ปฏิเสธการตอบ (Abstain)":
                tn += 1
            else:  # ตอบ
                fp += 1
                
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def calculate_confusion_matrix_by_evaluator(evaluations, evaluator_name):
    """Calculate confusion matrix for a specific evaluator."""
    evaluator_evals = [ev for ev in evaluations if ev.get("evaluator_name") == evaluator_name]
    return calculate_confusion_matrix(evaluator_evals)


with tab_ioc:
    st.markdown("### 📝 Expert Evaluation (IOC Form)")

    # Get latest Q&A from session state
    latest_q = ""
    latest_a = ""
    if st.session_state.messages:
        # Find last user-assistant pair
        for msg in reversed(st.session_state.messages):
            if msg.get("role") == "assistant":
                latest_a = msg.get("content", "")
            elif msg.get("role") == "user":
                latest_q = msg.get("content", "")
                break

    # Evaluation Form
    if latest_q and latest_a:
        with st.form("ioc_evaluation_form", clear_on_submit=True):
            st.markdown("**คำถามล่าสุด:**")
            st.info(latest_q[:200] + "..." if len(latest_q) > 200 else latest_q)
            st.markdown("**คำตอบของระบบ:**")
            st.success(latest_a[:200] + "..." if len(latest_a) > 200 else latest_a)

            st.divider()

            question_type = st.selectbox(
                "ประเภทคำถามที่ผู้ใช้ถาม",
                ["In-Scope (มีในเนื้อหา)", "Out-of-Scope (ไม่มีในเนื้อหา)"]
            )

            system_behavior = st.selectbox(
                "การทำงานของระบบ",
                ["ตอบคำถามปกติ", "ปฏิเสธการตอบ (Abstain)"]
            )

            ioc_score = st.radio(
                "IOC Score",
                ["+1: ตรงประเด็นถูกต้อง", "0: ไม่แน่ใจ/ต้องปรับปรุง", "-1: ผิดพลาด/ไม่ตรงประเด็น"],
                horizontal=True
            )

            comments = st.text_area("ข้อเสนอแนะเพิ่มเติม (Comments)", height=80)

            st.divider()
            st.markdown("**👤 ข้อมูลผู้ประเมิน (บังคับกรอก)**")

            evaluator_name = st.text_input(
                "ชื่อ-นามสกุล / รหัสผู้ประเมิน",
                placeholder="เช่น นายสมชาย ใจดี หรือ EVAL-001",
                help="สำหรับติดตามผู้ประเมินแต่ละคน (Production: ไม่นับผู้พัฒนา)"
            )

            submitted = st.form_submit_button("📋 บันทึกผลประเมิน", use_container_width=True)

            if submitted:
                # Validate evaluator name
                if not evaluator_name or not evaluator_name.strip():
                    st.error("❌ กรุณากรอกชื่อผู้ประเมินก่อนบันทึก")
                    st.stop()

                ioc_score_value = ioc_score.split(":")[0]  # Extract +1, 0, or -1
                data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "evaluator_name": evaluator_name.strip(),
                    "question": latest_q,
                    "answer": latest_a,
                    "question_type": question_type,
                    "system_behavior": system_behavior,
                    "ioc_score": ioc_score_value,
                    "comments": comments
                }
                if save_ioc_evaluation(data):
                    st.toast(f"✅ บันทึกผลประเมินสำเร็จ! (โดย: {evaluator_name.strip()})", icon="🎉")
                    st.success(f"✅ บันทึกสำเร็จ! ผู้ประเมิน: {evaluator_name.strip()}")
                    st.rerun()
    else:
        st.info("💬 ยังไม่มีประวัติการสนทนา กรุณาถามคำถามก่อนเพื่อทำการประเมิน")

    st.divider()

    # Confusion Matrix Dashboard
    st.markdown("### 📊 Confusion Matrix Dashboard")

    evaluations = load_ioc_evaluations()
    cm = calculate_confusion_matrix(evaluations)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ TP (True Positive)", cm["TP"], help="In-Scope + ตอบถูก")
    with col2:
        st.metric("✅ TN (True Negative)", cm["TN"], help="Out-of-Scope + Abstain ถูก")
    with col3:
        st.metric("⚠️ FP (Hallucination)", cm["FP"], help="Out-of-Scope แต่ดันตอบ")
    with col4:
        st.metric("❌ FN (Miss)", cm["FN"], help="In-Scope แต่ Abstain/ตอบผิด")

    # Confusion Matrix Table
    cm_data = {
        "": ["In-Scope", "Out-of-Scope"],
        "ตอบคำถาม": [cm["TP"], cm["FP"]],
        "Abstain": [cm["FN"], cm["TN"]]
    }
    st.dataframe(cm_data, use_container_width=True, hide_index=True)

    # Calculate and show accuracy/precision
    total = sum(cm.values())
    if total > 0:
        accuracy = (cm["TP"] + cm["TN"]) / total
        st.metric("🎯 Accuracy รวม", f"{accuracy:.1%}")

    # Per-Evaluator Confusion Matrix Summary
    st.markdown("### 📋 สรุปผลรายผู้ประเมิน")
    
    # Get unique evaluators
    evaluators = sorted(set(ev.get("evaluator_name", "") for ev in evaluations if ev.get("evaluator_name")))
    
    if evaluators:
        # Build summary table
        summary_data = []
        for evaluator in evaluators:
            ev_cm = calculate_confusion_matrix_by_evaluator(evaluations, evaluator)
            ev_total = sum(ev_cm.values())
            ev_accuracy = (ev_cm["TP"] + ev_cm["TN"]) / ev_total if ev_total > 0 else 0
            summary_data.append({
                "ผู้ประเมิน": evaluator,
                "จำนวนคำถาม": ev_total,
                "TP": ev_cm["TP"],
                "TN": ev_cm["TN"],
                "FP": ev_cm["FP"],
                "FN": ev_cm["FN"],
                "Accuracy": f"{ev_accuracy:.1%}"
            })
        
        # Add total row
        summary_data.append({
            "ผู้ประเมิน": "**รวม**",
            "จำนวนคำถาม": total,
            "TP": cm["TP"],
            "TN": cm["TN"],
            "FP": cm["FP"],
            "FN": cm["FN"],
            "Accuracy": f"{accuracy:.1%}" if total > 0 else "-"
        })
        
        st.dataframe(summary_data, use_container_width=True, hide_index=True)
    else:
        st.info("ยังไม่มีข้อมูลผู้ประเมิน")

    # Download button
    if IOC_EVAL_FILE.exists():
        with open(IOC_EVAL_FILE, "rb") as f:
            st.download_button(
                label="📥 ดาวน์โหลด CSV สำหรับรายงานวิจัย",
                data=f,
                file_name="expert_ioc_eval.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.caption("ยังไม่มีข้อมูลประเมินให้ดาวน์โหลด")

# Helper function to render OOS questions
def render_oos_questions(oos_list, set_name):
    """Render OOS questions list for a specific set."""
    import json
    OOS_FILE = PROJECT_ROOT / "eval" / "oos_questions_full.json"
    
    if not OOS_FILE.exists():
        st.warning(f"⚠️ ไม่พบไฟล์ {OOS_FILE}")
        return
    
    with open(OOS_FILE, "r", encoding="utf-8") as f:
        oos_data = json.load(f)
    
    all_questions = oos_data.get("oos_questions", [])
    
    # Filter to only show questions in the specified list
    questions = [q for q in all_questions if q['id'] in oos_list]
    
    st.info(f"📋 ชุดนี้มี {len(questions)} คำถาม (สำหรับผู้ประเมินคนหนึ่ง ไม่ซ้ำกับชุดอื่น)")
    
    st.divider()
    
    # Show questions in expanders with full text
    for i, q in enumerate(questions, 1):
        with st.expander(f"**{q['id']}** - {q['question']}"):
            st.markdown(f"**หมวดหมู่:** {q.get('category', '-')}")
            st.markdown(f"**เหตุผลที่เป็น OOS:** {q.get('reason', '-')}")
            st.markdown(f"**พฤติกรรมที่คาดหวัง:** {q.get('expected_behavior', '-')}")
            
            st.divider()
            
            # Copy section
            st.markdown("**📋 คัดลอกคำถาม:**")
            st.text_input(
                "",
                value=q['question'],
                key=f"copy_{set_name}_{q['id']}",
                label_visibility="collapsed"
            )
            st.caption("💡 กด **Ctrl+A** แล้ว **Ctrl+C** เพื่อคัดลอก → ไปที่แท็บ **💬 แชท** เพื่อถาม")

# Define OOS question sets (46 questions each, split from 138 total)
OOS_SET_A = [f"OOS-{i:02d}" for i in range(1, 47)]    # OOS-01 to OOS-46
OOS_SET_B = [f"OOS-{i:02d}" for i in range(47, 93)]   # OOS-47 to OOS-92
OOS_SET_C = [f"OOS-{i:02d}" for i in range(93, 139)]  # OOS-93 to OOS-138

with tab_oos_a:
    st.markdown("### ❓ ชุดคำถาม Out-of-Scope (OOS) - ชุด A")
    st.caption("สำหรับผู้ประเมินคนที่ 1 (OOS-01 ถึง OOS-46)")
    render_oos_questions(OOS_SET_A, "A")

with tab_oos_b:
    st.markdown("### ❓ ชุดคำถาม Out-of-Scope (OOS) - ชุด B")
    st.caption("สำหรับผู้ประเมินคนที่ 2 (OOS-47 ถึง OOS-92)")
    render_oos_questions(OOS_SET_B, "B")

with tab_oos_c:
    st.markdown("### ❓ ชุดคำถาม Out-of-Scope (OOS) - ชุด C")
    st.caption("สำหรับผู้ประเมินคนที่ 3 (OOS-93 ถึง OOS-138)")
    render_oos_questions(OOS_SET_C, "C")

# Per-Evaluator Confusion Matrix Tab
with tab_evaluators:
    st.markdown("### 👥 สถิติ Confusion Matrix รายผู้ประเมิน")
    st.caption("แยกตามผู้ประเมินแต่ละคน (DEV จะถูกกรองออกใน Production)")
    
    # Load all evaluations
    all_evals = load_ioc_evaluations()
    
    if not all_evals:
        st.info("ยังไม่มีข้อมูลการประเมิน")
    else:
        # Get unique evaluators
        evaluators = sorted(set(e.get("evaluator_name", "Unknown") for e in all_evals if e.get("evaluator_name")))
        
        if not evaluators:
            st.info("ไม่พบข้อมูลผู้ประเมิน")
        else:
            st.success(f"พบผู้ประเมิน {len(evaluators)} คน")
            
            # Select evaluator
            selected_evaluator = st.selectbox("👤 เลือกผู้ประเมิน", evaluators)
            
            # Filter evaluations for selected evaluator
            eval_filtered = [e for e in all_evals if e.get("evaluator_name") == selected_evaluator]
            
            # Calculate confusion matrix for this evaluator
            cm_eval = calculate_confusion_matrix(eval_filtered)
            
            st.divider()
            
            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ TP (True Positive)", cm_eval["TP"])
            with col2:
                st.metric("✅ TN (True Negative)", cm_eval["TN"])
            with col3:
                st.metric("⚠️ FP (Hallucination)", cm_eval["FP"])
            with col4:
                st.metric("❌ FN (Miss)", cm_eval["FN"])
            
            # Confusion matrix table
            cm_data = {
                "": ["In-Scope", "Out-of-Scope"],
                "ตอบคำถาม": [cm_eval["TP"], cm_eval["FP"]],
                "Abstain": [cm_eval["FN"], cm_eval["TN"]]
            }
            st.dataframe(cm_data, use_container_width=True, hide_index=True)
            
            # Accuracy
            total = sum(cm_eval.values())
            if total > 0:
                accuracy = (cm_eval["TP"] + cm_eval["TN"]) / total
                st.metric("🎯 Accuracy", f"{accuracy:.1%}")
            
            # Show evaluation count
            st.caption(f"จำนวนการประเมินทั้งหมด: {len(eval_filtered)} รายการ")
            
            # Show recent evaluations
            with st.expander("📋 รายการประเมินล่าสุด"):
                for ev in eval_filtered[-5:]:  # Last 5
                    q_type = ev.get("question_type", "-")
                    behavior = ev.get("system_behavior", "-")
                    score = ev.get("ioc_score", "-")
                    
                    # Color code based on score
                    if score in ["+1", "1"]:
                        st.success(f"**{q_type}** | {behavior} | **{score}**")
                    elif score == "-1":
                        st.error(f"**{q_type}** | {behavior} | **{score}**")
                    else:
                        st.warning(f"**{q_type}** | {behavior} | **{score}**")
                    st.caption(f"Q: {ev.get('question', '-')[:80]}...")

# Research Summary Tab
with tab_research:
    st.markdown("### 📊 สรุปผลการวิจัย (Research Summary)")
    st.caption("ผลการประเมินระบบ RAG ตามมาตรฐาน Grounded Only Policy")
    
    # Load all evaluations for research stats
    all_evals = load_ioc_evaluations()
    
    if all_evals:
        # Calculate overall confusion matrix
        cm_overall = calculate_confusion_matrix(all_evals)
        tp, tn, fp, fn = cm_overall["TP"], cm_overall["TN"], cm_overall["FP"], cm_overall["FN"]
        total = tp + tn + fp + fn
        
        if total > 0:
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Count by question type
            inscope = sum(1 for r in all_evals if r.get("question_type") == "In-Scope (มีในเนื้อหา)")
            oos = sum(1 for r in all_evals if r.get("question_type") == "Out-of-Scope (ไม่มีในเนื้อหา)")
            
            st.divider()
            
            # Main metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 IOC (Accuracy)", f"{accuracy:.1%}")
            with col2:
                st.metric("📏 Precision", f"{precision:.1%}")
            with col3:
                st.metric("🔁 Recall", f"{recall:.1%}")
            
            st.divider()
            
            # Confusion Matrix
            st.markdown("#### 📋 Confusion Matrix")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ TP", tp, f"{tp/total*100:.1f}%")
            with col2:
                st.metric("✅ TN", tn, f"{tn/total*100:.1f}%")
            with col3:
                st.metric("⚠️ FP", fp, f"{fp/total*100:.1f}%")
            with col4:
                st.metric("❌ FN", fn, f"{fn/total*100:.1f}%")
            
            st.divider()
            
            # Population stats
            st.markdown(f"**ประชากร (Population):** {total} รายการ")
            st.markdown(f"- In-Scope: {inscope} รายการ ({inscope/total*100:.1f}%)")
            st.markdown(f"- Out-of-Scope: {oos} รายการ ({oos/total*100:.1f}%)")
            
            # IOC Assessment
            st.divider()
            st.markdown("#### 🏆 การประเมิน IOC")
            if accuracy >= 0.8:
                st.success(f"✅ ระบบใช้ได้ (IOC = {accuracy:.1%} >= 80%)")
            elif accuracy >= 0.7:
                st.warning(f"⚠️ ระบบใช้ได้ (IOC = {accuracy:.1%} >= 70%)")
            else:
                st.error(f"❌ ระบบต้องปรับปรุง (IOC = {accuracy:.1%} < 70%)")
            
            # Strengths and Weaknesses
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ จุดแข็ง:**")
                st.markdown(f"- Precision สูง ({precision:.1%})")
                st.markdown(f"- FP ต่ำ ({fp/total*100:.1f}%)")
                st.markdown(f"- TN สูง ({tn/total*100:.1f}%)")
            with col2:
                st.markdown("**⚠️ จุดอ่อน:**")
                st.markdown(f"- FN สูง ({fn/total*100:.1f}%)")
                st.markdown(f"- Recall ต่ำ ({recall:.1f}%)")
                st.markdown("- Evidence Gate เก็บกว้างไป")
            
            # Recommendations
            st.divider()
            st.markdown("#### 💡 ข้อเสนอแนะสำหรับการพัฒนาต่อ")
            with st.expander("ดูข้อเสนอแนะ"):
                st.markdown("""
                1. **ปรับลด Evidence Threshold** (ปัจจุบัน 0.8 → แนะนำ 0.6)
                2. **เพิ่ม Corrective Retrieval Rounds** (ปัจจุบัน 1 → แนะนำ 3)
                3. **ปรับ ColPali Weight** (ลดลงสำหรับเอกสารข้อความหนาแน่น)
                4. **เพิ่ม Re-ranker แบบ Cross-encoder**
                5. **ปรับปรุง Citation Validation**
                6. **เพิ่ม Query Expansion**
                """)
            
            # Flowchart reference using Graphviz (Streamlit native support)
            st.divider()
            st.markdown("#### 🔄 Flowchart ระบบ")
            
            flowchart_dot = """
            digraph G {
                rankdir=TB;
                node [shape=box, style="rounded,filled", fillcolor="#e1f5fe", fontname="Arial", fontsize=10];
                edge [fontname="Arial", fontsize=9, color="#666"];
                
                // Section A - Sources
                A1 [label="PDF: data_ch1_to_ch5.pdf", fillcolor="#fff3e0"];
                A2 [label="สารบัญ: list_hitachi.txt", fillcolor="#fff3e0"];
                A3 [label="นโยบาย: Grounded Only", fillcolor="#fff3e0"];
                
                // Section B - Preparation
                B1 [label="OCR Extraction", fillcolor="#e8f5e9"];
                B2 [label="Extract Figures", fillcolor="#e8f5e9"];
                B3 [label="Visual Anchors", fillcolor="#e8f5e9"];
                B4 [label="Cleanup & QA", fillcolor="#e8f5e9"];
                
                // Section C - Indexing
                C1 [label="Build Hierarchy", fillcolor="#f3e5f5"];
                C2 [label="ColPali Corpus", fillcolor="#f3e5f5"];
                C3 [label="Hard Negatives", fillcolor="#f3e5f5"];
                C4 [label="Chapter Calibration", fillcolor="#f3e5f5"];
                
                // Section D - Online Retrieval
                D1 [label="User Query", shape=ellipse, fillcolor="#ffebee"];
                D2 [label="Query Profiling", fillcolor="#e3f2fd"];
                D3 [label="ColPali + Lexical\nRetrieval", fillcolor="#e3f2fd"];
                D4 [label="Late Fusion & RRF", fillcolor="#e3f2fd"];
                D5 [label="Apply Penalty", fillcolor="#e3f2fd"];
                
                // Section E - Validation
                E1 [label="Adaptive Crop", fillcolor="#fce4ec"];
                E2 [label="VLM Grounding\nEnsemble", fillcolor="#fce4ec"];
                E3 [label="Consensus Voting", fillcolor="#fce4ec"];
                E4 [label="Evidence Gate", shape=diamond, fillcolor="#ffccbc"];
                E5 [label="ABSTAIN", fillcolor="#ffcdd2"];
                E6 [label="Corrective Retrieval", fillcolor="#fff9c4"];
                
                // Section F - Generation
                F1 [label="Build Context", fillcolor="#e0f2f1"];
                F2 [label="Generate Answer\n(Temp=0.2)", fillcolor="#e0f2f1"];
                F3 [label="Citation Validation", shape=diamond, fillcolor="#ffccbc"];
                F4 [label="Citation Repair", fillcolor="#fff9c4"];
                F5 [label="Delivery to UI", fillcolor="#b2dfdb"];
                
                // Edges
                A1 -> B1 -> B4;
                A1 -> B2 -> B4;
                A1 -> B3 -> B4;
                B4 -> C1;
                B4 -> C2;
                A2 -> C1;
                C1 -> C3;
                C2 -> C3;
                C1 -> C4;
                C3 -> D5;
                C4 -> E4;
                
                D1 -> D2 -> D3 -> D4 -> D5;
                D5 -> E1 -> E2 -> E3 -> E4;
                E4 -> F1 [label="Yes"];
                E4 -> E5 [label="Fail"];
                E4 -> E6 [label="No"];
                E6 -> D3;
                E5 -> F5;
                
                F1 -> F2 -> F3;
                F3 -> F5 [label="Yes"];
                F3 -> F4 [label="No"];
                F4 -> F5;
                
                // Subgraph labels
                labelloc="t";
                label="RAG System Architecture";
            }
            """
            st.graphviz_chart(flowchart_dot, use_container_width=True)
            
            st.caption("📊 สถาปัตยกรรมระบบแบ่งเป็น 6 ส่วนหลัก: Preparation → Indexing → Retrieval → Validation → Generation → Delivery")
            
            # Link to full documentation
            st.divider()
            research_doc = PROJECT_ROOT / "docs" / "research_development.md"
            if research_doc.exists():
                st.info(f"📄 เอกสารฉบับเต็ม: `{research_doc}`")
    else:
        st.info("ยังไม่มีข้อมูลการประเมิน")
