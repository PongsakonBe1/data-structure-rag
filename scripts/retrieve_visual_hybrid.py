"""
Retrieval stack:
1) Topic filter from hierarchical index (list_hitachi)
2) Initial retrieval via hybrid metadata scoring (and optional local ColPali score)
3) VLM reranking with Qwen2.5-VL
4) Visual grounding on top hits

This script is designed to be robust on low-resource machines.
If local ColPali cannot be loaded, it falls back to metadata-only hybrid retrieval.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
from typing import Any
from pathlib import Path

import httpx
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

try:
    from pythainlp.tokenize import word_tokenize
except Exception:
    word_tokenize = None

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


FIGURE_REF_RE = re.compile(r"(ภาพที่|ตารางที่|figure|table)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
OPERATION_KEYWORDS = [
    "enqueue",
    "dequeue",
    "insert",
    "delete",
    "append",
    "traverse",
    "remove",
    "push",
    "pop",
    "front",
    "rear",
    "เพิ่ม",
    "ลบ",
    "แทรก",
    "เพิ่มโหนด",
    "ลบโหนด",
    "แทรกโหนด",
    "วงกลม",
    "circular",
    "ดำเนินการ",
    "ขั้นตอน",
    "ทำงาน",
]
STRUCTURE_KEYWORDS = [
    "โครงสร้าง",
    "structure",
    "stack",
    "queue",
    "linked",
    "list",
    "tree",
    "graph",
    "array",
    "อาร์เรย์",
    "ลิงค์ลิสต์",
    "คิว",
    "สแตก",
]

ENDPOINT_SCORE_CACHE: dict[str, dict] = {}
SPLADE_CACHE: dict[str, Any] = {
    "model_name": "",
    "device": "",
    "tokenizer": None,
    "model": None,
    "error": "",
}
BM25_CACHE: dict[str, Any] = {
    "corpus_sig": "",
    "model": None,
}
QUERY_SYNONYMS = {
    "คิว": ["queue", "enqueue", "dequeue", "front", "rear", "คิววงกลม", "วงกลม"],
    "queue": ["คิว", "enqueue", "dequeue", "front", "rear", "circular"],
    "วงกลม": ["circular", "mod", "wrap", "queue"],
    "อาร์เรย์": ["array", "index", "queue", "stack"],
    "array": ["อาร์เรย์", "index"],
    "ลิงค์ลิสต์": ["linked", "list", "node", "head", "next", "insert", "delete", "append", "traverse"],
    "linked": ["ลิงค์ลิสต์", "list", "node", "head", "next"],
    "list": ["ลิงค์ลิสต์", "node", "head", "next"],
    "โหนด": ["node", "linked", "list", "head", "next"],
    "node": ["โหนด", "linked", "list", "head", "next"],
}


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[\[\]\(\)\{\},.:;!?/\\|\"'`~@#$%^&*+=<>]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_thai_for_splade(text: str) -> str:
    """
    Preprocess Thai text for SPLADE by word segmentation.
    This helps SPLADE tokenizer understand Thai word boundaries better.
    """
    text = (text or "").strip()
    if not text:
        return ""
    
    # Check if text contains Thai characters
    thai_chars = re.compile(r'[\u0E00-\u0E7F]')
    if not thai_chars.search(text):
        # No Thai characters, use regular normalization
        return normalize_text(text)
    
    # For Thai text, segment words first then join with spaces
    if word_tokenize is not None:
        try:
            # Segment Thai words
            tokens = word_tokenize(text, engine="newmm")
            # Normalize each token
            normalized_tokens = []
            for tok in tokens:
                tok = tok.strip()
                if not tok:
                    continue
                # Clean punctuation but keep Thai characters
                tok = re.sub(r"[\[\]\(\)\{\},.:;!?/\\|\"'`~@#$%^&*+=<>]+", " ", tok)
                tok = tok.strip().lower()
                if tok:
                    normalized_tokens.append(tok)
            return " ".join(normalized_tokens)
        except Exception:
            pass
    
    # Fallback to regular normalization
    return normalize_text(text)


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if word_tokenize is not None:
        try:
            toks = [t.strip().lower() for t in word_tokenize(text, engine="newmm") if t.strip()]
            if toks:
                return toks
        except Exception:
            pass
    return [t for t in re.split(r"\s+", text) if t]


def expand_query_tokens(query: str, q_tokens: list[str]) -> list[str]:
    q_norm = normalize_text(query)
    expanded = list(q_tokens or [])
    seen = set(expanded)
    for k, vals in QUERY_SYNONYMS.items():
        if k in q_norm or k in seen:
            for v in vals:
                vv = normalize_text(v)
                if vv and vv not in seen:
                    seen.add(vv)
                    expanded.append(vv)
    return expanded


def lexical_score(query_tokens: list[str], rec: dict) -> float:
    if not query_tokens:
        return 0.0
    hay = " ".join(
        [
            str(rec.get("text", "")),
            " ".join(str(x) for x in rec.get("tags", []) if str(x).strip()),
            " ".join(str(x) for x in rec.get("metadata_tags", []) if str(x).strip()),
            " ".join(str(x) for x in rec.get("structure_labels", []) if str(x).strip()),
            " ".join(str(x) for x in rec.get("figure_refs", []) if str(x).strip()),
            str(rec.get("chapter_title", "")),
            str(rec.get("section_title", "")),
            str(rec.get("best_topic_title", "")),
            " ".join(str(x) for x in rec.get("topic_titles", []) if str(x).strip()),
            " ".join(str(x) for x in rec.get("topic_ids", []) if str(x).strip()),
        ]
    ).lower()
    if not hay:
        return 0.0
    hit = sum(1 for t in query_tokens if t and t in hay)
    return hit / max(1, len(query_tokens))


def _sparse_text_for_record(rec: dict) -> str:
    parts = [
        str(rec.get("text", "")),
        " ".join(str(x) for x in rec.get("tags", []) if str(x).strip()),
        " ".join(str(x) for x in rec.get("metadata_tags", []) if str(x).strip()),
        " ".join(str(x) for x in rec.get("structure_labels", []) if str(x).strip()),
        " ".join(str(x) for x in rec.get("figure_refs", []) if str(x).strip()),
        str(rec.get("chapter_title", "")),
        str(rec.get("section_title", "")),
        str(rec.get("best_topic_title", "")),
    ]
    return normalize_text(" ".join(parts))


def _resolve_device(device: str) -> str:
    real = str(device or "auto").strip().lower()
    if real in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if real == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return real


def _load_splade_runtime(model_name: str, device: str, hf_api_key: str = ""):
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except Exception as exc:
        return None, None, "cpu", f"splade_import_failed:{exc}"

    real_device = _resolve_device(device)
    if (
        SPLADE_CACHE.get("tokenizer") is not None
        and SPLADE_CACHE.get("model") is not None
        and SPLADE_CACHE.get("model_name") == model_name
        and SPLADE_CACHE.get("device") == real_device
    ):
        return SPLADE_CACHE.get("tokenizer"), SPLADE_CACHE.get("model"), real_device, "ok"

    try:
        token = str(hf_api_key or "").strip() or None
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForMaskedLM.from_pretrained(model_name, token=token)
        model = model.to(real_device).eval()
        SPLADE_CACHE["model_name"] = model_name
        SPLADE_CACHE["device"] = real_device
        SPLADE_CACHE["tokenizer"] = tokenizer
        SPLADE_CACHE["model"] = model
        SPLADE_CACHE["error"] = ""
        return tokenizer, model, real_device, "ok"
    except Exception as exc:
        SPLADE_CACHE["error"] = str(exc)
        SPLADE_CACHE["tokenizer"] = None
        SPLADE_CACHE["model"] = None
        return None, None, real_device, f"splade_load_failed:{exc}"


def _splade_encode_texts(
    texts: list[str],
    *,
    tokenizer,
    model,
    device: str,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    vecs: list[torch.Tensor] = []
    bs = max(1, int(batch_size))
    ml = max(32, int(max_length))
    for i in range(0, len(texts), bs):
        batch = [str(x or "") for x in texts[i : i + bs]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=ml,
            return_tensors="pt",
        )
        encoded = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(encoded).items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        attn = encoded.get("attention_mask")
        if attn is None:
            attn = torch.ones(logits.shape[:2], device=logits.device, dtype=torch.long)
        w = torch.log1p(torch.relu(logits)) * attn.unsqueeze(-1)
        pooled = torch.max(w, dim=1).values
        vecs.append(pooled.detach().cpu().float())
    return torch.cat(vecs, dim=0) if vecs else torch.zeros((0, 1), dtype=torch.float32)


def run_splade_scores(
    query: str,
    records: list[dict],
    *,
    model_name: str,
    device: str,
    max_length: int,
    batch_size: int,
    mode: str,
    hf_client: InferenceClient | None,
    hf_provider: str,
    hf_api_key: str,
) -> tuple[np.ndarray, str]:
    if not records:
        return np.zeros(0, dtype=np.float64), "splade_no_records"

    # Use Thai preprocessing for SPLADE to improve Thai word segmentation
    doc_texts = [preprocess_thai_for_splade(_sparse_text_for_record(r)) for r in records]
    if not any(doc_texts):
        return np.zeros(len(records), dtype=np.float64), "splade_no_text"

    def _vec_from_feature(obj) -> np.ndarray:
        arr = np.asarray(obj, dtype=np.float32)
        if arr.ndim == 1:
            vec = arr
        elif arr.ndim == 2:
            vec = arr.mean(axis=0)
        else:
            vec = arr.reshape(arr.shape[0], -1).mean(axis=0) if arr.size else np.zeros((1,), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-12 else vec

    def _router_feature_extraction(text: str) -> Any:
        payload = {"inputs": text}
        url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
        headers = {"Authorization": f"Bearer {hf_api_key}"}
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    def _run_hf_api() -> tuple[np.ndarray, str]:
        local_client = hf_client
        if local_client is None:
            if not str(hf_api_key or "").strip():
                return np.zeros(len(records), dtype=np.float64), "splade_hf_missing_client"
            try:
                local_client = InferenceClient(provider=str(hf_provider or "hf-inference"), api_key=hf_api_key)
            except TypeError:
                local_client = InferenceClient(api_key=hf_api_key)
        def _call_feature(text: str) -> Any:
            # Use Thai preprocessing for better tokenization
            norm = preprocess_thai_for_splade(text)
            last_exc: Exception | None = None
            if local_client is not None:
                for call in (
                    lambda: local_client.feature_extraction(text=norm, model=model_name),
                    lambda: local_client.feature_extraction(norm, model=model_name),
                ):
                    try:
                        return call()
                    except Exception as exc:  # try signature/router fallback
                        last_exc = exc
                        if "sparse_encoder" in str(exc).lower():
                            raise
            try:
                return _router_feature_extraction(norm)
            except Exception as router_exc:
                if last_exc is not None:
                    raise RuntimeError(f"{last_exc} | router:{router_exc}") from router_exc
                raise

        try:
            # Preprocess query with Thai segmentation
            q_processed = preprocess_thai_for_splade(query)
            q_raw = _call_feature(q_processed)
            q_vec = _vec_from_feature(q_raw)
            if q_vec.size == 0:
                return np.zeros(len(records), dtype=np.float64), "splade_hf_empty_query"
            vals = []
            for text in doc_texts:
                d_raw = _call_feature(text)
                d_vec = _vec_from_feature(d_raw)
                if d_vec.size == 0:
                    vals.append(0.0)
                    continue
                if d_vec.shape[0] != q_vec.shape[0]:
                    m = min(int(d_vec.shape[0]), int(q_vec.shape[0]))
                    if m <= 0:
                        vals.append(0.0)
                        continue
                    vals.append(float(np.dot(d_vec[:m], q_vec[:m])))
                else:
                    vals.append(float(np.dot(d_vec, q_vec)))
            return np.asarray(vals, dtype=np.float64), "ok_hf_api"
        except Exception as exc:
            if "sparse_encoder" in str(exc).lower():
                return np.zeros(len(records), dtype=np.float64), "splade_hf_backend_missing_sparse_encoder"
            return np.zeros(len(records), dtype=np.float64), f"splade_hf_failed:{exc}"

    def _run_local() -> tuple[np.ndarray, str]:
        home = Path.home()
        hf_cache_root = home / ".cache" / "huggingface" / "hub"
        try:
            usage = shutil.disk_usage(str(hf_cache_root.parent if hf_cache_root.parent.exists() else home))
            free_gb = usage.free / float(1024**3)
            if free_gb < 2.0:
                return np.zeros(len(records), dtype=np.float64), f"splade_local_disabled_low_disk:{round(free_gb,2)}GB"
        except Exception:
            pass

        tok, mdl, real_device, status = _load_splade_runtime(model_name, device, hf_api_key=hf_api_key)
        if status != "ok" or tok is None or mdl is None:
            return np.zeros(len(records), dtype=np.float64), status
        try:
            # Use Thai preprocessing for local SPLADE query encoding
            q_mat = _splade_encode_texts(
                [preprocess_thai_for_splade(query)],
                tokenizer=tok,
                model=mdl,
                device=real_device,
                max_length=max_length,
                batch_size=1,
            )
            d_mat = _splade_encode_texts(
                doc_texts,
                tokenizer=tok,
                model=mdl,
                device=real_device,
                max_length=max_length,
                batch_size=batch_size,
            )
            if q_mat.numel() == 0 or d_mat.numel() == 0:
                return np.zeros(len(records), dtype=np.float64), "splade_empty_embedding"
            q_vec = F.normalize(q_mat[0], dim=0, eps=1e-12)
            d_vecs = F.normalize(d_mat, dim=1, eps=1e-12)
            scores = torch.matmul(d_vecs, q_vec).detach().cpu().numpy().astype(np.float64)
            return scores, "ok_local"
        except Exception as exc:
            return np.zeros(len(records), dtype=np.float64), f"splade_runtime_failed:{exc}"

    mode_val = str(mode or "auto").strip().lower()
    if mode_val == "disabled":
        return np.zeros(len(records), dtype=np.float64), "splade_disabled"
    if mode_val in {"hf_api", "auto"}:
        vals, st = _run_hf_api()
        if st.startswith("ok"):
            return vals, st
        if mode_val == "hf_api":
            return vals, st
        hf_err = st
    if mode_val in {"local", "auto"}:
        vals2, st2 = _run_local()
        if mode_val == "auto" and "hf_err" in locals() and not str(st2).startswith("ok"):
            return vals2, f"{hf_err} | {st2}"
        return vals2, st2
    return np.zeros(len(records), dtype=np.float64), f"splade_invalid_mode:{mode_val}"


def run_bm25_scores(query: str, records: list[dict]) -> tuple[np.ndarray, str]:
    if not records:
        return np.zeros(0, dtype=np.float64), "bm25_no_records"
    try:
        from rank_bm25 import BM25Okapi
    except Exception as exc:
        return np.zeros(len(records), dtype=np.float64), f"bm25_import_failed:{exc}"

    doc_texts = [_sparse_text_for_record(r) for r in records]
    tokenized_docs = [tokenize(t) for t in doc_texts]
    if not any(tokenized_docs):
        return np.zeros(len(records), dtype=np.float64), "bm25_no_text"

    sig = hashlib.sha1()
    sig.update(str(len(records)).encode("utf-8"))
    for rec, toks in zip(records, tokenized_docs):
        sig.update(str(rec.get("id", "")).encode("utf-8", errors="ignore"))
        sig.update(str(len(toks)).encode("utf-8"))
    corpus_sig = sig.hexdigest()

    bm25_model = BM25_CACHE.get("model")
    if BM25_CACHE.get("corpus_sig") != corpus_sig or bm25_model is None:
        safe_docs = [toks if toks else ["_"] for toks in tokenized_docs]
        bm25_model = BM25Okapi(safe_docs)
        BM25_CACHE["corpus_sig"] = corpus_sig
        BM25_CACHE["model"] = bm25_model

    q_tokens = tokenize(query)
    if not q_tokens:
        q_tokens = ["_"]
    try:
        scores = bm25_model.get_scores(q_tokens)
        return np.asarray(scores, dtype=np.float64), "ok"
    except Exception as exc:
        return np.zeros(len(records), dtype=np.float64), f"bm25_failed:{exc}"


def normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        n = max(1, len(weights))
        return {k: 1.0 / n for k in weights}
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def extract_figure_refs(text: str) -> list[str]:
    refs: list[str] = []
    for m in FIGURE_REF_RE.finditer(text or ""):
        refs.append(m.group(2).strip())
    out = []
    seen = set()
    for r in refs:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def build_query_profile(query: str, q_tokens: list[str]) -> dict:
    qn = normalize_text(query)
    fig_refs = extract_figure_refs(query)
    has_figure_ref = len(fig_refs) > 0
    op_terms = [k for k in OPERATION_KEYWORDS if k in qn]
    struct_terms = [k for k in STRUCTURE_KEYWORDS if k in qn]
    return {
        "query_norm": qn,
        "query_tokens": q_tokens,
        "figure_refs": fig_refs,
        "has_figure_ref": has_figure_ref,
        "operation_terms": op_terms,
        "operation_intent": len(op_terms) > 0,
        "structure_terms": struct_terms,
        "structure_intent": len(struct_terms) > 0,
    }


def figure_ref_score(profile: dict, rec: dict) -> float:
    rec_refs = [str(x).strip() for x in rec.get("figure_refs", []) if str(x).strip()]
    rec_ref_nums = []
    for r in rec_refs:
        rec_ref_nums.extend(extract_figure_refs(r))
    rec_ref_nums = sorted(set(rec_ref_nums))
    if profile.get("has_figure_ref"):
        q_refs = set(profile.get("figure_refs", []))
        if not q_refs:
            return 0.0
        matched = len(q_refs.intersection(set(rec_ref_nums)))
        return matched / max(1, len(q_refs))
    # If query asks in visual style but no explicit figure number, prefer records with figure refs.
    qn = str(profile.get("query_norm", ""))
    if any(x in qn for x in ["ภาพ", "figure", "ตาราง", "table"]) and rec_ref_nums:
        return 0.65
    return 0.0


def operation_flow_score(profile: dict, rec: dict) -> float:
    terms = profile.get("operation_terms", []) or []
    if not terms:
        return 0.0
    hay = " ".join(
        [
            str(rec.get("text", "")),
            " ".join(str(x) for x in rec.get("tags", [])),
            " ".join(str(x) for x in rec.get("structure_labels", [])),
            " ".join(str(x) for x in rec.get("figure_refs", [])),
        ]
    ).lower()
    if not hay:
        return 0.0
    matched = sum(1 for t in terms if t in hay)
    if matched > 0:
        return matched / max(1, len(terms))
    if len(rec.get("figure_refs", []) or []) >= 3:
        return 0.55
    return 0.0


def structure_alignment_score(profile: dict, rec: dict) -> float:
    terms = profile.get("structure_terms", []) or []
    if not terms:
        return 0.0
    hay = " ".join(
        [
            " ".join(str(x) for x in rec.get("tags", [])),
            " ".join(str(x) for x in rec.get("structure_labels", [])),
            str(rec.get("text", "")),
        ]
    ).lower()
    if not hay:
        return 0.0
    matched = sum(1 for t in terms if t in hay)
    return matched / max(1, len(terms))


def topic_scope_alignment_score(
    rec: dict,
    *,
    target_topic_id: str,
    topic_scope_ids: set[str],
    query_profile: dict | None = None,
) -> float:
    target = str(target_topic_id or "").strip()
    scope_ids = {str(x).strip() for x in (topic_scope_ids or set()) if str(x).strip()}

    rec_best = str(rec.get("best_topic_id", "") or "").strip()
    rec_section = str(rec.get("section_id", "") or "").strip()
    rec_topic_ids = [str(x).strip() for x in (rec.get("topic_ids", []) or []) if str(x).strip()]
    rec_ids = {x for x in ([rec_best, rec_section] + rec_topic_ids) if x}
    if not rec_ids:
        return 0.0

    if target and (target in rec_ids):
        base = 1.0
    elif target and "." in target:
        parent = target.rsplit(".", 1)[0]
        if any((rid == parent) or rid.startswith(parent + ".") for rid in rec_ids):
            base = 0.82
        else:
            chap = target.split(".", 1)[0]
            base = 0.26 if any(rid.split(".", 1)[0] == chap for rid in rec_ids) else 0.0
    elif target:
        chap = target.split(".", 1)[0]
        base = 0.26 if any(rid.split(".", 1)[0] == chap for rid in rec_ids) else 0.0
    elif scope_ids and any(rid in scope_ids for rid in rec_ids):
        base = 0.9
    else:
        base = 0.0

    if base <= 0.0:
        return 0.0

    qn = str((query_profile or {}).get("query_norm", "") or "").lower()
    hay = " ".join(
        [
            str(rec.get("text", "")).lower(),
            str(rec.get("best_topic_title", "")).lower(),
            str(rec.get("section_title", "")).lower(),
            " ".join(str(x).lower() for x in (rec.get("figure_refs", []) or [])),
        ]
    )
    if ("วงกลม" in qn) or ("circular" in qn):
        if ("วงกลม" in hay) or ("circular" in hay):
            base = min(1.0, base + 0.12)
        else:
            base = max(0.0, base - 0.28)
    return max(0.0, min(1.0, float(base)))


def _record_figure_nums(rec: dict) -> set[str]:
    out = set()
    for fr in rec.get("figure_refs", []) or []:
        for n in extract_figure_refs(str(fr)):
            out.add(n)
    return out


def _record_page_key(rec: dict) -> tuple[str, int]:
    src = str(rec.get("source", "")).strip().lower()
    try:
        page = int(rec.get("page", 0) or 0)
    except Exception:
        page = 0
    return (src, page)


def coverage_aware_order(
    candidates: list[dict],
    *,
    query_profile: dict,
    top_k: int,
    novelty_weight: float,
    figure_weight: float,
    page_weight: float,
    step_target: int,
) -> list[dict]:
    """
    Greedy coverage-aware reordering:
    keeps relevance while encouraging diverse figure/page evidence,
    especially for step-by-step process questions.
    """
    if not candidates:
        return candidates

    remaining = list(candidates)
    selected: list[dict] = []
    seen_figs: set[str] = set()
    seen_pages: set[tuple[str, int]] = set()

    use_operation_boost = bool(query_profile.get("operation_intent"))
    use_figure_boost = bool(query_profile.get("has_figure_ref"))
    target_refs = set(query_profile.get("figure_refs", []) or [])
    k = max(1, min(int(top_k), len(remaining)))

    for _ in range(k):
        best_idx = 0
        best_score = -1e9
        for idx, c in enumerate(remaining):
            base = float(c.get("final_score", c.get("pre_vlm_score", c.get("base_score", 0.0))) or 0.0)
            rec_figs = _record_figure_nums(c)
            page_key = _record_page_key(c)

            new_figs = rec_figs - seen_figs
            fig_novelty = 1.0 if new_figs else 0.0
            if use_figure_boost and target_refs:
                fig_novelty = 1.0 if len(target_refs.intersection(rec_figs - seen_figs)) > 0 else fig_novelty

            page_novelty = 1.0 if page_key not in seen_pages else 0.0
            op_bonus = float(c.get("operation_score", 0.0) or 0.0) if use_operation_boost else 0.0
            struct_bonus = float(c.get("structure_score", 0.0) or 0.0) if query_profile.get("structure_intent") else 0.0

            coverage = (
                float(figure_weight) * fig_novelty
                + float(page_weight) * page_novelty
                + 0.08 * op_bonus
                + 0.05 * struct_bonus
            )
            score = ((1.0 - float(novelty_weight)) * base) + (float(novelty_weight) * coverage)
            if score > best_score:
                best_score = score
                best_idx = idx

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        seen_figs.update(_record_figure_nums(chosen))
        seen_pages.add(_record_page_key(chosen))
        if use_operation_boost and len(seen_figs) >= max(1, int(step_target)):
            # Enough distinct figure steps collected; keep remaining by relevance.
            break

    remaining = sorted(
        remaining,
        key=lambda x: float(x.get("final_score", x.get("pre_vlm_score", x.get("base_score", 0.0))) or 0.0),
        reverse=True,
    )
    return selected + remaining


def region_quality_score(rec: dict) -> float:
    """
    Proxy score for crop usability:
    - penalize very tiny crops (often arrows/noise only)
    - penalize very low-ink crops (likely blank)
    """
    meta = rec.get("region_meta", {}) if isinstance(rec.get("region_meta", {}), dict) else {}
    area = float(meta.get("area_ratio", 0.0) or 0.0)
    ink = float(meta.get("ink_ratio", 0.0) or 0.0)
    if area <= 0.0:
        return 0.0
    # Triangular area quality with sweet spot for figure regions.
    if area < 0.008:
        area_q = max(0.0, area / 0.008)
    elif area > 0.34:
        area_q = max(0.0, 1.0 - ((area - 0.34) / 0.66))
    else:
        area_q = 1.0
    ink_q = min(1.0, max(0.0, ink / 0.12))
    return float((0.72 * area_q) + (0.28 * ink_q))


def rank_positions(scores: np.ndarray) -> np.ndarray:
    """1-based rank positions (smaller is better)."""
    if scores.size == 0:
        return np.array([], dtype=np.float64)
    # Stable tie handling via mergesort.
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    return ranks


def weighted_rrf(
    score_channels: dict[str, np.ndarray],
    *,
    rrf_k: int = 60,
    channel_weights: dict[str, float] | None = None,
) -> np.ndarray:
    if not score_channels:
        return np.array([], dtype=np.float64)
    first = next(iter(score_channels.values()))
    out = np.zeros(first.size, dtype=np.float64)
    c_weights = normalize_weights(channel_weights or {k: 1.0 for k in score_channels})
    for name, vals in score_channels.items():
        if vals.size == 0:
            continue
        # Skip channels with no discrimination.
        if float(np.max(vals) - np.min(vals)) < 1e-12:
            continue
        ranks = rank_positions(vals)
        w = float(c_weights.get(name, 0.0))
        out += w * (1.0 / (float(rrf_k) + ranks))
    return out


def compute_task_metrics(
    query_profile: dict,
    hits: list[dict],
    grounding: list[dict],
    *,
    step_target: int = 4,
) -> dict:
    k = len(hits)
    if k == 0:
        return {
            "region_hit_ratio_at_k": 0.0,
            "figure_ref_hit_ratio_at_k": 0.0,
            "structure_hit_ratio_at_k": 0.0,
            "operation_coverage_ratio_at_k": 0.0,
            "query_figure_ref_coverage": 0.0 if query_profile.get("has_figure_ref") else None,
            "grounding_success_rate": 0.0,
            "grounding_consistency_mean": 0.0,
            "citation_ready_ratio": 0.0,
            "mean_final_score": 0.0,
            "crop_completeness_proxy": 0.0,
            "small_region_ratio_at_k": 0.0,
            "region_quality_mean": 0.0,
            "unique_figure_refs_at_k": 0,
            "page_diversity_at_k": 0.0,
            "procedure_step_coverage_proxy": 0.0,
        }

    region_hits = sum(1 for h in hits if str(h.get("image_level", "")).strip().lower() == "region")
    figure_ref_hits = sum(1 for h in hits if len(h.get("figure_refs", []) or []) > 0)
    structure_hits = sum(1 for h in hits if len(h.get("structure_labels", []) or []) > 0)
    op_terms = query_profile.get("operation_terms", []) or []
    op_hits = 0
    for h in hits:
        hay = " ".join(
            [
                str(h.get("preview", "")),
                " ".join(str(x) for x in h.get("figure_refs", []) if str(x).strip()),
                " ".join(str(x) for x in h.get("tags", []) if str(x).strip()),
            ]
        ).lower()
        op_score = float(h.get("operation_score", 0.0) or 0.0)
        if (op_terms and any(t in hay for t in op_terms)) or op_score >= 0.5:
            op_hits += 1

    q_refs = set(query_profile.get("figure_refs", []) or [])
    matched_q_refs = set()
    unique_refs = set()
    unique_pages = set()
    for h in hits:
        unique_pages.add((str(h.get("source", "")).strip().lower(), str(h.get("page", "")).strip()))
        for fr in h.get("figure_refs", []) or []:
            for n in extract_figure_refs(str(fr)):
                unique_refs.add(n)
                if n in q_refs:
                    matched_q_refs.add(n)

    grounding_ok = 0
    grounding_consistency_vals = []
    for g in grounding or []:
        payload = g.get("grounding", {}) if isinstance(g, dict) else {}
        parsed = payload.get("parsed") if isinstance(payload, dict) else None
        if isinstance(parsed, dict) and parsed:
            grounding_ok += 1
        cons = payload.get("consensus", {}) if isinstance(payload, dict) else {}
        if isinstance(cons, dict):
            ar = cons.get("agreement_ratio")
            if isinstance(ar, (int, float)):
                grounding_consistency_vals.append(float(ar))
    grounding_denom = max(1, len(grounding or []))

    ready_cites = 0
    for h in hits:
        if str(h.get("source", "")).strip() and str(h.get("page", "")).strip() and str(h.get("id", "")).strip():
            ready_cites += 1

    score_vals = [float(h.get("final_score", h.get("base_score", 0.0)) or 0.0) for h in hits]
    crop_good = 0
    crop_total = 0
    small_region_hits = 0
    region_quality_vals = []
    for h in hits:
        rm = h.get("region_meta", {}) if isinstance(h.get("region_meta", {}), dict) else {}
        if not rm:
            continue
        ar = float(rm.get("area_ratio", 0.0) or 0.0)
        ink = float(rm.get("ink_ratio", 0.0) or 0.0)
        crop_total += 1
        # Proxy: crop too tiny loses labels; too large behaves like full-page.
        if 0.015 <= ar <= 0.32:
            crop_good += 1
        if ar < 0.008:
            small_region_hits += 1
        region_quality_vals.append(region_quality_score({"region_meta": {"area_ratio": ar, "ink_ratio": ink}}))

    return {
        "region_hit_ratio_at_k": round(region_hits / k, 4),
        "figure_ref_hit_ratio_at_k": round(figure_ref_hits / k, 4),
        "structure_hit_ratio_at_k": round(structure_hits / k, 4),
        "operation_coverage_ratio_at_k": round(op_hits / k, 4) if op_terms else None,
        "query_figure_ref_coverage": round(len(matched_q_refs) / max(1, len(q_refs)), 4) if q_refs else None,
        "grounding_success_rate": round(grounding_ok / grounding_denom, 4),
        "grounding_consistency_mean": round(float(np.mean(grounding_consistency_vals)), 4) if grounding_consistency_vals else None,
        "citation_ready_ratio": round(ready_cites / k, 4),
        "mean_final_score": round(float(np.mean(score_vals)) if score_vals else 0.0, 6),
        "crop_completeness_proxy": round(crop_good / max(1, crop_total), 4) if crop_total else None,
        "small_region_ratio_at_k": round(small_region_hits / max(1, crop_total), 4) if crop_total else None,
        "region_quality_mean": round(float(np.mean(region_quality_vals)) if region_quality_vals else 0.0, 4),
        "unique_figure_refs_at_k": int(len(unique_refs)),
        "page_diversity_at_k": round(len(unique_pages) / k, 4) if k else 0.0,
        "procedure_step_coverage_proxy": (
            round(min(1.0, len(unique_refs) / max(1, int(step_target))), 4)
            if query_profile.get("operation_intent") else None
        ),
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_topic_hierarchy(path: Path) -> dict:
    if not path.exists():
        return {}
    hierarchy = json.loads(path.read_text(encoding="utf-8"))
    try:
        override_raw = os.getenv("SECTION_PAGE_OVERRIDE_FILE", "indexes/hierarchical/section_page_overrides.json")
        override_path = Path(override_raw)
        if not override_path.is_absolute():
            override_path = (Path.cwd() / override_path).resolve()
        if override_path.exists() and isinstance(hierarchy, dict):
            override_payload = json.loads(override_path.read_text(encoding="utf-8"))
            override_map = override_payload.get("topic_to_pages", override_payload if isinstance(override_payload, dict) else {})
            if isinstance(override_map, dict):
                topic_to_pages = hierarchy.get("topic_to_pages", {})
                if isinstance(topic_to_pages, dict):
                    for sid_raw, pages_raw in override_map.items():
                        sid = str(sid_raw).strip()
                        if not sid or not isinstance(pages_raw, list):
                            continue
                        pages = [str(p).strip() for p in pages_raw if str(p).strip()]
                        if pages:
                            topic_to_pages[sid] = pages
    except Exception:
        pass
    return hierarchy


def hierarchy_mojibake_ratio(hierarchy: dict) -> float:
    """
    Detect likely mojibake in topic titles/tokens (e.g., 'เธ...', 'เน...').
    """
    topics = hierarchy.get("topics", []) if isinstance(hierarchy, dict) else []
    if not isinstance(topics, list) or not topics:
        return 0.0
    total = 0
    garbled = 0
    for t in topics:
        title = str(t.get("title", "") or "")
        toks = " ".join(str(x) for x in (t.get("tokens", []) or []))
        text = f"{title} {toks}".strip()
        if not text:
            continue
        total += 1
        # common mojibake fingerprints observed in this project artifacts
        if ("เธ" in text and "เน" in text) or ("เธ" in text and text.count("เธ") >= 3):
            garbled += 1
    if total <= 0:
        return 0.0
    return float(garbled) / float(total)


def predict_topic_from_hierarchy(query: str, hierarchy: dict) -> tuple[str, float]:
    topics = hierarchy.get("topics", []) if isinstance(hierarchy, dict) else []
    if not topics:
        return "", 0.0
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return "", 0.0

    best_id = ""
    best_score = 0.0
    q_norm = normalize_text(query)
    for t in topics:
        tid = str(t.get("topic_id", "")).strip()
        title = str(t.get("title", "")).strip()
        t_tokens = set(str(x).lower() for x in t.get("tokens", [])) or set(tokenize(title))
        if not t_tokens:
            continue
        overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens | t_tokens))
        phrase_bonus = 0.25 if normalize_text(title) in q_norm or q_norm in normalize_text(title) else 0.0
        s = float(min(1.0, overlap + phrase_bonus))
        if s > best_score:
            best_score = s
            best_id = tid
    return best_id, best_score


def rank_topics_from_hierarchy(query: str, hierarchy: dict, top_n: int = 12) -> list[dict]:
    topics = hierarchy.get("topics", []) if isinstance(hierarchy, dict) else []
    if not topics:
        return []
    q_tokens = set(tokenize(query))
    q_norm = normalize_text(query)
    ranked = []
    for t in topics:
        tid = str(t.get("topic_id", "")).strip()
        title = str(t.get("title", "")).strip()
        if not tid:
            continue
        t_tokens = set(str(x).lower() for x in t.get("tokens", [])) or set(tokenize(title))
        if not t_tokens:
            continue
        overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens | t_tokens)) if q_tokens else 0.0
        phrase_bonus = 0.25 if (normalize_text(title) in q_norm or q_norm in normalize_text(title)) else 0.0
        score = float(min(1.0, overlap + phrase_bonus))
        ranked.append(
            {
                "topic_id": tid,
                "title": title,
                "score": score,
                "path": t.get("path", []),
            }
        )
    ranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return ranked[: max(1, int(top_n))]


def best_ranked_topic(ranked_topics: list[dict]) -> tuple[str, float]:
    if not ranked_topics:
        return "", 0.0
    try:
        max_score = max(float(t.get("score", 0.0) or 0.0) for t in ranked_topics)
    except Exception:
        max_score = 0.0
    eps = 1e-8
    candidates = []
    for t in ranked_topics:
        tid = str(t.get("topic_id", "")).strip()
        score = float(t.get("score", 0.0) or 0.0)
        if not tid:
            continue
        if abs(score - max_score) <= eps:
            candidates.append((tid, score))
    if not candidates:
        top = ranked_topics[0]
        return str(top.get("topic_id", "")).strip(), float(top.get("score", 0.0) or 0.0)
    candidates.sort(key=lambda x: (-x[0].count("."), len(x[0]), x[0]))
    return candidates[0][0], float(candidates[0][1])


def _valid_topic_ids(hierarchy: dict) -> set[str]:
    topics = hierarchy.get("topics", []) if isinstance(hierarchy, dict) else []
    out = set()
    for t in topics if isinstance(topics, list) else []:
        tid = str((t or {}).get("topic_id", "")).strip()
        if tid:
            out.add(tid)
    return out


def classify_query_intent_llm(
    *,
    hf_client: InferenceClient | None,
    model: str,
    query: str,
    ranked_topics: list[dict],
) -> tuple[dict, str]:
    if hf_client is None:
        return {}, "missing_hf_client"
    if not ranked_topics:
        return {}, "no_ranked_topics"
    topic_lines = []
    for t in ranked_topics:
        topic_lines.append(
            f"- topic_id={t.get('topic_id','')} | title={t.get('title','')} | score_hint={round(float(t.get('score',0.0)),4)}"
        )
    schema = (
        '{'
        '"intent_type":"operation|structure|definition|comparison|other",'
        '"target_chapter":"",'
        '"target_topic_ids":[""],'
        '"confidence":0.0,'
        '"require_structure":true,'
        '"reason":"..."'
        '}'
    )
    prompt = (
        "คุณคือโมดูลจัดหมวด intent สำหรับระบบค้นคืนเอกสารวิชาโครงสร้างข้อมูล\n"
        "ตอบ JSON เพียวเท่านั้น ห้ามมีข้อความอื่น\n"
        f"Schema: {schema}\n"
        "กติกา:\n"
        "1) target_topic_ids ต้องเลือกจาก candidate topic_id เท่านั้น\n"
        "2) target_chapter ใช้รหัสบทหลัก เช่น 2 หรือ 3 หรือ 2.4\n"
        "3) confidence อยู่ช่วง 0.0-1.0\n"
        "4) require_structure=true เมื่อคำถามเกี่ยวกับโครงสร้าง/ภาพ/ขั้นตอน\n\n"
        f"คำถามผู้ใช้: {query}\n"
        "Candidate Topics:\n"
        + "\n".join(topic_lines)
    )
    try:
        res = hf_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.0,
            top_p=0.9,
        )
        raw = (res.choices[0].message.content or "").strip()
        parsed = parse_json_object(raw)
        if isinstance(parsed, dict):
            q_lower = str(query or "").strip().lower()
            op_markers = (
                "การทำงาน",
                "ดำเนินการ",
                "ขั้นตอน",
                "เพิ่ม",
                "ลบ",
                "แทรก",
                "เพิ่มโหนด",
                "ลบโหนด",
                "แทรกโหนด",
                "insert",
                "delete",
                "remove",
                "enqueue",
                "dequeue",
                "push",
                "pop",
            )
            intent_type = str(parsed.get("intent_type", "") or "").strip().lower()
            if intent_type == "definition" and any(m in q_lower for m in op_markers):
                parsed["intent_type"] = "operation"
                intent_type = "operation"
            # Operation queries are frequently confused with structure sibling topics
            # (e.g., 2.4.1 vs 2.4.2). Normalize toward operation-labeled siblings.
            if intent_type == "operation" and any(m in q_lower for m in op_markers):
                target_ids = parsed.get("target_topic_ids", []) if isinstance(parsed.get("target_topic_ids", []), list) else []
                target_ids = [str(x).strip() for x in target_ids if str(x).strip()]
                op_candidates = []
                for cand in ranked_topics:
                    tid = str(cand.get("topic_id", "")).strip()
                    title = str(cand.get("title", "")).strip().lower()
                    if not tid:
                        continue
                    if any(k in title for k in ("การทำงาน", "ดำเนินการ", "operation")):
                        op_candidates.append((tid, title))
                if op_candidates:
                    chosen = ""
                    parent_ids = {t.rsplit(".", 1)[0] for t in target_ids if "." in t}
                    for tid, _ in op_candidates:
                        if tid.rsplit(".", 1)[0] in parent_ids:
                            chosen = tid
                            break
                    if not chosen and not target_ids:
                        chosen = op_candidates[0][0]
                    if not chosen and all(t.endswith(".1") for t in target_ids):
                        chosen = op_candidates[0][0]
                    if chosen:
                        parsed["target_topic_ids"] = [chosen]
            parsed["raw"] = raw
            return parsed, "ok"
        return {"raw": raw}, "invalid_json"
    except Exception as exc:
        return {"error": str(exc)}, f"llm_failed:{exc}"


def expand_query_terms_llm(
    *,
    hf_client: InferenceClient | None,
    model: str,
    query: str,
    ranked_topics: list[dict],
    max_terms: int,
) -> tuple[list[str], str]:
    if hf_client is None:
        return [], "missing_hf_client"
    if not str(query or "").strip():
        return [], "missing_query"
    if max_terms <= 0:
        return [], "disabled"

    topic_hints = []
    for t in ranked_topics[:8]:
        tid = str(t.get("topic_id", "")).strip()
        title = str(t.get("title", "")).strip()
        if tid and title:
            topic_hints.append(f"{tid}:{title}")

    schema = '{"expansion_terms":["term1","term2"],"reason":"..."}'
    prompt = (
        "คุณคือโมดูล query expansion สำหรับ retrieval เอกสารวิชาโครงสร้างข้อมูล\n"
        "ตอบเป็น JSON เท่านั้น\n"
        f"schema: {schema}\n"
        "กติกา:\n"
        f"- expansion_terms ไม่เกิน {max_terms} คำ/วลี\n"
        "- เน้นคำพ้อง/คำเทคนิคที่ช่วยค้นคืนหลักฐานในเอกสารเดียวกัน\n"
        "- ห้ามสร้างข้อเท็จจริงใหม่ ห้ามตอบเกินโจทย์\n"
        "- ควรคงทั้งไทยและอังกฤษเมื่อเป็นศัพท์เทคนิค\n"
        f"query: {query}\n"
        f"topic_hints: {', '.join(topic_hints)}"
    )
    try:
        res = hf_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,
            top_p=0.9,
        )
        raw = str(res.choices[0].message.content or "").strip()
        obj = parse_json_object(raw)
        raw_terms = []
        status = "ok"
        if isinstance(obj, dict):
            raw_terms = obj.get("expansion_terms", [])
            if not isinstance(raw_terms, list):
                raw_terms = []
                status = "invalid_schema"
        else:
            # Fallback: recover terms from plain text lists when model does not return JSON.
            status = "invalid_json"
            guessed = []
            for part in re.split(r"[\n,;|]+", raw):
                t = normalize_text(part)
                if not t:
                    continue
                if len(t) > 36:
                    continue
                if len(t.split()) > 5:
                    continue
                guessed.append(t)
            if guessed:
                raw_terms = guessed
                status = "ok_heuristic_parse"
        out: list[str] = []
        seen = set()
        for term in raw_terms:
            t = normalize_text(str(term or ""))
            if not t or len(t) < 2:
                continue
            if t.startswith("expansion_terms") or t in {"expansion_terms", "reason"}:
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= max_terms:
                break
        if out:
            return out, status
        return [], status
    except Exception as exc:
        return [], f"llm_failed:{exc}"


def resolve_intent_topic_ids(
    *,
    hierarchy: dict,
    ranked_topics: list[dict],
    intent_obj: dict,
    min_confidence: float,
    max_topic_ids: int,
) -> tuple[set[str], str, float]:
    valid_ids = _valid_topic_ids(hierarchy)
    fallback_tid, fallback_score = best_ranked_topic(ranked_topics or [])
    rank_scores = {
        str(t.get("topic_id", "")).strip(): float(t.get("score", 0.0) or 0.0)
        for t in (ranked_topics or [])
        if str(t.get("topic_id", "")).strip()
    }
    if not valid_ids:
        return ({fallback_tid} if fallback_tid else set()), "heuristic", fallback_score

    conf = 0.0
    try:
        conf = float((intent_obj or {}).get("confidence", 0.0) or 0.0)
    except Exception:
        conf = 0.0

    chapter = str((intent_obj or {}).get("target_chapter", "") or "").strip()
    target_ids_raw = (intent_obj or {}).get("target_topic_ids", []) if isinstance(intent_obj, dict) else []
    chosen = []
    explicit_ids = []
    for t in target_ids_raw if isinstance(target_ids_raw, list) else []:
        tid = str(t).strip()
        if tid and tid in valid_ids:
            explicit_ids.append(tid)
            chosen.append(tid)
    # Important: do NOT auto-expand to all topics in chapter when explicit topic IDs exist.
    # Broad chapter expansion was a major source of cross-topic evidence leakage.
    if chapter and not explicit_ids:
        # Prefer chapter node itself; if absent, keep only top scoped descendants.
        chapter_scoped = [tid for tid in sorted(valid_ids) if tid == chapter or tid.startswith(chapter + ".")]
        if chapter in chapter_scoped:
            chosen.append(chapter)
        else:
            chosen.extend(chapter_scoped[:4])

    dedup = []
    seen = set()
    for tid in chosen:
        if tid in seen:
            continue
        seen.add(tid)
        dedup.append(tid)
    # Prefer specific topics over ancestors to reduce cross-topic retrieval leakage.
    if dedup:
        kept: list[str] = []
        for tid in sorted(dedup, key=lambda x: (-x.count("."), len(x), x)):
            if any(k.startswith(tid + ".") for k in kept):
                continue
            kept.append(tid)
        # If both broad and specific topics remain, prefer specific leaves.
        if any(x.count(".") >= 2 for x in kept):
            leaf_like = [x for x in kept if x.count(".") >= 2]
            if leaf_like:
                kept = leaf_like
        kept = sorted(
            kept,
            key=lambda x: (-float(rank_scores.get(x, 0.0)), -x.count("."), len(x), x),
        )
        dedup = kept
    dedup = dedup[: max(1, int(max_topic_ids))]

    if conf >= float(min_confidence) and dedup:
        return set(dedup), "llm", conf
    if fallback_tid:
        return {fallback_tid}, "heuristic", fallback_score
    return set(), "none", 0.0


def pick_primary_topic_id(topic_ids: set[str], intent_obj: dict, heuristic_topic_id: str) -> str:
    if not topic_ids:
        return str(heuristic_topic_id or "").strip()
    explicit = []
    if isinstance(intent_obj, dict):
        raw_ids = intent_obj.get("target_topic_ids", [])
        if isinstance(raw_ids, list):
            for t in raw_ids:
                tid = str(t).strip()
                if tid and tid in topic_ids:
                    explicit.append(tid)
    if explicit:
        # Prefer the most specific ID first (deeper hierarchy).
        return sorted(set(explicit), key=lambda x: (-x.count("."), len(x), x))[0]
    return sorted(topic_ids, key=lambda x: (-x.count("."), len(x), x))[0]


def _topic_hier_match(rec_topic_id: str, target_topic_id: str) -> bool:
    rec = str(rec_topic_id or "").strip()
    tgt = str(target_topic_id or "").strip()
    if not rec or not tgt:
        return False
    if rec == tgt:
        return True
    if rec.startswith(tgt + "."):
        return True
    if tgt.startswith(rec + "."):
        return True
    return False


def _topic_strict_match(rec_topic_id: str, target_topic_id: str) -> bool:
    rec = str(rec_topic_id or "").strip()
    tgt = str(target_topic_id or "").strip()
    if not rec or not tgt:
        return False
    # Precision-first: exact target or descendant of target only.
    if rec == tgt:
        return True
    if rec.startswith(tgt + "."):
        return True
    return False


def filter_by_metadata_topic_ids(
    records: list[dict],
    topic_ids: set[str],
    *,
    target_chapter: str = "",
    strict_only: bool = False,
    allow_chapter_fallback: bool = True,
) -> tuple[list[dict], str]:
    if not topic_ids:
        return records, "not_applied"
    allowed_chapters = {str(tid).split(".", 1)[0].strip() for tid in topic_ids if str(tid).strip()}
    tc = str(target_chapter or "").strip()
    if tc:
        allowed_chapters.add(tc.split(".", 1)[0].strip())
    allowed_chapters = {c for c in allowed_chapters if c}

    def _rec_source_page(rec: dict) -> tuple[str, int]:
        src = str(rec.get("source", "") or "").strip()
        page_raw = rec.get("page", 0)
        try:
            page = int(page_raw or 0)
        except Exception:
            page = 0
        if src and page > 0:
            return src, page
        page_id = str(rec.get("page_id", "") or "").strip()
        if ":" in page_id:
            parts = page_id.rsplit(":", 1)
            src = src or parts[0]
            try:
                page = int(parts[1] or 0)
            except Exception:
                page = page
        return src, page

    # Pass 1 (strict): primary-topic exact/descendant match.
    strict: list[dict] = []
    strict_seed: list[tuple[str, int]] = []
    for r in records:
        chapter_id = str(r.get("chapter_id", "") or "").strip()
        if allowed_chapters and chapter_id and chapter_id not in allowed_chapters:
            continue
        best_id = str(r.get("best_topic_id", "") or "").strip()
        section_id = str(r.get("section_id", "") or "").strip()
        strict_topic_match = any(
            _topic_strict_match(best_id, tid) or _topic_strict_match(section_id, tid) for tid in topic_ids
        )
        if strict_topic_match:
            strict.append(r)
            strict_seed.append(_rec_source_page(r))

    # Strict extension: allow explicit topic tag only for adjacent pages to strict seed.
    if strict_seed:
        for r in records:
            if r in strict:
                continue
            chapter_id = str(r.get("chapter_id", "") or "").strip()
            if allowed_chapters and chapter_id and chapter_id not in allowed_chapters:
                continue
            rec_topic_ids = [str(x).strip() for x in (r.get("topic_ids", []) or []) if str(x).strip()]
            strict_tag_match = any(_topic_strict_match(rec_tid, tid) for rec_tid in rec_topic_ids for tid in topic_ids)
            if not strict_tag_match:
                continue
            src, page_no = _rec_source_page(r)
            near_seed = any(src == s and page_no > 0 and p > 0 and abs(page_no - p) <= 1 for s, p in strict_seed)
            if near_seed:
                strict.append(r)
    if strict:
        min_keep = max(6, int(0.08 * max(1, len(records))))
        if strict_only or len(strict) >= min_keep:
            return strict, "strict_topic_scope"
    if strict_only:
        # Precision-first rescue: strict misses can happen when sub-topic metadata
        # is sparse. In that case, keep retrieval inside the same chapter only.
        if allow_chapter_fallback and allowed_chapters:
            chapter_rescue: list[dict] = []
            for r in records:
                chapter_id = str(r.get("chapter_id", "") or "").strip()
                if chapter_id and chapter_id in allowed_chapters:
                    chapter_rescue.append(r)
            if chapter_rescue:
                return chapter_rescue, "strict_no_match_chapter_rescue"
        return [], "strict_no_match"

    # Pass 2 (relaxed): allow broader topic_ids and optional chapter-level hints.
    relaxed = []
    specific_topic_target = any("." in str(tid) for tid in topic_ids)
    for r in records:
        chapter_id = str(r.get("chapter_id", "") or "").strip()
        if allowed_chapters and chapter_id and chapter_id not in allowed_chapters:
            continue
        best_id = str(r.get("best_topic_id", "") or "").strip()
        section_id = str(r.get("section_id", "") or "").strip()
        hier_match = any(_topic_hier_match(best_id, tid) or _topic_hier_match(section_id, tid) for tid in topic_ids)
        chapter_hits = (
            any((tid == chapter_id) or tid.startswith(chapter_id + ".") for tid in topic_ids)
            if (allow_chapter_fallback and chapter_id and not specific_topic_target)
            else False
        )
        # Avoid noisy rec.topic_ids here; they can contain weak-label spillover.
        if hier_match or chapter_hits:
            relaxed.append(r)
    if relaxed:
        mode = "relaxed_topic_ids_with_chapter" if allow_chapter_fallback else "relaxed_topic_ids_no_chapter"
        return relaxed, mode
    return records, "fallback_all_records"


def filter_by_topic(records: list[dict], hierarchy: dict, topic_id: str) -> list[dict]:
    if not topic_id:
        return records
    topic_to_pages = hierarchy.get("topic_to_pages", {}) if isinstance(hierarchy, dict) else {}
    pages = set(topic_to_pages.get(topic_id, []))
    if not pages:
        return records
    out = [r for r in records if str(r.get("page_id", "")) in pages]
    return out or records


def expand_topic_ids(hierarchy: dict, topic_id: str, *, include_parent: bool, include_children: bool, include_siblings: bool) -> set[str]:
    if not topic_id:
        return set()
    topics = hierarchy.get("topics", []) if isinstance(hierarchy, dict) else []
    if not isinstance(topics, list):
        return {topic_id}
    by_id = {}
    children = {}
    for t in topics:
        tid = str(t.get("topic_id", "")).strip()
        if not tid:
            continue
        by_id[tid] = t
        pid = str(t.get("parent_id", "") or "").strip()
        children.setdefault(pid, set()).add(tid)

    expanded = {topic_id}
    parent_id = str((by_id.get(topic_id, {}) or {}).get("parent_id", "") or "").strip()
    if include_parent and parent_id:
        expanded.add(parent_id)
    if include_children:
        stack = [topic_id]
        while stack:
            cur = stack.pop()
            for ch in children.get(cur, set()):
                if ch not in expanded:
                    expanded.add(ch)
                    stack.append(ch)
    if include_siblings and parent_id:
        expanded.update(children.get(parent_id, set()))
    return expanded


def filter_by_topic_ids(records: list[dict], hierarchy: dict, topic_ids: set[str]) -> list[dict]:
    if not topic_ids:
        return records
    topic_to_pages = hierarchy.get("topic_to_pages", {}) if isinstance(hierarchy, dict) else {}
    pages = set()
    for tid in topic_ids:
        pages.update(topic_to_pages.get(tid, []) or [])
    if not pages:
        return records
    out = [r for r in records if str(r.get("page_id", "")) in pages]
    return out or records


def filter_by_best_topic_page_ids(records: list[dict], hierarchy: dict, topic_ids: set[str]) -> list[dict]:
    if not topic_ids:
        return records
    page_to_best = hierarchy.get("page_to_best_topic", {}) if isinstance(hierarchy, dict) else {}
    if not isinstance(page_to_best, dict) or not page_to_best:
        return []
    allowed_topic_ids = {str(t).strip() for t in topic_ids if str(t).strip()}
    allowed_pages = {
        str(pid).strip()
        for pid, tid in page_to_best.items()
        if str(pid).strip() and str(tid).strip() in allowed_topic_ids
    }
    if not allowed_pages:
        return []
    return [r for r in records if str(r.get("page_id", "")).strip() in allowed_pages]


def filter_by_chapters(records: list[dict], chapters: set[str]) -> list[dict]:
    if not chapters:
        return records
    ch = {str(x).strip() for x in chapters if str(x).strip()}
    if not ch:
        return records
    return [r for r in records if str(r.get("chapter_id", "") or "").strip() in ch]


def merge_unique_records(records: list[dict], limit: int = 0) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    lim = max(0, int(limit))
    for r in records or []:
        rid = str((r or {}).get("id", "")).strip()
        if not rid or rid in seen:
            continue
        out.append(r)
        seen.add(rid)
        if lim and len(out) >= lim:
            break
    return out


def rescue_unfiltered_topn(
    records: list[dict],
    *,
    query_tokens: list[str],
    top_n: int,
    allowed_chapters: set[str] | None = None,
) -> list[dict]:
    pool = list(records or [])
    if allowed_chapters:
        pool = filter_by_chapters(pool, allowed_chapters)
    if not pool:
        return []
    q_toks = [str(t).strip() for t in (query_tokens or []) if str(t).strip()]
    scored: list[tuple[float, dict]] = []
    for r in pool:
        lex = lexical_score(q_toks, r) if q_toks else 0.0
        fig = 0.25 if bool(r.get("figure_refs", [])) else 0.0
        lvl = str(r.get("image_level", "")).strip().lower()
        region_bonus = 0.15 if lvl == "region" else (0.08 if lvl == "page" else 0.0)
        struct = 0.12 if bool(r.get("has_structure", False)) else 0.0
        prior = float(r.get("region_score", 0.0) or 0.0)
        score = (0.58 * lex) + (0.17 * prior) + fig + region_bonus + struct
        scored.append((float(score), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [x[1] for x in scored[: max(1, int(top_n))]]
    return merge_unique_records(top, limit=top_n)


def _parse_source_page(rec: dict) -> tuple[str, int]:
    source = str(rec.get("source", "") or "").strip()
    try:
        page = int(rec.get("page", 0) or 0)
    except Exception:
        page = 0
    if source and page > 0:
        return source, page
    page_id = str(rec.get("page_id", "") or "").strip()
    m = re.match(r"^(.*):(\d+)$", page_id)
    if m:
        return m.group(1).strip(), int(m.group(2))
    return source, page


def _is_linkedlist_operation_query(profile: dict, topic_id: str) -> bool:
    qn = str(profile.get("query_norm", "") or "")
    terms = [
        "ลิงค์ลิสต์",
        "ลิงก์ลิสต์",
        "linked list",
        "linked",
        "node",
        "โหนด",
        "head",
    ]
    if any(t in qn for t in terms):
        return True
    tid = str(topic_id or "").strip()
    return tid.startswith("2.4")


def expand_with_adjacent_topic_pages(
    all_records: list[dict],
    filtered_records: list[dict],
    *,
    max_page_gap: int,
    max_extra_pages: int,
    forward_only: bool = False,
) -> list[dict]:
    """
    Recover step-by-step pages that are near topic anchors but were missed by hierarchy mapping.
    This is intentionally conservative: it only adds nearby pages from the same source.
    """
    if not filtered_records:
        return filtered_records
    page_gap = max(0, int(max_page_gap))
    extra_cap = max(0, int(max_extra_pages))
    if page_gap <= 0 or extra_cap <= 0:
        return filtered_records

    bucket: dict[tuple[str, int], list[dict]] = {}
    available_pages: dict[str, set[int]] = {}
    for rec in all_records:
        src, pg = _parse_source_page(rec)
        if not src or pg <= 0:
            continue
        bucket.setdefault((src, pg), []).append(rec)
        available_pages.setdefault(src, set()).add(pg)

    anchor_pages: dict[str, set[int]] = {}
    anchor_chapters: dict[str, set[str]] = {}
    for rec in filtered_records:
        src, pg = _parse_source_page(rec)
        if not src or pg <= 0:
            continue
        anchor_pages.setdefault(src, set()).add(pg)
        chap = str(rec.get("chapter_id", "") or "").strip()
        if chap:
            anchor_chapters.setdefault(src, set()).add(chap)
    if not anchor_pages:
        return filtered_records

    extra_page_candidates: list[tuple[int, str, int]] = []
    for src, pages in anchor_pages.items():
        avail = available_pages.get(src, set())
        if not avail:
            continue
        for ap in pages:
            lo = ap if forward_only else max(1, ap - page_gap)
            hi = ap + page_gap
            for pg in range(lo, hi + 1):
                if pg in avail and pg not in pages:
                    dist = abs(pg - ap)
                    extra_page_candidates.append((dist, src, pg))
    if not extra_page_candidates:
        return filtered_records

    extra_page_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    selected_pages: list[tuple[str, int]] = []
    seen_pages = set()
    for _, src, pg in extra_page_candidates:
        key = (src, pg)
        if key in seen_pages:
            continue
        seen_pages.add(key)
        selected_pages.append(key)
        if len(selected_pages) >= extra_cap:
            break

    out = list(filtered_records)
    seen_ids = {str(r.get("id", "")).strip() for r in out if str(r.get("id", "")).strip()}
    for src, pg in selected_pages:
        allowed_chaps = anchor_chapters.get(src, set())
        for rec in bucket.get((src, pg), []):
            rec_chap = str(rec.get("chapter_id", "") or "").strip()
            if allowed_chaps and rec_chap and rec_chap not in allowed_chaps:
                continue
            rid = str(rec.get("id", "")).strip()
            if not rid or rid in seen_ids:
                continue
            out.append(rec)
            seen_ids.add(rid)
    return out


def run_local_colpali_scores(
    query: str,
    records: list[dict],
    *,
    colpali_index_path: Path,
    model_name: str,
    device: str,
) -> tuple[np.ndarray, str]:
    if not colpali_index_path.exists():
        return np.zeros(len(records), dtype=np.float64), "no_colpali_index"

    payload = torch.load(colpali_index_path, map_location="cpu")
    idx_records: list[dict] = payload.get("records", [])
    idx_embs: list[torch.Tensor] = payload.get("embeddings", [])
    if not idx_records or not idx_embs or len(idx_records) != len(idx_embs):
        return np.zeros(len(records), dtype=np.float64), "invalid_colpali_index"

    # map id -> embedding index
    id_to_idx = {str(r.get("id", "")): i for i, r in enumerate(idx_records)}
    selected_idx = []
    for r in records:
        rid = str(r.get("id", ""))
        if rid in id_to_idx:
            selected_idx.append(id_to_idx[rid])
        else:
            selected_idx.append(-1)

    valid_positions = [i for i, idx in enumerate(selected_idx) if idx >= 0]
    if not valid_positions:
        return np.zeros(len(records), dtype=np.float64), "no_overlap_with_colpali_index"

    try:
        from transformers import ColPaliForRetrieval, ColPaliProcessor

        real_device = device
        if real_device == "auto":
            real_device = "cuda" if torch.cuda.is_available() else "cpu"
        if real_device == "cuda" and not torch.cuda.is_available():
            real_device = "cpu"

        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPaliForRetrieval.from_pretrained(model_name)
        model = model.to(real_device)
        model.eval()

        with torch.inference_mode():
            q_inputs = processor.process_queries(text=[query], return_tensors="pt")
            q_inputs = {k: (v.to(real_device) if hasattr(v, "to") else v) for k, v in dict(q_inputs).items()}
            q_out = model(**q_inputs)
            q_emb = q_out.embeddings[0].detach().cpu()
            q_mask = q_inputs.get("attention_mask")
            if q_mask is not None:
                m = q_mask[0].detach().cpu().bool()
                if m.ndim == 1 and m.shape[0] == q_emb.shape[0]:
                    kept = q_emb[m]
                    if kept.numel() > 0:
                        q_emb = kept

        passage_embs = [idx_embs[selected_idx[i]] for i in valid_positions]
        score_tensor = processor.score_retrieval(
            query_embeddings=[q_emb],
            passage_embeddings=passage_embs,
            batch_size=64,
            output_dtype=torch.float32,
            output_device="cpu",
        )
        scores_subset = score_tensor[0].detach().cpu().numpy().astype(np.float64)

        out = np.zeros(len(records), dtype=np.float64)
        for p, s in zip(valid_positions, scores_subset):
            out[p] = float(s)
        return out, "ok"
    except Exception as exc:
        return np.zeros(len(records), dtype=np.float64), f"colpali_local_failed:{exc}"


def run_colpali_engine_scores(
    query: str,
    records: list[dict],
    *,
    model_name: str,
    device: str,
    batch_size: int = 2,
    allow_cpu: bool = False,
) -> tuple[np.ndarray, str]:
    """
    Optional backend using `colpali-engine`.
    This computes image embeddings on-the-fly from record image paths.
    """
    try:
        from colpali_engine.models import ColPali, ColPaliProcessor
    except Exception as exc:
        return np.zeros(len(records), dtype=np.float64), f"colpali_engine_import_failed:{exc}"

    image_paths = [Path(str(r.get("image_path", ""))) for r in records]
    valid = [(i, p) for i, p in enumerate(image_paths) if p.exists()]
    if not valid:
        return np.zeros(len(records), dtype=np.float64), "no_valid_images_for_colpali_engine"

    try:
        real_device = device
        if real_device == "auto":
            real_device = "cuda" if torch.cuda.is_available() else "cpu"
        if real_device == "cuda" and not torch.cuda.is_available():
            real_device = "cpu"
        if real_device == "cpu" and not allow_cpu:
            return np.zeros(len(records), dtype=np.float64), "colpali_engine_cpu_disabled"

        dtype = torch.float32
        if real_device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=real_device,
        ).eval()
        processor = ColPaliProcessor.from_pretrained(model_name)

        # Query embedding
        with torch.inference_mode():
            q_batch = processor.process_queries([query])
            q_batch = {k: (v.to(real_device) if hasattr(v, "to") else v) for k, v in q_batch.items()}
            q_emb = model(**q_batch)

        # Passage embeddings
        p_emb_list = []
        valid_positions = []
        for start in range(0, len(valid), max(1, int(batch_size))):
            batch = valid[start : start + max(1, int(batch_size))]
            imgs = []
            pos = []
            for i, p in batch:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    pos.append(i)
                except Exception:
                    continue
            if not imgs:
                continue
            with torch.inference_mode():
                p_batch = processor.process_images(imgs)
                p_batch = {k: (v.to(real_device) if hasattr(v, "to") else v) for k, v in p_batch.items()}
                emb = model(**p_batch)
            # emb may be tensor [B, T, D] or list
            if isinstance(emb, torch.Tensor):
                for bi in range(emb.shape[0]):
                    p_emb_list.append(emb[bi].detach().cpu())
            elif isinstance(emb, list):
                for e in emb:
                    p_emb_list.append(e.detach().cpu() if hasattr(e, "detach") else e)
            else:
                return np.zeros(len(records), dtype=np.float64), "colpali_engine_unknown_embedding_format"
            valid_positions.extend(pos)

        if not p_emb_list:
            return np.zeros(len(records), dtype=np.float64), "colpali_engine_empty_passage_embeddings"

        scorer = getattr(processor, "score_retrieval", None) or getattr(processor, "score_multi_vector", None)
        if scorer is None:
            return np.zeros(len(records), dtype=np.float64), "colpali_engine_no_scorer"

        score_tensor = scorer(
            query_embeddings=[q_emb[0].detach().cpu() if isinstance(q_emb, torch.Tensor) else q_emb],
            passage_embeddings=p_emb_list,
        )
        # accept [1, N] or [N]
        if hasattr(score_tensor, "detach"):
            scores_subset = score_tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)
        else:
            scores_subset = np.asarray(score_tensor, dtype=np.float64).reshape(-1)

        out = np.zeros(len(records), dtype=np.float64)
        for p, s in zip(valid_positions, scores_subset):
            out[p] = float(s)
        return out, "ok"
    except Exception as exc:
        return np.zeros(len(records), dtype=np.float64), f"colpali_engine_failed:{exc}"


def run_byaldi_scores(
    query: str,
    records: list[dict],
    *,
    index_name: str,
    index_root: str | None,
    top_k: int,
) -> tuple[np.ndarray, str]:
    """
    Optional backend using Byaldi prebuilt index.
    Requires a byaldi index that maps to the same document pages.
    """
    try:
        from byaldi import RAGMultiModalModel
    except Exception as exc:
        return np.zeros(len(records), dtype=np.float64), f"byaldi_import_failed:{exc}"

    if not index_name:
        return np.zeros(len(records), dtype=np.float64), "byaldi_index_name_missing"

    try:
        if index_root:
            rag = RAGMultiModalModel.from_index(index_name, index_root=index_root)
        else:
            rag = RAGMultiModalModel.from_index(index_name)
        hits = rag.search(query, k=max(10, int(top_k)))
    except Exception as exc:
        return np.zeros(len(records), dtype=np.float64), f"byaldi_search_failed:{exc}"

    score_by_page: dict[str, float] = {}
    for h in hits or []:
        if isinstance(h, dict):
            # common keys observed in byaldi examples
            page_num = h.get("page_num") or h.get("page") or h.get("result_index")
            score = float(h.get("score", 0.0) or 0.0)
            metadata = h.get("metadata") or {}
            source = str(metadata.get("source", "") or metadata.get("file_name", "")).strip()
            if page_num is not None:
                pid = f"{source}:{int(page_num)}" if source else str(int(page_num))
                score_by_page[pid] = max(score_by_page.get(pid, 0.0), score)

    out = np.zeros(len(records), dtype=np.float64)
    if not score_by_page:
        return out, "byaldi_no_scores"

    for i, r in enumerate(records):
        source = str(r.get("source", "")).strip()
        page = int(r.get("page", 0) or 0)
        pid_full = f"{source}:{page}" if source and page else ""
        pid_page = str(page) if page else ""
        out[i] = max(
            float(score_by_page.get(pid_full, 0.0)),
            float(score_by_page.get(pid_page, 0.0)),
        )
    return out, "ok"


def run_endpoint_scores(
    query: str,
    records: list[dict],
    *,
    endpoint_url: str,
    api_key: str,
    timeout_sec: int = 120,
    endpoint_max_retries: int = 2,
    retry_backoff_ms: int = 700,
    local_cache_size: int = 256,
    endpoint_image_max_side: int = 896,
    endpoint_jpeg_quality: int = 80,
) -> tuple[np.ndarray, str, str, dict]:
    """
    Optional custom endpoint backend.
    Expected response schema:
      { "scores": [ {"id":"...", "score":0.87}, ... ] }
    """
    candidate_quality = {
        "candidate_count": 0,
        "image_usable_count": 0,
        "image_base64_coverage": 0.0,
        "dropped_no_image_count": 0,
    }
    if not endpoint_url.strip():
        return np.zeros(len(records), dtype=np.float64), "endpoint_url_missing", "endpoint_call_failed", candidate_quality

    def _encode_image_base64(path_str: str) -> str:
        p = Path(str(path_str or "").strip())
        if not p.exists() or not p.is_file():
            return ""
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                max_side = max(1, int(endpoint_image_max_side))
                if max(img.size) > max_side:
                    ratio = max_side / max(img.size)
                    new_size = (
                        max(1, int(img.size[0] * ratio)),
                        max(1, int(img.size[1] * ratio)),
                    )
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(
                    buf,
                    format="JPEG",
                    quality=max(40, min(95, int(endpoint_jpeg_quality))),
                    optimize=True,
                )
                return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            try:
                return base64.b64encode(p.read_bytes()).decode("ascii")
            except Exception:
                return ""

    # Build page-level fallback map once.
    page_fallback: dict[str, str] = {}
    for r in records:
        pid = str(r.get("page_id", "")).strip()
        pth = str(r.get("image_path", "")).strip()
        if not pid or not pth:
            continue
        p = Path(pth)
        if p.exists() and p.is_file():
            if str(r.get("image_level", "")).strip().lower() == "page":
                page_fallback[pid] = pth
            elif pid not in page_fallback:
                page_fallback[pid] = pth

    def _choose_image_path(rec: dict) -> str:
        pth = str(rec.get("image_path", "")).strip()
        p = Path(pth) if pth else None
        if pth and p is not None and p.exists() and p.is_file():
            return pth
        pid = str(rec.get("page_id", "")).strip()
        fallback = page_fallback.get(pid, "")
        if fallback:
            return fallback
        return ""

    def _build_candidates() -> tuple[list[dict], list[str], dict]:
        rows: list[dict] = []
        ids: list[str] = []
        seen: set[str] = set()
        dropped_no_image = 0
        for r in records:
            rid = str(r.get("id", "")).strip()
            if not rid or rid in seen:
                continue
            image_path = _choose_image_path(r)
            image_b64 = _encode_image_base64(image_path) if image_path else ""
            if not image_b64:
                dropped_no_image += 1
                continue
            rows.append(
                {
                    "id": rid,
                    "page_id": str(r.get("page_id", "")).strip(),
                    "source": str(r.get("source", "")).strip(),
                    "page": int(r.get("page", 0) or 0),
                    "image_path": image_path,
                    "image_base64": image_b64,
                    "text_preview": str(r.get("text", ""))[:600],
                }
            )
            ids.append(rid)
            seen.add(rid)
        cq = {
            "candidate_count": int(len(rows)),
            "image_usable_count": int(len(rows)),
            "image_base64_coverage": round(float(len(rows)) / max(1, len(records)), 4),
            "dropped_no_image_count": int(dropped_no_image),
        }
        return rows, ids, cq

    def _extract_scores_payload(data_obj) -> dict | None:
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
                if parsed is not None:
                    return parsed
        return None

    candidates_full, candidate_ids, candidate_quality = _build_candidates()
    if not candidates_full:
        return np.zeros(len(records), dtype=np.float64), "endpoint_no_scores", "endpoint_no_candidates", candidate_quality

    corpus_seed = "|".join(candidate_ids)
    corpus_hash = hashlib.sha1(corpus_seed.encode("utf-8", errors="ignore")).hexdigest()[:16]
    corpus_id = f"doc-corpus-{len(candidate_ids)}-{corpus_hash}"

    payload = {
        "query": query,
        "corpus_id": corpus_id,
        "candidate_ids": candidate_ids,
        "candidates": candidates_full,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoint_raw = endpoint_url.strip()
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "endpoint": endpoint_raw,
                "query": query,
                "candidate_ids": candidate_ids,
                "corpus_id": corpus_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8", errors="ignore")
    ).hexdigest()
    cached = ENDPOINT_SCORE_CACHE.get(cache_key)
    if isinstance(cached, dict):
        id_to_score = cached.get("id_to_score", {}) if isinstance(cached.get("id_to_score"), dict) else {}
        if id_to_score:
            out = np.zeros(len(records), dtype=np.float64)
            for i, r in enumerate(records):
                out[i] = float(id_to_score.get(str(r.get("id", "")), 0.0))
            return out, "ok", "ok", candidate_quality

    def _call_once() -> tuple[dict | None, str]:
        data_local = None
        last_err_local = ""
        is_http_endpoint = endpoint_raw.lower().startswith(("http://", "https://"))
        if is_http_endpoint:
            try:
                with httpx.Client(timeout=timeout_sec) as client:
                    res = client.post(endpoint_raw, json=payload, headers=headers)
                    res.raise_for_status()
                    data_local = res.json()
            except Exception as exc:
                last_err_local = str(exc)

        if data_local is None:
            try:
                from gradio_client import Client

                client = Client(endpoint_raw, hf_token=api_key or None, verbose=False)
                try:
                    one_payload = {
                        "query": query,
                        "corpus_id": corpus_id,
                        "candidate_ids": candidate_ids,
                        "candidates": candidates_full,
                    }
                    one_out = client.predict(one_payload, api_name="/register_and_score")
                    parsed_one = _extract_scores_payload(one_out)
                    if isinstance(parsed_one, dict) and isinstance(parsed_one.get("scores"), list):
                        data_local = parsed_one
                except Exception:
                    pass

                if data_local is None:
                    try:
                        reg_payload = {"corpus_id": corpus_id}
                        reg_out = client.predict(reg_payload, api_name="/register_corpus")
                        reg_parsed = _extract_scores_payload(reg_out)
                        reg_ok = bool(isinstance(reg_parsed, dict) and reg_parsed.get("ok", False))
                        if not reg_ok:
                            reg_payload = {"corpus_id": corpus_id, "candidates": candidates_full}
                            reg_out = client.predict(reg_payload, api_name="/register_corpus")
                            reg_parsed = _extract_scores_payload(reg_out)
                            reg_ok = bool(isinstance(reg_parsed, dict) and reg_parsed.get("ok", False))
                        if reg_ok:
                            score_cached_payload = {
                                "query": query,
                                "corpus_id": corpus_id,
                                "candidate_ids": candidate_ids,
                            }
                            cached_out = client.predict(score_cached_payload, api_name="/score_cached")
                            parsed_cached = _extract_scores_payload(cached_out)
                            if (
                                isinstance(parsed_cached, dict)
                                and str(parsed_cached.get("error", "")).strip() == "unknown_corpus_id"
                            ):
                                score_cached_payload["candidates"] = candidates_full
                                cached_out = client.predict(score_cached_payload, api_name="/score_cached")
                                parsed_cached = _extract_scores_payload(cached_out)
                            if isinstance(parsed_cached, dict) and isinstance(parsed_cached.get("scores"), list):
                                data_local = parsed_cached
                    except Exception:
                        pass

                if data_local is None:
                    for api_name in ("/score", "/api_endpoint"):
                        try:
                            out_obj = client.predict(payload, api_name=api_name)
                            parsed = _extract_scores_payload(out_obj)
                            if isinstance(parsed, dict):
                                data_local = parsed
                                break
                        except Exception:
                            try:
                                out_obj = client.predict(json.dumps(payload, ensure_ascii=False), api_name=api_name)
                                parsed = _extract_scores_payload(out_obj)
                                if isinstance(parsed, dict):
                                    data_local = parsed
                                    break
                            except Exception:
                                continue
            except Exception as exc:
                if last_err_local:
                    last_err_local = f"{last_err_local} | gradio_client:{exc}"
                else:
                    last_err_local = f"gradio_client:{exc}"
        return data_local, last_err_local

    max_attempts = max(1, int(endpoint_max_retries) + 1)
    data = None
    last_err = ""
    for attempt in range(max_attempts):
        data, last_err = _call_once()
        if isinstance(data, dict) and isinstance(data.get("scores"), list):
            break
        if attempt < (max_attempts - 1):
            sleep_sec = (max(0, int(retry_backoff_ms)) * (2**attempt)) / 1000.0
            time.sleep(min(5.0, sleep_sec))

    if data is None:
        return (
            np.zeros(len(records), dtype=np.float64),
            f"endpoint_call_failed:{last_err or 'unknown'}",
            "endpoint_call_failed",
            candidate_quality,
        )

    if not isinstance(data, dict):
        return np.zeros(len(records), dtype=np.float64), "endpoint_schema_invalid", "endpoint_schema_invalid", candidate_quality

    scores_obj = data.get("scores")
    if not isinstance(scores_obj, list):
        return np.zeros(len(records), dtype=np.float64), "endpoint_schema_invalid", "endpoint_schema_invalid", candidate_quality

    id_to_score = {}
    candidate_id_set = {str(x).strip() for x in candidate_ids if str(x).strip()}
    mismatched_ids = 0
    for item in (scores_obj or []):
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id", "")).strip()
        if not rid:
            continue
        if rid not in candidate_id_set:
            mismatched_ids += 1
            continue
        try:
            id_to_score[rid] = float(item.get("score", 0.0) or 0.0)
        except Exception:
            continue

    if not id_to_score:
        detail = "endpoint_id_mismatch" if mismatched_ids > 0 else "endpoint_empty_scores"
        return np.zeros(len(records), dtype=np.float64), "endpoint_no_scores", detail, candidate_quality

    out = np.zeros(len(records), dtype=np.float64)
    for i, r in enumerate(records):
        out[i] = float(id_to_score.get(str(r.get("id", "")), 0.0))
    ENDPOINT_SCORE_CACHE[cache_key] = {"id_to_score": id_to_score}
    if len(ENDPOINT_SCORE_CACHE) > max(8, int(local_cache_size)):
        # delete oldest inserted key
        oldest_key = next(iter(ENDPOINT_SCORE_CACHE))
        ENDPOINT_SCORE_CACHE.pop(oldest_key, None)

    status_detail = "ok"
    if mismatched_ids > 0 or len(id_to_score) < len(candidate_id_set):
        status_detail = "ok_partial"
    return out, "ok", status_detail, candidate_quality


def image_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = path.suffix.lower().lstrip(".") or "png"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


def parse_json_object(raw: str) -> dict | None:
    raw = (raw or "").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    s = raw.find("{")
    e = raw.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = json.loads(raw[s : e + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _normalize_fact_text(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"[\[\]\(\)\{\},.:;!?/\\|\"'`~@#$%^&*+=<>]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


GROUNDING_GLOSSARY = {
    "queue": ["queue", "คิว"],
    "array": ["array", "อาร์เรย์"],
    "linked_list": ["linked list", "ลิงค์ลิสต์", "ลิงก์ลิสต์"],
    "node": ["node", "โหนด"],
    "head": ["head", "เฮด"],
    "next": ["next", "ถัดไป"],
    "pointer": ["pointer", "พอยน์เตอร์", "ลิงก์", "link"],
    "front": ["front"],
    "rear": ["rear"],
    "enqueue": ["enqueue", "นำเข้า", "เพิ่มข้อมูลเข้าคิว"],
    "dequeue": ["dequeue", "นำออก", "ดึงข้อมูลออก"],
    "insert": ["insert", "เพิ่มโหนด", "แทรกโหนด", "แทรก"],
    "delete": ["delete", "remove", "ลบโหนด", "ลบ"],
    "circular": ["circular", "วงกลม", "วนกลับ", "wrap"],
    "full": ["full", "เต็ม"],
    "empty": ["empty", "ว่าง"],
}


CANONICAL_TERM_LABELS = {
    "queue": "queue/คิว",
    "array": "array/อาร์เรย์",
    "linked_list": "linked list/ลิงค์ลิสต์",
    "node": "node/โหนด",
    "head": "head",
    "next": "next",
    "pointer": "pointer/link",
    "front": "front",
    "rear": "rear",
    "enqueue": "enqueue",
    "dequeue": "dequeue",
    "insert": "insert",
    "delete": "delete",
    "circular": "circular",
    "full": "full",
    "empty": "empty",
}


CH2_OPERATION_PATTERNS = {
    "insert": ["insert", "เพิ่มโหนด", "แทรกโหนด", "แทรก"],
    "delete": ["delete", "remove", "ลบโหนด", "ลบ"],
    "node": ["node", "โหนด"],
    "head": ["head", "เฮด"],
    "next": ["next", "ถัดไป", "link"],
}


def _detect_canonical_terms(text: str) -> set[str]:
    q = _normalize_fact_text(text)
    found: set[str] = set()
    if not q:
        return found
    for term, synonyms in GROUNDING_GLOSSARY.items():
        for syn in synonyms:
            s = _normalize_fact_text(syn)
            if s and s in q:
                found.add(term)
                break
    return found


def _detect_ch2_operation_tags(text: str) -> set[str]:
    q = _normalize_fact_text(text)
    found: set[str] = set()
    if not q:
        return found
    for tag, patterns in CH2_OPERATION_PATTERNS.items():
        for p in patterns:
            pp = _normalize_fact_text(p)
            if pp and pp in q:
                found.add(tag)
                break
    return found


def _is_chapter2_operation_context(candidate: dict, query_profile: dict | None) -> bool:
    tid = str(candidate.get("best_topic_id", "") or "").strip()
    sec = str(candidate.get("section_id", "") or "").strip()
    chapter = str(candidate.get("chapter_id", "") or "").strip()
    if tid.startswith("2.4.2") or sec.startswith("2.4.2") or chapter == "2":
        if isinstance(query_profile, dict) and bool(query_profile.get("operation_intent")):
            return True
    return tid.startswith("2.4.2") or sec.startswith("2.4.2")


def postprocess_grounding_parsed(
    parsed: dict,
    *,
    query: str,
    candidate: dict,
    query_profile: dict | None,
) -> dict:
    if not isinstance(parsed, dict):
        return parsed

    facts_in = parsed.get("grounded_facts", [])
    facts = [str(x).strip() for x in facts_in if str(x).strip()] if isinstance(facts_in, list) else []
    source_blob = " ".join(
        [
            " ".join(facts),
            str(parsed.get("structure_type", "")),
            str(parsed.get("limitations", "")),
            str(candidate.get("text", "")),
            " ".join(str(x) for x in (candidate.get("tags", []) or []) if str(x).strip()),
            " ".join(str(x) for x in (candidate.get("structure_labels", []) or []) if str(x).strip()),
            " ".join(str(x) for x in (candidate.get("figure_refs", []) or []) if str(x).strip()),
            str(candidate.get("section_title", "")),
            str(candidate.get("best_topic_title", "")),
        ]
    )
    canonical_terms = _detect_canonical_terms(source_blob)
    chapter2_op_tags: set[str] = set()
    if _is_chapter2_operation_context(candidate, query_profile):
        chapter2_op_tags = _detect_ch2_operation_tags(source_blob)
        canonical_terms.update(chapter2_op_tags)

    extra_facts = []
    if canonical_terms:
        labels = [CANONICAL_TERM_LABELS.get(t, t) for t in sorted(canonical_terms)]
        extra_facts.append("คำสำคัญที่ตรวจพบจากหลักฐานภาพ: " + ", ".join(labels))
    if chapter2_op_tags:
        extra_facts.append("แท็กกระบวนการบท 2.4.2 ที่ตรวจพบ: " + ", ".join(sorted(chapter2_op_tags)))

    merged = []
    seen = set()
    for f in facts + extra_facts:
        key = _normalize_fact_text(f)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(f)

    parsed2 = dict(parsed)
    parsed2["grounded_facts"] = merged[:10]
    parsed2["normalized_terms"] = sorted(canonical_terms)
    parsed2["chapter_operation_tags"] = sorted(chapter2_op_tags)
    parsed2["postprocess"] = "glossary_synonym_normalization_v1"
    return parsed2


def _majority_string(values: list[str]) -> str:
    votes: dict[str, int] = {}
    first_seen: dict[str, str] = {}
    for v in values:
        raw = str(v or "").strip()
        if not raw:
            continue
        key = _normalize_fact_text(raw)
        if not key:
            continue
        votes[key] = votes.get(key, 0) + 1
        first_seen.setdefault(key, raw)
    if not votes:
        return ""
    top_key = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return first_seen.get(top_key, top_key)


def merge_grounding_samples(samples: list[dict], *, min_votes: int) -> tuple[dict, dict]:
    valid = [s for s in samples if isinstance(s, dict) and isinstance(s.get("parsed"), dict)]
    if not valid:
        return {}, {"runs": len(samples), "valid_runs": 0, "agreement_ratio": 0.0, "consensus_facts": 0}

    fact_votes: dict[str, int] = {}
    fact_display: dict[str, str] = {}
    for s in valid:
        parsed = s.get("parsed", {}) if isinstance(s.get("parsed", {}), dict) else {}
        facts = parsed.get("grounded_facts", [])
        if not isinstance(facts, list):
            continue
        seen_local = set()
        for f in facts:
            raw = str(f).strip()
            if not raw:
                continue
            key = _normalize_fact_text(raw)
            if not key or key in seen_local:
                continue
            seen_local.add(key)
            fact_votes[key] = fact_votes.get(key, 0) + 1
            fact_display.setdefault(key, raw)

    vote_threshold = max(1, min(int(min_votes), len(valid)))
    ranked = sorted(fact_votes.items(), key=lambda x: (-x[1], x[0]))
    consensus_keys = [k for k, v in ranked if v >= vote_threshold]
    if not consensus_keys and ranked:
        # Fallback: keep top-voted facts to avoid empty grounding.
        top_vote = ranked[0][1]
        consensus_keys = [k for k, v in ranked if v == top_vote][:3]

    consensus_facts = [fact_display.get(k, k) for k in consensus_keys[:8]]
    structure_type = _majority_string(
        [str((s.get("parsed", {}) or {}).get("structure_type", "")) for s in valid]
    )
    limitations = _majority_string(
        [str((s.get("parsed", {}) or {}).get("limitations", "")) for s in valid]
    )

    parsed = {
        "structure_type": structure_type or "unknown",
        "grounded_facts": consensus_facts,
        "limitations": limitations or "",
    }
    agreement_ratio = (
        float(len(consensus_keys) / max(1, len(fact_votes)))
        if fact_votes else 0.0
    )
    stats = {
        "runs": len(samples),
        "valid_runs": len(valid),
        "vote_threshold": vote_threshold,
        "raw_unique_facts": int(len(fact_votes)),
        "consensus_facts": int(len(consensus_keys)),
        "agreement_ratio": round(float(agreement_ratio), 4),
    }
    return parsed, stats


def load_hard_negative_rules(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rules = payload.get("rules", payload) if isinstance(payload, dict) else payload
    if not isinstance(rules, list):
        return []
    out = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        out.append(r)
    return out


def hard_negative_penalty(
    query_profile: dict,
    rec: dict,
    *,
    rules: list[dict],
    penalty_max: float,
) -> float:
    if not rules:
        return 0.0
    q_blob = " ".join(
        [
            str(query_profile.get("query_norm", "")),
            " ".join(str(x) for x in (query_profile.get("query_tokens", []) or [])),
            " ".join(str(x) for x in (query_profile.get("operation_terms", []) or [])),
            " ".join(str(x) for x in (query_profile.get("structure_terms", []) or [])),
        ]
    ).lower()
    r_blob = " ".join(
        [
            str(rec.get("text", "")),
            " ".join(str(x) for x in (rec.get("tags", []) or [])),
            " ".join(str(x) for x in (rec.get("structure_labels", []) or [])),
            " ".join(str(x) for x in (rec.get("figure_refs", []) or [])),
        ]
    ).lower()
    best = 0.0
    for rule in rules:
        q_any = [str(x).strip().lower() for x in (rule.get("query_any", []) or []) if str(x).strip()]
        q_all = [str(x).strip().lower() for x in (rule.get("query_all", []) or []) if str(x).strip()]
        n_any = [str(x).strip().lower() for x in (rule.get("negative_any", []) or []) if str(x).strip()]
        n_all = [str(x).strip().lower() for x in (rule.get("negative_all", []) or []) if str(x).strip()]
        if q_any and not any(x in q_blob for x in q_any):
            continue
        if q_all and not all(x in q_blob for x in q_all):
            continue
        if n_any and not any(x in r_blob for x in n_any):
            continue
        if n_all and not all(x in r_blob for x in n_all):
            continue
        p = float(rule.get("penalty", 0.0) or 0.0)
        if p > best:
            best = p
    return max(0.0, min(float(penalty_max), best))


def vlm_rerank(
    *,
    hf_client: InferenceClient,
    model: str,
    query: str,
    candidates: list[dict],
    require_structure: bool,
) -> list[dict]:
    out = []
    for c in candidates:
        img_path = Path(str(c.get("image_path", "")))
        if not img_path.exists():
            c2 = dict(c)
            c2["vlm_score"] = 0.0
            c2["vlm_reason"] = "image_not_found"
            out.append(c2)
            continue

        prompt = (
            "ให้คะแนนความเกี่ยวข้องของภาพกับคำถามผู้ใช้แบบ 0.0-1.0 แล้วตอบ JSON เท่านั้น:\n"
            "{\"score\":0.0,\"reason\":\"...\",\"structure_detected\":true,\"evidence\":[\"...\"]}\n"
            f"คำถาม: {query}\n"
            f"เงื่อนไข require_structure={str(require_structure).lower()}\n"
            "ให้ยึดเฉพาะสิ่งที่เห็นในภาพและข้อความประกอบสั้นๆ เท่านั้น"
        )

        try:
            data_url = image_to_data_url(img_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            res = hf_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=350,
                temperature=0.1,
                top_p=0.9,
            )
            raw = (res.choices[0].message.content or "").strip()
            obj = parse_json_object(raw) or {}
            score = float(obj.get("score", 0.0) or 0.0)
            score = max(0.0, min(1.0, score))
            reason = str(obj.get("reason", "")).strip()
            c2 = dict(c)
            c2["vlm_score"] = score
            c2["vlm_reason"] = reason
            c2["vlm_raw"] = raw
            out.append(c2)
        except Exception as exc:
            c2 = dict(c)
            c2["vlm_score"] = 0.0
            c2["vlm_reason"] = f"vlm_rerank_failed:{exc}"
            out.append(c2)
    return out


def visual_ground_once(
    *,
    hf_client: InferenceClient,
    model: str,
    query: str,
    candidate: dict,
    temperature: float,
    top_p: float,
) -> dict:
    img_path = Path(str(candidate.get("image_path", "")))
    if not img_path.exists():
        return {"error": "image_not_found"}

    prompt = (
        "วิเคราะห์ภาพแล้วทำ visual grounding แบบข้อความ JSON เท่านั้น:\n"
        "{\"structure_type\":\"...\",\"grounded_facts\":[\"...\"],\"limitations\":\"...\"}\n"
        f"คำถามผู้ใช้: {query}\n"
        "ให้ยึดเฉพาะหลักฐานที่มองเห็นในภาพ"
    )

    try:
        data_url = image_to_data_url(img_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        res = hf_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=450,
            temperature=max(0.0, float(temperature)),
            top_p=max(0.1, min(1.0, float(top_p))),
        )
        raw = (res.choices[0].message.content or "").strip()
        parsed = parse_json_object(raw)
        return {"parsed": parsed, "raw": raw}
    except Exception as exc:
        return {"error": str(exc)}


def visual_ground(
    *,
    hf_client: InferenceClient,
    model: str,
    query: str,
    candidate: dict,
    ensemble_runs: int,
    consensus_min_votes: int,
    temperature: float,
    top_p: float,
    query_profile: dict | None = None,
) -> dict:
    runs = max(1, int(ensemble_runs))
    samples = []
    for _ in range(runs):
        out = visual_ground_once(
            hf_client=hf_client,
            model=model,
            query=query,
            candidate=candidate,
            temperature=temperature,
            top_p=top_p,
        )
        samples.append(out)

    parsed, stats = merge_grounding_samples(samples, min_votes=max(1, int(consensus_min_votes)))
    if not parsed:
        return {
            "error": "grounding_consensus_failed",
            "samples": samples,
            "consensus": stats,
        }
    parsed = postprocess_grounding_parsed(
        parsed,
        query=query,
        candidate=candidate if isinstance(candidate, dict) else {},
        query_profile=query_profile if isinstance(query_profile, dict) else {},
    )
    # Keep one representative raw payload for compatibility.
    raw = ""
    for s in samples:
        if isinstance(s, dict) and str(s.get("raw", "")).strip():
            raw = str(s.get("raw", "")).strip()
            break
    return {
        "parsed": parsed,
        "raw": raw,
        "samples": samples,
        "consensus": stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid + ColPali + VLM rerank + grounding retrieval.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--pages-jsonl", default="indexes/colpali/pages.jsonl")
    ap.add_argument("--hierarchy-index", default="indexes/hierarchical/topic_hierarchy.json")
    ap.add_argument("--require-structure", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=20)

    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "metadata", "colpali_local", "colpali_engine", "byaldi", "colpali_endpoint"],
    )
    ap.add_argument("--colpali-index", default="indexes/colpali/colpali_index.pt")
    ap.add_argument("--colpali-model", default="vidore/colpali-v1.3-hf")
    ap.add_argument("--colpali-engine-model", default="vidore/colpali-v1.3")
    ap.add_argument("--colpali-engine-allow-cpu", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--byaldi-index-name", default=os.getenv("BYALDI_INDEX_NAME", ""))
    ap.add_argument("--byaldi-index-root", default=os.getenv("BYALDI_INDEX_ROOT", ""))
    ap.add_argument("--colpali-endpoint-url", default=os.getenv("COLPALI_ENDPOINT_URL", ""))
    ap.add_argument(
        "--use-splade",
        action="store_true",
        default=os.getenv("VISUAL_USE_SPLADE", "0").strip().lower() in {"1", "true", "yes", "on"},
    )
    ap.add_argument(
        "--sparse-strategy",
        default=os.getenv("VISUAL_SPARSE_STRATEGY", "bm25"),
        choices=["splade", "bm25"],
        help="Sparse channel strategy used alongside ColPali in hybrid fusion.",
    )
    ap.add_argument(
        "--splade-mode",
        default=os.getenv("VISUAL_SPLADE_MODE", "auto"),
        choices=["auto", "hf_api", "local", "disabled"],
    )
    ap.add_argument("--splade-model", default=os.getenv("VISUAL_SPLADE_MODEL", "naver/splade-v3"))
    ap.add_argument("--splade-provider", default=os.getenv("VISUAL_SPLADE_PROVIDER", "hf-inference"))
    ap.add_argument("--splade-device", default=os.getenv("VISUAL_SPLADE_DEVICE", "auto"))
    ap.add_argument("--splade-max-length", type=int, default=int(os.getenv("VISUAL_SPLADE_MAX_LENGTH", "256")))
    ap.add_argument("--splade-batch-size", type=int, default=int(os.getenv("VISUAL_SPLADE_BATCH_SIZE", "8")))
    ap.add_argument("--endpoint-timeout-sec", type=int, default=int(os.getenv("VISUAL_ENDPOINT_TIMEOUT_SEC", "120")))
    ap.add_argument("--endpoint-max-retries", type=int, default=int(os.getenv("VISUAL_ENDPOINT_MAX_RETRIES", "2")))
    ap.add_argument("--endpoint-retry-backoff-ms", type=int, default=int(os.getenv("VISUAL_ENDPOINT_RETRY_BACKOFF_MS", "700")))
    ap.add_argument("--endpoint-local-cache-size", type=int, default=int(os.getenv("VISUAL_ENDPOINT_LOCAL_CACHE_SIZE", "256")))
    ap.add_argument("--endpoint-image-max-side", type=int, default=int(os.getenv("VISUAL_ENDPOINT_IMAGE_MAX_SIDE", "896")))
    ap.add_argument("--endpoint-jpeg-quality", type=int, default=int(os.getenv("VISUAL_ENDPOINT_JPEG_QUALITY", "80")))

    ap.add_argument("--use-vlm-rerank", action="store_true")
    ap.add_argument("--vlm-rerank-top-m", type=int, default=int(os.getenv("VISUAL_VLM_RERANK_TOP_M", "12")))
    ap.add_argument("--use-visual-grounding", action="store_true")
    ap.add_argument("--ground-top-n", type=int, default=1)
    ap.add_argument("--grounding-ensemble-runs", type=int, default=int(os.getenv("VISUAL_GROUNDING_ENSEMBLE_RUNS", "2")))
    ap.add_argument("--grounding-consensus-min-votes", type=int, default=int(os.getenv("VISUAL_GROUNDING_CONSENSUS_MIN_VOTES", "2")))
    ap.add_argument("--grounding-temperature", type=float, default=float(os.getenv("VISUAL_GROUNDING_TEMPERATURE", "0.0")))
    ap.add_argument("--grounding-top-p", type=float, default=float(os.getenv("VISUAL_GROUNDING_TOP_P", "0.9")))
    ap.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-7B-Instruct")

    ap.add_argument("--topic-threshold", type=float, default=0.18)
    ap.add_argument(
        "--metadata-filter-strict",
        action="store_true",
        default=os.getenv("VISUAL_METADATA_FILTER_STRICT", "1").strip().lower() in {"1", "true", "yes", "on"},
        help="Use strict metadata pre-filtering (topic/chapter/section) before rerank; skip relaxed chapter fallback.",
    )
    ap.add_argument(
        "--use-intent-llm",
        action="store_true",
        default=os.getenv("VISUAL_INTENT_USE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"},
        help="Use LLM intent classification JSON in D2 before metadata pre-filtering",
    )
    ap.add_argument(
        "--intent-model",
        default=os.getenv("VISUAL_INTENT_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
    )
    ap.add_argument(
        "--intent-candidate-top",
        type=int,
        default=int(os.getenv("VISUAL_INTENT_CANDIDATE_TOP", "12")),
    )
    ap.add_argument(
        "--intent-min-confidence",
        type=float,
        default=float(os.getenv("VISUAL_INTENT_MIN_CONFIDENCE", "0.45")),
    )
    ap.add_argument(
        "--intent-max-topic-ids",
        type=int,
        default=int(os.getenv("VISUAL_INTENT_MAX_TOPIC_IDS", "14")),
    )
    ap.add_argument(
        "--use-query-expansion-llm",
        action="store_true",
        default=os.getenv("VISUAL_QUERY_EXPANSION_USE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"},
        help="Use LLM query expansion (Query2doc-style) to add sparse retrieval terms.",
    )
    ap.add_argument(
        "--query-expansion-model",
        default=os.getenv("VISUAL_QUERY_EXPANSION_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
    )
    ap.add_argument(
        "--query-expansion-max-terms",
        type=int,
        default=int(os.getenv("VISUAL_QUERY_EXPANSION_MAX_TERMS", "8")),
    )
    ap.add_argument(
        "--topic-adjacent-page-gap",
        type=int,
        default=int(os.getenv("VISUAL_TOPIC_ADJACENT_PAGE_GAP", "3")),
    )
    ap.add_argument(
        "--topic-adjacent-max-extra-pages",
        type=int,
        default=int(os.getenv("VISUAL_TOPIC_ADJACENT_MAX_EXTRA_PAGES", "8")),
    )
    ap.add_argument(
        "--prefilter-min-records",
        type=int,
        default=int(os.getenv("VISUAL_PREFILTER_MIN_RECORDS", "8")),
    )
    ap.add_argument(
        "--prefilter-rescue-topn",
        type=int,
        default=int(os.getenv("VISUAL_PREFILTER_RESCUE_TOPN", "24")),
    )
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--disable-figure-centric-rerank", action="store_true")
    ap.add_argument("--figure-boost-weight", type=float, default=0.18)
    ap.add_argument("--operation-boost-weight", type=float, default=0.10)
    ap.add_argument("--disable-coverage-rerank", action="store_true")
    ap.add_argument("--coverage-novelty-weight", type=float, default=0.26)
    ap.add_argument("--coverage-figure-weight", type=float, default=0.62)
    ap.add_argument("--coverage-page-weight", type=float, default=0.38)
    ap.add_argument("--coverage-step-target", type=int, default=4)
    ap.add_argument("--hard-negative-rules", default=os.getenv("VISUAL_HARD_NEGATIVE_RULES", "indexes/hierarchical/hard_negative_rules.json"))
    ap.add_argument("--hard-negative-penalty-max", type=float, default=float(os.getenv("VISUAL_HARD_NEGATIVE_PENALTY_MAX", "0.22")))
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    load_dotenv()
    hf_token = (os.getenv("HUGGINGFACE_READ_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
    hf_client = InferenceClient(api_key=hf_token) if hf_token else None

    pages_path = Path(args.pages_jsonl)
    hierarchy_path = Path(args.hierarchy_index)
    if not pages_path.exists():
        raise FileNotFoundError(f"pages jsonl not found: {pages_path}")

    records = load_jsonl(pages_path)
    hierarchy = load_topic_hierarchy(hierarchy_path)
    mojibake_ratio = hierarchy_mojibake_ratio(hierarchy)
    hierarchy_garbled = mojibake_ratio >= 0.35
    if hierarchy_garbled:
        hierarchy = {}
    q_tokens = tokenize(args.query)
    q_tokens_expanded = expand_query_tokens(args.query, q_tokens)
    query_profile = build_query_profile(args.query, q_tokens_expanded)
    hard_negative_rules = load_hard_negative_rules(Path(args.hard_negative_rules))

    # D2: Intent classification (LLM JSON + heuristic fallback) and metadata pre-filtering.
    ranked_topics = rank_topics_from_hierarchy(args.query, hierarchy, top_n=max(3, int(args.intent_candidate_top)))
    qe_terms: list[str] = []
    qe_status = "not_used"
    if bool(args.use_query_expansion_llm) and hf_client is not None:
        qe_terms, qe_status = expand_query_terms_llm(
            hf_client=hf_client,
            model=str(args.query_expansion_model),
            query=args.query,
            ranked_topics=ranked_topics,
            max_terms=max(0, int(args.query_expansion_max_terms)),
        )
        if qe_terms:
            seen_q = set(q_tokens_expanded)
            for term in qe_terms:
                if term not in seen_q:
                    seen_q.add(term)
                    q_tokens_expanded.append(term)
            query_profile = build_query_profile(args.query, q_tokens_expanded)
    elif bool(args.use_query_expansion_llm) and hf_client is None:
        qe_status = "missing_hf_token"
    heuristic_topic_id, heuristic_topic_score = best_ranked_topic(ranked_topics or [])
    if ranked_topics and bool(query_profile.get("operation_intent")):
        op_markers = (
            "การทำงาน",
            "ดำเนินการ",
            "operation",
            "insert",
            "delete",
            "remove",
            "enqueue",
            "dequeue",
            "push",
            "pop",
            "เพิ่ม",
            "ลบ",
            "แทรก",
        )
        parent_hint = ""
        if heuristic_topic_id and "." in heuristic_topic_id:
            parent_hint = heuristic_topic_id.rsplit(".", 1)[0]
        op_candidates: list[tuple[float, str]] = []
        for t in ranked_topics:
            tid = str(t.get("topic_id", "")).strip()
            if not tid:
                continue
            title = str(t.get("title", "") or "").strip().lower()
            if not any(m in title for m in op_markers):
                continue
            score = float(t.get("score", 0.0) or 0.0)
            if parent_hint and (tid == parent_hint or tid.startswith(parent_hint + ".")):
                score += 0.08
            score += min(0.08, 0.02 * tid.count("."))
            op_candidates.append((score, tid))
        if op_candidates:
            op_candidates.sort(key=lambda x: (-x[0], -x[1].count("."), len(x[1]), x[1]))
            heuristic_topic_id = op_candidates[0][1]
            heuristic_topic_score = float(min(1.0, max(0.0, op_candidates[0][0])))
    heuristic_intent_obj: dict[str, Any] = {}
    if bool(query_profile.get("operation_intent")) and heuristic_topic_id:
        heuristic_intent_obj = {
            "intent_type": "operation",
            "target_topic_ids": [str(heuristic_topic_id).strip()],
            "confidence": 1.0,
            "require_structure": True,
        }

    intent_llm_obj = {}
    intent_llm_status = "not_used"
    if bool(args.use_intent_llm) and hf_client is not None and ranked_topics:
        intent_llm_obj, intent_llm_status = classify_query_intent_llm(
            hf_client=hf_client,
            model=str(args.intent_model),
            query=args.query,
            ranked_topics=ranked_topics,
        )
    elif bool(args.use_intent_llm) and hf_client is None:
        intent_llm_status = "missing_hf_token"

    effective_intent_obj = (
        intent_llm_obj
        if isinstance(intent_llm_obj, dict) and bool(intent_llm_obj)
        else heuristic_intent_obj
    )
    topic_ids, topic_source, topic_score_val = resolve_intent_topic_ids(
        hierarchy=hierarchy,
        ranked_topics=ranked_topics,
        intent_obj=effective_intent_obj if isinstance(effective_intent_obj, dict) else {},
        min_confidence=float(args.intent_min_confidence),
        max_topic_ids=max(1, int(args.intent_max_topic_ids)),
    )
    if isinstance(effective_intent_obj, dict):
        intent_type = str(effective_intent_obj.get("intent_type", "") or "").strip().lower()
        if intent_type:
            query_profile["intent_type"] = intent_type
        if bool(effective_intent_obj.get("require_structure", False)):
            query_profile["structure_intent"] = True
        if intent_type in {"operation"}:
            query_profile["operation_intent"] = True
    topic_id = pick_primary_topic_id(topic_ids, intent_llm_obj if isinstance(intent_llm_obj, dict) else {}, heuristic_topic_id)
    topic_filtered = False
    topic_scope_ids = sorted(topic_ids) if topic_ids else []
    topic_filter_reason = "not_applied"
    prefilter_min_records = max(6, int(getattr(args, "prefilter_min_records", 24)))
    prefilter_rescue_topn = max(prefilter_min_records, int(getattr(args, "prefilter_rescue_topn", 64)))
    filter_trace = {
        "initial_records": int(len(records)),
        "after_strict": int(len(records)),
        "after_relax1": int(len(records)),
        "after_relax2": int(len(records)),
        "final_records": int(len(records)),
        "selected_mode": "not_applied",
    }
    if topic_ids and float(topic_score_val) >= float(args.topic_threshold):
        selected_ids = set(topic_ids)
        if query_profile.get("operation_intent") and topic_id:
            topic_depth = str(topic_id).count(".")
            selected_ids = expand_topic_ids(
                hierarchy,
                topic_id,
                include_parent=bool(topic_depth <= 1),
                include_children=True,
                include_siblings=False,
            ) or selected_ids

        target_chapter = ""
        if isinstance(intent_llm_obj, dict):
            target_chapter = str(intent_llm_obj.get("target_chapter", "") or "").strip()
        if not target_chapter and topic_id:
            target_chapter = str(topic_id).split(".", 1)[0].strip()

        filtered_meta, metadata_filter_mode = filter_by_metadata_topic_ids(
            records,
            selected_ids,
            target_chapter=target_chapter,
            strict_only=bool(args.metadata_filter_strict),
            allow_chapter_fallback=not bool(query_profile.get("operation_intent")),
        )
        if bool(args.metadata_filter_strict):
            # Strict-only policy: no relax cascade, no chapter fallback, no unfiltered rescue.
            filtered = list(filtered_meta)
            filter_trace["after_strict"] = int(len(filtered))

            topic_to_pages = hierarchy.get("topic_to_pages", {}) if isinstance(hierarchy, dict) else {}
            allowed_pages = set()
            for tid in selected_ids:
                allowed_pages.update(topic_to_pages.get(str(tid), []) or [])
            if allowed_pages and "chapter_rescue" not in str(metadata_filter_mode):
                filtered = [r for r in filtered if str(r.get("page_id", "")).strip() in allowed_pages]
            metadata_filter_mode = f"{metadata_filter_mode}->strict_only"
            filter_trace["after_relax1"] = int(len(filtered))
            filter_trace["after_relax2"] = int(len(filtered))
        else:
            filtered = filter_by_topic_ids(filtered_meta, hierarchy, selected_ids)
            filter_trace["after_strict"] = int(len(filtered_meta))
            filter_trace["after_relax1"] = int(len(filtered))
            filter_trace["after_relax2"] = int(len(filtered))
        # If metadata mapping is too sparse for a specific topic, allow a
        # conservative rescue using exact topic_ids (weak label) within the same chapter.
        # Keep this disabled in strict mode to protect precision-first behavior.
        if (not bool(args.metadata_filter_strict)) and len(filtered) < 2 and any("." in str(tid) for tid in selected_ids):
            target_chapters = {str(tid).split(".", 1)[0].strip() for tid in selected_ids if str(tid).strip()}
            q_norm = str(query_profile.get("query_norm", "") or "").lower()
            circular_query = ("วงกลม" in q_norm) or ("circular" in q_norm)
            q_focus_terms = [
                str(x).strip().lower()
                for x in (
                    list(query_profile.get("operation_terms", []) or [])
                    + list(query_profile.get("structure_terms", []) or [])
                )
                if str(x).strip()
            ]
            q_focus_terms = [t for t in q_focus_terms if len(t) >= 3 and t not in {"การ", "ทำงาน", "ดำเนินการ", "โครงสร้าง"}]
            seen_ids = {str(r.get("id", "")).strip() for r in filtered}
            rescued = list(filtered)
            for rec in records:
                rid = str(rec.get("id", "")).strip()
                if not rid or rid in seen_ids:
                    continue
                rec_chap = str(rec.get("chapter_id", "") or "").strip()
                if target_chapters and rec_chap and rec_chap not in target_chapters:
                    continue
                rec_topic_ids = [str(x).strip() for x in (rec.get("topic_ids", []) or []) if str(x).strip()]
                if not any(tid in rec_topic_ids for tid in selected_ids):
                    continue
                if q_focus_terms:
                    rec_blob = " ".join(
                        [
                            str(rec.get("text", "")).lower(),
                            " ".join(str(x).lower() for x in (rec.get("figure_refs", []) or [])),
                            str(rec.get("section_title", "")).lower(),
                            str(rec.get("best_topic_title", "")).lower(),
                        ]
                    )
                    if not any(t in rec_blob for t in q_focus_terms):
                        continue
                    if circular_query and not any(x in rec_blob for x in ("วงกลม", "circular", "3.8", "3.9")):
                        continue
                # Keep rescue conservative: prioritize figure-bearing/region candidates.
                if not (bool(rec.get("figure_refs", [])) or str(rec.get("image_level", "")).strip().lower() == "region"):
                    continue
                rescued.append(rec)
                seen_ids.add(rid)
                if len(rescued) >= 8:
                    break
            filtered = rescued
        if _is_linkedlist_operation_query(query_profile, topic_id):
            filtered = expand_with_adjacent_topic_pages(
                records,
                filtered,
                max_page_gap=max(0, int(args.topic_adjacent_page_gap)),
                max_extra_pages=max(0, int(args.topic_adjacent_max_extra_pages)),
                forward_only=True,
            )
        topic_scope_ids = sorted(selected_ids)
        topic_filtered = len(filtered) < len(records)
        topic_filter_reason = f"metadata_prefilter_plus_hierarchy:{metadata_filter_mode}"
        records = filtered
        filter_trace["selected_mode"] = str(metadata_filter_mode)
        filter_trace["final_records"] = int(len(records))
    elif heuristic_topic_id and float(heuristic_topic_score) >= float(args.topic_threshold):
        filtered = filter_by_topic(records, hierarchy, heuristic_topic_id)
        topic_filtered = len(filtered) < len(records)
        topic_filter_reason = "heuristic_hierarchy_fallback"
        topic_id = heuristic_topic_id
        topic_score_val = heuristic_topic_score
        topic_scope_ids = [heuristic_topic_id]
        records = filtered
        filter_trace["after_strict"] = int(len(records))
        filter_trace["after_relax1"] = int(len(records))
        filter_trace["after_relax2"] = int(len(records))
        filter_trace["final_records"] = int(len(records))
        filter_trace["selected_mode"] = "heuristic_hierarchy_fallback"

    if args.require_structure:
        tmp = [r for r in records if bool(r.get("has_structure", False))]
        if query_profile.get("operation_intent"):
            # Operation-heavy chapters (e.g., linked-list workflows) can have
            # critical step figures without explicit structure labels.
            # Keep figure-bearing candidates as supplemental evidence.
            supplemental = [r for r in records if bool(r.get("figure_refs", []))]
            if supplemental:
                seen_ids = {str(r.get("id", "")).strip() for r in tmp}
                for r in supplemental:
                    rid = str(r.get("id", "")).strip()
                    if not rid or rid in seen_ids:
                        continue
                    tmp.append(r)
                    seen_ids.add(rid)
        if tmp:
            records = tmp
    filter_trace["final_records"] = int(len(records))

    lex_scores = np.array([lexical_score(q_tokens_expanded, r) for r in records], dtype=np.float64)
    reg_scores = np.array([float(r.get("region_score", 0.0) or 0.0) for r in records], dtype=np.float64)
    struct_prior = np.array([1.0 if bool(r.get("has_structure", False)) else 0.0 for r in records], dtype=np.float64)
    fig_scores = np.array([figure_ref_score(query_profile, r) for r in records], dtype=np.float64)
    op_scores = np.array([operation_flow_score(query_profile, r) for r in records], dtype=np.float64)
    struct_scores = np.array([structure_alignment_score(query_profile, r) for r in records], dtype=np.float64)
    region_quality_scores = np.array([region_quality_score(r) for r in records], dtype=np.float64)
    topic_scope_scores = np.array(
        [
            topic_scope_alignment_score(
                r,
                target_topic_id=topic_id,
                topic_scope_ids=set(topic_scope_ids or []),
                query_profile=query_profile,
            )
            for r in records
        ],
        dtype=np.float64,
    )
    hard_negative_penalties = np.array(
        [
            hard_negative_penalty(
                query_profile,
                r,
                rules=hard_negative_rules,
                penalty_max=max(0.0, float(args.hard_negative_penalty_max)),
            )
            for r in records
        ],
        dtype=np.float64,
    )

    col_scores = np.zeros(len(records), dtype=np.float64)
    col_status = "not_used"
    col_status_detail = "not_used"
    chosen_backend = "metadata"
    col_score_ready = False
    candidate_quality = {
        "candidate_count": 0,
        "image_usable_count": 0,
        "image_base64_coverage": 0.0,
        "dropped_no_image_count": 0,
    }
    splade_scores = np.zeros(len(records), dtype=np.float64)
    splade_status = "not_used"
    splade_score_ready = False
    bm25_scores = np.zeros(len(records), dtype=np.float64)
    bm25_status = "not_used"
    bm25_score_ready = False
    sparse_channel_name = str(args.sparse_strategy).strip().lower()
    if sparse_channel_name not in {"splade", "bm25"}:
        sparse_channel_name = "splade"

    if sparse_channel_name == "splade" and bool(args.use_splade):
        splade_hf_client = None
        if str(hf_token or "").strip():
            try:
                splade_hf_client = InferenceClient(
                    provider=str(args.splade_provider or "hf-inference"),
                    api_key=hf_token,
                )
            except TypeError:
                splade_hf_client = InferenceClient(api_key=hf_token)
        splade_scores, splade_status = run_splade_scores(
            args.query,
            records,
            model_name=str(args.splade_model).strip(),
            device=str(args.splade_device).strip() or "auto",
            max_length=max(32, int(args.splade_max_length)),
            batch_size=max(1, int(args.splade_batch_size)),
            mode=str(args.splade_mode).strip() or "auto",
            hf_client=splade_hf_client,
            hf_provider=str(args.splade_provider or "hf-inference"),
            hf_api_key=hf_token,
        )
        splade_score_ready = str(splade_status).startswith("ok")
    elif sparse_channel_name == "splade":
        splade_status = "splade_disabled"
    elif sparse_channel_name == "bm25":
        bm25_scores, bm25_status = run_bm25_scores(args.query, records)
        bm25_score_ready = str(bm25_status).startswith("ok")
        splade_status = "disabled_by_strategy_bm25"

    if args.backend in {"auto", "colpali_local"}:
        col_scores, col_status = run_local_colpali_scores(
            args.query,
            records,
            colpali_index_path=Path(args.colpali_index),
            model_name=args.colpali_model,
            device=args.device,
        )
        if col_status == "ok":
            chosen_backend = "colpali_local"
            col_score_ready = True
        elif args.backend == "colpali_local":
            chosen_backend = "metadata"
            col_score_ready = False

        # Auto fallback path: local ColPali -> endpoint (if configured) -> metadata.
        if (
            args.backend == "auto"
            and not col_score_ready
            and str(args.colpali_endpoint_url).strip()
        ):
            ep_scores, ep_status, ep_detail, ep_quality = run_endpoint_scores(
                args.query,
                records,
                endpoint_url=args.colpali_endpoint_url,
                api_key=hf_token,
                timeout_sec=max(10, int(args.endpoint_timeout_sec)),
                endpoint_max_retries=max(0, int(args.endpoint_max_retries)),
                retry_backoff_ms=max(0, int(args.endpoint_retry_backoff_ms)),
                local_cache_size=max(16, int(args.endpoint_local_cache_size)),
                endpoint_image_max_side=max(256, int(args.endpoint_image_max_side)),
                endpoint_jpeg_quality=max(40, min(95, int(args.endpoint_jpeg_quality))),
            )
            if ep_status == "ok":
                col_scores = ep_scores
                col_status = ep_status
                col_status_detail = ep_detail
                candidate_quality = ep_quality
                chosen_backend = "colpali_endpoint"
                col_score_ready = True
            else:
                col_status = ep_status
                col_status_detail = ep_detail
                candidate_quality = ep_quality

    if args.backend == "colpali_engine":
        col_scores, col_status = run_colpali_engine_scores(
            args.query,
            records,
            model_name=args.colpali_engine_model,
            device=args.device,
            allow_cpu=bool(args.colpali_engine_allow_cpu),
        )
        chosen_backend = "colpali_engine" if col_status == "ok" else "metadata"
        col_score_ready = col_status == "ok"

    if args.backend == "byaldi":
        col_scores, col_status = run_byaldi_scores(
            args.query,
            records,
            index_name=args.byaldi_index_name,
            index_root=args.byaldi_index_root or None,
            top_k=max(10, int(args.candidate_k)),
        )
        chosen_backend = "byaldi" if col_status == "ok" else "metadata"
        col_score_ready = col_status == "ok"

    if args.backend == "colpali_endpoint":
        col_scores, col_status, col_status_detail, candidate_quality = run_endpoint_scores(
            args.query,
            records,
            endpoint_url=args.colpali_endpoint_url,
            api_key=hf_token,
            timeout_sec=max(10, int(args.endpoint_timeout_sec)),
            endpoint_max_retries=max(0, int(args.endpoint_max_retries)),
            retry_backoff_ms=max(0, int(args.endpoint_retry_backoff_ms)),
            local_cache_size=max(16, int(args.endpoint_local_cache_size)),
            endpoint_image_max_side=max(256, int(args.endpoint_image_max_side)),
            endpoint_jpeg_quality=max(40, min(95, int(args.endpoint_jpeg_quality))),
        )
        chosen_backend = "colpali_endpoint" if col_status == "ok" else "metadata"
        col_score_ready = col_status == "ok"

    lex_n = normalize_scores(lex_scores)
    reg_n = normalize_scores(reg_scores)
    struct_prior_n = normalize_scores(struct_prior)
    struct_align_n = normalize_scores(struct_scores)
    topic_scope_n = normalize_scores(topic_scope_scores)
    fig_n = normalize_scores(fig_scores)
    op_n = normalize_scores(op_scores)
    region_quality_n = normalize_scores(region_quality_scores)
    col_n = normalize_scores(col_scores) if col_score_ready else np.zeros_like(lex_n)
    splade_n = normalize_scores(splade_scores) if splade_score_ready else np.zeros_like(lex_n)
    bm25_n = normalize_scores(bm25_scores) if bm25_score_ready else np.zeros_like(lex_n)
    sparse_score_ready = splade_score_ready or bm25_score_ready
    sparse_scores = splade_scores if splade_score_ready else bm25_scores
    sparse_n = splade_n if splade_score_ready else bm25_n
    sparse_status = splade_status if sparse_channel_name == "splade" else bm25_status

    # Dynamic channel weights (research-inspired late fusion for multimodal retrieval).
    if col_score_ready:
        base_weights = {
            "colpali": 0.50,
            "sparse": 0.16 if sparse_score_ready else 0.0,
            "lexical": 0.08,
            "region_prior": 0.08,
            "region_quality": 0.06,
            "structure_prior": 0.07,
            "structure_align": 0.05,
            "topic_scope": 0.12 if topic_scope_ids else 0.0,
            "figure_match": 0.06,
            "operation_match": 0.04,
        }
    else:
        base_weights = {
            "sparse": 0.42 if sparse_score_ready else 0.0,
            "lexical": 0.30 if sparse_score_ready else 0.40,
            "region_prior": 0.15,
            "region_quality": 0.10,
            "structure_prior": 0.14,
            "structure_align": 0.11,
            "topic_scope": 0.16 if topic_scope_ids else 0.0,
            "figure_match": 0.12,
            "operation_match": 0.08,
        }

    if query_profile.get("has_figure_ref"):
        base_weights["figure_match"] = max(base_weights.get("figure_match", 0.0), float(args.figure_boost_weight))
        base_weights["lexical"] = max(0.04, base_weights.get("lexical", 0.0) - 0.04)
    if query_profile.get("operation_intent"):
        base_weights["operation_match"] = max(base_weights.get("operation_match", 0.0), float(args.operation_boost_weight))
        base_weights["lexical"] = max(0.03, base_weights.get("lexical", 0.0) - 0.03)
    if args.require_structure:
        base_weights["structure_prior"] = base_weights.get("structure_prior", 0.0) + 0.03
        base_weights["structure_align"] = base_weights.get("structure_align", 0.0) + 0.02
    base_weights = normalize_weights(base_weights)

    base = np.zeros_like(lex_n)
    if col_score_ready:
        base += base_weights.get("colpali", 0.0) * col_n
    if sparse_score_ready:
        base += base_weights.get("sparse", 0.0) * sparse_n
    base += base_weights.get("lexical", 0.0) * lex_n
    base += base_weights.get("region_prior", 0.0) * reg_n
    base += base_weights.get("region_quality", 0.0) * region_quality_n
    base += base_weights.get("structure_prior", 0.0) * struct_prior_n
    base += base_weights.get("structure_align", 0.0) * struct_align_n
    base += base_weights.get("topic_scope", 0.0) * topic_scope_n
    base += base_weights.get("figure_match", 0.0) * fig_n
    base += base_weights.get("operation_match", 0.0) * op_n

    # Weighted RRF fusion as robust rank aggregation.
    rrf_channels = {
        "lexical": lex_scores,
        "region_prior": reg_scores,
        "region_quality": region_quality_scores,
        "structure_prior": struct_prior,
        "structure_align": struct_scores,
        "topic_scope": topic_scope_scores,
        "figure_match": fig_scores,
        "operation_match": op_scores,
    }
    if sparse_score_ready:
        rrf_channels[sparse_channel_name] = sparse_scores
    if col_score_ready:
        rrf_channels["colpali"] = col_scores
    rrf_weights = {
        k: (base_weights.get("sparse", 0.0) if k in {"splade", "bm25"} else base_weights.get(k, 0.0))
        for k in rrf_channels
    }
    rrf_raw = weighted_rrf(rrf_channels, rrf_k=max(1, int(args.rrf_k)), channel_weights=rrf_weights)
    rrf_n = normalize_scores(rrf_raw)

    if args.disable_figure_centric_rerank:
        pre_vlm = base
        pre_vlm_weights = {"base_only": 1.0}
    else:
        # Blend calibrated score + robust rank fusion.
        pre_vlm = (0.68 * base) + (0.32 * rrf_n)
        pre_vlm_weights = {"base": 0.68, "rrf": 0.32}

    # Noise suppression: penalize very tiny region hits unless they have strong semantic support.
    area_vals = np.array(
        [
            float((r.get("region_meta", {}) or {}).get("area_ratio", 0.0) or 0.0)
            if isinstance(r.get("region_meta", {}), dict) else 0.0
            for r in records
        ],
        dtype=np.float64,
    )
    tiny_mask = area_vals < 0.006
    has_semantic_support = (fig_scores > 0.0) | (op_scores > 0.0) | (struct_scores >= 0.5)
    tiny_penalty_mask = tiny_mask & (~has_semantic_support)
    if tiny_penalty_mask.any():
        pre_vlm = np.where(tiny_penalty_mask, pre_vlm * 0.82, pre_vlm)

    if hard_negative_penalties.size and float(np.max(hard_negative_penalties)) > 0.0:
        pre_vlm = np.maximum(0.0, pre_vlm - hard_negative_penalties)

    scoring_weights = {
        "channel_weights": {k: round(float(v), 4) for k, v in base_weights.items()},
        "pre_vlm_blend": pre_vlm_weights,
        "rrf_k": int(args.rrf_k),
        "tiny_region_penalty": {"enabled": True, "area_ratio_lt": 0.006, "multiplier": 0.82},
        "hard_negative": {
            "rules_loaded": int(len(hard_negative_rules)),
            "penalty_max": float(args.hard_negative_penalty_max),
            "penalized_candidates": int(np.sum(hard_negative_penalties > 0.0)),
        },
    }

    ranked_idx = np.argsort(pre_vlm)[::-1][: max(1, int(args.candidate_k))]
    candidates = []
    for rank, i in enumerate(ranked_idx.tolist(), start=1):
        r = dict(records[i])
        r["rank_base"] = rank
        r["base_score"] = round(float(base[i]), 6)
        r["rrf_score"] = round(float(rrf_n[i]), 6)
        r["pre_vlm_score"] = round(float(pre_vlm[i]), 6)
        r["lexical_score"] = round(float(lex_scores[i]), 6)
        r["splade_score"] = round(float(splade_scores[i]), 6)
        r["bm25_score"] = round(float(bm25_scores[i]), 6)
        r["sparse_score"] = round(float(sparse_scores[i]), 6) if sparse_scores.size else 0.0
        r["colpali_score"] = round(float(col_scores[i]), 6)
        r["figure_score"] = round(float(fig_scores[i]), 6)
        r["operation_score"] = round(float(op_scores[i]), 6)
        r["structure_score"] = round(float(struct_scores[i]), 6)
        r["topic_scope_score"] = round(float(topic_scope_scores[i]), 6)
        r["region_quality_score"] = round(float(region_quality_scores[i]), 6)
        r["hard_negative_penalty"] = round(float(hard_negative_penalties[i]), 6)
        candidates.append(r)

    reranked = candidates
    vlm_rerank_status = "not_used"

    if args.use_vlm_rerank and hf_client is not None and candidates:
        top_m = max(1, int(args.vlm_rerank_top_m))
        head = candidates[:top_m]
        tail = candidates[top_m:]
        head_vlm = vlm_rerank(
            hf_client=hf_client,
            model=args.vlm_model,
            query=args.query,
            candidates=head,
            require_structure=bool(args.require_structure),
        )

        for c in head_vlm:
            vs = float(c.get("vlm_score", 0.0) or 0.0)
            bs = float(c.get("pre_vlm_score", c.get("base_score", 0.0)) or 0.0)
            c["final_score"] = round((0.63 * bs) + (0.37 * vs), 6)

        for c in tail:
            c["vlm_score"] = 0.0
            c["vlm_reason"] = "not_in_vlm_top_m"
            c["final_score"] = round(float(c.get("pre_vlm_score", c.get("base_score", 0.0))), 6)

        reranked = sorted(head_vlm + tail, key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
        vlm_rerank_status = "ok"
    elif args.use_vlm_rerank and hf_client is None:
        vlm_rerank_status = "missing_hf_token"

    top_k = max(1, int(args.top_k))
    if query_profile.get("has_figure_ref"):
        q_refs = set(query_profile.get("figure_refs", []))
        head = []
        tail = []
        for c in reranked:
            rec_refs = set()
            for fr in c.get("figure_refs", []) or []:
                rec_refs.update(extract_figure_refs(str(fr)))
            if q_refs.intersection(rec_refs):
                head.append(c)
            else:
                tail.append(c)
        reranked = head + tail
    if args.require_structure:
        head = [c for c in reranked if bool(c.get("has_structure", False))]
        tail = [c for c in reranked if not bool(c.get("has_structure", False))]
        if head:
            reranked = head + tail

    coverage_applied = False
    if (
        not args.disable_coverage_rerank
        and reranked
        and (query_profile.get("operation_intent") or query_profile.get("has_figure_ref"))
    ):
        reranked = coverage_aware_order(
            reranked,
            query_profile=query_profile,
            top_k=max(1, int(args.top_k)),
            novelty_weight=max(0.0, min(1.0, float(args.coverage_novelty_weight))),
            figure_weight=max(0.0, float(args.coverage_figure_weight)),
            page_weight=max(0.0, float(args.coverage_page_weight)),
            step_target=max(1, int(args.coverage_step_target)),
        )
        coverage_applied = True
    hits = reranked[:top_k]

    grounding = []
    grounding_status = "not_used"
    if args.use_visual_grounding and hf_client is not None and hits:
        ground_n = max(1, int(args.ground_top_n))
        for c in hits[:ground_n]:
            g = visual_ground(
                hf_client=hf_client,
                model=args.vlm_model,
                query=args.query,
                candidate=c,
                ensemble_runs=max(1, int(args.grounding_ensemble_runs)),
                consensus_min_votes=max(1, int(args.grounding_consensus_min_votes)),
                temperature=float(args.grounding_temperature),
                top_p=float(args.grounding_top_p),
                query_profile=query_profile,
            )
            grounding.append({"id": c.get("id"), "page_id": c.get("page_id"), "grounding": g})
        grounding_status = "ok"
    elif args.use_visual_grounding and hf_client is None:
        grounding_status = "missing_hf_token"

    hit_rows = [
        {
            "rank": i + 1,
            "id": h.get("id"),
            "page_id": h.get("page_id"),
            "source": h.get("source"),
            "page": h.get("page"),
            "image_path": h.get("image_path"),
            "image_level": h.get("image_level"),
            "region_meta": h.get("region_meta", {}),
            "structure_labels": h.get("structure_labels", []),
            "tags": h.get("tags", []),
            "figure_refs": h.get("figure_refs", []),
            "chapter_id": h.get("chapter_id", ""),
            "chapter_title": h.get("chapter_title", ""),
            "section_id": h.get("section_id", ""),
            "section_title": h.get("section_title", ""),
            "best_topic_id": h.get("best_topic_id", ""),
            "best_topic_title": h.get("best_topic_title", ""),
            "base_score": h.get("base_score"),
            "rrf_score": h.get("rrf_score", 0.0),
            "pre_vlm_score": h.get("pre_vlm_score", h.get("base_score")),
            "vlm_score": h.get("vlm_score", 0.0),
            "final_score": h.get("final_score", h.get("pre_vlm_score", h.get("base_score"))),
            "figure_score": h.get("figure_score", 0.0),
            "operation_score": h.get("operation_score", 0.0),
            "structure_score": h.get("structure_score", 0.0),
            "topic_scope_score": h.get("topic_scope_score", 0.0),
            "region_quality_score": h.get("region_quality_score", 0.0),
            "vlm_reason": h.get("vlm_reason", ""),
            "preview": str(h.get("text", ""))[:280],
        }
        for i, h in enumerate(hits)
    ]

    task_metrics = compute_task_metrics(
        query_profile,
        hit_rows,
        grounding,
        step_target=max(1, int(args.coverage_step_target)),
    )

    if col_status_detail == "not_used":
        if str(col_status).strip().lower() == "ok":
            col_status_detail = "ok"
        elif str(col_status).startswith("endpoint_call_failed"):
            col_status_detail = "endpoint_call_failed"
        elif str(col_status).strip().lower() == "endpoint_schema_invalid":
            col_status_detail = "endpoint_schema_invalid"
        elif str(col_status).strip().lower() == "endpoint_no_scores":
            col_status_detail = "endpoint_empty_scores"

    result = {
        "query": args.query,
        "backend": chosen_backend,
        "sparse_strategy": sparse_channel_name,
        "sparse_status": sparse_status,
        "sparse_score_ready": sparse_score_ready,
        "splade_status": splade_status,
        "splade_score_ready": splade_score_ready,
        "bm25_status": bm25_status,
        "bm25_score_ready": bm25_score_ready,
        "colpali_status": col_status,
        "colpali_status_detail": col_status_detail,
        "colpali_score_ready": col_score_ready,
        "vlm_rerank_status": vlm_rerank_status,
        "grounding_status": grounding_status,
        "candidate_quality": candidate_quality,
        "filter_trace": filter_trace,
        "endpoint_runtime": {
            "timeout_sec": int(args.endpoint_timeout_sec),
            "max_retries": int(args.endpoint_max_retries),
            "retry_backoff_ms": int(args.endpoint_retry_backoff_ms),
            "local_cache_size": int(args.endpoint_local_cache_size),
            "image_max_side": int(args.endpoint_image_max_side),
            "jpeg_quality": int(args.endpoint_jpeg_quality),
        },
        "splade_runtime": {
            "enabled": bool(args.use_splade) and sparse_channel_name == "splade",
            "mode": str(args.splade_mode),
            "provider": str(args.splade_provider),
            "model": str(args.splade_model),
            "device": str(args.splade_device),
            "max_length": int(args.splade_max_length),
            "batch_size": int(args.splade_batch_size),
        },
        "bm25_runtime": {
            "enabled": sparse_channel_name == "bm25",
            "status": bm25_status,
        },
        "scoring_weights": scoring_weights,
        "coverage_rerank": {
            "applied": coverage_applied,
            "novelty_weight": float(args.coverage_novelty_weight),
            "figure_weight": float(args.coverage_figure_weight),
            "page_weight": float(args.coverage_page_weight),
            "step_target": int(args.coverage_step_target),
        },
        "grounding_config": {
            "enabled": bool(args.use_visual_grounding),
            "top_n": int(args.ground_top_n),
            "ensemble_runs": int(args.grounding_ensemble_runs),
            "consensus_min_votes": int(args.grounding_consensus_min_votes),
            "temperature": float(args.grounding_temperature),
            "top_p": float(args.grounding_top_p),
        },
        "hard_negative_config": {
            "rules_path": str(args.hard_negative_rules),
            "rules_loaded": int(len(hard_negative_rules)),
            "penalty_max": float(args.hard_negative_penalty_max),
        },
        "topic_prediction": {
            "topic_id": topic_id,
            "topic_score": round(float(topic_score_val), 6),
            "topic_filtered": topic_filtered,
            "topic_threshold": float(args.topic_threshold),
            "topic_scope_ids": topic_scope_ids,
            "topic_source": topic_source,
            "topic_filter_reason": topic_filter_reason,
            "metadata_prefilter_enabled": bool(topic_scope_ids),
            "metadata_filter_strict": bool(args.metadata_filter_strict),
            "heuristic_topic_id": heuristic_topic_id,
            "heuristic_topic_score": round(float(heuristic_topic_score), 6),
        },
        "intent_classification": {
            "use_intent_llm": bool(args.use_intent_llm),
            "intent_model": str(args.intent_model),
            "intent_llm_status": intent_llm_status,
            "intent_llm": intent_llm_obj if isinstance(intent_llm_obj, dict) else {},
            "ranked_topics": [
                {
                    "topic_id": str(x.get("topic_id", "")),
                    "title": str(x.get("title", "")),
                    "score": round(float(x.get("score", 0.0) or 0.0), 6),
                }
                for x in ranked_topics[: max(1, int(args.intent_candidate_top))]
            ],
        },
        "query_expansion": {
            "use_query_expansion_llm": bool(args.use_query_expansion_llm),
            "query_expansion_model": str(args.query_expansion_model),
            "query_expansion_status": qe_status,
            "query_expansion_terms": qe_terms,
            "query_expansion_max_terms": int(args.query_expansion_max_terms),
        },
        "hierarchy_quality": {
            "garbled_detected": bool(hierarchy_garbled),
            "mojibake_ratio": round(float(mojibake_ratio), 4),
            "topic_filter_enabled": bool(not hierarchy_garbled),
        },
        "query_profile": {
            "has_figure_ref": bool(query_profile.get("has_figure_ref")),
            "figure_refs": query_profile.get("figure_refs", []),
            "operation_intent": bool(query_profile.get("operation_intent")),
            "structure_intent": bool(query_profile.get("structure_intent")),
            "operation_terms": query_profile.get("operation_terms", []),
            "structure_terms": query_profile.get("structure_terms", []),
            "intent_type": query_profile.get("intent_type", ""),
            "expanded_query_terms_count": len(q_tokens_expanded),
            "llm_expansion_status": qe_status,
            "llm_expansion_terms": qe_terms,
        },
        "task_metrics": task_metrics,
        "count": len(hits),
        "hits": hit_rows,
        "grounding": grounding,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_output={out}")


if __name__ == "__main__":
    main()
