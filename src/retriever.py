"""
retriever.py - Hybrid Search Engine with graceful fallbacks.

Pipeline priority:
1) Dense retrieval (FAISS) when available
2) Sparse retrieval (BM25)
3) Optional reranker when enabled and loadable
"""

import os
import pickle
import logging
import time
import re
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from pythainlp.tokenize import word_tokenize
try:
    from .index_integrity import INDEX_MANIFEST_NAME, verify_index_manifest
except ImportError:
    from index_integrity import INDEX_MANIFEST_NAME, verify_index_manifest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = BASE_DIR / "indexes"

EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_CANDIDATES = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "BAAI/bge-reranker-v2-m3",
]
LOGGER = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "1") -> bool:
    """Parse env bool-like values: 1/true/yes/on."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default_values: list[str]) -> list[str]:
    """Parse comma-separated env string into a non-empty model candidate list."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default_values)
    values = [v.strip() for v in raw.split(",")]
    values = [v for v in values if v]
    return values or list(default_values)


def thai_tokenize(text: str) -> list[str]:
    """Tokenize Thai text using PyThaiNLP (newmm engine)."""
    return word_tokenize(text, engine="newmm")


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[\[\]\(\)\{\},.:;!?/\\|\"'`~@#$%^&*+=<>]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


# ---------------------------------------------------------------------------
# RAG System
# ---------------------------------------------------------------------------
class RAGSystem:
    """Loads pre-built indexes and exposes a hybrid retrieve() method."""

    def __init__(self) -> None:
        self.vectorstore = None
        self.dense_error = None
        self.dense_enabled = _env_flag("DENSE_RETRIEVAL_ENABLED", "1")

        self.reranker = None
        self.reranker_model = None
        self.reranker_load_seconds = None
        self.reranker_error = None
        self.reranker_enabled = _env_flag("RERANKER_ENABLED", "1")
        self.reranker_candidates = _env_list("RERANKER_MODEL_CANDIDATES", DEFAULT_RERANKER_CANDIDATES)
        self.reranker_healthcheck_enabled = _env_flag("RERANKER_HEALTHCHECK_ENABLED", "1")
        self.reranker_startup_budget_sec = float(os.getenv("RERANKER_MAX_STARTUP_SECONDS", "45"))
        self.rrf_k = int(os.getenv("RRF_K", "60"))
        self.weight_dense = float(os.getenv("FUSION_WEIGHT_DENSE", "0.5"))
        self.weight_sparse = float(os.getenv("FUSION_WEIGHT_SPARSE", "0.5"))
        self.default_top_k = int(os.getenv("RETRIEVE_TOP_K", "20"))
        self.default_top_n = int(os.getenv("RETRIEVE_TOP_N", "8"))
        self.index_dir = Path(os.getenv("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR))).resolve()
        self.index_verify_enabled = _env_flag("INDEX_VERIFY_CHECKSUM", "1")
        self.index_verify_strict = _env_flag("INDEX_VERIFY_STRICT", "1")
        self.index_manifest = self.index_dir / INDEX_MANIFEST_NAME
        self.index_integrity_ok = None
        self.index_integrity_error = None
        self.topic_hierarchy_file = Path(
            os.getenv(
                "TOPIC_HIERARCHY_FILE",
                str(BASE_DIR / "indexes" / "hierarchical" / "topic_hierarchy.json"),
            )
        ).resolve()
        self.section_page_override_file = Path(
            os.getenv(
                "SECTION_PAGE_OVERRIDE_FILE",
                str(BASE_DIR / "indexes" / "hierarchical" / "section_page_overrides.json"),
            )
        ).resolve()
        self.topic_id_to_title: dict[str, str] = {}
        self.topic_to_pages: dict[str, list[str]] = {}
        self.page_to_best_topic: dict[str, str] = {}
        self.page_to_top_topics: dict[str, list[str]] = {}
        self._section_page_allowlist_cache: dict[str, set[str]] = {}

        self._verify_index_integrity()
        self._load_faiss()
        self._load_bm25()
        self._load_topic_hierarchy()
        self._load_reranker()

    # -- loaders -----------------------------------------------------------

    def _verify_index_integrity(self) -> None:
        """Validate SHA-256 checksums before loading index files."""
        if not self.index_verify_enabled:
            self.index_integrity_ok = None
            self.index_integrity_error = "Checksum verification disabled by env (INDEX_VERIFY_CHECKSUM=0)."
            LOGGER.warning(self.index_integrity_error)
            return

        try:
            ok, errors = verify_index_manifest(self.index_dir, self.index_manifest)
            if not ok:
                self.index_integrity_ok = False
                self.index_integrity_error = ", ".join(errors)
                message = f"Index integrity verification failed: {self.index_integrity_error}"
                if self.index_verify_strict:
                    raise ValueError(message)
                LOGGER.error(message)
            else:
                self.index_integrity_ok = True
                self.index_integrity_error = None
        except Exception as exc:
            self.index_integrity_ok = False
            self.index_integrity_error = str(exc) or exc.__class__.__name__
            if self.index_verify_strict:
                raise
            LOGGER.exception("Index integrity verification failed, continuing due to non-strict mode.")

    def _load_faiss(self) -> None:
        """Load FAISS vector store; degrade to sparse-only if unavailable."""
        if not self.dense_enabled:
            self.dense_error = "Dense retrieval disabled by env (DENSE_RETRIEVAL_ENABLED=0)."
            LOGGER.warning(self.dense_error)
            return

        faiss_path = self.index_dir / "faiss_index"
        if not faiss_path.exists():
            self.dense_error = "FAISS index not found."
            LOGGER.warning(self.dense_error)
            return

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self.vectorstore = FAISS.load_local(
                str(faiss_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            self.dense_error = None
        except Exception as exc:
            self.vectorstore = None
            self.dense_error = str(exc) or exc.__class__.__name__
            LOGGER.exception("Dense retrieval load failed. Falling back to BM25-only retrieval.")

    def _load_bm25(self) -> None:
        """Load BM25 index and chunk payload (required baseline)."""
        bm25_path = self.index_dir / "bm25_index.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError("BM25 index not found. Run ingest.py first.")

        with open(bm25_path, "rb") as f:
            payload = pickle.load(f)

        self.bm25: BM25Okapi = payload["bm25"]
        self.bm25_chunks: list[Document] = payload["chunks"]

    def _load_topic_hierarchy(self) -> None:
        """Load page/topic mapping used to enrich chunk metadata at retrieval time."""
        if not self.topic_hierarchy_file.exists():
            return
        try:
            payload = json.loads(self.topic_hierarchy_file.read_text(encoding="utf-8"))
            topics = payload.get("topics", [])
            self.topic_id_to_title = {
                str(t.get("topic_id", "")).strip(): str(t.get("title", "")).strip()
                for t in topics
                if isinstance(t, dict) and str(t.get("topic_id", "")).strip()
            }
            self.topic_to_pages = {}
            for tid, pages in (payload.get("topic_to_pages", {}) or {}).items():
                topic_id = str(tid).strip()
                if not topic_id:
                    continue
                if isinstance(pages, list):
                    self.topic_to_pages[topic_id] = [str(p).strip() for p in pages if str(p).strip()]
                else:
                    self.topic_to_pages[topic_id] = []
            self.page_to_best_topic = {
                str(k).strip(): str(v).strip()
                for k, v in (payload.get("page_to_best_topic", {}) or {}).items()
                if str(k).strip()
            }
            self.page_to_top_topics = {}
            for k, vals in (payload.get("page_to_top_topics", {}) or {}).items():
                key = str(k).strip()
                if not key:
                    continue
                arr = []
                if isinstance(vals, list):
                    arr = [str(x).strip() for x in vals if str(x).strip()]
                self.page_to_top_topics[key] = arr
            self._apply_section_page_overrides()
            self._section_page_allowlist_cache = {}
        except Exception:
            LOGGER.exception("Failed to load topic hierarchy metadata: %s", self.topic_hierarchy_file)

    def _apply_section_page_overrides(self) -> None:
        """
        Deterministic section->pages overrides (list_hitachi-curated).
        Applied after hierarchy load to stabilize strict section filtering.
        """
        path = self.section_page_override_file
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_map = payload.get("topic_to_pages", payload if isinstance(payload, dict) else {})
            if not isinstance(raw_map, dict):
                return
            for sid_raw, pages_raw in raw_map.items():
                sid = str(sid_raw).strip()
                if not sid:
                    continue
                pages = []
                if isinstance(pages_raw, list):
                    pages = [str(p).strip() for p in pages_raw if str(p).strip()]
                if not pages:
                    continue
                # Replace mapping deterministically for strict mode.
                self.topic_to_pages[sid] = pages
                # Enrich top-topic map so section match stays consistent.
                for page_key in pages:
                    cur = [str(x).strip() for x in (self.page_to_top_topics.get(page_key, []) or []) if str(x).strip()]
                    if sid not in cur:
                        cur = [sid] + [x for x in cur if x != sid]
                    self.page_to_top_topics[page_key] = cur
            LOGGER.info("Applied section page overrides from %s", path)
            self._section_page_allowlist_cache = {}
        except Exception:
            LOGGER.exception("Failed to apply section page overrides: %s", path)

    @staticmethod
    def _split_page_ref(page_ref: str) -> tuple[str, int]:
        ref = str(page_ref or "").strip()
        if not ref:
            return "", 10**9
        if ":" in ref:
            source, page = ref.rsplit(":", 1)
            try:
                return source.strip(), int(page)
            except Exception:
                return source.strip(), 10**9
        try:
            return "", int(ref)
        except Exception:
            return "", 10**9

    def _sort_page_refs(self, refs: set[str] | list[str]) -> list[str]:
        uniq = {str(x).strip() for x in refs if str(x).strip()}
        return sorted(uniq, key=lambda x: self._split_page_ref(x))

    def _derive_pages_from_chunk_metadata(self, section_id: str) -> set[str]:
        sid = str(section_id or "").strip()
        if not sid:
            return set()
        sid_norm = _normalize_text(sid)
        explicit_hits: set[str] = set()
        weak_hits: set[str] = set()
        for d in self.bm25_chunks:
            meta = getattr(d, "metadata", {}) or {}
            source = str(meta.get("source", "")).strip()
            page = str(meta.get("page", "")).strip()
            if not source or not page:
                continue
            page_ref = f"{source}:{page}"

            section = str(meta.get("section", "")).strip()
            section_norm = _normalize_text(section)
            if section_norm and (
                section_norm.startswith(sid_norm + " ")
                or section_norm.startswith(sid_norm + ".")
                or section_norm == sid_norm
            ):
                explicit_hits.add(page_ref)
                continue

            best_tid = _normalize_text(str(meta.get("best_topic_id", "")).strip())
            if best_tid == sid_norm:
                weak_hits.add(page_ref)
                continue

            top_ids = [
                _normalize_text(str(x).strip())
                for x in (meta.get("top_topic_ids", []) or [])
                if str(x).strip()
            ]
            if sid_norm in set(top_ids):
                weak_hits.add(page_ref)
        return explicit_hits if explicit_hits else weak_hits

    def _enrich_doc_with_hierarchy_topics(self, doc: Document) -> None:
        meta = getattr(doc, "metadata", {}) or {}
        source = str(meta.get("source", "")).strip()
        page = str(meta.get("page", "")).strip()
        if not source or not page:
            return
        page_key = f"{source}:{page}"

        best_topic = self.page_to_best_topic.get(page_key, "")
        top_topics = self.page_to_top_topics.get(page_key, [])
        if not best_topic and not top_topics:
            return

        topic_tags = [str(x).strip() for x in meta.get("topic_tags", []) if str(x).strip()]
        existing = set(topic_tags)

        if best_topic and best_topic not in existing:
            topic_tags.append(best_topic)
            existing.add(best_topic)
        if best_topic:
            best_title = self.topic_id_to_title.get(best_topic, "")
            if best_title and best_title not in existing:
                topic_tags.append(best_title)
                existing.add(best_title)

        for tid in top_topics:
            if tid and tid not in existing:
                topic_tags.append(tid)
                existing.add(tid)
            title = self.topic_id_to_title.get(tid, "")
            if title and title not in existing:
                topic_tags.append(title)
                existing.add(title)

        meta["topic_tags"] = topic_tags
        if best_topic and not str(meta.get("best_topic_id", "")).strip():
            meta["best_topic_id"] = best_topic
        if top_topics:
            meta["top_topic_ids"] = list(top_topics)
        doc.metadata = meta

    def _load_reranker(self) -> None:
        """Initialise optional cross-encoder reranker (tries candidates in order)."""
        if not self.reranker_enabled:
            LOGGER.warning("Reranker disabled by env (RERANKER_ENABLED=0).")
            return

        load_errors = []
        startup_begin = time.perf_counter()
        for model_name in self.reranker_candidates:
            elapsed = time.perf_counter() - startup_begin
            if elapsed >= self.reranker_startup_budget_sec:
                load_errors.append(
                    f"startup_budget_exceeded:{elapsed:.1f}s/{self.reranker_startup_budget_sec:.1f}s"
                )
                break
            try:
                candidate_start = time.perf_counter()
                self.reranker = CrossEncoder(model_name)

                if self.reranker_healthcheck_enabled:
                    # Quick health-check to verify inference path is usable.
                    _ = self.reranker.predict([("healthcheck query", "healthcheck passage")])

                self.reranker_model = model_name
                self.reranker_load_seconds = round(time.perf_counter() - candidate_start, 3)
                self.reranker_error = None
                LOGGER.info("Reranker loaded: %s (%.3fs)", model_name, self.reranker_load_seconds)
                return
            except Exception as exc:
                err = str(exc) or exc.__class__.__name__
                load_errors.append(f"{model_name}: {err}")
                LOGGER.warning("Reranker model load failed for %s", model_name)

        self.reranker = None
        self.reranker_model = None
        self.reranker_load_seconds = None
        self.reranker_error = " | ".join(load_errors) if load_errors else "No reranker candidates available."
        LOGGER.error("Reranker load failed for all candidates. Falling back without reranking.")

    # -- search ------------------------------------------------------------

    def _dense_search(self, query: str, top_k: int) -> list[Document]:
        """Retrieve top-K documents from FAISS (dense / vector search)."""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=top_k)

    def _sparse_search(self, query: str, top_k: int) -> list[Document]:
        """Retrieve top-K documents from BM25 (sparse / keyword search)."""
        tokenized_query = thai_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[Document] = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(self.bm25_chunks[idx])
        return results

    def _ensemble(
        self,
        dense_docs: list[Document],
        sparse_docs: list[Document],
        weight_dense: float | None = None,
        weight_sparse: float | None = None,
        rrf_k: int | None = None,
    ) -> list[Document]:
        """
        Canonical weighted Reciprocal Rank Fusion (RRF):
        score(d) = sum_i w_i / (k + rank_i(d))
        """
        wd = self.weight_dense if weight_dense is None else float(weight_dense)
        ws = self.weight_sparse if weight_sparse is None else float(weight_sparse)
        rrf_const = self.rrf_k if rrf_k is None else int(rrf_k)

        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        dense_rank: dict[str, int] = {}
        sparse_rank: dict[str, int] = {}

        def _doc_key(doc: Document) -> str:
            meta = getattr(doc, "metadata", {}) or {}
            source = str(meta.get("source", "")).strip()
            page = str(meta.get("page", "")).strip()
            chunk_id = str(meta.get("chunk_id", "")).strip()
            if source or page or chunk_id:
                return f"{source}|{page}|{chunk_id}|{doc.page_content[:80]}"
            return doc.page_content

        for rank, doc in enumerate(dense_docs, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + wd / (rrf_const + rank)
            doc_map.setdefault(key, doc)
            dense_rank[key] = rank

        for rank, doc in enumerate(sparse_docs, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + ws / (rrf_const + rank)
            doc_map.setdefault(key, doc)
            sparse_rank[key] = rank

        sorted_keys = sorted(scores, key=scores.get, reverse=True)
        results: list[Document] = []
        for key in sorted_keys:
            doc = doc_map[key]
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["rrf_score"] = round(float(scores[key]), 8)
            if key in dense_rank:
                doc.metadata["dense_rank"] = int(dense_rank[key])
            if key in sparse_rank:
                doc.metadata["sparse_rank"] = int(sparse_rank[key])
            results.append(doc)
        return results

    def _rerank(self, query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
        """Re-score candidates with cross-encoder when available."""
        if not docs:
            return []
        if self.reranker is None:
            return docs[:top_n]

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]

    def _topic_keywords(self, topic_hint: str) -> list[str]:
        text = _normalize_text(topic_hint)
        if not text:
            return []
        stop = {"topic", "other", "the", "of", "and"}
        kws = []
        for token in thai_tokenize(text):
            token = token.strip()
            if not token or token in stop:
                continue
            if len(token) < 2:
                continue
            kws.append(token)
        return kws

    def _stable_doc_key(self, doc: Document) -> str:
        meta = getattr(doc, "metadata", {}) or {}
        source = str(meta.get("source", "")).strip()
        page = str(meta.get("page", "")).strip()
        chunk_id = str(meta.get("chunk_id", "")).strip()
        if source or page or chunk_id:
            return f"{source}|{page}|{chunk_id}"
        return (doc.page_content or "")[:120]

    def _doc_page_key(self, doc: Document) -> str:
        meta = getattr(doc, "metadata", {}) or {}
        source = str(meta.get("source", "")).strip()
        page = str(meta.get("page", "")).strip()
        if source and page:
            return f"{source}:{page}"
        return ""

    def _doc_page_num(self, doc: Document) -> str:
        meta = getattr(doc, "metadata", {}) or {}
        return str(meta.get("page", "")).strip()

    def _doc_matches_page_allowlist(self, doc: Document, allowlist: set[str]) -> bool:
        if not allowlist:
            return False
        pkey = self._doc_page_key(doc)
        pnum = self._doc_page_num(doc)
        if pkey and pkey in allowlist:
            return True
        if pnum and pnum in allowlist:
            return True
        if pnum:
            # Allow entries like "source:page" by page suffix.
            for item in allowlist:
                if ":" in item and item.split(":")[-1] == pnum:
                    return True
        return False

    def _select_with_page_diversity(
        self,
        primary: list[Document],
        fallback: list[Document],
        *,
        limit: int,
        min_unique_pages: int,
    ) -> list[Document]:
        """
        Keep ranking order but guarantee minimum page diversity when possible.
        """
        k = max(1, int(limit))
        need_pages = max(1, int(min_unique_pages))

        selected: list[Document] = []
        seen_doc_keys: set[str] = set()
        seen_pages: set[str] = set()

        def add_if_new(doc: Document, *, prefer_new_page: bool = False) -> bool:
            if len(selected) >= k:
                return False
            dkey = self._stable_doc_key(doc)
            if dkey in seen_doc_keys:
                return False
            pkey = self._doc_page_key(doc)
            if prefer_new_page and pkey and pkey in seen_pages:
                return False
            selected.append(doc)
            seen_doc_keys.add(dkey)
            if pkey:
                seen_pages.add(pkey)
            return True

        # Pass 1: pick new pages from primary list.
        for d in primary:
            if len(selected) >= k:
                break
            add_if_new(d, prefer_new_page=True)
            if len(seen_pages) >= need_pages and len(selected) >= min(k, need_pages):
                break

        # Pass 2: if still under target, add new pages from fallback list.
        if len(seen_pages) < need_pages:
            for d in fallback:
                if len(selected) >= k:
                    break
                add_if_new(d, prefer_new_page=True)
                if len(seen_pages) >= need_pages:
                    break

        # Pass 3: fill remaining by rank (primary first, then fallback).
        for d in primary:
            if len(selected) >= k:
                break
            add_if_new(d, prefer_new_page=False)
        for d in fallback:
            if len(selected) >= k:
                break
            add_if_new(d, prefer_new_page=False)

        return selected[:k]

    def _section_page_allowlist(self, section_id: str) -> set[str]:
        sid = str(section_id or "").strip()
        if not sid:
            return set()
        if sid in self._section_page_allowlist_cache:
            return set(self._section_page_allowlist_cache[sid])

        mapped_pages = {
            str(p).strip()
            for p in (self.topic_to_pages.get(sid, []) or [])
            if str(p).strip()
        }
        derived_pages = self._derive_pages_from_chunk_metadata(sid)

        final_pages: set[str]
        if mapped_pages:
            # Trust deterministic mapping first (hierarchy/overrides).
            final_pages = set(mapped_pages)
            # If mapping is very narrow (single page), allow close neighbour from derived pages.
            if len(final_pages) <= 1 and derived_pages:
                expanded = set(final_pages)
                base = [self._split_page_ref(x) for x in final_pages]
                for ref in derived_pages:
                    src, pg = self._split_page_ref(ref)
                    for bsrc, bpg in base:
                        if src == bsrc and abs(pg - bpg) <= 1:
                            expanded.add(ref)
                final_pages = expanded
        else:
            final_pages = set(derived_pages)

        ordered = self._sort_page_refs(final_pages)
        # Keep allowlist tight for strict section filtering.
        if len(ordered) > 8:
            ordered = ordered[:8]
        result = set(ordered)
        self._section_page_allowlist_cache[sid] = set(result)
        return result

    def _section_page_keys(self, section_id: str) -> set[str]:
        sid = str(section_id or "").strip()
        if not sid:
            return set()
        out = set()
        for page_key, best in self.page_to_best_topic.items():
            if str(best).strip() == sid:
                out.add(page_key)
        for page_key, top_ids in self.page_to_top_topics.items():
            vals = [str(x).strip() for x in (top_ids or []) if str(x).strip()]
            if sid in vals:
                out.add(page_key)
        return out

    def _supplement_section_docs(
        self,
        section_id: str,
        query_text: str | None,
        query_profile: dict | None,
        limit: int = 12,
    ) -> list[Document]:
        page_keys = self._section_page_keys(section_id)
        page_allowlist = self._section_page_allowlist(section_id)
        if not page_keys and not page_allowlist:
            return []

        q_tokens = [t for t in thai_tokenize(query_text or "") if str(t).strip()]
        scored: list[tuple[float, Document]] = []
        for d in self.bm25_chunks:
            self._enrich_doc_with_hierarchy_topics(d)
            meta = getattr(d, "metadata", {}) or {}
            source = str(meta.get("source", "")).strip()
            page = str(meta.get("page", "")).strip()
            if not source or not page:
                continue
            if page_allowlist:
                # Deterministic override mode: honor explicit page allowlist first.
                page_match = self._doc_matches_page_allowlist(d, page_allowlist)
            else:
                page_match = f"{source}:{page}" in page_keys
            if not page_match:
                continue
            if query_profile and query_profile.get("enabled", False):
                if not self._doc_query_anchor_match(d, query_profile):
                    continue

            haystack_parts = [
                str(meta.get("chapter", "")),
                str(meta.get("section", "")),
                str(meta.get("best_topic_id", "")),
                " ".join(str(x) for x in meta.get("top_topic_ids", []) if str(x).strip()),
                str(meta.get("topic_path", "")),
                " ".join(str(x) for x in meta.get("topic_tags", []) if str(x).strip()),
                (d.page_content or "")[:900],
            ]
            hay = _normalize_text(" ".join(haystack_parts))

            score = 0.0
            if str(meta.get("best_topic_id", "")).strip() == str(section_id).strip():
                score += 3.0
            top_ids = [str(x).strip() for x in meta.get("top_topic_ids", []) if str(x).strip()]
            if str(section_id).strip() in top_ids:
                score += 2.0
            if q_tokens:
                score += sum(1 for t in q_tokens if _normalize_text(str(t)) in hay) * 0.25
            scored.append((score, d))

        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        seen = set()
        seen_pages = set()

        # Pass 1: ensure page diversity first (one doc per page).
        for _, doc in scored:
            pkey = self._doc_page_key(doc)
            if pkey and pkey in seen_pages:
                continue
            key = self._stable_doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            out.append(doc)
            if pkey:
                seen_pages.add(pkey)
            if len(out) >= max(1, int(limit)):
                break

        # Pass 2: fill remaining by score order.
        for _, doc in scored:
            if len(out) >= max(1, int(limit)):
                break
            key = self._stable_doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            out.append(doc)
        return out

    def _doc_has_structure(self, doc: Document) -> bool:
        meta = getattr(doc, "metadata", {}) or {}
        if bool(meta.get("has_structure", False)):
            return True
        text = (doc.page_content or "").lower()
        if "[structure:" in text:
            return True
        if "->" in text or "→" in text:
            return True
        hints = ("binary tree", "linked list", "graph", "\u0e44\u0e1a\u0e19\u0e32\u0e23\u0e35\u0e17\u0e23\u0e35", "\u0e25\u0e34\u0e07\u0e01\u0e4c\u0e25\u0e34\u0e2a\u0e15\u0e4c", "\u0e42\u0e04\u0e23\u0e07\u0e2a\u0e23\u0e49\u0e32\u0e07", "\u0e41\u0e1c\u0e19\u0e20\u0e32\u0e1e")
        return any(h in text for h in hints)

    def _build_query_anchor_profile(self, query_text: str | None) -> dict:
        text = _normalize_text(query_text or "")
        if not text:
            return {"enabled": False, "anchor_tokens": [], "min_match": 1}

        raw_tokens = [t.strip() for t in thai_tokenize(text) if str(t).strip()]
        stop_tokens = {
            "\u0e01\u0e32\u0e23",  # การ
            "\u0e02\u0e2d\u0e07",  # ของ
            "\u0e14\u0e49\u0e27\u0e22",  # ด้วย
            "\u0e41\u0e1a\u0e1a",  # แบบ
            "\u0e17\u0e35\u0e48",  # ที่
            "\u0e43\u0e19",  # ใน
            "\u0e08\u0e32\u0e01",  # จาก
            "\u0e44\u0e1b",  # ไป
            "\u0e41\u0e25\u0e30",  # และ
            "\u0e2b\u0e23\u0e37\u0e2d",  # หรือ
            "\u0e2d\u0e22\u0e32\u0e01",  # อยาก
            "\u0e23\u0e39\u0e49",  # รู้
            "\u0e40\u0e23\u0e37\u0e48\u0e2d\u0e07",  # เรื่อง
            "\u0e17\u0e33\u0e07\u0e32\u0e19",  # ทำงาน
            "\u0e14\u0e33\u0e40\u0e19\u0e34\u0e19\u0e01\u0e32\u0e23",  # ดำเนินการ
            "\u0e41\u0e17\u0e19",  # แทน
            "topic",
            "other",
            "the",
            "of",
            "and",
        }
        cleaned = []
        for token in raw_tokens:
            token = _normalize_text(token)
            if not token:
                continue
            if len(token) < 2:
                continue
            if token in stop_tokens:
                continue
            cleaned.append(token)

        # preserve order with de-dup and keep first few informative tokens
        seen = set()
        anchor_tokens = []
        for token in cleaned:
            if token in seen:
                continue
            seen.add(token)
            anchor_tokens.append(token)
            if len(anchor_tokens) >= 8:
                break

        must_groups = []
        queue_group = ["\u0e04\u0e34\u0e27", "queue"]
        circular_group = ["\u0e27\u0e07\u0e01\u0e25\u0e21", "circular"]
        linked_list_group = ["\u0e25\u0e34\u0e07\u0e04\u0e4c\u0e25\u0e34\u0e2a\u0e15\u0e4c", "\u0e25\u0e34\u0e07\u0e01\u0e4c\u0e25\u0e34\u0e2a\u0e15\u0e4c", "linked list"]
        stack_group = ["\u0e2a\u0e41\u0e15\u0e47\u0e01", "stack"]
        tree_group = ["\u0e17\u0e23\u0e35", "tree", "binary"]
        graph_group = ["\u0e01\u0e23\u0e32\u0e1f", "graph"]
        operation_group = [
            "\u0e14\u0e33\u0e40\u0e19\u0e34\u0e19\u0e01\u0e32\u0e23",  # ดำเนินการ
            "\u0e17\u0e33\u0e07\u0e32\u0e19",  # ทำงาน
            "insert",
            "remove",
            "enqueue",
            "dequeue",
            "push",
            "pop",
            "front",
            "rear",
        ]

        if any(t in text for t in queue_group):
            must_groups.append(queue_group)
        if any(t in text for t in circular_group):
            must_groups.append(circular_group)
        if any(t in text for t in linked_list_group):
            must_groups.append(linked_list_group)
        if any(t in text for t in stack_group):
            must_groups.append(stack_group)
        if any(t in text for t in tree_group):
            must_groups.append(tree_group)
        if any(t in text for t in graph_group):
            must_groups.append(graph_group)
        if any(t in text for t in operation_group):
            must_groups.append(operation_group)
        # Section 1.2 - Basic Data Structures (บิต ไบต์ ฟิลด์ เรคอร์ด ไฟล์ ฐานข้อมูล)
        basic_data_structure_group = [
            "\u0e1a\u0e34\u0e15", "bit",  # บิต
            "\u0e44\u0e1a\u0e15\u0e4c", "byte",  # ไบต์
            "\u0e1f\u0e34\u0e25\u0e14\u0e4c", "field",  # ฟิลด์
            "\u0e40\u0e23\u0e04\u0e2d\u0e23\u0e4c\u0e14", "record",  # เรคอร์ด
            "\u0e44\u0e1f\u0e25\u0e4c", "file",  # ไฟล์
            "\u0e10\u0e32\u0e19\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25", "database",  # ฐานข้อมูล
        ]
        if any(t in text for t in basic_data_structure_group):
            must_groups.append(basic_data_structure_group)

        enabled = bool(anchor_tokens) or bool(must_groups)
        min_match = 2 if len(anchor_tokens) >= 2 else 1
        return {
            "enabled": bool(enabled),
            "anchor_tokens": anchor_tokens,
            "min_match": int(min_match),
            "must_groups": must_groups,
        }

    def _doc_query_anchor_match(self, doc: Document, profile: dict | None) -> bool:
        if not profile or not profile.get("enabled", False):
            return True
        meta = getattr(doc, "metadata", {}) or {}
        haystack_parts = [
            str(meta.get("chapter", "")),
            str(meta.get("section", "")),
            str(meta.get("best_topic_title", "")),
            (doc.page_content or "")[:1200],
        ]
        haystack = _normalize_text(" ".join(haystack_parts))
        if not haystack:
            return False

        tokens = [str(t).strip() for t in profile.get("anchor_tokens", []) if str(t).strip()]
        must_groups = [g for g in profile.get("must_groups", []) if isinstance(g, list) and g]
        for group in must_groups:
            if not any(str(term).strip() and str(term) in haystack for term in group):
                return False
        if not tokens:
            return True
        matched = sum(1 for t in tokens if t in haystack)
        return matched >= int(profile.get("min_match", 1))

    def _doc_topic_match(self, doc: Document, topic_hint: str | None) -> bool:
        if not topic_hint:
            return True
        hint = _normalize_text(topic_hint)
        if not hint or hint == "other":
            return True

        meta = getattr(doc, "metadata", {}) or {}
        haystack_parts = [
            str(meta.get("chapter", "")),
            str(meta.get("section", "")),
            str(meta.get("best_topic_id", "")),
            " ".join(str(x) for x in meta.get("top_topic_ids", []) if str(x).strip()),
            str(meta.get("topic_path", "")),
            " ".join(str(x) for x in meta.get("topic_tags", []) if str(x).strip()),
            (doc.page_content or "")[:500],
        ]
        haystack = _normalize_text(" ".join(haystack_parts))
        if not haystack:
            return False
        if hint in haystack:
            return True

        for kw in self._topic_keywords(topic_hint):
            if _normalize_text(kw) in haystack:
                return True
        return False

    def _doc_section_match(self, doc: Document, section_id: str) -> bool:
        sid = _normalize_text(section_id or "")
        if not sid:
            return False
        meta = getattr(doc, "metadata", {}) or {}
        section_candidates = set()
        best_tid = str(meta.get("best_topic_id", "")).strip()
        if best_tid:
            section_candidates.add(_normalize_text(best_tid))
        for tid in meta.get("top_topic_ids", []) or []:
            t = str(tid).strip()
            if t:
                section_candidates.add(_normalize_text(t))
        if sid in section_candidates:
            return True

        # Keep fallback strict to metadata fields only (exclude free-text tags/content)
        # to avoid noisy cross-section leakage.
        haystack_parts = [
            str(meta.get("chapter", "")),
            str(meta.get("section", "")),
            str(meta.get("best_topic_id", "")),
            " ".join(str(x) for x in meta.get("top_topic_ids", []) if str(x).strip()),
        ]
        haystack = _normalize_text(" ".join(haystack_parts))
        return bool(sid and sid in haystack)

    def _doc_query_phrase_match(self, doc: Document, query_text: str | None) -> bool:
        q = _normalize_text(query_text or "")
        if not q or len(q) < 8:
            return False
        meta = getattr(doc, "metadata", {}) or {}
        haystack_parts = [
            str(meta.get("chapter", "")),
            str(meta.get("section", "")),
            str(meta.get("topic_path", "")),
            " ".join(str(x) for x in meta.get("topic_tags", []) if str(x).strip()),
            (doc.page_content or "")[:1400],
        ]
        haystack = _normalize_text(" ".join(haystack_parts))
        return q in haystack

    def _apply_filters(self, docs: list[Document], filters: dict | None) -> list[Document]:
        if not docs or not filters:
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
        query_profile = self._build_query_anchor_profile(query_text)

        enriched = []
        for doc in docs:
            self._enrich_doc_with_hierarchy_topics(doc)
            meta = getattr(doc, "metadata", {}) or {}
            topic_match = self._doc_topic_match(doc, topic_hint)
            structure_match = self._doc_has_structure(doc)
            query_anchor_match = self._doc_query_anchor_match(doc, query_profile)
            meta["topic_match"] = bool(topic_match)
            meta["structure_match"] = bool(structure_match)
            meta["query_anchor_match"] = bool(query_anchor_match)
            doc.metadata = meta
            enriched.append(doc)

        ordered = list(enriched)
        if topic_hint and topic_hint.lower() != "other":
            topic_docs = [d for d in enriched if d.metadata.get("topic_match")]
            off_topic_docs = [d for d in enriched if not d.metadata.get("topic_match")]
            if topic_docs:
                ordered = topic_docs + off_topic_docs[:max(0, allow_offtopic_docs)]
            elif strict_topic and not target_section_id:
                # If section is known, defer strictness to section/page gate.
                return []

        if target_section_id:
            sid_norm = _normalize_text(target_section_id)

            def _strict_section_hit(doc: Document) -> bool:
                meta = getattr(doc, "metadata", {}) or {}
                best_tid = _normalize_text(str(meta.get("best_topic_id", "")).strip())
                if sid_norm and best_tid == sid_norm:
                    return True
                top_ids = [_normalize_text(str(x).strip()) for x in (meta.get("top_topic_ids", []) or []) if str(x).strip()]
                return sid_norm in set(top_ids)

            section_docs = [d for d in ordered if _strict_section_hit(d)]
            non_section_docs = [d for d in ordered if not _strict_section_hit(d)]
            if section_docs:
                ordered = section_docs + non_section_docs[:max(0, allow_offtopic_docs)]

            page_allowlist = self._section_page_allowlist(target_section_id)
            if page_allowlist:
                section_page_docs = [d for d in ordered if self._doc_matches_page_allowlist(d, page_allowlist)]
                if section_page_docs:
                    ordered = section_page_docs
                else:
                    seed_docs = self._supplement_section_docs(
                        target_section_id,
                        query_text=query_text,
                        query_profile=None if strict_section_only else query_profile,
                        limit=max(8, self.default_top_n * 2),
                    )
                    if seed_docs:
                        ordered = seed_docs
                    elif strict_section_only:
                        return []

            section_pages = {
                str((getattr(d, "metadata", {}) or {}).get("page", "")).strip()
                for d in ordered
                if str((getattr(d, "metadata", {}) or {}).get("page", "")).strip()
            }
            min_section_pages = int(filters.get("min_section_page_diversity", 2) or 2)
            if page_allowlist:
                allow_pages = {x.split(":")[-1] for x in page_allowlist if str(x).strip()}
                if allow_pages:
                    min_section_pages = min(max(1, min_section_pages), len(allow_pages))
            if strict_section_only and (len(section_pages) < max(1, min_section_pages)):
                supplements = self._supplement_section_docs(
                    target_section_id,
                    query_text=query_text,
                    query_profile=None if strict_section_only else query_profile,
                    limit=max(8, self.default_top_n * 2),
                )
                if supplements:
                    merged = []
                    seen = set()
                    for d in ordered + supplements:
                        k = self._stable_doc_key(d)
                        if k in seen:
                            continue
                        seen.add(k)
                        merged.append(d)
                    ordered = merged
            elif (not strict_section_only) and ((not section_docs) or (len(section_pages) < max(1, min_section_pages))):
                supplements = self._supplement_section_docs(
                    target_section_id,
                    query_text=query_text,
                    query_profile=query_profile,
                    limit=max(8, allow_offtopic_docs + 8),
                )
                if supplements:
                    merged = []
                    seen = set()
                    for d in section_docs + supplements + non_section_docs[:max(0, allow_offtopic_docs)]:
                        k = self._stable_doc_key(d)
                        if k in seen:
                            continue
                        seen.add(k)
                        merged.append(d)
                    ordered = merged
        # Exact-phrase prefilter is useful for broad search, but too aggressive for
        # section-constrained operation queries (can collapse to one page).
        if not target_section_id:
            exact_query_docs = [d for d in ordered if self._doc_query_phrase_match(d, query_text)]
            if exact_query_docs:
                non_exact_docs = [d for d in ordered if not self._doc_query_phrase_match(d, query_text)]
                ordered = exact_query_docs + non_exact_docs[:max(0, allow_offtopic_docs)]

        if query_profile.get("enabled", False):
            for d in ordered:
                meta = getattr(d, "metadata", {}) or {}
                meta["query_anchor_match"] = bool(self._doc_query_anchor_match(d, query_profile))
                d.metadata = meta
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

    # -- public API --------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
        filters: dict | None = None,
    ) -> list[Document]:
        """
        Full retrieval pipeline with graceful degradation:
        - Dense search (if available)
        - Sparse search (always)
        - Fusion
        - Optional reranking
        """
        k = self.default_top_k if top_k is None else int(top_k)
        n = self.default_top_n if top_n is None else int(top_n)
        if k < 1:
            k = 1
        if n < 1:
            n = 1
        if n > k:
            n = k

        dense_docs = self._dense_search(query, k)
        sparse_docs = self._sparse_search(query, k)
        combined = self._ensemble(dense_docs, sparse_docs)
        filtered = self._apply_filters(combined, filters)
        if not filtered:
            return []

        # Ensure every candidate has numeric rrf_score, including docs injected by
        # section supplement logic that may bypass fusion score assignment.
        for idx, d in enumerate(filtered, start=1):
            meta = getattr(d, "metadata", {}) or {}
            rrf = meta.get("rrf_score")
            if not isinstance(rrf, (int, float)):
                # conservative rank-based fallback
                meta["rrf_score"] = round(1.0 / (float(self.rrf_k) + float(idx)), 8)
                d.metadata = meta

        rerank_depth = min(len(filtered), max(n, n * 3))
        reranked = self._rerank(query, filtered, rerank_depth)

        target_section_id = ""
        min_section_pages = 1
        if isinstance(filters, dict):
            target_section_id = str(filters.get("target_section_id", "")).strip()
            try:
                min_section_pages = int(filters.get("min_section_page_diversity", 1) or 1)
            except Exception:
                min_section_pages = 1

        if target_section_id and min_section_pages > 1:
            return self._select_with_page_diversity(
                reranked,
                filtered,
                limit=n,
                min_unique_pages=min_section_pages,
            )

        return reranked[:n]

    def get_runtime_status(self) -> dict:
        """Expose runtime retrieval status for app logging and diagnostics."""
        return {
            "index_dir": str(self.index_dir),
            "index_verify_enabled": bool(self.index_verify_enabled),
            "index_verify_strict": bool(self.index_verify_strict),
            "index_integrity_ok": self.index_integrity_ok,
            "index_integrity_error": self.index_integrity_error,
            "index_manifest": str(self.index_manifest),
            "dense_enabled": bool(self.dense_enabled),
            "dense_ready": self.vectorstore is not None,
            "dense_error": self.dense_error,
            "reranker_enabled": bool(self.reranker_enabled),
            "reranker_ready": self.reranker is not None,
            "reranker_model": self.reranker_model,
            "reranker_candidates": self.reranker_candidates,
            "reranker_healthcheck_enabled": bool(self.reranker_healthcheck_enabled),
            "reranker_startup_budget_sec": self.reranker_startup_budget_sec,
            "reranker_load_seconds": self.reranker_load_seconds,
            "reranker_error": self.reranker_error,
            "rrf_k": self.rrf_k,
            "fusion_weight_dense": self.weight_dense,
            "fusion_weight_sparse": self.weight_sparse,
            "default_top_k": self.default_top_k,
            "default_top_n": self.default_top_n,
        }
