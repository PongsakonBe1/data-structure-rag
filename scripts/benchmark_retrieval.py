"""
Benchmark hybrid retrieval modes for typhoon_rag.

Dataset format (JSONL, one object per line):
{"query": "...", "relevant_contains": ["keyword1", "keyword2"]}
{"query": "...", "relevant_sources": ["chapter_2.pdf"]}
{"query": "...", "relevant_sources": ["..."], "relevant_contains": ["..."]}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TYPE_CHECKING


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if TYPE_CHECKING:
    from retriever import RAGSystem  # noqa: F401


MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class EvalResult:
    mode: str
    queries: int
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            query = str(item.get("query", "")).strip()
            if not query:
                raise ValueError(f"Line {line_no}: missing 'query'")
            rows.append(item)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def is_relevant(doc, item: dict) -> bool:
    text = (doc.page_content or "").lower()
    source = str(doc.metadata.get("source", "")).lower() if hasattr(doc, "metadata") else ""

    source_targets = [str(x).lower() for x in item.get("relevant_sources", []) if str(x).strip()]
    text_targets = [str(x).lower() for x in item.get("relevant_contains", []) if str(x).strip()]

    source_hit = any(target in source for target in source_targets)
    text_hit = any(target in text for target in text_targets)
    return source_hit or text_hit


def dcg(binary_relevances: Iterable[int]) -> float:
    score = 0.0
    for idx, rel in enumerate(binary_relevances, start=1):
        if rel:
            score += rel / math.log2(idx + 1)
    return score


def get_docs_for_mode(rag: "RAGSystem", query: str, mode: str, k: int, top_n: int | None = None):
    dense = rag._dense_search(query, k)
    sparse = rag._sparse_search(query, k)
    n = k if top_n is None else max(1, min(int(top_n), k))

    if mode == "dense":
        return dense[:k]
    if mode == "bm25":
        return sparse[:k]

    combined = rag._ensemble(dense, sparse)
    if mode == "hybrid":
        return combined[:k]
    if mode == "hybrid_rerank":
        return rag._rerank(query, combined, top_n=n)

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_mode(rag: "RAGSystem", rows: list[dict], mode: str, k: int, top_n: int | None = None) -> EvalResult:
    recall_hits = 0
    mrr_total = 0.0
    ndcg_total = 0.0

    for item in rows:
        docs = get_docs_for_mode(rag, item["query"], mode, k, top_n=top_n)
        relevances = [1 if is_relevant(doc, item) else 0 for doc in docs]

        if any(relevances):
            recall_hits += 1
            first_idx = relevances.index(1) + 1
            mrr_total += 1.0 / first_idx

        ideal_hits = sum(relevances)
        idcg = dcg([1] * ideal_hits) if ideal_hits > 0 else 0.0
        ndcg_total += (dcg(relevances) / idcg) if idcg > 0 else 0.0

    n = len(rows)
    return EvalResult(
        mode=mode,
        queries=n,
        recall_at_k=recall_hits / n,
        mrr_at_k=mrr_total / n,
        ndcg_at_k=ndcg_total / n,
    )


def save_csv(path: Path, results: list[EvalResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "queries", "recall_at_k", "mrr_at_k", "ndcg_at_k"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "mode": r.mode,
                    "queries": r.queries,
                    "recall_at_k": f"{r.recall_at_k:.4f}",
                    "mrr_at_k": f"{r.mrr_at_k:.4f}",
                    "ndcg_at_k": f"{r.ndcg_at_k:.4f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark typhoon_rag retrieval modes.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL evaluation dataset.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for @k metrics.")
    parser.add_argument("--output", help="Optional output CSV path.")
    parser.add_argument("--index-dir", help="Optional index directory override (sets RAG_INDEX_DIR).")
    parser.add_argument("--disable-dense", action="store_true", help="Force sparse-only mode.")
    parser.add_argument("--enable-reranker", action="store_true", help="Enable reranker during benchmark.")
    parser.add_argument("--top-n", type=int, default=None, help="Final cutoff after reranking (default: k).")
    parser.add_argument("--rrf-k", type=int, default=None, help="RRF constant k for fusion.")
    parser.add_argument("--dense-weight", type=float, default=None, help="Dense fusion weight.")
    parser.add_argument("--sparse-weight", type=float, default=None, help="Sparse fusion weight.")
    parser.add_argument(
        "--verify-nonstrict",
        action="store_true",
        help="Continue even if checksum verification fails (INDEX_VERIFY_STRICT=0).",
    )
    args = parser.parse_args()

    if args.index_dir:
        os.environ["RAG_INDEX_DIR"] = str(Path(args.index_dir).resolve())
    if args.disable_dense:
        os.environ["DENSE_RETRIEVAL_ENABLED"] = "0"
    if args.enable_reranker:
        os.environ["RERANKER_ENABLED"] = "1"
    if args.rrf_k is not None:
        os.environ["RRF_K"] = str(args.rrf_k)
    if args.dense_weight is not None:
        os.environ["FUSION_WEIGHT_DENSE"] = str(args.dense_weight)
    if args.sparse_weight is not None:
        os.environ["FUSION_WEIGHT_SPARSE"] = str(args.sparse_weight)
    if args.top_n is not None:
        os.environ["RETRIEVE_TOP_N"] = str(args.top_n)
    if args.verify_nonstrict:
        os.environ["INDEX_VERIFY_STRICT"] = "0"

    from retriever import RAGSystem  # lazy import to keep --help fast

    rows = load_dataset(Path(args.dataset))
    rag = RAGSystem()
    status = rag.get_runtime_status() if hasattr(rag, "get_runtime_status") else {}
    print("Runtime status:", status)

    results = [evaluate_mode(rag, rows, mode, args.k, top_n=args.top_n) for mode in MODES]

    print(f"\nResults @k={args.k}")
    print("mode           recall    mrr      ndcg")
    print("-------------------------------------------")
    for r in results:
        print(f"{r.mode:<14} {r.recall_at_k:>7.4f}  {r.mrr_at_k:>7.4f}  {r.ndcg_at_k:>7.4f}")

    if args.output:
        save_csv(Path(args.output), results)
        print(f"\nSaved CSV: {args.output}")


if __name__ == "__main__":
    main()
