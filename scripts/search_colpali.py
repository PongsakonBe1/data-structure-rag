"""
Search ColPali index and return page/image evidence hits.

Input:
- indexes/colpali/colpali_index.pt

Output:
- prints top hits
- optional JSON output
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from pythainlp.tokenize import word_tokenize
except Exception:  # pragma: no cover
    word_tokenize = None

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def choose_device(device_arg: str) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def tokenize_query(text: str) -> list[str]:
    text = (text or "").strip().lower()
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


def lexical_score(query_tokens: list[str], rec: dict) -> float:
    if not query_tokens:
        return 0.0
    hay = " ".join(
        [
            str(rec.get("text", "")),
            " ".join(str(x) for x in rec.get("tags", []) if str(x).strip()),
            " ".join(str(x) for x in rec.get("structure_labels", []) if str(x).strip()),
            " ".join(str(x) for x in rec.get("figure_refs", []) if str(x).strip()),
        ]
    ).lower()
    if not hay:
        return 0.0
    hit = sum(1 for t in query_tokens if t and t in hay)
    return hit / max(1, len(query_tokens))


def to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def trim_query_embedding(emb: torch.Tensor, attn: torch.Tensor | None) -> torch.Tensor:
    if emb.ndim < 2:
        return emb.detach().cpu()
    q = emb[0].detach().cpu()
    if attn is None:
        return q
    mask = attn[0].detach().cpu().bool()
    if mask.ndim == 1 and mask.shape[0] == q.shape[0]:
        kept = q[mask]
        if kept.numel() > 0:
            return kept
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description="Search ColPali index.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--index", default="indexes/colpali/colpali_index.pt")
    ap.add_argument("--pages-jsonl", default="indexes/colpali/pages.jsonl")
    ap.add_argument("--model", default="vidore/colpali-v1.2-hf")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--require-structure", action="store_true")
    ap.add_argument("--require-example", action="store_true")
    ap.add_argument("--tag", action="append", default=[])
    ap.add_argument("--max-per-page", type=int, default=1)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    index_path = Path(args.index)
    pages_jsonl = Path(args.pages_jsonl)
    use_metadata_only = bool(args.metadata_only or not index_path.exists())

    records: list[dict]
    passages: list[torch.Tensor]
    if use_metadata_only:
        if not pages_jsonl.exists():
            raise FileNotFoundError(
                f"metadata-only mode requires --pages-jsonl, file not found: {pages_jsonl}"
            )
        records = []
        with pages_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        passages = []
    else:
        payload = torch.load(index_path, map_location="cpu")
        records = payload.get("records", [])
        passages = payload.get("embeddings", [])
        if not records or not passages:
            raise ValueError("Index payload is empty.")
        if len(records) != len(passages):
            raise ValueError("Corrupted index: records and embeddings length mismatch.")

    tags_required = [t.strip().lower() for t in args.tag if str(t).strip()]

    if use_metadata_only:
        score_np = np.zeros(len(records), dtype=np.float64)
    else:
        from transformers import ColPaliForRetrieval, ColPaliProcessor

        device = choose_device(args.device)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        print(f"loading_model={args.model}")
        processor = ColPaliProcessor.from_pretrained(args.model)
        model = ColPaliForRetrieval.from_pretrained(args.model)
        model = model.to(device)
        model.eval()

        with torch.inference_mode():
            q_inputs = processor.process_queries(text=[args.query], return_tensors="pt")
            q_inputs = to_device(dict(q_inputs), device)
            q_out = model(**q_inputs)
            q_emb = trim_query_embedding(q_out.embeddings, q_inputs.get("attention_mask"))

        score_tensor = processor.score_retrieval(
            query_embeddings=[q_emb],
            passage_embeddings=passages,
            batch_size=64,
            output_dtype=torch.float32,
            output_device="cpu",
        )
        score_np = score_tensor[0].detach().cpu().numpy().astype(np.float64)

    query_tokens = tokenize_query(args.query)
    lex_scores = np.array([lexical_score(query_tokens, r) for r in records], dtype=np.float64)
    region_scores = np.array([float(r.get("region_score", 0.0) or 0.0) for r in records], dtype=np.float64)

    col_norm = normalize_scores(score_np) if not use_metadata_only else np.zeros_like(lex_scores)
    lex_norm = normalize_scores(lex_scores)
    reg_norm = normalize_scores(region_scores)

    if use_metadata_only:
        combined = (0.90 * lex_norm) + (0.10 * reg_norm)
    else:
        combined = (0.82 * col_norm) + (0.13 * lex_norm) + (0.05 * reg_norm)

    # Hard filters by policy.
    keep_mask = np.ones(len(records), dtype=bool)
    if args.require_structure:
        keep_mask &= np.array([bool(r.get("has_structure", False)) for r in records], dtype=bool)
    if args.require_example:
        keep_mask &= np.array([bool(r.get("has_example", False)) for r in records], dtype=bool)
    if tags_required:
        rec_tags = [set(str(x).lower() for x in r.get("tags", [])) for r in records]
        keep_mask &= np.array([all(t in rec_tags[i] for t in tags_required) for i in range(len(records))], dtype=bool)

    filtered_idx = np.where(keep_mask)[0]
    if filtered_idx.size == 0:
        raise ValueError("No candidates left after filtering (structure/example/tag).")

    ranked = sorted(filtered_idx.tolist(), key=lambda i: combined[i], reverse=True)

    # Page diversification to avoid returning many crops from same page.
    max_per_page = max(1, int(args.max_per_page))
    seen_by_page: dict[str, int] = {}
    selected = []
    for i in ranked:
        page_id = str(records[i].get("page_id", ""))
        used = seen_by_page.get(page_id, 0)
        if used >= max_per_page:
            continue
        seen_by_page[page_id] = used + 1
        selected.append(i)
        if len(selected) >= max(1, int(args.top_k)):
            break

    rows = []
    for rank, idx in enumerate(selected, start=1):
        rec = records[idx]
        rows.append(
            {
                "rank": rank,
                "score": round(float(combined[idx]), 6),
                "colpali_score": round(float(score_np[idx]), 6),
                "lexical_score": round(float(lex_scores[idx]), 6),
                "source": rec.get("source"),
                "page": rec.get("page"),
                "page_id": rec.get("page_id"),
                "image_path": rec.get("image_path"),
                "image_level": rec.get("image_level"),
                "region_index": rec.get("region_index"),
                "has_structure": bool(rec.get("has_structure", False)),
                "has_example": bool(rec.get("has_example", False)),
                "tags": rec.get("tags", []),
                "structure_labels": rec.get("structure_labels", []),
                "figure_refs": rec.get("figure_refs", []),
                "preview": str(rec.get("text", ""))[:280],
            }
        )

    result = {
        "query": args.query,
        "mode": "metadata_only" if use_metadata_only else "colpali",
        "top_k": int(args.top_k),
        "require_structure": bool(args.require_structure),
        "require_example": bool(args.require_example),
        "required_tags": tags_required,
        "count": len(rows),
        "hits": rows,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_output={out}")


if __name__ == "__main__":
    main()
