import os
import io
import json
import base64
import hashlib
from collections import OrderedDict
from typing import Any

import gradio as gr
import spaces
import torch
import httpx
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
import torch.nn.functional as F

MODEL_NAME = os.getenv("COLPALI_MODEL", "vidore/colpali-v1.3")
BATCH_SIZE = int(os.getenv("COLPALI_BATCH_SIZE", "8"))
MAX_IMAGE_SIZE = int(os.getenv("COLPALI_MAX_IMAGE_SIZE", "1024"))
EMBED_CACHE_MAX = int(os.getenv("COLPALI_EMBED_CACHE_MAX", "4096"))
CORPUS_CACHE_MAX = int(os.getenv("COLPALI_CORPUS_CACHE_MAX", "8"))

# Load once; ZeroGPU will move model to GPU during @spaces.GPU execution.
model = ColPali.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu",
).eval()
processor = ColPaliProcessor.from_pretrained(MODEL_NAME)

EMBED_CACHE: "OrderedDict[str, torch.Tensor]" = OrderedDict()
CORPUS_CACHE: "OrderedDict[str, dict[str, torch.Tensor]]" = OrderedDict()


def _cache_get(cache: OrderedDict, key: str):
    if key in cache:
        val = cache.pop(key)
        cache[key] = val
        return val
    return None


def _cache_put(cache: OrderedDict, key: str, value, max_size: int):
    if key in cache:
        cache.pop(key)
    cache[key] = value
    while len(cache) > max(1, int(max_size)):
        cache.popitem(last=False)


def _to_device(batch: Any, device: str):
    if hasattr(batch, "to"):
        try:
            return batch.to(device)
        except Exception:
            pass
    if isinstance(batch, dict):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [_to_device(v, device) for v in batch]
        return type(batch)(moved)
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def _resize_if_needed(img: Image.Image) -> Image.Image:
    if max(img.size) <= MAX_IMAGE_SIZE:
        return img
    ratio = MAX_IMAGE_SIZE / max(img.size)
    new_size = (max(1, int(img.size[0] * ratio)), max(1, int(img.size[1] * ratio)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _decode_image_from_candidate(cand: dict) -> Image.Image | None:
    # 1) image_base64
    img_b64 = str(cand.get("image_base64", "") or "").strip()
    if img_b64:
        if img_b64.startswith("data:"):
            img_b64 = img_b64.split(",", 1)[-1]
        raw = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return _resize_if_needed(img)

    # 2) image_url
    image_url = str(cand.get("image_url", "") or "").strip()
    if image_url:
        with httpx.Client(timeout=20) as client:
            res = client.get(image_url)
            res.raise_for_status()
            img = Image.open(io.BytesIO(res.content)).convert("RGB")
            return _resize_if_needed(img)

    # 3) image_path (local path inside the Space runtime)
    image_path = str(cand.get("image_path", "") or "").strip()
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        return _resize_if_needed(img)

    return None


def _candidate_fingerprint(cand: dict) -> str:
    cid = str(cand.get("id", "") or "").strip()
    img_b64 = str(cand.get("image_base64", "") or "").strip()
    img_url = str(cand.get("image_url", "") or "").strip()
    img_path = str(cand.get("image_path", "") or "").strip()
    raw = f"{cid}|{img_path}|{img_url}|{img_b64}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def _extract_query_embedding(model_out, query_batch):
    # Compatible with different output formats
    if hasattr(model_out, "embeddings"):
        q = model_out.embeddings[0]
    elif isinstance(model_out, torch.Tensor):
        q = model_out[0]
    else:
        raise ValueError("Unsupported model output format for query embedding")

    q_mask = query_batch.get("attention_mask") if isinstance(query_batch, dict) else None
    if q_mask is not None:
        m = q_mask[0].detach().cpu().bool()
        q_cpu = q.detach().cpu()
        if m.ndim == 1 and m.shape[0] == q_cpu.shape[0]:
            kept = q_cpu[m]
            if kept.numel() > 0:
                return kept
    return q.detach().cpu()


def _compute_query_embedding(query: str, device: str) -> torch.Tensor:
    q_batch = processor.process_queries([query])
    q_batch = _to_device(q_batch, device)
    q_out = model(**q_batch)
    return _extract_query_embedding(q_out, q_batch)


def _build_passage_embeddings(candidates: list[dict], device: str) -> tuple[list[str], list[torch.Tensor]]:
    """
    Build passage embeddings in input order, reusing cache where possible.
    """
    rows: list[tuple[str, torch.Tensor] | None] = [None] * len(candidates)
    pending: list[tuple[int, str, str, Image.Image]] = []

    for i, c in enumerate(candidates):
        cid = str(c.get("id", "") or "").strip()
        if not cid:
            continue
        fp = _candidate_fingerprint(c)
        cached = _cache_get(EMBED_CACHE, fp)
        if isinstance(cached, torch.Tensor):
            rows[i] = (cid, cached.detach().cpu())
            continue
        try:
            img = _decode_image_from_candidate(c)
        except Exception:
            img = None
        if img is None:
            continue
        pending.append((i, cid, fp, img))

    for start in range(0, len(pending), max(1, BATCH_SIZE)):
        batch = pending[start : start + max(1, BATCH_SIZE)]
        imgs = [x[3] for x in batch]
        p_batch = processor.process_images(imgs)
        p_batch = _to_device(p_batch, device)
        p_out = model(**p_batch)

        if hasattr(p_out, "embeddings"):
            p_tensor = p_out.embeddings
        elif isinstance(p_out, torch.Tensor):
            p_tensor = p_out
        else:
            raise ValueError("Unsupported model output format for passage embedding")

        for bi, (idx, cid, fp, _) in enumerate(batch):
            emb = p_tensor[bi].detach().cpu()
            rows[idx] = (cid, emb)
            _cache_put(EMBED_CACHE, fp, emb, EMBED_CACHE_MAX)

    ids = []
    embs = []
    for row in rows:
        if isinstance(row, tuple):
            ids.append(row[0])
            embs.append(row[1])
    return ids, embs


def _score_retrieval_safe(query_embedding: torch.Tensor, passage_embeddings: list[torch.Tensor]) -> tuple[list[float], str]:
    """
    Prefer native ColPali retrieval scoring. If device mismatch occurs in some runtimes,
    fallback to pooled cosine for availability.
    """
    q_cpu = query_embedding.detach().cpu()
    p_cpu = [p.detach().cpu() for p in passage_embeddings]
    warn = ""

    try:
        score_tensor = processor.score_retrieval(
            query_embeddings=[q_cpu],
            passage_embeddings=p_cpu,
            batch_size=64,
            output_dtype=torch.float32,
            output_device="cpu",
        )
        vals = score_tensor[0].detach().cpu().numpy().tolist()
        return [float(v) for v in vals], warn
    except Exception as exc:
        warn = f"score_retrieval_fallback:{exc}"

    def _pool(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.ndim >= 2:
            x = x.mean(dim=0)
        else:
            x = x.reshape(-1)
        return F.normalize(x, dim=0)

    q_vec = _pool(q_cpu)
    vals = []
    for p in p_cpu:
        p_vec = _pool(p)
        vals.append(float(torch.dot(q_vec, p_vec).item()))
    return vals, warn


@spaces.GPU
@torch.no_grad()
def score_contract(payload: dict) -> dict:
    """
    Request contract:
      {
        "query": "...",
        "candidates": [{"id":"...", "image_base64":"..." | "image_url":"..." | "image_path":"..."}, ...]
      }

    Response contract:
      {"scores": [{"id":"...", "score": 0.87}, ...]}
    """
    try:
        query = str((payload or {}).get("query", "") or "").strip()
        candidates = (payload or {}).get("candidates", [])

        if not query:
            return {"scores": [], "error": "missing_query"}
        if not isinstance(candidates, list) or not candidates:
            return {"scores": [], "error": "missing_candidates"}

        ordered_ids = [str(c.get("id", f"candidate_{i}")) for i, c in enumerate(candidates)]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        q_emb = _compute_query_embedding(query, device)
        valid_ids, valid_embs = _build_passage_embeddings(candidates, device)

        score_map = {cid: 0.0 for cid in ordered_ids}
        score_warning = ""
        if valid_embs:
            vals, score_warning = _score_retrieval_safe(q_emb, valid_embs)
            for cid, s in zip(valid_ids, vals):
                score_map[cid] = float(s)
        out = {"scores": [{"id": cid, "score": float(score_map.get(cid, 0.0))} for cid in ordered_ids]}
        if score_warning:
            out["warning"] = score_warning
        return out
    except Exception as exc:
        fallback_ids = []
        if isinstance(payload, dict):
            cands = payload.get("candidates") or []
            if isinstance(cands, list):
                for i, c in enumerate(cands):
                    if isinstance(c, dict):
                        fallback_ids.append(str(c.get("id", f"candidate_{i}")))
        return {
            "scores": [{"id": cid, "score": 0.0} for cid in fallback_ids],
            "error": f"runtime_error:{exc}",
        }


@spaces.GPU
@torch.no_grad()
def register_corpus(payload: dict) -> dict:
    """
    Register candidate images into server-side embedding cache by corpus_id.
    Request: {"corpus_id":"...", "candidates":[...]}
    """
    try:
        corpus_id = str((payload or {}).get("corpus_id", "") or "").strip()
        if not corpus_id:
            return {"ok": False, "error": "missing_corpus_id"}

        existing = _cache_get(CORPUS_CACHE, corpus_id)
        if isinstance(existing, dict) and existing:
            return {"ok": True, "corpus_id": corpus_id, "registered": len(existing), "cached": True}

        candidates = (payload or {}).get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            return {"ok": False, "error": "missing_candidates", "corpus_id": corpus_id}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        ids, embs = _build_passage_embeddings(candidates, device)
        if not ids:
            return {"ok": False, "error": "no_valid_candidates", "corpus_id": corpus_id}

        corpus_map = {}
        for cid, emb in zip(ids, embs):
            corpus_map[str(cid)] = emb.detach().cpu()
        _cache_put(CORPUS_CACHE, corpus_id, corpus_map, CORPUS_CACHE_MAX)
        return {"ok": True, "corpus_id": corpus_id, "registered": len(corpus_map), "cached": False}
    except Exception as exc:
        return {"ok": False, "error": f"runtime_error:{exc}"}


@spaces.GPU
@torch.no_grad()
def score_cached(payload: dict) -> dict:
    """
    Score query against a pre-registered corpus.
    Request: {"query":"...", "corpus_id":"...", "candidate_ids":[...]}
    """
    try:
        query = str((payload or {}).get("query", "") or "").strip()
        corpus_id = str((payload or {}).get("corpus_id", "") or "").strip()
        if not query:
            return {"scores": [], "error": "missing_query"}
        if not corpus_id:
            return {"scores": [], "error": "missing_corpus_id"}

        corpus = _cache_get(CORPUS_CACHE, corpus_id)
        if not isinstance(corpus, dict) or not corpus:
            # Fallback: allow one-shot hydration for stateless workers.
            candidates = (payload or {}).get("candidates", [])
            if isinstance(candidates, list) and candidates:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                ids, embs = _build_passage_embeddings(candidates, device)
                if ids:
                    corpus = {str(cid): emb.detach().cpu() for cid, emb in zip(ids, embs)}
                    _cache_put(CORPUS_CACHE, corpus_id, corpus, CORPUS_CACHE_MAX)
            if not isinstance(corpus, dict) or not corpus:
                return {"scores": [], "error": "unknown_corpus_id"}

        candidate_ids = (payload or {}).get("candidate_ids", [])
        ordered_ids = [str(x) for x in candidate_ids] if isinstance(candidate_ids, list) and candidate_ids else list(corpus.keys())

        passage_ids = []
        passage_embs = []
        for cid in ordered_ids:
            emb = corpus.get(str(cid))
            if isinstance(emb, torch.Tensor):
                passage_ids.append(str(cid))
                passage_embs.append(emb.detach().cpu())

        if not passage_ids:
            return {"scores": [{"id": cid, "score": 0.0} for cid in ordered_ids], "error": "missing_candidate_ids"}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        q_emb = _compute_query_embedding(query, device)
        vals, warn = _score_retrieval_safe(q_emb, passage_embs)
        score_map = {cid: float(s) for cid, s in zip(passage_ids, vals)}
        out = {"scores": [{"id": cid, "score": float(score_map.get(cid, 0.0))} for cid in ordered_ids]}
        if warn:
            out["warning"] = warn
        return out
    except Exception as exc:
        return {"scores": [], "error": f"runtime_error:{exc}"}


@spaces.GPU
@torch.no_grad()
def register_and_score(payload: dict) -> dict:
    """
    Stateless-safe Phase-2 API: register + score in one call.
    Request:
      {
        "query":"...",
        "candidates":[...],
        "corpus_id":"optional",
        "candidate_ids":["optional ordered ids"]
      }
    """
    try:
        query = str((payload or {}).get("query", "") or "").strip()
        candidates = (payload or {}).get("candidates", [])
        corpus_id = str((payload or {}).get("corpus_id", "") or "").strip()
        if not query:
            return {"scores": [], "error": "missing_query"}
        if not isinstance(candidates, list) or not candidates:
            return {"scores": [], "error": "missing_candidates"}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        corpus = _cache_get(CORPUS_CACHE, corpus_id) if corpus_id else None
        cache_hit = isinstance(corpus, dict) and bool(corpus)
        if not cache_hit:
            ids, embs = _build_passage_embeddings(candidates, device)
            if not ids:
                return {"scores": [], "error": "no_valid_candidates"}
            corpus = {str(cid): emb.detach().cpu() for cid, emb in zip(ids, embs)}
            if corpus_id:
                _cache_put(CORPUS_CACHE, corpus_id, corpus, CORPUS_CACHE_MAX)

        candidate_ids = (payload or {}).get("candidate_ids", [])
        if isinstance(candidate_ids, list) and candidate_ids:
            ordered_ids = [str(x) for x in candidate_ids]
        else:
            ordered_ids = [str(c.get("id", f"candidate_{i}")) for i, c in enumerate(candidates)]

        passage_ids = []
        passage_embs = []
        for cid in ordered_ids:
            emb = corpus.get(str(cid)) if isinstance(corpus, dict) else None
            if isinstance(emb, torch.Tensor):
                passage_ids.append(str(cid))
                passage_embs.append(emb.detach().cpu())

        if not passage_ids:
            return {"scores": [{"id": cid, "score": 0.0} for cid in ordered_ids], "error": "missing_candidate_ids"}

        q_emb = _compute_query_embedding(query, device)
        vals, warn = _score_retrieval_safe(q_emb, passage_embs)
        score_map = {cid: float(s) for cid, s in zip(passage_ids, vals)}
        out = {
            "scores": [{"id": cid, "score": float(score_map.get(cid, 0.0))} for cid in ordered_ids],
            "phase2_mode": "register_and_score",
            "cache_hit": bool(cache_hit),
        }
        if corpus_id:
            out["corpus_id"] = corpus_id
        if warn:
            out["warning"] = warn
        return out
    except Exception as exc:
        fallback_ids = []
        cands = (payload or {}).get("candidates", [])
        if isinstance(cands, list):
            for i, c in enumerate(cands):
                if isinstance(c, dict):
                    fallback_ids.append(str(c.get("id", f"candidate_{i}")))
        return {
            "scores": [{"id": cid, "score": 0.0} for cid in fallback_ids],
            "error": f"runtime_error:{exc}",
        }


def score_contract_json(request_json: str) -> str:
    """Backward-compatible wrapper for old clients sending JSON as string."""
    try:
        payload = json.loads(request_json)
    except Exception as exc:
        return json.dumps({"scores": [], "error": f"invalid_json:{exc}"}, ensure_ascii=False)
    out = score_contract(payload)
    return json.dumps(out, ensure_ascii=False)


with gr.Blocks(title="CoPali Endpoint") as demo:
    gr.Markdown("# CoPali Endpoint (Contract-Compatible)")
    gr.Markdown("POST contract: `{query, candidates[]}` -> `{scores[]}`")
    gr.Markdown("Phase-2 APIs: `/register_and_score` (stateless-safe), `/register_corpus`, `/score_cached`.")

    with gr.Tab("JSON Contract"):
        req_json = gr.JSON(label="Request JSON")
        res_json = gr.JSON(label="Response JSON")
        run_btn = gr.Button("Run")
        run_btn.click(fn=score_contract, inputs=req_json, outputs=res_json, api_name="score")

    with gr.Tab("Register Corpus"):
        reg_req = gr.JSON(label="Register Request JSON")
        reg_res = gr.JSON(label="Register Response JSON")
        reg_btn = gr.Button("Register")
        reg_btn.click(fn=register_corpus, inputs=reg_req, outputs=reg_res, api_name="register_corpus")

    with gr.Tab("Score Cached Corpus"):
        cached_req = gr.JSON(label="Score Cached Request JSON")
        cached_res = gr.JSON(label="Score Cached Response JSON")
        cached_btn = gr.Button("Score Cached")
        cached_btn.click(fn=score_cached, inputs=cached_req, outputs=cached_res, api_name="score_cached")

    with gr.Tab("Register + Score (One Call)"):
        one_req = gr.JSON(label="Register+Score Request JSON")
        one_res = gr.JSON(label="Register+Score Response JSON")
        one_btn = gr.Button("Run Register+Score")
        one_btn.click(fn=register_and_score, inputs=one_req, outputs=one_res, api_name="register_and_score")

    with gr.Tab("Legacy String API"):
        req_text = gr.Textbox(label="Request JSON String", lines=12)
        res_text = gr.Textbox(label="Response JSON String", lines=12)
        run_legacy_btn = gr.Button("Run Legacy")
        run_legacy_btn.click(fn=score_contract_json, inputs=req_text, outputs=res_text, api_name="api_endpoint")


if __name__ == "__main__":
    demo.queue(max_size=16).launch()
