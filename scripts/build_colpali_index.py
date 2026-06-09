"""
Build ColPali image index from prepared corpus JSONL.

Input:
- indexes/colpali/pages.jsonl

Output:
- indexes/colpali/colpali_index.pt
- logs/colpali_index_report_latest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def choose_device(device_arg: str) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def trim_embeddings(emb: torch.Tensor, attention_mask: torch.Tensor | None) -> list[torch.Tensor]:
    items = []
    for i in range(emb.shape[0]):
        if attention_mask is None:
            items.append(emb[i].detach().cpu())
            continue
        mask = attention_mask[i].detach().cpu().bool()
        if mask.ndim == 0:
            items.append(emb[i].detach().cpu())
            continue
        if mask.shape[0] != emb.shape[1]:
            items.append(emb[i].detach().cpu())
            continue
        kept = emb[i].detach().cpu()[mask]
        if kept.numel() == 0:
            kept = emb[i].detach().cpu()
        items.append(kept)
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ColPali index from image corpus JSONL.")
    ap.add_argument("--input", default="indexes/colpali/pages.jsonl")
    ap.add_argument("--output", default="indexes/colpali/colpali_index.pt")
    ap.add_argument("--report", default="logs/colpali_index_report_latest.json")
    ap.add_argument("--model", default="vidore/colpali-v1.2-hf")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-records", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    report_path = Path(args.report)

    if not in_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")

    records = []
    skipped_missing_image = 0
    for rec in iter_jsonl(in_path):
        img_path = Path(str(rec.get("image_path", "")))
        if not img_path.exists():
            skipped_missing_image += 1
            continue
        records.append(rec)
        if args.max_records and len(records) >= int(args.max_records):
            break

    if not records:
        raise ValueError("No valid records with existing images were found.")

    device = choose_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dtype = torch.float32
    if device == "cuda":
        try:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            dtype = torch.float16

    print(f"loading_model={args.model}")
    try:
        processor = ColPaliProcessor.from_pretrained(args.model)
        model = ColPaliForRetrieval.from_pretrained(args.model, torch_dtype=dtype)
        model = model.to(device)
        model.eval()
    except Exception as exc:
        msg = str(exc)
        hint = (
            "Model load failed. On Windows this is often due to insufficient RAM/pagefile "
            "(os error 1455) or disk space. "
            "Use a machine with more memory/storage, or increase page file and free disk."
        )
        raise RuntimeError(f"{hint}\nOriginal error: {msg}") from exc

    embeddings: list[torch.Tensor] = []
    kept_records: list[dict] = []

    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = []
        good_batch = []
        for rec in batch:
            img_path = Path(str(rec.get("image_path", "")))
            try:
                images.append(Image.open(img_path).convert("RGB"))
                good_batch.append(rec)
            except Exception:
                continue

        if not images:
            continue

        with torch.inference_mode():
            proc = processor.process_images(images=images, return_tensors="pt")
            proc = to_device(dict(proc), device)
            out = model(**proc)
            emb = out.embeddings
            attn = proc.get("attention_mask")
            trimmed = trim_embeddings(emb, attn)

        for rec, e in zip(good_batch, trimmed):
            embeddings.append(e.to(torch.float16))
            kept_records.append(rec)

        print(f"encoded={min(start + batch_size, len(records))}/{len(records)}")

    if not embeddings:
        raise ValueError("ColPali encoding produced no embeddings.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "records": kept_records,
        "embeddings": embeddings,
    }
    torch.save(payload, out_path)

    token_lengths = [int(e.shape[0]) for e in embeddings if e.ndim >= 2]
    dims = [int(e.shape[-1]) for e in embeddings if e.ndim >= 2]
    report = {
        "input_records": len(records),
        "indexed_records": len(kept_records),
        "skipped_missing_image": skipped_missing_image,
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "avg_tokens_per_image": round(sum(token_lengths) / max(1, len(token_lengths)), 4),
        "embedding_dim": dims[0] if dims else None,
        "output": str(out_path.as_posix()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved_index={out_path}")
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    main()
