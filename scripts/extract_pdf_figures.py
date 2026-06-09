"""
Extract embedded figure images from PDF pages and write a manifest JSON.

Usage:
  python scripts/extract_pdf_figures.py --pdf data/data_structure_data_ch1_to_ch5.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF


def extract_figures(pdf_path: Path, output_dir: Path, min_side: int = 48) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    items = []

    for p_idx, page in enumerate(doc, start=1):
        page_items = []
        images = page.get_images(full=True)
        for img_idx, img in enumerate(images, start=1):
            xref = img[0]
            extracted = doc.extract_image(xref)
            if not extracted:
                continue
            img_bytes = extracted.get("image")
            ext = (extracted.get("ext") or "png").lower()
            width = int(extracted.get("width") or 0)
            height = int(extracted.get("height") or 0)
            if not img_bytes or min(width, height) < min_side:
                continue

            filename = f"page_{p_idx:03d}_img_{img_idx:02d}.{ext}"
            out_path = output_dir / filename
            out_path.write_bytes(img_bytes)

            rects = []
            try:
                for rect in page.get_image_rects(xref):
                    rects.append(
                        {
                            "x0": round(float(rect.x0), 2),
                            "y0": round(float(rect.y0), 2),
                            "x1": round(float(rect.x1), 2),
                            "y1": round(float(rect.y1), 2),
                        }
                    )
            except Exception:
                rects = []

            page_items.append(
                {
                    "page": p_idx,
                    "image_index": img_idx,
                    "xref": xref,
                    "path": str(out_path.as_posix()),
                    "width": width,
                    "height": height,
                    "rects": rects,
                }
            )

        items.extend(page_items)

    return {
        "pdf": str(pdf_path.as_posix()),
        "output_dir": str(output_dir.as_posix()),
        "total_images": len(items),
        "images": items,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract embedded figure images from PDF.")
    ap.add_argument("--pdf", required=True, help="Input PDF path.")
    ap.add_argument(
        "--output-dir",
        default="assets/figures",
        help="Directory to save extracted images.",
    )
    ap.add_argument(
        "--manifest",
        default="logs/figure_manifest_latest.json",
        help="Path for output manifest JSON.",
    )
    ap.add_argument(
        "--min-side",
        type=int,
        default=48,
        help="Ignore very small images whose min(width,height) is below this value.",
    )
    args = ap.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    result = extract_figures(pdf_path, output_dir, min_side=args.min_side)
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"pdf={pdf_path}")
    print(f"images={result['total_images']}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
