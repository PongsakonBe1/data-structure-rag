"""
Extract candidate visual regions from page images for Visual RAG.

Input:
- logs/figure_manifest_latest.json
- markdown file with page content (default: final_extracted_content.md)

Output:
- assets/figure_regions/*.png
- logs/figure_regions_manifest_latest.json

This version adds:
- operation-aware target page selection
- visual-density trigger for unlabeled step images
- fallback 3-zone splitting when no region is detected
- richer linkage metadata in manifest
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
FIGURE_REF_RE = re.compile(r"(ภาพที่|ตารางที่|figure|table)\s*\d+(?:\.\d+)?", re.IGNORECASE)
STRUCTURE_RE = re.compile(r"\[Structure:\s*([^\]]+)\]", re.IGNORECASE)
EXAMPLE_RE = re.compile(r"(ตัวอย่างที่|ตัวอย่าง|example)", re.IGNORECASE)
TABLE_HINT_RE = re.compile(r"(ตาราง|table|\|.+\|.+\|)", re.IGNORECASE)
INDEX_PATTERN_RE = re.compile(r"\[\s*\d+\s*\]")
OPERATION_KEYWORD_RE = re.compile(
    r"(insert|remove|enqueue|dequeue|push|pop|front|rear|"
    r"เพิ่ม|ลบ|แทรก|ดึงข้อมูลออก|นำเข้าข้อมูล|นำข้อมูลออก|วงกลม|ขั้นตอน)",
    re.IGNORECASE,
)
VISUAL_ANCHOR_BLOCK_RE = re.compile(
    r"\n### Visual Anchors \(Auto\)[\s\S]*?(?=\n### Visual Captions \(Auto\)|\Z)",
    re.MULTILINE,
)
VISUAL_CAPTION_BLOCK_RE = re.compile(
    r"\n### Visual Captions \(Auto\)[\s\S]*$",
    re.MULTILINE,
)


def strip_auto_visual_blocks(content: str) -> str:
    text = str(content or "")
    text = VISUAL_CAPTION_BLOCK_RE.sub("", text).strip()
    text = VISUAL_ANCHOR_BLOCK_RE.sub("", text).strip()
    return text


def parse_pages(md_text: str) -> list[dict[str, Any]]:
    headers = list(PAGE_HEADER_RE.finditer(md_text))
    pages: list[dict[str, Any]] = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        body = md_text[start:end].strip()
        body_clean = strip_auto_visual_blocks(body)
        figure_refs = [x.group(0).strip() for x in FIGURE_REF_RE.finditer(body_clean)]
        operation_hits = len(OPERATION_KEYWORD_RE.findall(body_clean))
        index_hits = len(INDEX_PATTERN_RE.findall(body_clean))
        table_hits = len(TABLE_HINT_RE.findall(body_clean))
        visual_density_score = (
            min(3.0, operation_hits / 2.0) + min(2.0, index_hits / 4.0) + min(2.0, table_hits / 2.0)
        )
        pages.append(
            {
                "source": m.group(1).strip(),
                "page": int(m.group(2)),
                "text": body_clean,
                "figure_refs_count": len(figure_refs),
                "has_structure": bool(STRUCTURE_RE.search(body_clean)),
                "has_example": bool(EXAMPLE_RE.search(body_clean)),
                "has_table_hint": bool(TABLE_HINT_RE.search(body_clean)),
                "operation_keyword_hits": int(operation_hits),
                "index_pattern_hits": int(index_hits),
                "visual_density_score": round(float(visual_density_score), 4),
            }
        )
    return pages


def load_page_images(path: Path) -> dict[tuple[str, int], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = Path(str(payload.get("pdf", ""))).name
    out: dict[tuple[str, int], str] = {}
    for item in payload.get("images", []):
        if not isinstance(item, dict):
            continue
        page = int(item.get("page", 0) or 0)
        img_path = str(item.get("path", "")).strip()
        if page < 1 or not img_path:
            continue
        out[(source, page)] = img_path
    return out


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0)
    ub = (bx1 - bx0) * (by1 - by0)
    return float(inter / max(1, ua + ub - inter))


def detect_regions(
    image: Image.Image,
    *,
    binary_threshold: int,
    component_min_area: int,
    min_area_ratio: float,
    min_width_ratio: float,
    min_height_ratio: float,
    max_ink_ratio: float,
    dilation_kernel: int,
    max_regions: int,
    iou_dedupe: float,
) -> list[dict[str, Any]]:
    arr = np.asarray(image.convert("L"))
    h, w = arr.shape

    ink = arr < int(binary_threshold)

    comp_labels, comp_count = ndimage.label(ink)
    filtered = np.zeros_like(ink)
    for i in range(1, comp_count + 1):
        ys, xs = np.where(comp_labels == i)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        if (x1 - x0) * (y1 - y0) < int(component_min_area):
            continue
        filtered[comp_labels == i] = True

    k = max(1, int(dilation_kernel))
    filtered = ndimage.binary_dilation(filtered, structure=np.ones((k, k)), iterations=1)

    labels, count = ndimage.label(filtered)
    candidates: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        ys, xs = np.where(labels == i)
        if ys.size == 0:
            continue

        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        bw = x1 - x0
        bh = y1 - y0
        area_ratio = (bw * bh) / float(max(1, w * h))

        if area_ratio < float(min_area_ratio):
            continue
        if bw < int(float(min_width_ratio) * w) or bh < int(float(min_height_ratio) * h):
            continue

        crop = arr[y0:y1, x0:x1]
        ink_ratio = float((crop < int(binary_threshold)).mean())
        if ink_ratio > float(max_ink_ratio):
            continue

        score = float(area_ratio * (1.0 - min(ink_ratio, 0.95)))
        candidates.append(
            {
                "bbox": (x0, y0, x1, y1),
                "bbox_norm": {
                    "x0": round(x0 / w, 6),
                    "y0": round(y0 / h, 6),
                    "x1": round(x1 / w, 6),
                    "y1": round(y1 / h, 6),
                },
                "area_ratio": round(area_ratio, 6),
                "ink_ratio": round(ink_ratio, 6),
                "score": round(score, 6),
                "is_fallback": False,
                "region_detection_method": "component_detect",
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    for cand in candidates:
        if len(selected) >= int(max_regions):
            break
        box = cand["bbox"]
        if any(_bbox_iou(box, x["bbox"]) >= float(iou_dedupe) for x in selected):
            continue
        selected.append(cand)
    return selected


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    *,
    img_w: int,
    img_h: int,
    factor: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    half_w = max(1.0, ((x1 - x0) / 2.0) * float(factor))
    half_h = max(1.0, ((y1 - y0) / 2.0) * float(factor))
    nx0 = max(0, int(round(cx - half_w)))
    ny0 = max(0, int(round(cy - half_h)))
    nx1 = min(int(img_w), int(round(cx + half_w)))
    ny1 = min(int(img_h), int(round(cy + half_h)))
    if nx1 <= nx0:
        nx1 = min(int(img_w), nx0 + 1)
    if ny1 <= ny0:
        ny1 = min(int(img_h), ny0 + 1)
    return nx0, ny0, nx1, ny1


def stabilize_tiny_regions(
    regions: list[dict[str, Any]],
    *,
    image: Image.Image,
    binary_threshold: int,
    tiny_area_ratio: float,
    tiny_expand_factor: float,
    tiny_max_expand: float,
    iou_dedupe: float,
) -> list[dict[str, Any]]:
    if not regions:
        return regions
    arr = np.asarray(image.convert("L"))
    h, w = arr.shape

    stabilized: list[dict[str, Any]] = []
    for r in regions:
        box = tuple(r.get("bbox", (0, 0, 0, 0)))
        x0, y0, x1, y1 = [int(v) for v in box]
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        area_ratio = (bw * bh) / float(max(1, w * h))
        is_tiny = area_ratio < float(tiny_area_ratio)
        out = dict(r)
        out["quality_flags"] = {"is_tiny_before": bool(is_tiny), "expanded_from_tiny": False}
        if is_tiny:
            scale = max(1.0, float(tiny_expand_factor))
            max_scale = max(scale, float(tiny_max_expand))
            while scale <= max_scale + 1e-9:
                nx0, ny0, nx1, ny1 = _expand_bbox((x0, y0, x1, y1), img_w=w, img_h=h, factor=scale)
                nbw = max(1, nx1 - nx0)
                nbh = max(1, ny1 - ny0)
                n_area_ratio = (nbw * nbh) / float(max(1, w * h))
                if n_area_ratio >= float(tiny_area_ratio):
                    x0, y0, x1, y1 = nx0, ny0, nx1, ny1
                    area_ratio = n_area_ratio
                    out["quality_flags"]["expanded_from_tiny"] = True
                    break
                scale *= 1.15
        crop = arr[y0:y1, x0:x1]
        ink_ratio = float((crop < int(binary_threshold)).mean()) if crop.size else 0.0
        out["bbox"] = (x0, y0, x1, y1)
        out["bbox_norm"] = {
            "x0": round(x0 / w, 6),
            "y0": round(y0 / h, 6),
            "x1": round(x1 / w, 6),
            "y1": round(y1 / h, 6),
        }
        out["area_ratio"] = round(float(area_ratio), 6)
        out["ink_ratio"] = round(float(ink_ratio), 6)
        out["quality_flags"]["is_tiny_after"] = bool(float(area_ratio) < float(tiny_area_ratio))
        stabilized.append(out)

    deduped: list[dict[str, Any]] = []
    for cand in sorted(stabilized, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        box = tuple(cand.get("bbox", (0, 0, 0, 0)))
        if any(_bbox_iou(box, tuple(x.get("bbox", (0, 0, 0, 0)))) >= float(iou_dedupe) for x in deduped):
            continue
        deduped.append(cand)
    return deduped


def fallback_3zone_regions(image: Image.Image) -> list[dict[str, Any]]:
    w, h = image.size
    zones = [
        ("top", (int(0.07 * w), int(0.10 * h), int(0.93 * w), int(0.36 * h))),
        ("mid", (int(0.07 * w), int(0.34 * h), int(0.93 * w), int(0.66 * h))),
        ("bottom", (int(0.07 * w), int(0.62 * h), int(0.93 * w), int(0.92 * h))),
    ]
    out: list[dict[str, Any]] = []
    for idx, (name, bbox) in enumerate(zones, start=1):
        x0, y0, x1, y1 = bbox
        area_ratio = ((x1 - x0) * (y1 - y0)) / float(max(1, w * h))
        out.append(
            {
                "bbox": (x0, y0, x1, y1),
                "bbox_norm": {
                    "x0": round(x0 / w, 6),
                    "y0": round(y0 / h, 6),
                    "x1": round(x1 / w, 6),
                    "y1": round(y1 / h, 6),
                },
                "area_ratio": round(area_ratio, 6),
                "ink_ratio": None,
                "score": round(0.01 - (idx * 0.0001), 6),
                "is_fallback": True,
                "fallback_zone": name,
                "region_detection_method": "fallback_3zone",
                "quality_flags": {"is_tiny_before": False, "expanded_from_tiny": False, "is_tiny_after": False},
            }
        )
    return out


def infer_target_page(
    page_row: dict[str, Any],
    *,
    operation_keyword_min_hits: int,
    visual_density_threshold: float,
) -> tuple[bool, list[str], bool]:
    reasons: list[str] = []
    if int(page_row.get("figure_refs_count", 0) or 0) > 0:
        reasons.append("figure_ref")
    if bool(page_row.get("has_structure", False)):
        reasons.append("structure_block")
    if bool(page_row.get("has_example", False)):
        reasons.append("example_marker")
    if bool(page_row.get("has_table_hint", False)):
        reasons.append("table_hint")

    op_hits = int(page_row.get("operation_keyword_hits", 0) or 0)
    if op_hits >= int(operation_keyword_min_hits):
        reasons.append("operation_keyword")

    density = float(page_row.get("visual_density_score", 0.0) or 0.0)
    if density >= float(visual_density_threshold):
        reasons.append("visual_density")

    target = bool(reasons)
    sequence_candidate = bool(op_hits >= int(operation_keyword_min_hits) or "visual_density" in reasons)
    return target, reasons, sequence_candidate


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract candidate figure regions from page images.")
    ap.add_argument("--markdown", default="final_extracted_content.md")
    ap.add_argument("--figure-manifest", default="logs/figure_manifest_latest.json")
    ap.add_argument("--output-dir", default="assets/figure_regions")
    ap.add_argument("--output-manifest", default="logs/figure_regions_manifest_latest.json")
    ap.add_argument("--max-regions-per-page", type=int, default=4)
    ap.add_argument("--binary-threshold", type=int, default=200)
    ap.add_argument("--component-min-area", type=int, default=60)
    ap.add_argument("--min-area-ratio", type=float, default=0.002)
    ap.add_argument("--min-width-ratio", type=float, default=0.05)
    ap.add_argument("--min-height-ratio", type=float, default=0.025)
    ap.add_argument("--max-ink-ratio", type=float, default=0.28)
    ap.add_argument("--dilation-kernel", type=int, default=5)
    ap.add_argument("--iou-dedupe", type=float, default=0.65)
    ap.add_argument("--tiny-area-ratio", type=float, default=0.012)
    ap.add_argument("--tiny-expand-factor", type=float, default=1.45)
    ap.add_argument("--tiny-max-expand", type=float, default=2.2)
    ap.add_argument("--target-only", action="store_true")
    ap.add_argument("--fallback-when-empty", action="store_true")
    ap.add_argument("--operation-keyword-min-hits", type=int, default=1)
    ap.add_argument("--visual-density-threshold", type=float, default=1.6)
    args = ap.parse_args()

    md_path = Path(args.markdown)
    fig_path = Path(args.figure_manifest)
    out_dir = Path(args.output_dir)
    out_manifest = Path(args.output_manifest)

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown not found: {md_path}")
    if not fig_path.exists():
        raise FileNotFoundError(f"Figure manifest not found: {fig_path}")

    pages = parse_pages(md_path.read_text(encoding="utf-8"))
    image_map = load_page_images(fig_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    targeted_pages = 0
    pages_with_regions = 0
    fallback_pages = 0
    total_regions = 0
    tiny_regions_before = 0
    tiny_regions_after = 0
    target_reason_hist: dict[str, int] = {}

    out_pages: list[dict[str, Any]] = []
    for p in pages:
        total_pages += 1
        source = p["source"]
        page = int(p["page"])
        target, target_reasons, sequence_candidate = infer_target_page(
            p,
            operation_keyword_min_hits=max(1, int(args.operation_keyword_min_hits)),
            visual_density_threshold=float(args.visual_density_threshold),
        )
        if args.target_only and not target:
            continue
        if target:
            targeted_pages += 1
            for r in target_reasons:
                target_reason_hist[r] = int(target_reason_hist.get(r, 0)) + 1

        key = (source, page)
        img_path = image_map.get(key)
        if not img_path:
            continue

        img_file = Path(img_path)
        if not img_file.exists():
            continue

        image = Image.open(img_file).convert("RGB")
        regions = detect_regions(
            image,
            binary_threshold=args.binary_threshold,
            component_min_area=args.component_min_area,
            min_area_ratio=args.min_area_ratio,
            min_width_ratio=args.min_width_ratio,
            min_height_ratio=args.min_height_ratio,
            max_ink_ratio=args.max_ink_ratio,
            dilation_kernel=args.dilation_kernel,
            max_regions=args.max_regions_per_page,
            iou_dedupe=args.iou_dedupe,
        )
        tiny_regions_before += sum(1 for r in regions if float(r.get("area_ratio", 0.0) or 0.0) < float(args.tiny_area_ratio))
        if regions:
            regions = stabilize_tiny_regions(
                regions,
                image=image,
                binary_threshold=args.binary_threshold,
                tiny_area_ratio=float(args.tiny_area_ratio),
                tiny_expand_factor=float(args.tiny_expand_factor),
                tiny_max_expand=float(args.tiny_max_expand),
                iou_dedupe=float(args.iou_dedupe),
            )
        tiny_regions_after += sum(1 for r in regions if float(r.get("area_ratio", 0.0) or 0.0) < float(args.tiny_area_ratio))

        if not regions and args.fallback_when_empty and target:
            regions = fallback_3zone_regions(image)
            fallback_pages += 1

        out_regions: list[dict[str, Any]] = []
        for ridx, region in enumerate(regions, start=1):
            x0, y0, x1, y1 = region["bbox"]
            crop = image.crop((x0, y0, x1, y1))
            fname = f"page_{page:03d}_region_{ridx:02d}.png"
            r_path = out_dir / fname
            crop.save(r_path)

            out_regions.append(
                {
                    "region_index": ridx,
                    "path": str(r_path.as_posix()),
                    "bbox": {
                        "x0": int(x0),
                        "y0": int(y0),
                        "x1": int(x1),
                        "y1": int(y1),
                    },
                    "bbox_norm": region["bbox_norm"],
                    "area_ratio": region["area_ratio"],
                    "ink_ratio": region["ink_ratio"],
                    "score": region["score"],
                    "is_fallback": bool(region["is_fallback"]),
                    "region_detection_method": str(region.get("region_detection_method", "component_detect")),
                    "fallback_zone": str(region.get("fallback_zone", "")),
                    "quality_flags": region.get("quality_flags", {}),
                }
            )

        if out_regions:
            pages_with_regions += 1
            total_regions += len(out_regions)

        out_pages.append(
            {
                "source": source,
                "page": page,
                "image_path": str(img_file.as_posix()),
                "target_page": target,
                "target_reason": target_reasons,
                "sequence_candidate": bool(sequence_candidate),
                "region_detection_method": (
                    "component_detect" if any(not bool(r.get("is_fallback", False)) for r in out_regions) else "fallback_3zone"
                ),
                "linkage_flags": {
                    "has_figure_refs": bool(int(p.get("figure_refs_count", 0) or 0) > 0),
                    "has_structure": bool(p.get("has_structure", False)),
                    "has_example": bool(p.get("has_example", False)),
                    "has_table_hint": bool(p.get("has_table_hint", False)),
                    "operation_keyword_hits": int(p.get("operation_keyword_hits", 0) or 0),
                    "index_pattern_hits": int(p.get("index_pattern_hits", 0) or 0),
                    "visual_density_score": float(p.get("visual_density_score", 0.0) or 0.0),
                },
                "figure_refs_count": int(p["figure_refs_count"]),
                "has_structure": bool(p["has_structure"]),
                "has_example": bool(p["has_example"]),
                "regions": out_regions,
            }
        )

    ratio_base = max(1, targeted_pages if args.target_only else total_pages)
    summary = {
        "total_pages": total_pages,
        "targeted_pages": targeted_pages,
        "pages_with_regions": pages_with_regions,
        "pages_with_regions_ratio": round(pages_with_regions / ratio_base, 6),
        "total_regions": total_regions,
        "avg_regions_per_page_with_regions": round(total_regions / max(1, pages_with_regions), 6),
        "fallback_pages": fallback_pages,
        "tiny_regions_before": int(tiny_regions_before),
        "tiny_regions_after": int(tiny_regions_after),
        "target_reason_hist": target_reason_hist,
    }

    payload = {
        "output_dir": str(out_dir.as_posix()),
        "source_manifest": str(fig_path.as_posix()),
        "summary": summary,
        "pages": out_pages,
    }
    out_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_manifest={out_manifest}")


if __name__ == "__main__":
    main()
