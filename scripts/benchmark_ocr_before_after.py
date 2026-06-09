"""
Compare OCR quality (CER/WER) before vs after using a GT JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PAGE_HEADER_RE = re.compile(r"# Source:\s*(.*?)\n## Page\s*(\d+)\n\n", re.MULTILINE)
ANCHOR_BLOCK_RE = re.compile(r"\n### Visual Anchors \(Auto\)[\s\S]*$", re.MULTILINE)
VISUAL_CAPTION_BLOCK_RE = re.compile(r"\n### Visual Captions \(Auto\)\n[\s\S]*$", re.MULTILINE)


def parse_pages(md_text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    headers = list(PAGE_HEADER_RE.finditer(md_text))
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        body = md_text[start:end].strip()
        body = ANCHOR_BLOCK_RE.sub("", body).strip()
        body = VISUAL_CAPTION_BLOCK_RE.sub("", body).strip()
        lines = []
        for ln in body.splitlines():
            t = ln.strip()
            if not t:
                continue
            t = t.lstrip("#").strip()
            lines.append(t)
        out[int(m.group(2))] = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return out


def edit_distance(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def cer_wer(pred: str, gt: str) -> tuple[float, float]:
    gt_chars = list(gt or "")
    pred_chars = list(pred or "")
    cer = edit_distance(pred_chars, gt_chars) / max(1, len(gt_chars))

    gt_words = [w for w in re.split(r"\s+", (gt or "").strip()) if w]
    pred_words = [w for w in re.split(r"\s+", (pred or "").strip()) if w]
    wer = edit_distance(pred_words, gt_words) / max(1, len(gt_words))
    return float(cer), float(wer)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"samples": 0}
    cer_vals = [float(r["cer"]) for r in rows]
    wer_vals = [float(r["wer"]) for r in rows]
    cer_vals_sorted = sorted(cer_vals)
    wer_vals_sorted = sorted(wer_vals)
    idx95 = int(0.95 * (len(rows) - 1))
    return {
        "samples": len(rows),
        "cer_mean": round(sum(cer_vals) / len(cer_vals), 6),
        "wer_mean": round(sum(wer_vals) / len(wer_vals), 6),
        "cer_p95": round(cer_vals_sorted[idx95], 6),
        "wer_p95": round(wer_vals_sorted[idx95], 6),
    }


def run_eval(
    markdown_path: Path,
    gt_rows: list[dict],
    *,
    manual_only: bool = False,
) -> tuple[list[dict], dict]:
    page_map = parse_pages(markdown_path.read_text(encoding="utf-8"))
    rows = []
    for g in gt_rows:
        if manual_only and (not bool(g.get("manual_verified", False))):
            continue
        page = int(g.get("page", 0) or 0)
        gt_text = str(g.get("text", "") or "")
        if page <= 0 or not gt_text:
            continue
        pred = str(page_map.get(page, "") or "")
        c, w = cer_wer(pred, gt_text)
        rows.append(
            {
                "page": page,
                "split": str(g.get("split", "")),
                "cer": c,
                "wer": w,
                "gt_chars": len(gt_text),
                "pred_chars": len(pred),
            }
        )
    return rows, summarize(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark OCR before/after against GT JSONL.")
    ap.add_argument("--gt-jsonl", default="eval/ocr_gt_first_mid_last_v1.jsonl")
    ap.add_argument("--before", default="logs/final_extracted_backup_before_rescan_20260213_193549.md")
    ap.add_argument("--after", default="final_extracted_text_only.md")
    ap.add_argument("--cer-threshold", type=float, default=0.05)
    ap.add_argument("--wer-threshold", type=float, default=0.2)
    ap.add_argument(
        "--strict-mode",
        choices=["internal", "external_publish_grade"],
        default="external_publish_grade",
        help="external_publish_grade: evaluate only manual_verified=true rows.",
    )
    ap.add_argument("--output-json", default="logs/ocr_before_after_latest.json")
    ap.add_argument("--output-csv", default="logs/ocr_before_after_latest.csv")
    args = ap.parse_args()

    gt_path = Path(args.gt_jsonl)
    if not gt_path.exists():
        raise FileNotFoundError(f"missing gt jsonl: {gt_path}")

    gt_rows = []
    invalid_rows = 0
    for ln in gt_path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            gt_rows.append(json.loads(s))
        except Exception:
            invalid_rows += 1
            continue

    manual_only = str(args.strict_mode or "external_publish_grade").strip() == "external_publish_grade"
    before_rows, before_sum = run_eval(Path(args.before), gt_rows, manual_only=manual_only)
    after_rows, after_sum = run_eval(Path(args.after), gt_rows, manual_only=manual_only)

    before_pass = (
        before_sum.get("samples", 0) > 0
        and float(before_sum.get("cer_mean", 1.0)) <= float(args.cer_threshold)
        and float(before_sum.get("wer_mean", 1.0)) <= float(args.wer_threshold)
    )
    after_pass = (
        after_sum.get("samples", 0) > 0
        and float(after_sum.get("cer_mean", 1.0)) <= float(args.cer_threshold)
        and float(after_sum.get("wer_mean", 1.0)) <= float(args.wer_threshold)
    )

    def _delta(k: str) -> float | None:
        if k not in before_sum or k not in after_sum:
            return None
        return round(float(after_sum[k]) - float(before_sum[k]), 6)

    payload = {
        "config": {
            "gt_jsonl": str(gt_path.as_posix()),
            "before": str(Path(args.before).as_posix()),
            "after": str(Path(args.after).as_posix()),
            "cer_threshold": float(args.cer_threshold),
            "wer_threshold": float(args.wer_threshold),
            "strict_mode": str(args.strict_mode),
            "manual_verified_only": bool(manual_only),
            "invalid_gt_rows_skipped": int(invalid_rows),
        },
        "before": {**before_sum, "pass": before_pass},
        "after": {**after_sum, "pass": after_pass},
        "delta_after_minus_before": {
            "cer_mean": _delta("cer_mean"),
            "wer_mean": _delta("wer_mean"),
            "cer_p95": _delta("cer_p95"),
            "wer_p95": _delta("wer_p95"),
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = Path(args.output_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["version", "page", "split", "cer", "wer", "gt_chars", "pred_chars"])
        w.writeheader()
        for r in before_rows:
            w.writerow({"version": "before", **r})
        for r in after_rows:
            w.writerow({"version": "after", **r})

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved_json={out_json}")
    print(f"saved_csv={out_csv}")


if __name__ == "__main__":
    main()
