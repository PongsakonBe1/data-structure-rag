"""
Health-check utility for visual retrieval endpoint contract.

Contract request:
{
  "query": "...",
  "candidates": [{"id","page_id","source","page","image_path","text_preview"}]
}

Contract response:
{
  "scores": [{"id":"...","score":0.0}, ...]
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import base64
from pathlib import Path

import httpx
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def build_payload(query: str) -> dict:
    # Prefer a real local region image if available; fallback to 1x1 pixel.
    img_path = Path("assets/figure_regions/page_031_region_01.png")
    if img_path.exists():
        img_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    else:
        img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7f0l8AAAAASUVORK5CYII="
    return {
        "query": query,
        "candidates": [
            {
                "id": "doc.pdf:31:region:1",
                "page_id": "doc.pdf:31",
                "source": "doc.pdf",
                "page": 31,
                "image_path": "assets/figure_regions/page_031_region_01.png",
                "image_base64": img_b64,
                "text_preview": "Figure 3.4 Queue enqueue/dequeue example",
            }
        ],
    }


def validate_response(data: dict) -> tuple[bool, str]:
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
    for i, item in enumerate(scores):
        if not isinstance(item, dict):
            return False, f"scores[{i}]_not_object"
        if "id" not in item or "score" not in item:
            return False, f"scores[{i}]_missing_id_or_score"
        try:
            _ = float(item.get("score"))
        except Exception:
            return False, f"scores[{i}]_score_not_numeric"
    return True, "ok"


def extract_scores_payload(data_obj):
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
            parsed = extract_scores_payload(item)
            if isinstance(parsed, dict):
                return parsed
    return None


def summarize_latency(values_ms: list[float]) -> dict:
    vals = [float(v) for v in values_ms if isinstance(v, (int, float))]
    if not vals:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None, "cv": None}
    vals = sorted(vals)
    n = len(vals)
    mean_v = sum(vals) / n
    med_v = vals[n // 2] if n % 2 == 1 else (vals[(n // 2) - 1] + vals[n // 2]) / 2.0
    p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    p95_v = vals[p95_idx]
    variance = sum((x - mean_v) ** 2 for x in vals) / max(1, n)
    std_v = variance**0.5
    cv = (std_v / mean_v) if mean_v > 0 else 0.0
    return {
        "mean_ms": round(mean_v, 2),
        "median_ms": round(med_v, 2),
        "p95_ms": round(p95_v, 2),
        "cv": round(cv, 4),
    }


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Check visual endpoint contract readiness.")
    ap.add_argument("--endpoint", default=os.getenv("COLPALI_ENDPOINT_URL", "").strip())
    ap.add_argument("--token", default=(os.getenv("HUGGINGFACE_READ_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip())
    ap.add_argument("--query", default="โครงสร้างของการแทนคิวด้วยอาร์เรย์")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--benchmark-runs", type=int, default=3)
    ap.add_argument("--warmup-runs", type=int, default=2)
    ap.add_argument("--require-phase2", action="store_true")
    ap.add_argument("--output", default="logs/visual_endpoint_healthcheck_latest.json")
    args = ap.parse_args()

    if not args.endpoint:
        raise SystemExit("missing --endpoint and COLPALI_ENDPOINT_URL")

    payload = build_payload(args.query)
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    started = time.perf_counter()
    status = None
    body = None
    data = None
    err = ""
    phase2 = {
        "supported": False,
        "register_and_score_ok": False,
        "register_ok": False,
        "score_cached_ok": False,
        "score_cached_error": "",
        "stateless_fallback_ok": False,
        "mode": "",
        "corpus_id": "",
        "latency": {
            "legacy_score": {"mean_ms": None, "median_ms": None, "p95_ms": None, "cv": None},
            "cached_score": {"mean_ms": None, "median_ms": None, "p95_ms": None, "cv": None},
            "speedup_ratio": None,
            "stable": None,
            "warmup_runs": max(0, int(args.warmup_runs)),
            "runs": max(1, int(args.benchmark_runs)),
        },
    }

    endpoint = (args.endpoint or "").strip()
    is_http_endpoint = endpoint.lower().startswith(("http://", "https://"))

    # Try direct contract endpoint first (only for raw HTTP endpoints)
    if is_http_endpoint:
        try:
            with httpx.Client(timeout=max(5, int(args.timeout))) as client:
                res = client.post(endpoint, json=payload, headers=headers)
                status = int(res.status_code)
                body = (res.text or "")[:2000]
                if status < 400:
                    data = res.json()
                else:
                    err = f"http_error:{status}:{body[:240]}"
        except Exception as exc:
            err = f"request_failed:{exc}"

    # Fallback: try Gradio API names
    if data is None:
        try:
            from gradio_client import Client

            client = Client(endpoint, hf_token=args.token or None, verbose=False)
            # Phase-2 preferred path: register corpus once then score by ids.
            try:
                cand0 = (payload.get("candidates", [{}])[0] or {})
                cid = str(cand0.get("id", "doc.pdf:31:region:1")).strip() or "doc.pdf:31:region:1"
                corpus_id = "healthcheck-corpus-v1"
                phase2["corpus_id"] = corpus_id
                one_payload = {
                    "query": payload.get("query", ""),
                    "corpus_id": corpus_id,
                    "candidate_ids": [cid],
                    "candidates": payload.get("candidates", []),
                }
                try:
                    one_out = client.predict(one_payload, api_name="/register_and_score")
                    one = extract_scores_payload(one_out) or {}
                    one_ok, _ = validate_response(one if isinstance(one, dict) else {})
                    if one_ok:
                        phase2["supported"] = True
                        phase2["register_and_score_ok"] = True
                        phase2["mode"] = "register_and_score"
                        data = one
                        err = ""
                        n_runs = max(1, int(args.benchmark_runs))
                        warmups = max(0, int(args.warmup_runs))
                        legacy_times = []
                        cached_times = []
                        for _ in range(warmups):
                            _ = client.predict(payload, api_name="/score")
                        for _ in range(warmups):
                            _ = client.predict(one_payload, api_name="/register_and_score")
                        for _ in range(n_runs):
                            t0 = time.perf_counter()
                            _ = client.predict(payload, api_name="/score")
                            legacy_times.append((time.perf_counter() - t0) * 1000.0)
                        for _ in range(n_runs):
                            t0 = time.perf_counter()
                            _ = client.predict(one_payload, api_name="/register_and_score")
                            cached_times.append((time.perf_counter() - t0) * 1000.0)
                        phase2["latency"]["legacy_score"] = summarize_latency(legacy_times)
                        phase2["latency"]["cached_score"] = summarize_latency(cached_times)
                        leg = phase2["latency"]["legacy_score"]["mean_ms"]
                        cch = phase2["latency"]["cached_score"]["mean_ms"]
                        lcv = phase2["latency"]["legacy_score"]["cv"]
                        ccv = phase2["latency"]["cached_score"]["cv"]
                        if isinstance(leg, (int, float)) and isinstance(cch, (int, float)) and cch > 0:
                            phase2["latency"]["speedup_ratio"] = round(float(leg) / float(cch), 3)
                        if isinstance(lcv, (int, float)) and isinstance(ccv, (int, float)):
                            phase2["latency"]["stable"] = bool((lcv <= 0.20) and (ccv <= 0.20))
                except Exception:
                    pass

                if data is None:
                    reg_out = client.predict({"corpus_id": corpus_id, "candidates": payload.get("candidates", [])}, api_name="/register_corpus")
                    reg = extract_scores_payload(reg_out) or {}
                    if isinstance(reg, dict) and bool(reg.get("ok", False)):
                        phase2["supported"] = True
                        phase2["register_ok"] = True
                        score_cached_payload = {
                            "query": payload.get("query", ""),
                            "corpus_id": corpus_id,
                            "candidate_ids": [cid],
                        }
                        cached_out = client.predict(score_cached_payload, api_name="/score_cached")
                        cached = extract_scores_payload(cached_out) or {}
                        if isinstance(cached, dict) and str(cached.get("error", "")).strip() == "unknown_corpus_id":
                            # Some Space deployments can route calls to different workers.
                            # Retry with candidates so the worker can hydrate corpus on-demand.
                            score_cached_payload["candidates"] = payload.get("candidates", [])
                            cached_out = client.predict(score_cached_payload, api_name="/score_cached")
                            cached = extract_scores_payload(cached_out) or {}
                        if isinstance(cached, dict):
                            phase2["score_cached_error"] = str(cached.get("error", "") or "")
                        c_ok, _ = validate_response(cached if isinstance(cached, dict) else {})
                        if c_ok:
                            phase2["score_cached_ok"] = True
                            phase2["mode"] = "register_corpus+score_cached"
                            data = cached
                            err = ""
                            # Phase-2 micro benchmark: /score vs /score_cached
                            n_runs = max(1, int(args.benchmark_runs))
                            warmups = max(0, int(args.warmup_runs))
                            legacy_times = []
                            cached_times = []
                            for _ in range(warmups):
                                _ = client.predict(payload, api_name="/score")
                            for _ in range(warmups):
                                _ = client.predict(
                                    {
                                        "query": payload.get("query", ""),
                                        "corpus_id": corpus_id,
                                        "candidate_ids": [cid],
                                        "candidates": payload.get("candidates", []),
                                    },
                                    api_name="/score_cached",
                                )
                            for _ in range(n_runs):
                                t0 = time.perf_counter()
                                _ = client.predict(payload, api_name="/score")
                                legacy_times.append((time.perf_counter() - t0) * 1000.0)
                            for _ in range(n_runs):
                                t0 = time.perf_counter()
                                _ = client.predict(
                                    {
                                        "query": payload.get("query", ""),
                                        "corpus_id": corpus_id,
                                        "candidate_ids": [cid],
                                        "candidates": payload.get("candidates", []),
                                    },
                                    api_name="/score_cached",
                                )
                                cached_times.append((time.perf_counter() - t0) * 1000.0)
                            phase2["latency"]["legacy_score"] = summarize_latency(legacy_times)
                            phase2["latency"]["cached_score"] = summarize_latency(cached_times)
                            leg = phase2["latency"]["legacy_score"]["mean_ms"]
                            cch = phase2["latency"]["cached_score"]["mean_ms"]
                            lcv = phase2["latency"]["legacy_score"]["cv"]
                            ccv = phase2["latency"]["cached_score"]["cv"]
                            if isinstance(leg, (int, float)) and isinstance(cch, (int, float)) and cch > 0:
                                phase2["latency"]["speedup_ratio"] = round(float(leg) / float(cch), 3)
                            if isinstance(lcv, (int, float)) and isinstance(ccv, (int, float)):
                                phase2["latency"]["stable"] = bool((lcv <= 0.20) and (ccv <= 0.20))
                        elif str(phase2.get("score_cached_error", "")).strip() == "unknown_corpus_id":
                            # Graceful fallback mode for stateless Spaces:
                            # APIs exist, but cached lookup cannot see registered corpus across calls.
                            out = client.predict(payload, api_name="/score")
                            parsed = extract_scores_payload(out)
                            p_ok, _ = validate_response(parsed if isinstance(parsed, dict) else {})
                            if p_ok:
                                phase2["stateless_fallback_ok"] = True
                                phase2["mode"] = "stateless_fallback_score"
                                data = parsed
                                err = ""
            except Exception:
                pass

            for api_name in ("/score", "/api_endpoint"):
                if data is not None:
                    break
                try:
                    out = client.predict(payload, api_name=api_name)
                    parsed = extract_scores_payload(out)
                    if isinstance(parsed, dict):
                        data = parsed
                        err = ""
                        break
                except Exception:
                    try:
                        out = client.predict(json.dumps(payload, ensure_ascii=False), api_name=api_name)
                        parsed = extract_scores_payload(out)
                        if isinstance(parsed, dict):
                            data = parsed
                            err = ""
                            break
                    except Exception:
                        continue
        except Exception as exc:
            if err:
                err = f"{err} | gradio_client:{exc}"
            else:
                err = f"gradio_client:{exc}"

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    ok = False
    validation = "not_validated"
    response_preview = []
    if err and data is None:
        ok = False
        validation = err
    elif isinstance(data, dict):
        ok, validation = validate_response(data)
        response_preview = (data.get("scores") or [])[:3]
    phase2_ready = bool(phase2.get("register_and_score_ok")) or (
        bool(phase2.get("register_ok")) and bool(
            phase2.get("score_cached_ok") or phase2.get("stateless_fallback_ok")
        )
    )
    if args.require_phase2 and not phase2_ready:
        ok = False
        validation = f"phase2_required_but_not_ready:{validation}"
    elif args.require_phase2 and phase2.get("register_and_score_ok"):
        validation = "ok_phase2_register_and_score"
    elif args.require_phase2 and phase2.get("stateless_fallback_ok") and not phase2.get("score_cached_ok"):
        validation = "ok_phase2_stateless_fallback"

    # Backward-compatible flattened fields.
    try:
        phase2["latency"]["legacy_score_mean_ms"] = (phase2.get("latency", {}).get("legacy_score", {}) or {}).get("mean_ms")
        phase2["latency"]["cached_score_mean_ms"] = (phase2.get("latency", {}).get("cached_score", {}) or {}).get("mean_ms")
    except Exception:
        pass

    report = {
        "ok": bool(ok),
        "endpoint_url": args.endpoint,
        "status_code": status,
        "latency_ms": elapsed_ms,
        "validation": validation,
        "phase2": phase2,
        "response_preview": response_preview,
        "response_text": body,
        "contract": {
            "query": payload.get("query", ""),
            "candidates": [
                {
                    **{k: v for k, v in (payload.get("candidates", [{}])[0] or {}).items() if k != "image_base64"},
                    "image_base64": "<omitted>" if (payload.get("candidates", [{}])[0] or {}).get("image_base64") else "",
                }
            ],
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
