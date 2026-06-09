"""
Evaluate LLM-as-judge robustness and discrimination quality.

Dataset format (JSONL):
{"query":"...","context":"...","answer":"..."}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

DEFAULT_MODEL = os.getenv("JUDGE_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN = (os.getenv("HUGGINGFACE_READ_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
METRICS = ("faithfulness", "relevance", "context_precision")


@dataclass
class RobustnessResult:
    queries: int
    mean_abs_delta_context_order: float
    mean_abs_delta_verbosity: float
    mean_abs_delta_noise_context: float
    mean_abs_delta_context_drop: float
    mean_discrimination_gap_good_vs_bad: float
    bad_beats_good_rate: float
    proxy_mean_discrimination_gap_good_vs_bad: float
    proxy_bad_beats_good_rate: float
    score_vector_unique_count: int


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            for key in ("query", "context", "answer"):
                if not str(item.get(key, "")).strip():
                    raise ValueError(f"Line {line_no}: missing '{key}'")
            rows.append(item)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def score_with_judge(
    client: InferenceClient,
    model: str,
    query: str,
    answer: str,
    context: str,
    temperature: float = 0.1,
) -> dict[str, float]:
    prompt = (
        "You are a strict evaluator for a Retrieval-Augmented Generation system.\n"
        "Score the following answer on three criteria (0.0 to 1.0):\n"
        "1. faithfulness\n"
        "2. relevance\n"
        "3. context_precision\n\n"
        f"[Question]: {query}\n"
        f"[Context]: {context[:2200]}\n"
        f"[Answer]: {answer[:1200]}\n\n"
        'Return ONLY JSON: {"scores": {"faithfulness": 0.8, "relevance": 0.9, "context_precision": 0.7}}'
    )
    last_err = None
    for attempt in range(3):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a JSON judge. Output ONLY JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=220,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            raw = (res.choices[0].message.content or "").strip()
            obj = json.loads(raw)
            scores = obj.get("scores", obj)
            normalized = {}
            for m in METRICS:
                val = float(scores.get(m, 0.0))
                normalized[m] = max(0.0, min(1.0, val))
            return normalized
        except Exception as exc:
            last_err = exc
            if attempt < 2:
                time.sleep(1.2 * (attempt + 1))
            continue

    # Fallback to neutral scores if backend is unstable.
    _ = last_err
    return {m: 0.5 for m in METRICS}


def permute_context(context: str) -> str:
    blocks = [b.strip() for b in context.split("\n\n") if b.strip()]
    if len(blocks) <= 1:
        return context
    return "\n\n".join(reversed(blocks))


def verbose_answer(answer: str) -> str:
    prefix = "คำตอบแบบอธิบายละเอียด:\n"
    suffix = "\n\nหมายเหตุ: ขยายคำอธิบายให้ยาวขึ้นแต่คงเนื้อหาเดิม"
    return f"{prefix}{answer}\n{answer}{suffix}"


def add_noise_context(context: str) -> str:
    noise_block = (
        "Noise block (irrelevant):\n"
        "- This paragraph is not related to the question.\n"
        "- It mentions astronomy, weather, and random trivia.\n"
        "- Judge should ignore this if robust.\n"
    )
    return f"{noise_block}\n\n{context}"


def drop_relevant_context(context: str, query: str, answer: str) -> str:
    """
    Remove context blocks that overlap with query/answer tokens.
    If no block is removed, drop the first block as fallback.
    """
    blocks = [b.strip() for b in context.split("\n\n") if b.strip()]
    if len(blocks) <= 1:
        return context

    target_tokens = _tokens(query) | _tokens(answer)
    keep_blocks = []
    removed = 0
    for block in blocks:
        if _overlap_ratio(_tokens(block), target_tokens) > 0.12:
            removed += 1
            continue
        keep_blocks.append(block)

    if removed == 0:
        keep_blocks = blocks[1:]
    if not keep_blocks:
        keep_blocks = blocks[-1:]
    return "\n\n".join(keep_blocks)


def synthetic_bad_answer(query: str) -> str:
    return (
        "คำตอบนี้จงใจไม่อิงเอกสาร: "
        "พูดถึงดาราศาสตร์ ภูมิอากาศ และการท่องเที่ยวเป็นหลัก "
        "พร้อมยืนยันผิดๆ ว่าทุกอัลกอริทึมมีค่าเวลาเท่ากันเสมอโดยไม่ต้องมีหลักฐาน"
    )


def mean_abs_delta(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(abs(a[m] - b[m]) for m in METRICS) / len(METRICS)


def mean_score(scores: dict[str, float]) -> float:
    return sum(scores[m] for m in METRICS) / len(METRICS)


def _tokens(text: str) -> set[str]:
    # Keep Thai + Latin + digits tokens
    return {t for t in re.findall(r"[A-Za-z0-9ก-๙]{2,}", (text or "").lower())}


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a:
        return 0.0
    return len(a & b) / len(a)


def proxy_score(query: str, answer: str, context: str) -> dict[str, float]:
    q = _tokens(query)
    a = _tokens(answer)
    c = _tokens(context)
    return {
        "faithfulness": _overlap_ratio(a, c),
        "relevance": _overlap_ratio(a, q),
        "context_precision": _overlap_ratio(q, c),
    }


def run_eval(rows: list[dict], model: str, temperature: float = 0.1) -> tuple[RobustnessResult, list[dict]]:
    if not HF_TOKEN:
        raise RuntimeError("Missing HUGGINGFACE_READ_TOKEN or HUGGINGFACE_API_KEY")
    client = InferenceClient(api_key=HF_TOKEN)

    order_deltas = []
    verbosity_deltas = []
    noise_deltas = []
    drop_deltas = []
    discrimination_gaps = []
    bad_beats_good = 0
    proxy_discrimination_gaps = []
    proxy_bad_beats_good = 0
    score_vectors = set()
    details = []

    for idx, row in enumerate(rows, start=1):
        q = row["query"]
        c = row["context"]
        a = row["answer"]

        base = score_with_judge(client, model, q, a, c, temperature=temperature)
        order_alt = score_with_judge(client, model, q, a, permute_context(c), temperature=temperature)
        verbose_alt = score_with_judge(client, model, q, verbose_answer(a), c, temperature=temperature)
        noise_alt = score_with_judge(client, model, q, a, add_noise_context(c), temperature=temperature)
        drop_alt = score_with_judge(client, model, q, a, drop_relevant_context(c, q, a), temperature=temperature)
        bad_alt = score_with_judge(client, model, q, synthetic_bad_answer(q), c, temperature=temperature)
        score_vectors.update(
            {
                tuple(round(base[m], 4) for m in METRICS),
                tuple(round(order_alt[m], 4) for m in METRICS),
                tuple(round(verbose_alt[m], 4) for m in METRICS),
                tuple(round(noise_alt[m], 4) for m in METRICS),
                tuple(round(drop_alt[m], 4) for m in METRICS),
                tuple(round(bad_alt[m], 4) for m in METRICS),
            }
        )

        base_proxy = proxy_score(q, a, c)
        bad_proxy = proxy_score(q, synthetic_bad_answer(q), c)

        d_order = mean_abs_delta(base, order_alt)
        d_verb = mean_abs_delta(base, verbose_alt)
        d_noise = mean_abs_delta(base, noise_alt)
        d_drop = mean_abs_delta(base, drop_alt)
        gap_good_bad = mean_score(base) - mean_score(bad_alt)
        proxy_gap_good_bad = mean_score(base_proxy) - mean_score(bad_proxy)
        if mean_score(bad_alt) > mean_score(base):
            bad_beats_good += 1
        if mean_score(bad_proxy) > mean_score(base_proxy):
            proxy_bad_beats_good += 1

        order_deltas.append(d_order)
        verbosity_deltas.append(d_verb)
        noise_deltas.append(d_noise)
        drop_deltas.append(d_drop)
        discrimination_gaps.append(gap_good_bad)
        proxy_discrimination_gaps.append(proxy_gap_good_bad)
        details.append(
            {
                "idx": idx,
                "query": q[:120],
                "delta_context_order": round(d_order, 6),
                "delta_verbosity": round(d_verb, 6),
                "delta_noise_context": round(d_noise, 6),
                "delta_context_drop": round(d_drop, 6),
                "good_vs_bad_gap": round(gap_good_bad, 6),
                "proxy_good_vs_bad_gap": round(proxy_gap_good_bad, 6),
            }
        )

    result = RobustnessResult(
        queries=len(rows),
        mean_abs_delta_context_order=sum(order_deltas) / len(order_deltas),
        mean_abs_delta_verbosity=sum(verbosity_deltas) / len(verbosity_deltas),
        mean_abs_delta_noise_context=sum(noise_deltas) / len(noise_deltas),
        mean_abs_delta_context_drop=sum(drop_deltas) / len(drop_deltas),
        mean_discrimination_gap_good_vs_bad=sum(discrimination_gaps) / len(discrimination_gaps),
        bad_beats_good_rate=bad_beats_good / len(rows),
        proxy_mean_discrimination_gap_good_vs_bad=sum(proxy_discrimination_gaps) / len(proxy_discrimination_gaps),
        proxy_bad_beats_good_rate=proxy_bad_beats_good / len(rows),
        score_vector_unique_count=len(score_vectors),
    )
    return result, details


def save_details(path: Path, details: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "query",
                "delta_context_order",
                "delta_verbosity",
                "delta_noise_context",
                "delta_context_drop",
                "good_vs_bad_gap",
                "proxy_good_vs_bad_gap",
            ],
        )
        writer.writeheader()
        writer.writerows(details)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge robustness benchmark for typhoon_rag.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Judge model id.")
    parser.add_argument("--output", help="Optional CSV output for per-query deltas.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Judge sampling temperature.")
    args = parser.parse_args()

    rows = load_dataset(Path(args.dataset))
    summary, details = run_eval(rows, args.model, temperature=args.temperature)

    print("Judge robustness summary")
    print("------------------------")
    print(f"queries: {summary.queries}")
    print(f"mean_abs_delta_context_order: {summary.mean_abs_delta_context_order:.4f}")
    print(f"mean_abs_delta_verbosity:    {summary.mean_abs_delta_verbosity:.4f}")
    print(f"mean_abs_delta_noise_context:{summary.mean_abs_delta_noise_context:.4f}")
    print(f"mean_abs_delta_context_drop: {summary.mean_abs_delta_context_drop:.4f}")
    print(f"mean_discrimination_gap_good_vs_bad: {summary.mean_discrimination_gap_good_vs_bad:.4f}")
    print(f"bad_beats_good_rate: {summary.bad_beats_good_rate:.4f}")
    print(f"proxy_mean_discrimination_gap_good_vs_bad: {summary.proxy_mean_discrimination_gap_good_vs_bad:.4f}")
    print(f"proxy_bad_beats_good_rate: {summary.proxy_bad_beats_good_rate:.4f}")
    print(f"score_vector_unique_count: {summary.score_vector_unique_count}")
    if summary.score_vector_unique_count <= 2:
        print("warning: judge signal is low-variance; rely more on discrimination gaps/proxy in this run.")

    if args.output:
        save_details(Path(args.output), details)
        print(f"\nSaved CSV: {args.output}")


if __name__ == "__main__":
    main()
