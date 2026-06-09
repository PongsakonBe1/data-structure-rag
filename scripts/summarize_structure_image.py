"""
Summarize a retrieved structure/example image using a vision model.

Input:
- image path from ColPali search output
- optional user question/context

Output:
- JSON summary and plain Thai explanation
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def image_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = path.suffix.lower().lstrip(".") or "png"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize structure image with Qwen2.5-VL.")
    ap.add_argument("--image", required=True, help="Path to image file.")
    ap.add_argument("--question", default="", help="Optional user question to focus the summary.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    load_dotenv()
    hf_token = (os.getenv("HUGGINGFACE_READ_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
    if not hf_token:
        raise RuntimeError("HUGGINGFACE_READ_TOKEN or HUGGINGFACE_API_KEY not found in environment/.env")

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    prompt = (
        "วิเคราะห์รูปโครงสร้างข้อมูลหรือรูปตัวอย่างจากตำรา แล้วตอบเป็น JSON เท่านั้นในรูปแบบ:\n"
        "{\n"
        "  \"structure_type\": \"...\",\n"
        "  \"what_it_shows\": \"...\",\n"
        "  \"key_components\": [\"...\"],\n"
        "  \"operations_or_flow\": [\"...\"],\n"
        "  \"limitations\": \"...\"\n"
        "}\n"
        "ข้อกำหนด: \n"
        "- ยึดเฉพาะข้อมูลที่เห็นในภาพ\n"
        "- ถ้าอ่านค่าไม่ได้บางส่วนให้ระบุว่าไม่ชัดเจน\n"
        "- เขียนภาษาไทย\n"
    )
    if args.question.strip():
        prompt += f"\nคำถามโฟกัสจากผู้ใช้: {args.question.strip()}\n"

    client = InferenceClient(api_key=hf_token)
    data_url = image_to_data_url(img_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    res = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=max(128, int(args.max_tokens)),
        temperature=0.1,
        top_p=0.9,
    )
    raw = (res.choices[0].message.content or "").strip()

    # Try extracting JSON body if model adds wrappers.
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
            except Exception:
                parsed = None

    output = {
        "image": str(img_path.as_posix()),
        "model": args.model,
        "question": args.question,
        "parsed": parsed,
        "raw": raw,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_output={out}")


if __name__ == "__main__":
    main()
