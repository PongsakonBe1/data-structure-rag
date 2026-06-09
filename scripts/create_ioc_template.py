#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create IOC evaluation CSV template with proper encoding for Excel."""

import csv
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
IOC_FILE = LOG_DIR / "expert_ioc_eval.csv"

fieldnames = [
    "timestamp",
    "evaluator_name",
    "question",
    "answer",
    "question_type",
    "system_behavior",
    "ioc_score",
    "comments"
]

# สร้างไฟล์ใหม่ด้วย utf-8-sig encoding (with BOM) เพื่อให้ Excel อ่านไทยได้
with open(IOC_FILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

print(f"✅ สร้างไฟล์ {IOC_FILE} สำเร็จ (UTF-8 with BOM)")
print(f"   Encoding: utf-8-sig")
print(f"   สามารถเปิดใน Excel ได้โดยภาษาไทยไม่เพี้ยน")
