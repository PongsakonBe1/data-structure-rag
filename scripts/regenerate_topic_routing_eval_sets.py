#!/usr/bin/env python
from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TOC_PATH = ROOT / "list_hitachi.txt"
DIRECT_OUT = ROOT / "eval" / "topic_routing_regression_queries.csv"
AMB_OUT = ROOT / "eval" / "topic_routing_ambiguous_queries.csv"
SEED_OUT = ROOT / "eval" / "topic_routing_answers_seed.csv"


def _parse_leaf_topics(toc_path: Path) -> list[tuple[str, str]]:
    lines = toc_path.read_text(encoding="utf-8").splitlines()
    items: list[tuple[str, str]] = []
    for ln in lines:
        m = re.search(r"\b(\d+(?:\.\d+)+)\b\s*(.*)", ln)
        if not m:
            continue
        tid = m.group(1).strip()
        title = m.group(2).strip(" \t-•")
        if tid and title:
            items.append((tid, title))

    ids = [x[0] for x in items]
    leaves: list[tuple[str, str]] = []
    for tid, title in items:
        if not any(other != tid and other.startswith(f"{tid}.") for other in ids):
            leaves.append((tid, title))
    return leaves


AMBIGUOUS_BY_ID = {
    "1.1.1": "เวลาออกแบบระบบ ควรจัดระเบียบข้อมูลอย่างไรให้ค้นหาและแก้ไขได้ง่าย",
    "1.1.2": "ก่อนเขียนโค้ดจริง ทำไมต้องกำหนดขั้นตอนแก้ปัญหาเป็นลำดับให้ชัดเจน",
    "1.2.1": "ข้อมูล 0 กับ 1 ที่ใช้แทนสถานะการทำงานของเครื่องเรียกว่าอะไร",
    "1.2.2": "หน่วยข้อมูลที่รวม 8 บิตเข้าด้วยกันเรียกว่าอะไร",
    "1.2.3": "ช่องข้อมูลย่อยอย่างชื่อ เพศ อายุ ในแบบฟอร์มควรเรียกว่าอะไร",
    "1.2.4": "การรวมหลายช่องข้อมูลของบุคคลหนึ่งคนเป็นรายการเดียวเรียกว่าอะไร",
    "1.2.5": "การรวมหลายเรคอร์ดที่เกี่ยวข้องกันเป็นชุดเดียวเรียกว่าอะไร",
    "1.2.6": "แหล่งเก็บข้อมูลขนาดใหญ่ที่หลายส่วนใช้งานร่วมกันควรเรียกว่าอะไร",
    "1.3.1": "ถ้าแบ่งโครงสร้างข้อมูลตามลักษณะการจัดเก็บ ควรแบ่งเป็นกลุ่มหลักอะไรบ้าง",
    "1.3.2": "แนวคิดขั้นตอนวิธีที่นิยมใช้แก้ปัญหาโดยรวมมีแบบไหนบ้าง",
    "1.4.1": "ถ้าจะอธิบายลำดับการทำงานให้เห็นเป็นภาพ ควรใช้เครื่องมืออะไร",
    "1.4.2": "ถ้ายังไม่เขียนโปรแกรมจริง แต่ต้องการเขียนขั้นตอนให้คนอ่านเข้าใจ ควรใช้อะไร",
    "1.5.1": "โปรแกรมที่ทำคำสั่งจากบนลงล่างตามลำดับอยู่ในโครงสร้างควบคุมแบบใด",
    "1.5.2": "การตัดสินใจด้วยเงื่อนไข if/else จัดอยู่ในโครงสร้างควบคุมแบบใด",
    "1.5.3": "การทำงานแบบวนซ้ำจนกว่าจะครบเงื่อนไขเป็นโครงสร้างควบคุมแบบใด",
    "2.1.1": "ตัวแปรชุดเดียวที่เก็บข้อมูลชนิดเดียวหลายค่าและเข้าถึงด้วย index คืออะไร",
    "2.1.2": "ถ้าดูเชิงโครงสร้าง อาร์เรย์มีองค์ประกอบสำคัญอะไรบ้าง",
    "2.2.1": "ตารางที่มีแถวเดียวแต่มีหลายตำแหน่งต่อเนื่องตรงกับอาร์เรย์ชนิดใด",
    "2.2.2": "ข้อมูลแบบตารางที่มีทั้งแถวและคอลัมน์ตรงกับอาร์เรย์ชนิดใด",
    "2.3.1": "โครงสร้างที่เก็บข้อมูลเป็นโหนดและเชื่อมกันด้วยพอยน์เตอร์เรียกว่าอะไร",
    "2.3.2": "ถ้ามององค์ประกอบของ linked list จะมีส่วนข้อมูลและส่วนใดอีกหนึ่งส่วน",
    "2.4.1": "ลิงก์ลิสต์ที่ลิงก์ชี้ไปทางเดียวจากต้นไปท้ายคือแบบไหน",
    "2.4.2": "การเพิ่ม แทรก และลบโหนดในลิสต์ที่ชี้ทางเดียวต้องจัดการลิงก์อย่างไรโดยรวม",
    "3.1": "โครงสร้างที่ใช้หลักเข้าก่อนออกก่อนหรือ FIFO คืออะไร",
    "3.2.1": "เวลาเพิ่มข้อมูลเข้า queue ต้องใส่ที่ปลายไหน",
    "3.2.2": "เวลานำข้อมูลออกจาก queue ต้องดึงจากปลายไหน",
    "3.3.1": "ถ้าจะแทนคิวด้วยอาร์เรย์ ต้องมีตัวชี้หลักอะไรบ้าง",
    "3.3.2": "ขั้นตอน insert/remove ของคิวแบบอาร์เรย์ทำงานกับตำแหน่งไหนบ้าง",
    "3.3.3": "จะทำให้คิวแบบอาร์เรย์กลับมาใช้ช่องว่างด้านหน้าได้ ควรใช้แนวคิดอะไร",
    "4.1": "โครงสร้างที่เข้าทีหลังออกก่อนหรือ LIFO คืออะไร",
    "4.2.1": "คำสั่งที่ใช้เพิ่มข้อมูลขึ้นบนสุดของ stack เรียกว่าอะไร",
    "4.2.2": "คำสั่งที่ใช้ดึงข้อมูลออกจากบนสุดของ stack เรียกว่าอะไร",
    "4.2.3": "ตัวชี้ที่บอกตำแหน่งข้อมูลบนสุดของสแตกเรียกว่าอะไร",
    "4.3.1": "ถ้าจะแทน stack ด้วยอาร์เรย์ ต้องจัดการ top และตรวจ overflow/underflow อย่างไร",
    "4.3.2": "การแปลงนิพจน์คณิตศาสตร์ด้วยสแตกเกี่ยวข้องกับ infix/prefix/postfix อย่างไร",
    "5.1": "โครงสร้างข้อมูลแบบลำดับชั้นที่มีรากและโหนดลูกเรียกว่าอะไร",
    "5.2.1": "ต้นไม้ที่แต่ละโหนดมีลูกได้หลายทางโดยไม่จำกัดสองทางเรียกว่าอะไร",
    "5.2.2": "ต้นไม้ที่แต่ละโหนดมีลูกได้ไม่เกินสองโหนดเรียกว่าอะไร",
    "5.2.3": "ต้นไม้ทวิภาคที่โหนดเติมเต็มตามเงื่อนไขระดับเรียกว่าอะไร",
    "5.3.1": "การท่องทรีแบบเข้า root ก่อนแล้วซ้ายขวาเรียกว่า traversal แบบใด",
    "5.3.2": "การท่องทรีแบบซ้ายก่อน root แล้วขวาเรียกว่า traversal แบบใด",
    "5.3.3": "การท่องทรีแบบซ้ายขวาก่อนแล้วค่อย root เรียกว่า traversal แบบใด",
    "5.4": "นิพจน์คณิตศาสตร์ที่แทนเป็นโหนดตัวดำเนินการและตัวถูกดำเนินการคือทรีแบบใด",
    "5.5": "การเก็บโหนดไบนารีทรีลงอาร์เรย์ตามดัชนีตำแหน่งเรียกการแทนแบบใด",
    "5.6": "ถ้าจะเปลี่ยนทรีทั่วไปให้แทนในรูปไบนารีทรี ควรทำอย่างไรโดยหลัก",
}


def _fallback_ambiguous_question(title: str) -> str:
    base = re.sub(r"\s*\([^)]*\)", "", title).strip()
    return f"ช่วยอธิบายแนวคิดของหัวข้อ \"{base}\" ในเชิงการใช้งานจริง"


def _save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    leaves = _parse_leaf_topics(TOC_PATH)

    direct_rows = []
    amb_rows = []
    seed_rows = []

    for idx, (tid, title) in enumerate(leaves, start=1):
        direct_question = title
        amb_question = AMBIGUOUS_BY_ID.get(tid, _fallback_ambiguous_question(title))

        direct_rows.append(
            {
                "order": idx,
                "category": "direct",
                "question": direct_question,
                "expected_topic_id": tid,
            }
        )
        amb_rows.append(
            {
                "order": idx,
                "question": amb_question,
                "should_attempt": 1,
                "expected_topic_id": tid,
            }
        )
        seed_rows.append(
            {
                "order": idx,
                "question": direct_question,
                "assistant_answer": "",
            }
        )

    _save_csv(
        DIRECT_OUT,
        direct_rows,
        ["order", "category", "question", "expected_topic_id"],
    )
    _save_csv(
        AMB_OUT,
        amb_rows,
        ["order", "question", "should_attempt", "expected_topic_id"],
    )
    _save_csv(SEED_OUT, seed_rows, ["order", "question", "assistant_answer"])

    print(
        {
            "direct_rows": len(direct_rows),
            "ambiguous_rows": len(amb_rows),
            "seed_rows": len(seed_rows),
            "direct_out": str(DIRECT_OUT),
            "ambiguous_out": str(AMB_OUT),
            "seed_out": str(SEED_OUT),
        }
    )


if __name__ == "__main__":
    main()

