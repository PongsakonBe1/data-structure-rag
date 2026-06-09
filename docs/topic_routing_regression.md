# Topic Routing Regression

รัน regression + สร้างไฟล์สรุปทั้งหมด:

```bash
python scripts/regression_topic_routing.py --enforce
```

ไฟล์ผลลัพธ์หลัก:

- `logs/topic_routing_regression_latest.csv`
- `logs/topic_confusion_old_latest.csv`
- `logs/topic_confusion_new_latest.csv`
- `logs/topic_routing_ambiguous_latest.csv`
- `logs/topic_routing_expert_review_latest.csv`
- `logs/topic_routing_dashboard_latest.html`
- `logs/topic_routing_expert_review_dashboard_latest.html`
- `logs/topic_routing_regression_summary_latest.json`

## Expert Review

ใช้ไฟล์ `logs/topic_routing_expert_review_latest.csv` ให้ผู้เชี่ยวชาญเติมคอลัมน์:

- `answer_correct(Y/N)`
- `note`
- `reviewer`
- `reviewed_at`

ถ้าต้องการกรอกบนหน้า dashboard แล้ว export CSV:

- เปิด `logs/topic_routing_expert_review_dashboard_latest.html`
- กรอกผลตรวจ
- กด `Export Reviewed CSV`

## Pre-release Gate

`scripts/run_baseline_validate.py` ผูก topic routing regression แล้ว (ค่าเริ่มต้น):

```bash
python scripts/run_baseline_validate.py --endpoint <owner/space>
```

ตัวเลือก gate:

- `--skip-topic-regression`
- `--min-topic-routing-accuracy 1.0`
- `--min-topic-ambiguous-pass-rate 1.0`

## CI

workflow: `.github/workflows/topic-routing-regression.yml`

- รันบน push / pull_request ที่กระทบไฟล์ routing regression
- fail ทันทีถ้า accuracy ต่ำกว่า threshold
- อัปโหลด CSV/summary/dashboard เป็น artifact
