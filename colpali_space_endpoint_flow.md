# Flow การใช้งาน CoPali ผ่าน Hugging Face Space Endpoint (สำหรับโปรเจ็กต์นี้)

เอกสารนี้สรุปเฉพาะ flow ที่โปรเจ็กต์ `typhoon_rag` ใช้งานอยู่จริง เมื่อเลือกให้ระบบ retrieval ไปเรียก CoPali score ผ่าน endpoint จาก Hugging Face Space ของคุณ

---

## 1) ภาพรวมสถาปัตยกรรม (Endpoint Mode)

1. ผู้ใช้ถามคำถามในหน้าแชท
2. `app.py` เข้าโหมด `visual` และเรียก `run_visual_retrieval_query(...)`
3. `app.py` รัน `scripts/retrieve_visual_hybrid.py` (subprocess)
4. สคริปต์สร้าง candidate จาก `indexes/colpali/pages.jsonl`
5. สคริปต์เรียก endpoint ของ Space เพื่อขอ score (`run_endpoint_scores`)
6. รวมคะแนน + จัดอันดับ + (optional) VLM re-rank/grounding
7. ส่ง hit กลับ `app.py` เพื่อประกอบ context
8. LLM ตอบโดยอ้างอิง citation จากหลักฐานที่ค้นคืนได้

---

## 2) Endpoint Contract ที่ระบบคาดหวัง (Phase-2)

ระบบนี้ใช้ contract แบบ custom (ไม่ใช่ raw text-generation endpoint)

### Request แบบเดิม (`/score`)

```json
{
  "query": "โครงสร้างของการแทนคิวด้วยอาร์เรย์",
  "candidates": [
    {
      "id": "doc.pdf:31:region:1",
      "page_id": "doc.pdf:31",
      "source": "doc.pdf",
      "page": 31,
      "image_path": "assets/figure_regions/page_031_region_01.png",
      "text_preview": "ภาพที่ 3.4 การนำเข้าและนำออกข้อมูลในโครงสร้างคิว"
    }
  ]
}
```

### Response แบบเดิม (`/score`)

```json
{
  "scores": [
    {"id": "doc.pdf:31:region:1", "score": 0.87}
  ]
}
```

ข้อกำหนด:
- ต้องมี key `scores` และเป็น list
- แต่ละ item ต้องมี `id` และ `score` (แปลงเป็น float ได้)

### Request ใหม่ Phase-2 (`/register_and_score`) [แนะนำ]

```json
{
  "query": "โครงสร้างของการแทนคิวด้วยอาร์เรย์",
  "corpus_id": "doc-corpus-8-abc123",
  "candidate_ids": ["doc.pdf:31:region:1"],
  "candidates": [
    {
      "id": "doc.pdf:31:region:1",
      "image_base64": "..."
    }
  ]
}
```

### Response (`/register_and_score`)

```json
{
  "scores": [
    {"id": "doc.pdf:31:region:1", "score": 0.87}
  ],
  "phase2_mode": "register_and_score",
  "cache_hit": false,
  "corpus_id": "doc-corpus-8-abc123"
}
```

### Request Phase-2 แบบแยก 2 ขั้น (`/register_corpus`)

```json
{
  "corpus_id": "doc-corpus-8-abc123",
  "candidates": [
    {
      "id": "doc.pdf:31:region:1",
      "image_base64": "..."
    }
  ]
}
```

### Response (`/register_corpus`)

```json
{
  "ok": true,
  "corpus_id": "doc-corpus-8-abc123",
  "registered": 8,
  "cached": false
}
```

### Request ใหม่ Phase-2 (`/score_cached`)

```json
{
  "query": "โครงสร้างของการแทนคิวด้วยอาร์เรย์",
  "corpus_id": "doc-corpus-8-abc123",
  "candidate_ids": ["doc.pdf:31:region:1"]
}
```

### Response (`/score_cached`)

```json
{
  "scores": [
    {"id": "doc.pdf:31:region:1", "score": 0.87}
  ]
}
```

---

## 3) การตั้งค่าที่ต้องมี

ใน `.env`:

```env
HUGGINGFACE_READ_TOKEN=hf_xxx
HUGGINGFACE_WRITE_TOKEN=hf_xxx
HUGGINGFACE_API_KEY=hf_xxx  # fallback compatibility; แนะนำให้เท่ากับ READ token
COLPALI_ENDPOINT_URL=macza5546/copali_endpoint
VISUAL_RETRIEVAL_ENABLED=1
VISUAL_RETRIEVAL_BACKEND=auto
```

คำแนะนำ:
- สำหรับ Space แบบ private แนะนำใช้ค่าเป็น `space_id` (เช่น `owner/space_name`) แล้วเชื่อมผ่าน `gradio_client` พร้อม token
- Runtime inference / เรียก private Space ใช้ **READ token**
- WRITE token ใช้เฉพาะงานจัดการทรัพยากร (เช่นสร้าง/แก้ไข Space, Endpoint, repo push)
- ถ้าต้องการบังคับใช้ endpoint ตรงๆ ให้เลือก backend เป็น `colpali_endpoint`
- ถ้าเลือก `auto` ระบบจะ fallback เป็นลำดับ: local ColPali -> endpoint -> metadata

---

## 4) Flow บนหน้า UI (Sidebar)

ใน Sidebar > โหมดค้นคืน:

1. ระบบบังคับโหมดค้นคืนเป็น `Visual Hybrid (CoPali/VLM)` เสมอ
2. Visual backend บังคับเป็น `colpali_endpoint`
3. Endpoint URL โหลดจาก `.env` อัตโนมัติ
4. กด `Run Visual Endpoint Health Check`

ผล health check จะถูกเขียนลงไฟล์:
- `logs/visual_endpoint_healthcheck_latest.json`

---

## 5) ปุ่มทดสอบมาตรฐาน 3 ข้อ (Pass/Fail)

หลัง health check ให้กด:
- `ทดสอบ query มาตรฐาน 3 ข้อ`

ระบบจะรัน 3 query:
1. โครงสร้างลิงค์ลิสต์แบบทิศทางเดียว
2. โครงสร้างของการแทนคิวด้วยอาร์เรย์
3. การดำเนินการแทนคิวด้วยวงกลม

เกณฑ์ pass ต่อข้อ (ตามโค้ดปัจจุบัน):
- มี hit (`hits > 0`)
- `evidence_ok = true`

รายงานถูกเขียนที่:
- `logs/visual_standard_query_test_latest.json`

---

## 6) Flow ภายใน `retrieve_visual_hybrid.py` (โหมด endpoint)

เมื่อ backend = `colpali_endpoint` หรือ `auto` ที่ fallback มา endpoint:

1. โหลด `pages.jsonl` และ `topic_hierarchy.json`
2. ทำนายหัวข้อ (topic prediction) และกรอง candidate ตาม hierarchy
3. คิด base lexical/region/structure priors
4. เรียก endpoint แบบ Phase-2:
   - พยายาม `register_and_score` ก่อน (แก้ปัญหา stateless worker ถาวร)
   - ถ้า endpoint ยังไม่มี route นี้ ค่อย fallback เป็น `register_corpus` + `score_cached`
   - สุดท้าย fallback เป็น `/score` แบบเดิม
5. normalize + fuse score
6. (optional) VLM rerank
7. (optional) visual grounding
8. คืนผล hits + metadata ไปยัง `app.py`

ไฟล์ผลรันชั่วคราว:
- `logs/visual_retrieval_runtime_latest.json`

---

## 7) การรันผ่าน CLI (ไม่ผ่าน UI)

### 7.1 เช็ก endpoint contract

```bash
python scripts/check_visual_endpoint.py --endpoint "https://<your-endpoint>"
```

บังคับให้ตรวจผ่านเฉพาะ Phase-2 API:

```bash
python scripts/check_visual_endpoint.py --endpoint "owner/space" --require-phase2 --benchmark-runs 3
```

หรือใช้จาก `.env`:

```bash
python scripts/check_visual_endpoint.py
```

### 7.2 ทดสอบ retrieval ผ่าน endpoint

```bash
python scripts/retrieve_visual_hybrid.py \
  --query "โครงสร้างของการแทนคิวด้วยอาร์เรย์" \
  --backend colpali_endpoint \
  --colpali-endpoint-url "https://<your-endpoint>" \
  --require-structure \
  --use-vlm-rerank \
  --use-visual-grounding \
  --output logs/visual_retrieval_endpoint_test.json
```

---

## 8) หมายเหตุสำคัญสำหรับ Hugging Face Space

ถ้า Space ของคุณเป็น Gradio ปกติ:
- endpoint ที่ระบบนี้ต้องใช้คือ endpoint ที่รับ/คืน JSON ตาม contract ข้างบน
- ถ้า API ของ Space เดิมไม่ตรง contract ให้ทำ adapter route เพิ่ม (แนะนำมี `/register_and_score` + `/score`, และอาจมี `/register_corpus`, `/score_cached`) แล้ว map request/response ให้ตรง

ข้อแนะนำเชิงใช้งาน:
- ตั้ง timeout และ error handling ให้ชัด (HTTP >= 400, invalid JSON)
- log เฉพาะสิ่งที่จำเป็น หลีกเลี่ยงเก็บ token

---

## 9) ไฟล์ที่เกี่ยวข้องในโปรเจ็กต์นี้

- UI + orchestration
  - `src/app.py`
- Retrieval core (endpoint scoring/fallback)
  - `scripts/retrieve_visual_hybrid.py`
- Endpoint contract health-check (CLI)
  - `scripts/check_visual_endpoint.py`
- Logs
  - `logs/visual_endpoint_healthcheck_latest.json`
  - `logs/visual_standard_query_test_latest.json`
  - `logs/visual_retrieval_runtime_latest.json`

---

## 10) Checklist สั้นก่อนใช้งานจริง

- ตั้ง `COLPALI_ENDPOINT_URL` ถูกต้อง
- Health check ผ่าน (`ok=true`)
- Standard query 3 ข้อผ่านทั้งหมด
- ตรวจว่าคำตอบสุดท้ายมี citation ที่อ้างจาก source/page/chunk จริง

