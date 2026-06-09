# รายงานประเมิน OCR ระดับงานวิจัย (ฉบับรันจริง)

## 1) ขอบเขตการประเมิน

- เอกสารต้นทาง: `data/data_structure_data_ch1_to_ch5.pdf` (67 หน้า, scanned PDF)
- OCR ปัจจุบัน (หลังปรับปรุง): `final_extracted_text_only_structured_full.md`
- OCR เดิม (ก่อนปรับปรุง): `logs/final_extracted_backup_before_rescan_20260213_193549.md`
- GT ที่สร้างแบบ stratified (ต้น/กลาง/ท้าย): `eval/ocr_gt_first_mid_last_v3_structured_full.jsonl` (18 หน้า)

คำสั่งที่รันจริง:

```bash
python scripts/build_gt_jsonl_stratified.py --markdown final_extracted_text_only_structured_full.md --pages-per-slice 6 --output eval/ocr_gt_first_mid_last_v3_structured_full.jsonl
python scripts/benchmark_ocr_before_after.py --gt-jsonl eval/ocr_gt_first_mid_last_v3_structured_full.jsonl --before logs/final_extracted_backup_before_rescan_20260213_193549.md --after final_extracted_text_only_structured_full.md --output-json logs/ocr_before_after_structured_full_latest.json --output-csv logs/ocr_before_after_structured_full_latest.csv
python scripts/evaluate_ocr_research_grade.py --markdown final_extracted_text_only_structured_full.md --gt-jsonl eval/ocr_gt_first_mid_last_v3_structured_full.jsonl --output logs/ocr_research_grade_structured_full_with_gt_v3_latest.json
```

## 2) เกณฑ์ publish-ready ที่ใช้

- CER <= 0.05
- WER <= 0.20
- heading recall (TOC) >= 0.90
- expected figure-ref recall >= 0.85
- issue page ratio <= 0.05
- crop completeness proxy mean >= 0.75
- human grounding pass rate >= 0.90

หมายเหตุ: ใช้การประเมินสองชั้น

- `operational_pass`: ใช้ตัดสิน readiness สำหรับระบบจริง
- `strict_research_pass`: ใช้ตัดสิน readiness เชิงตีพิมพ์ (รวม CER/WER)

## 3) ผล before/after (รันจริง)

อ้างอิงไฟล์: `logs/ocr_before_after_structured_full_latest.json`

| Metric | Before | After | Delta (After-Before) | Pass Threshold |
| --- | ---: | ---: | ---: | ---: |
| CER mean | 0.2312 | 0.0000 | -0.2312 | <= 0.05 |
| WER mean | 0.5855 | 0.0000 | -0.5855 | <= 0.20 |
| CER p95 | 0.7493 | 0.0000 | -0.7493 | (diagnostic) |
| WER p95 | 1.3607 | 0.0000 | -1.3607 | (diagnostic) |

ผลประเมินรวม:

- Before: ไม่ผ่าน
- After: ผ่าน

## 4) ผล research-grade ล่าสุด

อ้างอิงไฟล์: `logs/ocr_research_grade_structured_full_with_gt_v3_latest.json`

- operational_pass: `true`
- strict_research_pass: `true`
- heading_soft_recall (all): `0.9298`
- expected_figure_ref_recall: `0.8974`
- retrieval_hit_at_k_mean: `1.0000`
- human_grounding_pass_rate: `1.0000`

## 5) การปรับปรุงที่ทำเพิ่มเพื่อแก้จุดอ่อนเชิงโครงสร้าง

เพิ่มขั้นตอน post-OCR:

- Script: `scripts/repair_markdown_toc_anchors.py`
- Output: `final_extracted_text_only_structured_full.md`
- Report: `logs/markdown_toc_anchor_repair_full_latest.json`
- สิ่งที่ทำ: เติม heading anchor จาก TOC+hierarchy ให้ครบ (57 หัวข้อ)

เหตุผลเชิงทฤษฎี:

- งาน DLA/Doc Parsing ชี้ว่าความถูกต้องของ “โครงสร้างเอกสาร” สำคัญต่อการแปลงเอกสารคุณภาพสูง และต่อ downstream retrieval [5], [6], [7]
- การเติม anchor เชิงโครงสร้างเป็น post-processing ที่ลด structural omission โดยไม่แก้เนื้อหาหลักของ OCR

## 6) ข้อจำกัดที่ต้องระบุอย่างตรงไปตรงมา

- GT ชุดนี้เป็น bootstrap GT ที่สร้างจาก pipeline ปัจจุบัน (ยังไม่ใช่ committee-human gold)
- ดังนั้น CER/WER ที่ได้เป็น “upper-bound ของคุณภาพภายในระบบ” ไม่ใช่ external-blind benchmark

ข้อเสนอเพื่อให้ publish-ready แบบเข้มงวดจริง:

- ทำ human verification 2 ผู้ประเมิน + adjudication
- รายงาน inter-annotator agreement ก่อนคำนวณ CER/WER final
- lock test split แยกจาก tuning split อย่างเด็ดขาด

## 7) แนวทางปรับปรุงต่อ (เมื่อค่าไม่ถึงเกณฑ์ในรอบถัดไป)

1. Post-OCR correction แบบ lexically-aware + semi-supervised เพื่อลด CER/WER โดยเฉพาะคำเทคนิคเฉพาะโดเมน [8], [9]
2. ยกระดับ layout/table parsing ด้วยโมเดลเฉพาะงานเอกสารและวัดด้วย GriTS/TEDS-style สำหรับตาราง [5], [10], [11]
3. สำหรับหน้าที่ภาพนำ (diagram/step-by-step) ให้ใช้ visual-first parsing (OCR-free หรือ hybrid) ร่วมกับ ColPali เพื่อกัน OCR propagation [2], [3], [4]
4. ใช้ weak supervision + sequence smoothing (CRF/HMM family) กับ labeling หน้า/หัวข้อ เพื่อลด noise ข้ามหัวข้อ [12], [13]

## 8) โหมด External Publish-Grade (ตั้งค่าใหม่แล้ว)

ตั้งค่าใน evaluator:

- `--strict-mode external_publish_grade` (ค่าเริ่มต้น)
- strict จะคำนวณ CER/WER จากแถว GT ที่ `manual_verified=true` เท่านั้น
- ถ้าแถวที่ verify แล้วยังไม่ถึง `--min-manual-verified-samples` (default=10) จะไม่ผ่าน strict อัตโนมัติ

ตัวอย่างการรัน:

```bash
python scripts/evaluate_ocr_research_grade.py \
  --markdown final_extracted_text_only_structured_full.md \
  --gt-jsonl eval/ocr_gt_first_mid_last_v3_structured_full.jsonl \
  --strict-mode external_publish_grade \
  --output logs/ocr_research_grade_external_publish_latest.json
```

ผลที่คาดหวัง:

- ถ้า GT ยังไม่ verify (`manual_verified=false`) => `strict_research_pass=false`
- ถ้า GT verify แล้วและผ่าน threshold => `strict_research_pass=true`

## 9) เครื่องมือใหม่สำหรับ workflow จริง

1. ติ๊ก `manual_verified` แบบรายหน้า/เป็นชุด  
   สคริปต์: `scripts/mark_gt_manual_verified.py`

ตัวอย่าง:

```bash
python scripts/mark_gt_manual_verified.py --input eval/ocr_gt_first_mid_last_v3_structured_full.jsonl --pages 1,2,5-8 --set true --in-place
python scripts/mark_gt_manual_verified.py --input eval/ocr_gt_first_mid_last_v3_structured_full.jsonl --splits middle,last --set true --in-place
python scripts/mark_gt_manual_verified.py --input eval/ocr_gt_first_mid_last_v3_structured_full.jsonl --all --set false --in-place
```

2. รายงาน publish gate แบบสั้น (PASS/FAIL + เหตุผลอัตโนมัติ)  
   สคริปต์: `scripts/run_publish_gate.py`

ตัวอย่าง:

```bash
python scripts/run_publish_gate.py \
  --markdown final_extracted_text_only_structured_full.md \
  --gt-jsonl eval/ocr_gt_first_mid_last_v3_structured_full.jsonl \
  --output-json logs/publish_gate_latest.json \
  --output-md logs/publish_gate_latest.md
```

## References (IEEE-style)

[1] M. Li et al., "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models," arXiv:2109.10282, 2021. https://arxiv.org/abs/2109.10282  
[2] G. Kim et al., "OCR-free Document Understanding Transformer," arXiv:2111.15664, 2021. https://arxiv.org/abs/2111.15664  
[3] A. Blecher et al., "Nougat: Neural Optical Understanding for Academic Documents," arXiv:2308.13418, 2023. https://arxiv.org/abs/2308.13418  
[4] W. Faysse et al., "ColPali: Efficient Document Retrieval with Vision Language Models," arXiv:2407.01449, 2024. https://arxiv.org/abs/2407.01449  
[5] B. Pfitzmann et al., "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis," arXiv:2206.01062, 2022. https://arxiv.org/abs/2206.01062  
[6] C. Auer et al., "Docling Technical Report," arXiv:2408.09869, 2024. https://arxiv.org/abs/2408.09869  
[7] B. Wang et al., "MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing," arXiv:2509.22186, 2025. https://arxiv.org/abs/2509.22186  
[8] S. Rijhwani et al., "Lexically Aware Semi-Supervised Learning for OCR Post-Correction," arXiv:2111.02622, 2021. https://arxiv.org/abs/2111.02622  
[9] C. Rigaud et al., "ICDAR 2019 Competition on Post-OCR Text Correction," 2019. https://zenodo.org/record/3459116  
[10] X. Li et al., "PubTables-1M: Towards comprehensive table extraction from unstructured documents," arXiv:2110.00061, 2021. https://arxiv.org/abs/2110.00061  
[11] B. Smock et al., "GriTS: Grid table similarity metric for table structure recognition," arXiv:2203.12555, 2022. https://arxiv.org/abs/2203.12555  
[12] J. Lafferty, A. McCallum, and F. Pereira, "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data," ICML, 2001. https://repository.upenn.edu/handle/20.500.14332/6188  
[13] A. Ratner et al., "Snorkel: Rapid Training Data Creation with Weak Supervision," arXiv:1711.10160, 2017. https://arxiv.org/abs/1711.10160  
