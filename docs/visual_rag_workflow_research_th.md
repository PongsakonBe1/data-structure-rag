<!-- markdownlint-disable MD013 MD024 -->
# Research-Gate Checklist (Script-by-Script)

เอกสารนี้เป็นฉบับ `Research-gate only` และใช้เป็น checklist ลงมือทำจริงตามสคริปต์ใน repo

## 1) Gate Definition (ผ่านงานวิจัย)

ต้องผ่านทุกข้อ:

1. OCR gate: `strict_research_pass=true` หรือ `operational_pass=true`
2. Retrieval gate:
   1. `topic_f1_strict_mean >= 0.95`
   2. `retrieval_hit_at_k_mean >= 0.95`
   3. `figure_hit_at_k_mean >= 0.90`
   4. `endpoint_nonempty_rate >= 0.95`
   5. `candidate_image_coverage_mean >= 0.90`
3. Task gate:
   1. `page_recall_at_k_mean >= 0.95`
   2. `operation_coverage_ratio_at_k_mean >= 0.40`
   3. `filter_survival_rate > 0.0` (ห้าม prefilter ตัดเหลือศูนย์เป็นปกติ)
4. Citation/grounding gate: unknown citation = 0 ในคำตอบที่ผ่าน gate

## 2) Environment Baseline

- `COLPALI_ENDPOINT_URL=macza5546/copali_endpoint`
- `VISUAL_RETRIEVAL_ENABLED=1`
- `VISUAL_RETRIEVAL_USE_SPLADE=1`
- `VISUAL_SPARSE_STRATEGY=splade` (option: `splade` or `bm25`)
- `VISUAL_RETRIEVAL_SPLADE_MODE=hf_api`
- `VISUAL_RETRIEVAL_SPLADE_PROVIDER=hf-inference`
- `VISUAL_RETRIEVAL_SPLADE_MODEL=naver/splade-v3`
- `VISUAL_QUERY_EXPANSION_USE_LLM=1`
- `VISUAL_QUERY_EXPANSION_MODEL=Qwen/Qwen3-4B-Instruct-2507`
- `VISUAL_QUERY_EXPANSION_MAX_TERMS=8`
- `VISUAL_PREFILTER_MIN_RECORDS=24`
- `VISUAL_PREFILTER_RESCUE_TOPN=64`
- `VISUAL_VLM_RERANK_TOP_M=12`
- `VISUAL_ENDPOINT_AUTOWARM_ON_STARTUP=1`
- `STRICT_CITATION_ENFORCEMENT=1`

## 3) Checklist ตามสคริปต์ใน Repo

### Section A: Governance

- [ ] `list_hitachi.txt` เป็น UTF-8 และอ่านได้จริง
  - Script: `scripts/build_document_hierarchy.py`
  - Output: `indexes/hierarchical/topic_hierarchy.json`

### Section B: Preparation

- [ ] VLM caption enrichment (table/diagram) พร้อม cache + parallel workers
  - Script: `scripts/enrich_visual_captions_markdown.py`
  - Output: `final_extracted_text_only_structured_full.md`, `logs/visual_caption_cache_latest.json`
- [ ] OCR + structured markdown พร้อม anchor ภาพ/ตาราง
  - Script: `scripts/run_multimodal_pipeline.py --rescan`
  - Output: `final_extracted_text_only_structured_full.md`

- [ ] QA extraction ผ่านเกณฑ์ขั้นต่ำ
  - Script: `scripts/audit_extraction_quality.py`
  - Output: `logs/extraction_audit_latest.csv`

### Section C: Indexing

- [ ] สร้าง hierarchy index และ topic mapping
  - Script: `scripts/build_document_hierarchy.py`
  - Output: `indexes/hierarchical/topic_hierarchy.json`

- [ ] สร้าง CoPali corpus metadata
  - Script: `scripts/prepare_colpali_corpus.py`
  - Output: `indexes/colpali/pages.jsonl`

- [ ] อัปเดต hard negatives และ chapter calibration
  - Script: `scripts/update_chapter2_hard_negatives.py`
  - Script: `scripts/expand_chapter_calibration.py`
  - Output: `indexes/hierarchical/hard_negative_rules.json`, `indexes/hierarchical/chapter_calibration.json`

### Section D: Retrieval (Hybrid: CoPali + [SPLADE|BM25] + RRF)

- [ ] ตรวจว่า sparse mode ทำงาน (SPLADE หรือ BM25)
  - Script: `scripts/retrieve_visual_hybrid.py`
  - Required fields in output:
    - `sparse_strategy: splade|bm25`
    - `sparse_status: ok...`
    - `sparse_score_ready: true`
    - `colpali_status_detail: ok|ok_partial|endpoint_*`
    - `candidate_quality.image_base64_coverage`
    - `filter_trace.final_records`
    - `query_expansion.query_expansion_status`
    - `query_expansion.query_expansion_terms`

- [ ] ตรวจ metadata pre-filter (strict topic first)
  - Script: `scripts/retrieve_visual_hybrid.py`
  - Required fields in output:
    - `topic_prediction.topic_scope_ids`
    - `topic_prediction.topic_filter_reason`

### Section E: Rerank / Grounding Gate

- [ ] รัน grounding benchmark ด้วย human labels
  - Script: `scripts/benchmark_visual_grounding_human.py`
  - Output: `logs/visual_grounding_human_benchmark_*.json`

- [ ] ตรวจ false abstain / evidence gate
  - File: `logs/runtime_events.jsonl`
  - Event ที่ต้องตรวจ:
    - `visual_retrieve_success`
    - `citation_validation`
    - `citation_enforcement_abstain`

### Section F: Answering

- [ ] Citation repair ทำงานและไม่สร้าง unknown citation
  - File: `logs/runtime_events.jsonl`
  - ต้องเห็น:
    - `citation_repair_attempt`
    - unknown list เป็นค่าว่างในรอบที่ผ่าน

### Section G: Research Gate Summary

- [ ] สรุป PASS/FAIL อัตโนมัติ
  - Script: `scripts/run_research_gate.py`
  - Output:
    - `logs/research_gate_latest.json`
    - `docs/research_gate_latest.md`

## 4) คำสั่งรันแบบครบชุด (Research-Gate Runbook)

```bash
python scripts/run_multimodal_pipeline.py --rescan
python scripts/check_visual_endpoint.py --endpoint "macza5546/copali_endpoint" --benchmark-runs 3
python scripts/benchmark_visual_topic_retrieval.py --dataset eval/visual_task_dataset_top5.jsonl --endpoint macza5546/copali_endpoint --top-k 8 --candidate-k 20 --use-vlm-rerank --use-visual-grounding --output-json logs/visual_topic_benchmark_hotfix_latest.json --output-csv logs/visual_topic_benchmark_hotfix_latest.csv
python scripts/benchmark_visual_task_metrics.py --dataset eval/visual_task_dataset_top5.jsonl --endpoint macza5546/copali_endpoint --top-k 8 --candidate-k 20 --use-vlm-rerank --use-grounding --output-json logs/visual_task_metrics_hotfix_latest.json --output-csv logs/visual_task_metrics_hotfix_latest.csv
python scripts/evaluate_ocr_research_grade.py --markdown final_extracted_text_only_structured_full.md --gt-jsonl eval/ocr_gt_first_mid_last_v3_structured_full.jsonl --output logs/ocr_research_grade_structured_full_with_gt_v3_latest.json
python scripts/run_research_gate.py --ocr-report logs/ocr_research_grade_structured_full_with_gt_v3_latest.json --retrieval-report logs/visual_topic_benchmark_hotfix_latest.json --task-report logs/visual_task_metrics_hotfix_latest.json --output-json logs/research_gate_latest.json --output-md docs/research_gate_latest.md
```

### 4.1 เปรียบเทียบ Sparse Option (CoPali+SPLADE vs CoPali+BM25)

```bash
# CoPali + SPLADE
python scripts/benchmark_visual_topic_retrieval.py --dataset eval/visual_task_dataset_top5.jsonl --endpoint macza5546/copali_endpoint --top-k 8 --candidate-k 20 --sparse-strategy splade --use-vlm-rerank --use-visual-grounding --output-json logs/visual_topic_benchmark_splade_latest.json --output-csv logs/visual_topic_benchmark_splade_latest.csv
python scripts/benchmark_visual_task_metrics.py --dataset eval/visual_task_dataset_top5.jsonl --endpoint macza5546/copali_endpoint --top-k 8 --candidate-k 20 --sparse-strategy splade --use-vlm-rerank --use-grounding --output-json logs/visual_task_metrics_splade_latest.json --output-csv logs/visual_task_metrics_splade_latest.csv

# CoPali + BM25
python scripts/benchmark_visual_topic_retrieval.py --dataset eval/visual_task_dataset_top5.jsonl --endpoint macza5546/copali_endpoint --top-k 8 --candidate-k 20 --sparse-strategy bm25 --use-vlm-rerank --use-visual-grounding --output-json logs/visual_topic_benchmark_bm25_latest.json --output-csv logs/visual_topic_benchmark_bm25_latest.csv
python scripts/benchmark_visual_task_metrics.py --dataset eval/visual_task_dataset_top5.jsonl --endpoint macza5546/copali_endpoint --top-k 8 --candidate-k 20 --sparse-strategy bm25 --use-vlm-rerank --use-grounding --output-json logs/visual_task_metrics_bm25_latest.json --output-csv logs/visual_task_metrics_bm25_latest.csv
```

เกณฑ์ตัดสิน: เลือก strategy ที่ได้ `topic_f1_strict_mean`, `retrieval_hit_at_k_mean`, `figure_hit_at_k_mean` สูงกว่า และไม่ทำให้ `small_region_ratio_at_k_mean` สูงผิดปกติ
รวมถึง `endpoint_nonempty_rate` และ `candidate_image_coverage_mean` ต้องไม่ลดลง

## 5) Mapping ขั้นตอนกับหลักฐานงานวิจัย

| ขั้นตอนในระบบ | ตัวชี้วัดที่ใช้ตัดสิน | งานวิจัยหลักที่รองรับ |
| --- | --- | --- |
| Hybrid retrieval (dense + sparse + RRF) | topic-F1 / hit@k / figure-hit@k | [1], [2], [3] |
| Query expansion และ lexical enrichment | hit@k / coverage | [4], [5] |
| Visual-first retrieval ด้วย CoPali | page/figure recall, region hit ratio | [10] |
| Grounding/citation gate ก่อนตอบ | unknown citation = 0, abstain rate | [6], [7], [8], [9] |
| VLM caption enrichment ใน ingestion | OCR+structure coverage, figure structure coverage | [11] |

หมายเหตุเชิงทฤษฎี:

- RRF ช่วยลดความเสี่ยงจาก ranking variance ของช่องทางเดียว และมักเพิ่ม robustness ของ top-k ใน heterogeneous corpus [2], [3]
- SPLADE ช่วย sparse matching บน token expansion ที่ lexical drift สูง ส่วน BM25 เป็น baseline ที่เสถียรและตีความง่าย [1]
- เมื่องานมีภาพ/diagram หนัก CoPali ให้สัญญาณ visual relevance ดีกว่า text-only retriever และควรใช้ร่วม sparse channel เพื่อคุม recall [10]
- การบังคับ grounding/citation gate ก่อนตอบลด hallucination แต่ต้อง calibrate threshold เพื่อลด false abstain [6], [8], [9]

## 6) งานวิจัยอ้างอิงหลัก (IEEE, <= 5 ปี)

[1] T. Formal et al., "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval," arXiv:2109.10086, 2021.  
[2] N. Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models," arXiv:2104.08663, 2021.  
[3] O. Khattab et al., "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction," arXiv:2112.01488, 2021.  
[4] L. Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)," arXiv:2212.10496, 2022.  
[5] J. Wang et al., "Query2doc: Query Expansion with Large Language Models," arXiv:2303.07678, 2023.  
[6] S. Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation," arXiv:2309.15217, 2023.  
[7] N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," arXiv:2307.03172, 2023.  
[8] E. M. Wang et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," arXiv:2306.05685, 2023.  
[9] Z. Zheng et al., "Judging LLM-as-a-Judge: Bias and Limitations," arXiv:2305.17926, 2023.  
[10] A. Faysse et al., "ColPali: Efficient Document Retrieval with Vision Language Models," arXiv:2407.01449, 2024.  
[11] Qwen Team, "Qwen2.5-VL Technical Report," arXiv, 2024.

<!-- markdownlint-enable MD013 MD024 -->




