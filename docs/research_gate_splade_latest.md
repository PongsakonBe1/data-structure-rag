# Research Gate Summary

- Overall: FAIL
- Failed checks: retrieval_hit_at_k, figure_hit_at_k, task_page_recall_at_k

| Check | Pass | Value | Threshold |
| --- | --- | --- | --- |
| ocr_research_gate | True | {'strict_research_pass': True, 'operational_pass': True} | strict_research_pass=true (or operational_pass=true) |
| retrieval_topic_f1 | True | 1.0 | >=0.95 |
| retrieval_hit_at_k | False | 0.8 | >=0.95 |
| figure_hit_at_k | False | 0.8 | >=0.90 |
| endpoint_nonempty_rate | True | 1.0 | >=0.95 |
| candidate_image_coverage_mean | True | 1.0 | >=0.90 |
| filter_survival_rate | True | 0.1461 | >=0.01 |
| task_page_recall_at_k | False | 0.4 | >=0.95 |
| task_operation_coverage | True | 0.83335 | >=0.40 |
