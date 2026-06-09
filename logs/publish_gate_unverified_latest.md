# Publish Gate: FAIL

- gate_pass: `False`
- operational_pass: `True`
- strict_research_pass: `False`
- strict_mode: `external_publish_grade`
- manual_verified_required: `True`
- gt_metrics_available: `False`
- samples_used/total: `0/18`
- cer_mean: `None`
- wer_mean: `None`

## Reasons
- strict_research_failed
- missing_verified_gt_samples
- insufficient_verified_gt_samples
- failed_checks:cer_mean,wer_mean

## Failed Checks
- `cer_mean`: value=None threshold=0.05
- `wer_mean`: value=None threshold=0.2

- evaluator_report: `logs/ocr_research_grade_external_publish_unverified_latest.json`
