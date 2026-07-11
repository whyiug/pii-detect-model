# Validation-only span calibration

`scripts/calibrate_predictions.py` accepts exactly two data inputs: canonical synthetic
`validation` gold from `public_release_pool`, and raw-score prediction JSONL for the identical
document set. It rejects `test`, `evaluation_only`, non-synthetic, quarantined, private, or
quality-gate-failing records. PII Bench ZH and any other frozen test set must never be supplied to
this command.

```bash
python scripts/calibrate_predictions.py \
  --gold /data1/datasets/pii-detect-model/processed/public_release_pool/synthetic_v1/validation.jsonl \
  --predictions /data1/runs/pii-detect-model/seed42/validation.raw_predictions.jsonl \
  --output /data1/runs/pii-detect-model/seed42/calibration.json \
  --diagnostics /data1/runs/pii-detect-model/seed42/calibration.diagnostics.json \
  --model-version qwen3-pii-zh-0.6b-seed42 \
  --t0-recall-floor 0.90 \
  --t1-recall-floor 0.85 \
  --fit-temperature
```

Threshold candidates are observed emitted-span scores plus 0 and 1. A prediction is positive only
when `(doc_id, start, end, label)` exactly matches gold. All gold spans that were never emitted are
fixed false negatives for every threshold. They are not inserted as artificial score-zero positive
examples, so lowering a threshold can never pretend to recover a missing candidate. If the maximum
emitted-candidate recall cannot meet a configured T0/T1 floor, calibration fails.

T0 and T1 labels optimize F2 after satisfying their recall floor; T2 labels optimize F1. Ties prefer
recall, then precision, then the stricter threshold. The output is a serving-compatible
`CalibrationBundle`. Diagnostics are self-hashed and contain only input hashes, counts, per-label
TP/FP/FN, thresholds, ECE/Brier, and declared fitting parameters—never local paths, document IDs,
raw text, or entity values.

Optional temperature fitting is binary span-confidence fitting over emitted predictions only, with
strict exact-match correctness as the target. It does not make confidence claims about un-emitted
false negatives. When the configured sample/class minimum is not met, temperature remains `1.0`
and diagnostics report `insufficient_statistics`. ECE and Brier use the same emitted-span scope and
are reported both before and after temperature scaling.
