# Span calibration protocol

`scripts/calibrate_predictions.py` supports two deliberately separate calibration roles:

1. ordinary synthetic `validation` data in `public_release_pool`, for development experiments;
2. the frozen `synthetic_sota_release_eval_v2/calibration` split, only after cross-seed model and
   checkpoint selection is complete.

It rejects test data, private/quarantined records, failed quality gates, and unbound
`evaluation_only` data. PII Bench ZH, CLUENER, the v2 `internal_evaluation` split, and any other
final/public benchmark must never be supplied to this command.

## Confirmatory v2 calibration

The former `synthetic_sota_release_eval_v1` confirmatory role is withdrawn. One v1 row was inspected
structurally before model results existed (with text redacted and without predictions or metrics), so
v1 is conservatively excluded from the zero-read confirmatory claim. The replacement v2 design was
frozen before any v3 seed-42 validation result and its JSONL contents must remain unread until
cross-seed selection is frozen. See `docs/synthetic_sota_release_eval_v2.md`.

After selection, first generate raw-score predictions for the exact 10,000-document v2 calibration
split with the selected, immutable merged model. Then run the following once, replacing only the
prediction/output directory and the stable selected-model identifier. Set `$DATA_ROOT` to the
external dataset root and `$RUN_ROOT` to the external run-artifact root:

```bash
python scripts/calibrate_predictions.py \
  --gold "${DATA_ROOT}/processed/public_release_pool/synthetic_sota_release_eval_v2/calibration.jsonl" \
  --predictions "${RUN_ROOT}/release/<selected-model>/calibration.raw_predictions.jsonl" \
  --fit-prediction-manifest "${RUN_ROOT}/release/<selected-model>/calibration.model_raw.provenance.json" \
  --release-eval-v2-authorization "${RUN_ROOT}/release/<selected-model>/calibration.authorization.json" \
  --output "${RUN_ROOT}/release/<selected-model>/calibration.json" \
  --diagnostics "${RUN_ROOT}/release/<selected-model>/calibration.diagnostics.json" \
  --model-version <selected-model-id> \
  --dataset-manifest "${DATA_ROOT}/processed/public_release_pool/synthetic_sota_release_eval_v2/dataset_manifest.json" \
  --expected-dataset-manifest-sha256 bd99b9a858c07fe96f1effcc93d07defa17f7ce39a00712bcfe6b9fda56c19ad \
  --supersession-freeze configs/evaluation/synthetic_sota_release_eval_v2_supersession_freeze.json \
  --expected-supersession-freeze-sha256 06ebdce5e288b0e7699a8225e28c4c4f0a1762e76f6967cf990b14384101f0f0 \
  --t0-recall-floor 0.90 \
  --t1-recall-floor 0.85 \
  --fit-temperature
```

The authorization must first be built and strictly replayed with
`scripts/release_eval_v2_stage_gate.py`; see `docs/release_eval_v2_stage_gate.md`. For this formal
path, `--model-version` is the selected training manifest's logical `manifest_sha256`, not a model
directory alias. The expected dataset manifest value above is its canonical self-hash. The command additionally
checks the physical manifest bytes, calibration JSONL hash/count, all 24 labels, generation salt,
usage policy, supersession chronology, and the physical freeze-receipt hash. It explicitly rejects
v1. Outputs are immutable (`0444`), published atomically, and never overwrite existing paths.

Do not inspect `internal_evaluation` or generate its final metrics until the selected model,
calibration bundle, service profile, routing, validators, rules, fusion thresholds, and prediction
provenance have all been frozen. Its result is one-shot and may not be used to revise that candidate.

## Development-only calibration

For ordinary development runs, omit all four frozen-dataset arguments:

```bash
python scripts/calibrate_predictions.py \
  --gold "${DATA_ROOT}/processed/public_release_pool/<development-dataset>/validation.jsonl" \
  --predictions "${RUN_ROOT}/<development-run>/validation.raw_predictions.jsonl" \
  --output "${RUN_ROOT}/<development-run>/calibration.json" \
  --diagnostics "${RUN_ROOT}/<development-run>/calibration.diagnostics.json" \
  --model-version <development-model-id> \
  --t0-recall-floor 0.90 \
  --t1-recall-floor 0.85 \
  --fit-temperature
```

This path accepts only canonical synthetic `validation` records that are public-training eligible.
Its output is development evidence and must not be represented as independent confirmatory evidence.

## Selection and diagnostics semantics

Threshold candidates are observed emitted-span scores plus 0 and 1. A prediction is positive only
when `(doc_id, start, end, label)` exactly matches gold. Gold spans never emitted by the model remain
false negatives for every threshold; lowering the threshold cannot pretend to recover a missing
candidate. Calibration fails if maximum emitted-candidate recall cannot meet a configured T0/T1
floor.

T0 and T1 labels optimize F2 after satisfying their recall floor; T2 labels optimize F1. Ties prefer
recall, then precision, then the stricter threshold. The serving-compatible `CalibrationBundle`
contains the global temperature and per-entity thresholds.

Optional temperature fitting is binary span-confidence fitting over emitted predictions only, with
strict exact-match correctness as the target. It does not claim calibration for un-emitted false
negatives. When sample/class minima are not met, temperature remains `1.0` and diagnostics report
`insufficient_statistics`. ECE and Brier use the same emitted-span scope.

Diagnostics are self-hashed and contain only input hashes, counts, per-label TP/FP/FN, thresholds,
ECE/Brier, and declared fitting parameters—never local paths, document IDs, raw text, or entity
values.
