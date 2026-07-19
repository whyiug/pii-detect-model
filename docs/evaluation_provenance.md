# Evaluation provenance and rule baselines

Evaluation reports are raw-text-free, but that alone does not prove which frozen data, model, or
prediction file produced a metric. The evaluation CLIs therefore use self-hashed JSON manifests.
The logical `manifest_sha256` is SHA-256 over the manifest without that field, encoded as sorted,
ASCII JSON with compact separators. Reports also record the SHA-256 of each exact manifest file.

## Freeze PII Bench ZH

The raw and canonical SHA-256 values, expected record counts, and expected span counts are fixed in
`scripts/prepare_evaluation_data.py`. Verify the already-prepared files and generate the read-only
dataset manifest without rewriting either benchmark. Set `$DATA_ROOT` to the external dataset root
before running the command:

```bash
python scripts/prepare_evaluation_data.py \
  --formal "${DATA_ROOT}/evaluation_only/pii_bench_zh/raw/pii_bench_zh.jsonl" \
  --chat "${DATA_ROOT}/evaluation_only/pii_bench_zh/raw/pii_bench_zh_chat.jsonl" \
  --output-dir "${DATA_ROOT}/evaluation_only/pii_bench_zh/canonical" \
  --verify-existing \
  --prepared-at 2026-07-11T05:35:50Z
```

For a reproducible build, `SOURCE_DATE_EPOCH` may replace `--prepared-at`. The command refuses to
use the wall clock. The manifest contains upstream source/revision/license, raw and canonical
hashes, converter version and implementation hash, record/span/label counts, and the irreversible
`evaluation_only` policy. It contains no paths or source text and is written mode `0444`.

## Bind predictions and release evaluation

Model inference can atomically create a deterministic prediction manifest immediately after the
prediction JSONL. The three provenance arguments are an all-or-none release contract. The manifest
binds the exact prediction bytes to the frozen dataset manifest, self-hashed training manifest,
training seed, actual attention mode, and hashes of the loaded config/weights/label schema:

```bash
python scripts/predict.py \
  --model /path/to/model \
  --input /path/to/formal.canonical.jsonl \
  --output /path/to/formal.predictions.jsonl \
  --dataset-manifest /path/to/pii_bench_zh.dataset_manifest.json \
  --training-manifest /path/to/training_manifest.json \
  --prediction-manifest /path/to/formal.prediction_manifest.json
```

The standalone helper remains available for already-created predictions, with its smaller legacy
identity contract:

```bash
python scripts/create_prediction_manifest.py \
  --predictions /path/to/formal.predictions.jsonl \
  --dataset-manifest "${DATA_ROOT}/evaluation_only/pii_bench_zh/canonical/pii_bench_zh.dataset_manifest.json" \
  --model-training-manifest /path/to/training_manifest.json \
  --prediction-id qwen3-bi-seed13-formal-v1 \
  --output /path/to/formal.prediction_manifest.json
```

Development evaluation always records the exact gold and prediction SHA-256 values. Supplying any
manifest enables its validation. Release mode requires all three and rejects a gold/manifest,
prediction/manifest, prediction/dataset, or prediction/training mismatch:

```bash
python scripts/evaluate.py \
  --gold /path/to/pii_bench_zh_formal.canonical.jsonl \
  --predictions /path/to/formal.predictions.jsonl \
  --dataset-manifest /path/to/pii_bench_zh.dataset_manifest.json \
  --prediction-manifest /path/to/formal.prediction_manifest.json \
  --model-training-manifest /path/to/training_manifest.json \
  --release-mode \
  --output /path/to/formal.evaluation.json
```

The report's `provenance` object contains only hashes, counters, controlled source/model identity,
evaluation parameters, and a content hash of the evaluator implementation. It never copies a local
path, raw text, entity value, or manifest body.

`confidence_calibration` evaluates confidence on emitted prediction spans. A positive is an exact
`(start, end, label)` gold match in the same document. Brier score and 10-bin equal-width ECE use
left-closed/right-open bins, with 1.0 included in the last bin. False negatives have no emitted
confidence and are outside this precision-calibration scope. If there are no scores, no prediction
spans, or only some spans are scored, the metrics are `null` with an explicit `not_applicable`
reason; a selectively scored subset is never presented as calibrated evidence.

Threshold selection and temperature fitting use synthetic validation data only; see
[`calibration.md`](calibration.md). Confidence diagnostics here describe emitted-span precision
calibration and do not imply that missed gold spans were recoverable.

## Publication-safe rule baseline summary

Run the rule predictor and evaluator separately for formal and chat. `predict_rules.py` optionally
writes a dataset-bound rule manifest when `--dataset-manifest` and `--prediction-manifest` are both
provided. Its identity contains hashes of the rules implementation and configuration and never
fabricates a model training hash. Both evaluation reports must
be bound to the same frozen dataset manifest and include document-level bootstrap intervals. Then
produce the compact, self-hashed summary:

```bash
python scripts/summarize_rule_baseline.py \
  --formal-report /path/to/rules_formal.evaluation.json \
  --chat-report /path/to/rules_chat.evaluation.json \
  --output /path/to/rules_baseline.summary.json
```

The summary binds each report and prediction SHA-256, the evaluation identity, the frozen subset,
and a content hash over the rule implementation and configuration. It includes only aggregate
metrics and confidence intervals; output is written atomically as mode `0444`.
