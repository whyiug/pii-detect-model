from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pii_zh.calibration import CalibrationBundle
from pii_zh.data.schema import (
    DocumentRecord,
    EntitySpan,
    Provenance,
    QualityMetadata,
    QualityTier,
    write_jsonl,
)
from pii_zh.evaluation import (
    PredictionRecord,
    Span,
    canonical_json_hash,
    write_prediction_jsonl,
)


def _synthetic_record(*, doc_id: str, text: str, entity_text: str, split: str) -> DocumentRecord:
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=[
            EntitySpan(
                start=0,
                end=len(entity_text),
                label="PERSON_NAME",
                text=entity_text,
                source="synthetic",
            )
        ],
        language="zh-Hans-CN",
        domain="fixture",
        scene="calibration",
        provenance=Provenance(
            source_id="synthetic-fixture",
            source_kind="synthetic",
            source_revision="v1",
            license="Apache-2.0",
            synthetic=True,
            generator_name="fixture",
            generator_version="v1",
        ),
        quality=QualityMetadata(
            tier=QualityTier.SYNTHETIC_VALIDATED,
            quality_gate=True,
            validators_passed=True,
            review_status="validated",
        ),
        public_weight_training_allowed=True,
        generator_version="v1",
        split=split,
    )


def _run_calibration(
    tmp_path: Path, *, split: str = "validation"
) -> subprocess.CompletedProcess[str]:
    gold_path = tmp_path / "gold.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "calibration.json"
    diagnostics_path = tmp_path / "calibration.diagnostics.json"
    write_jsonl(
        [
            _synthetic_record(
                doc_id="validation-one",
                text="张三登记",
                entity_text="张三",
                split=split,
            ),
            _synthetic_record(
                doc_id="validation-two",
                text="李四登记",
                entity_text="李四",
                split=split,
            ),
        ],
        gold_path,
    )
    write_prediction_jsonl(
        [
            PredictionRecord(
                "validation-one",
                (
                    Span(0, 2, "PERSON_NAME", score=0.9),
                    Span(2, 4, "PERSON_NAME", score=0.8),
                ),
            ),
            PredictionRecord("validation-two", ()),
        ],
        prediction_path,
    )
    repository_root = Path(__file__).resolve().parents[3]
    return subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/calibrate_predictions.py"),
            "--gold",
            str(gold_path),
            "--predictions",
            str(prediction_path),
            "--output",
            str(output_path),
            "--diagnostics",
            str(diagnostics_path),
            "--model-version",
            "fixture-model-v1",
            "--t0-recall-floor",
            "0.5",
            "--t1-recall-floor",
            "0.5",
            "--fit-temperature",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_writes_bundle_and_self_hashed_raw_free_diagnostics(tmp_path: Path) -> None:
    completed = _run_calibration(tmp_path)
    assert completed.returncode == 0, completed.stderr

    output_path = tmp_path / "calibration.json"
    diagnostics_path = tmp_path / "calibration.diagnostics.json"
    bundle = CalibrationBundle.from_json(output_path)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    unsigned = dict(diagnostics)
    logical_hash = unsigned.pop("manifest_sha256")
    assert canonical_json_hash(unsigned) == logical_hash
    assert bundle.global_temperature == 1.0
    assert bundle.entity_thresholds["PERSON_NAME"] == 0.9
    assert diagnostics["temperature"]["status"] == "insufficient_statistics"
    assert diagnostics["inputs"]["un_emitted_gold_span_count"] == 1
    person = diagnostics["per_label"]["PERSON_NAME"]
    assert person["un_emitted_gold_fixed_fn"] == 1
    assert person["tp"] == 1
    assert person["fp"] == 0
    assert person["fn"] == 1
    serialized = diagnostics_path.read_text(encoding="utf-8")
    assert "张三" not in serialized
    assert "李四" not in serialized
    assert str(tmp_path) not in serialized
    assert output_path.stat().st_mode & 0o777 == 0o444
    assert diagnostics_path.stat().st_mode & 0o777 == 0o444


def test_cli_rejects_test_split_without_writing_outputs(tmp_path: Path) -> None:
    completed = _run_calibration(tmp_path, split="test")
    assert completed.returncode != 0
    assert "only the validation split" in completed.stderr
    assert not (tmp_path / "calibration.json").exists()
    assert not (tmp_path / "calibration.diagnostics.json").exists()
