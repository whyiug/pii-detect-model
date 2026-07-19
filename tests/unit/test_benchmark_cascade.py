from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import benchmark_cascade

from pii_zh.data.schema import (
    DocumentRecord,
    Provenance,
    QualityMetadata,
    QualityTier,
    write_jsonl,
)


def _canonical_record(*, doc_id: str, text: str) -> DocumentRecord:
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=(),
        language="zh-Hans-CN",
        domain="synthetic-test",
        scene="resource-smoke",
        provenance=Provenance(
            source_id="unit-test",
            source_kind="synthetic",
            source_revision="v1",
            license="Apache-2.0",
            synthetic=True,
            generator_name="unit-test",
            generator_version="v1",
        ),
        quality=QualityMetadata(
            tier=QualityTier.SYNTHETIC_VALIDATED,
            quality_gate=True,
            validators_passed=True,
            review_status="synthetic_fixture",
        ),
        public_weight_training_allowed=True,
        generator_version="v1",
    )


def test_builtin_benchmark_calls_published_rules_only_pipeline_and_is_aggregate_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, ...]] = []
    configs = []

    class SpyPipeline:
        def __init__(self, *, config: object) -> None:
            configs.append(config)

        def detect_batch(self, texts: object) -> list[list[object]]:
            values = tuple(texts)  # type: ignore[arg-type]
            calls.append(values)
            return [[] for _ in values]

    monkeypatch.setattr(benchmark_cascade, "CascadePipeline", SpyPipeline)
    output = tmp_path / "resource-report.json"

    status = benchmark_cascade.main(
        [
            "--builtin-synthetic",
            "--output",
            str(output),
            "--batch-size",
            "3",
            "--warmup",
            "1",
            "--repetitions",
            "2",
        ]
    )

    assert status == 0
    assert len(configs) == 1
    assert configs[0].mode == "rules-only"
    assert configs[0].uses_model is False
    assert len(calls) == 6
    report_text = output.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["schema_version"] == benchmark_cascade.SCHEMA_VERSION
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["claim_scope"]["release_performance_conclusion"] is False
    assert report["runtime"]["mode"] == "rules-only"
    assert report["runtime"]["gpu_used"] is False
    assert report["runtime"]["model_loaded"] is False
    assert len(report["runtime"]["configuration_sha256"]) == 64
    assert len(report["runtime"]["implementation"]["implementation_sha256"]) == 64
    assert report["workload"]["source_kind"] == "builtin_synthetic"
    assert report["measurements"]["warmup"]["passes"] == 1
    assert report["measurements"]["hot_path"]["repetitions"] == 2
    assert (
        report["measurements"]["hot_path"]["latency_milliseconds"]["per_batch_call"]["sample_count"]
        == 4
    )
    assert report["measurements"]["hot_path"]["throughput"]["docs_per_second"] > 0
    assert report["measurements"]["hot_path"]["throughput"]["chars_per_second"] > 0
    assert report["measurements"]["model_call_rate"]["calls_per_document"] == 0.0
    assert report["measurements"]["cpu"]["max_rss_after"]["status"] == "available"
    assert report["measurements"]["startup"]["cold_process_start"] == {
        "status": "unavailable",
        "milliseconds": None,
        "reason": (
            "reliable process cold-start measurement requires an external supervisor "
            "and controlled filesystem cache state"
        ),
    }
    assert report["privacy"] == {
        "aggregate_only": True,
        "contains_absolute_paths": False,
        "contains_document_ids": False,
        "contains_entity_values": False,
        "contains_raw_text": False,
    }
    for text in benchmark_cascade._BUILTIN_SYNTHETIC_TEXTS:
        assert text not in report_text
    assert str(tmp_path) not in report_text
    assert output.stat().st_mode & 0o222 == 0
    captured = capsys.readouterr()
    assert str(tmp_path) not in captured.out


def test_local_canonical_jsonl_is_hashed_but_content_and_paths_are_not_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = "do-not-retain@example.com"
    input_path = tmp_path / "private-looking-name.canonical.jsonl"
    write_jsonl(
        [_canonical_record(doc_id="private-looking-document-id", text=sentinel)],
        input_path,
    )
    output = tmp_path / "aggregate.json"

    status = benchmark_cascade.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--warmup",
            "0",
            "--repetitions",
            "1",
        ]
    )

    assert status == 0
    serialized = output.read_text(encoding="utf-8")
    report = json.loads(serialized)
    assert report["workload"]["source_kind"] == "local_canonical_jsonl"
    assert report["workload"]["document_count"] == 1
    assert report["workload"]["character_count"] == len(sentinel)
    assert len(report["workload"]["content_sha256"]) == 64
    assert sentinel not in serialized
    assert "private-looking-document-id" not in serialized
    assert input_path.name not in serialized
    assert str(tmp_path) not in serialized
    captured = capsys.readouterr()
    assert sentinel not in captured.out
    assert str(tmp_path) not in captured.out


def test_invalid_input_is_sanitized_and_does_not_publish_partial_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = "must-not-echo-secret-value"
    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text(json.dumps({"text": sentinel}) + "\n", encoding="utf-8")
    output = tmp_path / "aggregate.json"

    status = benchmark_cascade.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output),
        ]
    )

    assert status == 2
    assert not output.exists()
    captured = capsys.readouterr()
    assert sentinel not in captured.err
    assert str(tmp_path) not in captured.err
    assert "canonical input validation failed" in captured.err


def test_existing_output_is_never_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("sentinel", encoding="utf-8")

    status = benchmark_cascade.main(
        [
            "--builtin-synthetic",
            "--output",
            str(output),
            "--warmup",
            "0",
            "--repetitions",
            "1",
        ]
    )

    assert status == 2
    assert output.read_text(encoding="utf-8") == "sentinel"
    assert "refusing to overwrite" in capsys.readouterr().err


def test_single_measurement_percentiles_are_well_defined() -> None:
    assert benchmark_cascade._summary_ms([1_250_000]) == {
        "sample_count": 1,
        "p50": 1.25,
        "p95": 1.25,
    }
