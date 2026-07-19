from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pii_zh.data.schema import (
    DocumentRecord,
    Provenance,
    QualityMetadata,
    QualityTier,
    write_jsonl,
)
from pii_zh.data.validators import luhn_check_digit
from pii_zh.evaluation import (
    PredictionRecord,
    Span,
    add_manifest_hash,
    canonical_json_hash,
    load_prediction_jsonl,
    write_prediction_jsonl,
)


def _synthetic_record(*, doc_id: str, text: str) -> DocumentRecord:
    return DocumentRecord.create(
        doc_id=doc_id,
        text=text,
        entities=[],
        language="zh-Hans-CN",
        domain="fixture",
        scene="rule-cli",
        provenance=Provenance(
            source_id="synthetic-rule-fixture",
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
        split="validation",
    )


def test_predict_rules_cli_emits_v5_account_semantics_and_path_free_manifest(
    tmp_path: Path,
) -> None:
    prefix = "622202123456789"
    number = prefix + luhn_check_digit(prefix)
    account_text = f"收款账户：{number}"
    card_text = f"工资账户绑定的银行卡号：{number}"
    source = tmp_path / "input.jsonl"
    output = tmp_path / "rules.jsonl"
    manifest_path = tmp_path / "rules.manifest.json"
    dataset_manifest_path = tmp_path / "dataset.manifest.json"
    write_jsonl(
        [
            _synthetic_record(doc_id="account-fixture", text=account_text),
            _synthetic_record(doc_id="card-fixture", text=card_text),
        ],
        source,
    )
    dataset_manifest = add_manifest_hash(
        {"schema_version": 1, "manifest_type": "evaluation_dataset"}
    )
    dataset_manifest_path.write_text(json.dumps(dataset_manifest), encoding="utf-8")
    repository_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/predict_rules.py"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--dataset-manifest",
            str(dataset_manifest_path),
            "--prediction-manifest",
            str(manifest_path),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    records = load_prediction_jsonl(output)
    assert [[span.label for span in record.spans] for record in records] == [
        ["BANK_ACCOUNT_NUMBER"],
        ["BANK_CARD_NUMBER"],
    ]
    assert [
        account_text[records[0].spans[0].start : records[0].spans[0].end],
        card_text[records[1].spans[0].start : records[1].spans[0].end],
    ] == [number, number]
    serialized = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(serialized)
    unsigned = dict(manifest)
    assert unsigned.pop("manifest_sha256") == canonical_json_hash(unsigned)
    assert manifest["rules_identity"]["ruleset_id"] == "cn_common_v5"
    assert str(tmp_path) not in serialized
    assert account_text not in serialized
    assert card_text not in serialized
    assert "account-fixture" not in serialized
    assert manifest_path.stat().st_mode & 0o777 == 0o444


def test_predict_rules_cli_restricts_outputs_to_closed_protocol(tmp_path: Path) -> None:
    prefix = "622202123456789"
    number = prefix + luhn_check_digit(prefix)
    source = tmp_path / "input.jsonl"
    output = tmp_path / "rules.jsonl"
    write_jsonl(
        [
            _synthetic_record(doc_id="account-fixture", text=f"收款账户：{number}"),
            _synthetic_record(doc_id="card-fixture", text=f"银行卡号：{number}"),
        ],
        source,
    )
    repository_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/predict_rules.py"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--allowed-label",
            "BANK_CARD_NUMBER",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    records = load_prediction_jsonl(output)
    assert records[0].spans == ()
    assert [span.label for span in records[1].spans] == ["BANK_CARD_NUMBER"]


def test_refinement_cli_suppresses_business_dob_and_keeps_path_text_free_audit(
    tmp_path: Path,
) -> None:
    business_text = "版本计划发布日期为2026-07-11"
    birth_text = "出生日期：1990-01-02"
    documents = tmp_path / "documents.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "refined.jsonl"
    audit_path = tmp_path / "refinement.audit.json"
    write_jsonl(
        [
            _synthetic_record(doc_id="business-date", text=business_text),
            _synthetic_record(doc_id="birth-date", text=birth_text),
        ],
        documents,
    )
    business_value = "2026-07-11"
    birth_value = "1990-01-02"
    business_start = business_text.index(business_value)
    birth_start = birth_text.index(birth_value)
    write_prediction_jsonl(
        [
            PredictionRecord(
                "business-date",
                (
                    Span(
                        business_start,
                        business_start + len(business_value),
                        "DATE_OF_BIRTH",
                        0.9,
                    ),
                ),
            ),
            PredictionRecord(
                "birth-date",
                (
                    Span(
                        birth_start,
                        birth_start + len(birth_value),
                        "DATE_OF_BIRTH",
                        0.9,
                    ),
                ),
            ),
        ],
        predictions,
    )
    repository_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/refine_predictions.py"),
            "--documents",
            str(documents),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
            "--audit",
            str(audit_path),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    records = load_prediction_jsonl(output)
    assert records[0].spans == ()
    assert [span.label for span in records[1].spans] == ["DATE_OF_BIRTH"]
    serialized = audit_path.read_text(encoding="utf-8")
    audit = json.loads(serialized)
    unsigned = dict(audit)
    assert unsigned.pop("manifest_sha256") == canonical_json_hash(unsigned)
    assert audit["refinement_id"] == "structured_refinement_v4"
    assert audit["suppressed_by_label"] == {"DATE_OF_BIRTH": 1}
    assert all(value is False for value in audit["privacy"].values())
    assert str(tmp_path) not in serialized
    assert business_text not in serialized
    assert birth_text not in serialized
    assert "business-date" not in serialized
    assert "birth-date" not in serialized
    assert audit_path.stat().st_mode & 0o777 == 0o444
