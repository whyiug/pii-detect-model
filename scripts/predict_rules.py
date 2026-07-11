#!/usr/bin/env python3
"""Generate raw-text-free cn_common rule predictions for a canonical dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pii_zh.data.schema import iter_jsonl  # noqa: E402
from pii_zh.evaluation import (  # noqa: E402
    PredictionRecord,
    Span,
    build_rule_prediction_manifest,
    canonical_json_hash,
    sha256_file,
    write_prediction_jsonl,
)
from pii_zh.rules import CN_COMMON_RULES, CnCommonRulePack  # noqa: E402
from pii_zh.rules import cn_common as cn_common_module  # noqa: E402

RULE_TO_MODEL = {
    "CN_ID_CARD": "CN_RESIDENT_ID",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "BANK_CARD_NUMBER": "BANK_CARD_NUMBER",
    "IP_ADDRESS": "IP_ADDRESS",
    "MAC_ADDRESS": "MAC_ADDRESS",
    "GEO_COORDINATE": "GEO_COORDINATE",
    "PASSPORT_NUMBER": "PASSPORT_NUMBER",
    "DRIVER_LICENSE_NUMBER": "DRIVER_LICENSE_NUMBER",
    "CN_VEHICLE_LICENSE_PLATE": "VEHICLE_LICENSE_PLATE",
    "SECRET": "SECRET",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--prediction-manifest", type=Path)
    return parser.parse_args()


def _write_manifest_atomic(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
        prefix=f".{destination.name}.",
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)
    destination.chmod(0o444)


def main() -> int:
    args = parse_args()
    if (args.dataset_manifest is None) != (args.prediction_manifest is None):
        raise ValueError("--dataset-manifest and --prediction-manifest must be supplied together")
    if args.prediction_manifest is not None and (
        args.prediction_manifest.resolve() == args.output.resolve()
    ):
        raise ValueError("--prediction-manifest and --output must be different files")
    rule_pack = CnCommonRulePack()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False, prefix=f".{args.output.name}."
    ) as handle:
        temporary = Path(handle.name)
        try:
            for record in iter_jsonl(args.input):
                if record.doc_id in seen:
                    raise ValueError("input contains a duplicate doc_id")
                seen.add(record.doc_id)
                matches = rule_pack.analyze(record.text, entities=list(RULE_TO_MODEL))
                prediction = PredictionRecord(
                    doc_id=record.doc_id,
                    spans=tuple(
                        Span(
                            start=match.start,
                            end=match.end,
                            label=RULE_TO_MODEL[match.entity_type],
                            score=match.score,
                        )
                        for match in matches
                        if match.entity_type in RULE_TO_MODEL
                    ),
                )
                write_prediction_jsonl([prediction], handle)
                count += 1
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, args.output)
    prediction_manifest_sha256 = None
    if args.prediction_manifest is not None:
        rule_configuration = {
            "mapping": RULE_TO_MODEL,
            "rules": [
                {
                    "already_masked": rule.already_masked,
                    "entity_type": rule.entity_type,
                    "pattern_id": rule.pattern_id,
                    "required_context": list(rule.required_context),
                    "score": rule.score,
                    "validator_label": rule.validator_label,
                }
                for rule in CN_COMMON_RULES
            ],
        }
        manifest = build_rule_prediction_manifest(
            predictions_path=args.output,
            dataset_manifest_path=args.dataset_manifest,
            prediction_document_count=count,
            ruleset_id="cn_common_v1",
            implementation_sha256=sha256_file(Path(cn_common_module.__file__)),
            configuration_sha256=canonical_json_hash(rule_configuration),
        )
        _write_manifest_atomic(manifest, args.prediction_manifest)
        prediction_manifest_sha256 = manifest["manifest_sha256"]
    print(
        json.dumps(
            {
                "documents": count,
                "output": args.output.name,
                "prediction_manifest_sha256": prediction_manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
