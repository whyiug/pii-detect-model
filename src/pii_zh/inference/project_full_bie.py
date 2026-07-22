"""Independent full-attention BIE73 artifact verification and inference.

This protocol is deliberately separate from the native-causal BIE loader.  It
validates raw ``config.json`` before Transformers config/model construction,
requires one merged safetensors weight file, and keeps the research artifact
release-ineligible after Qwen3Bi deserialization.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional inference dependency
    raise ImportError("Full BIE73 inference requires PyTorch.") from exc

from pii_zh.inference.aiguard import decode_bie_ids
from pii_zh.inference.bie_decoding import (
    BieDecoderId,
    BieDecodingError,
    decode_bie_logits,
    normalize_bie_decoder_id,
)
from pii_zh.inference.project_bie import (
    PROJECT_CLOSED8_LABELS,
    normalize_project_bie_id2label,
    project_bie_allowed_label_ids,
)
from pii_zh.models.aiguard24 import AIGUARD_SOURCE_MODEL_ID, AIGUARD_SOURCE_REVISION
from pii_zh.models.aiguard24_bie import (
    AIGUARD24_BIE_INITIALIZATION_STRATEGY,
    build_core_bie_label_maps,
)
from pii_zh.models.bie_attention_conversion import (
    BIE73_FULL_ATTENTION_CONVERSION_STRATEGY,
)
from pii_zh.models.qwen3_bi import (
    ARCHITECTURE_VERSION,
    Qwen3BiForTokenClassification,
)
from pii_zh.taxonomy import load_taxonomy
from pii_zh.tokenization import load_serialized_fast_tokenizer
from pii_zh.training.manifest import (
    canonical_json_hash,
    verify_output_artifact_binding,
    verify_training_manifest,
)

FULL_BIE73_MANIFEST_TYPE = "aiguard24_full_attention_bie_training_v1"
FULL_BIE73_PROTOCOL_ID = "pii_zh_core24_full_attention_bie_v1"
FULL_BIE73_MODEL_PROTOCOL = "FULL_BIE73"
FULL_BIE73_MODEL_IDENTITY_SCHEMA_VERSION = "pii-zh.full-bie73-model-identity.v1"
FULL_BIE73_INITIALIZATION_STRATEGY = "aiguard24_bie_projection_then_full_attention_conversion_v1"
FULL_BIE73_ATTENTION_BACKEND = "sdpa"
FULL_BIE73_CLOSED8_LABELS = PROJECT_CLOSED8_LABELS
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_AUTO_MAP = {
    "AutoConfig": "configuration_qwen3_bi.Qwen3BiConfig",
    "AutoModelForTokenClassification": "modeling_qwen3_bi.Qwen3BiForTokenClassification",
}
_REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_type",
        "status",
        "release_eligible",
        "created_at",
        "completed_at",
        "seed",
        "attention_mode",
        "tag_scheme",
        "update_mode",
        "validation_split_name",
        "base_source_id",
        "source_revision",
        "training_source_ids",
        "taxonomy_version",
        "label2id",
        "label_schema_sha256",
        "recipe",
        "recipe_sha256",
        "initialization",
        "parameter_update_audit",
        "tokenizer",
        "datasets",
        "split_isolation",
        "dev_selection_gates",
        "benchmark_isolation",
        "versions",
        "code_revision",
        "privacy",
        "runtime_environment",
        "checkpoint_selection",
        "independent_validation",
        "output_artifact",
        "manifest_sha256",
    }
)
_CONVERSION_AUDIT_FIELDS = frozenset(
    {
        "strategy",
        "source_architecture",
        "target_architecture",
        "source_model_type",
        "target_model_type",
        "source_attention_mode",
        "target_attention_mode",
        "attention_backend",
        "tagging_scheme",
        "label_count",
        "state_dict_key_count",
        "state_dict_keys_identical",
        "state_dict_values_identical",
        "classifier_keys",
        "classifier_values_identical",
        "config_payload_identical_except_attention_contract",
        "preserved_config_key_count",
        "trainability_preserved",
        "source_training_mode_preserved",
        "source_device",
        "target_device",
        "lora_target_module_counts",
        "newly_initialized_parameter_keys",
        "discarded_parameter_keys",
        "release_eligible",
    }
)


class FullBie73Error(RuntimeError):
    """Raised when a full-attention BIE73 artifact or runtime fails closed."""


@dataclass(frozen=True, slots=True)
class FullBie73ModelIdentity:
    """Path-free identity for a verified full-attention BIE73 artifact."""

    schema_version: str
    manifest_type: str
    manifest_sha256: str
    model_protocol: str
    protocol_id: str
    model_type: str
    attention_mode: str
    attention_backend: str
    tag_scheme: str
    taxonomy_version: str
    label_count: int
    label_schema_sha256: str
    weights_combined_sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerifiedFullBie73Model:
    root: Path
    identity: FullBie73ModelIdentity


def _json_object(path: Path, *, kind: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FullBie73Error(f"full BIE73 {kind} must be a regular file")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise FullBie73Error(f"full BIE73 {kind} contains duplicate keys")
            value[key] = item
        return value

    def reject_constant(_value: str) -> None:
        raise FullBie73Error(f"full BIE73 {kind} contains a non-finite number")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except FullBie73Error:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise FullBie73Error(f"full BIE73 {kind} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise FullBie73Error(f"full BIE73 {kind} must be a JSON object")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise FullBie73Error(f"full BIE73 {field} must be a SHA-256 value")
    return value


def _verify_checkpoint_receipt(value: object) -> None:
    if not isinstance(value, Mapping):
        raise FullBie73Error("full BIE73 checkpoint selection receipt is missing")
    step = value.get("selected_global_step")
    best = value.get("best_metric")
    replay = value.get("final_validation_replay_metric")
    hashes = value.get("safetensor_sha256")
    if (
        value.get("selection_split") != "independent_validation"
        or value.get("selection_metric") != "independent_dev_selection_score"
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 1
        or value.get("selected_checkpoint_id") != f"checkpoint-{step}"
        or isinstance(best, bool)
        or not isinstance(best, (int, float))
        or not math.isfinite(float(best))
        or isinstance(replay, bool)
        or not isinstance(replay, (int, float))
        or not math.isfinite(float(replay))
        or not math.isclose(float(best), float(replay), rel_tol=1.0e-6, abs_tol=1.0e-8)
        or not isinstance(hashes, Mapping)
        or not hashes
        or any(not isinstance(name, str) or not name.endswith(".safetensors") for name in hashes)
        or any(
            not isinstance(item, str) or _HEX_SHA256.fullmatch(item) is None
            for item in hashes.values()
        )
    ):
        raise FullBie73Error("full BIE73 checkpoint selection receipt is invalid")


def _verify_conversion_audit(value: object, *, recipe: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != _CONVERSION_AUDIT_FIELDS:
        raise FullBie73Error("full BIE73 attention-conversion audit is incomplete")
    state_count = value.get("state_dict_key_count")
    preserved_config_count = value.get("preserved_config_key_count")
    if (
        value.get("strategy") != BIE73_FULL_ATTENTION_CONVERSION_STRATEGY
        or value.get("source_architecture") != "Qwen3ForTokenClassification"
        or value.get("target_architecture") != "Qwen3BiForTokenClassification"
        or value.get("source_model_type") != "qwen3"
        or value.get("target_model_type") != "qwen3_bi"
        or value.get("source_attention_mode") != "causal"
        or value.get("target_attention_mode") != "full"
        or value.get("attention_backend") != FULL_BIE73_ATTENTION_BACKEND
        or value.get("tagging_scheme") != "BIE"
        or value.get("label_count") != 73
        or isinstance(state_count, bool)
        or not isinstance(state_count, int)
        or state_count < 1
        or value.get("state_dict_keys_identical") is not True
        or value.get("state_dict_values_identical") is not True
        or value.get("classifier_keys") != ["score.weight", "score.bias"]
        or value.get("classifier_values_identical") is not True
        or value.get("config_payload_identical_except_attention_contract") is not True
        or isinstance(preserved_config_count, bool)
        or not isinstance(preserved_config_count, int)
        or preserved_config_count < 1
        or value.get("trainability_preserved") is not True
        or value.get("source_training_mode_preserved") is not True
        or value.get("source_device") != "cpu"
        or value.get("target_device") != "cpu"
        or value.get("newly_initialized_parameter_keys") != []
        or value.get("discarded_parameter_keys") != []
        or value.get("release_eligible") is not False
    ):
        raise FullBie73Error("full BIE73 attention-conversion audit is invalid")

    update_mode = recipe.get("update_mode")
    counts = value.get("lora_target_module_counts")
    lora = recipe.get("lora")
    if update_mode == "lora":
        targets = lora.get("target_modules") if isinstance(lora, Mapping) else None
        if (
            not isinstance(targets, list)
            or not targets
            or not isinstance(counts, list)
            or len(counts) != len(targets)
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or item[0] != target
                or isinstance(item[1], bool)
                or not isinstance(item[1], int)
                or item[1] < 1
                for item, target in zip(counts, targets, strict=True)
            )
        ):
            raise FullBie73Error("full BIE73 LoRA conversion audit is invalid")
    elif counts != []:
        raise FullBie73Error("head-only full BIE73 conversion must not bind LoRA targets")


def verify_full_bie73_training_manifest(
    manifest: dict[str, Any],
) -> FullBie73ModelIdentity:
    """Verify the exact trainer-facing full-BIE73 manifest contract."""

    label2id, _ = build_core_bie_label_maps()
    taxonomy = load_taxonomy()
    if not _REQUIRED_MANIFEST_FIELDS <= manifest.keys():
        raise FullBie73Error("full BIE73 completed training manifest is incomplete")
    if (
        not verify_training_manifest(manifest)
        or manifest.get("schema_version") != 1
        or manifest.get("manifest_type") != FULL_BIE73_MANIFEST_TYPE
        or manifest.get("status") != "completed"
        or manifest.get("release_eligible") is not False
        or manifest.get("attention_mode") != "full"
        or manifest.get("tag_scheme") != "BIE"
        or manifest.get("update_mode") not in {"head_only", "lora"}
        or manifest.get("validation_split_name") != "dev"
        or manifest.get("base_source_id") != AIGUARD_SOURCE_MODEL_ID
        or manifest.get("source_revision") != AIGUARD_SOURCE_REVISION
        or manifest.get("taxonomy_version") != taxonomy.taxonomy_version
        or manifest.get("label2id") != label2id
        or manifest.get("label_schema_sha256") != canonical_json_hash(label2id)
    ):
        raise FullBie73Error("full BIE73 training manifest protocol is invalid")
    recipe = manifest.get("recipe")
    update_mode = manifest.get("update_mode")
    if (
        not isinstance(recipe, Mapping)
        or recipe.get("attention_mode") != "full"
        or recipe.get("attention_backend") != FULL_BIE73_ATTENTION_BACKEND
        or recipe.get("tag_scheme") != "BIE"
        or recipe.get("label_count") != 73
        or recipe.get("update_mode") != update_mode
        or recipe.get("initialization_strategy") != FULL_BIE73_INITIALIZATION_STRATEGY
        or recipe.get("checkpoint_selection_split") != "independent_validation"
        or recipe.get("checkpoint_selection_metric") != "independent_dev_selection_score"
        or manifest.get("recipe_sha256") != canonical_json_hash(recipe)
    ):
        raise FullBie73Error("full BIE73 training recipe receipt is invalid")
    initialization = manifest.get("initialization")
    if (
        not isinstance(initialization, Mapping)
        or initialization.get("strategy") != AIGUARD24_BIE_INITIALIZATION_STRATEGY
        or initialization.get("source_model_id") != AIGUARD_SOURCE_MODEL_ID
        or initialization.get("source_revision") != AIGUARD_SOURCE_REVISION
        or initialization.get("target_tag_scheme") != "BIE"
        or initialization.get("target_label_count") != 73
        or initialization.get("target_label2id") != label2id
        or initialization.get("attention_mode") != "causal"
        or initialization.get("release_eligible") is not False
    ):
        raise FullBie73Error("full BIE73 projection initialization receipt is invalid")
    _verify_conversion_audit(initialization.get("attention_conversion"), recipe=recipe)

    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != {"train", "independent_validation"}:
        raise FullBie73Error("full BIE73 training dataset receipts are incomplete")
    for receipt in datasets.values():
        dataset_sha256 = receipt.get("sha256") if isinstance(receipt, Mapping) else None
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(dataset_sha256, str)
            or _HEX_SHA256.fullmatch(dataset_sha256) is None
            or not isinstance(receipt.get("summary"), Mapping)
            or not isinstance(receipt.get("admission"), Mapping)
        ):
            raise FullBie73Error("full BIE73 training dataset receipt is invalid")
    isolation = manifest.get("split_isolation")
    if (
        not isinstance(isolation, Mapping)
        or isolation.get("policy") != "strict_all_groups"
        or isolation.get("collision_counts")
        != {
            "document_id": 0,
            "template_group": 0,
            "entity_value_group": 0,
            "source_group": 0,
        }
        or isolation.get("template_group_overlap_allowed") is not False
    ):
        raise FullBie73Error("full BIE73 split-isolation receipt is invalid")
    benchmark = manifest.get("benchmark_isolation")
    if not isinstance(benchmark, Mapping) or any(
        benchmark.get(field) is not False
        for field in (
            "public_test_read_for_training",
            "public_test_read_for_validation",
            "public_test_read_for_checkpoint_selection",
            "pii_bench_zh_read",
        )
    ):
        raise FullBie73Error("full BIE73 benchmark-isolation receipt is invalid")
    privacy = manifest.get("privacy")
    if (
        not isinstance(privacy, Mapping)
        or privacy.get("contains_raw_text") is not False
        or privacy.get("contains_entity_values") is not False
        or privacy.get("trainer_reporting_integrations") != []
    ):
        raise FullBie73Error("full BIE73 privacy receipt is invalid")
    validation = manifest.get("independent_validation")
    if not isinstance(validation, Mapping) or validation.get("eval_dev_gate_passed") != 1.0:
        raise FullBie73Error("full BIE73 independent-validation gate is incomplete")
    _verify_checkpoint_receipt(manifest.get("checkpoint_selection"))
    output = manifest.get("output_artifact")
    if not isinstance(output, Mapping):
        raise FullBie73Error("full BIE73 output artifact receipt is missing")
    return FullBie73ModelIdentity(
        schema_version=FULL_BIE73_MODEL_IDENTITY_SCHEMA_VERSION,
        manifest_type=FULL_BIE73_MANIFEST_TYPE,
        manifest_sha256=_sha256(manifest.get("manifest_sha256"), field="manifest hash"),
        model_protocol=FULL_BIE73_MODEL_PROTOCOL,
        protocol_id=FULL_BIE73_PROTOCOL_ID,
        model_type="qwen3_bi",
        attention_mode="full",
        attention_backend=FULL_BIE73_ATTENTION_BACKEND,
        tag_scheme="BIE",
        taxonomy_version=taxonomy.taxonomy_version,
        label_count=73,
        label_schema_sha256=_sha256(manifest.get("label_schema_sha256"), field="label schema hash"),
        weights_combined_sha256=_sha256(
            output.get("weights_combined_sha256"), field="combined weights hash"
        ),
    )


def _verify_full_bie73_config(config: Mapping[str, Any]) -> None:
    """Validate raw JSON or a post-load config projection identically."""

    label2id, expected_id2label = build_core_bie_label_maps()
    try:
        id2label = normalize_project_bie_id2label(config.get("id2label", {}))
    except (TypeError, ValueError) as exc:
        raise FullBie73Error("full BIE73 config label schema is invalid") from exc
    num_labels = config.get("num_labels")
    layer_types = config.get("layer_types")
    num_hidden_layers = config.get("num_hidden_layers")
    if (
        config.get("model_type") != "qwen3_bi"
        or config.get("architectures") != ["Qwen3BiForTokenClassification"]
        or config.get("pii_attention_mode") != "full"
        or config.get("bi_attention_backend") != FULL_BIE73_ATTENTION_BACKEND
        or config.get("architecture_version") != ARCHITECTURE_VERSION
        or config.get("pii_tagging_scheme") != "BIE"
        or config.get("pii_training_status") != "completed_independent_dev_feasible_candidate"
        or config.get("pii_taxonomy_version") != load_taxonomy().taxonomy_version
        or config.get("pii_release_eligible") is not False
        or config.get("use_cache") is not False
        or config.get("auto_map") != _EXPECTED_AUTO_MAP
        or not isinstance(layer_types, list)
        or isinstance(num_hidden_layers, bool)
        or not isinstance(num_hidden_layers, int)
        or num_hidden_layers < 1
        or len(layer_types) != num_hidden_layers
        or any(layer_type != "full_attention" for layer_type in layer_types)
        or config.get("label2id") != label2id
        or id2label != expected_id2label
        or (num_labels is not None and num_labels != 73)
    ):
        raise FullBie73Error("full BIE73 config protocol is invalid")


def verify_full_bie73_artifact(model_path: str | Path) -> VerifiedFullBie73Model:
    """Verify raw manifest/config and a merged safetensors-only local artifact."""

    candidate = Path(model_path).expanduser()
    if candidate.is_symlink():
        raise FullBie73Error("full BIE73 model root must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FullBie73Error("full BIE73 model root must be a local directory") from exc
    if not root.is_dir() or any(item.is_symlink() for item in root.rglob("*")):
        raise FullBie73Error("full BIE73 artifact must contain no symlinks")
    if any(
        item.is_file() and item.suffix.lower() in {".bin", ".pkl", ".pickle", ".pt", ".pth"}
        for item in root.rglob("*")
    ) or any(root.glob("adapter_*")):
        raise FullBie73Error("full BIE73 artifact must be merged safetensors-only")
    weights = tuple(
        sorted(
            item.relative_to(root).as_posix()
            for item in root.rglob("*.safetensors")
            if item.is_file()
        )
    )
    if weights != ("model.safetensors",):
        raise FullBie73Error("full BIE73 artifact must contain exactly model.safetensors")
    for name in (
        "training_manifest.json",
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        artifact = root / name
        if artifact.is_symlink() or not artifact.is_file():
            raise FullBie73Error("full BIE73 standalone artifact inventory is incomplete")

    manifest = _json_object(root / "training_manifest.json", kind="training manifest")
    identity = verify_full_bie73_training_manifest(manifest)
    # This raw JSON check intentionally precedes Qwen3BiConfig.from_pretrained.
    _verify_full_bie73_config(_json_object(root / "config.json", kind="raw model config"))
    if not verify_output_artifact_binding(manifest, root, require_single_weight=True):
        raise FullBie73Error("full BIE73 manifest-to-safetensors binding failed")
    return VerifiedFullBie73Model(root=root, identity=identity)


class FullBie73SpanPredictor:
    """Window-compatible full-attention BIE73 predictor with a bound decoder."""

    protocol_id = FULL_BIE73_PROTOCOL_ID
    attention_mode = "full"

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        decoder_id: str,
        device: str | torch.device | None = None,
        micro_batch_size: int = 16,
        allowed_project_labels: Sequence[str] | None = None,
    ) -> None:
        self.decoder_id: BieDecoderId = normalize_bie_decoder_id(decoder_id)
        if (
            isinstance(micro_batch_size, bool)
            or not isinstance(micro_batch_size, int)
            or micro_batch_size < 1
        ):
            raise FullBie73Error("micro_batch_size must be a positive integer")
        if not getattr(tokenizer, "is_fast", False):
            raise FullBie73Error("exact character inference requires a fast tokenizer")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise FullBie73Error("tokenizer lacks pad and EOS tokens")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        config = getattr(model, "config", None)
        raw_id2label = getattr(config, "id2label", None)
        if not isinstance(raw_id2label, Mapping):
            raise FullBie73Error("model config must provide id2label")
        if (
            getattr(config, "model_type", None) != "qwen3_bi"
            or getattr(config, "pii_attention_mode", None) != "full"
            or getattr(config, "pii_tagging_scheme", None) != "BIE"
            or getattr(config, "pii_release_eligible", None) is not False
            or getattr(config, "use_cache", None) is not False
        ):
            raise FullBie73Error("model is not a full-attention BIE73 research artifact")
        try:
            self.id2label = normalize_project_bie_id2label(raw_id2label)
            self.allowed_label_ids = project_bie_allowed_label_ids(
                self.id2label, allowed_project_labels
            )
        except (TypeError, ValueError) as exc:
            raise FullBie73Error("full BIE73 allowed-label protocol is invalid") from exc
        self.allowed_project_labels = (
            frozenset(allowed_project_labels)
            if allowed_project_labels is not None
            else frozenset(
                label.split("-", 1)[1] for label in self.id2label.values() if label != "O"
            )
        )
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.micro_batch_size = micro_batch_size
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            self.model.to(self.device)

    def _predict_micro_batch(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        if not isinstance(offsets, torch.Tensor) or offsets.ndim != 3 or offsets.shape[-1] != 2:
            raise FullBie73Error("tokenizer returned invalid batched offsets")
        attention_mask = encoded.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor):
            raise FullBie73Error("tokenizer did not return an attention mask")
        valid_tokens = (offsets[..., 1] > offsets[..., 0]) & attention_mask.to(dtype=torch.bool)
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits.float()
            try:
                label_ids, token_scores = decode_bie_logits(
                    logits,
                    allowed_label_ids=self.allowed_label_ids,
                    id2label=self.id2label,
                    decoder_id=self.decoder_id,
                    valid_token_mask=valid_tokens.to(self.device),
                )
            except BieDecodingError as exc:
                raise FullBie73Error("full BIE73 decoder failed") from exc
        results: list[list[dict[str, Any]]] = []
        for row in range(len(texts)):
            row_offsets = [tuple(map(int, pair)) for pair in offsets[row].tolist()]
            decoded = decode_bie_ids(
                label_ids[row].tolist(),
                row_offsets,
                self.id2label,
                token_scores=token_scores[row].tolist(),
            )
            results.append(
                [
                    {
                        "entity_type": span.entity_type,
                        "start": span.start,
                        "end": span.end,
                        "score": span.score,
                        "offset_mode": "relative",
                        "metadata": {
                            "boundary_refined": False,
                            "protocol_id": self.protocol_id,
                            "decoder_id": self.decoder_id,
                        },
                    }
                    for span in decoded
                ]
            )
        return results

    def predict_batch(self, texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("texts must contain strings")
        results: list[list[dict[str, Any]]] = [[] for _ in values]
        non_empty = [(index, text) for index, text in enumerate(values) if text]
        for start in range(0, len(non_empty), self.micro_batch_size):
            batch = non_empty[start : start + self.micro_batch_size]
            predictions = self._predict_micro_batch([text for _, text in batch])
            for (index, _), prediction in zip(batch, predictions, strict=True):
                results[index] = prediction
        return results

    def predict(self, text: str) -> list[dict[str, Any]]:
        return self.predict_batch([text])[0]


def load_local_full_bie73_predictor(
    model_path: str | Path,
    *,
    decoder_id: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    micro_batch_size: int = 16,
    allowed_project_labels: Sequence[str] | None = None,
) -> FullBie73SpanPredictor:
    """Load only an independently verified full-BIE73 local artifact."""

    selected_decoder = normalize_bie_decoder_id(decoder_id)
    verified_before = verify_full_bie73_artifact(model_path)
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "weights_only": True,
        "low_cpu_mem_usage": True,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        model = Qwen3BiForTokenClassification.from_pretrained(
            verified_before.root,
            **kwargs,
        )
    except Exception as exc:
        raise FullBie73Error("full BIE73 safetensors model failed local loading") from exc
    if type(model) is not Qwen3BiForTokenClassification:
        raise FullBie73Error("full BIE73 loader returned an unexpected model class")

    # Qwen3BiConfig currently initializes this field to true.  The raw config
    # was already required to contain false; restore and revalidate the research
    # contract after model construction rather than trusting that default.
    model.config.pii_release_eligible = False
    _verify_full_bie73_config(model.config.to_dict())
    verified_after = verify_full_bie73_artifact(verified_before.root)
    if verified_after != verified_before:
        raise FullBie73Error("full BIE73 artifact changed while it was being loaded")
    tokenizer = load_serialized_fast_tokenizer(verified_before.root, require_boundary=True)
    return FullBie73SpanPredictor(
        model,
        tokenizer,
        decoder_id=selected_decoder,
        device=device,
        micro_batch_size=micro_batch_size,
        allowed_project_labels=allowed_project_labels,
    )


__all__ = [
    "FULL_BIE73_ATTENTION_BACKEND",
    "FULL_BIE73_CLOSED8_LABELS",
    "FULL_BIE73_INITIALIZATION_STRATEGY",
    "FULL_BIE73_MANIFEST_TYPE",
    "FULL_BIE73_MODEL_IDENTITY_SCHEMA_VERSION",
    "FULL_BIE73_MODEL_PROTOCOL",
    "FULL_BIE73_PROTOCOL_ID",
    "FullBie73Error",
    "FullBie73ModelIdentity",
    "FullBie73SpanPredictor",
    "VerifiedFullBie73Model",
    "load_local_full_bie73_predictor",
    "verify_full_bie73_artifact",
    "verify_full_bie73_training_manifest",
]
