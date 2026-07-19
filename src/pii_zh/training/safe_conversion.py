"""Restricted one-time conversion of legacy PyTorch state dicts to safetensors.

This module exists only for pinned upstream checkpoints which have not
published safetensors.  It never invokes an unrestricted pickle loader and it
rejects every payload which is not a flat ``str -> Tensor`` state dictionary.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class RestrictedConversionError(RuntimeError):
    """Raised without including serialized payload values or local paths."""


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_state_dict(value: object) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional training boundary
        raise RestrictedConversionError("PyTorch is required for restricted conversion") from exc

    if not isinstance(value, Mapping) or not value:
        raise RestrictedConversionError("restricted payload is not a non-empty state dictionary")
    tensors: dict[str, Any] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise RestrictedConversionError("restricted payload is not a flat str-to-Tensor map")
        if tensor.layout is not torch.strided or tensor.is_sparse or tensor.is_quantized:
            raise RestrictedConversionError("state dictionary contains an unsupported tensor")
        if tensor.device.type == "meta":
            raise RestrictedConversionError("state dictionary contains a meta tensor")
        tensors[key] = tensor.detach().to(device="cpu").contiguous().clone()
    return tensors


def convert_restricted_state_dict(
    source: str | Path,
    destination: str | Path,
    *,
    delete_source: bool = False,
) -> dict[str, Any]:
    """Convert a legacy checkpoint using only ``torch.load(weights_only=True)``.

    The output is atomically installed, loaded back through safetensors, and
    compared tensor-for-tensor before optional source deletion.
    """

    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - optional training boundary
        raise RestrictedConversionError("PyTorch and safetensors are required") from exc

    source_path = Path(source).expanduser().resolve(strict=True)
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_file() or source_path.is_symlink():
        raise RestrictedConversionError("source must be a regular non-symlink file")
    if source_path.suffix.lower() != ".bin" or destination_path.suffix != ".safetensors":
        raise RestrictedConversionError("conversion requires .bin input and .safetensors output")
    if destination_path.exists() or destination_path.is_symlink():
        raise RestrictedConversionError("destination must not already exist")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    source_sha256 = sha256_file(source_path)
    try:
        loaded = torch.load(source_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RestrictedConversionError("restricted weights-only load failed") from exc
    tensors = _validate_state_dict(loaded)
    shape_schema = [
        {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        for name, tensor in sorted(tensors.items())
    ]

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination_path.parent,
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        save_file(tensors, temporary_path, metadata={"format": "pt"})
        restored = load_file(temporary_path, device="cpu")
        if set(restored) != set(tensors):
            raise RestrictedConversionError("safetensors key verification failed")
        for name, tensor in tensors.items():
            candidate = restored[name]
            if (
                tensor.dtype != candidate.dtype
                or tensor.shape != candidate.shape
                or not torch.equal(tensor, candidate)
            ):
                raise RestrictedConversionError("safetensors tensor verification failed")
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, destination_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    destination_sha256 = sha256_file(destination_path)
    if delete_source:
        source_path.unlink()
        if source_path.exists():  # pragma: no cover - operating-system boundary
            destination_path.unlink(missing_ok=True)
            raise RestrictedConversionError("legacy source deletion could not be verified")

    dtype_counts = Counter(str(tensor.dtype) for tensor in tensors.values())
    return {
        "schema_version": 1,
        "loader": "torch.load(weights_only=True,map_location=cpu)",
        "payload_contract": "flat_str_to_tensor_state_dict",
        "source_sha256": source_sha256,
        "source_deleted": delete_source,
        "destination_sha256": destination_sha256,
        "tensor_count": len(tensors),
        "total_elements": sum(tensor.numel() for tensor in tensors.values()),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "shape_schema_sha256": canonical_json_hash(shape_schema),
        "verification": "safetensors_full_tensor_equality",
    }


__all__ = [
    "RestrictedConversionError",
    "canonical_json_hash",
    "convert_restricted_state_dict",
    "sha256_file",
]
