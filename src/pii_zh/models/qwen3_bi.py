"""Bidirectional Qwen3 token classification.

Transformers 5.13's :class:`~transformers.Qwen3Model` accepts a mapping of
precomputed attention masks.  Passing such a mapping deliberately bypasses
the model's internal causal-mask construction.  This module uses that public
forward behaviour to give every non-padding query access to every
non-padding key while retaining the upstream Qwen3 blocks and state-dict key
layout.

Only eager attention and PyTorch SDPA are supported.  FlashAttention/flex
attention are intentionally rejected because a non-causal, padding-aware 4-D
additive mask is part of this model's correctness contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on optional environment
    raise ImportError("qwen3_bi requires the optional PyTorch dependency.") from exc

try:
    from transformers import Qwen3Config, Qwen3Model
    from transformers.modeling_outputs import TokenClassifierOutput
    from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel
except ImportError as exc:  # pragma: no cover - depends on optional environment
    raise ImportError(
        "qwen3_bi requires a Transformers release with Qwen3 support "
        "(tested with transformers>=5.13,<6)."
    ) from exc


SUPPORTED_ATTENTION_BACKENDS = frozenset({"eager", "sdpa"})
ARCHITECTURE_VERSION = "qwen3_bi_token_cls_v1"


class Qwen3BiConfig(Qwen3Config):
    """Configuration for a full-attention Qwen3 token classifier.

    ``bi_attention_backend`` is persisted because Transformers deliberately
    omits its private ``_attn_implementation`` setting from ``config.json``.
    Persisting the choice makes a save/load round trip deterministic while an
    explicit ``attn_implementation=...`` argument can still override it.
    """

    model_type = "qwen3_bi"

    def __init__(
        self,
        *,
        bi_attention_backend: str = "sdpa",
        architecture_version: str = ARCHITECTURE_VERSION,
        **kwargs: Any,
    ) -> None:
        requested_backend = kwargs.get("attn_implementation") or bi_attention_backend
        if requested_backend not in SUPPORTED_ATTENTION_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_ATTENTION_BACKENDS))
            raise ValueError(
                f"Unsupported bidirectional attention backend {requested_backend!r}; "
                f"choose one of: {supported}."
            )

        # KV caching is invalid for a representation whose earlier token
        # states depend on later tokens.  Ignore stale upstream config values.
        kwargs["use_cache"] = False
        kwargs["attn_implementation"] = requested_backend
        # Transformers does not publish complete annotations for Qwen config
        # construction.  Keep the suppression at that external call boundary.
        super().__init__(**kwargs)  # type: ignore[no-untyped-call]

        self.bi_attention_backend = requested_backend
        self.architecture_version = architecture_version
        self.use_cache = False
        self.pii_attention_mode = "full"
        self.pii_release_eligible = True
        self.auto_map = {
            "AutoConfig": "configuration_qwen3_bi.Qwen3BiConfig",
            "AutoModelForTokenClassification": ("modeling_qwen3_bi.Qwen3BiForTokenClassification"),
        }
        if self.architectures is None:
            self.architectures = ["Qwen3BiForTokenClassification"]

        layer_types = set(getattr(self, "layer_types", []) or [])
        if layer_types and layer_types != {"full_attention"}:
            raise ValueError(
                "Qwen3BiConfig requires every decoder layer to use "
                f"'full_attention'; received {sorted(layer_types)!r}."
            )

    @classmethod
    def from_qwen3_config(
        cls,
        config: Qwen3Config,
        *,
        bi_attention_backend: str | None = None,
    ) -> Qwen3BiConfig:
        """Copy a standard Qwen3 config into the bidirectional architecture."""

        if not isinstance(config, Qwen3Config):
            raise TypeError(f"Expected Qwen3Config, received {type(config).__name__}.")
        values = config.to_dict()
        values.pop("model_type", None)
        values.pop("architectures", None)
        values["use_cache"] = False
        values["bi_attention_backend"] = bi_attention_backend or getattr(
            config, "bi_attention_backend", getattr(config, "_attn_implementation", None) or "sdpa"
        )
        return cls(**values)


def build_full_attention_mask(
    attention_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a 2-D valid-token mask to a 4-D additive key-padding mask.

    The result has shape ``[batch, 1, query_length, key_length]``.  Every query
    row has the same visible keys; padding *queries* are not fully masked and
    are instead discarded by the task loss/decoder.  Using the finite minimum
    for ``dtype`` avoids all-``-inf`` rows and the resulting softmax NaNs.
    """

    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor.")
    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape [batch, sequence], "
            f"received {tuple(attention_mask.shape)!r}."
        )
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"The additive attention mask requires a floating dtype, got {dtype}.")

    valid_keys = attention_mask.to(dtype=torch.bool)
    if valid_keys.shape[1] == 0:
        raise ValueError("Cannot build attention for an empty sequence.")
    if not bool(valid_keys.any(dim=-1).all()):
        raise ValueError("Each example must contain at least one non-padding token.")

    batch_size, sequence_length = valid_keys.shape
    additive_mask = torch.zeros(
        (batch_size, 1, sequence_length, sequence_length),
        dtype=dtype,
        device=attention_mask.device,
    )
    padding_keys = ~valid_keys[:, None, None, :]
    additive_mask.masked_fill_(padding_keys, torch.finfo(dtype).min)
    return additive_mask


def convert_qwen3_token_classifier_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Validate and copy an upstream Qwen3 token-classifier state dict.

    Upstream ``Qwen3ForTokenClassification`` and this class intentionally use
    the same ``model.*`` and ``score.*`` keys, so conversion is a validated
    no-op.  The explicit function makes the trust boundary visible and avoids
    silently accepting a causal-LM checkpoint with no classification head.
    """

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping of parameter names to tensors.")
    keys = tuple(state_dict)
    if not any(key.startswith("model.") for key in keys):
        raise ValueError("State dict does not contain the expected Qwen3 'model.*' weights.")
    if not any(key.startswith("score.") for key in keys):
        raise ValueError("State dict does not contain a Qwen3 token-classification 'score.*' head.")
    return dict(state_dict)


class Qwen3BiForTokenClassification(Qwen3PreTrainedModel):  # type: ignore[no-untyped-call]
    """Qwen3 token classifier with non-causal, key-padding-only attention."""

    # The upstream base leaves ``config_class`` inferred as ``None`` even
    # though subclasses are expected to replace it with their config type.
    config_class = Qwen3BiConfig  # type: ignore[assignment]
    base_model_prefix = "model"
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_sdpa = True

    def __init__(self, config: Qwen3BiConfig) -> None:
        # Transformers 5 may import configuration and modeling remote-code
        # modules under distinct cache namespaces.  Class identity is then
        # different even though both classes inherit the same Qwen3Config.
        # Validate the complete persisted security/correctness contract
        # instead of accepting a class-name-only duck type.
        compatible_remote_config = (
            isinstance(config, Qwen3Config)
            and getattr(config, "model_type", None) == Qwen3BiConfig.model_type
            and getattr(config, "architecture_version", None) == ARCHITECTURE_VERSION
            and getattr(config, "pii_attention_mode", None) == "full"
            and getattr(config, "pii_release_eligible", None) is True
            and getattr(config, "use_cache", None) is False
            and getattr(config, "bi_attention_backend", None) in SUPPORTED_ATTENTION_BACKENDS
            and set(getattr(config, "layer_types", []) or []) == {"full_attention"}
        )
        if not isinstance(config, Qwen3BiConfig) and not compatible_remote_config:
            raise TypeError(
                "Qwen3BiForTokenClassification requires the complete Qwen3BiConfig contract; "
                "use Qwen3BiConfig.from_qwen3_config(...) for an upstream config."
            )
        super().__init__(config)
        backend = getattr(config, "_attn_implementation", None)
        if backend not in SUPPORTED_ATTENTION_BACKENDS:
            raise ValueError(
                f"Qwen3 bidirectional attention supports only eager/SDPA, got {backend!r}."
            )

        self.num_labels = config.num_labels
        self.model = Qwen3Model(config)
        # Keep this import inside the extracted class so the generated
        # Hugging Face remote-code module remains self-contained.
        from numbers import Real

        classifier_dropout = getattr(config, "classifier_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = getattr(config, "hidden_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = 0.1
        if isinstance(classifier_dropout, bool) or not isinstance(classifier_dropout, Real):
            raise TypeError("classifier dropout must be a real number")
        self.dropout = nn.Dropout(float(classifier_dropout))
        self.score = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()  # type: ignore[no-untyped-call]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> TokenClassifierOutput | tuple[torch.Tensor, ...]:
        """Run full-attention token classification.

        ``attention_mask`` is always a 2-D valid-token mask.  Callers cannot
        inject a prepared mask mapping; this keeps the key-padding isolation
        invariant under the model's control.
        """

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if past_key_values is not None or use_cache:
            raise ValueError("KV caching is not supported by bidirectional token classification.")
        if isinstance(attention_mask, dict):
            raise TypeError("Pass a 2-D valid-token attention_mask, not a prepared mask mapping.")

        source = input_ids if input_ids is not None else inputs_embeds
        assert source is not None  # narrowed by the exclusive-input check above
        batch_size, sequence_length = source.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length), dtype=torch.bool, device=source.device
            )
        elif tuple(attention_mask.shape) != (batch_size, sequence_length):
            raise ValueError(
                "attention_mask shape must match the input's first two dimensions: "
                f"expected {(batch_size, sequence_length)!r}, got {tuple(attention_mask.shape)!r}."
            )
        elif attention_mask.device != source.device:
            attention_mask = attention_mask.to(source.device)

        mask_dtype = (
            inputs_embeds.dtype
            if inputs_embeds is not None
            else self.get_input_embeddings().weight.dtype
        )
        if not isinstance(mask_dtype, torch.dtype):
            raise TypeError("model embedding dtype must be a torch.dtype")
        full_attention_mask = build_full_attention_mask(attention_mask, dtype=mask_dtype)
        prepared_mask = {"full_attention": full_attention_mask}

        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=prepared_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )
        logits = self.score(self.dropout(outputs.last_hidden_state))

        loss = None
        if labels is not None:
            if tuple(labels.shape) != (batch_size, sequence_length):
                raise ValueError(
                    f"labels must have shape {(batch_size, sequence_length)!r}, "
                    f"got {tuple(labels.shape)!r}."
                )
            loss = self.loss_function(logits, labels, self.config)

        result = TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
        return result if return_dict else result.to_tuple()

    @classmethod
    def from_qwen3_token_classifier_state_dict(
        cls,
        config: Qwen3Config,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
        bi_attention_backend: str | None = None,
    ) -> Qwen3BiForTokenClassification:
        """Construct from an upstream ``Qwen3ForTokenClassification`` state dict."""

        bi_config = Qwen3BiConfig.from_qwen3_config(
            config, bi_attention_backend=bi_attention_backend
        )
        model = cls(bi_config)
        model.load_state_dict(convert_qwen3_token_classifier_state_dict(state_dict), strict=strict)
        return model

    @classmethod
    def from_qwen3_token_classifier_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        config: Qwen3Config | None = None,
        bi_attention_backend: str | None = None,
        **kwargs: Any,
    ) -> Qwen3BiForTokenClassification:
        """Load an upstream token-classifier checkpoint with bidirectional config.

        This method is suitable for local paths or explicitly approved Hub IDs;
        it performs no network policy decisions on behalf of its caller.
        """

        if config is None:
            config_keys = {
                "cache_dir",
                "force_download",
                "local_files_only",
                "revision",
                "subfolder",
                "token",
            }
            config_kwargs = {key: value for key, value in kwargs.items() if key in config_keys}
            config = Qwen3Config.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
        bi_config = Qwen3BiConfig.from_qwen3_config(
            config, bi_attention_backend=bi_attention_backend
        )
        # Transformers leaves this generic factory return type as ``Any``.
        return cls.from_pretrained(  # type: ignore[no-any-return]
            pretrained_model_name_or_path, config=bi_config, **kwargs
        )
