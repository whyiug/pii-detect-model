"""Reusable, explicitly selected greedy and constrained BIE decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional inference dependency
    raise ImportError("BIE decoding requires PyTorch.") from exc


GREEDY_BIE_DECODER_ID = "greedy"
CONSTRAINED_VITERBI_BIE_DECODER_ID = "constrained_viterbi"
BieDecoderId = Literal["greedy", "constrained_viterbi"]
BIE_DECODER_IDS: tuple[BieDecoderId, ...] = (
    GREEDY_BIE_DECODER_ID,
    CONSTRAINED_VITERBI_BIE_DECODER_ID,
)


class BieDecodingError(ValueError):
    """Raised when a decoder contract or BIE path is invalid."""


def normalize_bie_decoder_id(value: object) -> BieDecoderId:
    """Validate a decoder ID without a fallback or implicit default."""

    if not isinstance(value, str) or value not in BIE_DECODER_IDS:
        raise BieDecodingError("decoder_id must be greedy or constrained_viterbi")
    return value


def _parse_tag(value: object) -> tuple[str, str | None]:
    if value == "O":
        return "O", None
    if not isinstance(value, str):
        raise BieDecodingError("BIE labels must be strings")
    prefix, separator, entity = value.partition("-")
    if separator != "-" or prefix not in {"B", "I", "E"} or not entity:
        raise BieDecodingError("label inventory is not BIE-compatible")
    return prefix, entity


def _decoder_contract(
    allowed_label_ids: Sequence[int],
    id2label: Mapping[int, str],
) -> tuple[tuple[int, ...], int, tuple[tuple[int, int, int], ...]]:
    if isinstance(allowed_label_ids, (str, bytes)):
        raise TypeError("allowed_label_ids must be an integer sequence")
    allowed = tuple(allowed_label_ids)
    if (
        not allowed
        or len(set(allowed)) != len(allowed)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in allowed
        )
    ):
        raise BieDecodingError("allowed_label_ids must contain unique non-negative integers")
    try:
        tags = tuple(id2label[label_id] for label_id in allowed)
    except (KeyError, TypeError):
        raise BieDecodingError("allowed_label_ids are absent from id2label") from None
    if len(set(tags)) != len(tags):
        raise BieDecodingError("allowed BIE labels must be unique")
    local = {tag: index for index, tag in enumerate(tags)}
    if "O" not in local:
        raise BieDecodingError("allowed BIE labels must include O")
    entities = sorted(
        entity for prefix, entity in map(_parse_tag, tags) if prefix != "O" and entity is not None
    )
    unique_entities = tuple(dict.fromkeys(entities))
    triples: list[tuple[int, int, int]] = []
    for entity in unique_entities:
        required = (f"B-{entity}", f"I-{entity}", f"E-{entity}")
        if any(tag not in local for tag in required):
            raise BieDecodingError("each allowed entity must include complete B/I/E states")
        triples.append(tuple(local[tag] for tag in required))  # type: ignore[arg-type]
    if not triples:
        raise BieDecodingError("allowed BIE labels must include at least one entity")
    return allowed, local["O"], tuple(triples)


def validate_bie_sequence(
    label_ids: Sequence[int],
    *,
    id2label: Mapping[int, str],
) -> None:
    """Reject orphan/mismatched I/E states and unterminated multi-token spans.

    A standalone ``B`` is a valid singleton span.  Once a matching ``I`` is
    observed, the span must end with a matching ``E``.
    """

    active: str | None = None
    continuation_seen = False
    for label_id in label_ids:
        try:
            prefix, entity = _parse_tag(id2label[label_id])
        except (KeyError, TypeError):
            raise BieDecodingError("decoded label is absent from id2label") from None
        if prefix in {"O", "B"}:
            if active is not None and continuation_seen:
                raise BieDecodingError("multi-token BIE span ended without E")
            active = entity if prefix == "B" else None
            continuation_seen = False
        elif prefix == "I":
            if active is None or active != entity:
                raise BieDecodingError("decoded sequence contains orphan or mismatched I")
            continuation_seen = True
        else:
            if active is None or active != entity:
                raise BieDecodingError("decoded sequence contains orphan or mismatched E")
            active = None
            continuation_seen = False
    if active is not None and continuation_seen:
        raise BieDecodingError("multi-token BIE span ended without E")


def _valid_token_mask(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    time_steps: int,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.ones((batch_size, time_steps), dtype=torch.bool, device=device)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch_size, time_steps):
        raise BieDecodingError("valid_token_mask must match the logits batch/time dimensions")
    return value.to(device=device, dtype=torch.bool)


def _constrained_local_path(
    logits: torch.Tensor,
    *,
    o_state: int,
    triples: tuple[tuple[int, int, int], ...],
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] < 1:
        raise BieDecodingError("constrained Viterbi expects non-empty [time, labels] logits")
    if tuple(valid_tokens.shape) != (logits.shape[0],):
        raise BieDecodingError("constrained Viterbi valid-token mask has an invalid shape")
    emissions = logits.to(dtype=torch.float64)
    if not bool(torch.isfinite(emissions).all()):
        raise BieDecodingError("BIE logits must be finite")
    emissions = emissions.clone()
    invalid = ~valid_tokens
    if bool(invalid.any()):
        emissions[invalid] = -torch.inf
        emissions[invalid, o_state] = 0.0

    b_states = torch.tensor([item[0] for item in triples], device=logits.device)
    e_states = torch.tensor([item[2] for item in triples], device=logits.device)
    terminal_states = torch.cat(
        (torch.tensor([o_state], device=logits.device), b_states, e_states)
    )
    time_steps, state_count = emissions.shape
    previous = torch.full(
        (state_count,), -torch.inf, dtype=torch.float64, device=logits.device
    )
    previous[o_state] = emissions[0, o_state]
    previous[b_states] = emissions[0, b_states]
    backpointers = torch.full(
        (time_steps, state_count), -1, dtype=torch.long, device=logits.device
    )

    for step in range(1, time_steps):
        current = torch.full_like(previous, -torch.inf)
        terminal_source = terminal_states[torch.argmax(previous[terminal_states])]
        current[o_state] = previous[terminal_source] + emissions[step, o_state]
        backpointers[step, o_state] = terminal_source
        current[b_states] = previous[terminal_source] + emissions[step, b_states]
        backpointers[step, b_states] = terminal_source
        for b_state, i_state, e_state in triples:
            sources = torch.tensor((b_state, i_state), dtype=torch.long, device=logits.device)
            source = sources[torch.argmax(previous[sources])]
            current[i_state] = previous[source] + emissions[step, i_state]
            current[e_state] = previous[source] + emissions[step, e_state]
            backpointers[step, i_state] = source
            backpointers[step, e_state] = source
        previous = current

    final_state = terminal_states[torch.argmax(previous[terminal_states])]
    decoded = torch.empty((time_steps,), dtype=torch.long, device=logits.device)
    decoded[-1] = final_state
    for step in range(time_steps - 1, 0, -1):
        final_state = backpointers[step, final_state]
        if int(final_state) < 0:
            raise BieDecodingError("constrained Viterbi produced an invalid backpointer")
        decoded[step - 1] = final_state
    return decoded


def constrained_bie_viterbi(
    logits: torch.Tensor,
    *,
    allowed_label_ids: Sequence[int],
    id2label: Mapping[int, str],
    valid_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a globally valid BIE path from already scope-restricted logits."""

    allowed, o_state, triples = _decoder_contract(allowed_label_ids, id2label)
    if (
        not isinstance(logits, torch.Tensor)
        or not torch.is_floating_point(logits)
        or logits.ndim != 2
        or logits.shape[-1] != len(allowed)
    ):
        raise BieDecodingError("constrained Viterbi logits do not match allowed labels")
    valid = _valid_token_mask(
        None if valid_token_mask is None else valid_token_mask.unsqueeze(0),
        batch_size=1,
        time_steps=logits.shape[0],
        device=logits.device,
    )[0]
    local_path = _constrained_local_path(
        logits,
        o_state=o_state,
        triples=triples,
        valid_tokens=valid,
    )
    allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=logits.device)
    full_path = allowed_tensor[local_path]
    validate_bie_sequence(full_path.tolist(), id2label=id2label)
    return full_path


def decode_bie_logits(
    logits: torch.Tensor,
    *,
    allowed_label_ids: Sequence[int],
    id2label: Mapping[int, str],
    decoder_id: str,
    valid_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask logits before an explicitly selected decoder and return IDs/scores."""

    selected_decoder = normalize_bie_decoder_id(decoder_id)
    allowed, o_state, triples = _decoder_contract(allowed_label_ids, id2label)
    if (
        not isinstance(logits, torch.Tensor)
        or not torch.is_floating_point(logits)
        or logits.ndim not in {2, 3}
    ):
        raise BieDecodingError(
            "BIE decode expects floating [time, labels] or [batch, time, labels]"
        )
    single = logits.ndim == 2
    batched = logits.unsqueeze(0) if single else logits
    batch_size, time_steps, label_count = batched.shape
    if max(allowed) >= label_count or not bool(torch.isfinite(batched).all()):
        raise BieDecodingError("BIE logits are non-finite or narrower than allowed labels")
    valid = _valid_token_mask(
        (
            valid_token_mask.unsqueeze(0)
            if single and valid_token_mask is not None
            else valid_token_mask
        ),
        batch_size=batch_size,
        time_steps=time_steps,
        device=batched.device,
    )
    allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=batched.device)
    restricted = batched.index_select(-1, allowed_tensor)
    probabilities = torch.softmax(restricted, dim=-1)

    if selected_decoder == GREEDY_BIE_DECODER_ID:
        local_paths = restricted.argmax(dim=-1)
        local_paths = torch.where(valid, local_paths, torch.full_like(local_paths, o_state))
    else:
        local_paths = torch.stack(
            [
                _constrained_local_path(
                    restricted[row],
                    o_state=o_state,
                    triples=triples,
                    valid_tokens=valid[row],
                )
                for row in range(batch_size)
            ]
        )
    label_ids = allowed_tensor[local_paths]
    token_scores = probabilities.gather(-1, local_paths.unsqueeze(-1)).squeeze(-1)
    for row in range(batch_size):
        if selected_decoder == CONSTRAINED_VITERBI_BIE_DECODER_ID:
            validate_bie_sequence(label_ids[row].tolist(), id2label=id2label)
    if single:
        return label_ids[0], token_scores[0]
    return label_ids, token_scores


__all__ = [
    "BIE_DECODER_IDS",
    "CONSTRAINED_VITERBI_BIE_DECODER_ID",
    "GREEDY_BIE_DECODER_ID",
    "BieDecoderId",
    "BieDecodingError",
    "constrained_bie_viterbi",
    "decode_bie_logits",
    "normalize_bie_decoder_id",
    "validate_bie_sequence",
]
