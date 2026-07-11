from __future__ import annotations

from pii_zh.presidio import TokenizerAwareChunker


class CharacterTokenizer:
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is False
        return 2

    def __call__(self, text: str, **_: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def test_chunker_prefers_structure_honors_token_limit_and_stride() -> None:
    chunker = TokenizerAwareChunker(CharacterTokenizer(), max_tokens=8, stride_fraction=0.20)
    windows = chunker.chunk("甲乙丙丁。戊己庚辛壬癸子丑")

    assert windows[0].end == 5  # sentence boundary before the hard six-token limit
    assert all(window.token_count + 2 <= 8 for window in windows)
    for current, following in zip(windows, windows[1:], strict=False):
        overlap = current.token_end - following.token_start
        assert overlap == round(current.token_count * 0.20)
        assert current.owner_end == following.owner_start


def test_center_ownership_selects_exactly_one_overlapping_window() -> None:
    chunker = TokenizerAwareChunker(CharacterTokenizer(), max_tokens=8)
    windows = chunker.chunk("甲乙丙丁张三戊己庚辛")
    containing = [window for window in windows if window.start <= 4 < 6 <= window.end]

    assert len(containing) == 2
    assert sum(window.owns(4, 6) for window in containing) == 1
