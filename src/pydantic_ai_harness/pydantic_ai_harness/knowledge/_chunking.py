"""Split document text into the chunks that `Knowledge` embeds and returns."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.exceptions import UserError


class Chunker(Protocol):
    """Splits one document's text into the passages that are embedded and returned by search.

    Chunking runs on every sync, so keep it deterministic: the same text must give the same
    chunks, or unchanged documents are embedded again.
    """

    def chunk(self, text: str) -> Sequence[str]:
        """Return `text` split into chunks, in document order."""
        ...


@dataclass(frozen=True, kw_only=True)
class RecursiveChunker:
    """Split text at the coarsest separator that keeps chunks within `chunk_size` characters.

    Text is split at each separator in turn (paragraphs, then lines, then sentences, then
    words), and adjacent pieces are merged back into chunks of up to `chunk_size` characters,
    each starting with up to `chunk_overlap` characters from the end of the previous one. A
    piece with no separator left that is still too long is cut every `chunk_size` characters.
    """

    chunk_size: int = 1_000
    """Maximum chunk length in characters. Must be positive."""

    chunk_overlap: int = 200
    """Characters repeated from the end of one chunk at the start of the next, so a passage cut at a
    boundary is still found whole. Must be non-negative and smaller than `chunk_size`."""

    separators: tuple[str, ...] = ('\n\n', '\n', '. ', ' ')
    """Separators to split at, coarsest first. Each stays at the end of the piece before it."""

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise UserError('`RecursiveChunker.chunk_size` must be positive.')
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise UserError('`RecursiveChunker.chunk_overlap` must be non-negative and smaller than `chunk_size`.')
        if '' in self.separators:
            raise UserError('`RecursiveChunker.separators` cannot contain an empty string.')

    def chunk(self, text: str) -> list[str]:
        """Return `text` split into chunks of at most `chunk_size` characters, without surrounding whitespace."""
        return [stripped for piece in self._split(text, self.separators) if (stripped := piece.strip())]

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        index = next((i for i, separator in enumerate(separators) if separator in text), None)
        if index is None:
            step = self.chunk_size - self.chunk_overlap
            return [text[start : start + self.chunk_size] for start in range(0, len(text) - self.chunk_overlap, step)]
        separator, finer = separators[index], separators[index + 1 :]
        parts = text.split(separator)
        pieces = [part + separator for part in parts[:-1]] + [parts[-1]]

        chunks: list[str] = []
        window: list[str] = []
        for piece in pieces:
            if len(piece) > self.chunk_size:
                if window:
                    chunks.append(''.join(window))
                window = []
                chunks.extend(self._split(piece, finer))
                continue
            if window and sum(map(len, window)) + len(piece) > self.chunk_size:
                chunks.append(''.join(window))
                window = self._overlap(window, len(piece))
            window.append(piece)
        if window:
            chunks.append(''.join(window))
        return chunks

    def _overlap(self, window: list[str], incoming: int) -> list[str]:
        """The trailing pieces of `window` to repeat: at most `chunk_overlap` characters, leaving room for `incoming`."""
        kept: list[str] = []
        size = 0
        # Always ends at a `break`: the window was flushed because it and `incoming` do not fit in `chunk_size`.
        for piece in reversed(window):  # pragma: no branch
            size += len(piece)
            if size > self.chunk_overlap or size + incoming > self.chunk_size:
                break
            kept.insert(0, piece)
        return kept
