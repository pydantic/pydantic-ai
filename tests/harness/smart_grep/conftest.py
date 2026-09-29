from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

import pytest

from pydantic_ai_harness.smart_grep import _index
from pydantic_ai_harness.smart_grep._chunks import Chunk, source_chunks

# Syntax-aware chunking beyond Python needs the `smart-grep` extra; without it those files use line windows.
collect_ignore = ['test_treesitter.py'] if importlib.util.find_spec('tree_sitter') is None else []


@dataclass
class CountingChunker:
    """Wraps `source_chunks`, recording each path it chunks."""

    paths: list[str] = field(default_factory=list[str])

    def __call__(self, text: str, path: str) -> tuple[list[Chunk], str]:
        self.paths.append(path)
        return source_chunks(text, path)


@pytest.fixture
def chunker(monkeypatch: pytest.MonkeyPatch) -> CountingChunker:
    """Count the files the index chunks, to tell a rebuilt index from a reused one."""
    counting = CountingChunker()
    monkeypatch.setattr(_index, 'source_chunks', counting)
    return counting
