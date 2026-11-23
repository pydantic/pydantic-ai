"""An edit through a harness text store leaves every byte it did not edit as it was.

Line endings, a byte order mark, NUL and bytes that are not UTF-8 each broke one store's edit separately:
https://github.com/pydantic/pydantic-ai/pull/11851, https://github.com/pydantic/pydantic-ai/pull/13613 and
https://github.com/pydantic/pydantic-ai/pull/13278. Each case writes text holding one of them, edits a word in it,
and reads the result back. A new text store gets cases here.

A case's expectation is what the store's docs promise, with `source` naming where. Where they promise nothing,
`source` is `UNDOCUMENTED` and the expectation records what the store does. `AgentSpec.from_file` and
`Dataset.from_file` read a byte order mark as https://github.com/pydantic/pydantic-ai/pull/8317 made them, and its
tests cover that.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.file_system import FileSystemToolset
from pydantic_ai_harness.memory import (
    FileMemoryStore,
    InMemoryMemoryStore,
    Memory,
    MemoryStore,
    MemoryToolset,
    SqliteMemoryStore,
)

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='`LocalWorkspaceBackend` needs a POSIX host')

Feature = Literal['crlf', 'bom', 'nul', 'invalid_utf8']
REFUSED = 'refused'
"""What a case observes when the store refuses the edit."""
UNDOCUMENTED = 'undocumented: what the store does on main'


@dataclass(frozen=True)
class Expect:
    preserved: bool
    """Whether the edit leaves every other byte as it was; `False` means the store refuses the edit."""
    source: str


# --- FileSystem: `edit_file` on a file the bytes were written to ---

FILE_BYTES: dict[Feature, bytes] = {
    'crlf': b'one\r\nalpha\r\ntwo\r\n',
    'bom': b'\xef\xbb\xbfalpha\nnext\n',
    'nul': b'one\x00two alpha\n',
    'invalid_utf8': b'caf\xe9 alpha\n',
}

_FILESYSTEM_ENCODINGS = 'docs/harness/filesystem.md "Security model": Text encodings'

FILESYSTEM: dict[Feature, Expect] = {
    'crlf': Expect(True, UNDOCUMENTED),
    'bom': Expect(True, UNDOCUMENTED),
    'nul': Expect(False, UNDOCUMENTED),
    'invalid_utf8': Expect(True, _FILESYSTEM_ENCODINGS),
}


@pytest.mark.parametrize('feature', FILESYSTEM)
async def test_file_system_edit(feature: Feature, tmp_path: Path) -> None:
    """`edit_file` replaces `alpha` and nothing else, or refuses the file."""
    original = FILE_BYTES[feature]
    (tmp_path / 'notes.txt').write_bytes(original)
    toolset = FileSystemToolset[None](
        allowed_patterns=[],
        denied_patterns=[],
        max_read_lines=2000,
        max_list_results=1000,
        max_search_results=1000,
        max_find_results=1000,
    )
    try:
        await toolset.edit_file('notes.txt', 'alpha', 'omega', workspace=LocalWorkspaceBackend(tmp_path))
    except ModelRetry:
        observed: bytes | str = REFUSED
    else:
        observed = (tmp_path / 'notes.txt').read_bytes()

    expected = FILESYSTEM[feature]
    assert observed == (original.replace(b'alpha', b'omega') if expected.preserved else REFUSED), expected.source


# --- Memory: `write_memory` edits on each bundled store ---

MEMORY_TEXT: dict[Feature, tuple[str, str]] = {
    # An `old_text` with `\n` line endings, as the injected snapshot shows the file.
    'crlf': ('- one\r\n- alpha\r\n- two', '- one\n- alpha'),
    'bom': ('﻿- alpha\n- two', '- alpha'),
    'nul': ('- one\x00two\n- alpha', '- alpha'),
}
"""Each feature's seeded content and the `old_text` that selects `alpha` in it."""

_MEMORY_CRLF = 'docs/harness/memory.md `write_memory`: an `old_text` with `\\n` line endings matches a `\\r\\n` file'

MEMORY: dict[tuple[str, Feature], Expect] = {
    **{(store, 'crlf'): Expect(True, _MEMORY_CRLF) for store in ('file', 'in_memory', 'sqlite')},
    **{
        (store, feature): Expect(True, UNDOCUMENTED)
        for store in ('file', 'in_memory', 'sqlite')
        for feature in ('bom', 'nul')
    },
}


def _stores(tmp_path: Path) -> dict[str, Callable[[], MemoryStore]]:
    return {
        'file': lambda: FileMemoryStore('memory', workspace=LocalWorkspaceBackend(tmp_path)),
        'in_memory': InMemoryMemoryStore,
        'sqlite': lambda: SqliteMemoryStore(database=tmp_path / 'memory.sqlite3'),
    }


def _ctx(tool_call_id: str) -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), tool_call_id=tool_call_id, run_id='run')


async def _edit(store: MemoryStore, content: str, old_text: str) -> tuple[str, str]:
    """The file's content after seeding it, and after editing `alpha` in it."""
    capability = Memory[None](store=store)
    toolset = MemoryToolset(capability)
    await toolset.write_memory(_ctx('seed'), content)
    seeded = await capability.store.read('main/MEMORY.md', max_chars=10_000)
    assert seeded is not None
    await toolset.write_memory(_ctx('edit'), old_text.replace('alpha', 'omega'), old_text=old_text)
    edited = await capability.store.read('main/MEMORY.md', max_chars=10_000)
    assert edited is not None
    return seeded.content, edited.content


@pytest.mark.parametrize(('store', 'feature'), MEMORY)
async def test_memory_edit(store: str, feature: Feature, tmp_path: Path) -> None:
    """Seeding keeps the feature, and editing `alpha` changes nothing else."""
    content, old_text = MEMORY_TEXT[feature]

    seeded, edited = await _edit(_stores(tmp_path)[store](), content, old_text)

    expected = MEMORY[(store, feature)]
    assert seeded.startswith(content), expected.source
    assert (edited == seeded.replace('alpha', 'omega')) == expected.preserved, expected.source
