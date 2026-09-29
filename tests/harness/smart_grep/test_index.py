"""Tests for `SmartFileSearch`'s index: discovery through the workspace, BM25, caching, and parallel searches."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace
from pydantic_ai_harness.smart_grep import _chunks, _index
from pydantic_ai_harness.smart_grep._chunks import Chunk, searchable_root
from pydantic_ai_harness.smart_grep._index import Shortlist, SnippetIndexes, read_snippets
from pydantic_ai_harness.smart_grep._retrieve import Bm25

from ..filesystem.conftest import tools_path
from .conftest import CountingChunker
from .test_smart_grep import PY_SOURCE, VanishingBackend


def _workspace(root: Path) -> Workspace:
    return Workspace(LocalWorkspaceBackend(root))


@dataclass
class Found:
    shortlist: Shortlist
    chunks: list[Chunk]

    @property
    def files(self) -> int:
        return self.shortlist.files

    @property
    def skipped(self) -> list[tuple[str, str]]:
        return self.shortlist.skipped


async def _discover(
    workspace: Workspace,
    directory: str,
    glob: str | None = None,
    *,
    indexes: SnippetIndexes | None = None,
    query: str = 'x',
) -> Found:
    """Every snippet under `directory`, through the same steps `search_code` takes before judging."""
    root = await searchable_root(workspace, directory)
    found = await (indexes or SnippetIndexes(0)).shortlist(workspace, root, glob, query, 10_000)
    return Found(found, await read_snippets(workspace, root, directory, found.hits))


# ---------------------------------------------------------------- discovery


async def test_discover_skips_binary_non_utf8_and_minified(tmp_path: Path) -> None:
    (tmp_path / 'good.py').write_text(PY_SOURCE)
    (tmp_path / 'bin.dat').write_bytes(b'abc\0def')
    (tmp_path / 'latin.txt').write_bytes('caf\xe9'.encode('latin-1'))
    (tmp_path / 'min.js').write_text('x' * 13_000)
    found = await _discover(_workspace(tmp_path), '.')
    assert found.files == 4
    assert sorted(reason.split(':')[0] for _, reason in found.skipped) == [
        'binary',
        'line exceeds 12000 characters',
        'not UTF-8',
    ]
    assert {c.path for c in found.chunks} == {'good.py'}


async def test_discover_respects_gitignore_and_reports_paths_as_spelled(tmp_path: Path) -> None:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'a.py').write_text('x = 1\n')
    (tmp_path / 'src' / 'ignored.py').write_text('y = 2\n')
    (tmp_path / '.gitignore').write_text('src/ignored.py\n')
    (tmp_path / '.git').mkdir()
    found = await _discover(_workspace(tmp_path), 'src')
    assert [c.path for c in found.chunks] == ['src/a.py']
    found = await _discover(_workspace(tmp_path), str(tmp_path / 'src'))
    assert [c.path for c in found.chunks] == [str(tmp_path / 'src' / 'a.py')]
    assert (await _discover(_workspace(tmp_path), 'src', glob='*.txt')).chunks == []


async def test_glob_cannot_reach_ignored_or_hidden_files(tmp_path: Path) -> None:
    (tmp_path / '.git').mkdir()
    (tmp_path / '.gitignore').write_text('ignored.py\n.env\n')
    (tmp_path / '.env').write_text('SECRET=1\n')
    (tmp_path / 'ignored.py').write_text('x = 1\n')
    (tmp_path / 'kept.py').write_text('y = 2\n')
    workspace = _workspace(tmp_path)
    assert (await _discover(workspace, '.', glob='.env')).chunks == []  # ripgrep's `--glob` alone would list it
    assert [c.path for c in (await _discover(workspace, '.', glob='*.py')).chunks] == ['kept.py']


async def test_discover_stays_inside_the_working_directory(tmp_path: Path) -> None:
    project, outside = tmp_path / 'project', tmp_path / 'outside'
    project.mkdir()
    outside.mkdir()
    (outside / 'secret.py').write_text('token = 1\n')
    (project / 'link').symlink_to(outside)
    workspace = _workspace(project)
    for directory in ('..', str(outside), 'link'):
        with pytest.raises(ModelRetry, match='outside the working directory'):
            await _discover(workspace, directory)


async def test_discover_enforces_line_and_file_budgets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / 'a.py').write_text('x = 1\n' * 50)
    (tmp_path / 'b.py').write_text('y = 1\n')
    monkeypatch.setattr(_index, 'MAX_TOTAL_LINES', 10)
    with pytest.raises(ModelRetry, match='exceeds 10 lines of source'):
        await _discover(_workspace(tmp_path), '.')
    monkeypatch.setattr(_chunks, 'MAX_FILES', 1)
    with pytest.raises(ModelRetry, match='exceeds 1 files'):
        await _discover(_workspace(tmp_path), '.')


async def test_discover_without_ripgrep(tmp_path: Path) -> None:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    workspace = Workspace(LocalWorkspaceBackend(tmp_path, env={'PATH': tools_path(bin_dir)}))
    with pytest.raises(ToolFailed, match='needs ripgrep'):
        await _discover(workspace, '.')


async def test_discover_skips_files_that_vanish_after_listing(tmp_path: Path) -> None:
    (tmp_path / 'gone.py').write_text('x = 1\n')
    (tmp_path / 'odd.py').write_text('x = 1\n')
    (tmp_path / 'kept.py').write_text('y = 2\n')
    found = await _discover(Workspace(VanishingBackend(tmp_path)), '.')
    assert found.skipped == [('gone.py', 'No such file or directory'), ('odd.py', 'OSError')]
    assert [c.path for c in found.chunks] == ['kept.py']


# ---------------------------------------------------------------- caching


async def test_cached_index_rechunks_only_changed_files(tmp_path: Path, chunker: CountingChunker) -> None:
    (tmp_path / 'a.py').write_text('def alpha():\n    return 1\n')
    (tmp_path / 'b.py').write_text('def beta():\n    return 2\n')
    (tmp_path / 'c.py').write_text('def gamma():\n    return 3\n')
    workspace, indexes = _workspace(tmp_path), SnippetIndexes()
    await _discover(workspace, '.', indexes=indexes)
    assert sorted(chunker.paths) == ['a.py', 'b.py', 'c.py']

    chunker.paths.clear()
    (tmp_path / 'b.py').write_text('def beta_renamed():\n    return 2\n')
    (tmp_path / 'c.py').unlink()
    found = await _discover(workspace, '.', indexes=indexes, query='beta renamed')
    assert chunker.paths == ['b.py']  # unchanged `a.py` is not chunked again
    assert found.shortlist.snippets == 2 and found.files == 2  # `c.py`'s snippet is retired
    assert found.chunks[0].symbol == 'beta_renamed'


async def test_uncached_index_rebuilds_every_search(tmp_path: Path, chunker: CountingChunker) -> None:
    (tmp_path / 'a.py').write_text('x = 1\n')
    workspace, indexes = _workspace(tmp_path), SnippetIndexes(0)
    await _discover(workspace, '.', indexes=indexes)
    await _discover(workspace, '.', indexes=indexes)
    assert chunker.paths == ['a.py', 'a.py']


async def test_parallel_searches_build_the_index_once(tmp_path: Path, chunker: CountingChunker) -> None:
    for i in range(5):
        (tmp_path / f'm{i}.py').write_text(f'def f{i}():\n    return {i}\n')
    workspace, indexes = _workspace(tmp_path), SnippetIndexes()
    results = await asyncio.gather(*(_discover(workspace, '.', indexes=indexes, query=f'f{i}') for i in range(3)))
    assert sorted(chunker.paths) == [f'm{i}.py' for i in range(5)]  # the first call builds, the others reuse
    assert [found.shortlist.snippets for found in results] == [5, 5, 5]


async def test_least_recently_searched_index_is_evicted(tmp_path: Path, chunker: CountingChunker) -> None:
    for name in 'abc':
        (tmp_path / name).mkdir()
        (tmp_path / name / 'x.py').write_text('x = 1\n')
    workspace, indexes = _workspace(tmp_path), SnippetIndexes(2)
    for name in 'abca':  # `a` was evicted by `c`, so it is rebuilt
        await _discover(workspace, name, indexes=indexes)
    await _discover(workspace, 'c', indexes=indexes)  # still cached
    assert chunker.paths == ['x.py'] * 4


async def test_an_index_being_searched_is_not_evicted() -> None:
    indexes = SnippetIndexes(1)
    slots = indexes._slots  # pyright: ignore[reportPrivateUsage]
    busy = indexes._slot(('a', None))  # pyright: ignore[reportPrivateUsage]
    async with busy.lock:
        indexes._slot(('b', None))  # pyright: ignore[reportPrivateUsage]
        assert list(slots) == [('a', None), ('b', None)]  # over the limit while `a` is searched
    indexes._slot(('c', None))  # pyright: ignore[reportPrivateUsage]
    assert list(slots) == [('c', None)]


async def test_line_budget_counts_unicode_line_separators(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / 'a.py').write_text('x = 1\u2028' * 50)  # 50 lines to the chunker, no newline at all
    monkeypatch.setattr(_index, 'MAX_TOTAL_LINES', 10)
    with pytest.raises(ModelRetry, match='lines of source'):
        await _discover(_workspace(tmp_path), '.')


async def test_retired_snippets_are_compacted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_index, 'COMPACT_AFTER', 0)
    workspace, indexes = _workspace(tmp_path), SnippetIndexes()
    for version in range(3):
        (tmp_path / 'a.py').write_text(f'def retry_{version}():\n    return {version}\n')
        found = await _discover(workspace, '.', indexes=indexes, query=f'retry {version}')
        assert [c.symbol for c in found.chunks] == [f'retry_{version}']


async def test_files_changed_after_shortlisting_are_left_out(tmp_path: Path) -> None:
    for name in ('edited.py', 'deleted.py', 'kept.py'):
        (tmp_path / name).write_text('x = 1\n')
    workspace = _workspace(tmp_path)
    root = await searchable_root(workspace, '.')
    found = await SnippetIndexes(0).shortlist(workspace, root, None, 'x', 10)
    (tmp_path / 'edited.py').write_text('x = 2\n')
    (tmp_path / 'deleted.py').unlink()
    assert [c.path for c in await read_snippets(workspace, root, '.', found.hits)] == ['kept.py']


async def test_search_over_budget_keeps_no_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / 'a.py').write_text('x = 1\n' * 50)
    workspace, indexes = _workspace(tmp_path), SnippetIndexes()
    await _discover(workspace, '.', indexes=indexes)
    monkeypatch.setattr(_index, 'MAX_TOTAL_LINES', 10)
    with pytest.raises(ModelRetry, match='lines of source'):
        await _discover(workspace, '.', indexes=indexes)
    assert indexes._slots == {}  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------- BM25


def test_bm25_prefers_lexical_and_synonym_hits() -> None:
    index = Bm25()
    a = index.add('def render(): pass', 'a.py')
    b = index.add('def backoff(): attempt()', 'b.py')
    c = index.add('x = 1', 'retry.py')  # the path is boosted metadata
    scores = index.scores('retry the request')
    assert a not in scores  # no overlap at all
    assert set(scores) == {b, c}
    assert Bm25().scores('anything') == {}


def test_bm25_retires_and_compacts() -> None:
    index = Bm25()
    gone = index.add('unique retry', 'a.py')
    kept = index.add('retry again', 'b.py')
    index.retire(gone)
    assert (index.size, index.retired) == (1, 1)
    assert set(index.scores('unique retry')) == {kept}
    renumber = index.compact()
    assert (list(renumber), index.retired) == ([-1, 0], 0)
    assert set(index.scores('unique retry')) == {0}  # `unique` had no live postings left, and is dropped
