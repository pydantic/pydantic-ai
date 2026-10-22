"""Tests for `SmartSearch`'s index: discovery through the workspace, budgets, the lexical shortlist, and caching."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest

from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace

from ..filesystem.conftest import tools_path
from .conftest import PY_SOURCE, FakeJev, VanishingBackend, judged, search, search_agent, search_all, search_error

pytestmark = pytest.mark.usefixtures('allow_model_requests')  # `FakeJev` is a real `DecisionModel`


class GatedBackend(LocalWorkspaceBackend):
    """Holds reads under the directories named in `held` until `release[name]` is set.

    A search holds its index while it reads, so this keeps that index busy for as long as a test needs.
    """

    def __init__(self, root: Path, held: str) -> None:
        super().__init__(root)
        self.reading = {name: anyio.Event() for name in held}
        self.release = {name: anyio.Event() for name in held}

    async def read_bytes(self, path: str) -> bytes:
        name = Path(path).parent.name
        if name in self.release:
            self.reading[name].set()
            await self.release[name].wait()
        return await super().read_bytes(path)


class EditingBackend(LocalWorkspaceBackend):
    """Serves every file as it is for indexing, then edits `edited.py` and deletes `deleted.py` before a re-read."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.reads = Counter[str]()

    async def read_bytes(self, path: str) -> bytes:
        name = Path(path).name
        self.reads[name] += 1
        if self.reads[name] > 1 and name == 'edited.py':
            return b'x = 2\n'
        if self.reads[name] > 1 and name == 'deleted.py':
            raise FileNotFoundError(2, 'No such file or directory', path)
        return await super().read_bytes(path)


def _judged_paths(model: FakeJev) -> list[str]:
    return [snippet.path for snippet in judged(model)]


class TestSmartSearchIndex:
    # ------------------------------------------------------------ discovery

    async def test_skips_binary_non_utf8_and_minified_files(self, tmp_path: Path) -> None:
        (tmp_path / 'good.py').write_text(PY_SOURCE)
        (tmp_path / 'bin.dat').write_bytes(b'abc\0def')
        (tmp_path / 'latin.txt').write_bytes('caf\xe9'.encode('latin-1'))
        (tmp_path / 'min.js').write_text('x' * 13_000)
        model = FakeJev('')
        result = await search(search_agent(tmp_path, model))
        assert (result.coverage.files, result.coverage.skipped_files) == (4, 3)
        assert result.warnings == ['3 files skipped (binary, non-UTF-8, minified or unreadable).']
        assert set(_judged_paths(model)) == {'good.py'}

    async def test_respects_gitignore_and_reports_paths_as_spelled(self, tmp_path: Path) -> None:
        (tmp_path / 'src').mkdir()
        (tmp_path / 'src' / 'a.py').write_text('x = 1\n')
        (tmp_path / 'src' / 'ignored.py').write_text('y = 2\n')
        (tmp_path / '.gitignore').write_text('src/ignored.py\n')
        (tmp_path / '.git').mkdir()
        model = FakeJev('')
        agent = search_agent(tmp_path, model)

        [match] = (await search(agent, directory='src')).matches
        assert match.file_path == 'src/a.py'
        [match] = (await search(agent, directory=str(tmp_path / 'src'))).matches
        assert match.file_path == str(tmp_path / 'src' / 'a.py')
        model.requests.clear()
        assert (await search(agent, directory='src', glob='*.txt')).coverage.files == 0
        assert model.requests == []

    async def test_glob_cannot_reach_ignored_or_hidden_files(self, tmp_path: Path) -> None:
        (tmp_path / '.git').mkdir()
        (tmp_path / '.gitignore').write_text('ignored.py\n.env\n')
        (tmp_path / '.env').write_text('SECRET=1\n')
        (tmp_path / 'ignored.py').write_text('x = 1\n')
        (tmp_path / 'kept.py').write_text('y = 2\n')
        model = FakeJev('')
        agent = search_agent(tmp_path, model)
        # ripgrep's `--glob` alone would list `.env` and send it to the judge
        assert (await search(agent, glob='.env')).coverage.files == 0
        assert model.requests == []
        await search(agent, glob='*.py')
        assert _judged_paths(model) == ['kept.py']

    async def test_stays_inside_the_working_directory(self, tmp_path: Path) -> None:
        project, outside = tmp_path / 'project', tmp_path / 'outside'
        project.mkdir()
        outside.mkdir()
        (outside / 'secret.py').write_text('token = 1\n')
        (project / 'link').symlink_to(outside)
        model = FakeJev('')
        agent = search_agent(project, model)
        for directory in ('..', str(outside), 'link'):
            assert await search_error(agent, directory=directory) == (
                f'{directory!r} is outside the working directory; `smart_search` only searches inside it.'
            )
        assert model.requests == []

    async def test_without_ripgrep_the_failure_is_reported(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / 'bin'
        bin_dir.mkdir()
        workspace = Workspace(LocalWorkspaceBackend(tmp_path, env={'PATH': tools_path(bin_dir)}))
        assert 'needs ripgrep' in await search_error(search_agent(None, FakeJev('')), workspace=workspace)

    async def test_skips_files_that_vanish_after_listing(self, tmp_path: Path) -> None:
        (tmp_path / 'gone.py').write_text('x = 1\n')
        (tmp_path / 'odd.py').write_text('x = 1\n')
        (tmp_path / 'kept.py').write_text('y = 2\n')
        model = FakeJev('')
        result = await search(search_agent(None, model), workspace=Workspace(VanishingBackend(tmp_path)))
        assert (result.coverage.files, result.coverage.skipped_files) == (3, 2)
        assert _judged_paths(model) == ['kept.py']

    async def test_files_changed_after_shortlisting_are_left_out(self, tmp_path: Path) -> None:
        for name in ('edited.py', 'deleted.py', 'kept.py'):
            (tmp_path / name).write_text('x = 1\n')
        model = FakeJev('')
        result = await search(search_agent(None, model), workspace=Workspace(EditingBackend(tmp_path)))
        assert _judged_paths(model) == ['kept.py']
        assert (result.coverage.snippets, result.coverage.evaluated) == (3, 1)
        assert result.warnings == [
            '2 shortlisted snippets were left out because their files changed during the search; search again '
            'to include them.'
        ]

    # ------------------------------------------------------------ budgets

    async def test_enforces_line_and_file_budgets(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / 'a.py').write_text('x = 1\n' * 50)
        (tmp_path / 'b.py').write_text('y = 1\n')
        model = FakeJev('')
        agent = search_agent(tmp_path, model)
        # The real budgets (50M lines, 200k files) are too large to reach in a test.
        monkeypatch.setattr('pydantic_ai_harness.smart_search._index.MAX_TOTAL_LINES', 10)
        assert await search_error(agent) == 'Search exceeds 10 lines of source. Narrow the directory or glob.'
        monkeypatch.setattr('pydantic_ai_harness.smart_search._chunks.MAX_FILES', 1)
        assert await search_error(agent) == 'Search exceeds 1 files. Narrow the directory.'
        assert await search_error(agent, glob='*.py') == 'Search exceeds 1 files. Narrow the directory or glob.'
        assert model.requests == []
        # The file cap applies after the glob: one matching file is searchable in a larger workspace.
        await search(agent, glob='b.py')
        assert _judged_paths(model) == ['b.py']

    async def test_line_budget_counts_carriage_returns(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / 'a.py').write_bytes(b'x = 1\r' * 50)  # 50 lines to the chunker, no newline at all
        monkeypatch.setattr('pydantic_ai_harness.smart_search._index.MAX_TOTAL_LINES', 10)
        assert 'lines of source' in await search_error(search_agent(tmp_path, FakeJev('')))

    async def test_enforces_byte_budget(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / 'a.py').write_text('x = 1\n' * 50)
        agent = search_agent(tmp_path, FakeJev(''))
        assert (await search(agent)).matches
        monkeypatch.setattr('pydantic_ai_harness.smart_search._index.MAX_TOTAL_BYTES', 100)  # really 2 GiB
        assert 'GiB of source' in await search_error(agent)

    async def test_search_over_budget_keeps_no_index(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, indexed: Callable[[], list[int | None]]
    ) -> None:
        (tmp_path / 'a.py').write_text('x = 1\n' * 50)
        agent = search_agent(tmp_path, FakeJev(''), cache_index=True)
        await search(agent)
        with monkeypatch.context() as patch:
            patch.setattr('pydantic_ai_harness.smart_search._index.MAX_TOTAL_LINES', 10)
            assert 'lines of source' in await search_error(agent)
        await search(agent)
        assert indexed() == [1, None, 1]  # the index cached by the first search was dropped, so it is rebuilt

    # ------------------------------------------------------------ shortlist

    async def test_shortlist_prefers_lexical_and_synonym_hits(self, tmp_path: Path) -> None:
        (tmp_path / 'a.py').write_text('def render(): pass\n')  # no overlap at all
        (tmp_path / 'b.py').write_text('def backoff(): attempt()\n')  # synonyms of `retry`
        (tmp_path / 'retry.py').write_text('x = 1\n')  # the path is boosted metadata
        model = FakeJev('')
        await search(search_agent(tmp_path, model), query='retry the request', candidates=2)
        assert _judged_paths(model) == ['b.py', 'retry.py']

    @pytest.mark.parametrize(
        'query',
        ['where is the token', 'where is the value', 'where is the httpserver'],
        ids=['camel-case', 'snake-case', 'acronym'],
    )
    async def test_shortlist_splits_identifiers_and_drops_stopwords(self, tmp_path: Path, query: str) -> None:
        (tmp_path / 'a.txt').write_text('where is the where is the\n')  # only stopwords
        (tmp_path / 'z.txt').write_text('refreshToken_value HTTPServer\n')
        model = FakeJev('')
        await search(search_agent(tmp_path, model), query=query, candidates=1)
        assert _judged_paths(model) == ['z.txt']

    # ------------------------------------------------------------ caching

    @pytest.mark.parametrize(('cache_index', 'second'), [(True, 0), (False, 1)])
    async def test_cache_index_keeps_the_index_across_runs(
        self, tmp_path: Path, indexed: Callable[[], list[int | None]], cache_index: bool, second: int
    ) -> None:
        (tmp_path / 'a.py').write_text('expired = True\n')
        agent = search_agent(tmp_path, FakeJev('x'), cache_index=cache_index)
        await search(agent)
        await search(agent)
        assert indexed() == [1, second]

    async def test_cached_index_rechunks_only_changed_files(
        self, tmp_path: Path, indexed: Callable[[], list[int | None]]
    ) -> None:
        (tmp_path / 'a.py').write_text('def alpha():\n    return 1\n')
        (tmp_path / 'b.py').write_text('def beta():\n    return 2\n')
        (tmp_path / 'c.py').write_text('def gamma():\n    return 3\n')
        model = FakeJev('')
        agent = search_agent(tmp_path, model, cache_index=True)
        await search(agent)
        model.requests.clear()
        (tmp_path / 'b.py').write_text('def beta_renamed():\n    return 2\n')
        (tmp_path / 'c.py').unlink()
        result = await search(agent, query='beta renamed', candidates=1)
        assert indexed() == [3, 1]  # only the edited `b.py` is chunked again
        assert (result.coverage.snippets, result.coverage.files) == (2, 2)  # `c.py`'s snippet is retired
        assert [snippet.symbol for snippet in judged(model)] == ['beta_renamed']

    async def test_retired_snippets_are_compacted(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Compaction waits for 10,000 retired snippets; indexing that many takes seconds, so compact at once.
        monkeypatch.setattr('pydantic_ai_harness.smart_search._index.COMPACT_AFTER', 0)
        model = FakeJev('')
        agent = search_agent(tmp_path, model, cache_index=True)
        for name in ('alpha', 'beta', 'gamma'):  # each edit retires the last version's snippet
            (tmp_path / 'a.py').write_text(f'def retry_{name}():\n    return {name!r}\n')
            model.requests.clear()
            result = await search(agent, query=f'retry {name}', candidates=1)
            assert [snippet.symbol for snippet in judged(model)] == [f'retry_{name}']
            assert result.coverage.snippets == 1

    async def test_parallel_searches_build_the_index_once(
        self, tmp_path: Path, indexed: Callable[[], list[int | None]]
    ) -> None:
        for i in range(5):
            (tmp_path / f'm{i}.py').write_text(f'def f{i}():\n    return {i}\n')
        agent = search_agent(tmp_path, FakeJev(''), cache_index=True)
        results = await search_all(agent, *({'query': f'f{i}'} for i in range(3)))
        assert sorted(indexed(), key=str) == [0, 0, 5]  # the first call builds, the others reuse
        assert [result.coverage.snippets for result in results] == [5, 5, 5]

    async def test_least_recently_searched_index_is_evicted(
        self, tmp_path: Path, indexed: Callable[[], list[int | None]]
    ) -> None:
        for name in 'abcde':
            (tmp_path / name).mkdir()
            (tmp_path / name / f'{name}.py').write_text('x = 1\n')
        agent = search_agent(tmp_path, FakeJev(''), cache_index=True)
        for name in 'abcadeba':
            await search(agent, directory=name)
        # Under four, nothing is evicted: `a` is still cached. Then `e` evicts `b`, the least recently searched,
        # and the rebuilt `b` evicts `c`, leaving `a` cached.
        assert indexed() == [1, 1, 1, 0, 1, 1, 1, 0]

    async def test_an_index_being_searched_is_not_evicted(
        self, tmp_path: Path, indexed: Callable[[], list[int | None]]
    ) -> None:
        for name in 'abcde':
            (tmp_path / name).mkdir()
            (tmp_path / name / f'{name}.py').write_text('x = 1\n')
        backend = GatedBackend(tmp_path, held='a')
        agent = search_agent(None, FakeJev(''), cache_index=True)

        async def run(directory: str) -> None:
            await search(agent, workspace=Workspace(backend), directory=directory)

        async with anyio.create_task_group() as group:
            group.start_soon(run, 'a')
            await backend.reading['a'].wait()
            for name in 'bcde':  # the fifth index evicts the least recently searched idle one: `b`, not `a`
                await run(name)
            backend.release['a'].set()
        await run('a')  # still cached
        await run('b')  # rebuilt
        assert indexed() == [1, 1, 1, 1, 1, 0, 1]  # `b`-`e`, then the held `a`; `a` again cached, `b` rebuilt

    async def test_parallel_searches_trim_the_cache_when_done(
        self, tmp_path: Path, indexed: Callable[[], list[int | None]]
    ) -> None:
        for name in 'abcde':
            (tmp_path / name).mkdir()
            (tmp_path / name / f'{name}.py').write_text('x = 1\n')
        backend = GatedBackend(tmp_path, held='abcde')
        agent = search_agent(None, FakeJev(''), cache_index=True)

        async def run(directory: str) -> None:
            await search(agent, workspace=Workspace(backend), directory=directory)

        async with anyio.create_task_group() as others:
            for name in 'bcde':
                others.start_soon(run, name)
            async with anyio.create_task_group() as first:
                first.start_soon(run, 'a')
                for name in 'abcde':  # five indexes held at once: one more than are kept
                    await backend.reading[name].wait()
                backend.release['a'].set()
            # `a` finished while the others were busy, so it was the one dropped to get back to four.
            for name in 'bcde':
                backend.release[name].set()
        for name in 'bcdea':
            await run(name)
        assert indexed() == [1] * 5 + [0, 0, 0, 0, 1]  # `b`-`e` still cached, `a` rebuilt
