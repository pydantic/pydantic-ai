"""Tests for how `SmartSearch` cuts files into snippets, read back from what the judge is asked about."""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import PY_SOURCE, FakeJev, Judged, judged, search, search_agent

pytestmark = pytest.mark.usefixtures('allow_model_requests')  # `FakeJev` is a real `DecisionModel`


async def _snippets(root: Path, name: str, source: str) -> list[Judged]:
    """Every snippet `name` is cut into: all of them are judged."""
    (root / name).write_text(source)
    model = FakeJev('')
    result = await search(search_agent(root, model), candidates=256)
    assert result.coverage.selection_complete
    return judged(model)


async def _spans(root: Path, name: str, source: str) -> list[tuple[int, int, str | None]]:
    return [snippet.span for snippet in await _snippets(root, name, source)]


class TestSmartSearchChunking:
    async def test_python_chunks_follow_declarations(self, tmp_path: Path) -> None:
        by_symbol = {s.symbol: s for s in await _snippets(tmp_path, 'a.py', PY_SOURCE)}
        imports = by_symbol[None]  # adjacent one-liners merge into one target
        assert (imports.line, imports.end_line) == (1, 2)
        fn = by_symbol['reject_expired']
        assert (fn.line, fn.end_line) == (3, 8)  # preceding blank gap is absorbed
        assert 'raise PermissionError' in fn.text
        assert by_symbol['Cache.get'].text.strip().startswith('def get')

    async def test_decorated_python_declaration_starts_at_its_decorator(self, tmp_path: Path) -> None:
        source = 'import os\n\n\n@cache\ndef load():\n    return os.environ\n'
        assert await _spans(tmp_path, 'a.py', source) == [(1, 1, None), (2, 6, 'load')]

    async def test_line_numbers_match_the_file_past_a_form_feed(self, tmp_path: Path) -> None:
        # `str.splitlines` would also split on the form feed and the U+2028 in the string, shifting later lines.
        source = "def load():\n    return 1\n\x0c\ndef save():\n    return '2\u2028'\n"
        assert await _spans(tmp_path, 'a.py', source) == [(1, 2, 'load'), (3, 5, 'save')]

    async def test_lines_after_the_last_python_statement_are_windowed(self, tmp_path: Path) -> None:
        source = 'def load():\n    return 1\n# trailing note\n'
        assert await _spans(tmp_path, 'a.py', source) == [(1, 2, 'load'), (3, 3, None)]

    async def test_long_function_is_split_into_blocks_covering_every_line(self, tmp_path: Path) -> None:
        body = '\n'.join(
            f'    if x == {i}:\n        a = {i}\n        b = {i}\n        c = {i}\n        d = {i}\n        e = {i}'
            for i in range(6)
        )
        source = f'def router(x):\n{body}\n'
        snippets = await _snippets(tmp_path, 'r.py', source)
        assert len(snippets) > 1
        assert all(s.symbol == 'router' for s in snippets)
        covered = {n for s in snippets for n in range(s.line, s.end_line + 1)}
        assert covered == set(range(1, len(source.splitlines()) + 1))

    async def test_long_body_groups_short_statements_and_isolates_blocks(self, tmp_path: Path) -> None:
        source = (
            'def f(x):\n    a = 1\n    b = 2\n    if x:\n'
            + ''.join(f'        a = {i}\n' for i in range(5))  # a 6-line block: lines 4-9
            + '    c = 3\n    if a:\n        y = [\n'
            + ''.join(f'            {i},\n' for i in range(27))
            + '        ]\n'  # a 30-line block of one statement: lines 11-40
        )
        assert await _spans(tmp_path, 'f.py', source) == [(1, 3, 'f'), (4, 9, 'f'), (10, 10, 'f'), (11, 40, 'f')]

    @pytest.mark.parametrize(('lines', 'expected'), [(23, [(1, 23, 'f')]), (24, [(1, 2, 'f'), (3, 24, 'f')])])
    async def test_declarations_of_24_lines_or_more_are_split(
        self, tmp_path: Path, lines: int, expected: list[tuple[int, int, str | None]]
    ) -> None:
        source = 'def f():\n    a = 1\n    b = [\n' + ''.join(f'        {i},\n' for i in range(lines - 4)) + '    ]\n'
        assert await _spans(tmp_path, 'f.py', source) == expected

    async def test_body_whose_block_ends_on_the_last_line_stays_whole(self, tmp_path: Path) -> None:
        # `y()` follows the 29-line assignment on its closing line, so no lines are left for it.
        source = 'def f():\n    x = [\n' + ''.join(f'        {i},\n' for i in range(27)) + '    ]; y()\n'
        assert await _spans(tmp_path, 'f.py', source) == [(1, 30, 'f')]

    async def test_deeply_nested_single_statement_bodies_stop_splitting(self, tmp_path: Path) -> None:
        source = 'def f():\n  with a:\n    with b:\n      with c:\n        with d:\n' + ''.join(
            f'          x{i} = {i}\n' for i in range(30)
        )
        assert await _spans(tmp_path, 'nested.py', source) == [(1, 35, 'f')]

    async def test_container_whose_first_member_shares_its_line_has_no_header(self, tmp_path: Path) -> None:
        assert await _spans(tmp_path, 'a.py', 'class A: x = 1\n') == [(1, 1, 'A.body')]

    async def test_python_without_statements_uses_windows(self, tmp_path: Path) -> None:
        assert await _spans(tmp_path, 'empty.py', '# only a comment\n') == [(1, 1, None)]

    @pytest.mark.parametrize('name', ['x.txt', 'broken.py'])
    async def test_non_python_and_broken_python_fall_back_to_windows(self, tmp_path: Path, name: str) -> None:
        text = '\n'.join(f'line {i}' for i in range(130))
        source = 'def (:\n' + text if name == 'broken.py' else text
        last = len(source.splitlines())
        # 60-line windows with a 10-line overlap
        assert await _spans(tmp_path, name, source) == [(1, 60, None), (51, 110, None), (101, last, None)]

    async def test_whitespace_only_windows_are_dropped(self, tmp_path: Path) -> None:
        (tmp_path / 'blank.txt').write_text('\n   \n')
        result = await search(search_agent(tmp_path, FakeJev('')))
        assert (result.coverage.files, result.coverage.snippets, result.coverage.skipped_files) == (1, 0, 0)

    async def test_long_whitespace_lines_are_skipped_not_fatal(self, tmp_path: Path) -> None:
        snippets = await _snippets(tmp_path, 'w.txt', 'x = 1\n' + ' ' * 13_000 + '\ny = 2\n')
        assert [(s.span, s.text) for s in snippets] == [((1, 1, None), 'x = 1'), ((3, 3, None), 'y = 2')]

    async def test_giant_window_halves_and_giant_line_skips_the_file(self, tmp_path: Path) -> None:
        assert await _spans(tmp_path, 'big.txt', ('y' * 3_500 + '\n') * 4) == [(1, 3, None), (4, 4, None)]

        (tmp_path / 'min.js').write_text('x' * 13_000)
        result = await search(search_agent(tmp_path, FakeJev('')), candidates=256)
        assert result.coverage.skipped_files == 1  # too long to judge, never clipped
        assert result.coverage.snippets == 2
