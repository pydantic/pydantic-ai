"""Searching a spill with `grep_tool_result`."""

from __future__ import annotations

from pathlib import Path

import pytest
from inline_snapshot import snapshot

from pydantic_ai_harness.tool_output_limits import GREP_TOOL_NAME
from tests.harness.tool_output_limits._spill_query import query_spill

_LOG = '\n'.join(f'ERROR {i}' if i in {5, 7, 15} else f'line {i}' for i in range(1, 21))


class TestGrepToolResult:
    @pytest.mark.parametrize(
        'text',
        ['\u00e91\r\nb\rc\u00e9\n\nd\u2028\u00e92\x85f\r\n\r\u00e93', '\u00e91\r\nb\r\u00e92\r', 'ab\r\nc\r\r\u00e9'],
    )
    async def test_streams_in_chunks_like_splitlines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.tool_output_limits._store._CHUNK_SIZE', 3)
        payload = text + 'x' * 60
        out = await query_spill(tmp_path, payload, GREP_TOOL_NAME, {'pattern': '\u00e9', 'context': 0})
        matches = [f'{n + 1}:{line}' for n, line in enumerate(payload.splitlines()) if '\u00e9' in line]
        assert [line for line in out.split('\n')[1:] if line != '--'] == matches

    async def test_numbers_lines_and_merges_overlapping_context(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'ERROR', 'context': 1})
        assert out.split('\n', 1)[1] == snapshot("""\
4-line 4
5:ERROR 5
6-line 6
7:ERROR 7
8-line 8
--
14-line 14
15:ERROR 15
16-line 16\
""")
        assert out.startswith('[handle ') and out.split('\n', 1)[0].endswith(': 3 matching line(s); showing 3]')

    async def test_match_cap_names_the_offset_to_continue_from(self, tmp_path: Path) -> None:
        first = await query_spill(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'ERROR', 'context': 0, 'max_matches': 2})
        assert first.endswith(
            ': 3 matching line(s); showing 2, 1 more; continue with offset=2]\n5:ERROR 5\n--\n7:ERROR 7'
        )
        rest = await query_spill(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'ERROR', 'context': 0, 'offset': 2})
        assert rest.endswith(': 3 matching line(s); showing 1]\n15:ERROR 15')

    async def test_pattern_is_literal(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, _LOG + '\n.* literal', GREP_TOOL_NAME, {'pattern': '.*', 'context': 0})
        assert out.endswith(': 1 matching line(s); showing 1]\n21:.* literal')

    async def test_no_match(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'WARN'})
        assert out.endswith(': 0 matching line(s); showing 0]')
        assert '\n' not in out

    async def test_context_is_clamped(self, tmp_path: Path) -> None:
        payload = '\n'.join([f'line {i}' for i in range(1, 31)] + ['hit'] + [f'line {i}' for i in range(32, 61)])
        out = await query_spill(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit', 'context': 50})
        body = out.split('\n', 1)[1].splitlines()
        assert (body[0], body[10], body[-1]) == ('21-line 21', '31:hit', '41-line 41')

    async def test_matches_are_clamped(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, 'hit\n' * 300, GREP_TOOL_NAME, {'pattern': 'hit', 'max_matches': 1_000})
        assert out.split('\n', 1)[0].endswith(
            ': 300 matching line(s); showing 200, 100 more; continue with offset=200]'
        )

    async def test_long_match_cut_by_the_cap_gives_its_byte_offset(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, 'é\n' + 'x' * 60_000 + '\nafter', GREP_TOOL_NAME, {'pattern': 'x'})
        header, body = out.split('\n', 1)
        assert header.endswith(
            ': 1 matching line(s); showing 1, output capped at 50,000 chars; '
            "context lines 1-1 omitted (read_tool_result(handle='<h>', offset=0, limit=1)); context lines 3-3 omitted (read_tool_result(handle='<h>', offset=2, limit=1)); "
            "the cut line continues at read_tool_result(handle='<h>', unit='bytes', offset=50001, limit=50000)]"
        )
        assert len(body) == 50_000 and body.startswith('2:x')

    async def test_long_context_before_the_first_match_is_dropped(self, tmp_path: Path) -> None:
        payload = '\n'.join(['a' * 30_000, 'b' * 30_000, 'hit ' + 'c' * 30_000, 'after'])
        out = await query_spill(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit'})
        header, body = out.split('\n', 1)
        assert header.endswith(
            ': 1 matching line(s); showing 1, output capped at 50,000 chars; '
            "context lines 1-2 omitted (read_tool_result(handle='<h>', offset=0, limit=2))]"
        )
        assert body.startswith('3:hit ') and body.endswith('\n4-after')
        out = await query_spill(
            tmp_path, '\n'.join(['a' * 30_000, 'hit ' + 'c' * 30_000]), GREP_TOOL_NAME, {'pattern': 'hit'}
        )
        header, body = out.split('\n', 1)
        assert header.endswith("context lines 1-1 omitted (read_tool_result(handle='<h>', offset=0, limit=1))]")
        assert body == '2:hit ' + 'c' * 30_000

    async def test_long_context_after_the_last_match_is_reported(self, tmp_path: Path) -> None:
        payload = '\n'.join(['hit ' + 'a' * 30_000, 'b' * 30_000, 'c'])
        out = await query_spill(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit'})
        header, body = out.split('\n', 1)
        assert header.endswith(
            ': 1 matching line(s); showing 1, output capped at 50,000 chars; '
            "context lines 2-3 omitted (read_tool_result(handle='<h>', offset=1, limit=2))]"
        )
        assert body == '1:hit ' + 'a' * 30_000

    async def test_omitted_context_stops_at_the_next_group(self, tmp_path: Path) -> None:
        payload = '\n'.join(['hit a ' + 'x' * 30_000, 'a' * 30_000, 'g1', 'g2', 'hit b'])
        out = await query_spill(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit', 'context': 1})
        assert out.split('\n', 1)[0].endswith(
            ': 2 matching line(s); showing 1, 1 more; continue with offset=1, output capped at 50,000 chars; '
            "context lines 2-2 omitted (read_tool_result(handle='<h>', offset=1, limit=1))]"
        )

    async def test_cap_leaves_the_next_groups_context_to_the_next_page(self, tmp_path: Path) -> None:
        payload = '\n'.join(['hit a ' + 'x' * 30_000, 'g1', 'g2', 'g3', 'c' * 30_000, 'hit b'])
        out = await query_spill(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit', 'context': 1})
        header, body = out.split('\n', 1)
        assert header.endswith(': 2 matching line(s); showing 1, 1 more; continue with offset=1]')
        assert body.endswith('\n2-g1')

    async def test_cap_stops_at_whole_lines_and_continues_from_there(self, tmp_path: Path) -> None:
        payload = '\n'.join(f'hit {i} ' + 'x' * 5_000 for i in range(25))
        first = await query_spill(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit', 'context': 0})
        header, body = first.split('\n', 1)
        assert header.endswith(': 25 matching line(s); showing 9, 16 more; continue with offset=9]')
        assert body.splitlines()[-1].startswith('9:hit 8 ')
        rest = await query_spill(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit', 'context': 0, 'offset': 9})
        assert rest.split('\n', 2)[1].startswith('10:hit 9 ')

    async def test_cap_drops_a_trailing_group_separator(self, tmp_path: Path) -> None:
        payload = '\n'.join(['hit a ' + 'x' * 30_000, 'gap', 'hit b ' + 'x' * 30_000])
        out = await query_spill(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit', 'context': 0})
        header, body = out.split('\n', 1)
        assert header.endswith(': 2 matching line(s); showing 1, 1 more; continue with offset=1]')
        assert body.startswith('1:hit a ') and '\n' not in body

    async def test_only_matches_on_this_page_are_marked(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'ERROR', 'max_matches': 1})
        assert out.split('\n', 1)[1] == snapshot("""\
3-line 3
4-line 4
5:ERROR 5
6-line 6
7-ERROR 7\
""")

    async def test_missing_handle(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'ERROR'}, handle='missing/1.0')
        assert out.startswith("[No stored tool result for handle 'missing/1.0'.")

    @pytest.mark.parametrize(
        ('args', 'error'),
        [
            ({'pattern': ''}, '`pattern` must not be empty.'),
            ({'pattern': 'x', 'context': -1}, '`context` must be >= 0.'),
            ({'pattern': 'x', 'max_matches': 0}, '`max_matches` must be >= 1.'),
            ({'pattern': 'x', 'offset': -1}, '`offset` must be >= 0.'),
        ],
    )
    async def test_invalid_input_asks_for_a_retry(self, tmp_path: Path, args: dict[str, object], error: str) -> None:
        assert await query_spill(tmp_path, _LOG, GREP_TOOL_NAME, args) == f'retry: {error}'

    async def test_result_is_not_reduced(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'line'})
        assert len(out) > 50
        assert 'Tool output too large' not in out
