"""Querying a spill: `grep_tool_result`, and `read_tool_result` with `unit='bytes'`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.tool_output_limits import (
    GREP_TOOL_NAME,
    GREP_TOOL_RESULT_KIND,
    READ_TOOL_NAME,
    READ_TOOL_RESULT_KIND,
    Band,
    LocalFileOverflowStore,
    Spill,
    ToolOutputLimits,
)

_LOG = '\n'.join(f'ERROR {i}' if i in {5, 7, 15} else f'line {i}' for i in range(1, 21))


async def _query(tmp_path: Path, payload: str, tool_name: str, args: dict[str, Any], handle: str | None = None) -> str:
    """Spill `payload`, call `tool_name` on its handle with `args`, and return what the model got back."""
    offered: dict[str, str | None] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.update({tool.name: tool.tool_kind for tool in info.function_tools})
        parts = [part for message in messages for part in message.parts]
        if answer := [part for part in parts if isinstance(part, ToolReturnPart) and part.tool_name == tool_name]:
            retried = 'retry: ' if answer[0].outcome == 'retried' else ''
            return ModelResponse(parts=[TextPart(f'{retried}{answer[0].content}')])
        if spilled := [part for part in parts if isinstance(part, ToolReturnPart)]:
            assert spilled[0].metadata is not None
            stored = handle if handle is not None else spilled[0].metadata['overflow_handle']
            return ModelResponse(parts=[ToolCallPart(tool_name, {'handle': stored, **args})])
        return ModelResponse(parts=[ToolCallPart('big_tool', {})])

    limits: ToolOutputLimits[None] = ToolOutputLimits(
        bands=[Band(over=50, action=Spill())], store=LocalFileOverflowStore(base_dir=tmp_path)
    )
    agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[limits])

    @agent.tool_plain
    def big_tool() -> str:
        return payload

    result = await agent.run('go')
    assert offered[READ_TOOL_NAME] == READ_TOOL_RESULT_KIND.id
    assert offered[GREP_TOOL_NAME] == GREP_TOOL_RESULT_KIND.id
    return result.output.replace(str(tmp_path), '<tmp>')


class TestGrepToolResult:
    async def test_numbers_lines_and_merges_overlapping_context(self, tmp_path: Path) -> None:
        out = await _query(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'ERROR', 'context': 1})
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
        first = await _query(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'ERROR', 'context': 0, 'max_matches': 2})
        assert first.endswith(
            ': 3 matching line(s); showing 2, 1 more; continue with offset=2]\n5:ERROR 5\n--\n7:ERROR 7'
        )
        rest = await _query(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'ERROR', 'context': 0, 'offset': 2})
        assert rest.endswith(': 3 matching line(s); showing 1]\n15:ERROR 15')

    async def test_pattern_is_literal(self, tmp_path: Path) -> None:
        out = await _query(tmp_path, _LOG + '\n.* literal', GREP_TOOL_NAME, {'pattern': '.*', 'context': 0})
        assert out.endswith(': 1 matching line(s); showing 1]\n21:.* literal')

    async def test_no_match(self, tmp_path: Path) -> None:
        out = await _query(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'WARN'})
        assert out.endswith(': 0 matching line(s); showing 0]')
        assert '\n' not in out

    async def test_context_is_clamped(self, tmp_path: Path) -> None:
        payload = '\n'.join([f'line {i}' for i in range(1, 31)] + ['hit'] + [f'line {i}' for i in range(32, 61)])
        out = await _query(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit', 'context': 50})
        body = out.split('\n', 1)[1].splitlines()
        assert (body[0], body[10], body[-1]) == ('21-line 21', '31:hit', '41-line 41')

    async def test_matches_are_clamped(self, tmp_path: Path) -> None:
        out = await _query(tmp_path, 'hit\n' * 300, GREP_TOOL_NAME, {'pattern': 'hit', 'max_matches': 1_000})
        assert out.split('\n', 1)[0].endswith(
            ': 300 matching line(s); showing 200, 100 more; continue with offset=200]'
        )

    async def test_long_line_cut_by_the_cap_names_what_was_omitted(self, tmp_path: Path) -> None:
        out = await _query(tmp_path, 'x' * 60_000, GREP_TOOL_NAME, {'pattern': 'x'})
        assert out.split('\n', 1)[0].endswith(
            ': 1 matching line(s); showing 1, output capped at 50,000 chars with 10,002 omitted; '
            "read the rest of a long line with read_tool_result's unit='bytes']"
        )

    async def test_cap_stops_at_whole_lines_and_continues_from_there(self, tmp_path: Path) -> None:
        payload = '\n'.join(f'hit {i} ' + 'x' * 5_000 for i in range(25))
        first = await _query(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit', 'context': 0})
        header, body = first.split('\n', 1)
        assert header.endswith(': 25 matching line(s); showing 9, 16 more; continue with offset=9]')
        assert body.splitlines()[-1].startswith('9:hit 8 ')
        rest = await _query(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit', 'context': 0, 'offset': 9})
        assert rest.split('\n', 2)[1].startswith('10:hit 9 ')

    async def test_cap_drops_a_trailing_group_separator(self, tmp_path: Path) -> None:
        payload = '\n'.join(['hit a ' + 'x' * 30_000, 'gap', 'hit b ' + 'x' * 30_000])
        out = await _query(tmp_path, payload, GREP_TOOL_NAME, {'pattern': 'hit', 'context': 0})
        header, body = out.split('\n', 1)
        assert header.endswith(': 2 matching line(s); showing 1, 1 more; continue with offset=1]')
        assert body.startswith('1:hit a ') and '\n' not in body

    async def test_only_matches_on_this_page_are_marked(self, tmp_path: Path) -> None:
        out = await _query(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'ERROR', 'max_matches': 1})
        assert out.split('\n', 1)[1] == snapshot("""\
3-line 3
4-line 4
5:ERROR 5
6-line 6
7-ERROR 7\
""")

    async def test_missing_handle(self, tmp_path: Path) -> None:
        out = await _query(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'ERROR'}, handle='missing/1.0')
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
    async def test_invalid_input_asks_for_a_retry(self, tmp_path: Path, args: dict[str, Any], error: str) -> None:
        assert await _query(tmp_path, _LOG, GREP_TOOL_NAME, args) == f'retry: {error}'

    async def test_result_is_not_reduced(self, tmp_path: Path) -> None:
        out = await _query(tmp_path, _LOG, GREP_TOOL_NAME, {'pattern': 'line'})
        assert len(out) > 50
        assert 'Tool output too large' not in out


class TestReadToolResultBytes:
    async def test_slice(self, tmp_path: Path) -> None:
        payload = '{"a":' + '1' * 100 + '}'
        out = await _query(tmp_path, payload, READ_TOOL_NAME, {'unit': 'bytes', 'offset': 2, 'limit': 5})
        assert out.endswith(': bytes 2-7 of 106 (end exclusive)]\na":11')

    async def test_from_end(self, tmp_path: Path) -> None:
        payload = 'x' * 100 + 'tail'
        out = await _query(tmp_path, payload, READ_TOOL_NAME, {'unit': 'bytes', 'limit': 4, 'from_end': True})
        assert out.endswith(': bytes 100-104 of 104 (end exclusive)]\ntail')

    async def test_range_is_clamped(self, tmp_path: Path) -> None:
        past = await _query(tmp_path, 'x' * 100, READ_TOOL_NAME, {'unit': 'bytes', 'offset': 500})
        assert past.endswith(': bytes 100-100 of 100 (end exclusive)]')
        whole = await _query(tmp_path, 'x' * 60_000, READ_TOOL_NAME, {'unit': 'bytes', 'limit': 100_000})
        assert whole.split('\n', 1)[0].endswith(': bytes 0-50000 of 60000 (end exclusive)]')

    async def test_cut_utf8_sequence_renders_as_replacement_character(self, tmp_path: Path) -> None:
        payload = 'é' * 60
        out = await _query(tmp_path, payload, READ_TOOL_NAME, {'unit': 'bytes', 'offset': 1, 'limit': 4})
        assert out.endswith(': bytes 1-5 of 120 (end exclusive)]\n\ufffdé\ufffd')

    async def test_pattern_is_rejected(self, tmp_path: Path) -> None:
        out = await _query(tmp_path, _LOG, READ_TOOL_NAME, {'unit': 'bytes', 'pattern': 'ERROR'})
        assert out == "retry: `pattern` filters lines, so it cannot be combined with `unit='bytes'`."

    async def test_capped_line_read_names_the_recovery(self, tmp_path: Path) -> None:
        out = await _query(tmp_path, 'x' * 60_000, READ_TOOL_NAME, {})
        assert out.split('\n', 1)[0].endswith(
            ': 1 matching line(s); showing 1, output capped at 50,000 chars with 10,000 omitted; '
            "read fewer lines, or long lines with unit='bytes']"
        )
