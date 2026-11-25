"""Tests for pydantic_ai_harness.tool_output_limits."""

from __future__ import annotations

import base64
import dataclasses
import io
import os
import re
import sys
import time
from collections.abc import AsyncIterable, AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import anyio
import anyio.lowlevel
import pytest
from inline_snapshot import snapshot
from pydantic_core import to_json

from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.capabilities import Capability
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryContent,
    CompactionPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import AbstractModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace
from pydantic_ai_harness.compaction import SummarizingCompaction
from pydantic_ai_harness.tool_output_limits import (
    READ_TOOL_RESULT_KIND,
    Band,
    LocalFileOverflowStore,
    Passthrough,
    Spill,
    Summarize,
    ToolOutputLimits,
    Truncate,
    TruncationStrategy,
    WorkspaceOverflowStore,
    drain_summary_events,
    indented_json,
    json_lines,
)
from pydantic_ai_harness.tool_output_limits._capability import (
    READ_TOOL_NAME,
    _build_spill_preview,  # pyright: ignore[reportPrivateUsage]
    _head_tail_preview,  # pyright: ignore[reportPrivateUsage]
    _read_slice,  # pyright: ignore[reportPrivateUsage]
    _select_action,  # pyright: ignore[reportPrivateUsage]
    _Unit,  # pyright: ignore[reportPrivateUsage]
    _with_handles,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai_harness.tool_output_limits._payload import (
    is_binary,
    json_sketch,
    measure,
    strip_ansi,
    to_bytes,
    to_text,
)
from pydantic_ai_harness.tool_output_limits._store import _safe_segment  # pyright: ignore[reportPrivateUsage]
from tests.conftest import try_import
from tests.function_model_prefix import COMPACTION_REWRITES_HISTORY
from tests.harness._recording_durability import RecordingDurability
from tests.harness.conftest import agent_run_names
from tests.harness.tool_output_limits._spill_query import query_spill

with try_import() as ag_ui_imports_successful:
    from ag_ui.core import CustomEvent, RunAgentInput, UserMessage

    from pydantic_ai.ui.ag_ui import AGUIAdapter

with try_import() as vercel_ai_imports_successful:
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter
    from pydantic_ai.ui.vercel_ai.request_types import DataUIPart
    from pydantic_ai.ui.vercel_ai.response_types import DataChunk

if TYPE_CHECKING:
    from logfire.testing import CaptureLogfire


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    *, run_id: str | None = 'run-1', retry: int = 0, model: Any = None, usage_limits: UsageLimits | None = None
) -> Any:
    """Build a minimal RunContext-like object for testing the hook directly."""

    @dataclasses.dataclass
    class _FakeModel:
        model_id: str = 'test-model'

    @dataclasses.dataclass
    class _FakeCtx:
        usage: RunUsage
        run_id: str | None
        retry: int
        usage_limits: UsageLimits | None = None
        tool_call_id: str | None = 'call-1'
        model: Any = dataclasses.field(default_factory=_FakeModel)
        deps: None = None
        conversation_id: str | None = None
        # The hook reads the step the run is on, so a fake context needs one; nothing here may read
        # the history, which the integration tests cover by compacting it away mid-run.
        run_step: int = 0

        def tool_name_for_kind(self, kind: object) -> str | None:
            return None

    ctx = _FakeCtx(usage=RunUsage(), run_id=run_id, retry=retry, usage_limits=usage_limits)
    if model is not None:
        ctx.model = model
    return ctx


def _fixed_model(text: str) -> FunctionModel:
    """A `FunctionModel` whose single text response is `text` (no tool calls)."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(respond)


def _stream_only_summarizer(chunks: list[str] | None = None) -> FunctionModel:
    """A summarizer that only accepts streaming requests (#9249).

    No `function` is provided, so a non-streaming request fails outright -- mirroring an
    endpoint that returns HTTP 400 on `stream:false`.
    """
    parts = chunks if chunks is not None else ['STREAMED ', 'SUMMARY']

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        for part in parts:
            yield part

    return FunctionModel(stream_function=stream_fn)


def _ui_event_agent(tmp_path: Path, metadata: object) -> Agent[None, str]:
    """An agent whose `big_tool` returns spilling output with `metadata` attached."""
    cap: ToolOutputLimits[object] = ToolOutputLimits(
        bands=[Band(over=100, action=Spill())], store=LocalFileOverflowStore(base_dir=tmp_path)
    )
    agent = Agent(TestModel(call_tools=['big_tool']), capabilities=[cap])

    @agent.tool_plain
    def big_tool() -> ToolReturn:
        return ToolReturn(return_value='data line\n' * 500, metadata=metadata)

    return agent


def _big_tool_return(messages: list[ModelMessage]) -> ToolReturnPart:
    return next(p for m in messages for p in m.parts if isinstance(p, ToolReturnPart) and p.tool_name == 'big_tool')


def _call(tool_name: str = 'big_tool', tool_call_id: str = 'call-1') -> ToolCallPart:
    return ToolCallPart(tool_name=tool_name, args='{}', tool_call_id=tool_call_id)


def _tool_def(name: str = 'big_tool') -> ToolDefinition:
    return ToolDefinition(name=name)


def _summary(text: str, chars: int = 100) -> str:
    """A summary as the model sees it, after the marker that says the output was summarized."""
    return f'[summary of {chars:,} chars of output; rerun the original tool for the full output]\n{text}'


async def _run(cap: ToolOutputLimits[object], result: Any, *, ctx: Any = None, tool_name: str = 'big_tool') -> Any:
    return await cap.after_tool_execute(
        ctx if ctx is not None else _make_ctx(),
        call=_call(tool_name),
        tool_def=_tool_def(tool_name),
        args={},
        result=result,
    )


# ---------------------------------------------------------------------------
# _payload helpers
# ---------------------------------------------------------------------------


class TestPayloadHelpers:
    def test_strip_ansi(self):
        assert strip_ansi('\x1b[31mred\x1b[0m') == 'red'

    @pytest.mark.parametrize(
        ('text', 'expected'),
        [
            ('\x1b]0;title\x07body', 'body'),
            ('a\x1b]8;;https://x\x07link\x1b]8;;\x07b', 'alinkb'),
            ('\x1b]0;a\x1b]0;b\x07c', 'c'),
            ('\x1b]unterminated \x1b[1mbold', '\x1b]unterminated bold'),
            ('\x1b]split\nline\x07end', '\x1b]split\nline\x07end'),
            ('\x1b]x\x1b\nnext', '\x1b]xnext'),
            ('\x1bMreverse \x1b(B', 'reverse \x1b(B'),
        ],
    )
    def test_strip_ansi_osc(self, text: str, expected: str):
        assert strip_ansi(text) == expected

    def test_strip_ansi_is_linear_on_unterminated_osc(self):
        """A run of unterminated OSC openers is scanned once, not once per opener.

        A time bound, like `test_a_run_of_header_lines_is_scanned_in_linear_time`'s: `re` reports no step count to
        assert on. On this input, a pattern that rescans to the end for each opener takes far longer.
        """
        text = '\x1b]' * 50_000
        start = time.perf_counter()
        result = strip_ansi(text + '\x1b[1m')

        assert time.perf_counter() - start < 1
        assert result == text

    def test_is_binary(self):
        assert is_binary(b'x') is True
        assert is_binary(bytearray(b'x')) is True
        assert is_binary('x') is False

    def test_to_bytes_variants(self):
        assert to_bytes('hi') == b'hi'
        assert to_bytes(memoryview(b'mv')) == b'mv'
        assert to_bytes(bytearray(b'ba')) == b'ba'
        assert to_bytes({'a': 1}) == b'{"a":1}'

    def test_to_text_variants(self):
        assert to_text('hi') == 'hi'
        assert to_text({'a': 1}) == '{"a":1}'

    def test_values_without_json_form_render_as_repr(self):
        """A `type` leaf must not abort the after-hook, which would lose the tool's output."""
        value = {'kind': int, 'nested': [ValueError]}
        expected = '{"kind":"<class \'int\'>","nested":["<class \'ValueError\'>"]}'
        assert to_text(value) == expected
        assert to_bytes(value) == expected.encode('utf-8')

    def test_indented_json(self):
        assert indented_json({'a': 1}) == '{\n  "a": 1\n}'

    def test_json_lines_list(self):
        assert json_lines([{'a': 1}, {'b': 2}]) == '{"a":1}\n{"b":2}'

    def test_json_lines_non_sequence_falls_back_to_indented(self):
        assert json_lines({'a': 1}) == '{\n  "a": 1\n}'

    def test_json_lines_tuple_is_a_sequence(self):
        assert json_lines(({'a': 1}, {'b': 2})) == '{"a":1}\n{"b":2}'

    def test_json_lines_empty_list(self):
        assert json_lines([]) == ''

    def test_presets_render_bytes_that_are_not_utf8_as_base64(self):
        assert indented_json({'text': b'hi'}) == '{\n  "text": "hi"\n}'
        assert indented_json({'blob': b'\xff', 'text': b'hi'}) == '{\n  "blob": "_w==",\n  "text": "aGk="\n}'
        assert json_lines([{'text': b'hi'}, {'blob': b'\xff'}]) == '{"text":"hi"}\n{"blob":"_w=="}'

    def test_presets_escape_line_separator_chars(self):
        rendered = json_lines([{'text': 'a\N{LINE SEPARATOR}b'}, {'ok': True}])
        assert rendered.splitlines() == ['{"text":"a\\u2028b"}', '{"ok":true}']
        indented = indented_json({'text': 'a\N{PARAGRAPH SEPARATOR}b\x85c'})
        assert indented.splitlines() == ['{', '  "text": "a\\u2029b\\u0085c"', '}']

    def test_measure_chars_and_tokens(self):
        assert measure('x' * 100, over_tokens=False, tokenizer=None) == 100
        assert measure('x' * 100, over_tokens=True, tokenizer=None) == 25
        assert measure('abcd', over_tokens=True, tokenizer=lambda s: len(s)) == 4

    def test_json_sketch_mapping(self):
        assert json_sketch({'a': 1, 'b': 'x'}) == "{'a': int, 'b': str}"

    def test_json_sketch_mapping_truncated(self):
        big = {f'k{i}': i for i in range(12)}
        assert json_sketch(big).endswith('... (12 keys)}')

    def test_json_sketch_sequence(self):
        assert json_sketch([1, 2, 3]) == '[3 items of int]'

    def test_json_sketch_empty_sequence(self):
        assert json_sketch([]) == '[0 items of empty]'

    def test_json_sketch_scalar(self):
        assert json_sketch(42) == ''
        assert json_sketch('plain') == ''


# ---------------------------------------------------------------------------
# Store: write/read, S1 hardening
# ---------------------------------------------------------------------------


class TestStore:
    def test_safe_segment(self):
        assert _safe_segment('a b!@#') == 'a_b_'
        assert _safe_segment('') == '_'
        assert _safe_segment('..') == '_'
        assert _safe_segment('.') == '_'
        assert _safe_segment('ok-1.2') == 'ok-1.2'

    def test_default_root(self):
        store = LocalFileOverflowStore()
        assert store._root.name.startswith('pyai_harness_overflow')  # pyright: ignore[reportPrivateUsage]

    def test_default_root_is_per_user(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(os, 'geteuid', lambda: 1001, raising=False)
        first = LocalFileOverflowStore()._root  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(os, 'geteuid', lambda: 1002, raising=False)
        second = LocalFileOverflowStore()._root  # pyright: ignore[reportPrivateUsage]
        assert first.name == 'pyai_harness_overflow-1001'
        assert second.name == 'pyai_harness_overflow-1002'

    def test_default_root_without_uid(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delattr(os, 'geteuid', raising=False)
        assert LocalFileOverflowStore()._root.name == 'pyai_harness_overflow'  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.skipif(not hasattr(os, 'geteuid'), reason='POSIX ownership check')
    async def test_new_root_created_0700_regardless_of_umask(self, tmp_path: Path):
        root = tmp_path / 'store'
        store = LocalFileOverflowStore(base_dir=root)
        old_umask = os.umask(0)
        try:
            with patch.object(Path, 'chmod', side_effect=AssertionError('root was not created 0700')):
                await store.write('run/c.0', b'x')
        finally:
            os.umask(old_umask)
        assert oct(root.stat().st_mode & 0o777) == '0o700'

    @pytest.mark.skipif(not hasattr(os, 'geteuid'), reason='POSIX ownership check')
    async def test_preexisting_open_root_is_tightened(self, tmp_path: Path):
        root = tmp_path / 'store'
        root.mkdir()
        root.chmod(0o777)
        store = LocalFileOverflowStore(base_dir=root)
        await store.write('run/c.0', b'x')
        assert oct(root.stat().st_mode & 0o777) == '0o700'

    @pytest.mark.skipif(not hasattr(os, 'geteuid'), reason='POSIX ownership check')
    async def test_root_owned_by_another_user_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        root = tmp_path / 'store'
        root.mkdir()
        root.chmod(0o777)
        other_uid = root.stat().st_uid + 1
        monkeypatch.setattr(os, 'geteuid', lambda: other_uid)
        store = LocalFileOverflowStore(base_dir=root)
        with pytest.raises(PermissionError, match='not owned by the current user'):
            await store.write('run/c.0', b'secret')
        assert list(root.iterdir()) == []

    async def test_symlink_root_is_refused_for_write_and_read(self, tmp_path: Path):
        target = tmp_path / 'target'
        target.mkdir()
        (target / 'secret').write_bytes(b'secret')
        root = tmp_path / 'store'
        root.symlink_to(target, target_is_directory=True)
        store = LocalFileOverflowStore(base_dir=root)

        with pytest.raises(PermissionError, match='is a symbolic link'):
            await store.write('run/c.0', b'spilled')
        with pytest.raises(PermissionError, match='is a symbolic link'):
            await store.read('secret')
        assert list(target.iterdir()) == [target / 'secret']

    async def test_root_check_skipped_without_uid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delattr(os, 'geteuid', raising=False)
        store = LocalFileOverflowStore(base_dir=tmp_path / 'store')
        handle = await store.write('run/c.0', b'x')
        assert await store.read(handle) == b'x'

    async def test_write_read_roundtrip(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path / 'store')
        handle = await store.write('run-1/call-1.0', b'payload')
        assert handle == 'run-1/call-1.0'
        assert await store.read(handle) == b'payload'

    async def test_root_created_0700(self, tmp_path: Path):
        root = tmp_path / 'store'
        store = LocalFileOverflowStore(base_dir=root)
        await store.write('run/c.0', b'x')
        assert oct(root.stat().st_mode & 0o777) == '0o700'

    async def test_empty_key(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path / 'store')
        handle = await store.write('', b'data')
        assert await store.read(handle) == b'data'

    async def test_read_missing_raises(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path / 'store')
        with pytest.raises(OSError):
            await store.read('nope/x.0')

    async def test_dotdot_handle_stays_in_root(self, tmp_path: Path):
        # `_safe_segment` neutralizes `..`, so the read resolves inside the root and 404s
        # rather than escaping.
        store = LocalFileOverflowStore(base_dir=tmp_path / 'store')
        await store.write('run/c.0', b'inside')
        with pytest.raises(OSError):
            await store.read('../c.0')

    async def test_symlink_escape_rejected(self, tmp_path: Path):
        secret = tmp_path / 'secret.txt'
        secret.write_bytes(b'top secret')
        root = tmp_path / 'store'
        store = LocalFileOverflowStore(base_dir=root)
        await store.write('run/c.0', b'inside')  # creates the root
        (root / 'evil').symlink_to(secret)
        with pytest.raises(PermissionError, match='outside the store root'):
            await store.read('evil')

    @pytest.mark.parametrize('link', ['file', 'directory'])
    async def test_symlink_escape_rejected_for_write(self, tmp_path: Path, link: str):
        outside = tmp_path / 'outside'
        outside.mkdir()
        root = tmp_path / 'store'
        store = LocalFileOverflowStore(base_dir=root)
        await store.write('run/c.0', b'inside')  # creates the root
        if link == 'file':
            (root / 'run' / 'evil.0').symlink_to(outside / 'planted')
            key = 'run/evil.0'
        else:
            (root / 'evil').symlink_to(outside, target_is_directory=True)
            key = 'evil/sub/c.0'
        with pytest.raises(PermissionError, match='outside the store root'):
            await store.write(key, b'spilled')
        assert list(outside.iterdir()) == []


# ---------------------------------------------------------------------------
# Store: opt-in TTL cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_prune_removes_old_keeps_new(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path, cleanup_after=timedelta(seconds=1))
        old = tmp_path / 'old.bin'
        old.write_bytes(b'x')
        new = tmp_path / 'new.bin'
        new.write_bytes(b'y')
        (tmp_path / 'sub').mkdir()  # a directory rglob yields -- must be skipped
        past = time.time() - 100
        os.utime(old, (past, past))

        store._prune_sync()  # pyright: ignore[reportPrivateUsage]

        assert not old.exists()
        assert new.exists()

    def test_prune_refuses_symlink_root(self, tmp_path: Path):
        target = tmp_path / 'target'
        target.mkdir()
        old = target / 'old.bin'
        old.write_bytes(b'x')
        past = time.time() - 100
        os.utime(old, (past, past))
        root = tmp_path / 'store'
        root.symlink_to(target, target_is_directory=True)
        store = LocalFileOverflowStore(base_dir=root, cleanup_after=timedelta(seconds=1))

        with pytest.raises(PermissionError, match='is a symbolic link'):
            store._prune_sync()  # pyright: ignore[reportPrivateUsage]

        assert old.exists()

    def test_run_prune_swallows_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        store = LocalFileOverflowStore(base_dir=tmp_path, cleanup_after=timedelta(seconds=1))

        def boom() -> None:
            raise OSError('disk gone')

        monkeypatch.setattr(store, '_prune_sync', boom)
        with pytest.warns(UserWarning, match='cleanup failed'):
            store._run_prune()  # pyright: ignore[reportPrivateUsage]

    def test_schedule_none_when_disabled(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        assert store._schedule_cleanup() is None  # pyright: ignore[reportPrivateUsage]

    def test_schedule_starts_thread(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path, cleanup_after=timedelta(seconds=1))
        (tmp_path / 'f.bin').write_bytes(b'z')
        thread = store._schedule_cleanup()  # pyright: ignore[reportPrivateUsage]
        assert thread is not None
        thread.join(timeout=5)
        assert not thread.is_alive()

    async def test_write_schedules_cleanup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        store = LocalFileOverflowStore(base_dir=tmp_path / 'store', cleanup_after=timedelta(seconds=1))
        scheduled: list[int] = []
        monkeypatch.setattr(store, '_schedule_cleanup', lambda: scheduled.append(1))
        await store.write('run/c.0', b'data')
        assert scheduled == [1]


# ---------------------------------------------------------------------------
# Capability construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_band_is_spill_then_truncate(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits()
        assert len(cap._bands) == 1  # pyright: ignore[reportPrivateUsage]
        action = cap._bands[0].action  # pyright: ignore[reportPrivateUsage]
        assert isinstance(action, Spill)
        assert isinstance(action.then, Truncate)

    def test_bands_sorted_descending(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=10, action=Truncate()), Band(over=100, action=Spill())]
        )
        assert [b.over for b in cap._bands] == [100, 10]  # pyright: ignore[reportPrivateUsage]

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match='non-negative'):
            ToolOutputLimits(bands=[Band(over=-1, action=Passthrough())])

    def test_provided_store_used(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(store=store)
        assert cap._store is store  # pyright: ignore[reportPrivateUsage]

    def test_per_tool_prepared(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(per_tool={'read_file': [Band(over=5, action=Truncate())]})
        assert 'read_file' in cap._per_tool  # pyright: ignore[reportPrivateUsage]


class TestCombine:
    async def test_merged_bands_run_largest_threshold_first(self):
        merged = ToolOutputLimits[object].combine(
            [
                ToolOutputLimits(bands=[Band(over=100, action=Truncate())]),
                ToolOutputLimits(bands=[Band(over=1000, action=Summarize(model=_fixed_model('summary')))]),
            ]
        )
        assert isinstance(merged, ToolOutputLimits)
        assert [b.over for b in merged._bands] == [1000, 100]  # pyright: ignore[reportPrivateUsage]
        assert await _run(merged, 'x' * 1500) == _summary('summary', 1_500)

    def test_later_band_wins_a_shared_threshold(self):
        later = Truncate(max_chars=20)
        merged = ToolOutputLimits[object].combine(
            [
                ToolOutputLimits(bands=[Band(over=100, action=Truncate(max_chars=10)), Band(over=500, action=Spill())]),
                ToolOutputLimits(bands=[Band(over=100, action=later)]),
            ]
        )
        assert isinstance(merged, ToolOutputLimits)
        assert [(b.over, b.action) for b in merged._bands] == [(500, Spill()), (100, later)]  # pyright: ignore[reportPrivateUsage]
        assert _select_action(merged._bands, 150) is later  # pyright: ignore[reportPrivateUsage]

    def test_stated_store_survives_the_merge(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        merged = ToolOutputLimits[object].combine([ToolOutputLimits(store=store), ToolOutputLimits()])
        assert isinstance(merged, ToolOutputLimits)
        assert merged._store is store  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Passthrough / filtering / guards
# ---------------------------------------------------------------------------


class TestPassthrough:
    async def test_read_tool_exempt_by_kind(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=1, action=Truncate(max_chars=2))])
        out = await cap.after_tool_execute(
            _make_ctx(),
            call=_call('renamed_reader'),
            tool_def=ToolDefinition(name='renamed_reader', tool_kind=READ_TOOL_RESULT_KIND),
            args={},
            result='x' * 100,
        )
        assert out == 'x' * 100

    async def test_read_tool_name_alone_is_not_exempt(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=1, action=Truncate(max_chars=2))])
        out = await _run(cap, 'x' * 100, tool_name=READ_TOOL_NAME)
        assert out != 'x' * 100

    async def test_tool_filter_skips_unmatched(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=1, action=Truncate(max_chars=2))], tool_filter=['other']
        )
        out = await _run(cap, 'x' * 100)
        assert out == 'x' * 100

    async def test_callable_filter(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=1, action=Truncate(max_chars=150))],
            tool_filter=lambda ctx, td: td.name == 'big_tool',
        )
        out = await _run(cap, 'x' * 300)
        assert isinstance(out, str) and 'truncated' in out

    async def test_below_threshold_passthrough(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=1000, action=Truncate())])
        out = await _run(cap, 'small')
        assert out == 'small'

    async def test_below_threshold_structured_passthrough_with_serializer(self):
        records = [{'record_id': 1}]
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=1000, action=Truncate())], serializer=indented_json
        )
        assert await _run(cap, records) is records

    async def test_exception_result_passthrough(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=1, action=Truncate(max_chars=2))])
        err = ValueError('boom')
        assert await _run(cap, err) is err


# ---------------------------------------------------------------------------
# Truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    @pytest.mark.parametrize('strategy', TruncationStrategy)
    @pytest.mark.parametrize(
        'total,kept',
        [(1000, 1), (1000, 9), (1000, 10), (1000, 99), (1000, 100), (10000, 999), (10000, 1000), (1001, 2)],
    )
    async def test_marker_counts_toward_budget(self, strategy: TruncationStrategy, total: int, kept: int):
        text = ''.join(chr(0x4E00 + i) for i in range(total))
        if strategy is TruncationStrategy.head:
            expected = (
                text[:kept]
                + f'\n\n[truncated: showing first {kept:,} of {total:,} chars; rerun the original tool for the full output]'
            )
        elif strategy is TruncationStrategy.tail:
            expected = (
                f'[... output truncated, showing last {kept} chars; rerun the original tool for the full output]\n'
                + text[-kept:]
            )
        else:
            head = kept * 2 // 5
            tail = kept - head
            expected = (
                f'{text[:head]}\n\n[truncated: {total - kept:,} chars omitted from the middle; '
                f'showing first {head:,} + last {tail:,} of {total:,} chars; rerun the original tool for the full output]\n\n{text[-tail:]}'
            )
        max_chars = len(expected)
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=0, action=Truncate(max_chars=max_chars, strategy=strategy))]
        )
        agent = Agent(TestModel(call_tools=['big_tool']), capabilities=[cap])

        @agent.tool_plain
        def big_tool() -> str:
            return text

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 1
        assert returns[0].content == expected

    @pytest.mark.parametrize(
        'strategy,max_chars',
        [(strategy, cap) for strategy in TruncationStrategy for cap in [-1, 0, 1, 2, 10]]
        + [(TruncationStrategy.head, 45), (TruncationStrategy.tail, 45), (TruncationStrategy.head_tail, 91)],
    )
    async def test_budget_too_small_for_marker(self, strategy: TruncationStrategy, max_chars: int):
        text = ''.join(chr(0x4E00 + i) for i in range(1000))
        kept = max(0, max_chars)
        if strategy is TruncationStrategy.head:
            expected = text[:kept]
        elif strategy is TruncationStrategy.tail:
            expected = text[len(text) - kept :]
        else:
            head = kept * 2 // 5
            expected = text[:head] + text[len(text) - (kept - head) :]
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=0, action=Truncate(max_chars=max_chars, strategy=strategy))]
        )
        agent = Agent(TestModel(call_tools=['big_tool']), capabilities=[cap])

        @agent.tool_plain
        def big_tool() -> str:
            return text

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 1
        assert returns[0].content == expected

    @pytest.mark.parametrize('strategy', TruncationStrategy)
    @pytest.mark.parametrize('max_chars', [5, 6])
    async def test_output_that_fits_is_unchanged(self, strategy: TruncationStrategy, max_chars: int):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=0, action=Truncate(max_chars=max_chars, strategy=strategy))]
        )
        agent = Agent(TestModel(call_tools=['small_tool']), capabilities=[cap])

        @agent.tool_plain
        def small_tool() -> str:
            return 'short'

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 1
        assert returns[0].content == 'short'

    @pytest.mark.parametrize('strategy', TruncationStrategy)
    async def test_tool_return_value_and_content_are_bounded(self, strategy: TruncationStrategy):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=0, action=Truncate(max_chars=200, strategy=strategy))]
        )
        agent = Agent(TestModel(call_tools=['big_tool']), capabilities=[cap])

        @agent.tool_plain
        def big_tool() -> ToolReturn:
            return ToolReturn(return_value='v' * 1000, content='c' * 2000, metadata={'k': 1})

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 1
        assert returns[0].metadata == {'k': 1}
        value = returns[0].content
        assert isinstance(value, str) and len(value) <= 200 and 'truncated' in value
        prompts = [p for m in result.all_messages() for p in m.parts if isinstance(p, UserPromptPart)]
        assert len(prompts) == 2
        content = prompts[-1].content
        assert isinstance(content, str) and len(content) <= 200 and 'truncated' in content

    async def test_strip_ansi_applied(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=5, action=Truncate(max_chars=1000))], strip_ansi=True
        )
        out = await _run(cap, '\x1b[31m' + 'red text ' * 10 + '\x1b[0m')
        assert isinstance(out, str) and '\x1b[' not in out

    async def test_binary_truncate_falls_back_to_passthrough(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=1, action=Truncate())])
        data = b'\x00\x01' * 100
        assert await _run(cap, data) == data

    async def test_tool_return_envelope_preserved(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Truncate(max_chars=20))])
        out = await _run(cap, ToolReturn(return_value='a' * 100, content='note', metadata={'k': 1}))
        assert isinstance(out, ToolReturn)
        assert out.content == 'note'
        assert out.metadata == {'k': 1}

    async def test_dense_output_obeys_token_budget(self):
        """over_tokens=True prices this 348-character base64 payload at 278 tokens, and 4 characters per token gave 87.

        Not a VCR test: the estimate is a pure function of the text, so there is no request to record.
        """
        payload = base64.b64encode(b'binary blob 0123 4567 89ab' * 10).decode()

        def limits(over: int) -> ToolOutputLimits[object]:
            return ToolOutputLimits(bands=[Band(over=over, action=Truncate(max_chars=50))], over_tokens=True)

        assert await _run(limits(300), payload) == payload
        out = await _run(limits(200), payload)
        assert isinstance(out, str) and len(out) == 50 and out == payload[:20] + payload[-30:]


# ---------------------------------------------------------------------------
# Spill
# ---------------------------------------------------------------------------


class TestSpill:
    async def test_spill_roundtrip(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=10, action=Spill(preview_chars=20))], store=store
        )
        text = 'line\n' * 1000
        out = await _run(cap, text)
        assert isinstance(out, ToolReturn)
        assert isinstance(out.return_value, str) and 'too large' in out.return_value
        handle = out.metadata['overflow_handle']
        assert handle == 'run-1/call-1.0'
        assert await store.read(handle) == text.encode('utf-8')

    async def test_spill_binary_verbatim(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=1, action=Spill())], store=store)
        data = b'\x00\xff' * 100
        out = await _run(cap, data)
        assert isinstance(out, ToolReturn)
        assert 'binary' in out.return_value  # type: ignore[operator]
        assert await store.read(out.metadata['overflow_handle']) == data

    async def test_spill_structured_includes_sketch(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=5, action=Spill())], store=store)
        out = await _run(cap, {'rows': list(range(1000)), 'ok': True})
        assert isinstance(out, ToolReturn)
        assert 'shape:' in out.return_value  # type: ignore[operator]

    async def test_spill_serializer_json_lines_is_pageable(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=5, action=Spill(preview_chars=80))], serializer=json_lines, store=store
        )
        records = [{'record_id': record_id, 'item': f'item-{record_id}'} for record_id in range(10)]
        out = await _run(cap, records)
        assert isinstance(out, ToolReturn)
        handle = out.metadata['overflow_handle']
        assert await store.read(handle) == json_lines(records).encode('utf-8')
        toolset = cap.get_toolset()
        assert toolset is not None
        read = toolset.tools[READ_TOOL_NAME].function  # type: ignore[union-attr]
        page_1 = await read(_make_ctx(), handle, offset=0, limit=1)  # type: ignore[attr-defined]
        page_2 = await read(_make_ctx(), handle, offset=1, limit=1)  # type: ignore[attr-defined]
        assert '"record_id":0' in page_1 and '"record_id":1' not in page_1
        assert '"record_id":1' in page_2 and '"record_id":0' not in page_2

    async def test_spill_serializer_inside_tool_return_envelope(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=5, action=Spill())], serializer=json_lines, store=store
        )
        out = await _run(cap, ToolReturn(return_value=[{'record_id': 1}, {'record_id': 2}], metadata={'orig': True}))
        assert isinstance(out, ToolReturn)
        assert out.metadata['orig'] is True
        assert await store.read(out.metadata['overflow_handle']) == b'{"record_id":1}\n{"record_id":2}'

    async def test_serializer_layout_reaches_band_compact_would_not(self):
        value = {'a': 1, 'b': 2, 'c': 3}
        bands = [Band(over=25, action=Truncate(max_chars=10, strategy=TruncationStrategy.head))]
        plain: ToolOutputLimits[object] = ToolOutputLimits(bands=bands)
        assert await _run(plain, value) is value
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=bands, serializer=indented_json)
        out = await _run(cap, value)
        assert isinstance(out, str) and out.startswith('{\n  "a": 1')

    async def test_serializer_error_warns_and_falls_back_to_compact(self):
        def broken(value: object) -> str:
            raise RuntimeError('boom')

        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=10, action=Truncate(max_chars=20, strategy=TruncationStrategy.head))],
            serializer=broken,
        )
        with pytest.warns(UserWarning, match='serializer raised'):
            out = await _run(cap, {'rows': list(range(100))})
        assert isinstance(out, str) and out.startswith('{"rows":[0,1,')

    async def test_serializer_non_text_return_warns_and_falls_back_to_compact(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=10, action=Truncate(max_chars=20, strategy=TruncationStrategy.head))],
            serializer=lambda value: to_json(value, indent=2),  # type: ignore[arg-type,return-value]
        )
        with pytest.warns(UserWarning, match='non-text'):
            out = await _run(cap, {'rows': list(range(100))})
        assert isinstance(out, str) and out.startswith('{"rows":[0,1,')

    async def test_spill_serializer_skips_strings(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=5, action=Spill())], serializer=indented_json, store=store
        )
        text = 'line\n' * 100
        out = await _run(cap, text)
        assert isinstance(out, ToolReturn)
        assert await store.read(out.metadata['overflow_handle']) == text.encode('utf-8')

    async def test_spill_failure_falls_back_to_truncate(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=10, action=Spill(then=Truncate(max_chars=150)))], store=_BrokenStore()
        )
        out = await _run(cap, 'a' * 300)
        assert isinstance(out, str) and 'truncated' in out

    async def test_spill_failure_logs_warning(self, caplog: pytest.LogCaptureFixture):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=10, action=Spill(then=Truncate(max_chars=150)))], store=_BrokenStore()
        )
        with caplog.at_level('WARNING', logger='pydantic_ai_harness.tool_output_limits'):
            out = await _run(cap, 'a' * 300)
        assert isinstance(out, str) and 'truncated' in out
        [record] = caplog.records
        assert record.levelname == 'WARNING'
        assert 'OSError' in record.getMessage()
        assert 'big_tool' in record.getMessage()
        assert record.exc_info is not None

    async def test_spill_failure_no_fallback_returns_original(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Spill())], store=_BrokenStore())
        out = await _run(cap, 'a' * 100)
        assert out == 'a' * 100

    async def test_handle_distinct_per_retry(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=5, action=Spill())], store=store)
        out0 = await _run(cap, 'a' * 100, ctx=_make_ctx(retry=0))
        out1 = await _run(cap, 'b' * 100, ctx=_make_ctx(retry=1))
        assert out0.metadata['overflow_handle'] != out1.metadata['overflow_handle']

    async def test_spill_merges_existing_metadata(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=5, action=Spill())], store=store)
        out = await _run(cap, ToolReturn(return_value='a' * 100, metadata={'orig': True}))
        assert isinstance(out, ToolReturn)
        assert out.metadata['orig'] is True
        assert 'overflow_handle' in out.metadata

    async def test_spill_preserves_non_mapping_metadata(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=5, action=Spill())], store=store)
        out = await _run(cap, ToolReturn(return_value='a' * 100, metadata='app-request-id-123'))
        assert isinstance(out, ToolReturn)
        assert out.metadata['original_metadata'] == 'app-request-id-123'
        assert 'overflow_handle' in out.metadata


class _BrokenStore:
    """An `OverflowStore` whose writes always fail (for fallback tests)."""

    async def write(self, key: str, data: bytes) -> str:
        raise OSError('disk full')

    async def read(self, handle: str) -> bytes:  # pragma: no cover - never reached
        raise FileNotFoundError(handle)


# ---------------------------------------------------------------------------
# C1: model-visible ToolReturn.content is reduced too
# ---------------------------------------------------------------------------


class TestFilesInReturnValue:
    image = BinaryContent(data=b'\x89PNG\r\n\x1a\n' + b'x' * 100_000, media_type='image/png')

    async def test_file_alone_passes_through(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Truncate(max_chars=150))])
        assert await _run(cap, self.image) is self.image

    async def test_short_text_beside_a_file_passes_through(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10_000, action=Truncate(max_chars=150))])
        value = ['header', self.image]
        assert await _run(cap, value) is value

    async def test_data_beside_files_is_reduced_and_the_files_kept(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Truncate(max_chars=150))])
        assert await _run(cap, ['A' * 300, self.image, 'B' * 300]) == [
            snapshot("""\
["AAAA

[truncated: 592 chars omitted from the middle; showing first 6 + last 9 of 607 chars; rerun the original tool for the full output]

BBBBBBB"]\
"""),
            self.image,
        ]

    async def test_data_beside_a_file_spills(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Spill())], store=store)
        assert await _run(cap, ToolReturn(return_value=['A' * 300, self.image], metadata={'k': 'v'})) == ToolReturn(
            return_value=[
                snapshot("""\
[Tool output too large (300 chars); stored to handle 'run-1/call-1.0'. Read it with read_tool_result(handle='run-1/call-1.0', offset=0, limit=200, from_end=False, pattern=None).]
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\
"""),
                self.image,
            ],
            metadata=snapshot({'k': 'v', 'overflow_handle': 'run-1/call-1.0', 'overflow_bytes': 300}),
        )
        assert await store.read('run-1/call-1.0') == ('A' * 300).encode('utf-8')

    async def test_structured_data_beside_a_file_is_reduced_and_the_file_kept(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Truncate(max_chars=150))])
        assert await _run(cap, [{'rows': list(range(1_000))}, self.image, b'raw bytes']) == [
            snapshot("""\
[{"r

[truncated: 3,902 chars omitted from the middle; showing first 4 + last 8 of 3,914 chars; rerun the original tool for the full output]

 bytes"]\
"""),
            self.image,
        ]

    async def test_raw_bytes_beside_a_file_are_reduced_as_text(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Truncate(max_chars=150))])
        assert await _run(cap, [b'R' * 5_000, self.image]) == [
            snapshot("""\
"RRR

[truncated: 4,990 chars omitted from the middle; showing first 4 + last 8 of 5,002 chars; rerun the original tool for the full output]

RRRRRRR"\
"""),
            self.image,
        ]

    async def test_content_beside_a_file_is_still_reduced(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Truncate(max_chars=150))])
        out = await _run(cap, ToolReturn(return_value=['image', self.image], content='C' * 5_000))
        assert out == ToolReturn(
            return_value=['image', self.image],
            content=snapshot("""\
CCCC

[truncated: 4,988 chars omitted from the middle; showing first 4 + last 8 of 5,000 chars; rerun the original tool for the full output]

CCCCCCCC\
"""),
        )


class TestContentReduction:
    async def test_large_content_spilled(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=100, action=Spill(preview_chars=20))], store=store
        )
        out = await _run(cap, ToolReturn(return_value='small', content='C' * 5000))
        assert isinstance(out, ToolReturn)
        assert out.return_value == 'small'  # small return_value untouched
        assert isinstance(out.content, str) and 'too large' in out.content
        handle = out.metadata['overflow_content_handle']
        assert await store.read(handle) == ('C' * 5000).encode('utf-8')

    async def test_large_content_truncated(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Truncate(max_chars=150))])
        out = await _run(cap, ToolReturn(return_value='small', content='C' * 400))
        assert isinstance(out, ToolReturn)
        assert isinstance(out.content, str) and 'truncated' in out.content

    async def test_both_value_and_content_reduced(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=50, action=Spill())], store=store)
        out = await _run(cap, ToolReturn(return_value='v' * 500, content='c' * 500))
        assert isinstance(out, ToolReturn)
        assert out.metadata['overflow_handle'] != out.metadata['overflow_content_handle']

    async def test_nontext_content_warns_and_passes_through(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Truncate())])
        content = ['x' * 5000, BinaryContent(data=b'\x00', media_type='application/octet-stream')]
        original = ToolReturn(return_value='small', content=content)
        with pytest.warns(UserWarning, match='non-text content'):
            out = await _run(cap, original)
        assert out is original

    async def test_nontext_content_passthrough_action_no_warn(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=1, action=Passthrough())])
        content = ['x' * 5000, BinaryContent(data=b'\x00', media_type='application/octet-stream')]
        original = ToolReturn(return_value='small', content=content)
        out = await _run(cap, original)  # Passthrough action -> no warning, returned unchanged
        assert out is original

    async def test_small_nontext_content_no_warn(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10_000, action=Truncate())])
        content = ['tiny', BinaryContent(data=b'\x00', media_type='application/octet-stream')]
        original = ToolReturn(return_value='small', content=content)
        out = await _run(cap, original)
        assert out is original


# ---------------------------------------------------------------------------
# Summarize (M1: assert model + usage, not just a wholesale mock)
# ---------------------------------------------------------------------------


class TestSummarize:
    async def test_mutating_source_bands_does_not_desync_summarizer_resolution(self):
        bands = [Band(over=5, action=Summarize(model=_fixed_model('THE SUMMARY')))]
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=bands)
        bands.clear()

        out = await _run(cap, 'x' * 100)

        assert out == _summary('THE SUMMARY')

    @pytest.mark.usefixtures('instrument_all_agents')
    async def test_summarizer_run_is_named_after_the_capability(self, capfire: CaptureLogfire):
        def large_output() -> str:
            return 'x' * 100

        agent = Agent(
            TestModel(call_tools='all'),
            name='outer',
            capabilities=[ToolOutputLimits(bands=[Band(over=5, action=Summarize(model=_fixed_model('THE SUMMARY')))])],
            toolsets=[FunctionToolset(tools=[large_output], id='large-output')],
        )
        await agent.run('call the tool')

        assert 'tool_output_limits' in agent_run_names(capfire)

    async def test_summarizer_run_belongs_to_the_tool_caller_conversation(self):
        summary_conversations: set[str | None] = set()

        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            summary_conversations.update(m.conversation_id for m in messages)
            return ModelResponse(parts=[TextPart(content='THE SUMMARY')])

        def large_output() -> str:
            return 'x' * 100

        agent = Agent(
            TestModel(call_tools='all'),
            capabilities=[ToolOutputLimits(bands=[Band(over=5, action=Summarize(model=FunctionModel(summarize)))])],
            toolsets=[FunctionToolset(tools=[large_output], id='large-output')],
        )
        await agent.run('call the tool', conversation_id='conversation-1')

        assert summary_conversations == {'conversation-1'}

    async def test_model_summarizer_dispatches_as_durable_operation(self):
        def large_output() -> str:
            return 'x' * 100

        durability = RecordingDurability()
        cap: ToolOutputLimits[Any] = ToolOutputLimits(
            bands=[Band(over=5, action=Summarize(model=_fixed_model('THE SUMMARY')))]
        )
        assert cap.id == 'tool_output_limits'

        agent = Agent(
            TestModel(call_tools='all'),
            name='tool_output_limits',
            capabilities=[cap, durability],
            toolsets=[FunctionToolset(tools=[large_output], id='large-output')],
        )
        await agent.run('call the tool')

        bound = RecordingDurability.from_agent(agent)
        assert bound is not None
        assert 'tool_output_limits__capability__tool_output_limits.summarize' in {name for name, _ in bound.calls}

    async def test_custom_summarizer_bypasses_durable_operation(self):
        def large_output() -> str:
            return 'x' * 100

        durability = RecordingDurability()
        cap: ToolOutputLimits[Any] = ToolOutputLimits(
            id='tool_output_limits', bands=[Band(over=5, action=Summarize(summarize=lambda _name, _text: 'summary'))]
        )

        agent = Agent(
            TestModel(call_tools='all'),
            name='custom_tool_output_limits',
            capabilities=[cap, durability],
            toolsets=[FunctionToolset(tools=[large_output], id='custom-large-output')],
        )
        await agent.run('call the tool')

        bound = RecordingDurability.from_agent(agent)
        assert bound is not None
        assert 'custom_tool_output_limits__capability__tool_output_limits.summarize' not in {
            name for name, _ in bound.calls
        }

    async def test_custom_sync_summarizer(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=5, action=Summarize(summarize=lambda name, text: f'{name}:{len(text)}'))]
        )
        out = await _run(cap, 'x' * 100)
        assert out == _summary('big_tool:100')

    async def test_custom_async_summarizer(self):
        async def summ(name: str, text: str) -> str:
            return f'async:{len(text)}'

        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=5, action=Summarize(summarize=summ))])
        out = await _run(cap, 'x' * 100)
        assert out == _summary('async:100')

    async def test_inherited_model_and_usage(self):
        # model=None resolves to ctx.model, and the call threads usage=ctx.usage.
        ctx = _make_ctx(model=_fixed_model('THE SUMMARY'))
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=5, action=Summarize())])
        out = await _run(cap, 'x' * 100, ctx=ctx)
        assert out == _summary('THE SUMMARY')
        assert ctx.usage.requests == 1

    async def test_inherited_model_reserves_parent_usage_limits(self):
        ctx = _make_ctx(
            model=_fixed_model('THE SUMMARY'), usage_limits=UsageLimits(request_limit=5, tool_calls_limit=2)
        )
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=5, action=Summarize())])
        mock_result = AsyncMock()
        mock_result.output = 'THE SUMMARY'
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch('pydantic_ai.Agent', return_value=mock_agent):
            out = await _run(cap, 'x' * 100, ctx=ctx)

        assert out == _summary('THE SUMMARY')
        assert mock_agent.run.call_args.kwargs['usage_limits'] == UsageLimits(request_limit=4, tool_calls_limit=2)
        assert mock_agent.run.call_args.kwargs['event_stream_handler'] is None

    async def test_explicit_model_overrides_ctx(self):
        ctx = _make_ctx(model=_fixed_model('FROM CTX MODEL'))
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=5, action=Summarize(model=_fixed_model('FROM EXPLICIT MODEL')))]
        )
        out = await _run(cap, 'x' * 100, ctx=ctx)
        assert out == _summary('FROM EXPLICIT MODEL')
        assert ctx.usage.requests == 1

    async def test_explicit_model_name_overrides_ctx(self):
        ctx = _make_ctx(model=_fixed_model('FROM CTX MODEL'))
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=5, action=Summarize(model='test:summary'))])
        mock_result = AsyncMock(output='FROM NAMED MODEL')
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch('pydantic_ai.Agent', return_value=mock_agent) as agent_type:
            assert await _run(cap, 'x' * 100, ctx=ctx) == _summary('FROM NAMED MODEL')

        agent_type.assert_called_once_with(
            'test:summary', name='tool_output_limits', instructions='You summarize oversized tool output.'
        )

    async def test_realtime_run_without_a_model_raises(self):
        """A realtime run has no request-response model to summarize with; ask for one (#585)."""

        class _RealtimeModel(AbstractModel):
            @property
            def model_name(self) -> str:
                return 'gpt-4o-realtime-preview'

            @property
            def system(self) -> str:
                return 'openai'

        model = _RealtimeModel()
        # A genuine `AbstractModel` identity, not a mock: it is rejected for being a non-`Model`.
        assert model.model_id == 'openai:gpt-4o-realtime-preview'
        ctx = _make_ctx(model=model)
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=5, action=Summarize())])

        with pytest.raises(UserError, match='needs a request-response model'):
            await _run(cap, 'x' * 100, ctx=ctx)

    async def test_binary_summarize_falls_back(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=1, action=Summarize(then=Passthrough()))])
        data = b'\x00' * 100
        assert await _run(cap, data) == data

    async def test_summarize_failure_falls_back(self):
        def boom(name: str, text: str) -> str:
            raise RuntimeError('model down')

        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=5, action=Summarize(summarize=boom, then=Truncate(max_chars=150)))]
        )
        out = await _run(cap, 'a' * 300)
        assert isinstance(out, str) and 'truncated' in out

    async def test_summarize_failure_logs_warning(self, caplog: pytest.LogCaptureFixture):
        def boom(name: str, text: str) -> str:
            raise RuntimeError('model down')

        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=5, action=Summarize(summarize=boom, then=Truncate(max_chars=150)))]
        )
        with caplog.at_level('WARNING', logger='pydantic_ai_harness.tool_output_limits'):
            await _run(cap, 'a' * 300)
        [record] = caplog.records
        assert 'RuntimeError' in record.getMessage()
        assert 'big_tool' in record.getMessage()
        assert record.exc_info is not None

    async def test_stream_only_summarizer_without_a_handler_falls_back(self):
        # No handler means the nested run takes the non-streaming request path, which this
        # summarizer rejects; the failure degrades to the `then` fallback (#9249).
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=5, action=Summarize(model=_stream_only_summarizer(), then=Truncate(max_chars=95)))]
        )
        out = await _run(cap, 'x' * 100)
        assert isinstance(out, str) and 'truncated' in out

    async def test_stream_only_summarizer_completes_with_drain_handler(self):
        # `drain_summary_events` selects the streaming request path and discards the events,
        # which is what a summarizer endpoint that rejects non-streaming requests needs.
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[
                Band(
                    over=5,
                    action=Summarize(model=_stream_only_summarizer(), event_stream_handler=drain_summary_events),
                )
            ]
        )
        out = await _run(cap, 'x' * 100)
        assert out == 'STREAMED SUMMARY'

    async def test_summary_events_reach_a_caller_supplied_handler(self):
        deltas: list[str] = []

        async def collect(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
            # The opening chunk of a part arrives as `PartStartEvent`; the rest are deltas.
            async for event in events:
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    deltas.append(event.part.content)
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    deltas.append(event.delta.content_delta)

        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[
                Band(
                    over=5,
                    action=Summarize(
                        model=_stream_only_summarizer(['STREA', 'MED SU', 'MMARY']),
                        event_stream_handler=collect,
                    ),
                )
            ]
        )
        out = await _run(cap, 'x' * 100)
        assert out == 'STREAMED SUMMARY'
        assert ''.join(deltas) == 'STREAMED SUMMARY'

    async def test_stream_only_summarizer_completes_the_parent_run(self):
        # Through the public surface: an oversized tool return is summarized over the
        # streaming path inside a full agent run (#9249).
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[
                Band(
                    over=100,
                    action=Summarize(model=_stream_only_summarizer(), event_stream_handler=drain_summary_events),
                )
            ]
        )
        agent = Agent(TestModel(), capabilities=[cap])

        @agent.tool_plain
        def read_log() -> str:
            return 'log line\n' * 100

        result = await agent.run('go')
        returns = [
            part.content
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'read_log'
        ]
        assert returns == ['STREAMED SUMMARY']

    async def test_nested_per_tool_model_summarizer(self):
        summarize = Summarize(model=_fixed_model('NESTED SUMMARY'))
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=5, action=Passthrough())],
            per_tool={'big_tool': [Band(over=5, action=Spill(then=summarize))]},
            store=_BrokenStore(),
        )

        assert await _run(cap, 'x' * 100) == _summary('NESTED SUMMARY')


# ---------------------------------------------------------------------------
# Passthrough action + per-tool + band selection
# ---------------------------------------------------------------------------


class TestActionsAndSelection:
    async def test_passthrough_action(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=1, action=Passthrough())])
        assert await _run(cap, 'x' * 100) == 'x' * 100

    async def test_per_tool_replaces_bands(self):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=1, action=Truncate(max_chars=5))],
            per_tool={'big_tool': [Band(over=100_000, action=Truncate())]},
        )
        # global band would truncate, but per_tool threshold is huge -> passthrough
        assert await _run(cap, 'x' * 100) == 'x' * 100

    def test_select_action_no_match(self):
        assert _select_action([Band(over=100, action=Passthrough())], 50) is None

    def test_select_action_first_match(self):
        bands = [Band(over=100, action=Spill()), Band(over=10, action=Truncate())]
        assert isinstance(_select_action(bands, 50), Truncate)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# `ToolOutputLimits._handle_key` is private, but its output is the documented on-disk handle layout, so the
# tests below pin it directly.
_handle_key = ToolOutputLimits._handle_key  # pyright: ignore[reportPrivateUsage]


class TestInternals:
    def test_handle_key_defaults(self):
        limits: ToolOutputLimits[object] = ToolOutputLimits()
        ctx = _make_ctx(run_id=None, retry=2)
        ctx.tool_call_id = None
        key = _handle_key(limits, ctx, ToolCallPart(tool_name='t', args='{}', tool_call_id=''))
        assert key == 'run/call.2'

    def test_handle_key_suffix(self):
        limits: ToolOutputLimits[object] = ToolOutputLimits()
        key = _handle_key(limits, _make_ctx(), ToolCallPart(tool_name='t', args='{}', tool_call_id='c'), '.content')
        assert key.endswith('.content')

    def test_with_handles_non_mapping(self):
        meta = _with_handles('not-a-mapping', 'h/1.0', 42)
        assert meta == {'original_metadata': 'not-a-mapping', 'overflow_handle': 'h/1.0', 'overflow_bytes': 42}

    def test_with_handles_none(self):
        meta = _with_handles(None, 'h/1.0', 42)
        assert meta == {'overflow_handle': 'h/1.0', 'overflow_bytes': 42}

    def test_with_handles_content_only(self):
        meta = _with_handles({'orig': 1}, None, 0, 'h/1.0.content')
        assert meta == {'orig': 1, 'overflow_content_handle': 'h/1.0.content'}

    def test_head_tail_preview_under(self):
        assert _head_tail_preview('short', 1000) == 'short'

    def test_head_tail_preview_over(self):
        assert 'omitted' in _head_tail_preview('a' * 100, 10)

    def test_build_spill_preview_tokens_unit(self):
        unit = _Unit(binary=False, text='x' * 100, data=b'x' * 100, value='x' * 100, suffix='')
        assert 'tokens' in _build_spill_preview('h/1.0', unit, 20, over_tokens=True, read_hint='')


# ---------------------------------------------------------------------------
# read_tool_result / _read_slice (C2 bounds + literal pattern)
# ---------------------------------------------------------------------------


class TestReadBack:
    async def test_read_slice_basic(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        await store.write('h/1.0', '\n'.join(f'line {i}' for i in range(50)).encode('utf-8'))
        out = await _read_slice(
            store,
            'h/1.0',
            offset=0,
            limit=3,
            from_end=False,
            pattern=None,
            unit='lines',
            reader_name=READ_TOOL_NAME,
        )
        assert 'line 0' in out and 'line 2' in out and 'line 3' not in out

    async def test_read_slice_from_end(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        await store.write('h/1.0', '\n'.join(f'line {i}' for i in range(50)).encode('utf-8'))
        out = await _read_slice(
            store,
            'h/1.0',
            offset=0,
            limit=2,
            from_end=True,
            pattern=None,
            unit='lines',
            reader_name=READ_TOOL_NAME,
        )
        assert 'line 49' in out and 'line 48' in out

    async def test_read_slice_literal_pattern(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        await store.write('h/1.0', b'apple\nbanana\navocado\ncherry')
        out = await _read_slice(
            store,
            'h/1.0',
            offset=0,
            limit=200,
            from_end=False,
            pattern='av',
            unit='lines',
            reader_name=READ_TOOL_NAME,
        )
        assert 'avocado' in out and 'apple' not in out and 'banana' not in out

    async def test_read_slice_pattern_is_literal_not_regex(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        await store.write('h/1.0', b'plain line\n^anchored')
        # A regex metacharacter is matched literally, so it cannot trigger backtracking.
        out = await _read_slice(
            store,
            'h/1.0',
            offset=0,
            limit=200,
            from_end=False,
            pattern='^a',
            unit='lines',
            reader_name=READ_TOOL_NAME,
        )
        assert 'anchored' in out and 'plain line' not in out

    async def test_read_slice_offset_negative(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        await store.write('h/1.0', b'data')
        with pytest.raises(ModelRetry, match='offset'):
            await _read_slice(
                store,
                'h/1.0',
                offset=-1,
                limit=10,
                from_end=False,
                pattern=None,
                unit='lines',
                reader_name=READ_TOOL_NAME,
            )

    async def test_read_slice_limit_too_small(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        await store.write('h/1.0', b'data')
        with pytest.raises(ModelRetry, match='limit'):
            await _read_slice(
                store,
                'h/1.0',
                offset=0,
                limit=0,
                from_end=False,
                pattern=None,
                unit='lines',
                reader_name=READ_TOOL_NAME,
            )

    async def test_read_slice_limit_clamped(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        await store.write('h/1.0', '\n'.join(f'l{i}' for i in range(2000)).encode('utf-8'))
        out = await _read_slice(
            store,
            'h/1.0',
            offset=0,
            limit=10_000,
            from_end=False,
            pattern=None,
            unit='lines',
            reader_name=READ_TOOL_NAME,
        )
        assert out.count('\n') <= 1_000  # clamped to the line cap

    async def test_read_slice_missing_handle(self, tmp_path: Path):
        # A missing/wrong handle returns a guiding message (it does NOT raise): a bad
        # handle must not consume a retry and escalate to a fatal UnexpectedModelBehavior.
        store = LocalFileOverflowStore(base_dir=tmp_path)
        out = await _read_slice(
            store,
            'missing/1.0',
            offset=0,
            limit=10,
            from_end=False,
            pattern=None,
            unit='lines',
            reader_name=READ_TOOL_NAME,
        )
        assert 'No stored tool result' in out
        assert 're-run the original tool' in out
        # The store's error (which can carry the resolved filesystem path) is not leaked.
        assert str(tmp_path) not in out

    @pytest.mark.parametrize('from_end', [False, True])
    @pytest.mark.parametrize('pattern', [None, 'é'])
    @pytest.mark.parametrize(
        'text', ['é1\r\nb\rcé\n\nd\u2028é2\x85f\r\n\ré3', 'é1\r\nb\ré2\r', 'ab\r\nc\r\ré', 'éabcdefgh\n\nijklmé']
    )
    async def test_read_slice_streams_like_splitlines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, from_end: bool, pattern: str | None, text: str
    ):
        monkeypatch.setattr('pydantic_ai_harness.tool_output_limits._store._CHUNK_SIZE', 3)
        store = LocalFileOverflowStore(base_dir=tmp_path)
        await store.write('h/1.0', text.encode('utf-8'))
        lines = [line for line in text.splitlines() if pattern is None or pattern in line]
        for offset in range(len(lines) + 1):
            out = await _read_slice(
                store,
                'h/1.0',
                offset=offset,
                limit=2,
                from_end=from_end,
                pattern=pattern,
                unit='lines',
                reader_name=READ_TOOL_NAME,
            )
            end = len(lines) - offset
            body = '\n'.join(lines[max(0, end - 2) : end] if from_end else lines[offset : offset + 2])
            assert out.split('\n', 1)[1:] == ([body] if body else [])

    async def test_read_slice_streams_a_local_payload_in_chunks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        class CountingReader(io.BufferedReader):
            def __init__(self, raw: io.RawIOBase) -> None:
                super().__init__(raw)
                self.reads: list[int] = []

            def read(self, size: int | None = -1, /) -> bytes:
                data = super().read(size)
                self.reads.append(len(data))
                return data

        monkeypatch.setattr('pydantic_ai_harness.tool_output_limits._store._CHUNK_SIZE', 8)
        payload = '\n'.join(f'line {i}' for i in range(1_000)).encode('utf-8')
        store = LocalFileOverflowStore(base_dir=tmp_path)
        await store.write('h/1.0', payload)
        opened: list[CountingReader] = []
        open_file = LocalFileOverflowStore._open  # pyright: ignore[reportPrivateUsage]

        def spy(self: LocalFileOverflowStore, handle: str) -> io.BufferedReader:
            opened.append(CountingReader(open_file(self, handle).detach()))
            return opened[-1]

        monkeypatch.setattr(LocalFileOverflowStore, '_open', spy)
        out = await _read_slice(
            store,
            'h/1.0',
            offset=10,
            limit=3,
            from_end=False,
            pattern=None,
            unit='lines',
            reader_name=READ_TOOL_NAME,
        )
        assert out == "[handle 'h/1.0': 1,000 matching line(s); showing 3]\nline 10\nline 11\nline 12"
        out = await _read_slice(
            store,
            'h/1.0',
            offset=0,
            limit=2,
            from_end=True,
            pattern=None,
            unit='lines',
            reader_name=READ_TOOL_NAME,
        )
        assert out == "[handle 'h/1.0': 1,000 matching line(s); showing 2]\nline 998\nline 999"
        for file in opened:
            assert file.closed
            assert sum(file.reads) == len(payload)
            assert max(file.reads) == 8

    async def test_read_slice_splits_a_whole_read_in_chunks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A store with no ranged read is read whole, then split one chunk at a time, yielding to the loop in between."""
        monkeypatch.setattr('pydantic_ai_harness.tool_output_limits._store._CHUNK_SIZE', 8)
        payload = '\n'.join(f'line {i}' for i in range(1_000)).encode('utf-8')
        store = WorkspaceOverflowStore().bind(Workspace(LocalWorkspaceBackend(working_dir=tmp_path)))
        handle = await store.write('run/call.0', payload)
        turns = 0

        async def count_turns() -> None:
            nonlocal turns
            while True:
                turns += 1
                await anyio.lowlevel.checkpoint()

        async with anyio.create_task_group() as tg:
            tg.start_soon(count_turns)
            out = await _read_slice(
                store,
                handle,
                offset=10,
                limit=3,
                from_end=False,
                pattern=None,
                unit='lines',
                reader_name=READ_TOOL_NAME,
            )
            tg.cancel_scope.cancel()
            assert out == f'[handle {handle!r}: 1,000 matching line(s); showing 3]\nline 10\nline 11\nline 12'
        assert turns >= len(payload) // 8

    @pytest.mark.parametrize('from_end', [False, True])
    async def test_read_slice_accepts_an_offset_past_sys_maxsize(self, tmp_path: Path, from_end: bool):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        await store.write('h/1.0', b'a\nb')
        out = await _read_slice(
            store,
            'h/1.0',
            offset=sys.maxsize + 1,
            limit=5,
            from_end=from_end,
            pattern=None,
            unit='lines',
            reader_name=READ_TOOL_NAME,
        )
        assert out == "[handle 'h/1.0': 2 matching line(s); showing 0]"

    async def test_read_slice_uses_an_overridden_read(self, tmp_path: Path):
        class UpperStore(LocalFileOverflowStore):
            async def read(self, handle: str) -> bytes:
                return (await super().read(handle)).upper()

        store = UpperStore(base_dir=tmp_path)
        await store.write('h/1.0', b'a\nb')
        out = await _read_slice(
            store,
            'h/1.0',
            offset=0,
            limit=5,
            from_end=False,
            pattern=None,
            unit='lines',
            reader_name=READ_TOOL_NAME,
        )
        assert out == "[handle 'h/1.0': 2 matching line(s); showing 2]\nA\nB"

    async def test_read_slice_reads_other_stores_whole(self):
        store = AsyncMock()
        store.read.return_value = b'a\nb\nc'
        out = await _read_slice(
            store,
            'h',
            offset=1,
            limit=5,
            from_end=False,
            pattern=None,
            unit='lines',
            reader_name=READ_TOOL_NAME,
        )
        assert out == "[handle 'h': 3 matching line(s); showing 2]\nb\nc"

    async def test_get_toolset_registers_read_tool(self, tmp_path: Path):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        await store.write('h/1.0', b'hello\nworld')
        cap: ToolOutputLimits[object] = ToolOutputLimits(store=store)
        toolset = cap.get_toolset()
        assert toolset is not None
        tool = toolset.tools[READ_TOOL_NAME]  # type: ignore[union-attr]
        out = await tool.function(_make_ctx(), 'h/1.0')  # type: ignore[attr-defined]
        assert 'hello' in out


class TestReadToolResultBytes:
    @pytest.mark.parametrize('from_end', [False, True])
    async def test_streams_in_chunks_like_a_whole_slice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, from_end: bool
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.tool_output_limits._store._CHUNK_SIZE', 3)
        data = ('é' * 30 + 'abc' * 10).encode('utf-8')
        for offset in (0, 1, 7, 64, 89, 90, 200):
            out = await query_spill(
                tmp_path,
                data.decode(),
                READ_TOOL_NAME,
                {'unit': 'bytes', 'offset': offset, 'limit': 11, 'from_end': from_end},
            )
            end = max(0, len(data) - offset) if from_end else min(offset + 11, len(data))
            start = max(0, end - 11) if from_end else min(offset, len(data))
            body = data[start:end].decode('utf-8', errors='replace')
            header = f': bytes {start}-{end} of {len(data)} (end exclusive)]'
            assert out.endswith(f'{header}\n{body}' if body else header)

    async def test_slice(self, tmp_path: Path) -> None:
        payload = '{"a":' + '1' * 100 + '}'
        out = await query_spill(tmp_path, payload, READ_TOOL_NAME, {'unit': 'bytes', 'offset': 2, 'limit': 5})
        assert out.endswith(': bytes 2-7 of 106 (end exclusive)]\na":11')

    async def test_from_end(self, tmp_path: Path) -> None:
        payload = 'x' * 100 + 'tail'
        out = await query_spill(tmp_path, payload, READ_TOOL_NAME, {'unit': 'bytes', 'limit': 4, 'from_end': True})
        assert out.endswith(': bytes 100-104 of 104 (end exclusive)]\ntail')

    async def test_range_is_clamped(self, tmp_path: Path) -> None:
        past = await query_spill(tmp_path, 'x' * 100, READ_TOOL_NAME, {'unit': 'bytes', 'offset': 500})
        assert past.endswith(': bytes 100-100 of 100 (end exclusive)]')
        whole = await query_spill(tmp_path, 'x' * 60_000, READ_TOOL_NAME, {'unit': 'bytes', 'limit': 100_000})
        assert whole.split('\n', 1)[0].endswith(': bytes 0-50000 of 60000 (end exclusive)]')

    async def test_cut_utf8_sequence_renders_as_replacement_character(self, tmp_path: Path) -> None:
        payload = 'é' * 60
        out = await query_spill(tmp_path, payload, READ_TOOL_NAME, {'unit': 'bytes', 'offset': 1, 'limit': 4})
        assert out.endswith(': bytes 1-5 of 120 (end exclusive)]\n\ufffdé\ufffd')

    async def test_pattern_is_rejected(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, 'x' * 100, READ_TOOL_NAME, {'unit': 'bytes', 'pattern': 'x'})
        assert out == "retry: `pattern` filters lines, so it cannot be combined with `unit='bytes'`."


class TestReadToolResultCap:
    async def test_cut_offset_counts_a_crlf_split_across_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.tool_output_limits._store._CHUNK_SIZE', 3)
        out = await query_spill(tmp_path, 'ab\r\ncd\n' + 'x' * 50_010, READ_TOOL_NAME, {'offset': 2})
        assert out.split('\n', 1)[0].endswith(
            "the cut line continues at read_tool_result(handle='<h>', unit='bytes', offset=50007, limit=50000)]"
        )

    async def test_cut_first_line_continues_after_the_shown_bytes(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, 'x' * 60_000, READ_TOOL_NAME, {})
        assert out.split('\n', 1)[0].endswith(
            "the cut line continues at read_tool_result(handle='<h>', unit='bytes', offset=50000, limit=50000)]"
        )

    async def test_line_like_a_group_separator_is_kept(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, 'x' * 60 + '\n--\n', READ_TOOL_NAME, {})
        assert out.endswith(': 2 matching line(s); showing 2]\n' + 'x' * 60 + '\n--')

    async def test_cut_line_continues_at_the_reported_byte_offset(self, tmp_path: Path) -> None:
        payload = 'short\r\n' + 'x' * 50_000 + 'TAIL\nnext'
        out = await query_spill(tmp_path, payload, READ_TOOL_NAME, {'offset': 1})
        header, body = out.split('\n', 1)
        assert header.endswith(
            ': 3 matching line(s); showing 1, output capped at 50,000 chars; the cut line continues at '
            "read_tool_result(handle='<h>', unit='bytes', offset=50007, limit=50000); the next lines are at read_tool_result(handle='<h>', offset=2)]"
        )
        assert body == 'x' * 50_000
        hint = re.search(r"unit='bytes', offset=(\d+), limit=(\d+)", header)
        assert hint is not None
        args: dict[str, object] = {'unit': 'bytes', 'offset': int(hint[1]), 'limit': int(hint[2])}
        rest = await query_spill(tmp_path, payload, READ_TOOL_NAME, args)
        assert rest.endswith(': bytes 50007-50016 of 50016 (end exclusive)]\nTAIL\nnext')

    async def test_cut_hint_after_a_filtered_read_is_a_fresh_call(self, tmp_path: Path) -> None:
        out = await query_spill(tmp_path, 'short\n' + 'x' * 50_010, READ_TOOL_NAME, {'pattern': 'x'})
        assert out.split('\n', 1)[0].endswith(
            ': 1 matching line(s); showing 1, output capped at 50,000 chars; the cut line continues at '
            "read_tool_result(handle='<h>', unit='bytes', offset=50006, limit=50000)]"
        )

    async def test_stops_before_a_line_that_would_cross_the_cap(self, tmp_path: Path) -> None:
        payload = '\n'.join(['a' * 30_000, 'b' * 30_000, 'c'])
        out = await query_spill(tmp_path, payload, READ_TOOL_NAME, {'limit': 3, 'from_end': True})
        header, body = out.split('\n', 1)
        assert header.endswith(
            ': 3 matching line(s); showing 1, output capped at 50,000 chars; the next lines are at '
            "read_tool_result(handle='<h>', offset=1)]"
        )
        assert body == 'a' * 30_000
        out = await query_spill(tmp_path, '\n'.join(['xa' * 15_000, 'xb' * 15_000]), READ_TOOL_NAME, {'pattern': 'x'})
        assert out.split('\n', 1)[0].endswith("read_tool_result(handle='<h>', offset=1, pattern='x')]")


# ---------------------------------------------------------------------------
# Agent-path integration
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    async def test_spill_persists_in_history(self, tmp_path: Path, anyio_backend: str):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=100, action=Spill(preview_chars=50))], store=store
        )
        agent = Agent(TestModel(call_tools=['big_tool']), capabilities=[cap])

        @agent.tool_plain
        def big_tool() -> str:
            return 'data line\n' * 500

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        spilled = [p for p in returns if p.tool_name == 'big_tool']
        assert spilled
        part = spilled[0]
        assert isinstance(part.content, str) and 'too large' in part.content
        assert part.metadata is not None and 'overflow_handle' in part.metadata
        assert await store.read(part.metadata['overflow_handle']) == ('data line\n' * 500).encode('utf-8')

    async def test_spill_preserves_non_mapping_metadata_through_agent_run(self, tmp_path: Path, anyio_backend: str):
        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=100, action=Spill())], store=store)
        agent = Agent(TestModel(call_tools=['big_tool']), capabilities=[cap])

        @agent.tool_plain
        def big_tool() -> ToolReturn:
            return ToolReturn(return_value='data line\n' * 500, metadata='app-request-id-123')

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        spilled = [p for p in returns if p.tool_name == 'big_tool']
        assert spilled
        part = spilled[0]
        assert part.metadata is not None
        assert part.metadata['original_metadata'] == 'app-request-id-123'
        assert 'overflow_handle' in part.metadata

    @pytest.mark.parametrize('metadata', [['a', 'b'], ('a', 'b')], ids=['list', 'tuple'])
    async def test_spill_appends_handles_to_list_metadata(self, tmp_path: Path, metadata: object, anyio_backend: str):
        result = await _ui_event_agent(tmp_path, metadata).run('go')
        part = _big_tool_return(result.all_messages())
        assert part.metadata[:2] == ['a', 'b']
        assert part.metadata[2].keys() == {'overflow_handle', 'overflow_bytes'}

    @pytest.mark.skipif(not ag_ui_imports_successful(), reason='ag-ui-protocol not installed')
    @pytest.mark.parametrize('as_list', [True, False], ids=['list', 'single'])
    async def test_spill_keeps_ag_ui_events(self, tmp_path: Path, as_list: bool, anyio_backend: str):
        event = CustomEvent(name='cards', value={'n': 100})
        run_input = RunAgentInput(
            thread_id='t',
            run_id='r',
            messages=[UserMessage(id='u', content='go')],
            state={},
            tools=[],
            context=[],
            forwarded_props={},
        )
        adapter = AGUIAdapter(agent=_ui_event_agent(tmp_path, [event] if as_list else event), run_input=run_input)
        events = [e async for e in adapter.run_stream()]
        assert [(e.name, e.value) for e in events if isinstance(e, CustomEvent)] == [('cards', {'n': 100})]
        assert list(tmp_path.rglob('*'))

    @pytest.mark.skipif(not vercel_ai_imports_successful(), reason='vercel ai dependencies not installed')
    @pytest.mark.parametrize('as_list', [True, False], ids=['list', 'single'])
    async def test_spill_keeps_vercel_ai_data_chunks(self, tmp_path: Path, as_list: bool, anyio_backend: str):
        chunk = DataChunk(type='data-search-results', data={'count': 100})
        result = await _ui_event_agent(tmp_path, [chunk] if as_list else chunk).run('go')
        assert 'overflow_handle' in _big_tool_return(result.all_messages()).metadata[-1]
        ui_parts = [p for m in VercelAIAdapter.dump_messages(result.all_messages()) for p in m.parts]
        assert [(p.type, p.data) for p in ui_parts if isinstance(p, DataUIPart)] == [
            ('data-search-results', {'count': 100})
        ]

    async def test_structured_return_with_bytes_that_are_not_utf8(self, anyio_backend: str):
        """Binary in a structured return is measured as base64 rather than failing the run."""

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[_call('read_blob')])
            return ModelResponse(parts=[TextPart('done')])

        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Truncate(max_chars=1000))])
        agent = Agent(FunctionModel(respond), capabilities=[cap])

        @agent.tool_plain
        def read_blob() -> dict[str, bytes]:
            return {'blob': b'\xff' * 3}

        result = await agent.run('go')
        returns = [p.content for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        assert returns == ['{"blob":"____"}']

    async def test_file_return_reaches_the_model_untruncated(self):
        image = BinaryContent(data=b'\x89PNG\r\n\x1a\n' + b'x' * 100_000, media_type='image/png')
        requests: list[list[ModelMessage]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            requests.append(messages.copy())
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name='read_image', args='{}')])
            return ModelResponse(parts=[TextPart(content='done')])

        agent = Agent(
            FunctionModel(respond),
            capabilities=[ToolOutputLimits(bands=[Band(over=10, action=Truncate(max_chars=10))])],
        )

        @agent.tool_plain
        def read_image() -> BinaryContent:
            return image

        result = await agent.run('read image')

        # The follow-up request is what the model receives; history is what later requests replay.
        for messages in (requests[1], result.all_messages()):
            tool_return = messages[2].parts[0]
            assert isinstance(tool_return, ToolReturnPart)
            assert tool_return.content == image

    async def test_small_output_untouched(self, tmp_path: Path, anyio_backend: str):
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=10_000, action=Spill())], store=LocalFileOverflowStore(base_dir=tmp_path)
        )
        agent = Agent(TestModel(call_tools=['small_tool']), capabilities=[cap])

        @agent.tool_plain
        def small_tool() -> str:
            return 'tiny'

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        small = [p for p in returns if p.tool_name == 'small_tool']
        assert small and small[0].content == 'tiny'

    async def test_capability_load_is_not_reduced(self, anyio_backend: str):
        """A reduced `load_capability` return would not count as a load, leaving the capability's tools locked."""

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'refunds'})])
            if len(messages) == 3:
                return ModelResponse(parts=[ToolCallPart('refund_status', {'order_id': 'A1'})])
            return ModelResponse(parts=[TextPart('done')])

        refunds = Capability[object](id='refunds', description='Refunds.', defer_loading=True)

        @refunds.tool_plain
        def refund_status(order_id: str) -> str:
            return f'Order {order_id}: refunded'

        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Truncate(max_chars=20))])
        agent = Agent(FunctionModel(respond), capabilities=[refunds, cap])

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        assert [type(p) for p in returns] == [LoadCapabilityReturnPart, ToolReturnPart]
        assert returns[0].content == {'message': "Capability 'refunds' loaded. Its instructions, if any, follow."}
        assert returns[1].content == 'Order A1: refunded'

    async def test_tool_search_is_not_reduced(self, anyio_backend: str):
        """A reduced `search_tools` return would not reveal the tools it found."""

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('search_tools', {'queries': ['refund']})])
            if len(messages) == 3:
                return ModelResponse(parts=[ToolCallPart('refund_status', {'order_id': 'A1'})])
            return ModelResponse(parts=[TextPart('done')])

        cap: ToolOutputLimits[object] = ToolOutputLimits(bands=[Band(over=10, action=Truncate(max_chars=20))])
        agent = Agent(FunctionModel(respond), capabilities=[cap])

        @agent.tool_plain(defer_loading=True)
        def refund_status(order_id: str) -> str:
            """Look up a refund."""
            return f'Order {order_id}: refunded'

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        assert [type(p) for p in returns] == [ToolSearchReturnPart, ToolReturnPart]
        assert returns[0].content == {'discovered_tools': [{'name': 'refund_status'}]}
        assert returns[1].content == 'Order A1: refunded'

    async def test_a_reused_tool_call_id_spills_each_call_separately(self, tmp_path: Path, anyio_backend: str):
        """A provider that numbers calls per response (`call_0`, `call_0`) must not overwrite the first spill."""

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('dump', {'tag': 'first'}, tool_call_id='call_0')])
            if len(messages) == 3:
                return ModelResponse(parts=[ToolCallPart('dump', {'tag': 'second'}, tool_call_id='call_0')])
            return ModelResponse(parts=[TextPart('done')])

        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=100, action=Spill(preview_chars=20))], store=store
        )
        agent = Agent(FunctionModel(respond), capabilities=[cap])

        @agent.tool_plain
        def dump(tag: str) -> str:
            return f'{tag} payload\n' * 100

        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        handles = [p.metadata['overflow_handle'] for p in returns if p.metadata is not None]
        assert len(handles) == 2
        assert len(set(handles)) == 2
        # The first call keeps the documented `run_id/tool_call_id.retry` handle and only the repeated
        # id is parked one level down, so the shape is what says which of the two moved.
        assert handles[0].count('/') == 1
        assert handles[0].endswith('/call_0.0')
        assert handles[1].count('/') == 2
        assert handles[1].endswith('/call_0.0')
        assert await store.read(handles[0]) == b'first payload\n' * 100
        assert await store.read(handles[1]) == b'second payload\n' * 100

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_a_reused_tool_call_id_survives_a_compaction_between_the_responses(
        self, tmp_path: Path, anyio_backend: str
    ):
        """A compaction rewrites `ctx.messages`, so a later call reusing an id must not reuse the first handle."""
        issued: list[int] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            issued.append(1)
            if len(issued) == 1:
                return ModelResponse(parts=[ToolCallPart('dump', {'tag': 'first'}, tool_call_id='call_0')])
            # An unrelated call in between gives the compaction a boundary to cut at, so the first
            # response and its return leave the history before the second `call_0` is issued.
            if len(issued) == 2:
                return ModelResponse(parts=[ToolCallPart('dump', {'tag': 'middle'}, tool_call_id='call_9')])
            if len(issued) == 3:
                return ModelResponse(parts=[ToolCallPart('dump', {'tag': 'second'}, tool_call_id='call_0')])
            return ModelResponse(parts=[TextPart('done')])

        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=100, action=Spill(preview_chars=20))], store=store
        )
        compaction: SummarizingCompaction[object] = SummarizingCompaction(
            model=FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('earlier turns')])),
            max_messages=4,
            keep_messages=2,
        )
        agent = Agent(FunctionModel(respond), capabilities=[cap, compaction])

        @agent.tool_plain
        def dump(tag: str) -> str:
            return f'{tag} payload\n' * 100

        result = await agent.run('go')
        messages = result.all_messages()
        assert any(isinstance(part, CompactionPart) for message in messages for part in message.parts), (
            'the test must compact between the two responses'
        )
        # The first return left the history with its response, so the spills are read off the store.
        # Deriving the key from the retained history would reuse `call_0`'s handle here and leave two
        # files, the first payload overwritten by the second.
        spilled = sorted(path.read_bytes() for path in tmp_path.rglob('*') if path.is_file())
        assert spilled == [b'first payload\n' * 100, b'middle payload\n' * 100, b'second payload\n' * 100]

    async def test_a_reused_tool_call_id_is_tracked_per_run(self, tmp_path: Path, anyio_backend: str):
        """A later run's first spill of a name keeps the plain handle, whatever an earlier run spilled."""
        runs: list[int] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                runs.append(1)
                # The second run spills `call_0` first at a later step than the first run did.
                size = 100 if len(runs) == 1 else 1
                return ModelResponse(parts=[ToolCallPart('dump', {'size': size}, tool_call_id='call_0')])
            if len(runs) == 2 and len(messages) == 3:
                return ModelResponse(parts=[ToolCallPart('dump', {'size': 100}, tool_call_id='call_0')])
            return ModelResponse(parts=[TextPart('done')])

        store = LocalFileOverflowStore(base_dir=tmp_path)
        cap: ToolOutputLimits[object] = ToolOutputLimits(
            bands=[Band(over=100, action=Spill(preview_chars=20))], store=store
        )
        agent = Agent(FunctionModel(respond), capabilities=[cap])

        @agent.tool_plain
        def dump(size: int) -> str:
            return 'payload\n' * size

        handles: list[str] = []
        for _ in range(2):
            result = await agent.run('go')
            returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
            handles += [p.metadata['overflow_handle'] for p in returns if p.metadata is not None]
        assert len(handles) == 2
        assert [handle.count('/') for handle in handles] == [1, 1]
