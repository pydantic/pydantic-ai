"""Fails the test that leaves a resource for someone else to clean up.

Registered by `tests/conftest.py`; see "Leaks" in `tests/AGENTS.md`.

A leak the garbage collector finalizes is reported in whichever test happens to collect it, if anywhere, and a leak
that something keeps alive is never reported at all. `tests/gc_isolation.py` already makes a test collect its own
garbage. This plugin fails the test for what that cannot see:

- An event loop it created on a thread other than the main one and never closed, such as the one `Agent.run_sync()`
  sets on a worker thread. While the thread lives it keeps the loop alive, so no `ResourceWarning` reports it, and
  once the thread exits the warning lands in whichever test is running then. Loops a `module`, `class`, `package`
  or `session`-scoped fixture creates are left to that fixture, and loops third-party code creates to that code. A
  loop left on the main thread is the one `run_sync()` reuses there; once dropped, the collector finalizes it in the
  test that dropped it.
- An HTTP client an agent run built and left open, such as the one a run gets for a model it builds from a name
  (https://github.com/pydantic/pydantic-ai/pull/10067). The autouse `track_httpx_clients` fixture closes every client
  a test leaves open, and fails the test only for one a run built: a client that test code built, or that a provider
  or agent built outside a run, is its builder's to close.
- An async generator of this repository's code that the collector finalized, because nothing closed it after its
  consumer stopped early (https://github.com/pydantic/pydantic-ai/pull/13859). asyncio then schedules its `aclose()` on
  a loop that may never run it, so its `finally` blocks run late or not at all. `GC_FINALIZED_BASELINE` lists the ones
  `main` still leaves to the collector.
"""

from __future__ import annotations as _annotations

import asyncio
import inspect
import itertools
import os
import threading
import traceback
import weakref
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from types import AsyncGeneratorType
from typing import Any

import pytest

PLUGIN_NAME = 'pydantic_ai_leak_guards'

_ROOT = Path(__file__).resolve().parent.parent
_ASYNCIO = Path(asyncio.__file__).resolve().parent
_TESTS = f'{_ROOT / "tests"}{os.sep}'
_AGENT = str(_ROOT / 'pydantic_ai_slim' / 'pydantic_ai' / 'agent' / '__init__.py')
_AGENT_GRAPH = str(_ROOT / 'pydantic_ai_slim' / 'pydantic_ai' / '_agent_graph.py')


GC_FINALIZED_BASELINE: frozenset[str] = frozenset(
    {
        # Each is left suspended when its consumer stops early and nothing closes it; fixing each is product work.
        'pydantic_ai_slim/pydantic_ai/_agent_graph.py:buffered_events',
        'pydantic_ai_slim/pydantic_ai/_ssrf.py:_aiter_raw',
        'pydantic_ai_slim/pydantic_ai/_tool_execution.py:_ToolCallProcessor._validate_function_calls',
        'pydantic_ai_slim/pydantic_ai/_tool_execution.py:_iter_completed_or_buffered',
        'pydantic_ai_slim/pydantic_ai/durable_exec/_base.py:BaseDurabilityCapability._single_event_stream',
        'pydantic_ai_slim/pydantic_ai/models/__init__.py:StreamedResponse.__aiter__.<locals>.iterator_with_cancel_guard',
        'pydantic_ai_slim/pydantic_ai/models/__init__.py:StreamedResponse.__aiter__.<locals>.iterator_with_final_event',
        'pydantic_ai_slim/pydantic_ai/models/__init__.py:StreamedResponse.__aiter__.<locals>.iterator_with_part_end',
        'pydantic_ai_slim/pydantic_ai/models/_continuation.py:_ContinuationStreamedResponse._iterator_with_cancel_guard',
        'pydantic_ai_slim/pydantic_ai/models/anthropic/__init__.py:AnthropicModel._map_user_prompt',
        'pydantic_ai_slim/pydantic_ai/models/function.py:FunctionStreamedResponse._get_event_iterator',
        'pydantic_ai_slim/pydantic_ai/models/openai/_responses_streaming.py:OpenAIResponsesStreamedResponse._get_event_iterator',
        'pydantic_ai_slim/pydantic_ai/models/test.py:TestStreamedResponse._get_event_iterator',
        'pydantic_ai_slim/pydantic_ai/realtime/_inferred_lifecycle.py:InferredLifecycle.frames',
        'pydantic_ai_slim/pydantic_ai/realtime/_session.py:RealtimeSession.__aiter__',
        'pydantic_ai_slim/pydantic_ai/realtime/_session.py:RealtimeSession.stream_audio.<locals>.iterate',
        'pydantic_ai_slim/pydantic_ai/realtime/_session.py:RealtimeSession.stream_transcripts.<locals>.iterate',
        'pydantic_ai_slim/pydantic_ai/realtime/codec.py:RealtimeConnection.__aiter__',
        'pydantic_ai_slim/pydantic_ai/realtime/elevenlabs_agents.py:ElevenLabsAgentsRealtimeConnection.frames',
        'pydantic_ai_slim/pydantic_ai/realtime/google.py:GoogleRealtimeConnection.frames',
        'pydantic_ai_slim/pydantic_ai/realtime/openai.py:OpenAIRealtimeConnection.frames',
        'pydantic_ai_slim/pydantic_ai/realtime/openai_live.py:OpenAILiveConnection.frames',
        'pydantic_ai_slim/pydantic_ai/result.py:AgentStream._pull_shared',
        'pydantic_ai_slim/pydantic_ai/result.py:AgentStream._stream_response_text',
        'pydantic_ai_slim/pydantic_ai/result.py:AgentStream.stream_response',
        'pydantic_ai_slim/pydantic_ai/result.py:AgentStream.stream_text',
        'pydantic_ai_slim/pydantic_ai/result.py:StreamedRunResult.stream_response',
        'pydantic_ai_slim/pydantic_ai/result.py:StreamedRunResult.stream_text',
        'pydantic_ai_slim/pydantic_ai/ui/_adapter.py:UIAdapter._saving_conversation',
        'pydantic_ai_slim/pydantic_ai/ui/_event_stream.py:UIEventStream._turn_to',
        'pydantic_ai_slim/pydantic_ai/ui/_event_stream.py:UIEventStream.handle_part_end',
        'pydantic_ai_slim/pydantic_ai/ui/_event_stream.py:UIEventStream.transform_stream',
        'pydantic_ai_slim/pydantic_ai/ui/vercel_ai/_event_stream.py:VercelAIEventStream.before_response',
        'pydantic_ai_slim/pydantic_ai/ui/vercel_ai/_event_stream.py:VercelAIEventStream.before_stream',
        'src/pydantic_ai_harness/pydantic_ai_harness/tool_output_limits/_capability.py:_aiter',
        # Test code, left suspended the same way.
        'tests/continuation_utils.py:ScriptedStreamedResponse._get_event_iterator',
        'tests/harness/guardrails/test_stream_guardrail.py:TestStreamGuardrail.test_block_and_redaction_spans_gate_content.<locals>.events',
        'tests/harness/sub_agents/test_tasks.py:parent_model.<locals>.stream',
        'tests/models/google/test_agentic_video.py:_stream_response_parts.<locals>.response_chunks',
        'tests/models/test_continuation_stream.py:_FakeStream._get_event_iterator',
        'tests/models/test_openai_responses.py:test_background_mode_streaming_continuation_vcr.<locals>.<genexpr>',
        'tests/models/test_openai_responses.py:test_background_mode_streaming_starting_after_vcr.<locals>.<genexpr>',
        'tests/models/test_streamed_continuation.py:_ScriptedStreamedResponse._get_event_iterator',
        'tests/realtime/test_google.py:_DroppableSession.receive',
        'tests/realtime/test_google.py:_RecordingSession.receive',
        'tests/realtime/test_openai.py:FakeWebSocket.__aiter__',
        'tests/realtime/test_openai.py:_DroppableWebSocket.__aiter__',
        'tests/realtime/test_openai.py:_QueuedWebSocket.__aiter__',
        'tests/realtime/test_session.py:FakeRealtimeConnection.codec_events',
        'tests/realtime/test_session.py:test_send_audio_bad_later_chunk_keeps_earlier_chunks.<locals>.chunks',
        'tests/realtime/test_session.py:test_send_audio_first_chunk_bad_rolls_back_turn_state.<locals>.first_chunk_bad',
        'tests/realtime/test_session.py:test_send_audio_iterable_waits_for_next_chunk_after_close.<locals>.microphone',
        'tests/realtime/test_session.py:test_tool_close_ends_async_audio_sender_cleanly.<locals>.microphone',
        'tests/realtime/ws_cassettes.py:ReplayWebSocket.__aiter__',
        'tests/realtime_connections.py:lifecycle_events',
        'tests/refactor_capture.py:RefactorCapture._recorded_ui_events.<locals>.recorded',
        'tests/test_ui.py:DummyUIEventStream.before_request',
        'tests/test_ui.py:DummyUIEventStream.before_response',
        'tests/test_ui.py:DummyUIEventStream.handle_part_end',
        'tests/test_ui.py:DummyUIEventStream.handle_tool_call_end',
    }
)
"""This repository's async generators that the collector finalizes on `main` today, as `<path>:<qualname>`.

`scripts/check_test_baselines.py` fails a change that adds one: fix the generator instead.
"""


_CREATOR = '_pydantic_ai_leak_guards_creator'
"""The attribute that holds, on a loop the running test created, the frame that created it."""


@dataclass
class _State:
    test_running: bool = False
    shared_fixture_depth: int = 0
    generation: int = 0
    """Numbers the running test, so a loop finalized in a later test is not blamed on that one."""
    loops: weakref.WeakSet[asyncio.AbstractEventLoop] = field(
        default_factory=weakref.WeakSet[asyncio.AbstractEventLoop]
    )
    """Loops first-party code created off the main thread during the running test."""
    finalized_open: list[str] = field(default_factory=list[str])
    """Where each such loop that the collector finalized while it was still open was created."""
    run_built_clients: weakref.WeakKeyDictionary[object, str] = field(
        default_factory=weakref.WeakKeyDictionary[object, str]
    )
    """HTTP clients an agent run built during the running test, each with the run's code that built it."""
    clients_left_open: list[str] = field(default_factory=list[str])
    """Where each such client that the test left for `track_httpx_clients` to close was built."""
    finalized_generators: list[str] = field(default_factory=list[str])
    """Each of this repository's async generators outside the baseline that the collector finalized in the test."""


_state = _State()
"""The running test's state; a test that runs a nested pytest session (`pytester`) gets its own back afterwards."""
_generations = itertools.count(1)


def _first_party_creator() -> str | None:
    """The frame that created the loop being initialized, when it is this repository's own code."""
    for frame in reversed(traceback.extract_stack()[:-2]):
        path = Path(frame.filename).resolve()
        if path.is_relative_to(_ASYNCIO) or path == Path(__file__).resolve():
            continue
        if not path.is_relative_to(_ROOT) or '.venv' in path.parts or 'site-packages' in path.parts:
            return None
        return f'{path.relative_to(_ROOT)}:{frame.lineno} in {frame.name}'
    return None  # pragma: no cover


_original_init = asyncio.BaseEventLoop.__init__


def _tracking_init(self: asyncio.BaseEventLoop, *args: Any, **kwargs: Any) -> None:
    _original_init(self, *args, **kwargs)
    on_worker_thread = threading.current_thread() is not threading.main_thread()
    if (
        on_worker_thread
        and _state.test_running
        and not _state.shared_fixture_depth
        and (creator := _first_party_creator())
    ):
        setattr(self, _CREATOR, (_state.generation, creator))
        _state.loops.add(self)


_original_del = asyncio.BaseEventLoop.__del__


def _tracking_del(self: asyncio.BaseEventLoop, *args: Any) -> None:
    # A thread that exits drops the loop it holds, so the loop can be finalized before its test reports.
    created: tuple[int, str] | None = getattr(self, _CREATOR, None)
    if created is not None and created[0] == _state.generation and not self.is_closed():
        _state.finalized_open.append(created[1])
    _original_del(self, *args)


# Not in the stubs. asyncio calls it when the collector finalizes an async generator iterated on the loop and not closed.
_original_finalizer_hook: Callable[[asyncio.BaseEventLoop, AsyncGeneratorType[object, object]], None] = getattr(
    asyncio.BaseEventLoop, '_asyncgen_finalizer_hook'
)


def _tracking_finalizer_hook(self: asyncio.BaseEventLoop, agen: AsyncGeneratorType[object, object]) -> None:
    path = Path(agen.ag_code.co_filename).resolve()
    if _state.test_running and path.is_relative_to(_ROOT) and not {'.venv', 'site-packages'} & set(path.parts):
        site = f'{path.relative_to(_ROOT)}:{agen.__qualname__}'
        if site not in GC_FINALIZED_BASELINE:
            _state.finalized_generators.append(site)
    _original_finalizer_hook(self, agen)


# Patched once, at import, so a nested pytest session (`pytester`) that registers this module too cannot undo it.
setattr(asyncio.BaseEventLoop, '__init__', _tracking_init)
setattr(asyncio.BaseEventLoop, '__del__', _tracking_del)
setattr(asyncio.BaseEventLoop, '_asyncgen_finalizer_hook', _tracking_finalizer_hook)


def note_client_created(client: object) -> None:
    """Note the HTTP client that `track_httpx_clients`' factory just created, if an agent run built it.

    A run built it when an `Agent.iter` or agent graph frame asked for it with no test code in between.
    """
    frame = inspect.currentframe()
    # This function and the fixture's factory come first; the frame after them asked for the client.
    frame = frame and frame.f_back and frame.f_back.f_back
    while frame is not None:
        code = frame.f_code
        if code.co_filename.startswith(_TESTS):
            return
        if code.co_filename == _AGENT_GRAPH or (code.co_filename == _AGENT and code.co_name == 'iter'):
            site = f'{Path(code.co_filename).relative_to(_ROOT)}:{frame.f_lineno} in {code.co_name}'
            _state.run_built_clients[client] = site
            return
        frame = frame.f_back


def note_client_left_open(client: object) -> None:
    """Note that `track_httpx_clients` is closing a client the test left open."""
    if (site := _state.run_built_clients.get(client)) is not None:
        _state.clients_left_open.append(site)


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef: pytest.FixtureDef[Any]) -> Generator[None, object, object]:
    shared = fixturedef.scope != 'function'
    _state.shared_fixture_depth += shared
    try:
        return (yield)
    finally:
        _state.shared_fixture_depth -= shared


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    global _state
    outer, _state = _state, _State(test_running=True, generation=next(_generations))
    try:
        return (yield)
    finally:
        _state = outer


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if call.when != 'teardown':
        return report
    open_loops = [loop for loop in _state.loops if not loop.is_closed()]
    leaked = [*(getattr(loop, _CREATOR)[1] for loop in open_loops), *_state.finalized_open]
    for loop in open_loops:
        if not loop.is_running():  # pragma: no branch
            loop.close()
    problems: list[str] = []
    if leaked:
        sites = ''.join(f'\n  {creator}' for creator in leaked)
        problems.append(
            f'{item.nodeid} left {len(leaked)} event loop(s) open, created at:{sites}\n'
            'Close each loop where it is created, as `asyncio.get_event_loop().close()` on the thread that ran '
            '`run_sync()`.'
        )
    if clients := _state.clients_left_open:
        sites = ''.join(f'\n  {site}' for site in clients)
        problems.append(
            f'{item.nodeid} left {len(clients)} HTTP client(s) that an agent run built open, built by:{sites}\n'
            'The run that builds a client must close it, as it closes the model it builds from a name.'
        )
    if generators := _state.finalized_generators:
        sites = ''.join(f'\n  {site}' for site in sorted(set(generators)))
        problems.append(
            f'{item.nodeid} left {len(generators)} async generator(s) for the collector to finalize:{sites}\n'
            'Close each one where its consumer stops, with `aclose()` or `contextlib.aclosing()`.'
        )
    if problems and report.passed:
        report.outcome = 'failed'
        report.longrepr = '\n'.join([*problems, 'See "Leaks" in tests/AGENTS.md.'])
    return report
