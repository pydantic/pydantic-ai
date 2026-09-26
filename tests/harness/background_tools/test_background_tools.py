"""Tests for the `BackgroundTools` capability."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest
from pydantic_ai import Agent, AgentRunResultEvent, CancellationToken, RunCancelled, UsageLimits
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    ToolFailed,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime import RealtimeModel, RealtimeModelProfile, RealtimeModelSettings
from pydantic_ai.realtime.codec import (
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ResponseDone,
    ToolCall,
    ToolResult,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import DeferredToolRequests, RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import BackgroundTools

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _ack_seen(messages: list[ModelMessage]) -> bool:
    """True if any tool return in the history is a background-execution ack."""
    return any(
        isinstance(part, ToolReturnPart) and 'running in background' in str(part.content)
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


def _follow_up_seen(messages: list[ModelMessage], needle: str) -> bool:
    """True if any drained user prompt in the history contains `needle`."""
    return any(
        isinstance(part, UserPromptPart)
        and any(
            needle in item
            for item in ([part.content] if isinstance(part.content, str) else part.content)
            if isinstance(item, str)
        )
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


def _model_calling(tool_name: str, args: str = '{}', ack_callback: Callable[[], None] | None = None) -> FunctionModel:
    """A model that calls `tool_name` once, idles while the task runs, and answers `done` on the follow-up."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _follow_up_seen(messages, f"Background tool '{tool_name}'"):
            return ModelResponse(parts=[TextPart(content='done')])
        if _ack_seen(messages):
            if ack_callback is not None:
                ack_callback()
            return ModelResponse(parts=[TextPart(content='waiting')])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])

    return FunctionModel(model_fn)


class _BackgroundRealtimeConnection(RealtimeConnection):
    def __init__(self) -> None:
        self.sent: list[RealtimeInput] = []
        self.sent_event = asyncio.Event()

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)
        self.sent_event.set()

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        yield ToolCall(tool_call_id='tc1', tool_name='structured', args='{}')
        yield ResponseDone()
        await asyncio.Event().wait()


class _BackgroundRealtimeModel(RealtimeModel):
    def __init__(self, connection: _BackgroundRealtimeConnection) -> None:
        self.connection = connection
        self.messages: Sequence[ModelMessage] | None = None

    @property
    def model_name(self) -> str:
        return 'fake-realtime'

    @property
    def system(self) -> str:
        return 'fake'

    @property
    def profile(self) -> RealtimeModelProfile:
        return RealtimeModelProfile(supports_text_output=True, supports_image_input=True)

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[RealtimeConnection]:
        self.messages = messages
        yield self.connection


class TestBackgroundTools:
    """The metadata-default selector path: spawn, ack, deliver, error, cancel."""

    async def test_metadata_marked_tool_acks_then_delivers_result_as_follow_up(self) -> None:
        release = asyncio.Event()
        ack_seen = asyncio.Event()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _follow_up_seen(messages, "Background tool 'slow_research'"):
                return ModelResponse(parts=[TextPart(content='done')])
            if _ack_seen(messages):
                ack_seen.set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            return ModelResponse(parts=[ToolCallPart(tool_name='slow_research', args='{"query": "topic"}')])

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[BackgroundTools()],
        )

        @agent.tool_plain(metadata={'background': True})
        async def slow_research(query: str) -> str:  # pyright: ignore[reportUnusedFunction]
            await release.wait()
            return f'researched {query}'

        run = asyncio.ensure_future(agent.run('go'))
        await asyncio.wait_for(ack_seen.wait(), timeout=5)
        assert not run.done()
        release.set()
        result = await asyncio.wait_for(run, timeout=5)

        assert result.output == 'done'
        parts = [part for message in result.all_messages() for part in message.parts]
        ack = next(part.content for part in parts if isinstance(part, ToolReturnPart))
        assert isinstance(ack, str)
        match = re.search(r'\(task (?P<task_id>[^)]+)\)', ack)
        assert match is not None
        assert ack == (
            f"Tool 'slow_research' is running in background (task {match['task_id']}). This call is pending. "
            'Do not repeat or poll it; its result will arrive automatically. '
            'Continue independent work, or end your response.'
        )
        follow_up = next(
            part.content
            for part in parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str) and 'Background tool' in part.content
        )
        assert follow_up == (
            f"Background tool 'slow_research' (task {match['task_id']}) completed.\nResult: researched topic"
        )

    async def test_realtime_session_keeps_native_tool_path_and_instructions(self) -> None:
        image = BinaryContent(data=b'image bytes', media_type='image/png')
        connection = _BackgroundRealtimeConnection()
        model = _BackgroundRealtimeModel(connection)
        agent = Agent(capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def structured() -> ToolReturn[str]:  # pyright: ignore[reportUnusedFunction]
            return ToolReturn(return_value='public answer', content=['supporting detail', image])

        async with agent.realtime(model).session() as session:

            async def consume() -> None:
                async for _ in session:  # pragma: no branch
                    pass

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(consume)
                await asyncio.wait_for(connection.sent_event.wait(), timeout=5)

                assert connection.sent == [
                    ToolResult(
                        tool_call_id='tc1',
                        output='public answer',
                        content=['supporting detail', image],
                    )
                ]
                assert model.messages is not None
                initial = next(message for message in model.messages if isinstance(message, ModelRequest))
                assert 'follow-up message' not in (initial.instructions or '')
                task_group.cancel_scope.cancel()

    @pytest.mark.parametrize(
        ('error_factory', 'expected', 'private_detail'),
        [
            (lambda: ModelRetry('use different arguments'), 'failed: use different arguments', None),
            (lambda: ToolFailed('service unavailable'), 'failed: service unavailable', None),
            (
                lambda: CallDeferred(metadata={'private_job_id': 'secret'}),
                'failed: CallDeferred was raised; background tools cannot defer a running task.',
                'secret',
            ),
            (lambda: ModelRetry(''), 'failed: ToolRetryError', None),
        ],
        ids=['retry', 'tool-failed', 'deferred', 'empty-retry'],
    )
    async def test_tool_signalled_errors_have_readable_follow_ups(
        self, error_factory: Callable[[], Exception], expected: str, private_detail: str | None
    ) -> None:
        release = asyncio.Event()
        agent = Agent(_model_calling('broken', ack_callback=release.set), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def broken() -> str:  # pyright: ignore[reportUnusedFunction]
            await release.wait()
            raise error_factory()

        result = await agent.run('go')

        assert _follow_up_seen(result.all_messages(), expected)
        if private_detail is not None:
            assert not _follow_up_seen(result.all_messages(), private_detail)

    async def test_retry_exhaustion_terminates_run(self) -> None:
        agent = Agent(_model_calling('broken'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True}, retries=0)
        async def broken() -> str:  # pyright: ignore[reportUnusedFunction]
            raise ModelRetry('try again')

        with pytest.raises(UnexpectedModelBehavior, match="Tool 'broken' exceeded max retries count of 0"):
            await agent.run('go')

    async def test_completed_background_call_is_counted_once(self) -> None:
        agent = Agent(_model_calling('slow'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'value'

        result = await agent.run('go')

        assert result.usage.tool_calls == 1

    async def test_pending_background_call_counts_toward_tool_call_limit(self) -> None:
        started = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='slow', args='{}')])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            nonlocal started
            started += 1
            await asyncio.Event().wait()
            return 'unreachable'  # pragma: no cover -- the run cancels the pending call

        with pytest.raises(UsageLimitExceeded):
            await agent.run('go', usage_limits=UsageLimits(tool_calls_limit=1))

        assert started == 1

    async def test_pending_background_call_leaves_capacity_for_another_call(self) -> None:
        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _follow_up_seen(messages, "Background tool 'slow'"):
                return ModelResponse(parts=[TextPart(content='done')])
            if _ack_seen(messages):
                return ModelResponse(parts=[ToolCallPart(tool_name='ordinary', args='{}')])
            return ModelResponse(parts=[ToolCallPart(tool_name='slow', args='{}')])

        release = asyncio.Event()
        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            await release.wait()
            return 'slow result'

        @agent.tool_plain
        async def ordinary() -> str:  # pyright: ignore[reportUnusedFunction]
            release.set()
            return 'ordinary result'

        result = await agent.run('go', usage_limits=UsageLimits(tool_calls_limit=2))

        assert result.usage.tool_calls == 2

    async def test_sequential_tool_stays_on_normal_execution_path(self) -> None:
        sequential_active = False
        overlapped = False

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _follow_up_seen(messages, "Background tool 'later'"):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='barrier', args='{}'),
                    ToolCallPart(tool_name='later', args='{}'),
                ]
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True}, sequential=True)
        async def barrier() -> str:  # pyright: ignore[reportUnusedFunction]
            nonlocal sequential_active
            sequential_active = True
            await asyncio.sleep(0)
            sequential_active = False
            return 'barrier result'

        @agent.tool_plain(metadata={'background': True})
        async def later() -> str:  # pyright: ignore[reportUnusedFunction]
            nonlocal overlapped
            overlapped = sequential_active
            return 'later result'

        result = await agent.run('go')

        assert result.output == 'done'
        assert not overlapped
        barrier_return = next(
            part
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'barrier'
        )
        assert barrier_return.content == 'barrier result'

    async def test_unexpected_error_reports_type_without_exposing_details_to_model(self) -> None:
        agent = Agent(_model_calling('broken'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def broken() -> str:  # pyright: ignore[reportUnusedFunction]
            raise RuntimeError('private backend detail')

        result = await agent.run('go')

        assert _follow_up_seen(result.all_messages(), 'failed: RuntimeError')
        assert not _follow_up_seen(result.all_messages(), 'private backend detail')

    async def test_run_stream_waits_for_live_task_then_drops_its_result(self) -> None:
        started = asyncio.Event()
        response_streamed = asyncio.Event()
        release = asyncio.Event()

        async def stream_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            if _ack_seen(messages):
                yield 'final answer'
                response_streamed.set()
            else:
                yield {0: DeltaToolCall(name='slow', json_args='{}')}

        agent = Agent(FunctionModel(stream_function=stream_model), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            started.set()
            await release.wait()
            return 'late result'

        async with agent.run_stream('go') as stream_result:
            output = asyncio.ensure_future(stream_result.get_output())
            await asyncio.wait_for(started.wait(), timeout=5)
            await asyncio.wait_for(response_streamed.wait(), timeout=5)
            assert not output.done()
            release.set()
            assert await asyncio.wait_for(output, timeout=5) == 'final answer'

        assert not _follow_up_seen(stream_result.all_messages(), 'late result')

    async def test_run_stream_events_delivers_background_result(self) -> None:
        release = asyncio.Event()

        async def model_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            if _follow_up_seen(messages, "Background tool 'slow'"):
                yield 'done'
            elif _ack_seen(messages):
                release.set()
                yield 'waiting'
            else:
                yield {0: DeltaToolCall(name='slow', json_args='{}')}

        agent = Agent(FunctionModel(stream_function=model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            await release.wait()
            return 'value'

        output = None
        async with agent.run_stream_events('go') as events:
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    output = event.result.output

        assert output == 'done'

    async def test_iter_delivers_background_result(self) -> None:
        agent = Agent(_model_calling('slow'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'value'

        async with agent.iter('go') as run:
            async for _ in run:
                pass

        result = run.result
        assert result is not None and result.output == 'done'

    async def test_unmarked_tool_runs_normally(self) -> None:
        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            returned = any(
                isinstance(part, ToolReturnPart) and part.content == 'sync result'
                for msg in messages
                if isinstance(msg, ModelRequest)
                for part in msg.parts
            )
            if returned:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='plain', args='{}')])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain
        def plain() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'sync result'

        result = await agent.run('go')

        assert result.output == 'done'
        assert not _ack_seen(result.all_messages())
        assert not _follow_up_seen(result.all_messages(), 'Background tool')

    async def test_run_scoped_sequential_mode_uses_normal_execution(self) -> None:
        agent = Agent(TestModel(), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def selected() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'normal result'

        with ToolManager.parallel_execution_mode('sequential'):
            result = await agent.run('go')

        assert any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'selected' and part.content == 'normal result'
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
        )

    async def test_sync_tool_can_enqueue_from_its_worker_thread(self) -> None:
        agent = Agent(_model_calling('sync_bg'), capabilities=[BackgroundTools()])

        @agent.tool(metadata={'background': True})
        def sync_bg(ctx: RunContext[object]) -> str:  # pyright: ignore[reportUnusedFunction]
            ctx.enqueue('message from sync background tool')
            return 'sync result'

        result = await agent.run('go')

        assert _follow_up_seen(result.all_messages(), 'message from sync background tool')

    async def test_multiple_capabilities_combine_selectors_without_double_execution(self) -> None:
        calls = {'first': 0, 'shared': 0, 'plain': 0}
        release = asyncio.Event()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _follow_up_seen(messages, 'first result') and _follow_up_seen(messages, 'shared result'):
                return ModelResponse(parts=[TextPart(content='done')])
            if _ack_seen(messages):
                release.set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='first', args='{}'),
                    ToolCallPart(tool_name='shared', args='{}'),
                    ToolCallPart(tool_name='plain', args='{}'),
                ]
            )

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[BackgroundTools(tools=['first', 'shared']), BackgroundTools(tools=['shared'])],
        )

        @agent.tool_plain
        async def first() -> str:  # pyright: ignore[reportUnusedFunction]
            calls['first'] += 1
            await release.wait()
            return 'first result'

        @agent.tool_plain
        async def shared() -> str:  # pyright: ignore[reportUnusedFunction]
            calls['shared'] += 1
            await release.wait()
            return 'shared result'

        @agent.tool_plain
        async def plain() -> str:  # pyright: ignore[reportUnusedFunction]
            calls['plain'] += 1
            return 'plain result'

        result = await agent.run('go')

        assert calls == {'first': 1, 'shared': 1, 'plain': 1}
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: first result')
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: shared result')

    async def test_background_tool_uses_context_cancel_to_stop_the_run(self) -> None:
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        first_cancelled = asyncio.Event()
        second_cancelled = asyncio.Event()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='stop', args='{}'),
                    ToolCallPart(tool_name='first', args='{}'),
                    ToolCallPart(tool_name='second', args='{}'),
                ]
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool(metadata={'background': True})
        async def stop(ctx: RunContext[object]) -> str:  # pyright: ignore[reportUnusedFunction]
            await first_started.wait()
            await second_started.wait()
            ctx.cancel()
            await asyncio.sleep(0)
            return 'discarded'  # pragma: no cover -- cancellation is delivered at the await

        @agent.tool_plain(metadata={'background': True})
        async def first() -> str:  # pyright: ignore[reportUnusedFunction]
            first_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_cancelled.set()
                raise
            return 'unreachable'  # pragma: no cover

        @agent.tool_plain(metadata={'background': True})
        async def second() -> str:  # pyright: ignore[reportUnusedFunction]
            second_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                second_cancelled.set()
                raise
            return 'unreachable'  # pragma: no cover

        with pytest.raises(RunCancelled):
            await agent.run('go')

        assert first_cancelled.is_set()
        assert second_cancelled.is_set()

    async def test_tool_return_preserves_model_content_without_exposing_metadata(self) -> None:
        image = BinaryContent(data=b'image bytes', media_type='image/png')
        agent = Agent(_model_calling('structured'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def structured() -> ToolReturn[str]:  # pyright: ignore[reportUnusedFunction]
            return ToolReturn(
                return_value='public answer',
                content=['supporting image', image],
                metadata={'api_key': 'secret application metadata'},
                tools=['deferred_tool'],
            )

        result = await agent.run('go')

        follow_up = next(
            part
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart) and 'Background tool' in str(part.content)
        )
        assert isinstance(follow_up.content, list)
        assert len(follow_up.content) == 3
        assert isinstance(follow_up.content[0], str)
        assert follow_up.content[0].startswith("Background tool 'structured' (task ")
        assert follow_up.content[0].endswith(') completed.\nResult: public answer')
        assert follow_up.content[1:] == ['supporting image', image]
        assert 'secret application metadata' not in str(follow_up.content)
        assert 'deferred_tool' not in str(follow_up.content)

    async def test_tool_return_preserves_string_content(self) -> None:
        agent = Agent(_model_calling('structured'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def structured() -> ToolReturn[str]:  # pyright: ignore[reportUnusedFunction]
            return ToolReturn(return_value='public answer', content='supporting detail')

        result = await agent.run('go')

        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: public answer\nsupporting detail')

    async def test_structured_return_value_uses_core_model_serialization(self) -> None:
        agent = Agent(_model_calling('structured'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def structured() -> list[int]:  # pyright: ignore[reportUnusedFunction]
            return [1, 2]

        result = await agent.run('go')

        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: [1,2]')

    async def test_instructions_explain_pending_calls(self) -> None:
        seen: list[str | None] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.instructions)
            return ModelResponse(parts=[TextPart(content='done')])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])
        await agent.run('go')

        assert seen == [
            'Some tools reply that a call is running in the background. This means that exact call is pending. '
            'Do not repeat or poll it. Its result will arrive automatically in a later message. Continue only '
            'with independent work. If none remains, end your response; the run will resume when the result arrives.'
        ]

    async def test_concurrent_runs_do_not_share_tasks(self) -> None:
        release = {'first': asyncio.Event(), 'second': asyncio.Event()}
        started = {'first': asyncio.Event(), 'second': asyncio.Event()}

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            first_part = messages[0].parts[-1] if isinstance(messages[0], ModelRequest) else None
            run_name = first_part.content if isinstance(first_part, UserPromptPart) else ''
            if _follow_up_seen(messages, 'completed'):
                return ModelResponse(parts=[TextPart(content='done')])
            if _ack_seen(messages):
                return ModelResponse(parts=[TextPart(content='waiting')])
            return ModelResponse(parts=[ToolCallPart(tool_name='waiter', args=f'{{"name": "{run_name}"}}')])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def waiter(name: str) -> str:  # pyright: ignore[reportUnusedFunction]
            started[name].set()
            await release[name].wait()
            return name

        async def run_and_check(name: str) -> None:
            result = await agent.run(name)
            assert result.output == 'done'
            assert _follow_up_seen(result.all_messages(), f'completed.\nResult: {name}')

        first = asyncio.ensure_future(run_and_check('first'))
        second = asyncio.ensure_future(run_and_check('second'))
        await asyncio.wait_for(started['first'].wait(), timeout=5)
        await asyncio.wait_for(started['second'].wait(), timeout=5)

        # Finish the first run completely while the second's task is still live;
        # shared state would let the first run's cleanup cancel the second's task.
        release['first'].set()
        await asyncio.wait_for(first, timeout=5)
        release['second'].set()
        await asyncio.wait_for(second, timeout=5)

    async def test_deferred_tool_pause_is_not_held_behind_background_tasks(self) -> None:
        cancel_seen = asyncio.Event()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='slow', args='{}'),
                        ToolCallPart(tool_name='needs_approval', args='{}'),
                    ]
                )
            # The pause must end the run without another model turn.
            return ModelResponse(parts=[TextPart(content='unreachable')])  # pragma: no cover

        agent = Agent(
            FunctionModel(model_fn),
            output_type=[str, DeferredToolRequests],
            capabilities=[BackgroundTools()],
        )

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_seen.set()
                raise
            return 'never'  # pragma: no cover -- task is cancelled when the run pauses

        @agent.tool_plain(requires_approval=True)
        def needs_approval() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'approved'  # pragma: no cover -- never approved in this test

        result = await asyncio.wait_for(agent.run('go'), timeout=1)

        assert isinstance(result.output, DeferredToolRequests)
        await asyncio.wait_for(cancel_seen.wait(), timeout=1)

    async def test_deferred_tool_pause_wins_over_a_completed_background_result(self) -> None:
        fast_done = asyncio.Event()
        model_calls = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_calls
            model_calls += 1
            return ModelResponse(
                parts=[ToolCallPart(tool_name='fast', args='{}'), ToolCallPart(tool_name='needs_approval', args='{}')]
            )

        agent = Agent(
            FunctionModel(model_fn),
            output_type=[str, DeferredToolRequests],
            capabilities=[BackgroundTools()],
        )

        @agent.tool_plain(metadata={'background': True})
        async def fast() -> str:  # pyright: ignore[reportUnusedFunction]
            fast_done.set()
            return 'finished'

        @agent.tool_plain
        async def needs_approval() -> str:  # pyright: ignore[reportUnusedFunction]
            await fast_done.wait()
            raise ApprovalRequired

        result = await agent.run('go')

        assert isinstance(result.output, DeferredToolRequests)
        assert result.output.approvals[0].tool_name == 'needs_approval'
        assert model_calls == 1

    async def test_queued_result_is_not_delayed_by_an_unrelated_live_task(self) -> None:
        release = {'fast_bg': asyncio.Event(), 'slow_bg': asyncio.Event()}

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _follow_up_seen(messages, 'fast value') and _follow_up_seen(messages, 'slow value'):
                return ModelResponse(parts=[TextPart(content='done')])
            if _follow_up_seen(messages, 'fast value'):
                # The fast result arrived without waiting on `slow_bg`; let it finish now.
                release['slow_bg'].set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            if _ack_seen(messages):
                release['fast_bg'].set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            return ModelResponse(
                parts=[ToolCallPart(tool_name='fast_bg', args='{}'), ToolCallPart(tool_name='slow_bg', args='{}')]
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def fast_bg() -> str:  # pyright: ignore[reportUnusedFunction]
            await release['fast_bg'].wait()
            return 'fast value'

        @agent.tool_plain(metadata={'background': True})
        async def slow_bg() -> str:  # pyright: ignore[reportUnusedFunction]
            await release['slow_bg'].wait()
            return 'slow value'

        result = await asyncio.wait_for(agent.run('go'), timeout=5)

        assert result.output == 'done'
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: fast value')
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: slow value')

    async def test_failed_background_tool_does_not_cancel_its_sibling(self) -> None:
        release_broken = asyncio.Event()
        release_slow = asyncio.Event()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _follow_up_seen(messages, 'failed: RuntimeError') and _follow_up_seen(messages, 'slow value'):
                return ModelResponse(parts=[TextPart(content='done')])
            if _follow_up_seen(messages, 'failed: RuntimeError'):
                release_slow.set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            if _ack_seen(messages):
                release_broken.set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            return ModelResponse(
                parts=[ToolCallPart(tool_name='broken', args='{}'), ToolCallPart(tool_name='slow', args='{}')]
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def broken() -> str:  # pyright: ignore[reportUnusedFunction]
            await release_broken.wait()
            raise RuntimeError('private detail')

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            await release_slow.wait()
            return 'slow value'

        result = await asyncio.wait_for(agent.run('go'), timeout=5)

        assert result.output == 'done'
        assert _follow_up_seen(result.all_messages(), 'failed: RuntimeError')
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: slow value')

    async def test_background_tool_ending_in_cancelled_error_cancels_the_run(self) -> None:
        release = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _ack_seen(messages):
                release.set()
                return ModelResponse(parts=[TextPart(content='waiting')])
            return ModelResponse(
                parts=[ToolCallPart(tool_name='cancelled', args='{}'), ToolCallPart(tool_name='slow', args='{}')]
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def cancelled() -> str:  # pyright: ignore[reportUnusedFunction]
            await release.wait()
            raise asyncio.CancelledError

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise
            return 'unreachable'  # pragma: no cover

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(agent.run('go'), timeout=5)
        assert sibling_cancelled.is_set()

    async def test_background_base_exception_stops_the_run(self) -> None:
        class FatalBackgroundError(BaseException):
            pass

        slow_started = asyncio.Event()
        slow_cancelled = asyncio.Event()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='fatal', args='{}'), ToolCallPart(tool_name='slow', args='{}')]
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def fatal() -> str:  # pyright: ignore[reportUnusedFunction]
            await slow_started.wait()
            raise FatalBackgroundError

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            slow_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                slow_cancelled.set()
                raise
            return 'unreachable'  # pragma: no cover

        with pytest.raises(FatalBackgroundError):
            await asyncio.wait_for(agent.run('go'), timeout=5)
        await asyncio.wait_for(slow_cancelled.wait(), timeout=5)

    async def test_pre_start_cancellation_releases_reserved_tool_usage(self) -> None:
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), pending_messages=[])
        capability = await BackgroundTools().for_run(ctx)
        handler_started = False

        async def tool_handler(args: dict[str, Any]) -> str:
            nonlocal handler_started  # pragma: no cover -- the task is cancelled before its first step
            handler_started = True  # pragma: no cover
            return 'unreachable'  # pragma: no cover

        async def run_handler() -> AgentRunResult[str]:
            await capability.wrap_tool_execute(
                ctx,
                call=ToolCallPart(tool_name='slow', args='{}'),
                tool_def=ToolDefinition(name='slow', metadata={'background': True}),
                args={},
                handler=tool_handler,
            )
            return AgentRunResult('done')

        result = await capability.wrap_run(ctx, handler=run_handler)

        assert result.output == 'done'
        assert not handler_started
        assert ctx.usage.tool_calls == 0

    async def test_fatal_error_during_run_teardown_is_propagated(self) -> None:
        class FatalBackgroundError(BaseException):
            pass

        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), pending_messages=[])
        capability = await BackgroundTools().for_run(ctx)
        raised = asyncio.Event()

        async def tool_handler(args: dict[str, Any]) -> str:
            raised.set()
            raise FatalBackgroundError

        async def run_handler() -> AgentRunResult[str]:
            await capability.wrap_tool_execute(
                ctx,
                call=ToolCallPart(tool_name='fatal', args='{}'),
                tool_def=ToolDefinition(name='fatal', metadata={'background': True}),
                args={},
                handler=tool_handler,
            )
            await raised.wait()
            return AgentRunResult('done')

        with pytest.raises(FatalBackgroundError):
            await capability.wrap_run(ctx, handler=run_handler)

    async def test_run_abort_waits_for_background_task_cancellation_cleanup(self) -> None:
        started = asyncio.Event()
        cancellation_seen = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()
        agent = Agent(_model_calling('stubborn'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def stubborn() -> str:  # pyright: ignore[reportUnusedFunction]
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError('unreachable')  # pragma: no cover
            except asyncio.CancelledError:
                cancellation_seen.set()
                with anyio.CancelScope(shield=True):
                    await release.wait()
                finished.set()
                raise

        run_returned = asyncio.Event()

        async def run() -> None:
            try:
                await agent.run('go')
            finally:
                run_returned.set()

        async def release_after_a_while() -> None:
            await asyncio.sleep(0.3)
            assert not run_returned.is_set()
            release.set()

        releaser = asyncio.ensure_future(release_after_a_while())
        async with anyio.create_task_group() as tg:
            tg.start_soon(run)
            await asyncio.wait_for(started.wait(), timeout=5)
            tg.cancel_scope.cancel()

        await asyncio.wait_for(cancellation_seen.wait(), timeout=5)
        await asyncio.wait_for(releaser, timeout=5)
        assert finished.is_set()

    async def test_run_abort_waits_for_default_sync_worker(self) -> None:
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        agent = Agent(_model_calling('blocking'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        def blocking() -> str:  # pyright: ignore[reportUnusedFunction]
            started.set()
            assert release.wait(timeout=5)
            finished.set()
            return 'late result'

        run = asyncio.ensure_future(agent.run('go'))
        assert await asyncio.to_thread(started.wait, 5)
        run.cancel()

        await asyncio.sleep(0.3)
        assert not run.done()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, timeout=5)
        assert finished.is_set()

    async def test_cancellation_token_cancels_live_tasks(self) -> None:
        cancel_seen = asyncio.Event()
        started = asyncio.Event()
        token = CancellationToken()
        agent = Agent(_model_calling('slow'), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancel_seen.set()
                raise
            return 'never'  # pragma: no cover -- task is cancelled before completing

        run = asyncio.ensure_future(agent.run('go', cancellation_token=token))
        await asyncio.wait_for(started.wait(), timeout=5)
        token.cancel()

        with pytest.raises(RunCancelled):
            await asyncio.wait_for(run, timeout=5)
        await asyncio.wait_for(cancel_seen.wait(), timeout=5)

    async def test_constructs_from_agent_spec(self) -> None:
        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'BackgroundTools': {}}]},
            custom_capability_types=[BackgroundTools],
        )

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'value'

        result = await agent.run('go')

        assert _ack_seen(result.all_messages())

    async def test_only_optional_tools_advertise_run_in_background(self) -> None:
        seen: list[ToolDefinition] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.extend(info.function_tools)
            return ModelResponse(parts=[TextPart(content='done')])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def always() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'always'  # pragma: no cover

        @agent.tool_plain(metadata={'background': 'optional'})
        async def optional() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'optional'  # pragma: no cover

        @agent.tool_plain
        async def normal() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'normal'  # pragma: no cover

        await agent.run('go')

        properties = {tool.name: tool.parameters_json_schema['properties'] for tool in seen}
        assert 'run_in_background' in properties['optional']
        assert 'run_in_background' not in properties['always']
        assert 'run_in_background' not in properties['normal']

    async def test_model_opting_in_runs_tool_in_background(self) -> None:
        agent = Agent(
            _model_calling('slow_research', '{"query": "topic", "run_in_background": true}'),
            capabilities=[BackgroundTools()],
        )

        @agent.tool_plain(metadata={'background': 'optional'})
        async def slow_research(query: str) -> str:  # pyright: ignore[reportUnusedFunction]
            return f'researched {query}'

        result = await agent.run('go')

        assert _ack_seen(result.all_messages())
        assert _follow_up_seen(result.all_messages(), 'completed.\nResult: researched topic')

    async def test_callable_selector_decision_stays_stable_for_each_model_response(self) -> None:
        selector_calls = 0

        def changing_selector(ctx: RunContext[object], tool_def: ToolDefinition) -> bool:
            nonlocal selector_calls
            selector_calls += 1
            return selector_calls > 1

        agent = Agent(
            _model_calling('research', '{"run_in_background": true}'),
            capabilities=[BackgroundTools(tools=changing_selector)],
        )

        @agent.tool_plain(metadata={'background': 'optional'})
        async def research() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'researched'

        result = await agent.run('go')

        assert result.output == 'done'

    async def test_model_omitting_flag_runs_tool_inline(self) -> None:
        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='research', args={'query': 'topic'})])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': 'optional'})
        async def research(query: str) -> str:  # pyright: ignore[reportUnusedFunction]
            return f'researched {query}'

        result = await agent.run('go')

        assert not _ack_seen(result.all_messages())
        assert any(
            isinstance(part, ToolReturnPart) and part.content == 'researched topic'
            for message in result.all_messages()
            for part in message.parts
        )

    async def test_null_background_flag_runs_tool_inline(self) -> None:
        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='research', args={'run_in_background': None})])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': 'optional'}, strict=True)
        async def research() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'researched'

        result = await agent.run('go')

        assert result.output == 'done'

    async def test_always_background_selector_wins_over_optional_metadata(self) -> None:
        seen: list[ToolDefinition] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.extend(info.function_tools)
            if _ack_seen(messages):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='research', args='{}')])

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[BackgroundTools(tools=['research'])],
        )

        @agent.tool_plain(metadata={'background': 'optional'})
        async def research() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'researched'

        result = await agent.run('go')

        research_tool = next(tool for tool in seen if tool.name == 'research')
        assert 'run_in_background' not in research_tool.parameters_json_schema['properties']
        assert _ack_seen(result.all_messages())

    async def test_optional_tool_with_existing_run_in_background_parameter_is_rejected(self) -> None:
        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='research', args='{}')])  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': 'optional'})
        async def research(run_in_background: bool) -> str:  # pyright: ignore[reportUnusedFunction]
            return str(run_in_background)  # pragma: no cover

        with pytest.raises(UserError, match='already has a'):
            await agent.run('go')

    @pytest.mark.parametrize(
        'args', ['{', '[]', '{"run_in_background": "yes"}'], ids=['not-json', 'not-object', 'flag-not-boolean']
    )
    async def test_invalid_args_for_optional_tool_are_retried(self, args: str) -> None:
        retry_seen = False

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal retry_seen
            retry_seen = retry_seen or any(
                isinstance(part, RetryPromptPart) for message in messages for part in message.parts
            )
            if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[TextPart(content='researched')])
            if retry_seen:
                return ModelResponse(parts=[ToolCallPart(tool_name='research', args='{}')])
            return ModelResponse(parts=[ToolCallPart(tool_name='research', args=args)])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': 'optional'})
        async def research() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'researched'

        await agent.run('go')

        assert retry_seen

    async def test_run_scoped_sequential_mode_does_not_advertise_run_in_background(self) -> None:
        seen: list[ToolDefinition] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.extend(info.function_tools)
            return ModelResponse(parts=[TextPart(content='done')])

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': 'optional'})
        async def research() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'researched'  # pragma: no cover

        with ToolManager.parallel_execution_mode('sequential'):
            await agent.run('go')

        assert 'run_in_background' not in seen[0].parameters_json_schema['properties']
