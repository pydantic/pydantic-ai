"""Node-lifecycle hook behavior when the run is driven by `run_stream()`.

Split out of `test_capabilities.py` (which sits at pre-commit's large-file limit): these tests
pin the documented `run_stream()` exception to wrap-outermost node ordering — `before_node_run`
fires pre-stream for streamed nodes — together with its error, short-circuit, and replacement arms.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_graph import End

from ._inline_snapshot import snapshot
from .capability_models import (
    make_text_response,
    simple_model_function,
    simple_stream_function,
    tool_calling_model,
    tool_calling_stream_function,
)

pytestmark = [
    pytest.mark.anyio,
]


@dataclass
class _ReplacingCapability(AbstractCapability[Any]):
    """Capability that replaces ModelRequestNode with a fresh copy in before_node_run.

    Used to test that streaming + node replacement doesn't cause double model execution.
    """

    replaced: bool = field(default=False, init=False)

    async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
        from pydantic_ai import ModelRequestNode

        if isinstance(node, ModelRequestNode) and not self.replaced:
            self.replaced = True
            return ModelRequestNode(request=node.request)  # pyright: ignore[reportUnknownVariableType]
        return node  # pyright: ignore[reportUnknownVariableType]


class TestNodeStreamingWithHooks:
    """Tests that node streaming with event_stream_handler doesn't cause double model execution
    when before_node_run replaces a node."""

    async def test_run_stream_on_node_run_error_recovery_syncs_graph_state(self):
        """`run_stream()` node recovery: `on_node_run_error` returning `End` ends the run with
        the recovery result, keeping the graph runner's state in sync."""
        from pydantic_ai.result import FinalResult
        from pydantic_graph import End

        @dataclass
        class RecoverStreamingNodeCap(AbstractCapability[Any]):
            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                raise RuntimeError('node wrapper exploded')

            async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: BaseException) -> Any:
                return End(FinalResult(output='recovered'))

        agent = Agent(FunctionModel(simple_model_function), capabilities=[RecoverStreamingNodeCap()])
        async with agent.run_stream('hello') as result:
            output = await result.get_output()
        assert output == 'recovered'

    async def test_run_stream_after_node_run_result_change_syncs_graph_state(self):
        """`run_stream()`: `after_node_run` converting the advanced result to `End` ends the run
        with the converted result, keeping the graph runner's state in sync."""
        from pydantic_ai.result import FinalResult
        from pydantic_graph import End

        model_called = False

        @dataclass
        class EndAfterFirstAdvanceCap(AbstractCapability[Any]):
            async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
                # The run ends on the swapped `End`, so this hook only sees the first advance.
                assert Agent.is_model_request_node(result)
                return End(FinalResult(output='cut short'))

        def recording_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
            nonlocal model_called
            model_called = True
            return make_text_response('model output')

        agent = Agent(FunctionModel(recording_model), capabilities=[EndAfterFirstAdvanceCap()])
        async with agent.run_stream('hello') as result:
            output = await result.get_output()
        assert output == 'cut short'
        assert not model_called

    async def test_before_node_run_replacement_no_double_execution(self):
        """When before_node_run replaces a ModelRequestNode and event_stream_handler is set,
        the model should be called exactly once (not twice)."""
        model_call_count = 0

        async def counting_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            nonlocal model_call_count
            model_call_count += 1
            yield 'streamed response'

        cap = _ReplacingCapability()
        agent = Agent(FunctionModel(simple_model_function, stream_function=counting_stream), capabilities=[cap])

        events_received: list[AgentStreamEvent] = []

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                events_received.append(event)

        result = await agent.run('hello', event_stream_handler=handler)
        assert result.output == 'streamed response'
        assert model_call_count == 1, f'Model was called {model_call_count} times, expected 1'
        assert len(events_received) > 0

    async def test_hook_ordering_with_event_stream_handler(self):
        """`agent.run()` keeps the full lifecycle inside the wrapper while streaming events.

        The documented exception applies only to `run_stream()`, where the caller regains control
        mid-node and `before_node_run` must fire before streaming.
        """
        log: list[str] = []

        @dataclass
        class OrderTrackingCapability(AbstractCapability[Any]):
            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                log.append(f'before:{type(node).__name__}')
                return node

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                log.append(f'wrap:enter:{type(node).__name__}')
                result = await handler(node)
                log.append(f'wrap:exit:{type(node).__name__}')
                return result

            async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
                log.append(f'after:{type(node).__name__}')
                return result

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[OrderTrackingCapability()],
        )

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass
            log.append('stream:consumed')

        await agent.run('hello', event_stream_handler=handler)

        # `agent.run()` keeps streaming inside the full wrap-outermost lifecycle.
        mr_before = log.index('before:ModelRequestNode')
        mr_wrap_enter = log.index('wrap:enter:ModelRequestNode')
        stream_consumed_idx = log.index('stream:consumed')
        mr_wrap_exit = log.index('wrap:exit:ModelRequestNode')
        mr_after = log.index('after:ModelRequestNode')
        assert mr_wrap_enter < mr_before < stream_consumed_idx < mr_after < mr_wrap_exit

    async def test_run_stream_before_node_run_replacement_no_double_execution(self):
        """Same as the run() test but for run_stream(): before_node_run replacement
        should not cause double model execution."""
        model_call_count = 0

        async def counting_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            nonlocal model_call_count
            model_call_count += 1
            yield 'streamed response'

        cap = _ReplacingCapability()
        agent = Agent(FunctionModel(simple_model_function, stream_function=counting_stream), capabilities=[cap])

        async with agent.run_stream('hello') as streamed:
            output = await streamed.get_output()

        assert output == 'streamed response'
        assert model_call_count == 1, f'Model was called {model_call_count} times, expected 1'

    async def test_run_stream_skips_wrap_and_after_for_the_final_model_request(self):
        """`run_stream()` hands back the result mid-stream, so the final `ModelRequestNode` only gets `before_node_run`.

        Pinning the documented exception to "node hooks fire however the run is driven": that node's
        `wrap_node_run`/`after_node_run` are deliberately skipped, while the `SetFinalResult` node
        that ends the run gets the full lifecycle.
        """
        log: list[str] = []

        @dataclass
        class NodeHookCap(AbstractCapability[Any]):
            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                log.append(f'before:{type(node).__name__}')
                return node

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                log.append(f'wrap:{type(node).__name__}')
                return await handler(node)

            async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
                log.append(f'after:{type(node).__name__}')
                return result

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[NodeHookCap()],
        )

        async with agent.run_stream('hello') as streamed:
            await streamed.get_output()

        assert log == snapshot(
            [
                'before:UserPromptNode',
                'wrap:UserPromptNode',
                'after:UserPromptNode',
                'before:ModelRequestNode',
                'wrap:SetFinalResult',
                'before:SetFinalResult',
                'after:SetFinalResult',
            ]
        )

    async def test_on_node_run_error_fires_in_run_stream(self):
        """on_node_run_error in run_stream() fires when wrap_node_run raises during graph advancement."""
        error_log: list[str] = []

        @dataclass
        class WrapErrorCap(AbstractCapability[Any]):
            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                # Raise on CallToolsNode — after UserPromptNode and ModelRequestNode pass through.
                # ModelRequestNode with tool calls doesn't produce a FinalResultEvent in run_stream(),
                # so it falls through to wrap_node_run; CallToolsNode is next and triggers the error.
                from pydantic_ai._agent_graph import CallToolsNode

                if isinstance(node, CallToolsNode):
                    raise RuntimeError('wrap error')
                return await handler(node)

            async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
                error_log.append(type(node).__name__)
                raise error

        agent = Agent(
            FunctionModel(tool_calling_model, stream_function=tool_calling_stream_function),
            capabilities=[WrapErrorCap()],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        with pytest.raises(RuntimeError, match='wrap error'):
            async with agent.run_stream('hello') as _streamed:
                pass

        assert error_log == ['CallToolsNode']

    async def test_on_node_run_error_recovery_updates_run_stream_result(self):
        from pydantic_ai._agent_graph import CallToolsNode
        from pydantic_ai.result import FinalResult

        @dataclass
        class RecoverWrapErrorCap(AbstractCapability[Any]):
            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                if isinstance(node, CallToolsNode):
                    raise RuntimeError('wrap error')
                return await handler(node)

            async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
                assert isinstance(node, CallToolsNode)
                assert str(error) == 'wrap error'
                return End(FinalResult(output='recovered'))

        agent = Agent(
            FunctionModel(tool_calling_model, stream_function=tool_calling_stream_function),
            capabilities=[RecoverWrapErrorCap()],
        )

        @agent.tool_plain
        def my_tool() -> str:
            # Runs while `CallToolsNode` streams its events; `wrap_node_run` only raises
            # afterwards, when the node advances.
            return 'tool result'

        async with agent.run_stream('hello') as streamed:
            output = await streamed.get_output()

        assert output == 'recovered'

    async def test_wrap_node_run_short_circuit_updates_run_stream_result(self):
        """A `wrap_node_run` short-circuit during `run_stream()` graph advancement syncs the graph.

        The wrapper returns `End` without calling its handler, so the graph runner is still
        pending on the short-circuited node and must be overridden to reflect the hook's outcome."""
        from pydantic_ai._agent_graph import CallToolsNode
        from pydantic_ai.result import FinalResult

        @dataclass
        class ShortCircuitCap(AbstractCapability[Any]):
            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                if isinstance(node, CallToolsNode):
                    return End(FinalResult(output='short-circuited'))
                return await handler(node)

        agent = Agent(
            FunctionModel(tool_calling_model, stream_function=tool_calling_stream_function),
            capabilities=[ShortCircuitCap()],
        )

        @agent.tool_plain
        def my_tool() -> str:
            # Runs while `CallToolsNode` streams its events; the wrapper only short-circuits
            # afterwards, when the node advances.
            return 'tool result'

        async with agent.run_stream('hello') as streamed:
            output = await streamed.get_output()

        assert output == 'short-circuited'

    async def test_after_node_run_replacement_updates_run_stream_result(self):
        from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
        from pydantic_ai.result import FinalResult

        tool_call_count = 0

        @dataclass
        class ReplaceAfterNodeCap(AbstractCapability[Any]):
            async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
                if isinstance(node, ModelRequestNode) and isinstance(result, CallToolsNode):
                    return End(FinalResult(output='replaced'))
                return result

        agent = Agent(
            FunctionModel(tool_calling_model, stream_function=tool_calling_stream_function),
            capabilities=[ReplaceAfterNodeCap()],
        )

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1  # pragma: no cover
            return 'tool result'  # pragma: no cover

        async with agent.run_stream('hello') as streamed:
            output = await streamed.get_output()

        assert output == 'replaced'
        assert tool_call_count == 0
