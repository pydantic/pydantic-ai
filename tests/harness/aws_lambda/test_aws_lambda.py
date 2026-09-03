"""Tests for the AWS Lambda durability capability."""

from __future__ import annotations

import asyncio
import gc
import re
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aws_durable_execution_sdk_python import durable_execution  # pyright: ignore[reportUnknownVariableType]
from aws_durable_execution_sdk_python.config import StepConfig, StepSemantics
from aws_durable_execution_sdk_python.exceptions import ExecutionError
from aws_durable_execution_sdk_python.retries import RetryPresets
from aws_durable_execution_sdk_python.serdes import DEFAULT_JSON_SERDES
from pydantic_ai import Agent, RunContext
from pydantic_ai._run_context import get_current_run_context  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryContent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.toolsets.external import ExternalToolset

from pydantic_ai_harness.aws_lambda import (
    AWSLambdaDurability,
    _bridge,  # pyright: ignore[reportPrivateUsage]
    _operation_backend,  # pyright: ignore[reportPrivateUsage]
    durable_agent_handler,
    run_durable,
)

from .conftest import FakeDurableContext
from .test_aws_lambda_mcp import FakeMCPToolset


def tool_then_text(tool_name: str = 'act', args: dict[str, Any] | None = None) -> FunctionModel:
    """A model that calls `tool_name` once, then answers."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, args or {})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model_fn)


def build_agent(tool: Any, *, output_type: Any = None, **kwargs: Any) -> Agent[Any, Any]:
    extra = {'output_type': output_type} if output_type is not None else {}
    agent = Agent(tool_then_text(), name='a', capabilities=[AWSLambdaDurability(**kwargs)], **extra)
    tool.__name__ = 'act'
    agent.tool_plain(tool)
    return agent


class TestDurableRun:
    def test_checkpoints_model_and_tool_steps(self) -> None:
        agent = build_agent(lambda: 'sunny')
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('go'), context=ctx)

        assert result.output == 'done'
        assert ctx.step_names == [
            'a__model.request',
            'a__function_toolset__<agent>.call_tool:act',
            'a__model.request',
        ]
        assert ctx.failed == []

    def test_resume_serves_completed_steps_from_checkpoints(self) -> None:
        calls: list[str] = []

        def tool() -> str:
            calls.append('tool')
            return 'sunny'

        agent = build_agent(tool)
        first = FakeDurableContext()
        run_durable(lambda: agent.run('go'), context=first)
        assert calls == ['tool']

        resumed = FakeDurableContext(journal=first.operations)
        result = run_durable(lambda: agent.run('go'), context=resumed)

        assert result.output == 'done'
        # No step body ran again: the model was not called and the tool did not re-execute.
        assert resumed.invoked == []
        assert calls == ['tool']

    def test_transparent_outside_a_durable_handler(self) -> None:
        agent = build_agent(lambda: 'sunny')

        result = agent.run_sync('go')

        assert result.output == 'done'

    def test_steps_run_on_the_handler_thread(self) -> None:
        agent = build_agent(lambda: 'sunny')
        ctx = FakeDurableContext()
        handler_thread = threading.get_ident()

        run_durable(lambda: agent.run('go'), context=ctx)

        assert {op.thread_id for op in ctx.operations} == {handler_thread}

    def test_agent_error_propagates_to_the_handler(self) -> None:
        def tool() -> str:
            raise RuntimeError('boom')

        agent = build_agent(tool)
        ctx = FakeDurableContext()

        with pytest.raises(RuntimeError, match='boom'):
            run_durable(lambda: agent.run('go'), context=ctx)

        assert [op.status for op in ctx.failed] == ['failed']


class TestDurableAgentHandler:
    def test_runs_async_handler(self) -> None:
        agent = Agent(TestModel(custom_output_text='done'), name='a', capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        @durable_agent_handler
        async def handler(event: dict[str, str], context: _bridge.DurableStepContext) -> str:
            result = await agent.run(event['prompt'])
            return result.output

        assert handler({'prompt': 'go'}, ctx) == 'done'
        assert ctx.step_names == ['a__model.request']

    def test_parameterized_form_passes_cancel_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: list[float] = []

        def fake_run_durable(agent_run: Any, *, context: _bridge.DurableStepContext, cancel_timeout: float) -> str:
            del context
            received.append(cancel_timeout)
            return asyncio.run(agent_run())

        monkeypatch.setattr(_bridge, 'run_durable', fake_run_durable)

        @durable_agent_handler(cancel_timeout=30)
        async def handler(event: object, context: _bridge.DurableStepContext) -> str:
            del event, context
            return 'handler result'

        assert handler({}, FakeDurableContext()) == 'handler result'
        assert received == [30]

    def test_wrong_decorator_order_is_rejected(self) -> None:
        async def handler(event: object, context: object) -> None:
            """Stub used only to verify decorator composition."""

        sdk_wrapped = durable_execution(handler)
        with pytest.raises(
            UserError,
            match='`@durable_execution` must be the outermost decorator.*synchronous handler should call `run_durable`',
        ):
            durable_agent_handler(sdk_wrapped)  # pyright: ignore[reportCallIssue, reportArgumentType]

    def test_multiple_agent_runs_share_one_handler_invocation(self) -> None:
        first = Agent(TestModel(custom_output_text='one'), name='first', capabilities=[AWSLambdaDurability()])
        second = Agent(TestModel(custom_output_text='two'), name='second', capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        @durable_agent_handler
        async def handler(event: object, context: _bridge.DurableStepContext) -> tuple[str, str]:
            del event, context
            first_result = await first.run('go')
            second_result = await second.run('go')
            return first_result.output, second_result.output

        assert handler({}, ctx) == ('one', 'two')
        assert ctx.step_names == ['first__model.request', 'second__model.request']


class TestCapabilityOperation:
    def test_operation_is_journaled_and_replayed_without_rerunning_handler(self) -> None:
        class Contributor(AbstractCapability[Any]):
            id = 'contributor'

            def __init__(self) -> None:
                self.calls = 0

            async def before_run(self, ctx: RunContext[Any]) -> None:
                await self.record(ctx, 'value')

            @durable_operation('record')
            async def record(self, ctx: RunContext[Any], value: str) -> str:
                self.calls += 1
                return value

        contributor = Contributor()
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='a',
            capabilities=[contributor, AWSLambdaDurability()],
        )

        first = FakeDurableContext()
        run_durable(lambda: agent.run('go'), context=first)

        assert first.step_names[0] == 'a__capability__contributor.record'
        assert contributor.calls == 1

        resumed = FakeDurableContext(journal=first.operations)
        result = run_durable(lambda: agent.run('go'), context=resumed)

        assert result.output == 'done'
        assert resumed.invoked == []
        assert contributor.calls == 1

    def test_operation_receives_base_step_config(self) -> None:
        class Contributor(AbstractCapability[Any]):
            id = 'contributor'

            async def before_run(self, ctx: RunContext[Any]) -> None:
                await self.record(ctx)

            @durable_operation('record')
            async def record(self, ctx: RunContext[Any]) -> None:
                pass

        agent = Agent(
            TestModel(custom_output_text='done'),
            name='a',
            capabilities=[
                Contributor(),
                AWSLambdaDurability(step_config={'step_semantics': StepSemantics.AT_MOST_ONCE_PER_RETRY}),
            ],
        )
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('go'), context=ctx)

        operation = next(op for op in ctx.operations if op.name == 'a__capability__contributor.record')
        assert operation.config is not None
        assert operation.config.step_semantics is StepSemantics.AT_MOST_ONCE_PER_RETRY


class TestControlFlowSignals:
    """Control-flow signals must cross a step as values, not as step failures.

    A step body that raises is recorded as failed and retried, so an approval request or a
    deferred call would be retried and then fail the execution instead of pausing the run.
    """

    def test_approval_required_pauses_the_run(self) -> None:
        def tool() -> str:
            raise ApprovalRequired

        agent = build_agent(tool, output_type=[str, DeferredToolRequests])
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('go'), context=ctx)

        assert isinstance(result.output, DeferredToolRequests)
        assert len(result.output.approvals) == 1
        assert ctx.failed == []

    def test_call_deferred_pauses_the_run(self) -> None:
        def tool() -> str:
            raise CallDeferred

        agent = build_agent(tool, output_type=[str, DeferredToolRequests])
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('go'), context=ctx)

        assert isinstance(result.output, DeferredToolRequests)
        assert len(result.output.calls) == 1
        assert ctx.failed == []

    def test_model_retry_is_not_a_step_failure(self) -> None:
        def tool() -> str:
            raise ModelRetry('try again')

        agent = build_agent(tool)
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('go'), context=ctx)

        assert result.output == 'done'
        assert ctx.failed == []


class TestToolResultSerialization:
    """A tool result is checkpointed through the SDK serializer, so it must round-trip."""

    def test_tool_return_object(self) -> None:
        agent = build_agent(lambda: ToolReturn(return_value='ok', content='extra'))
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('go'), context=ctx)

        assert result.output == 'done'
        assert ctx.failed == []

    def test_binary_content(self) -> None:
        agent = build_agent(lambda: BinaryContent(data=b'\x89PNG', media_type='image/png'))
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('go'), context=ctx)

        assert result.output == 'done'
        assert ctx.failed == []

    def test_non_serializable_tool_result_fails_without_retry(self) -> None:
        agent = build_agent(lambda: object())
        ctx = FakeDurableContext()

        with pytest.raises(ExecutionError, match='Unable to serialize unknown type'):
            run_durable(lambda: agent.run('go'), context=ctx)

        assert len(ctx.failed) == 1

    def test_fake_records_result_serialization_failure(self) -> None:
        ctx = FakeDurableContext()

        with pytest.raises(ExecutionError, match='Serialization failed for id: operation'):
            ctx.step(lambda _: object(), name='non-serializable')

        assert len(ctx.failed) == 1
        assert ctx.failed[0].name == 'non-serializable'


class TestCrashMidRunRetry:
    def test_completed_model_step_is_reused_while_the_failed_tool_reruns(self) -> None:
        tool_calls: list[int] = []

        def tool() -> str:
            tool_calls.append(1)
            if len(tool_calls) == 1:
                raise RuntimeError('transient')
            return 'sunny'

        agent = build_agent(tool)

        first = FakeDurableContext()
        with pytest.raises(RuntimeError, match='transient'):
            run_durable(lambda: agent.run('go'), context=first)
        assert first.step_names == ['a__model.request', 'a__function_toolset__<agent>.call_tool:act']

        # Resume from the completed prefix: the model step replays, the failed tool runs again.
        resumed = FakeDurableContext(journal=first.operations[:1])
        result = run_durable(lambda: agent.run('go'), context=resumed)

        assert result.output == 'done'
        # The first model request came from the checkpoint; only the second one ran.
        assert resumed.invoked == ['a__function_toolset__<agent>.call_tool:act', 'a__model.request']
        assert len(tool_calls) == 2


class TestStepConfig:
    def test_base_config_is_applied_to_every_step(self) -> None:
        agent = build_agent(lambda: 'sunny', step_config={'step_semantics': StepSemantics.AT_MOST_ONCE_PER_RETRY})
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('go'), context=ctx)

        configs = [op.config for op in ctx.operations]
        assert all(c is not None and c.step_semantics is StepSemantics.AT_MOST_ONCE_PER_RETRY for c in configs)

    def test_base_config_accepts_valid_retry_strategy_and_serdes(self) -> None:
        capability = AWSLambdaDurability(
            step_config={'retry_strategy': RetryPresets.none(), 'serdes': DEFAULT_JSON_SERDES}
        )

        assert capability is not None

    def test_per_tool_metadata_overrides_the_base_config_key_by_key(self) -> None:
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain(metadata={'aws_lambda': {'step_semantics': StepSemantics.AT_MOST_ONCE_PER_RETRY}})
        def act() -> str:
            return 'sunny'

        agent = Agent(tool_then_text(), name='a', toolsets=[toolset], capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('go'), context=ctx)

        tool_op = next(op for op in ctx.operations if op.name == 'a__function_toolset__tools.call_tool:act')
        assert tool_op.config is not None
        assert tool_op.config.step_semantics is StepSemantics.AT_MOST_ONCE_PER_RETRY

    def test_metadata_false_runs_the_tool_inline(self) -> None:
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain(metadata={'aws_lambda': False})
        def act() -> str:
            return 'sunny'

        agent = Agent(tool_then_text(), name='a', toolsets=[toolset], capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('go'), context=ctx)

        assert ctx.step_names == ['a__model.request', 'a__model.request']

    def test_unknown_config_key_is_rejected_at_construction(self) -> None:
        with pytest.raises(UserError, match="Unknown 'aws_lambda' step config key 'retries'"):
            AWSLambdaDurability(step_config={'retries': 3})

    @pytest.mark.parametrize(
        ('key', 'value', 'expected'),
        [
            ('retry_strategy', 3, 'a callable or None'),
            ('step_semantics', 'at-least-once', 'StepSemantics'),
            ('serdes', 3, 'SerDes or None'),
        ],
    )
    def test_wrong_base_config_value_type_is_rejected_at_construction(
        self, key: str, value: object, expected: str
    ) -> None:
        with pytest.raises(
            UserError,
            match=rf"Invalid 'aws_lambda' step config value for '{key}': expected {expected}, got",
        ):
            AWSLambdaDurability(step_config={key: value})

    def test_new_sdk_step_config_field_passes_through_unvalidated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def future_step_config(**config: Any) -> object:
            config.pop('timeout')
            return StepConfig(**config)

        monkeypatch.setattr(
            _operation_backend, '_STEP_CONFIG_FIELDS', _operation_backend._STEP_CONFIG_FIELDS | {'timeout'}
        )
        monkeypatch.setattr(_operation_backend, 'StepConfig', future_step_config)

        capability = AWSLambdaDurability(step_config={'timeout': 30.0})

        assert capability is not None

    def test_unknown_per_tool_config_key_is_rejected(self) -> None:
        toolset = FunctionToolset[object](id='tools')
        toolset.add_function(act, metadata={'aws_lambda': {'retries': 3}})

        agent = Agent(tool_then_text(), name='a', toolsets=[toolset], capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match="Unknown 'aws_lambda' step config key 'retries'"):
            run_durable(lambda: agent.run('go'), context=ctx)

    def test_wrong_per_tool_config_value_type_is_rejected(self) -> None:
        toolset = FunctionToolset[object](id='tools')
        toolset.add_function(act, metadata={'aws_lambda': {'step_semantics': 'at-least-once'}})

        agent = Agent(tool_then_text(), name='a', toolsets=[toolset], capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        with pytest.raises(
            UserError,
            match="Invalid 'aws_lambda' step config value for 'step_semantics': expected StepSemantics, got str",
        ):
            run_durable(lambda: agent.run('go'), context=ctx)

    def test_non_mapping_per_tool_config_is_rejected(self) -> None:
        toolset = FunctionToolset[object](id='tools')
        toolset.add_function(act, metadata={'aws_lambda': 'invalid'})

        agent = Agent(tool_then_text(), name='a', toolsets=[toolset], capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match='expected a dict .* or `False`, got str'):
            run_durable(lambda: agent.run('go'), context=ctx)


class TestNesting:
    """A nested durable run cannot make progress: the handler thread is blocked servicing the
    outer step, so it cannot service the inner one. Both entry points reject it rather than hang.
    """

    def test_nested_agent_run_is_rejected_instead_of_deadlocking(self) -> None:
        inner = Agent(TestModel(), name='inner', capabilities=[AWSLambdaDurability()])

        async def act() -> str:
            return str((await inner.run('nested')).output)

        agent = Agent(tool_then_text(), name='a', capabilities=[AWSLambdaDurability()])
        agent.tool_plain(act)
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match='from inside another AWS Lambda durable step'):
            run_durable(lambda: agent.run('go'), context=ctx)

    def test_run_durable_inside_a_running_loop_is_rejected(self) -> None:
        inner = Agent(TestModel(), name='inner', capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        async def act() -> str:
            return str(run_durable(lambda: inner.run('nested'), context=ctx).output)

        agent = Agent(tool_then_text(), name='a', capabilities=[AWSLambdaDurability()])
        agent.tool_plain(act)

        with pytest.raises(UserError, match='cannot be called from a running event loop'):
            run_durable(lambda: agent.run('go'), context=ctx)


def act() -> str:
    return 'sunny'


class TestBinding:
    def test_binding_does_not_mutate_the_template_capability(self) -> None:
        capability = AWSLambdaDurability()

        agent = Agent('test', name='a', capabilities=[capability])
        bound = AWSLambdaDurability.from_agent(agent)

        assert capability.agent is None
        assert capability.default_model_id is None
        assert bound is not None
        assert bound.agent is agent
        assert bound.default_model_id == 'test'

    def test_agent_without_a_name_is_rejected(self) -> None:
        with pytest.raises(UserError, match='unique `name`'):
            Agent(TestModel(), capabilities=[AWSLambdaDurability()])

    def test_capability_name_overrides_the_agent_name(self) -> None:
        agent = Agent(tool_then_text(), name='a', capabilities=[AWSLambdaDurability(name='custom')])
        agent.tool_plain(act)
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('go'), context=ctx)

        assert ctx.step_names[0] == 'custom__model.request'

    def test_model_id_is_folded_into_the_step_name(self) -> None:
        extra = TestModel(custom_output_text='from extra')
        agent = Agent(tool_then_text(), name='a', capabilities=[AWSLambdaDurability(models={'extra': extra})])
        agent.tool_plain(act)
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('go', model='extra'), context=ctx)

        assert ctx.step_names[0] == 'a__model.request.extra'


class TestEventStreamHandler:
    def test_agent_events_are_checkpointed(self) -> None:
        seen: list[AgentStreamEvent] = []

        async def handler(ctx: RunContext[object], stream: Any) -> None:
            async for event in stream:
                seen.append(event)

        # An event stream handler makes the run stream, so use a model that supports streaming.
        agent = Agent(TestModel(), name='a', capabilities=[AWSLambdaDurability(event_stream_handler=handler)])
        agent.tool_plain(act)
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('go'), context=ctx)

        assert 'a__model.request_stream' in ctx.step_names
        assert 'a__event_stream_handler' in ctx.step_names
        assert seen


class TestCoverageOfRemainingPaths:
    def test_sync_tool_calling_run_durable_is_rejected(self) -> None:
        """A sync tool runs on a worker thread with the handler's context copied, so there is no
        running loop but the bridge is still active."""
        inner = Agent(TestModel(), name='inner', capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        def act() -> str:
            return str(run_durable(lambda: inner.run('nested'), context=ctx).output)

        agent = Agent(tool_then_text(), name='a', capabilities=[AWSLambdaDurability()])
        agent.tool_plain(act)

        with pytest.raises(UserError, match='already active on this thread'):
            run_durable(lambda: agent.run('go'), context=ctx)

    def test_concurrent_run_durable_calls_from_different_threads_are_rejected(self) -> None:
        start = threading.Barrier(2)
        run_started = threading.Event()
        rejected = threading.Event()
        release = threading.Event()
        results: list[str] = []
        errors: list[BaseException] = []

        async def hold_run() -> str:
            run_started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            return 'done'

        def invoke() -> None:
            start.wait(timeout=5)
            try:
                results.append(run_durable(hold_run, context=FakeDurableContext()))
            except BaseException as exc:
                errors.append(exc)
                rejected.set()

        threads = [threading.Thread(target=invoke) for _ in range(2)]
        for thread in threads:
            thread.start()
        assert run_started.wait(timeout=5)
        assert rejected.wait(timeout=5)
        release.set()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

        assert results == ['done']
        assert len(errors) == 1
        assert isinstance(errors[0], UserError)
        assert 'another thread' in str(errors[0])

    def test_process_guard_is_released_after_a_failed_run(self) -> None:
        async def fail() -> None:
            raise RuntimeError('boom')

        with pytest.raises(RuntimeError, match='boom'):
            run_durable(fail, context=FakeDurableContext())

        assert run_durable(lambda: asyncio.sleep(0, result='recovered'), context=FakeDurableContext()) == 'recovered'

    def test_toolsets_without_a_durable_wrapper_pass_through(self) -> None:
        external = ExternalToolset[object]([ToolDefinition(name='remote')], id='ext')
        agent = Agent(tool_then_text(), name='a', toolsets=[external], capabilities=[AWSLambdaDurability()])
        agent.tool_plain(act)
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('go'), context=ctx)

        assert 'a__function_toolset__<agent>.call_tool:act' in ctx.step_names

    def test_a_string_model_default_keeps_one_step_name(self) -> None:
        """The default model carries its own name as provenance; the suffix is suppressed so the
        default keeps a single step name."""
        agent = Agent('test', name='a', capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('go'), context=ctx)

        assert ctx.step_names == ['a__model.request']

    def test_a_model_instance_default_keeps_one_step_name(self) -> None:
        """The twin of the string case, by a different route. An agent built from a `Model`
        instance leaves `ModelRequestContext.model_id` unset, so there is no provenance string to
        compare against and the base resolves the id through its model registry -- which already
        answers `None` for the agent's own model. The name is short because nothing named it, not
        because the suffix was suppressed."""
        model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')]))
        agent = Agent(model, name='a', capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        run_durable(lambda: agent.run('go'), context=ctx)

        assert ctx.step_names == ['a__model.request']

    def test_cancel_suspended_response_is_checkpointed(self) -> None:
        # The model returns a suspended response, the agent re-issues it as a continuation, the
        # continuation fails, and the graph tears the suspended job down via
        # `cancel_suspended_response`, which is itself checkpointed.
        cancelled: list[ModelResponse] = []

        class CancellableModel(FunctionModel):
            async def cancel_suspended_response(self, response: ModelResponse) -> None:
                cancelled.append(response)

        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(m.parts and getattr(m, 'state', None) == 'suspended' for m in messages):
                return ModelResponse(parts=[TextPart(content='partial')], state='suspended')
            raise RuntimeError('continuation failed')

        agent = Agent(CancellableModel(fn, model_name='fn'), name='a', capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()

        with pytest.raises(RuntimeError, match='continuation failed'):
            run_durable(lambda: agent.run('go'), context=ctx)

        assert len(cancelled) == 1
        assert 'a__model.cancel_suspended_response' in ctx.step_names


def shutdown(loop: asyncio.AbstractEventLoop) -> None:
    """Stop and close a loop a test built, so it does not outlive the test as a leak.

    Retired and stopped bridge loops close themselves on their owning thread. Tests also pass
    foreign loops that have no owning thread, so this helper closes those directly.
    """
    if loop.is_closed():
        return
    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
        deadline = time.monotonic() + 5
        while loop.is_running() and time.monotonic() < deadline:  # pragma: no branch - stops promptly
            time.sleep(0.01)
    # Unwind anything the loop was still holding, so closing it does not log a pending task.
    outstanding = asyncio.all_tasks(loop)
    if outstanding:  # pragma: no cover - test cleanup normally leaves no pending tasks
        for task in outstanding:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*outstanding, return_exceptions=True))
    if not loop.is_closed():
        loop.close()


class TestBridgeFailureModes:
    """Regressions for ways the bridge could strand the handler thread or swallow SDK control flow."""

    def test_a_cancelled_step_operation_does_not_hang_the_handler(self) -> None:
        # `Task.exception()` raises for a cancelled task, so a naive done-callback would strand the
        # handler thread inside `context.step(...)` until the function timed out.
        def act() -> str:
            raise asyncio.CancelledError('inner cancel')

        agent = build_agent(act)
        ctx = FakeDurableContext()

        # The point of the assertion is that the call returns at all rather than wedging the thread.
        with pytest.raises(BaseException):
            run_durable(lambda: agent.run('go'), context=ctx)

    def test_sdk_control_flow_propagates_out_of_the_handler(self) -> None:
        """`SuspendExecution` and friends are `BaseException`s the SDK raises so user code cannot
        catch them. Routing one into the agent loop would turn a suspension into a failure."""

        class Suspend(BaseException):
            pass

        class SuspendingContext(FakeDurableContext):
            def step(self, func, name=None, config=None):  # type: ignore[no-untyped-def]
                if name is not None and 'call_tool' in name:
                    raise Suspend('retry scheduled')
                return super().step(func, name=name, config=config)

        agent = build_agent(act)
        ctx = SuspendingContext()

        with pytest.raises(Suspend):
            run_durable(lambda: agent.run('go'), context=ctx)

    @pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
    def test_a_dead_agent_loop_fails_the_step_instead_of_blocking_forever(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A step body blocks the handler thread on a result only the agent loop can deliver. If
        that loop stops, nothing will ever resolve it, and an unconditional wait would hang until
        Lambda timed the function out with an opaque error. Steps can legitimately run for minutes,
        so the wait polls for the loop's liveness rather than imposing a deadline of its own."""
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(_bridge, '_agent_loop', loops)
        monkeypatch.setattr(_bridge, '_LOOP_LIVENESS_POLL_SECONDS', 0.05)

        async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Stop the loop from inside the step body, then suspend: the task never completes, so
            # the done-callback that would resolve the handler thread's future can never run.
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
            return ModelResponse(parts=[TextPart('unreachable')])  # pragma: no cover - never resumed

        agent = Agent(FunctionModel(model_fn), name='a', capabilities=[AWSLambdaDurability()])
        ctx = FakeDurableContext()
        dead = loops.get()

        with pytest.raises(_bridge.AgentLoopGone, match='agent event loop stopped'):
            run_durable(lambda: agent.run('go'), context=ctx, cancel_timeout=0.05)

        assert dead.is_closed()
        gc.collect()

    @pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
    def test_an_agent_loop_closed_before_step_scheduling_fails_instead_of_blocking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(_bridge, '_agent_loop', loops)

        class CloseBeforeBodyContext(FakeDurableContext):
            def step(self, func, name=None, config=None):  # type: ignore[no-untyped-def]
                loop = loops.get()
                thread = loops._thread  # pyright: ignore[reportPrivateUsage]
                assert thread is not None
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=5)
                assert loop.is_closed()
                return super().step(func, name=name, config=config)

        with pytest.raises(_bridge.AgentLoopGone, match='agent event loop stopped'):
            run_durable(lambda: build_agent(act).run('go'), context=CloseBeforeBodyContext(), cancel_timeout=0.05)

        gc.collect()

    def test_a_step_that_outlasts_the_liveness_poll_keeps_waiting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The poll interval is not a deadline. A step that runs longer than it simply keeps
        waiting, because the loop that owes it a result is still there to deliver one."""
        monkeypatch.setattr(_bridge, '_LOOP_LIVENESS_POLL_SECONDS', 0.01)

        async def act() -> str:
            await asyncio.sleep(0.05)
            return 'sunny'

        agent = build_agent(act)
        ctx = FakeDurableContext()

        result = run_durable(lambda: agent.run('go'), context=ctx)

        assert result.output == 'done'

    def test_a_scheduled_run_that_never_starts_fails_and_closes_its_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(_bridge, '_agent_loop', loops)
        monkeypatch.setattr(_bridge, '_LOOP_START_TIMEOUT_SECONDS', 0.05)
        loop = loops.get()
        loop_thread = loops._thread  # pyright: ignore[reportPrivateUsage]
        assert loop_thread is not None
        blocking = threading.Event()
        blocker_started = threading.Event()

        def block_loop() -> None:
            blocker_started.set()
            blocking.wait(timeout=5)

        loop.call_soon_threadsafe(block_loop)
        assert blocker_started.wait(timeout=5)

        with pytest.raises(_bridge.AgentLoopGone, match='did not start the scheduled run'):
            run_durable(lambda: build_agent(act).run('go'), context=FakeDurableContext())

        blocking.set()
        loop_thread.join(timeout=5)
        assert loop.is_closed()
        assert not loop_thread.is_alive()

    @pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
    def test_an_unwind_that_outlives_the_cancel_timeout_closes_the_retired_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The loop is shared across warm invocations, so a run abandoned by a suspension is
        cancelled before the handler returns. When that unwind does not finish in time, the loop is
        given up rather than reused, so the next invocation cannot inherit its loop-bound state."""
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(_bridge, '_agent_loop', loops)
        monkeypatch.setattr(_bridge, '_RETIRED_LOOP_GRACE_SECONDS', 0.05)

        class Suspend(BaseException):
            pass

        class SuspendingContext(FakeDurableContext):
            def step(self, func, name=None, config=None):  # type: ignore[no-untyped-def]
                raise Suspend('retry scheduled')

        cleanup_reached = threading.Event()

        async def run_with_slow_cleanup() -> str:
            try:
                return await agent.run('go')  # type: ignore[return-value]
            finally:
                cleanup_reached.set()
                # Refuse cancellation past the retirement deadline. The loop must still stop at
                # the grace deadline rather than letting this abandoned run continue indefinitely.
                while True:
                    try:
                        await asyncio.sleep(10)
                    except asyncio.CancelledError:
                        continue  # pragma: no cover - the forced deadline closes the loop first

        agent = build_agent(act)
        abandoned = loops.get()
        abandoned_thread = loops._thread  # pyright: ignore[reportPrivateUsage]
        assert abandoned_thread is not None

        with pytest.raises(Suspend):
            run_durable(run_with_slow_cleanup, context=SuspendingContext(), cancel_timeout=0.01)

        assert cleanup_reached.is_set()
        # The handler returned while the abandoned run still holds `abandoned`, so the next
        # invocation must not be handed that loop.
        replacement = loops.get()
        assert replacement is not abandoned

        started_waiting = time.monotonic()
        deadline = time.monotonic() + 5
        while not abandoned.is_closed() and time.monotonic() < deadline:
            time.sleep(0.01)  # pragma: no cover - the retired loop normally closes before polling

        assert abandoned.is_closed()
        assert not abandoned_thread.is_alive()
        assert time.monotonic() - started_waiting < 0.5
        shutdown(replacement)
        gc.collect()

    def test_retirement_drains_cleanup_scheduled_when_the_main_task_finishes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(_bridge, '_agent_loop', loops)
        monkeypatch.setattr(_bridge, '_RETIRED_LOOP_GRACE_SECONDS', 0.5)
        cleanup_finished = threading.Event()

        class Suspend(BaseException):
            pass

        class SuspendingContext(FakeDurableContext):
            def step(self, func, name=None, config=None):  # type: ignore[no-untyped-def]
                raise Suspend('retry scheduled')

        async def delayed_cleanup() -> None:
            await asyncio.sleep(0.02)
            cleanup_finished.set()

        agent = build_agent(act)

        async def abandoned_run() -> str:
            try:
                return await agent.run('go')  # type: ignore[return-value]
            finally:
                await asyncio.sleep(0.03)
                asyncio.create_task(delayed_cleanup())

        retired = loops.get()
        thread = loops._thread  # pyright: ignore[reportPrivateUsage]
        assert thread is not None

        with pytest.raises(Suspend):
            run_durable(abandoned_run, context=SuspendingContext(), cancel_timeout=0.01)

        assert cleanup_finished.wait(timeout=5)
        thread.join(timeout=5)
        assert retired.is_closed()
        assert not thread.is_alive()

    def test_retirement_cancels_and_drains_pending_tasks_without_a_main_task(self) -> None:
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        loop = loops.get()
        thread = loops._thread  # pyright: ignore[reportPrivateUsage]
        assert thread is not None
        task_started = threading.Event()
        task_cancelled = threading.Event()

        async def pending_work() -> None:
            task_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                task_cancelled.set()

        loop.call_soon_threadsafe(loop.create_task, pending_work())
        assert task_started.wait(timeout=5)

        loops.retire(loop, None)

        thread.join(timeout=5)
        assert task_cancelled.is_set()
        assert loop.is_closed()
        assert not thread.is_alive()

    def test_retirement_closes_a_live_loop_with_no_pending_tasks(self) -> None:
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        loop = loops.get()
        thread = loops._thread  # pyright: ignore[reportPrivateUsage]
        assert thread is not None

        loops.retire(loop, None)

        thread.join(timeout=5)
        assert loop.is_closed()
        assert not thread.is_alive()

    def test_retirement_drains_default_executor_before_closing_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(_bridge, '_RETIRED_LOOP_GRACE_SECONDS', 1)
        loop = loops.get()
        thread = loops._thread  # pyright: ignore[reportPrivateUsage]
        assert thread is not None
        worker_started = threading.Event()
        release_worker = threading.Event()
        worker_finished = threading.Event()

        def executor_work() -> None:
            worker_started.set()
            release_worker.wait(timeout=5)
            worker_finished.set()

        loop.call_soon_threadsafe(loop.run_in_executor, None, executor_work)
        assert worker_started.wait(timeout=5)

        loops.retire(loop, None)

        assert thread.is_alive()
        assert not loop.is_closed()
        assert not worker_finished.is_set()
        release_worker.set()
        thread.join(timeout=5)
        assert worker_finished.is_set()
        assert loop.is_closed()
        assert not thread.is_alive()

    def test_retirement_cancels_a_waiter_when_the_cleanup_budget_expires(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(_bridge, '_RETIRED_LOOP_GRACE_SECONDS', 0)
        loop = loops.get()
        thread = loops._thread  # pyright: ignore[reportPrivateUsage]
        assert thread is not None
        task_started = threading.Event()
        task_cancelled = threading.Event()
        task_holder: list[asyncio.Task[None]] = []

        async def pending_work() -> None:
            task_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                task_cancelled.set()

        def start_task() -> None:
            task_holder.append(loop.create_task(pending_work()))

        loop.call_soon_threadsafe(start_task)
        assert task_started.wait(timeout=5)
        assert task_holder

        # Keep the forced-stop timer from winning the race with the cleanup coroutine. The cleanup
        # still has a zero budget and stops the loop itself after cancelling its timed-out waiters.
        original_stop = loop.stop
        forced_stop_called = False

        def stop_after_cleanup() -> None:
            nonlocal forced_stop_called
            if not forced_stop_called:
                forced_stop_called = True
                return
            original_stop()

        monkeypatch.setattr(loop, 'stop', stop_after_cleanup)

        loops.retire(loop, task_holder[0])

        thread.join(timeout=5)
        assert task_cancelled.is_set()
        assert loop.is_closed()
        assert not thread.is_alive()

    def test_a_stopped_loop_is_not_handed_to_the_next_invocation(self) -> None:
        """`is_closed()` answers `False` for a stopped-but-open loop, so deciding reuse on that
        alone would hand the next invocation a loop that never runs a callback: `schedule_run`
        would sit in its queue, `bridge.finish()` would never be called, and the handler would
        block in `consume()` until Lambda timed the function out."""
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        stopped = loops.get()
        stopped.call_soon_threadsafe(stopped.stop)
        deadline = time.monotonic() + 5
        while stopped.is_running() and time.monotonic() < deadline:  # pragma: no branch - stops promptly
            time.sleep(0.01)

        replacement = loops.get()

        assert replacement is not stopped
        assert replacement.is_running()
        shutdown(replacement)
        shutdown(stopped)

    def test_retiring_an_unexpectedly_stopped_loop_finds_it_closed(self) -> None:
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        stopped = loops.get()
        thread = loops._thread  # pyright: ignore[reportPrivateUsage]
        assert thread is not None

        stopped.call_soon_threadsafe(stopped.stop)
        thread.join(timeout=5)
        loops.retire(stopped, None)

        assert stopped.is_closed()
        assert not thread.is_alive()

    def test_retiring_a_loop_that_was_already_replaced_leaves_the_current_one_alone(self) -> None:
        loops = _bridge._AgentLoop()  # pyright: ignore[reportPrivateUsage]
        live = loops.get()
        foreign = asyncio.new_event_loop()

        loops.retire(foreign, None)

        assert loops.get() is live
        shutdown(foreign)
        shutdown(live)


class TestRuntimeToolsets:
    @pytest.mark.parametrize('kind', ['function', 'mcp', 'dynamic'])
    def test_executing_toolsets_added_per_run_are_rejected(self, kind: str) -> None:
        toolset: object
        if kind == 'function':
            toolset = FunctionToolset[object](id='late')
        elif kind == 'dynamic':
            toolset = DynamicToolset[object](lambda ctx: FunctionToolset[object](), id='late')
        else:
            pytest.importorskip('pydantic_ai.mcp')
            toolset = FakeMCPToolset(id='late')

        agent = build_agent(act)
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match=re.escape('cannot be added at runtime with AWS Lambda')):
            run_durable(lambda: agent.run('go', toolsets=[toolset]), context=ctx)

    def test_non_executing_runtime_toolsets_pass_through(self) -> None:
        agent = build_agent(act)
        ctx = FakeDurableContext()

        result = run_durable(
            lambda: agent.run('go', toolsets=[ExternalToolset[object]([ToolDefinition(name='remote')], id='ext')]),
            context=ctx,
        )

        assert result.output == 'done'


class TestEnqueueGuard:
    def test_enqueue_inside_a_checkpointed_tool_raises(self) -> None:
        def act(ctx: RunContext[object]) -> None:
            ctx.enqueue('later')

        agent = Agent(tool_then_text(), name='a', capabilities=[AWSLambdaDurability()])
        agent.tool(act)
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match='enqueue'):
            run_durable(lambda: agent.run('go'), context=ctx)

    def test_enqueue_via_the_ambient_run_context_inside_a_step_raises(self) -> None:
        """Guarding only the context handed to user code leaves the ambient getter as a way around
        the guard, and a message enqueued through it is dropped just as silently on replay."""

        def act() -> None:
            ambient = get_current_run_context()
            assert ambient is not None
            ambient.enqueue('later')

        agent = Agent(tool_then_text(), name='a', capabilities=[AWSLambdaDurability()])
        agent.tool_plain(act)
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match='enqueue'):
            run_durable(lambda: agent.run('go'), context=ctx)

    def test_enqueue_from_a_streaming_model_body_raises(self) -> None:
        """The model body runs inside the model step too, so a custom streaming model that enqueues
        would have its messages dropped on replay -- the stream is served from the checkpoint and
        the body never runs again."""

        class EnqueueingModel(FunctionModel):
            @asynccontextmanager
            async def request_stream(  # type: ignore[override]
                self,
                messages: list[ModelMessage],
                model_settings: Any,
                model_request_parameters: Any,
                run_context: RunContext[Any] | None = None,
            ) -> AsyncGenerator[Any]:
                assert run_context is not None
                run_context.enqueue('later')
                yield  # pragma: no cover - the enqueue above always raises

        async def handler(ctx: RunContext[object], stream: Any) -> None:  # pragma: no cover - never reached
            async for _event in stream:
                pass

        model = EnqueueingModel(lambda messages, info: ModelResponse(parts=[TextPart('done')]))
        agent = Agent(model, name='a', capabilities=[AWSLambdaDurability(event_stream_handler=handler)])
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match='enqueue'):
            run_durable(lambda: agent.run('go'), context=ctx)

    def test_enqueue_from_an_event_handler_on_a_model_event_raises(self) -> None:
        """Model events reach the handler inside the model step (`capture_event_stream`); a replay
        serves the recorded step, so an enqueue there would be dropped. The first handler call is
        the model-request capture, so enqueuing after draining the stream raises on that path."""

        async def handler(ctx: RunContext[object], stream: Any) -> None:
            async for _event in stream:
                pass
            ctx.enqueue('later')

        agent = Agent(TestModel(), name='a', capabilities=[AWSLambdaDurability(event_stream_handler=handler)])
        agent.tool_plain(act)
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match='enqueue'):
            run_durable(lambda: agent.run('go'), context=ctx)

    def test_enqueue_from_an_event_handler_on_an_agent_event_raises(self) -> None:
        """Agent-level events reach the handler in their own step (`_dispatch_event_stream_event`);
        that step is replayed too, so an enqueue there would be dropped. Enqueuing only once a
        `FunctionToolCallEvent` is seen skips the model-capture call and hits the dispatch path."""

        async def handler(ctx: RunContext[object], stream: Any) -> None:
            saw_tool_call = False
            async for event in stream:
                if isinstance(event, FunctionToolCallEvent):
                    saw_tool_call = True
            if saw_tool_call:
                ctx.enqueue('later')

        agent = Agent(TestModel(), name='a', capabilities=[AWSLambdaDurability(event_stream_handler=handler)])
        agent.tool_plain(act)
        ctx = FakeDurableContext()

        with pytest.raises(UserError, match='enqueue'):
            run_durable(lambda: agent.run('go'), context=ctx)


class TestReplayFidelity:
    """The fake models what the service does on resume; these pin that behaviour."""

    def test_a_diverging_step_sequence_is_detected(self) -> None:
        agent = build_agent(act)
        first = FakeDurableContext()
        run_durable(lambda: agent.run('go'), context=first)

        # Resume an execution whose recorded operations belong to a different agent shape.
        other = Agent(tool_then_text(), name='b', capabilities=[AWSLambdaDurability()])
        other.tool_plain(act)
        resumed = FakeDurableContext(journal=first.operations)

        with pytest.raises(AssertionError, match='replay divergence'):
            run_durable(lambda: other.run('go'), context=resumed)

    def test_a_recorded_step_failure_is_raised_again_on_resume(self) -> None:
        # Once a step's retries are exhausted the failure is checkpointed, and resuming the
        # execution re-raises it rather than re-attempting the step.
        def failing() -> str:
            raise RuntimeError('permanent')

        agent = build_agent(failing)
        first = FakeDurableContext()
        with pytest.raises(RuntimeError, match='permanent'):
            run_durable(lambda: agent.run('go'), context=first)

        resumed = FakeDurableContext(journal=first.operations)
        with pytest.raises(RuntimeError, match='permanent'):
            run_durable(lambda: agent.run('go'), context=resumed)
        assert resumed.invoked == []
