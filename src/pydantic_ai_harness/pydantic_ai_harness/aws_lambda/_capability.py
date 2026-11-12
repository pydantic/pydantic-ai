"""Durable execution for Pydantic AI agents on AWS Lambda durable functions.

Lambda's durable API is synchronous and an agent run is async, so every step goes through the
bridge in `_bridge.py`.
"""

from __future__ import annotations

try:
    import aws_durable_execution_sdk_python  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:
    raise ImportError(
        'Please install the `aws-durable-execution-sdk-python` package to use the AWS Lambda durability '
        'capability, you can use the `aws-lambda` optional group -- '
        '`pip install "pydantic-ai-harness[aws-lambda]"`'
    ) from _import_error

import functools
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, ClassVar

from aws_durable_execution_sdk_python.config import CallbackConfig, Duration, ParallelConfig, StepConfig
from aws_durable_execution_sdk_python.exceptions import CallbackError, ExecutionError, SuspendExecution
from pydantic import JsonValue
from pydantic_core import from_json, to_json

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.capabilities import WrapRunHandler
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    DurabilityEngineSpec,
    DurableOperationBackend,
    ExternalEventError,
    ExternalEventWaiter,
    external_event_scope,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext

from ._bridge import ENGINE_NAME as _ENGINE_NAME, current_bridge, in_durable_context
from ._operation_backend import TOOL_CONFIG_KEY as _TOOL_CONFIG_KEY, AWSLambdaOperationBackend, AWSLambdaOperationConfig


@dataclass(init=False)
class AWSLambdaDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that checkpoints an agent's I/O into AWS Lambda durable steps.

    Attach it with `capabilities=[AWSLambdaDurability()]` and decorate an async handler with
    `durable_agent_handler`: every model request, function tool call, MCP call, and dynamic-toolset
    resolution is wrapped in `DurableContext.step(...)`. A completed step is served from its
    checkpoint when the execution resumes, so finished work is not repeated.

    A step is checkpointed after it runs, so an interruption between a tool's side effect and its
    checkpoint re-runs the tool when the execution resumes: keep tool side effects idempotent, or
    set `step_semantics` to `AT_MOST_ONCE_PER_RETRY` for the tools that cannot tolerate it.

    Outside a durable handler the capability is transparent and the run is an ordinary agent run.

    Step results are checkpointed through the Lambda SDK's serializer, so a checkpointed tool's
    return value must survive that round trip. Control-flow signals (`ModelRetry`,
    `ApprovalRequired`, `CallDeferred`, `ToolFailed`) cross the boundary as values rather than
    exceptions, so approval and deferred-tool flows work inside a durable execution. With
    `RequestInput` or harness `AskUser` attached, a call that needs an answer from outside waits on
    a Lambda callback instead of ending the run. The waits between polls of a background response or
    a `BatchModelRequests` batch are durable Lambda waits, which suspend the invocation until they're
    over.

    Example:
        ```python {test="skip"}
        from aws_durable_execution_sdk_python import DurableContext, durable_execution
        from pydantic_ai import Agent
        from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, durable_agent_handler

        agent = Agent('bedrock:us.amazon.nova-pro-v1:0', name='support', capabilities=[AWSLambdaDurability()])


        @agent.tool_plain
        def get_weather(city: str) -> str:
            return f'It is sunny in {city}.'


        @durable_execution
        @durable_agent_handler
        async def handler(event: dict[str, object], context: DurableContext) -> str:
            result = await agent.run(str(event['prompt']))
            return result.output
        ```
    """

    engine_spec: ClassVar = DurabilityEngineSpec(
        engine_name=_ENGINE_NAME,
        durable_unit_noun='step',
        durable_container_noun='handler',
        codec=JSON_CODEC,
        serialization_failure=lambda exc: ExecutionError(str(exc)),
        sequential_tools_in_durable_context=True,
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        tool_config_key=_TOOL_CONFIG_KEY,
        suspension_error_types=(SuspendExecution,),
    )

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        step_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Create an `AWSLambdaDurability` capability.

        The agent's model, name, and toolsets are discovered when the capability is bound.

        Args:
            models: Optional additional models keyed by ID for run-time model switching via
                `agent.run(model='<id>')`. The ID is folded into the step name so the execution
                history shows which model each checkpoint came from.
            event_stream_handler: Optional event stream handler. Model events are handled live
                inside the model-request step; each agent-level event is handled in its own
                checkpointed step.
            name: Unique agent name used as the prefix for every step name. Defaults to the
                agent's `name` when the capability is bound.
            step_config: Base `StepConfig` fields applied to every step, as a mapping of
                `retry_strategy`, `step_semantics`, and `serdes`. Per-tool
                `metadata={'aws_lambda': {...}}` overrides it key by key for that tool.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self._operation_config = AWSLambdaOperationConfig(step_config)

    @property
    def in_durable_context(self) -> bool:
        return in_durable_context()

    async def durable_sleep(self, seconds: float) -> None:
        """Wait with a durable Lambda wait, which suspends the invocation until it's over.

        Lambda times waits in whole seconds, at least one, so `seconds` is rounded up.
        """
        bridge = current_bridge()
        # Only called in the durable context, which guarantees a bridge.
        assert bridge is not None
        await bridge.wait(f'{self.name}__sleep', Duration(seconds=max(1, math.ceil(seconds))))

    def get_durable_operation_backend(self) -> DurableOperationBackend[StepConfig | None]:
        return AWSLambdaOperationBackend(
            agent_name=self.name,
            default_model_id=self.default_model_id,
            config=self._operation_config,
        )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Let the run wait for external events, such as `RequestInput` answers, through Lambda callbacks."""
        if not in_durable_context():
            return await super().wrap_run(ctx, handler=handler)
        waiter = _CallbackWaiter(agent_name=self.name, ask_config=self._operation_config.base_step_config)
        with external_event_scope(waiter):
            return await super().wrap_run(ctx, handler=handler)


class _CallbackWaiter(ExternalEventWaiter):
    """Waits for an external event with a Lambda callback: its ID is the resume token."""

    def __init__(self, *, agent_name: str, ask_config: StepConfig | None) -> None:
        self._agent_name = agent_name
        self._ask_config = ask_config

    async def wait(
        self,
        name: str,
        *,
        ask: Callable[[str], Awaitable[JsonValue]],
        timeout: timedelta | None,
    ) -> JsonValue:
        # Lambda times callbacks in whole seconds, and zero means no timeout, so round up.
        seconds = 0 if timeout is None else math.ceil(timeout.total_seconds())
        callback_name = f'{self._agent_name}__external_event.{name}'
        # Looked up when the wait starts rather than when the run does, so a wait in a branch the
        # execution runs in parallel uses that branch's bridge.
        bridge = current_bridge()
        if bridge is None:
            raise UserError(
                f'The wait for {_ENGINE_NAME} callback {callback_name!r} started outside the durable execution '
                'that ran the agent, so there is no execution to suspend.'
            )
        try:
            result = await bridge.wait_for_callback(
                callback_name,
                callback_config=CallbackConfig(timeout=Duration(seconds=seconds)),
                ask=functools.partial(_ask_for_json, ask),
                ask_config=self._ask_config,
            )
        except CallbackError as error:
            raise ExternalEventError(str(error)) from error
        # An answer from `ask` is a JSON string, like a resolution: the `CallbackConfig` sets no `serdes`,
        # so on SDK 1.x and 3.x `Callback.result()` returns the result as the resolver sent it
        # (`PASS_THROUGH_SERDES`), or `None` when it sent none.
        if result is None:
            return None
        try:
            return from_json(result)
        except ValueError as error:
            raise UserError(
                f'The result sent to {_ENGINE_NAME} callback {callback_name!r} is not valid JSON: {error}'
            ) from error


async def _ask_for_json(ask: Callable[[str], Awaitable[JsonValue]], token: str) -> str | None:
    """Call `ask`, and record an answer as the JSON string a resolver would send, so both are read back the same way."""
    answer = await ask(token)
    return None if answer is None else to_json(answer).decode()


def parallel_config(durability: AWSLambdaDurability[Any]) -> ParallelConfig | None:
    """The config that checkpoints a `parallel` operation's results with `durability`'s `serdes`, if it sets one."""
    return durability._operation_config.parallel_config  # pyright: ignore[reportPrivateUsage]
