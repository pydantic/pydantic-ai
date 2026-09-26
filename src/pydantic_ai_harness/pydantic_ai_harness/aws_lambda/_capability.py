"""Durable execution for Pydantic AI agents on AWS Lambda durable functions.

Checkpoints an agent's I/O -- model requests, function tool calls, MCP calls, and
dynamic-toolset resolution -- into Lambda durable steps (`DurableContext.step(...)`), so an
invocation that times out, fails, or is retried resumes from the last completed step instead of
replaying the work.

Lambda's durable API is synchronous and an agent run is async, so the capability drives every
step through the bridge in `_bridge.py`. See `durable_agent_handler`, which adapts an async handler
body, and `run_durable`, the lower-level bridge entrypoint.
"""

from __future__ import annotations

try:
    import aws_durable_execution_sdk_python  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `aws-durable-execution-sdk-python` package to use the AWS Lambda durability '
        'capability, you can use the `aws-lambda` optional group -- '
        '`pip install "pydantic-ai-harness[aws-lambda]"`'
    ) from _import_error

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from aws_durable_execution_sdk_python.config import StepConfig
from aws_durable_execution_sdk_python.exceptions import ExecutionError
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    DurabilityEngineSpec,
    DurableOperationBackend,
)
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT

from ._bridge import ENGINE_NAME as _ENGINE_NAME
from ._bridge import in_durable_context
from ._operation_backend import AWSLambdaOperationBackend, AWSLambdaOperationConfig

_TOOL_CONFIG_KEY = 'aws_lambda'


@dataclass(init=False)
class AWSLambdaDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that checkpoints an agent's I/O into AWS Lambda durable steps.

    Attach it with `capabilities=[AWSLambdaDurability()]` and decorate an async handler with
    `durable_agent_handler`: every model request, function tool call, MCP call, and dynamic-toolset
    resolution is wrapped in `DurableContext.step(...)`. A completed step is served from its
    checkpoint when the execution resumes, so finished work is not repeated and tokens are not
    re-spent.

    A step is checkpointed after it runs, so an interruption between a tool's side effect and its
    checkpoint re-runs the tool when the execution resumes: keep tool side effects idempotent, or
    set `step_semantics` to `AT_MOST_ONCE_PER_RETRY` for the tools that cannot tolerate it.

    Outside a durable handler the capability is transparent and the run is an ordinary agent run.

    Step results are checkpointed through the Lambda SDK's serializer, so a checkpointed tool's
    return value must survive that round trip. Control-flow signals (`ModelRetry`,
    `ApprovalRequired`, `CallDeferred`, `ToolFailed`) cross the boundary as values rather than
    exceptions, so approval and deferred-tool flows work inside a durable execution.

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
                `agent.run(model='<id>')`. The ID is folded into the step name so a resumed
                execution maps each checkpoint back to the model it was recorded for.
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

    def get_durable_operation_backend(self) -> DurableOperationBackend[StepConfig | None]:
        return AWSLambdaOperationBackend(
            agent_name=self.name,
            default_model_id=self.default_model_id,
            config=self._operation_config,
        )
