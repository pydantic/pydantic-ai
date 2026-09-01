from __future__ import annotations

import os
import sys
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from importlib.metadata import version
from typing import Any, cast

import httpx2
import pytest
from packaging.version import Version
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    CodeExecutionTool,
    ExternalToolset,
    FunctionToolset,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPart,
    ToolReturn,
    UserPromptPart,
    WebSearchTool,
)
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import (
    ProcessEventStream,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from pydantic_ai.tools import ToolDefinition
from pydantic_graph import GraphBuilder, StepContext
from pydantic_graph.join import reduce_list_append

try:
    from temporalio import activity, workflow
    from temporalio.client import WorkflowFailureError, WorkflowHistory
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ApplicationError
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import TemporalAgent, TemporalDurability  # pyright: ignore[reportDeprecated]

except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)


# Nothing imports this module on 3.14: the test modules carry the same gate and skip first, and the
# conftest fixture that imports it is never requested once nothing is collected.
if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

try:
    import logfire
except ImportError:  # pragma: lax no cover
    pytest.skip('logfire not installed', allow_module_level=True)

try:
    from fastmcp.client.transports import StdioTransport

    from pydantic_ai.mcp import MCPToolset
except ImportError:  # pragma: lax no cover
    pytest.skip('mcp not installed', allow_module_level=True)

try:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)


with workflow.unsafe.imports_passed_through():
    # Workaround for a race condition when running `logfire.info` inside an activity with attributes to serialize and pandas importable:
    # AttributeError: partially initialized module 'pandas' has no attribute '_pandas_parser_CAPI' (most likely due to a circular import)
    try:
        import pandas  # pyright: ignore[reportUnusedImport] # noqa: F401
    except ImportError:  # pragma: lax no cover
        pass

    # https://github.com/temporalio/sdk-python/blob/3244f8bffebee05e0e7efefb1240a75039903dda/tests/test_client.py#L112C1-L113C1

    # Loads `vcr`, which Temporal doesn't like without passing through the import


# `TemporalAgent` is deprecated in favor of `capabilities=[TemporalDurability(...)]`.
# These tests exercise the wrapper-agent path on purpose; suppress the warning here
# rather than globally in `pyproject.toml`. The `pytestmark` entry below covers warnings
# emitted *inside* test functions; the `filterwarnings` call below covers warnings emitted
# at module import time (e.g. module-level construction of `TemporalAgent`).
warnings.filterwarnings('ignore', message='`TemporalAgent` is deprecated', category=PydanticAIDeprecationWarning)


# We need to use a custom cached HTTP client here as the default one created for OpenAIProvider will be closed automatically
# at the end of each test, but we need this one to live longer.
http_client = httpx2.AsyncClient()


@contextmanager
def workflow_raises(exc_type: type[Exception], exc_message: str) -> Generator[None]:
    """Helper for asserting that a Temporal workflow fails with the expected error."""
    with pytest.raises(WorkflowFailureError) as exc_info:
        yield
    assert isinstance(exc_info.value.__cause__, ApplicationError)
    assert exc_info.value.__cause__.type == exc_type.__name__
    assert exc_info.value.__cause__.message == exc_message


@contextmanager
def workflow_activity_raises(exc_type: type[Exception], exc_message: str) -> Generator[None]:
    """Assert an activity failure preserves the user exception through Temporal's cause chain."""
    with pytest.raises(WorkflowFailureError) as exc_info:
        yield
    causes: list[BaseException] = []
    error: BaseException | None = exc_info.value
    while error is not None:
        causes.append(error)
        error = error.__cause__
    assert any(
        isinstance(cause, ApplicationError) and cause.type == exc_type.__name__ and cause.message == exc_message
        for cause in causes
    ), f'{exc_type.__name__}({exc_message!r}) not found in the workflow failure cause chain: {causes}'


TASK_QUEUE = 'pydantic-ai-agent-task-queue'

BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)


# Can't use the `openai_api_key` fixture here because the workflow needs to be defined at the top level of the file.
model = OpenAIChatModel(
    'gpt-4o',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)


simple_agent = Agent(model, name='simple_agent')


# This needs to be done before the `TemporalAgent` is bound to the workflow.
simple_temporal_agent = TemporalAgent(simple_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


class Deps(BaseModel):
    country: str


async def event_stream_handler(
    ctx: RunContext[Deps],
    stream: AsyncIterable[AgentStreamEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info('event', event=event)


async def get_country(ctx: RunContext[Deps]) -> str:
    return ctx.deps.country


class WeatherArgs(BaseModel):
    city: str


def get_weather(args: WeatherArgs) -> str:
    if args.city == 'Mexico City':
        return 'sunny'
    else:
        return 'unknown'  # pragma: no cover


@dataclass
class Answer:
    label: str
    answer: str


@dataclass
class Response:
    answers: list[Answer]


complex_agent = Agent(
    model,
    deps_type=Deps,
    output_type=Response,
    toolsets=[
        FunctionToolset[Deps](tools=[get_country], id='country'),
        MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='mcp', init_timeout=20),
        ExternalToolset(tool_defs=[ToolDefinition(name='external')], id='external'),
    ],
    tools=[get_weather],
    name='complex_agent',
)


# This needs to be done before the `TemporalAgent` is bound to the workflow.
complex_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    complex_agent,
    event_stream_handler=event_stream_handler,
    activity_config=BASE_ACTIVITY_CONFIG,
    model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=90)),
    toolset_activity_config={
        'country': ActivityConfig(start_to_close_timeout=timedelta(seconds=120)),
    },
    tool_activity_config={
        'country': {
            'get_country': False,
        },
        'mcp': {
            'get_product_name': ActivityConfig(start_to_close_timeout=timedelta(seconds=150)),
        },
        '<agent>': {
            'get_weather': ActivityConfig(start_to_close_timeout=timedelta(seconds=180)),
        },
    },
)


@workflow.defn
class ComplexAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> Response:
        result = await complex_temporal_agent.run(prompt, deps=deps)
        return result.output


@dataclass
class BasicSpan:
    content: str
    children: list[BasicSpan] = field(default_factory=list['BasicSpan'])
    parent_id: int | None = field(repr=False, compare=False, default=None)


# --- DynamicToolset / @agent.toolset tests ---


@dataclass
class DynamicToolsetDeps:
    user_name: str


dynamic_toolset_agent = Agent(TestModel(), name='dynamic_toolset_agent', deps_type=DynamicToolsetDeps)


@dynamic_toolset_agent.toolset(id='my_dynamic_tools')
def my_dynamic_toolset(ctx: RunContext[DynamicToolsetDeps]) -> FunctionToolset[DynamicToolsetDeps]:
    toolset = FunctionToolset[DynamicToolsetDeps](id='dynamic_weather')

    @toolset.tool_plain
    def get_dynamic_weather(location: str) -> str:
        """Get the weather for a location."""
        user = ctx.deps.user_name
        return f'Weather in {location} for {user}: sunny.'

    return toolset


dynamic_toolset_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    dynamic_toolset_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


class CustomModelSettings(ModelSettings, total=False):
    custom_setting: str


model_settings = CustomModelSettings(max_tokens=123, custom_setting='custom_value')


def payload_limit_detail(size: int) -> str:
    """Temporal's own sentence inside the guard's message, which differs across the range we support.

    `temporalio` 1.31 moved the payload-size check out of the Python SDK and into Temporal's Rust core,
    which reports the breach without the byte counts the SDK's own check appended. Both shapes are inside
    the `>=1.24` range the `temporal` extra declares, so which one to expect is read off the installed
    SDK rather than pinned.
    """
    exceeded = '[TMPRL1103] Attempted to upload payloads with size that exceeded the error limit'
    if Version(version('temporalio')) >= Version('1.31'):
        return exceeded
    return f'{exceeded}. Size: {size} bytes, Limit: 2097152 bytes'


# Can't use the `openai_api_key` fixture here because the workflow needs to be defined at the top level of the file.
web_search_model = OpenAIResponsesModel(
    'gpt-5',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)


# ============================================================================
# Beta Graph API Tests - Tests for running pydantic-graph beta API in Temporal
# ============================================================================


@dataclass
class GraphState:
    """State for the graph execution test."""

    values: list[int] = field(default_factory=list[int])


# Create a graph with parallel execution using the beta API
graph_builder = GraphBuilder(
    name='parallel_test_graph',
    state_type=GraphState,
    input_type=int,
    output_type=list[int],
)


@graph_builder.step
async def source(ctx: StepContext[GraphState, None, int]) -> int:
    """Source step that passes through the input value."""
    return ctx.inputs


@graph_builder.step
async def multiply_by_two(ctx: StepContext[GraphState, None, int]) -> int:
    """Multiply input by 2."""
    return ctx.inputs * 2


@graph_builder.step
async def multiply_by_three(ctx: StepContext[GraphState, None, int]) -> int:
    """Multiply input by 3."""
    return ctx.inputs * 3


@graph_builder.step
async def multiply_by_four(ctx: StepContext[GraphState, None, int]) -> int:
    """Multiply input by 4."""
    return ctx.inputs * 4


# Create a join to collect results
result_collector = graph_builder.join(reduce_list_append, initial_factory=list[int])


# Build the graph with parallel edges (broadcast pattern)
graph_builder.add(
    graph_builder.edge_from(graph_builder.start_node).to(source),
    # Broadcast: send value to all three parallel steps
    graph_builder.edge_from(source).to(multiply_by_two, multiply_by_three, multiply_by_four),
    # Collect all results
    graph_builder.edge_from(multiply_by_two, multiply_by_three, multiply_by_four).to(result_collector),
    graph_builder.edge_from(result_collector).to(graph_builder.end_node),
)


parallel_test_graph = graph_builder.build()


# Module-level test models for error test
test_model_error_1 = TestModel()

test_model_error_2 = TestModel()


class _BuiltinToolModel(TestModel):
    SUPPORTED_TOOLS: frozenset[type[AbstractNativeTool]] = frozenset()

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return cls.SUPPORTED_TOOLS

    def _request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        # Override to skip TestModel._request's builtin tools rejection
        return ModelResponse(parts=[TextPart(self.custom_output_text or '')], model_name=self.model_name)


class _WebSearchOnlyModel(_BuiltinToolModel):
    SUPPORTED_TOOLS = frozenset({WebSearchTool})


class _CodeExecutionOnlyModel(_BuiltinToolModel):
    SUPPORTED_TOOLS = frozenset({CodeExecutionTool})


def _select_builtin_tool(ctx: RunContext[Any]) -> AbstractNativeTool:
    # `RunContext.model` is an `AbstractModel`; narrow to a request-response model to read its profile.
    ctx_model = ctx.model
    assert isinstance(ctx_model, Model)
    model = cast('Model[Any]', ctx_model)
    if WebSearchTool in model.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS):
        return WebSearchTool()
    return CodeExecutionTool()


web_search_builtin_model = _WebSearchOnlyModel(custom_output_text='search model', model_name='web-search')

code_execution_builtin_model = _CodeExecutionOnlyModel(custom_output_text='code model', model_name='code-exec')


# ==========================================
# TemporalDurability capability tests
# ==========================================


def _durability_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Simple model function for durability tests that echoes the last user prompt."""
    # The first message always carries the prompt and its first part is always the `UserPromptPart`, so none branch.
    for msg in reversed(messages):  # pragma: no branch
        for part in msg.parts:  # pragma: no branch
            if isinstance(part, UserPromptPart):  # pragma: no branch
                return ModelResponse(parts=[TextPart(content=f'Echo: {part.content}')])
    return ModelResponse(parts=[TextPart(content='no prompt')])  # pragma: no cover


_durability_fn_model = FunctionModel(_durability_model_fn)


# --- Streaming in workflow (event_stream_handler) ---


async def _stream_model_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'Stream'
    yield 'ed '
    yield 'response'


_stream_fn_model = FunctionModel(_durability_model_fn, stream_function=_stream_model_fn)


_stream_events_collected: list[AgentStreamEvent] = []
_stream_model_events_in_activity: list[bool] = []


async def _durability_event_stream_handler(
    ctx: RunContext[object],
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in stream:
        if isinstance(event, (PartStartEvent, PartDeltaEvent)):
            _stream_model_events_in_activity.append(activity.in_activity())
        _stream_events_collected.append(event)


_stream_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)

_stream_durable_agent = Agent(
    _stream_fn_model,
    name='durability_stream_agent',
    capabilities=[ProcessEventStream(_durability_event_stream_handler), _stream_durability],
)


@workflow.defn
class StreamDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> tuple[str, list[bool]]:
        result = await _stream_durable_agent.run(prompt)
        return result.output, _stream_model_events_in_activity


async def _durability_reveal_tool() -> ToolReturn[str]:
    return ToolReturn(return_value='handled', tools=['hidden_tool'])


# --- Continuation chains (suspended → complete) run one activity per segment ---
#
# When a model suspends a turn (Anthropic `pause_turn`, OpenAI background mode), the
# continuation loop in the innermost `model_request`/`model_request_stream` helpers runs
# workflow-side under `TemporalDurability`, dispatching each segment through its own
# model-request activity, so a failed segment retries alone and the suspended response is
# checkpointed in workflow history between segments. These tests use a scripted model (no
# cassettes: `FunctionModel` can't emit suspended streaming segments, and VCR matchers
# wouldn't pin the chain shape).


def _workflow_failure_cause(exc: WorkflowFailureError) -> ApplicationError:
    """The innermost `ApplicationError` of a workflow failure (walking through `ActivityError`)."""
    cause: BaseException | None = exc.__cause__
    while cause is not None and not isinstance(cause, ApplicationError):
        cause = cause.__cause__
    assert isinstance(cause, ApplicationError), f'expected ApplicationError in cause chain of {exc!r}'
    return cause


def _scheduled_activity_count(history: WorkflowHistory) -> int:
    return len([e for e in history.events if e.HasField('activity_task_scheduled_event_attributes')])
