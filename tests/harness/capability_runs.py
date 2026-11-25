"""Running each capability case's calls, and what the model sees of the parts they add, for the conformance suites.

`run_native` runs a case's calls as plain tool calls, with every capability it builds lowered to
`with_authority('none')`, and `typed_history` reads back each part the calls added besides what a call returned:
its type, source, capability id, authority and frame. The route matrix (C1, `test_capability_routes.py`) compares that
across the routes a tool call can take, and the wire suite (`test_capability_voice_wire.py`) sends the history
`run_native` ends with to each provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, AgentRunResult, RunContext, ToolReturn
from pydantic_ai.capabilities import AbstractCapability, Capability, ValidatedToolArgs
from pydantic_ai.messages import (
    CapabilityPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.background_tools import BackgroundToolResultPart

from .. import capability_cases as core
from ..append_only import ToolArgs, ToolCall, discover_capabilities, drain_events
from ..capability_cases import CORE_CAPABILITY_PACKAGES, Calls, CapabilityCase
from .capability_cases import CASES as HARNESS_CASES, HARNESS_CAPABILITY_PACKAGES

CASES: dict[str, CapabilityCase] = {**core.CASES, **HARNESS_CASES}
PUBLIC = {**discover_capabilities(CORE_CAPABILITY_PACKAGES), **discover_capabilities(HARNESS_CAPABILITY_PACKAGES)}


def fetch_page() -> ToolReturn:
    """Fetch a page, with what it says beside the return value."""
    return ToolReturn(return_value='ok', content='the page says hello')


def fetch_and_note(ctx: RunContext[None]) -> str:
    """Fetch a page, and note what it says for the next request."""
    ctx.enqueue('the page says hello')
    return 'ok'


PROBES: dict[str, Calls] = {
    'probe:tool_return_content': Calls(
        (('fetch_page', {}),), lambda _: Capability[None](id='plugin', tools=[fetch_page])
    ),
    'probe:enqueued': Calls((('fetch_and_note', {}),), lambda _: Capability[None](id='plugin', tools=[fetch_and_note])),
}
"""Capabilities built only here, whose calls put more than their return value in the history, so every route has a
part to compare while few cases' calls do: one returns `ToolReturn.content`, the other calls `ctx.enqueue`."""


@dataclass(frozen=True)
class Seen:
    """What the model sees of one part the calls put in the history, besides what a call returned."""

    part: str
    """The part's type, or its kind for a capability part."""
    source: object = None
    capability_id: str | None = None
    authority: str | None = None
    frame: tuple[str, str | None] | None = None
    """A capability part's voice and tag."""


def seen_part(part: object) -> Seen:
    """What the model sees of `part`."""
    if not isinstance(part, CapabilityPart):
        return Seen(type(part).__name__)
    return Seen(
        part.kind, part.source, part.capability_id, part.source.authority, (part.rendering.voice, part.rendering.tag)
    )


CARRIERS = (ToolReturnPart, BackgroundToolResultPart)
"""Parts that carry what a call returned, which each route wraps its own way."""


def typed_history(messages: Sequence[ModelMessage]) -> list[Seen]:
    """What the model sees of each part the calls put in the history, besides the prompt and what each call returned."""
    requests = [message for message in messages if isinstance(message, ModelRequest)]
    return [seen_part(part) for request in requests[1:] for part in request.parts if not isinstance(part, CARRIERS)]


@dataclass
class Observer(AbstractCapability[None]):
    """Records the tools each run executes and the history it ends with: a sub-agent's run doesn't return its own."""

    ran: list[str] = field(default_factory=list[str])
    histories: list[list[ModelMessage]] = field(default_factory=list[list[ModelMessage]])

    async def before_tool_execute(
        self, ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        self.ran.append(call.tool_name)
        return args

    async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[object]) -> AgentRunResult[object]:
        self.histories.append(result.all_messages())
        return result


@dataclass
class Script:
    """A model that makes the calls in order, one per response, then answers.

    With `wrap`, a call to a tool the model isn't offered directly goes through the route's dispatcher.
    """

    calls: Sequence[ToolCall]
    wrap: Callable[[str, ToolArgs, AgentInfo], ToolCall] | None = None
    wrapped: int = 0

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step >= len(self.calls):
            return ModelResponse(parts=[TextPart('done')])
        name, args = self.calls[step]
        if self.wrap is not None and name not in {tool.name for tool in info.function_tools}:
            self.wrapped += 1
            name, args = self.wrap(name, args, info)
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=f'call_{step}')])

    async def stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        # Streamed, so a tool that emits a capability event has a handler to deliver it to.
        [part] = self.respond(messages, info).parts
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {0: DeltaToolCall(part.tool_name, part.args_as_json_str(), tool_call_id=part.tool_call_id)}

    @property
    def model(self) -> FunctionModel:
        return FunctionModel(self.respond, stream_function=self.stream)


@dataclass
class RouteRun:
    """What a case's calls did on one route."""

    messages: list[ModelMessage]
    ran: list[str]
    """The tools the calls executed, in order, leaving out the route's own dispatchers."""


def lowered(calls: Calls, directory: Path) -> list[AbstractCapability[None]]:
    built = calls.build(directory)
    capabilities = [built] if isinstance(built, AbstractCapability) else list(built)
    return [capability.with_authority('none') for capability in capabilities]


async def build_lowered(calls: Calls, directory: Path) -> tuple[list[AbstractCapability[None]], Observer]:
    directory.mkdir()
    # Built off the event loop, as an app builds its agent: building can import a client library.
    capabilities = await anyio.to_thread.run_sync(lowered, calls, directory)
    return capabilities, Observer()


def tools_ran(calls: Calls, observer: Observer) -> list[str]:
    names = {name for name, _ in calls.made}
    return [name for name in observer.ran if name in names]


async def run_native(calls: Calls, directory: Path) -> RouteRun:
    """What `calls` do as plain tool calls."""
    capabilities, observer = await build_lowered(calls, directory)
    agent = Agent(Script(calls.made).model, deps_type=type(None), capabilities=[*capabilities, observer])
    result = await agent.run('go', event_stream_handler=drain_events)
    return RouteRun(result.all_messages(), tools_ran(calls, observer))


def case_calls(name: str) -> Calls:
    """The calls of the case or probe named `name`, skipping a case whose extra isn't installed."""
    if (probe := PROBES.get(name)) is not None:
        return probe
    case = CASES[name]
    if case.requires is not None:
        pytest.importorskip(case.requires)
    assert case.calls is not None
    return case.calls
