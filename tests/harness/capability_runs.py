"""Running each capability case's calls, and what the model sees of the parts they add, for the conformance suites.

`run_native` runs a case's calls as plain tool calls, with every capability it builds lowered to
`with_authority('none')`, and `typed_history` reads back each part the calls added besides what a call returned:
its type, source, capability id, authority and frame. The route matrix (C1, `test_capability_routes.py`) compares that
across the routes a tool call can take, and the wire suite (`test_capability_voice_wire.py`) sends the history
`run_native` ends with to each provider.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pydantic_ai import Agent, AgentRunResult, RunContext, ToolReturn
from pydantic_ai.capabilities import AbstractCapability, Capability, ValidatedToolArgs
from pydantic_ai.messages import (
    CapabilityPart,
    InstructionCapabilityPart,
    ModelMessage,
    ModelRequest,
    OperatorSource,
    OperatorText,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.background_tools import BackgroundToolResultPart

from .. import capability_cases as core
from ..append_only import OneCallPerResponse, build_calls, discover_capabilities, drain_events
from ..capability_cases import CORE_CAPABILITY_PACKAGES, Calls, CapabilityCase
from .capability_cases import CASES as HARNESS_CASES, HARNESS_CAPABILITY_PACKAGES

CASES: dict[str, CapabilityCase] = {**core.CASES, **HARNESS_CASES}
PUBLIC = {**discover_capabilities(CORE_CAPABILITY_PACKAGES), **discover_capabilities(HARNESS_CAPABILITY_PACKAGES)}


PAYLOAD = 'the page says hello'
"""What every probe's calls put in the history, which each route must deliver unchanged."""


def fetch_page() -> ToolReturn:
    """Fetch a page, with what it says beside the return value."""
    return ToolReturn(return_value='ok', content=PAYLOAD)


def fetch_and_note(ctx: RunContext[None]) -> str:
    """Fetch a page, and note what it says for the next request."""
    ctx.enqueue(PAYLOAD)
    return 'ok'


@dataclass(kw_only=True)
class _ReminderPart(InstructionCapabilityPart, namespace='test_capability_routes', name='reminder'):
    """A reminder for the next request only."""


def fetch_and_remind(ctx: RunContext[None]) -> str:
    """Fetch a page, and remind the model of what it says for the next request only."""
    ctx.enqueue(_ReminderPart(content=OperatorText(PAYLOAD), source=OperatorSource(), scope='request'))
    return 'ok'


PROBES: dict[str, Calls] = {
    'probe:tool_return_content': Calls(
        (('fetch_page', {}),), lambda _: Capability[None](id='plugin', tools=[fetch_page])
    ),
    'probe:enqueued': Calls((('fetch_and_note', {}),), lambda _: Capability[None](id='plugin', tools=[fetch_and_note])),
    'probe:request_scoped': Calls(
        (('fetch_and_remind', {}),),
        lambda _: Capability[None](id='plugin', tools=[fetch_and_remind]),
        moves_cache_prefix='a request-scoped reminder is gone from the request after the one it was for',
    ),
}
"""Capabilities built only here, whose calls put more than their return value in the history, so every route has a
part to compare while few cases' calls do: one returns `ToolReturn.content`, one calls `ctx.enqueue` with text, and one
enqueues a request-scoped system prompt. What they put there is fixed, so their routes are also compared on content."""

SUBJECTS: dict[str, Calls] = {
    **{name: case.calls for name, case in sorted(CASES.items()) if name in PUBLIC and case.calls is not None},
    **PROBES,
}
"""What the conformance suites run: every public case with `calls`, then the probes."""


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
    scope: str | None = None
    """How long the model sees a capability part: a route mustn't turn a request-scoped part into a lasting one."""
    content: object = None
    """What a capability part tells the model, compared only for the probes, whose content is fixed."""


def seen_part(part: object, *, with_content: bool = False) -> Seen:
    """What the model sees of `part`."""
    if not isinstance(part, CapabilityPart):
        return Seen(type(part).__name__)
    return Seen(
        part.kind,
        part.source,
        part.capability_id,
        part.source.authority,
        (part.rendering.voice, part.rendering.tag),
        part.scope,
        part.model_content() if with_content else None,
    )


CARRIERS = (ToolReturnPart, BackgroundToolResultPart)
"""Parts that carry what a call returned, which each route wraps its own way."""


def typed_history(messages: Sequence[ModelMessage], *, with_content: bool = False) -> list[Seen]:
    """What the model sees of each part the calls put in the history, besides the prompt and what each call returned.

    `with_content` compares what each capability part says too, for calls whose content doesn't vary between runs.
    """
    requests = [message for message in messages if isinstance(message, ModelRequest)]
    return [
        seen_part(part, with_content=with_content)
        for request in requests[1:]
        for part in request.parts
        if not isinstance(part, CARRIERS)
    ]


@dataclass
class Observer(AbstractCapability[None]):
    """Records the tools each run executes and the history it ends with: a sub-agent's run doesn't return its own."""

    ran: list[str] = field(default_factory=list[str])
    histories: list[list[ModelMessage]] = field(default_factory=list[list[ModelMessage]])
    params: list[ModelRequestParameters] = field(default_factory=list[ModelRequestParameters])

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.params.append(request_context.model_request_parameters)
        return request_context

    async def before_tool_execute(
        self, ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        self.ran.append(call.tool_name)
        return args

    async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[object]) -> AgentRunResult[object]:
        self.histories.append(result.all_messages())
        return result


@dataclass
class RouteRun:
    """What a case's calls did on one route."""

    messages: list[ModelMessage]
    ran: list[str]
    """The tools the calls executed, in order, leaving out the route's own dispatchers."""
    params: ModelRequestParameters | None = None
    """The parameters of the run's last model request, such as the tools it offered."""


async def build_lowered(calls: Calls, directory: Path) -> tuple[list[AbstractCapability[None]], Observer]:
    capabilities = await build_calls(calls.build, directory)
    return [capability.with_authority('none') for capability in capabilities], Observer()


def tools_ran(calls: Calls, observer: Observer) -> list[str]:
    names = {name for name, _ in calls.made}
    return [name for name in observer.ran if name in names]


async def run_native(calls: Calls, directory: Path) -> RouteRun:
    """What `calls` do as plain tool calls."""
    capabilities, observer = await build_lowered(calls, directory)
    agent = Agent(OneCallPerResponse(calls.made).model, deps_type=type(None), capabilities=[*capabilities, observer])
    result = await agent.run('go', event_stream_handler=drain_events)
    return RouteRun(result.all_messages(), tools_ran(calls, observer), observer.params[-1])


def case_calls(name: str) -> Calls:
    """The calls of the case or probe named `name`, skipping a case whose extra isn't installed."""
    if (probe := PROBES.get(name)) is not None:
        return probe
    case = CASES[name]
    if case.requires is not None:
        pytest.importorskip(case.requires)
    assert case.calls is not None
    return case.calls
