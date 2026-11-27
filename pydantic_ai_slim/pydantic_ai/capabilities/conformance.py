"""Conformance data for Pydantic AI capabilities: tool calls that run each capability's tools offline, and scenarios.

A suite that runs every capability under some machinery, such as
[`DurableEngineSuite`][pydantic_ai.durable_exec.conformance.DurableEngineSuite] under a durable engine, reads its
cases from here: `CALLS` holds, by capability class name, the tool calls a scripted model makes and what they run
against, and `SCENARIO_RUNS` holds runs that check more than one call per response.
[`pydantic_ai_harness.conformance`](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/conformance.py)
holds the same for the Pydantic AI Harness capabilities.

`Calls`, `CapabilityBuilder`, `ScenarioRun` and `ToolCall` are stable in 2.x, and so are the case and scenario
names, which are renamed only with a deprecation alias. Minor releases can add cases and scenarios. The stand-ins a
`build` uses are internal: what they are and their names can change.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any, Protocol

from ..messages import CapabilityEvent, CustomEvent, ModelRequest, ToolReturnPart
from ..models import ModelRequestContext
from ..tools import RunContext
from ..toolsets import AbstractToolset, FunctionToolset
from . import (
    MCP,
    AbstractCapability,
    IncludeToolReturnSchemas,
    Instrumentation,
    PrefixTools,
    SetToolMetadata,
    Toolset,
    durable_operation,
)

__all__ = ('CALLS', 'SCENARIO_RUNS', 'CapabilityBuilder', 'Calls', 'ScenarioRun', 'ToolCall')

ToolCall = tuple[str, dict[str, Any]]
"""A tool call a scripted model makes: the tool's name and its arguments."""


class CapabilityBuilder(Protocol):
    """Builds what a case's calls run against.

    That's the capability, any it needs, such as a workspace, and any toolset the agent has directly, as a function
    tool the user wrote is.
    """

    def __call__(self, directory: Path, /) -> Sequence[AbstractCapability[None] | AbstractToolset[None]]:
        """The capabilities and toolsets the calls run against, built offline in `directory`.

        `directory` is a new, empty directory the build may write files to, such as the files a tool reads, and
        point a workspace at. The build needs nothing else: no network, no credentials, and no state of the test
        that runs it. A toolset it builds has an `id`, which a durable engine names its units after.
        """
        ...


@dataclass(frozen=True)
class Calls:
    """Tool calls that run a capability's tools offline, made one per model response, in order.

    The scripted model that makes them answers once it has made the last, and answers every request past them.
    """

    made: tuple[ToolCall, ...]
    """The tool calls the scripted model makes, one per response, before it answers."""
    _: KW_ONLY
    build: CapabilityBuilder
    """What the calls run against, offline: the capability, with stand-ins for any client its tools send requests
    through."""
    follow_ups: int = 0
    """How many model requests the calls cause beyond one per call: the results that arrive as follow-up messages,
    such as a backgrounded call's result, which `BackgroundTools` delivers in a later request.

    It's a total for all the calls. `DurableEngineSuite` counts it: a run makes one model request per call, one per
    follow-up and one to answer. The count is part of this stable contract.
    """


@dataclass(frozen=True)
class ScenarioRun:
    """A run that checks more than one call per response: its calls, and what a correct run observes."""

    calls: Calls
    """The calls the scenario's scripted model makes, and what they run against."""
    _: KW_ONLY
    observed: tuple[str, ...]
    """What a correct run observes, each once, in order: `event:<text>` per custom or capability event,
    `enqueued:<text>` per enqueued message's event, and `message:<text>` per enqueued message in the history."""


def _echo_toolset(*, background: bool = False) -> FunctionToolset[None]:
    """A toolset with one `echo` tool, with the `id` a durable engine names its units after."""

    def echo(text: str) -> str:
        """Return `text`."""
        return text

    return FunctionToolset([echo], id='echo', metadata={'background': True} if background else None)


_ECHO: tuple[ToolCall, ...] = (('echo', {'text': 'hi'}),)
"""A call to the `echo` tool `_with_echo` adds."""


def _with_echo(build: Callable[[], AbstractCapability[None]]) -> CapabilityBuilder:
    """The capability beside a toolset with one `echo` tool, for a capability whose hooks run around other tools."""
    return lambda _: [build(), Toolset(_echo_toolset())]


def _mcp(_: Path) -> list[AbstractCapability[None]]:
    """`MCP` with an in-process server, which has instructions and one tool. Needs `fastmcp`'s server."""
    from fastmcp.server import FastMCP

    from ..mcp import MCPToolset

    server: FastMCP[None] = FastMCP('extra', instructions='Extra server instructions.')

    @server.tool()
    def extra_tool() -> str:
        return 'extra'

    return [MCP(local=MCPToolset(server, id='extra'))]


CALLS: Mapping[str, Calls] = {
    'IncludeToolReturnSchemas': Calls(_ECHO, build=_with_echo(IncludeToolReturnSchemas)),
    'Instrumentation': Calls(_ECHO, build=_with_echo(Instrumentation)),
    'MCP': Calls((('extra_tool', {}),), build=_mcp),
    'PrefixTools': Calls(
        (('extra_echo', {'text': 'hi'}),), build=lambda _: [PrefixTools(Toolset(_echo_toolset()), prefix='extra')]
    ),
    'SetToolMetadata': Calls(_ECHO, build=_with_echo(lambda: SetToolMetadata(source='case'))),
    'Toolset': Calls(_ECHO, build=lambda _: [Toolset(_echo_toolset())]),
}
"""The calls that run each core capability's tools offline, keyed by capability class name."""


@dataclass(kw_only=True)
class _ScenarioNote(CustomEvent, name='scenario_note'):
    """What a scenario's function tool emits: an application event."""

    text: str


@dataclass(kw_only=True)
class _ScenarioCapabilityNote(CapabilityEvent, namespace='scenario', name='scenario_capability_note'):
    """What a scenario's capability emits, from a hook or a durable unit."""

    text: str


def _scenario_tool(*, emit: bool = False, enqueue: bool = False) -> FunctionToolset[None]:
    """A toolset with one `work` tool, which emits or enqueues from where a function tool runs."""
    toolset = FunctionToolset[None](id='scenario')

    @toolset.tool
    async def work(ctx: RunContext[None]) -> str:
        if emit:
            await ctx.emit(_ScenarioNote(text='tool'))
        if enqueue:
            ctx.enqueue('note from the tool')
        return 'worked'

    return toolset


def _after_the_tool(ctx: RunContext[None]) -> bool:
    """Whether the request about to be made is the one after `work` returned: true once per run, deterministically."""
    last = ctx.messages[-1] if ctx.messages else None
    return isinstance(last, ModelRequest) and any(
        isinstance(part, ToolReturnPart) and part.tool_name == 'work' for part in last.parts
    )


@dataclass
class _InUnit(AbstractCapability[None]):
    """Emits or enqueues inside a `@durable_operation`, before the request after `work`."""

    emit: bool = False
    enqueue: bool = False
    id: str | None = 'in_unit'

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if _after_the_tool(ctx):
            await self.act(ctx)
        return request_context

    @durable_operation('act')
    async def act(self, ctx: RunContext[None]) -> None:
        if self.emit:
            await ctx.emit(_ScenarioCapabilityNote(text='unit'))
        if self.enqueue:
            ctx.enqueue('note from the unit')


@dataclass
class _FromHook(AbstractCapability[None]):
    """Emits and enqueues from a hook, in workflow code, before the request after `work`."""

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if _after_the_tool(ctx):
            await ctx.emit(_ScenarioCapabilityNote(text='hook'))
            ctx.enqueue('note from the hook')
        return request_context


def _scenario(
    build: Callable[[], Sequence[AbstractCapability[None] | AbstractToolset[None]]], *observed: str
) -> ScenarioRun:
    """A scenario whose model makes one `work` call, a function tool of the agent's own, then answers."""
    return ScenarioRun(Calls((('work', {}),), build=lambda _: build()), observed=observed)


SCENARIO_RUNS: Mapping[str, ScenarioRun] = {
    'RunContext.emit_from_tool': _scenario(lambda: [_scenario_tool(emit=True)], 'event:tool'),
    'RunContext.enqueue_from_tool': _scenario(
        lambda: [_scenario_tool(enqueue=True)], 'enqueued:note from the tool', 'message:note from the tool'
    ),
    'RunContext.emit_in_unit': _scenario(lambda: [_scenario_tool(), _InUnit(emit=True)], 'event:unit'),
    'RunContext.enqueue_in_unit': _scenario(
        lambda: [_scenario_tool(), _InUnit(enqueue=True)], 'enqueued:note from the unit', 'message:note from the unit'
    ),
    'RunContext.emit_enqueue_from_hook': _scenario(
        lambda: [_scenario_tool(), _FromHook()],
        'event:hook',
        'enqueued:note from the hook',
        'message:note from the hook',
    ),
}
"""Runs that check more than one call per response, keyed `<owner>.<scenario>`.

The `RunContext` ones call [`ctx.emit`][pydantic_ai.tools.RunContext.emit] and
[`ctx.enqueue`][pydantic_ai.tools.RunContext.enqueue] from a function tool, from inside a durable unit (a
[`@durable_operation`][pydantic_ai.capabilities.durable_operation]), and from a hook.
"""
