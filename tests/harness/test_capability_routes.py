"""C1, the route matrix: each case's calls reach the model the same on every route a tool call can take.

Each case's `calls` run with every capability it builds lowered to `with_authority('none')`, once as plain tool calls
(the native route) and once through each other route in `ROUTES`. What a call returns travels in a carrier of the
route's own, such as the `run_code` result or a background result, so the check compares everything else the calls
put in the history, including the content a carrier keeps with what a call returned (`ToolReturnPart.attachments`),
as the parts it reaches the model as: each part's type, source, capability id, authority, frame and scope must be the
native route's, and for `PROBES`, whose calls put fixed content there, its content too. The same tools must run, so a
call a route never makes can't pass.

A route that breaks this for a case is a known finding, marked `xfail(strict=True)` in `KNOWN` with what fixes it, so
the fix fails the suite until its entry is removed. A route that doesn't apply to a case is skipped with its reason.
To add a route, add it to `Route` and give `_through` the way to run calls on it.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Literal, get_args

import anyio
import pytest
from _pytest.mark import ParameterSet

from pydantic_ai import Agent, AgentRunResult, RunContext, ToolReturn
from pydantic_ai.capabilities import AbstractCapability, Capability, PrepareTools, ValidatedToolArgs
from pydantic_ai.messages import (
    CapabilityPart,
    CapabilitySource,
    InstructionCapabilityPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    OperatorSource,
    OperatorText,
    StructuredContent,
    ToolCallPart,
    ToolResultAttachment,
    ToolReturnPart,
    UserContent,
)
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness import BackgroundTools, CodeMode, SubAgent, SubAgents, ToolBridge
from pydantic_ai_harness.background_tools import BackgroundToolResultPart

from .. import capability_cases as core
from ..append_only import (
    OneCallPerResponse,
    ToolArgs,
    ToolCall,
    discover_capabilities,
    drain_events,
    run_through_ui,
    ui_imports_successful,
)
from ..capability_cases import CORE_CAPABILITY_PACKAGES, Calls, CapabilityCase
from ..conftest import try_import
from .capability_cases import CASES as HARNESS_CASES, HARNESS_CAPABILITY_PACKAGES

with try_import() as ipython_installed:
    from pydantic_ai_harness.ipython_code_mode import IPythonCodeMode, IPythonKernel

CASES: dict[str, CapabilityCase] = {**core.CASES, **HARNESS_CASES}
PUBLIC = {**discover_capabilities(CORE_CAPABILITY_PACKAGES), **discover_capabilities(HARNESS_CAPABILITY_PACKAGES)}

Route = Literal[
    'code_mode',
    'ipython_code_mode',
    'tool_bridge',
    'background_tools',
    'sub_agent',
    'prepare_hook',
    'type_adapter',
    'vercel_ai',
    'ag_ui',
]
"""Every route besides the native one, which each is compared with.

- `code_mode`, `ipython_code_mode`, `tool_bridge`: each call is made from `run_code` or `tool_call`, unless the route
  keeps the tool native.
- `background_tools`: `BackgroundTools(tools='all')` runs each call as a background task.
- `sub_agent`: the parent delegates to a sub-agent that `SubAgents(shared_capabilities=...)` gives the capabilities,
  and the sub-agent's history is compared.
- `prepare_hook`: a `PrepareTools` hook builds each tool definition anew, leaving out the ids the framework stamps.
- `type_adapter`: the native history, dumped and loaded with `ModelMessagesTypeAdapter`.
- `vercel_ai`, `ag_ui`: the calls run through the protocol's adapter with `manage_message_history='server'`, then a
  second run sends the whole conversation back and the server-held history is compared.

Temporal and DBOS are durable routes, which D1 covers per case.
"""

ROUTES: tuple[Route, ...] = get_args(Route)


def _python_call(name: str, args: ToolArgs) -> str:
    return f'{name}({", ".join(f"{key}={value!r}" for key, value in args.items())})'


def _code_mode_call(name: str, args: ToolArgs, info: AgentInfo) -> ToolCall:
    """Code Mode's catalog renders a sequential tool as a sync function, and the rest as async ones."""
    sync = re.search(rf'(?<!async )def {re.escape(name)}\(', info.instructions or '') is not None
    return 'run_code', {'code': f'{"" if sync else "await "}{_python_call(name, args)}'}


_WRAPS: dict[Route, Callable[[str, ToolArgs, AgentInfo], ToolCall]] = {
    'code_mode': _code_mode_call,
    'ipython_code_mode': lambda name, args, info: ('run_code', {'code': _python_call(name, args)}),
    'tool_bridge': lambda name, args, info: ('tool_call', {'name': name, 'args_json': json.dumps(args)}),
}
"""How a route that dispatches calls itself makes one, given what the model is offered."""


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

_ISSUE = 'https://github.com/pydantic/pydantic-ai/issues/'
_PLAN_IN_CARRIER = (
    f"{_ISSUE}14441: `Planning` finds the plan a plan tool returned only in that tool's own `ToolReturnPart`, so after a plan tool "
    'runs inside another carrier it records the plan again in a `<plan_reminder>`'
)

KNOWN: dict[tuple[str, Route], tuple[str, type[Exception]]] = {
    ('Planning', 'code_mode'): (_PLAN_IN_CARRIER, AssertionError),
    ('Planning', 'ipython_code_mode'): (_PLAN_IN_CARRIER, AssertionError),
    ('Planning', 'background_tools'): (_PLAN_IN_CARRIER, AssertionError),
}
"""A route on which a case's calls reach the model differently than natively, with what fixes it and the error the
check fails with."""


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


def _said(content: str | StructuredContent | Sequence[UserContent]) -> object:
    """`content` as the model reads it: one item alone reads the same as that item."""
    if isinstance(content, str | StructuredContent):
        return content
    items = tuple(content)
    return items[0] if len(items) == 1 else items


def _seen(part: CapabilityPart, *, with_content: bool) -> Seen:
    return Seen(
        part.kind,
        part.source,
        part.capability_id,
        part.source.authority,
        (part.rendering.voice, part.rendering.tag),
        part.scope,
        _said(part.model_content()) if with_content else None,
    )


_CARRIERS = (ToolReturnPart, BackgroundToolResultPart)
"""Parts that carry what a call returned, which each route wraps its own way."""


def _attachments_seen(attachments: Sequence[ToolResultAttachment], *, with_content: bool) -> Iterator[Seen]:
    """What the model sees of a tool result's attachments, which are always data.

    Each run of items one call made reaches the model as one framed part: content the capability that contributed the
    tool would enqueue at `'none'`. Every case's tools come from a capability.
    """
    for (_, _, capability_id), group in groupby(
        attachments, key=lambda attachment: (attachment.tool_call_id, attachment.tool_name, attachment.capability_id)
    ):
        assert capability_id is not None
        items = [attachment.content for attachment in group]
        yield Seen(
            'pydantic_ai.capability_content',
            CapabilitySource(capability_id, authority='none'),
            capability_id,
            'none',
            ('data', 'capability_content'),
            'conversation',
            _said(items) if with_content else None,
        )


def _parts_seen(part: object, *, with_content: bool) -> Iterator[Seen]:
    """What the model sees of `part`: a carrier's attachments, or the part."""
    if isinstance(part, ToolReturnPart):
        yield from _attachments_seen(part.attachments, with_content=with_content)
    elif not isinstance(part, _CARRIERS):
        # Besides the prompt and the carriers, the calls put only capability parts in the history.
        assert isinstance(part, CapabilityPart), part
        yield _seen(part, with_content=with_content)


def typed_history(messages: Sequence[ModelMessage], *, with_content: bool = False) -> list[Seen]:
    """What the model sees of each part the calls put in the history, besides the prompt and what each call returned.

    `with_content` compares what each capability part says too, for calls whose content doesn't vary between runs.
    """
    requests = [message for message in messages if isinstance(message, ModelRequest)]
    return [
        seen
        for request in requests[1:]
        for part in request.parts
        for seen in _parts_seen(part, with_content=with_content)
    ]


@dataclass
class _Observer(AbstractCapability[None]):
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


async def _rebuild_definitions(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    """A prepare hook that builds each definition anew from the fields a user sets, leaving out the stamped ids."""
    return [
        ToolDefinition(
            name=tool_def.name,
            parameters_json_schema=tool_def.parameters_json_schema,
            description=tool_def.description,
            outer_typed_dict_key=tool_def.outer_typed_dict_key,
            strict=tool_def.strict,
            sequential=tool_def.sequential,
            kind=tool_def.kind,
            metadata=tool_def.metadata,
            timeout=tool_def.timeout,
            durable=tool_def.durable,
            defer_loading=tool_def.defer_loading,
            unless_native=tool_def.unless_native,
            with_native=tool_def.with_native,
            return_schema=tool_def.return_schema,
            include_return_schema=tool_def.include_return_schema,
            ui_visibility=tool_def.ui_visibility,
        )
        for tool_def in tool_defs
    ]


@dataclass
class RouteRun:
    """What a case's calls did on one route."""

    messages: list[ModelMessage]
    ran: list[str]
    """The tools the calls executed, in order, leaving out the route's own dispatchers."""


def _lowered(calls: Calls, directory: Path) -> list[AbstractCapability[None]]:
    built = calls.build(directory)
    capabilities = [built] if isinstance(built, AbstractCapability) else list(built)
    return [capability.with_authority('none') for capability in capabilities]


async def _built(calls: Calls, directory: Path) -> tuple[list[AbstractCapability[None]], _Observer]:
    directory.mkdir()
    # Built off the event loop, as an app builds its agent: building can import a client library.
    capabilities = await anyio.to_thread.run_sync(_lowered, calls, directory)
    return capabilities, _Observer()


def _ran(calls: Calls, observer: _Observer) -> list[str]:
    names = {name for name, _ in calls.made}
    return [name for name in observer.ran if name in names]


async def _native(calls: Calls, directory: Path) -> RouteRun:
    capabilities, observer = await _built(calls, directory)
    agent = Agent(
        OneCallPerResponse(calls.made, unstreamed=True).model,
        deps_type=type(None),
        capabilities=[*capabilities, observer],
    )
    result = await agent.run('go', event_stream_handler=drain_events)
    return RouteRun(result.all_messages(), _ran(calls, observer))


async def _through(route: Route, calls: Calls, directory: Path, kernel: IPythonKernel | None = None) -> RouteRun:
    """What the calls do on `route`."""
    if route == 'type_adapter':
        native = await _native(calls, directory)
        loaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(native.messages))
        return RouteRun(loaded, native.ran)
    capabilities, observer = await _built(calls, directory)
    script = OneCallPerResponse(calls.made, wrap=_WRAPS.get(route), unstreamed=True)
    if route in ('vercel_ai', 'ag_ui'):
        agent = Agent(script.model, deps_type=type(None), capabilities=[*capabilities, observer])
        protocol: Literal['vercel', 'agui'] = 'vercel' if route == 'vercel_ai' else 'agui'
        first = (await run_through_ui(protocol, agent, 'go')).all_messages()
        # The client sends the whole conversation back, and the server keeps the history it stored.
        second = await run_through_ui(protocol, agent, 'next', message_history=first)
        return RouteRun(second.all_messages()[: len(first)], _ran(calls, observer))
    if route == 'sub_agent':
        child = Agent(script.model, deps_type=type(None), name='child', capabilities=[observer])
        parent = Agent(
            OneCallPerResponse([('delegate_task', {'agent_name': 'child', 'task': 'go'})], unstreamed=True).model,
            deps_type=type(None),
            capabilities=[SubAgents(agents=[SubAgent(child)], shared_capabilities=capabilities)],
        )
        await parent.run('go', event_stream_handler=drain_events)
        [history] = observer.histories
        return RouteRun(history, _ran(calls, observer))
    around: dict[Route, Callable[[], AbstractCapability[None]]] = {
        'code_mode': CodeMode,
        'ipython_code_mode': lambda: IPythonCodeMode(kernel=kernel),
        'tool_bridge': ToolBridge,
        'background_tools': lambda: BackgroundTools(tools='all'),
        'prepare_hook': lambda: PrepareTools(_rebuild_definitions),
    }
    agent = Agent(script.model, deps_type=type(None), capabilities=[*capabilities, around[route](), observer])
    result = await agent.run('go', event_stream_handler=drain_events)
    if route in _WRAPS and not script.wrapped:
        pytest.skip(f'{route} keeps every tool the calls make native')
    return RouteRun(result.all_messages(), _ran(calls, observer))


def _matrix(routes: Sequence[Route]) -> Iterator[ParameterSet]:
    subjects: dict[str, Calls] = {
        **{name: case.calls for name, case in sorted(CASES.items()) if name in PUBLIC and case.calls is not None},
        **PROBES,
    }
    for name, calls in subjects.items():
        for route in routes:
            marks: list[pytest.MarkDecorator] = []
            if (known := KNOWN.get((name, route))) is not None:
                reason, raises = known
                marks.append(pytest.mark.xfail(strict=True, raises=raises, reason=reason))
            if route in ('vercel_ai', 'ag_ui') and not ui_imports_successful():
                marks.append(pytest.mark.skip(reason='the UI adapters are not installed'))
            if route == 'ag_ui':
                # It loses only what the client sends back; the server-held history this route compares keeps it.
                marks.append(pytest.mark.filterwarnings('ignore:ag-ui-protocol .* predates the `encrypted_value`'))
            if calls.moves_cache_prefix is not None:
                marks.append(pytest.mark.moves_cache_prefix(reason=calls.moves_cache_prefix))
            yield pytest.param(name, route, marks=marks, id=f'{name}-{route}')


def _calls(name: str) -> Calls:
    if (probe := PROBES.get(name)) is not None:
        return probe
    case = CASES[name]
    if case.requires is not None:
        pytest.importorskip(case.requires)
    assert case.calls is not None
    return case.calls


async def _assert_same(name: str, route: Route, directory: Path, kernel: IPythonKernel | None = None) -> None:
    calls = _calls(name)
    native = await _native(calls, directory / 'native')
    assert native.ran, 'the calls ran no tool'
    routed = await _through(route, calls, directory / route, kernel)
    with_content = name in PROBES
    expected = typed_history(native.messages, with_content=with_content)
    actual = typed_history(routed.messages, with_content=with_content)
    if route == 'background_tools':
        # Background tasks run concurrently, so they end, and report, in any order.
        native.ran.sort()
        routed.ran.sort()
        expected.sort(key=repr)
        actual.sort(key=repr)
    assert routed.ran == native.ran, f'on {route} the calls ran {routed.ran}, not {native.ran}'
    assert actual == expected


@pytest.mark.parametrize(('name', 'route'), list(_matrix([route for route in ROUTES if route != 'ipython_code_mode'])))
async def test_calls_reach_the_model_the_same_on_every_route(name: str, route: Route, tmp_path: Path) -> None:
    await _assert_same(name, route, tmp_path)


@pytest.fixture(scope='module')
def ipython_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Gives the kernel its own IPython profile, so it shares no history file with other processes."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv('IPYTHONDIR', str(tmp_path_factory.mktemp('ipython')))
        yield


@pytest.fixture(scope='module')
async def ipython_kernel(ipython_dir: None) -> AsyncIterator[IPythonKernel]:
    """One kernel for every case, which keeps the route to a single process launch."""
    async with IPythonKernel() as kernel:
        yield kernel


@pytest.mark.skipif(not ipython_installed(), reason='jupyter-client not installed')
@pytest.mark.xdist_group(name='harness-ipython-kernel')
@pytest.mark.parametrize(('name', 'route'), list(_matrix(['ipython_code_mode'])))
async def test_calls_reach_the_model_the_same_through_ipython_code_mode(
    name: str, route: Route, tmp_path: Path, ipython_kernel: IPythonKernel
) -> None:
    await _assert_same(name, route, tmp_path, ipython_kernel)


def test_every_known_entry_names_a_case_and_route() -> None:
    """An entry for a case that no longer has calls, or a route that's gone, would hide the next finding there."""
    subjects = {name for name, case in CASES.items() if name in PUBLIC and case.calls is not None} | set(PROBES)
    assert [key for key in KNOWN if key[0] not in subjects or key[1] not in ROUTES] == []


async def test_the_native_route_attributes_the_probes(tmp_path: Path) -> None:
    """The baseline the probes' routes are compared with: `plugin`'s content at `'none'`, as written."""
    for name, calls in PROBES.items():
        [seen] = typed_history((await _native(calls, tmp_path / name)).messages, with_content=True)
        assert seen.content == PAYLOAD
        assert seen.scope == ('request' if name == 'probe:request_scoped' else 'conversation')
        assert seen.source == CapabilitySource('plugin', authority='none')
        # A reminder keeps its instruction voice, rendered with a note that it has no authority; the others are data.
        expected_frame = ('instruction', None) if name == 'probe:request_scoped' else ('data', 'capability_content')
        assert seen.frame == expected_frame
