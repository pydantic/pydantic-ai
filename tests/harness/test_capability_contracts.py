"""The contracts in `tests/capability_cases.py`, run over every core and Harness case they apply to.

They run here, where the Harness is installed, because its cases are most of what they apply to, over the public
capabilities the suites build. A case a contract exempts must still break it, so fixing the capability fails the
suite until the exemption is removed.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeVar

import anyio
import pytest
from _pytest.mark import ParameterSet

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, LocalWorkspace, Toolset
from pydantic_ai.capabilities.abstract import leaf_capabilities
from pydantic_ai.exceptions import UsageLimitExceeded, UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    Authority,
    CapabilityPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.usage import UsageLimits
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.memory import FileMemoryStore, Memory
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.skills import Skills
from pydantic_ai_harness.sub_agents import DynamicAgents, SubAgent, SubAgents

from .. import capability_cases as core
from ..append_only import ToolCall, build_capabilities, discover_capabilities, drain_events, skip_unless_installed
from ..capability_cases import (
    COMBINE_FIXED_POINT,
    CONTRACTS as CORE_CONTRACTS,
    CORE_CAPABILITY_PACKAGES,
    CUSTOM_ID,
    FORWARDS_USAGE,
    RENAMED,
    UNSET_ID,
    WORKSPACE_FROM_FUNCTION,
    Anonymous,
    Calls,
    CapabilityCase,
    Combines,
    Contract,
)
from ..function_model_prefix import RequestRecorder, patch_models
from ._authority import untrusted
from ._mcp_server import MCPServer
from .capability_cases import (
    CASES as HARNESS_CASES,
    CONTRACTS as HARNESS_CONTRACTS,
    HARNESS_CAPABILITY_PACKAGES,
    HOSTED_MCP_CONNECTION,
    hosted_server,
)

DepsT = TypeVar('DepsT')

CASES: dict[str, CapabilityCase] = {**core.CASES, **HARNESS_CASES}
CONTRACTS: tuple[Contract, ...] = (*CORE_CONTRACTS, *HARNESS_CONTRACTS)
PUBLIC = {**discover_capabilities(CORE_CAPABILITY_PACKAGES), **discover_capabilities(HARNESS_CAPABILITY_PACKAGES)}
"""The public capabilities, which the suites build; a private class only has to say what two of it mean."""


def _cases(contract: Contract) -> list[ParameterSet]:
    """The cases `contract` applies to, each whose calls rewrite history marked `moves_cache_prefix`."""
    return [
        pytest.param(name, marks=pytest.mark.moves_cache_prefix(reason=reason))
        if case.calls is not None and (reason := case.calls.moves_cache_prefix)
        else pytest.param(name)
        for name, case in sorted(CASES.items())
        if name in PUBLIC and contract.applies(name, case)
    ]


def test_every_exemption_names_a_case_the_contract_applies_to() -> None:
    """An exemption for a case the contract no longer applies to would hide the next one added under its name."""
    problems = [
        f'`{contract.name}` exempts {name!r}, which it does not apply to: remove the exemption.'
        for contract in CONTRACTS
        for name in contract.exempt
        if name not in PUBLIC or not contract.applies(name, CASES[name])
    ]
    problems += [
        f'`{contract.name}` exempts {name!r} with no reason.'
        for contract in CONTRACTS
        for name, exempt in contract.exempt.items()
        if not exempt.reason.strip()
    ]
    assert not problems, '\n'.join(problems)


def _assert_holds(contract: Contract, name: str, breaks: Sequence[str]) -> None:
    """`breaks` says how the case breaks `contract`: it must say nothing, unless `contract` exempts the case."""
    if exempt := contract.exempt.get(name):
        assert breaks, f'{name} now passes: remove its `{contract.name}` exemption ({exempt.reason})'
    else:
        assert not breaks, '; '.join(breaks)


def _one_call_per_response(made: tuple[ToolCall, ...], prefix: str) -> FunctionModel:
    """A model that streams the calls in order, one per response, then answers; call `i` gets the ID `call_{i}`.

    The run streams its requests, so a request that isn't streamed is a capability's own run on the run's model, such
    as `BrowserUse`'s browser agent, and gets the answer.
    """

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        del info
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step == len(made):
            yield 'done'
            return
        name, args = made[step]
        yield {0: DeltaToolCall(name=f'{prefix}{name}', json_args=json.dumps(args), tool_call_id=f'call_{step}')}

    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')]), stream_function=respond)


_GENERATED_ID = re.compile(r'\b[0-9a-f]{8}\b')
"""An ID a tool generates at random, such as a `Planning` step's, which no two runs share."""


def _comparable(content: object, directory: Path) -> object:
    """`content` with the run's directory written as `<dir>` and each generated ID as `<id>`."""
    if not isinstance(content, str):
        return content
    return _GENERATED_ID.sub('<id>', content.replace(str(directory), '<dir>'))


def _capabilities(
    built: AbstractCapability[DepsT] | Sequence[AbstractCapability[DepsT]],
) -> list[AbstractCapability[DepsT]]:
    return [built] if isinstance(built, AbstractCapability) else list(built)


async def _run_calls(
    case: CapabilityCase, directory: Path, *, prefixed: bool = False, usage_limits: UsageLimits | None = None
) -> AgentRunResult[str]:
    """A run that makes the case's calls one per response, against what its calls build in `directory`."""
    if case.requires is not None:
        pytest.importorskip(case.requires)
    assert case.calls is not None
    directory.mkdir()
    # Built off the event loop, as an app builds its agent: building can import a client library.
    capabilities = _capabilities(await anyio.to_thread.run_sync(case.calls.build, directory))
    if prefixed:
        capabilities = [capability.prefix_tools('p') for capability in capabilities]
    model = _one_call_per_response(case.calls.made, 'p_' if prefixed else '')
    # Streamed, so a tool that emits a capability event has a handler to deliver it to.
    return await Agent(model, capabilities=capabilities).run(
        'go', event_stream_handler=drain_events, usage_limits=usage_limits
    )


async def _tool_results(case: CapabilityCase, directory: Path, *, prefixed: bool) -> list[tuple[str, str, object]]:
    """How each call ended, and what it returned or the retry it asked for, as `_comparable` writes it."""
    result = await _run_calls(case, directory, prefixed=prefixed)
    return [
        (
            part.tool_call_id,
            part.outcome,
            _comparable(part.content, directory),
        )
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


@pytest.mark.parametrize('name', _cases(RENAMED))
async def test_tools_behave_the_same_under_a_prefix(name: str, tmp_path: Path) -> None:
    """Each call returns what it returns unprefixed: a tool, hook or event a capability finds by name breaks this."""
    plain = await _tool_results(CASES[name], tmp_path / 'plain', prefixed=False)
    assert plain, 'the calls returned nothing to compare'
    prefixed = await _tool_results(CASES[name], tmp_path / 'prefixed', prefixed=True)
    _assert_holds(RENAMED, name, [] if prefixed == plain else [f'under a prefix the calls end {prefixed}, not {plain}'])


async def _usage_breaks(case: CapabilityCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """How `case`'s calls break `forwards_usage`, run unlimited, then with a `request_limit` one below the requests
    that sent, counting every request a model receives, a nested run's included."""
    recorder = RequestRecorder()
    patch_models(monkeypatch, recorder)
    used = (await _run_calls(case, tmp_path / 'unlimited')).usage.requests
    sent = len(recorder.requests)
    breaks = [] if used == sent else [f'{sent} requests reached a model, but `usage.requests` is {used}']
    recorder.requests.clear()
    limit = sent - 1
    # A capability may keep within the limit instead, as `SmartSearch` leaves chunks unjudged.
    with suppress(UsageLimitExceeded):
        used = (
            await _run_calls(case, tmp_path / 'limited', usage_limits=UsageLimits(request_limit=limit))
        ).usage.requests
        if used != len(recorder.requests):
            breaks.append(
                f'with `request_limit={limit}`, {len(recorder.requests)} requests reached a model, '
                f'but `usage.requests` is {used}'
            )
    if len(recorder.requests) > limit:
        breaks.append(f'with `request_limit={limit}`, {len(recorder.requests)} requests reached a model')
    return breaks


@pytest.mark.parametrize('name', _cases(FORWARDS_USAGE))
async def test_every_request_counts_toward_the_run_usage(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_holds(FORWARDS_USAGE, name, await _usage_breaks(CASES[name], tmp_path, monkeypatch))


async def test_a_nested_run_that_drops_its_usage_breaks_forwards_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool whose own run leaves out `usage=ctx.usage`, as `BrowserUse`'s did before #11860, breaks every clause."""

    async def ask() -> str:
        return (await Agent(TestModel()).run('nested')).output

    case = CapabilityCase(
        Anonymous('built only here'), calls=Calls((('ask', {}),), lambda _: Toolset(FunctionToolset[None]([ask])))
    )
    assert await _usage_breaks(case, tmp_path, monkeypatch) == [
        '3 requests reached a model, but `usage.requests` is 2',
        'with `request_limit=2`, 3 requests reached a model, but `usage.requests` is 2',
        'with `request_limit=2`, 3 requests reached a model',
    ]


def _prepare(name: str, case: CapabilityCase, monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the case of `name` unless its dependency is installed, and set the credentials it checks."""
    skip_unless_installed(name, case.requires)
    for key, value in case.env.items():
        monkeypatch.setenv(key, value)


def _build(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[AbstractCapability[None]]:
    """Everything the case of `name` builds: the capability and what it runs with."""
    case = CASES[name]
    _prepare(name, case, monkeypatch)
    return build_capabilities(name, PUBLIC[name].load, case.fixture, tmp_path)


def _built(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AbstractCapability[None]:
    """The capability `name`, as its case builds it."""
    [capability] = [
        capability for capability in _build(name, tmp_path, monkeypatch) if type(capability).__name__ == name
    ]
    return capability


def _toolset_ids(capability: AbstractCapability[DepsT]) -> list[str | None] | None:
    """The id of every toolset `capability` builds itself, or `None` when it builds none or is made of others."""
    if leaf_capabilities(capability) != [capability]:
        return None
    toolset = capability.get_toolset()
    if toolset is None:
        return None
    if not isinstance(toolset, AbstractToolset):
        # The agent wraps a toolset function in one `DynamicToolset`, with no id of its own.
        return [None]
    leaves: list[AbstractToolset[DepsT]] = []
    # Pyright can't narrow the toolset function out of `AgentToolset`, as core's `CombinedCapability` notes.
    toolset.apply(leaves.append)  # pyright: ignore[reportUnknownMemberType]
    return [leaf.id for leaf in leaves]


def test_a_toolset_function_has_no_toolset_id() -> None:
    """No case's capability returns a toolset function, so this is the one place `_toolset_ids` reads one."""
    assert _toolset_ids(Toolset[None](lambda ctx: FunctionToolset[None]([]))) == [None]


def _with_id(capability: AbstractCapability[DepsT], id: str | None) -> AbstractCapability[DepsT] | None:
    """`capability` built again with `id`, or `None` when its `__init__` takes no `id` to build it with."""
    parameters = inspect.signature(type(capability).__init__).parameters.values()
    if not any(p.name == 'id' or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return None
    return replace(capability, id=id)


def _unset_id_breaks(capability: AbstractCapability[DepsT]) -> list[str]:
    breaks: list[str] = []
    unset = _with_id(capability, None)
    if unset is None:
        breaks.append('its `__init__` takes no `id`, so `id=None` cannot be set')
    for label, built in (('its defaults', capability), ('`id=None`', unset)):
        if built is None:
            continue
        ids = _toolset_ids(built)
        if ids is not None and any(toolset_id != built.id for toolset_id in ids):
            breaks.append(f'with {label}, the capability id is {built.id!r} and its toolset ids are {ids}')
    return breaks


def _assert_not_exempt(contract: Contract, name: str, why: str) -> None:
    """`contract` has nothing to check on the case of `name`, so an exemption for it would never be found stale."""
    assert name not in contract.exempt, f'{name} {why}: remove its `{contract.name}` exemption'


@pytest.mark.parametrize('name', _cases(UNSET_ID))
def test_an_unset_id_is_every_toolset_id(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capability = _built(name, tmp_path, monkeypatch)
    if _toolset_ids(capability) is None:
        _assert_not_exempt(UNSET_ID, name, 'builds no toolset of its own')
        return
    _assert_holds(UNSET_ID, name, _unset_id_breaks(capability))


@pytest.mark.parametrize('name', _cases(CUSTOM_ID))
def test_a_custom_id_reaches_every_toolset(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capability = _built(name, tmp_path, monkeypatch)
    if _toolset_ids(capability) is None:
        _assert_not_exempt(CUSTOM_ID, name, 'builds no toolset of its own')
        return
    custom = _with_id(capability, 'custom')
    if custom is None:
        _assert_holds(CUSTOM_ID, name, ["its `__init__` takes no `id`, so `id='custom'` cannot be set"])
        return
    ids = _toolset_ids(custom)
    reached = ids is None or all(toolset_id == 'custom' for toolset_id in ids)
    _assert_holds(CUSTOM_ID, name, [] if reached else [f"with `id='custom'`, its toolset ids are {ids}"])


def _through_client(cls: Callable[..., AbstractCapability[DepsT]], *, auth: str) -> AbstractCapability[DepsT]:
    return cls(client=hosted_server(), auth=auth)


@pytest.mark.parametrize('name', _cases(HOSTED_MCP_CONNECTION))
def test_a_hosted_connection_takes_a_client_or_auth(name: str) -> None:
    cls = PUBLIC[name].load()
    with pytest.raises(UserError, match='`client` owns the connection'):
        _through_client(cls, auth='key')
    capability = _through_client(cls, auth='')
    assert _toolset_ids(capability) == [capability.id]


async def _first_request(capabilities: Sequence[AgentCapability[None]], directory: Path) -> object:
    """What reaches the model in a run's first request, or why the agent refused the capabilities."""
    seen: list[object] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools = [(tool.name, tool.description, tool.parameters_json_schema) for tool in info.function_tools]
        seen.append((info.instructions, tools, info.model_settings, info.model_request_parameters.native_tools))
        return ModelResponse(parts=[TextPart('done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        respond(messages, info)
        yield 'done'

    # A capability that uses the run's workspace needs one, unless it or a capability function supplies its own.
    supplies_workspace = any(
        callable(capability) or type(capability).get_workspace is not AbstractCapability.get_workspace
        for capability in capabilities
    )
    workspace = [] if supplies_workspace else [LocalWorkspace[None](directory)]
    try:
        # Built off the event loop, as an app builds its agent: combining can construct capabilities, which may
        # read files. Streamed when a capability streams the run's requests.
        agent = await anyio.to_thread.run_sync(
            lambda: Agent(
                FunctionModel(respond, stream_function=stream),
                deps_type=type(None),
                capabilities=[*capabilities, *workspace],
            )
        )
        await agent.run('go')
    except (UserError, ValueError) as exc:
        return f'{type(exc).__name__}: {exc}'
    return seen[0]


@pytest.mark.parametrize('name', _cases(COMBINE_FIXED_POINT))
async def test_stating_a_configuration_twice_states_it_once(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = CASES[name]
    assert isinstance(case.combine, Combines)
    _prepare(name, case, monkeypatch)
    # Built off the event loop, as an app builds them: a capability may read files when constructed.
    first, second = await anyio.to_thread.run_sync(case.combine.make)
    merged = await anyio.to_thread.run_sync(type(first).combine, [first, second])
    breaks: list[str] = []
    for label, capability in (('one of the pair', first), ('the pair combined', merged)):
        once = await _first_request([capability], tmp_path)
        # Stated once it must reach the model: two runs that fail alike would compare equal and check nothing.
        assert not isinstance(once, str), f'{name}: {label} alone does not run: {once}'
        if await _first_request([capability, capability], tmp_path) != once:
            breaks.append(label)
    _assert_holds(
        COMBINE_FIXED_POINT, name, [f'stated twice, {label} reaches the model differently' for label in breaks]
    )


@pytest.mark.parametrize('name', _cases(WORKSPACE_FROM_FUNCTION))
async def test_a_workspace_from_a_function_works_like_a_static_one(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for directory in ('static', 'function'):
        (tmp_path / directory).mkdir()
    static = _build(name, tmp_path / 'static', monkeypatch)
    if not any(isinstance(capability, LocalWorkspace) for capability in static):
        _assert_not_exempt(WORKSPACE_FROM_FUNCTION, name, 'builds no `LocalWorkspace`')
        return
    supplied: list[AgentCapability[None]] = [
        (lambda ctx, workspace=capability: workspace) if isinstance(capability, LocalWorkspace) else capability
        for capability in _build(name, tmp_path / 'function', monkeypatch)
    ]
    static_request = await _first_request(static, tmp_path / 'static')
    assert not isinstance(static_request, str), f'{name} does not run with a static workspace: {static_request}'
    expected = _comparable(repr(static_request), tmp_path / 'static')
    actual = _comparable(repr(await _first_request(supplied, tmp_path / 'function')), tmp_path / 'function')
    _assert_holds(
        WORKSPACE_FROM_FUNCTION,
        name,
        [] if actual == expected else [f'from a capability function, the first request is {actual}, not {expected}'],
    )


# C6, instruction sources: text a capability reads from a source below `'operator'` never reaches the instructions, a
# system or user part, or a tool definition, only tool output or a framed data part naming its source.

CANARY = 'CANARY-c6'


@dataclass
class _Seen:
    """Where the canary reached the model, each place with the authority it reached it at."""

    places: list[tuple[str, Authority]] = field(default_factory=list[tuple[str, Authority]])

    def record(self, messages: list[ModelMessage], info: AgentInfo) -> None:
        """Note the canary in what a model was sent in the operator's voice: instructions, system parts, tool definitions.

        Every model in the run records, a sub-agent's too. The user channel is judged from the typed history instead,
        where a framed data part still says which source it came from.
        """
        for message in messages:
            if not isinstance(message, ModelRequest):
                continue
            if CANARY in (message.instructions or ''):
                self.places.append(('instructions', 'operator'))
            self.places.extend(
                ('system part', 'operator')
                for part in message.parts
                if isinstance(part, SystemPromptPart) and CANARY in part.content
            )
        self.places.extend(
            (f'`{tool.name}` definition', 'operator')
            for tool in info.function_tools
            if CANARY in f'{tool.name} {tool.description} {json.dumps(tool.parameters_json_schema)}'
        )

    def record_history(self, messages: Sequence[ModelMessage]) -> None:
        """Note the canary in the run's typed request parts, at the authority each part carries it with."""
        for message in messages:
            if isinstance(message, ModelRequest):
                self.places.extend(_place(part) for part in message.parts if CANARY in repr(part))


def _place(part: ModelRequestPart) -> tuple[str, Authority]:
    if isinstance(part, ToolReturnPart):
        return 'tool output', 'none'
    if isinstance(part, CapabilityPart):
        return f'`{part.kind}` part', part.source.authority
    if isinstance(part, UserPromptPart):
        return 'user part', 'user'
    # A system part, or a place this test doesn't know: counted as the operator's voice, so it fails at `'none'`.
    return type(part).__name__, 'operator'


def _scripted(seen: _Seen, calls: Sequence[ToolCallPart] = ()) -> FunctionModel:
    """A model that makes `calls` one per step and then answers, recording what it was sent."""
    step = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        seen.record(messages, info)
        step += 1
        return ModelResponse(parts=[calls[step - 1]] if step <= len(calls) else [TextPart('done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        [part] = respond(messages, info).parts
        if isinstance(part, ToolCallPart):
            yield {
                0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id)
            }
        else:
            assert isinstance(part, TextPart)
            yield part.content

    return FunctionModel(respond, stream_function=stream)


@dataclass
class _InstructionSource:
    """One source of text a capability can put in front of the model, holding the canary."""

    build: Callable[[Path, Authority, _Seen], Sequence[AgentCapability[None]]]
    """Writes the source's files under the directory and returns the capabilities that read them."""
    calls: Callable[[Authority], Sequence[ToolCallPart]] = lambda _: ()
    """The tool calls that bring the source's text to the model at the given authority."""
    working_dir: str = '.'
    operator_reaches: Authority = 'operator'
    """The highest authority the text reaches the model at when its source is the operator's."""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def _skills(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    skill = directory / 'skills' / 'guide'
    _write(skill / 'SKILL.md', f'---\ndescription: Guide help {CANARY}.\n---\n\nGuide directions {CANARY}.\n')
    _write(skill / 'references' / 'guide.md', f'Use tabs {CANARY}.')
    return [Skills('skills', read_files=True)]


def _skills_calls(authority: Authority) -> list[ToolCallPart]:
    # Below `'operator'` the catalog is left out and these tools offer it; at `'operator'` it's in the instructions.
    if authority == 'operator':
        return []
    return [
        ToolCallPart('list_skills', {}),
        ToolCallPart('load_skill', {'name': 'guide'}),
        ToolCallPart('read_skill_file', {'skill': 'guide', 'path': 'references/guide.md'}),
    ]


def _repo_context_working_dir(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    _write(directory / 'packages' / 'foo' / 'AGENTS.md', f'Package rules {CANARY}.')
    return [RepoContext(home_dir=str(directory), nested_traversal=False), FileSystem()]


def _repo_context_ancestor(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    _write(directory / 'AGENTS.md', f'Repo rules {CANARY}.')
    (directory / 'packages' / 'foo').mkdir(parents=True)
    return [RepoContext(home_dir=str(directory), nested_traversal=False)]


def _repo_context_nested(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    _write(directory / 'sub' / 'AGENTS.md', f'Nested rules {CANARY}.')
    _write(directory / 'sub' / 'x.py', 'x = 1')
    return [RepoContext(home_dir=str(directory), autoload_instructions=False), FileSystem()]


def _memory(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    _write(directory / '.memory' / 'main' / 'MEMORY.md', f'- a fact {CANARY}')
    return [Memory(store=FileMemoryStore('.memory'))]


def _mcp_instructions(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    server = MCPServer('docs', instructions=f'Server rules {CANARY}.')

    @server.tool()
    def search() -> str:
        return 'found'  # pragma: no cover

    return [Toolset(MCPToolset(server, include_instructions=True))]


def _mcp_tool_description(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    server = MCPServer('docs')

    @server.tool(description=f'Search the docs {CANARY}.')
    def search() -> str:
        return 'found'  # pragma: no cover

    return [Toolset(MCPToolset(server))]


def _sub_agents(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    child = _scripted(seen)
    if authority == 'operator':
        # The operator's own sub-agent: its instructions are the operator's, the control for the model-written ones.
        helper = Agent(child, deps_type=object, name='helper', instructions=f'Helper rules {CANARY}.')
        return [SubAgents(agents=[SubAgent(helper)], agent_folders=None)]
    return [SubAgents(models={'child': child}, dynamic_agents=DynamicAgents(models=['child']), agent_folders=None)]


def _sub_agents_calls(authority: Authority) -> list[ToolCallPart]:
    delegate = ToolCallPart('delegate_task', {'agent_name': 'helper', 'task': 'Help.'})
    if authority == 'operator':
        return [delegate]
    create = {'name': 'helper', 'description': 'Helps.', 'instructions': f'Helper rules {CANARY}.', 'model': 'child'}
    return [ToolCallPart('create_agent', create), delegate]


INSTRUCTION_SOURCES: dict[str, _InstructionSource] = {
    'skills': _InstructionSource(_skills, _skills_calls),
    'repo_context_working_dir': _InstructionSource(
        _repo_context_working_dir, lambda _: [ToolCallPart('read_file', {'path': 'AGENTS.md'})], 'packages/foo'
    ),
    'repo_context_ancestor': _InstructionSource(_repo_context_ancestor, working_dir='packages/foo'),
    'repo_context_nested': _InstructionSource(
        _repo_context_nested, lambda _: [ToolCallPart('read_file', {'path': 'sub/x.py'})]
    ),
    # Memory is data even from the operator's own store, attributed to the agent that wrote it.
    'memory': _InstructionSource(_memory, operator_reaches='none'),
    'mcp_instructions': _InstructionSource(_mcp_instructions),
    'mcp_tool_description': _InstructionSource(_mcp_tool_description),
    'sub_agents_create_agent': _InstructionSource(_sub_agents, _sub_agents_calls),
}

_KNOWN_BREAKS = {
    'mcp_instructions': 'MCP servers have no authority yet; their text goes out as the operator (#13117)',
    'mcp_tool_description': 'MCP servers have no authority yet; their text goes out as the operator (#13117)',
    'sub_agents_create_agent': 'model-written instructions become the child agent system prompt (#12433)',
}


async def _reached(name: str, directory: Path, authority: Authority) -> list[tuple[str, Authority]]:
    source = INSTRUCTION_SOURCES[name]
    seen = _Seen()
    capabilities = source.build(directory, authority, seen)
    working_dir = directory / source.working_dir
    workspace = LocalWorkspaceBackend(working_dir) if authority == 'operator' else untrusted(working_dir)
    result = await Agent(_scripted(seen, source.calls(authority)), capabilities=capabilities).run(
        'go', workspace=workspace
    )
    seen.record_history(result.all_messages())
    return seen.places


@pytest.mark.parametrize(
    'name',
    [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=_KNOWN_BREAKS[name]))
        if name in _KNOWN_BREAKS
        else name
        for name in INSTRUCTION_SOURCES
    ],
)
async def test_text_below_operator_reaches_the_model_only_as_tool_output_or_framed_data(
    name: str, tmp_path: Path
) -> None:
    places = await _reached(name, tmp_path, 'none')
    assert places, 'the canary never reached the model, so the case checks nothing'
    assert [place for place in places if place[1] != 'none'] == []


@pytest.mark.parametrize('name', list(INSTRUCTION_SOURCES))
async def test_the_same_text_from_the_operator_reaches_the_model_as_the_operator(name: str, tmp_path: Path) -> None:
    # The control for the test above: each case is wired so its source's level is what decides where the text goes.
    places = await _reached(name, tmp_path, 'operator')
    assert any(authority == INSTRUCTION_SOURCES[name].operator_reaches for _, authority in places), places
