"""The contracts in `tests/capability_cases.py`, run over every core and Harness case they apply to.

They run here, where the Harness is installed, because its cases are most of what they apply to, over the public
capabilities the suites build. A case a contract exempts must still break it, so fixing the capability fails the
suite until the exemption is removed. The durable contracts run under their engines, in
`tests/harness/test_durable_tool_io.py` (DBOS) and `tests/harness/test_durable_temporal_replay.py` (Temporal); how
they read a run's journal is checked here.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import anyio
import pytest

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, LocalWorkspace, Toolset
from pydantic_ai.capabilities.abstract import leaf_capabilities
from pydantic_ai.exceptions import UsageLimitExceeded, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.usage import UsageLimits

from ..append_only import ToolCall, build_capabilities, drain_events, skip_unless_installed
from ..capability_cases import (
    COMBINE_FIXED_POINT,
    CONTRACTS as CORE_CONTRACTS,
    CUSTOM_ID,
    DECLARES_CALLS,
    FORWARDS_USAGE,
    PENDING_CALLS,
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
from ._contracts import CASES, PUBLIC, assert_holds as _assert_holds, contract_cases as _cases
from ._counting_clients import REQUESTS
from ._durable_cases import PROMPT, UNPROMPTED_REQUEST, journal_breaks, one_call_per_response
from .capability_cases import (
    CONTRACTS as HARNESS_CONTRACTS,
    HOSTED_MCP_CONNECTION,
    hosted_server,
)

DepsT = TypeVar('DepsT')

CONTRACTS: tuple[Contract, ...] = (*CORE_CONTRACTS, *HARNESS_CONTRACTS)


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


def _one_call_per_response(made: tuple[ToolCall, ...], prefix: str) -> FunctionModel:
    """A model that streams the calls in order, one per response, then answers every request past them, such as one a
    call's follow-up causes; call `i` gets the ID `call_{i}`.

    The run streams its requests, so a request that isn't streamed is a capability's own run on the run's model, such
    as `BrowserUse`'s browser agent, and gets the answer.
    """

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        del info
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step >= len(made):
            yield 'done'
            return
        name, args = made[step]
        yield {0: DeltaToolCall(name=f'{prefix}{name}', json_args=json.dumps(args), tool_call_id=f'call_{step}')}

    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')]), stream_function=respond)


_GENERATED_ID = re.compile(r'\b[0-9a-f]{8}\b')
"""An ID a tool generates at random, such as a `Planning` step's, which no two runs share."""


_PREFIX = 'p'
"""The prefix the `renamed` contract runs a case's tools under."""


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
        capabilities = [capability.prefix_tools(_PREFIX) for capability in capabilities]
    model = _one_call_per_response(case.calls.made, f'{_PREFIX}_' if prefixed else '')
    # Streamed, so a tool that emits a capability event has a handler to deliver it to.
    return await Agent(model, capabilities=capabilities).run(
        'go', event_stream_handler=drain_events, usage_limits=usage_limits
    )


def _unprefixed(content: object, names: dict[str, str]) -> object:
    if not isinstance(content, str):
        return content
    for prefixed, plain in names.items():
        content = content.replace(prefixed, plain)
    return content


async def _tool_results(case: CapabilityCase, directory: Path, *, prefixed: bool) -> list[tuple[str, str, object]]:
    """How each call ended, and what it returned or the retry it asked for, as `_comparable` writes it."""
    result = await _run_calls(case, directory, prefixed=prefixed)
    assert case.calls is not None
    # A reply that quotes its tool by the name the model sees, as `BackgroundTools`' acknowledgement does, is the same
    # reply under a prefix.
    names = {f"'{_PREFIX}_{name}'": f"'{name}'" for name, _ in case.calls.made} if prefixed else {}
    return [
        (
            part.tool_call_id,
            part.outcome,
            _comparable(_unprefixed(part.content, names), directory),
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


def _unset_id_breaks(capability: AbstractCapability[DepsT]) -> list[str]:
    breaks: list[str] = []
    for label, built in (('its defaults', capability), ('`id=None`', replace(capability, id=None))):
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
    ids = _toolset_ids(replace(capability, id='custom'))
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


_RUNS_WITH_TOOLS = (
    'get_wrapper_toolset',
    'before_tool_validate',
    'after_tool_validate',
    'wrap_tool_validate',
    'on_tool_validate_error',
    'before_tool_execute',
    'after_tool_execute',
    'wrap_tool_execute',
    'on_tool_execute_error',
)
"""The methods that run when a tool is called, besides the toolset `get_toolset` returns."""


def _tool_contributions(capability: AbstractCapability[DepsT]) -> list[str]:
    """What `capability` contributes that runs when a tool is called, or nothing when it is made of others."""
    if leaf_capabilities(capability) != [capability]:
        return []
    contributes = ['get_toolset'] if capability.get_toolset() is not None else []
    return contributes + [
        method
        for method in _RUNS_WITH_TOOLS
        if getattr(type(capability), method) is not getattr(AbstractCapability, method)
    ]


@pytest.mark.parametrize('name', _cases(DECLARES_CALLS))
def test_a_capability_that_runs_with_tools_declares_calls(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contributes = _tool_contributions(_built(name, tmp_path, monkeypatch))
    if not contributes:
        _assert_not_exempt(DECLARES_CALLS, name, 'contributes no toolset or tool hook')
        return
    declared = CASES[name].calls is not None
    _assert_holds(DECLARES_CALLS, name, [] if declared else [f'it contributes {contributes} but declares no `calls`'])


async def test_a_nested_run_on_a_durable_case_model_is_answered_and_counted() -> None:
    """The durable suites count it, so a nested run's request a recovery repeats breaks them like any other."""
    REQUESTS.clear()

    model = one_call_per_response((('ask', {}),))

    async def ask() -> str:
        return (await Agent(model).run('nested')).output

    try:
        result = await Agent(model, tools=[ask]).run(PROMPT)
        assert (result.output, REQUESTS) == ('done', {UNPROMPTED_REQUEST: 1})
    finally:
        REQUESTS.clear()


def test_a_drifted_journal_breaks_the_durable_contracts() -> None:
    """A unit renamed, added or dropped breaks them, and so does a model request too many or too few."""
    expected = ('capability__memory.write_memory',)
    recorded = ['a__model.request', 'a__capability__memory.write_memory', 'a__model.request']

    def read(units: list[str]) -> list[str]:
        return journal_breaks(
            units, prefix='a__', model_request='model.request', calls=1, follow_ups=0, expected=expected, engine='dbos'
        )

    assert read(recorded) == []
    renamed = ['a__model.request', 'a__capability__memory.write', 'a__model.request']
    assert len(read(renamed)) == 1
    assert len(read(['a__model.request', 'a__model.request'])) == 1, 'a dropped unit breaks them'
    assert read(['a__capability__memory.write_memory', 'a__model.request']) == [
        'the run recorded 1 `model.request` units, not one per call, one per follow-up and one to answer'
    ]
    assert read([*recorded, 'a__model.request', 'a__capability__memory.read_memory']) == [
        'the run recorded 3 `model.request` units, not one per call, one per follow-up and one to answer',
        "the calls record `dbos=('capability__memory.write_memory', 'capability__memory.read_memory')`, not "
        "('capability__memory.write_memory',). A recorded name is what a workflow from an earlier release replays by: "
        'change the case only with a compatibility note (agent_docs/capability-conformance.md, "Durable execution")',
    ]


def test_pending_calls_backfill() -> None:
    """The cases whose `calls` are still to be written: a PR that writes one lowers the count, and none raises it."""
    pending = sorted(name for name, exempt in DECLARES_CALLS.exempt.items() if exempt.reason.startswith(PENDING_CALLS))
    assert len(pending) == 14, pending
