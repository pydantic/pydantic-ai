"""The contracts in `tests/capability_cases.py`, run over every core and Harness case they apply to.

They run here, where the Harness is installed, because its cases are most of what they apply to, over the public
capabilities the suites build. A case a contract exempts must still break it, so fixing the capability fails the
suite until the exemption is removed.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import leaf_capabilities
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.toolsets import AbstractToolset

from .. import capability_cases as core
from ..append_only import ToolCall, build_capabilities, discover_capabilities, drain_events, skip_unless_installed
from ..capability_cases import (
    CONTRACTS,
    CORE_CAPABILITY_PACKAGES,
    CUSTOM_ID,
    RENAMED,
    UNSET_ID,
    CapabilityCase,
    Contract,
)
from .capability_cases import CASES as HARNESS_CASES, HARNESS_CAPABILITY_PACKAGES

CASES: dict[str, CapabilityCase] = {**core.CASES, **HARNESS_CASES}
PUBLIC = {**discover_capabilities(CORE_CAPABILITY_PACKAGES), **discover_capabilities(HARNESS_CAPABILITY_PACKAGES)}
"""The public capabilities, which the suites build; a private class only has to say what two of it mean."""


def _cases(contract: Contract) -> list[Any]:
    """The cases `contract` applies to, each whose calls rewrite history marked `moves_cache_prefix`."""
    return [
        pytest.param(name, marks=pytest.mark.moves_cache_prefix(reason=reason))
        if case.calls is not None and (reason := case.calls.moves_cache_prefix)
        else name
        for name, case in sorted(CASES.items())
        if name in PUBLIC and contract.applies(case)
    ]


def test_every_exemption_names_a_case_the_contract_applies_to() -> None:
    """An exemption for a case the contract no longer applies to would hide the next one added under its name."""
    problems = [
        f'`{contract.name}` exempts {name!r}, which it does not apply to: remove the exemption.'
        for contract in CONTRACTS
        for name in contract.exempt
        if name not in PUBLIC or not contract.applies(CASES[name])
    ]
    problems += [
        f'`{contract.name}` exempts {name!r} with no reason.'
        for contract in CONTRACTS
        for name, exempt in contract.exempt.items()
        if not exempt.reason.strip()
    ]
    assert not problems, '\n'.join(problems)


def _one_call_per_response(made: tuple[ToolCall, ...], prefix: str) -> FunctionModel:
    """A model that streams the calls in order, one per response, then answers; call `i` gets the ID `call_{i}`."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        del info
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step == len(made):
            yield 'done'
            return
        name, args = made[step]
        yield {0: DeltaToolCall(name=f'{prefix}{name}', json_args=json.dumps(args), tool_call_id=f'call_{step}')}

    return FunctionModel(stream_function=respond)


_GENERATED_ID = re.compile(r'\b[0-9a-f]{8}\b')
"""An ID a tool generates at random, such as a `Planning` step's, which no two runs share."""


def _comparable(content: object, directory: Path) -> object:
    """`content` with the run's directory written as `<dir>` and each generated ID as `<id>`."""
    if not isinstance(content, str):
        return content
    return _GENERATED_ID.sub('<id>', content.replace(str(directory), '<dir>'))


def _capabilities(built: AbstractCapability[Any] | Sequence[AbstractCapability[Any]]) -> list[AbstractCapability[Any]]:
    return [built] if isinstance(built, AbstractCapability) else list(built)


async def _tool_results(case: CapabilityCase, directory: Path, *, prefixed: bool) -> list[tuple[str, str, object]]:
    """How each call ended, and what it returned or the retry it asked for, as `_comparable` writes it."""
    assert case.calls is not None
    directory.mkdir()
    capabilities = _capabilities(case.calls.build(directory))
    if prefixed:
        capabilities = [capability.prefix_tools('p') for capability in capabilities]
    model = _one_call_per_response(case.calls.made, 'p_' if prefixed else '')
    # Streamed, so a tool that emits a capability event has a handler to deliver it to.
    result = await Agent(model, capabilities=capabilities).run('go', event_stream_handler=drain_events)
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
    case = CASES[name]
    if case.requires is not None:
        pytest.importorskip(case.requires)
    plain = await _tool_results(case, tmp_path / 'plain', prefixed=False)
    assert plain, 'the calls returned nothing to compare'
    prefixed = await _tool_results(case, tmp_path / 'prefixed', prefixed=True)
    if exempt := RENAMED.exempt.get(name):
        assert prefixed != plain, f'{name} now behaves the same under a prefix: remove its exemption ({exempt.reason})'
    else:
        assert prefixed == plain


def _built(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AbstractCapability[Any]:
    """The capability `name`, as its case builds it."""
    case = CASES[name]
    skip_unless_installed(name, case.requires)
    for key, value in case.env.items():
        monkeypatch.setenv(key, value)
    built = build_capabilities(name, PUBLIC[name].load, case.fixture, tmp_path)
    [capability] = [capability for capability in built if type(capability).__name__ == name]
    return capability


def _toolset_ids(capability: AbstractCapability[Any]) -> list[str | None] | None:
    """The id of every toolset `capability` builds itself, or `None` when it builds none or is made of others."""
    if leaf_capabilities(capability) != [capability]:
        return None
    toolset = capability.get_toolset()
    if toolset is None:
        return None
    if not isinstance(toolset, AbstractToolset):
        # The agent wraps a toolset function in one `DynamicToolset`, with no id of its own.
        return [None]
    leaves: list[AbstractToolset[Any]] = []
    # Pyright can't narrow the toolset function out of `AgentToolset`, as core's `CombinedCapability` notes.
    toolset.apply(leaves.append)  # pyright: ignore[reportUnknownMemberType]
    return [leaf.id for leaf in leaves]


def _unset_id_breaks(capability: AbstractCapability[Any]) -> list[str]:
    breaks: list[str] = []
    for label, built in (('its defaults', capability), ('`id=None`', replace(capability, id=None))):
        ids = _toolset_ids(built)
        if ids is not None and any(toolset_id != built.id for toolset_id in ids):
            breaks.append(f'with {label}, the capability id is {built.id!r} and its toolset ids are {ids}')
    return breaks


@pytest.mark.parametrize('name', _cases(UNSET_ID))
def test_an_unset_id_is_every_toolset_id(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capability = _built(name, tmp_path, monkeypatch)
    if _toolset_ids(capability) is None:
        return
    breaks = _unset_id_breaks(capability)
    if exempt := UNSET_ID.exempt.get(name):
        assert breaks, f'{name} now passes: remove its `unset_id` exemption ({exempt.reason})'
    else:
        assert not breaks, '; '.join(breaks)


@pytest.mark.parametrize('name', _cases(CUSTOM_ID))
def test_a_custom_id_reaches_every_toolset(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capability = _built(name, tmp_path, monkeypatch)
    if _toolset_ids(capability) is None:
        return
    ids = _toolset_ids(replace(capability, id='custom'))
    reached = ids is None or all(toolset_id == 'custom' for toolset_id in ids)
    if exempt := CUSTOM_ID.exempt.get(name):
        assert not reached, f'{name} now passes: remove its `custom_id` exemption ({exempt.reason})'
    else:
        assert reached, f"with `id='custom'`, its toolset ids are {ids}"
