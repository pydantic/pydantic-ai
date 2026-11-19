"""The contracts in `tests/capability_cases.py`, run over every core and Harness case they apply to.

They run here, where the Harness is installed, because its cases are most of what they apply to. A case a
contract exempts must still break it, so fixing the capability fails the suite until the exemption is removed.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from .. import capability_cases as core
from ..append_only import ToolCall, drain_events
from ..capability_cases import CONTRACTS, RENAMED, CapabilityCase, Contract
from .capability_cases import CASES as HARNESS_CASES

CASES: dict[str, CapabilityCase] = {**core.CASES, **HARNESS_CASES}


def _cases(contract: Contract) -> list[Any]:
    """The cases `contract` applies to, each whose calls rewrite history marked `moves_cache_prefix`."""
    return [
        pytest.param(name, marks=pytest.mark.moves_cache_prefix(reason=reason))
        if case.calls is not None and (reason := case.calls.moves_cache_prefix)
        else name
        for name, case in sorted(CASES.items())
        if contract.applies(case)
    ]


def test_every_exemption_names_a_case_the_contract_applies_to() -> None:
    """An exemption for a case the contract no longer applies to would hide the next one added under its name."""
    problems = [
        f'`{contract.name}` exempts {name!r}, which it does not apply to: remove the exemption.'
        for contract in CONTRACTS
        for name in contract.exempt
        if name not in CASES or not contract.applies(CASES[name])
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
