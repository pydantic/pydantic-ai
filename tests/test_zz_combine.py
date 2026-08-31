from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, PrefixTools
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel


@dataclass
class Plain(AbstractCapability[Any]):
    """No combine: duplicates are a mistake."""

    marker: str = 'x'


@dataclass
class LastWins(AbstractCapability[Any]):
    marker: str = 'x'

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[Any]]) -> AbstractCapability[Any]:
        return capabilities[-1]


async def _registered(agent: Agent[Any, Any]) -> dict[str, AbstractCapability[Any]]:
    seen: dict[str, AbstractCapability[Any]] = {}

    @dataclass
    class Probe(AbstractCapability[Any]):
        async def before_run(self, ctx: RunContext[Any]) -> None:
            seen.update(ctx.capabilities)

    await Agent(TestModel(), capabilities=[agent._root_capability, Probe()]).run('hi')  # type: ignore[attr-defined]
    return seen


def test_duplicate_explicit_ids_without_combine_raise_at_construction() -> None:
    with pytest.raises(UserError, match="Capability id 'dup' is used by multiple capabilities"):
        Agent(TestModel(), capabilities=[Plain(id='dup'), Plain(id='dup')])


async def test_duplicate_ids_with_combine_collapse() -> None:
    agent = Agent(TestModel(), capabilities=[LastWins(id='dup', marker='first'), LastWins(id='dup', marker='last')])
    seen = await _registered(agent)
    assert isinstance(seen['dup'], LastWins)
    assert seen['dup'].marker == 'last'


async def test_run_level_supersedes_agent_level() -> None:
    seen: dict[str, AbstractCapability[Any]] = {}

    @dataclass
    class Probe(AbstractCapability[Any]):
        async def before_run(self, ctx: RunContext[Any]) -> None:
            seen.update(ctx.capabilities)

    agent = Agent(TestModel(), capabilities=[LastWins(id='dup', marker='agent'), Probe()])
    await agent.run('hi', capabilities=[LastWins(id='dup', marker='run')])
    assert seen['dup'].marker == 'run'  # type: ignore[attr-defined]


async def test_nested_wrapper_supersession() -> None:
    """The #7248 bug: a superseded capability nested inside a wrapper must not survive."""
    seen: dict[str, AbstractCapability[Any]] = {}

    @dataclass
    class Probe(AbstractCapability[Any]):
        async def before_run(self, ctx: RunContext[Any]) -> None:
            seen.update(ctx.capabilities)

    layer = PrefixTools(
        wrapped=CombinedCapability[Any]([LastWins(id='dup', marker='nested'), Plain(id='bundle')]),
        prefix='p',
    )
    agent = Agent(TestModel(), capabilities=[layer, Probe()])
    await agent.run('hi', capabilities=[LastWins(id='dup', marker='run')])
    assert seen['dup'].marker == 'run'  # type: ignore[attr-defined]
    # `bundle` survives exactly once, and nothing is duplicated.
    ids = sorted(seen)
    assert ids.count('bundle') == 1
    assert 'dup_2' not in seen and 'bundle_2' not in seen


async def test_mixed_types_under_one_id_raise() -> None:
    agent = Agent(TestModel(), capabilities=[LastWins(id='dup')])
    with pytest.raises(UserError, match='used by capabilities of different types'):
        await agent.run('hi', capabilities=[Plain(id='dup')])


async def test_anonymous_capabilities_are_untouched() -> None:
    agent = Agent(TestModel(), capabilities=[Plain(), Plain()])
    seen = await _registered(agent)
    assert 'plain' in seen and 'plain_2' in seen
