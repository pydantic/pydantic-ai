from dataclasses import dataclass
from typing import Literal
from unittest.mock import Mock

import pytest

from pydantic_ai import Agent, AgentSpec, RunContext
from pydantic_ai.models.test import TestModel


@dataclass
class MetadataCase:
    name: str
    base: dict[str, object] | None
    run: dict[str, object] | None
    expected: dict[str, object] | None
    identity: Literal['base', 'run', 'new', 'none']


@pytest.mark.parametrize(
    'case',
    [
        MetadataCase('no metadata', None, None, None, 'none'),
        MetadataCase('empty base', {}, None, {}, 'base'),
        MetadataCase('empty run', None, {}, None, 'none'),
        MetadataCase('truthy base and empty run', {'base': True}, {}, {'base': True}, 'base'),
        MetadataCase('truthy run only', None, {'run': True}, {'run': True}, 'run'),
        MetadataCase(
            'two truthy dicts',
            {'shared': 'base', 'base': True},
            {'shared': 'run', 'run': True},
            {'shared': 'run', 'base': True, 'run': True},
            'new',
        ),
    ],
    ids=lambda case: case.name,
)
async def test_agent_metadata_resolution(case: MetadataCase) -> None:
    agent = Agent(TestModel(), metadata=case.base)

    result = await agent.run('hello', metadata=case.run)

    assert result.metadata == case.expected
    if case.identity == 'base':
        assert result.metadata is case.base
    elif case.identity == 'run':
        assert result.metadata is case.run
    elif case.identity == 'new':
        assert result.metadata is not case.base
        assert result.metadata is not case.run
    else:
        assert result.metadata is None


async def test_agent_metadata_factories_resolve_in_order() -> None:
    calls: list[str] = []
    spec_metadata: dict[str, object] = {'shared': 'spec', 'spec': True}

    def base_factory(_ctx: RunContext[object]) -> dict[str, object]:
        calls.append('base')
        return {'shared': 'base', 'base': True}

    def run_factory(_ctx: RunContext[object]) -> dict[str, object]:
        calls.append('run')
        return {'shared': 'run', 'run': True}

    agent = Agent(TestModel(), metadata=base_factory)
    result = await agent.run('hello', spec=AgentSpec(metadata=spec_metadata), metadata=run_factory)

    assert calls == ['base', 'run'] * 3
    assert result.metadata == {'shared': 'run', 'spec': True, 'base': True, 'run': True}


async def test_run_metadata_override_skips_factories() -> None:
    base_metadata: dict[str, object] = {'base': True}
    run_metadata: dict[str, object] = {'run': True}
    base_factory = Mock(return_value=base_metadata)
    run_factory = Mock(return_value=run_metadata)
    override: dict[str, object] = {}
    agent = Agent(TestModel(), metadata=base_factory)

    with agent.override(metadata=override):
        result = await agent.run('hello', metadata=run_factory)

    assert result.metadata is override
    base_factory.assert_not_called()
    run_factory.assert_not_called()
