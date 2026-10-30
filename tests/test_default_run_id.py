"""Compatibility of the public no-argument capability identity hook."""

from __future__ import annotations

import uuid
from unittest.mock import Mock

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.models.test import TestModel


@pytest.mark.parametrize('legacy', [False, True])
async def test_plain_and_legacy_runs_do_not_assemble_identity_inputs(
    legacy: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LegacyCapability(AbstractCapability[object]):
        def get_default_run_id(self) -> str:
            return 'legacy-run-id'

    capabilities: list[AbstractCapability[object]] = [LegacyCapability()] if legacy else []
    agent = Agent(TestModel(), capabilities=capabilities)
    # Observe the assembly boundary while exercising the public bootstrap path.
    factory = Mock(wraps=agent._get_run_id_inputs)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(agent, '_get_run_id_inputs', factory)
    await agent.run('hello', metadata={'run': True})
    factory.assert_not_called()


@pytest.mark.parametrize('provided', [None, 'legacy-run-id'])
@pytest.mark.parametrize('container', ['leaf', 'wrapper', 'combined', 'wrapper_override'])
async def test_legacy_default_run_id_hook(provided: str | None, container: str) -> None:
    calls: list[str] = []

    class LegacyCapability(AbstractCapability[object]):
        def get_default_run_id(self) -> str | None:
            calls.append('leaf')
            return 'child-run-id' if container == 'wrapper_override' else provided

    class LegacyWrapper(WrapperCapability[object]):
        def get_default_run_id(self) -> str | None:
            calls.append('wrapper')
            return provided

    capability: AbstractCapability[object] = LegacyCapability()
    if container == 'wrapper':
        capability = WrapperCapability(capability)
    elif container == 'combined':
        capability = CombinedCapability([capability, AbstractCapability()])
    elif container == 'wrapper_override':
        capability = LegacyWrapper(capability)

    agent = Agent(TestModel(), capabilities=[capability])
    result = await agent.run('hello')
    if provided is not None:
        assert result.run_id == provided
    else:
        assert uuid.UUID(result.run_id).version == 7
    assert calls == ['wrapper' if container == 'wrapper_override' else 'leaf']
    calls.clear()
    assert (await agent.run('hello', run_id='explicit')).run_id == 'explicit'
    assert calls == []


@pytest.mark.parametrize('provided', [None, 'container-run-id'])
async def test_combined_default_run_id_override(provided: str | None) -> None:
    """A wrapped container override owns bootstrap even when it declines to supply an ID."""

    class LegacyCombined(CombinedCapability[object]):
        def get_default_run_id(self) -> str | None:
            return provided

    class ChildCapability(AbstractCapability[object]):
        def get_default_run_id(self) -> str:
            return 'child-run-id'

    child = ChildCapability()
    assert child.get_default_run_id() == 'child-run-id'
    # A wrapper retains the group; plain nested combined capabilities are flattened during composition.
    capability = WrapperCapability(LegacyCombined([child]))
    result = await Agent(TestModel(), capabilities=[capability]).run('hello')
    if provided is None:
        assert uuid.UUID(result.run_id).version == 7
    else:
        assert result.run_id == provided


@pytest.mark.parametrize('provided', [None, 'legacy-run-id'])
def test_public_default_run_id_container_delegation(provided: str | None) -> None:
    class LegacyCapability(AbstractCapability[object]):
        def get_default_run_id(self) -> str | None:
            return provided

    capability = LegacyCapability()
    assert WrapperCapability(capability).get_default_run_id() == provided
    assert CombinedCapability([capability, AbstractCapability[object]()]).get_default_run_id() == provided
