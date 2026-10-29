"""Compatibility of the public no-argument capability identity hook."""

from __future__ import annotations

import uuid

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.models.test import TestModel


@pytest.mark.parametrize('provided', [None, 'legacy-run-id'])
@pytest.mark.parametrize('container', ['leaf', 'wrapper', 'combined', 'wrapper_override'])
async def test_legacy_default_run_id_hook(provided: str | None, container: str) -> None:
    calls: list[str] = []

    class LegacyCapability(AbstractCapability[object]):
        def get_default_run_id(self) -> str | None:
            calls.append('leaf')
            return provided

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
def test_combined_default_run_id_override(provided: str | None) -> None:
    """A container override owns bootstrap dispatch even when it declines to supply an ID."""

    class LegacyCombined(CombinedCapability[object]):
        def get_default_run_id(self) -> str | None:
            return provided

    capability = LegacyCombined([AbstractCapability[object]()])
    assert capability._get_default_run_id_for_inputs({}) == provided  # pyright: ignore[reportPrivateUsage]
