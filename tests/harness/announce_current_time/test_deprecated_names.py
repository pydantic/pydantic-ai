"""The old `pydantic_ai_harness.current_time` name still imports, with a `HarnessDeprecationWarning`, and old specs still load."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass

import pytest

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.announce_current_time import AnnounceCurrentTime


def test_shim_warns_and_aliases_the_old_name() -> None:
    # Import once quietly so the deprecation fires on the reload we assert on, even if
    # another test already imported the shim.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shim = importlib.import_module('pydantic_ai_harness.current_time')

    with pytest.warns(
        HarnessDeprecationWarning, match=r'renamed to `pydantic_ai_harness\.announce_current_time`.*verb phrase'
    ):
        importlib.reload(shim)

    assert issubclass(shim.CurrentTime, AnnounceCurrentTime)


def _import_old_class() -> type[AnnounceCurrentTime[None]]:
    """The shim warns once on import; these tests assert the class's own warnings instead."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from pydantic_ai_harness.current_time import CurrentTime  # pyright: ignore[reportDeprecated]

    return CurrentTime  # pyright: ignore[reportDeprecated]


_CLASS_WARNING = r'`CurrentTime` is now `AnnounceCurrentTime`.*rename the `CurrentTime` block to `AnnounceCurrentTime`'


def test_root_old_name_resolves_to_shim() -> None:
    """The old name resolves from the root, but stays out of `__all__`, as other renamed names do.

    A star import resolves every name in `__all__`, so listing it would import the shim and warn for every user.
    """
    import pydantic_ai_harness

    assert 'CurrentTime' not in pydantic_ai_harness.__all__
    old = _import_old_class()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert pydantic_ai_harness.CurrentTime is old  # pyright: ignore[reportDeprecated]


def test_constructing_old_class_warns_and_keeps_the_id() -> None:
    """Every construction warns, not only the first import, so a long-running process that loads old specs sees it.

    The `id` stays `current_time`, so conversations recorded before the rename keep their instruction key.
    """
    old = _import_old_class()
    with pytest.warns(HarnessDeprecationWarning, match=_CLASS_WARNING):
        capability = old(timezone='Europe/Amsterdam')
    assert isinstance(capability, AnnounceCurrentTime)
    assert capability.id == AnnounceCurrentTime[None]().id == 'current_time'


def test_subclass_of_old_name_warns_once_and_keeps_its_own_spec_name() -> None:
    """Defining a subclass warns, since its base class has to migrate; creating one doesn't warn again.

    As before the rename, the subclass's spec block name is its own class name.
    """
    old = _import_old_class()
    with pytest.warns(HarnessDeprecationWarning, match=_CLASS_WARNING):

        @dataclass(init=False)
        class MyClock(old):
            pass

    assert MyClock.get_serialization_name() == 'MyClock'
    agent = Agent.from_spec(
        {'model': 'test', 'capabilities': [{'MyClock': {'timezone': 'Europe/Amsterdam'}}]},
        custom_capability_types=[MyClock],
    )
    assert [type(c) for c in agent.root_capability.capabilities if isinstance(c, AnnounceCurrentTime)] == [MyClock]


def test_old_spec_block_name_loads() -> None:
    old = _import_old_class()
    with pytest.warns(HarnessDeprecationWarning, match=_CLASS_WARNING):
        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'CurrentTime': {'timezone': 'Europe/Amsterdam'}}]},
            custom_capability_types=[old],
        )

    assert [c.timezone for c in agent.root_capability.capabilities if isinstance(c, AnnounceCurrentTime)] == [
        'Europe/Amsterdam'
    ]


def test_old_spec_block_name_publishes_its_fields() -> None:
    """The alias's `init=False` keeps its fields in the `AgentSpec` schema, so old spec blocks keep validating."""
    old = _import_old_class()
    properties = AgentSpec.model_json_schema_with_capabilities([old])['$defs']['spec_params_CurrentTime']['properties']

    assert set(properties) == {'id', 'description', 'defer_loading', 'timezone', 'precision'}
