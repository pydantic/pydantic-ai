"""Wrap an agent in one of the deprecated durable wrapper agents, which refuse some capabilities at construction.

The wrappers add no durability capability, so a capability's own `for_agent` refusal cannot see them; each wrapper
refuses such a capability itself (#14259).
"""

from __future__ import annotations

import warnings
from typing import Literal, TypeAlias

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import PydanticAIDeprecationWarning

__all__ = ('LEGACY_DURABLE_WRAPPERS', 'LegacyDurableWrapper', 'wrap_in_legacy_durable_agent')

LegacyDurableWrapper: TypeAlias = Literal['TemporalAgent', 'DBOSAgent', 'PrefectAgent']

LEGACY_DURABLE_WRAPPERS: tuple[LegacyDurableWrapper, ...] = ('TemporalAgent', 'DBOSAgent', 'PrefectAgent')


def wrap_in_legacy_durable_agent(wrapper: LegacyDurableWrapper, agent: Agent[None, str]) -> None:
    """Wrap `agent` in `wrapper`, skipping the test when the engine's optional extra is not installed."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PydanticAIDeprecationWarning)
        if wrapper == 'TemporalAgent':
            pytest.importorskip('temporalio')
            from pydantic_ai.durable_exec.temporal import TemporalAgent  # needs the temporal extra

            TemporalAgent(agent)  # pyright: ignore[reportDeprecated]
        elif wrapper == 'DBOSAgent':
            pytest.importorskip('dbos')
            from pydantic_ai.durable_exec.dbos import DBOSAgent  # needs the dbos extra

            DBOSAgent(agent)  # pyright: ignore[reportDeprecated]
        else:
            pytest.importorskip('prefect')
            from pydantic_ai.durable_exec.prefect import PrefectAgent  # needs the prefect extra

            PrefectAgent(agent)  # pyright: ignore[reportDeprecated]
