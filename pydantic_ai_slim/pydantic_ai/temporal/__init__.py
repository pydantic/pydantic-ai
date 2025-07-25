from __future__ import annotations

import contextlib
from typing import Any, Callable

from temporalio import workflow

from pydantic_ai.agent import Agent
from pydantic_ai.toolsets.abstract import AbstractToolset

from ..models import Model
from ._model import temporalize_model
from ._run_context import TemporalRunContext
from ._settings import TemporalSettings
from ._toolset import temporalize_toolset

__all__ = [
    'initialize_temporal',
    'TemporalSettings',
    'TemporalRunContext',
]


def initialize_temporal():
    """Initialize Temporal."""
    with workflow.unsafe.imports_passed_through():
        with contextlib.suppress(ModuleNotFoundError):
            import pandas  # pyright: ignore[reportUnusedImport]  # noqa: F401


def temporalize_agent(
    agent: Agent[Any, Any],
    settings: TemporalSettings | None = None,
    temporalize_toolset_func: Callable[
        [AbstractToolset, TemporalSettings | None], list[Callable[..., Any]]
    ] = temporalize_toolset,
) -> list[Callable[..., Any]]:
    """Temporalize an agent.

    Args:
        agent: The agent to temporalize.
        settings: The temporal settings to use.
        temporalize_toolset_func: The function to use to temporalize the toolsets.
    """
    if existing_activities := getattr(agent, '__temporal_activities', None):
        return existing_activities

    settings = settings or TemporalSettings()

    # TODO: Doesn't consider model/toolsets passed at iter time.

    activities: list[Callable[..., Any]] = []
    if isinstance(agent.model, Model):
        activities.extend(temporalize_model(agent.model, settings, agent._event_stream_handler))  # pyright: ignore[reportPrivateUsage]

    def temporalize_toolset(toolset: AbstractToolset) -> None:
        activities.extend(temporalize_toolset_func(toolset, settings))

    agent.toolset.apply(temporalize_toolset)

    setattr(agent, '__temporal_activities', activities)
    return activities


# TODO: untemporalize_agent
