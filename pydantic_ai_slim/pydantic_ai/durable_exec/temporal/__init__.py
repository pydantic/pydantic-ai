from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Callable

from pydantic.errors import PydanticUserError
from temporalio.client import ClientConfig, Plugin as ClientPlugin
from temporalio.contrib.pydantic import PydanticPayloadConverter, pydantic_data_converter
from temporalio.converter import DefaultPayloadConverter
from temporalio.worker import Plugin as WorkerPlugin, WorkerConfig
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from ...exceptions import UserError
from ._agent import TemporalAgent
from ._logfire import LogfirePlugin
from ._run_context import TemporalRunContext
from ._toolset import TemporalWrapperToolset

__all__ = [
    'TemporalAgent',
    'PydanticAIPlugin',
    'LogfirePlugin',
    'AgentPlugin',
    'TemporalRunContext',
    'TemporalWrapperToolset',
]


class PydanticAIPlugin(ClientPlugin, WorkerPlugin):
    """Temporal client and worker plugin for Pydantic AI."""

    def configure_client(self, config: ClientConfig) -> ClientConfig:
        if (data_converter := config.get('data_converter')) and data_converter.payload_converter_class not in (
            DefaultPayloadConverter,
            PydanticPayloadConverter,
        ):
            warnings.warn(  # pragma: no cover
                'A non-default Temporal data converter was used which has been replaced with the Pydantic data converter.'
            )

        config['data_converter'] = pydantic_data_converter
        return super().configure_client(config)

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        runner = config.get('workflow_runner')  # pyright: ignore[reportUnknownMemberType]
        if isinstance(runner, SandboxedWorkflowRunner):  # pragma: no branch
            config['workflow_runner'] = replace(
                runner,
                restrictions=runner.restrictions.with_passthrough_modules(
                    'pydantic_ai',
                    'logfire',
                    'rich',
                    'httpx',
                    # Imported inside `logfire._internal.json_encoder` when running `logfire.info` inside an activity with attributes to serialize
                    'attrs',
                    # Imported inside `logfire._internal.json_schema` when running `logfire.info` inside an activity with attributes to serialize
                    'numpy',
                    'pandas',
                ),
            )

        config['workflow_failure_exception_types'] = [
            *config.get('workflow_failure_exception_types', []),  # pyright: ignore[reportUnknownMemberType]
            UserError,
            PydanticUserError,
        ]

        return super().configure_worker(config)


class AgentPlugin(WorkerPlugin):
    """Temporal worker plugin for a specific Pydantic AI agent."""

    def __init__(self, agent: TemporalAgent[Any, Any]):
        self.agent = agent

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        activities: Sequence[Callable[..., Any]] = config.get('activities', [])  # pyright: ignore[reportUnknownMemberType]
        # Activities are checked for name conflicts by Temporal.
        config['activities'] = [*activities, *self.agent.temporal_activities]
        return super().configure_worker(config)
