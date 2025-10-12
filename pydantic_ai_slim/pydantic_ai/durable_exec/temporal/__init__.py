from __future__ import annotations

import warnings
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from typing import Any

from pydantic.errors import PydanticUserError
from temporalio.client import ClientConfig, Plugin as ClientPlugin, WorkflowHistory
from temporalio.contrib.pydantic import PydanticPayloadConverter, pydantic_data_converter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.service import ConnectConfig, ServiceClient
from temporalio.worker import (
    Plugin as WorkerPlugin,
    Replayer,
    ReplayerConfig,
    Worker,
    WorkerConfig,
    WorkflowReplayResult,
)
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

    def init_client_plugin(self, next: ClientPlugin) -> None:
        self.next_client_plugin = next

    def init_worker_plugin(self, next: WorkerPlugin) -> None:
        self.next_worker_plugin = next

    def configure_client(self, config: ClientConfig) -> ClientConfig:
        config['data_converter'] = self._get_new_data_converter(config.get('data_converter'))
        return self.next_client_plugin.configure_client(config)

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        runner = config.get('workflow_runner')  # pyright: ignore[reportUnknownMemberType]
        if isinstance(runner, SandboxedWorkflowRunner):  # pragma: no branch
            config['workflow_runner'] = replace(
                runner,
                restrictions=runner.restrictions.with_passthrough_modules(
                    'pydantic_ai',
                    'pydantic',
                    'pydantic_core',
                    'logfire',
                    'rich',
                    'httpx',
                    'httpcore',
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

        return self.next_worker_plugin.configure_worker(config)

    async def connect_service_client(self, config: ConnectConfig) -> ServiceClient:
        return await self.next_client_plugin.connect_service_client(config)

    async def run_worker(self, worker: Worker) -> None:
        await self.next_worker_plugin.run_worker(worker)

    def configure_replayer(self, config: ReplayerConfig) -> ReplayerConfig:  # pragma: no cover
        config['data_converter'] = self._get_new_data_converter(config.get('data_converter'))  # pyright: ignore[reportUnknownMemberType]
        return self.next_worker_plugin.configure_replayer(config)

    def run_replayer(
        self,
        replayer: Replayer,
        histories: AsyncIterator[WorkflowHistory],
    ) -> AbstractAsyncContextManager[AsyncIterator[WorkflowReplayResult]]:  # pragma: no cover
        return self.next_worker_plugin.run_replayer(replayer, histories)

    def _get_new_data_converter(self, converter: DataConverter | None) -> DataConverter:
        if converter and converter.payload_converter_class not in (
            DefaultPayloadConverter,
            PydanticPayloadConverter,
        ):
            warnings.warn(  # pragma: no cover
                'A non-default Temporal data converter was used which has been replaced with the Pydantic data converter.'
            )

        return pydantic_data_converter


class AgentPlugin(WorkerPlugin):
    """Temporal worker plugin for a specific Pydantic AI agent."""

    def __init__(self, agent: TemporalAgent[Any, Any]):
        self.agent = agent

    def init_worker_plugin(self, next: WorkerPlugin) -> None:
        self.next_worker_plugin = next

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        activities: Sequence[Callable[..., Any]] = config.get('activities', [])  # pyright: ignore[reportUnknownMemberType]
        # Activities are checked for name conflicts by Temporal.
        config['activities'] = [*activities, *self.agent.temporal_activities]
        return self.next_worker_plugin.configure_worker(config)

    async def run_worker(self, worker: Worker) -> None:
        await self.next_worker_plugin.run_worker(worker)

    def configure_replayer(self, config: ReplayerConfig) -> ReplayerConfig:  # pragma: no cover
        return self.next_worker_plugin.configure_replayer(config)

    def run_replayer(
        self,
        replayer: Replayer,
        histories: AsyncIterator[WorkflowHistory],
    ) -> AbstractAsyncContextManager[AsyncIterator[WorkflowReplayResult]]:  # pragma: no cover
        return self.next_worker_plugin.run_replayer(replayer, histories)
