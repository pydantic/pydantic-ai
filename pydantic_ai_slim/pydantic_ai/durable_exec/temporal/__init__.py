from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from pydantic.errors import PydanticUserError
from temporalio.contrib.pydantic import PydanticPayloadConverter, pydantic_data_converter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkerConfig, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from ...exceptions import UserError
from ._agent import TemporalAgent
from ._logfire import LogfirePlugin
from ._run_context import TemporalRunContext
from ._toolset import TemporalWrapperToolset
from ._workflow import PydanticAIWorkflow

__all__ = [
    'TemporalAgent',
    'PydanticAIPlugin',
    'LogfirePlugin',
    'AgentPlugin',
    'TemporalRunContext',
    'TemporalWrapperToolset',
    'PydanticAIWorkflow',
]

# We need eagerly import the anyio backends or it will happens inside workflow code and temporal has issues
# Note: It's difficult to add a test that covers this because pytest presumably does these imports itself
# when you have a @pytest.mark.anyio somewhere.
# I suppose we could add a test that runs a python script in a separate process, but I have not done that...
import anyio._backends._asyncio  # pyright: ignore[reportUnusedImport]  # noqa: F401

try:
    import anyio._backends._trio  # pyright: ignore[reportUnusedImport]  # noqa: F401
except ImportError:
    pass


def _data_converter(converter: DataConverter | None) -> DataConverter:
    if converter is None:
        return pydantic_data_converter

    # If the payload converter class is already a subclass of PydanticPayloadConverter,
    # the converter is already compatible with Pydantic AI - return it as-is.
    if issubclass(converter.payload_converter_class, PydanticPayloadConverter):
        return converter

    # If using a non-Pydantic payload converter, warn and replace just the payload converter class,
    # preserving any custom payload_codec or failure_converter_class.
    if converter.payload_converter_class is not DefaultPayloadConverter:
        warnings.warn(
            'A non-Pydantic Temporal payload converter was used which has been replaced with PydanticPayloadConverter. '
            'To suppress this warning, ensure your payload_converter_class inherits from PydanticPayloadConverter.'
        )

    return replace(converter, payload_converter_class=PydanticPayloadConverter)


def _workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
    if not runner:
        raise ValueError('No WorkflowRunner provided to the Pydantic AI plugin.')  # pragma: no cover

    if not isinstance(runner, SandboxedWorkflowRunner):
        return runner  # pragma: no cover

    return replace(
        runner,
        restrictions=runner.restrictions.with_passthrough_modules(
            'pydantic_ai',
            'pydantic',
            'pydantic_core',
            'logfire',
            'rich',
            'httpx',
            'anyio',
            'sniffio',
            'httpcore',
            # Used by fastmcp via py-key-value-aio
            'beartype',
            # Imported inside `logfire._internal.json_encoder` when running `logfire.info` inside an activity with attributes to serialize
            'attrs',
            # Imported inside `logfire._internal.json_schema` when running `logfire.info` inside an activity with attributes to serialize
            'numpy',
            'pandas',
        ),
    )


class PydanticAIPlugin(SimplePlugin):
    """Temporal client and worker plugin for Pydantic AI."""

    def __init__(self):
        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='PydanticAIPlugin',
            data_converter=_data_converter,
            workflow_runner=_workflow_runner,
            workflow_failure_exception_types=[UserError, PydanticUserError],
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        config = super().configure_worker(config)

        workflows = list(config.get('workflows', []))  # type: ignore[reportUnknownMemberType]
        activities = list(config.get('activities', []))  # type: ignore[reportUnknownMemberType]

        for workflow_class in workflows:  # type: ignore[reportUnknownMemberType]
            agents = getattr(workflow_class, '__pydantic_ai_agents__', None)  # type: ignore[reportUnknownMemberType]
            if agents is None:
                continue
            if not isinstance(agents, Sequence):
                raise TypeError(  # pragma: no cover
                    f'__pydantic_ai_agents__ must be a Sequence of TemporalAgent instances, got {type(agents)}'
                )
            for agent in agents:  # type: ignore[reportUnknownVariableType]
                if not isinstance(agent, TemporalAgent):
                    raise TypeError(  # pragma: no cover
                        f'__pydantic_ai_agents__ must be a Sequence of TemporalAgent, got {type(agent)}'  # type: ignore[reportUnknownVariableType]
                    )
                activities.extend(agent.temporal_activities)  # type: ignore[reportUnknownMemberType]

        config['activities'] = activities

        return config


class AgentPlugin(SimplePlugin):
    """Temporal worker plugin for a specific Pydantic AI agent."""

    def __init__(self, agent: TemporalAgent[Any, Any]):
        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='AgentPlugin',
            activities=agent.temporal_activities,
        )
