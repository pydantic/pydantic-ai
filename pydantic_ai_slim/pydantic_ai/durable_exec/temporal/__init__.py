from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

from pydantic.errors import PydanticUserError
from temporalio.contrib.pydantic import PydanticPayloadConverter, pydantic_data_converter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from ...exceptions import UserError
from ._agent import TemporalAgent
from ._logfire import LogfirePlugin
from ._run_context import TemporalRunContext, get_activity_deps, register_activity_deps
from ._toolset import TemporalWrapperToolset

__all__ = [
    'TemporalAgent',
    'PydanticAIPlugin',
    'LogfirePlugin',
    'AgentPlugin',
    'TemporalRunContext',
    'TemporalWrapperToolset',
    'get_activity_deps',
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
    if converter and converter.payload_converter_class not in (
        DefaultPayloadConverter,
        PydanticPayloadConverter,
    ):
        warnings.warn(  # pragma: no cover
            'A non-default Temporal data converter was used which has been replaced with the Pydantic data converter.'
        )

    return pydantic_data_converter


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


class AgentPlugin(SimplePlugin):
    """Temporal worker plugin for a specific Pydantic AI agent.

    Args:
        agent: The TemporalAgent to register activities for.
        activity_deps: Optional non-serializable dependencies that will be available
            inside activities via [`get_activity_deps()`][pydantic_ai.durable_exec.temporal.get_activity_deps].
            This is useful for injecting database clients, Temporal clients, or other
            resources that should be initialized once at worker startup.

    Example:
    ```python {test="skip" lint="skip"}
    from dataclasses import dataclass
    from temporalio.client import Client
    from temporalio.worker import Worker
    from pydantic_ai.durable_exec.temporal import AgentPlugin, get_activity_deps

    @dataclass
    class ActivityDeps:
        temporal_client: Client
        db_pool: DatabasePool

    # At worker startup:
    activity_deps = ActivityDeps(
        temporal_client=temporal_client,
        db_pool=get_db_pool(),
    )

    worker = Worker(
        temporal_client,
        task_queue='my-queue',
        workflows=[MyWorkflow],
        plugins=[AgentPlugin(agent=temporal_agent, activity_deps=activity_deps)],
    )

    # In a tool:
    @agent.tool
    async def my_tool(ctx: RunContext[MyDeps]) -> str:
        activity_deps = get_activity_deps()
        return await activity_deps.db_pool.fetch('...')
    ```
    """

    def __init__(self, agent: TemporalAgent[Any, Any], activity_deps: Any = None):
        if activity_deps is not None:
            # TemporalAgent validates that name is set at construction
            assert agent.name is not None
            register_activity_deps(agent.name, activity_deps)
        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='AgentPlugin',
            activities=agent.temporal_activities,
        )
