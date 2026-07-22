from __future__ import annotations

try:
    import temporalio  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:
    raise ImportError(
        'Please install the `temporalio` package to use the Temporal integration, '
        'you can use the `temporal` optional group — `pip install "pydantic-ai-slim[temporal]"`'
    ) from _import_error

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

from ...agent.abstract import AbstractAgent
from ...exceptions import AgentRunError, UserError
from ._agent import TemporalAgent  # pyright: ignore[reportDeprecated]
from ._durability import TemporalDurability
from ._logfire import LogfirePlugin
from ._run_context import TemporalRunContext
from ._toolset import TemporalWrapperToolset
from ._workflow import PydanticAIWorkflow

__all__ = [
    'TemporalAgent',
    'TemporalDurability',
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
import anyio._backends._asyncio  # pyright: ignore[reportUnusedImport]  #noqa: F401

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
        return runner

    return replace(
        runner,
        restrictions=runner.restrictions.with_passthrough_modules(
            'pydantic_ai',
            'pydantic',
            'pydantic_core',
            'pydantic_monty',
            'logfire',
            'rich',
            'httpx',
            'anyio',
            'sniffio',
            'httpcore',
            # `certifi` is imported lazily by `httpx`/`ssl` when a client builds its TLS context. A
            # model constructed inside the workflow (e.g. a `gateway/` model resolved via
            # `infer_model`) creates its own HTTP client there, so without passing `certifi` through
            # alongside the rest of the HTTP stack Temporal warns that it was "imported after initial
            # workflow load" (a hard error under `filterwarnings=error`).
            'certifi',
            # `fastmcp` (and the `mcp` SDK it transitively imports) calls `Path.expanduser` at
            # import time when resolving its config directory — restricted by the workflow
            # sandbox. Safe to pass through: the call only happens once at module init.
            'fastmcp',
            'mcp',
            # The `anthropic` SDK (>=0.99.0) calls `Path.home()` during client construction to
            # resolve its credentials/profile config directory (`~/.config/anthropic`) — restricted
            # by the workflow sandbox. This trips when a model is constructed inside the workflow,
            # e.g. a `gateway/anthropic:` or `anthropic:` model resolved lazily via `infer_model`.
            # Safe to pass through: a deterministic, read-only config lookup.
            'anthropic',
            # The `google-genai` SDK lazily imports `google.auth` submodules (e.g.
            # `google.auth.aio.credentials`) while constructing its client, which Temporal flags as
            # "imported after initial workflow load" when a `gateway/google-cloud:` (or `google-*:`)
            # model is built inside the workflow.
            'google.auth',
            # Used by fastmcp via py-key-value-aio
            'beartype',
            # Imported inside `logfire._internal.json_encoder` when running `logfire.info` inside an activity with attributes to serialize
            'attrs',
            # Imported inside `logfire._internal.json_schema` when running `logfire.info` inside an activity with attributes to serialize
            'numpy',
            'pandas',
            # `response.cost()` lazily imports `genai_prices` (and its `httpx2` dependency) on first call.
            # When cost is calculated inside a workflow, the sandbox re-imports that chain and `httpx2._models`
            # subclasses `urllib.request.Request`, which is restricted unless `genai_prices`/`httpx2` are passed
            # through alongside the rest of the HTTP stack.
            'genai_prices',
            'httpx2',
        ),
    )


class PydanticAIPlugin(SimplePlugin):
    """Temporal client and worker plugin for Pydantic AI."""

    def __init__(self) -> None:
        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='PydanticAIPlugin',
            data_converter=_data_converter,
            workflow_runner=_workflow_runner,
            # `AgentRunError` covers deterministic run failures that can now surface in
            # workflow code, like `UsageLimitExceeded` and the `UnexpectedModelBehavior`
            # continuation ceilings raised by the workflow-side continuation loop: they
            # must fail the workflow (preserving the exception type for the caller)
            # rather than fail the workflow *task*, which Temporal would retry forever.
            workflow_failure_exception_types=[UserError, PydanticUserError, AgentRunError],
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        config = super().configure_worker(config)

        workflows = list(config.get('workflows', []))  # type: ignore[reportUnknownMemberType]
        activities = list(config.get('activities', []))  # type: ignore[reportUnknownMemberType]

        for workflow_class in workflows:
            agents = getattr(workflow_class, '__pydantic_ai_agents__', None)
            if agents is None:
                continue
            if not isinstance(agents, Sequence):
                raise TypeError(  # pragma: no cover
                    f'__pydantic_ai_agents__ must be a Sequence of TemporalAgent instances, got {type(agents)}'
                )
            for agent in agents:  # type: ignore[reportUnknownVariableType]
                if isinstance(agent, TemporalAgent):  # pyright: ignore[reportDeprecated]
                    # Deprecated path: `TemporalAgent` is being phased out in favor of
                    # `capabilities=[TemporalDurability(...)]` on a regular `Agent`. Kept
                    # working so existing workers keep loading without changes.
                    activities.extend(agent.temporal_activities)  # type: ignore[reportUnknownMemberType]
                elif isinstance(agent, AbstractAgent):
                    durability = TemporalDurability.from_agent(agent)  # type: ignore[reportUnknownArgumentType]
                    if durability is None:
                        raise UserError(
                            f'Agent {agent.name!r} listed in `__pydantic_ai_agents__` has no '
                            '`TemporalDurability` capability; add one to `capabilities=[...]`.'
                        )
                    activities.extend(durability.temporal_activities)  # type: ignore[reportUnknownMemberType]
                else:
                    raise TypeError(  # pragma: no cover
                        f'__pydantic_ai_agents__ items must be TemporalAgent or AbstractAgent, got {type(agent)}'  # type: ignore[reportUnknownVariableType]
                    )

        config['activities'] = activities

        return config


class AgentPlugin(SimplePlugin):
    """Temporal worker plugin for a specific Pydantic AI agent.

    Accepts either a regular `Agent` carrying a
    [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]
    capability (whose chain is walked to find the bound capability), or the
    deprecated [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent]
    wrapper, and registers the agent's activities on the worker.
    """

    def __init__(self, agent: AbstractAgent[Any, Any]):
        if isinstance(agent, TemporalAgent):  # pyright: ignore[reportDeprecated]
            activities = agent.temporal_activities
        else:
            durability = TemporalDurability.from_agent(agent)
            if durability is None:
                raise UserError(
                    f'Agent {agent.name!r} has no `TemporalDurability` capability; '
                    'add one to `capabilities=[...]` before constructing the plugin.'
                )
            activities = durability.temporal_activities
        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='AgentPlugin',
            activities=activities,
        )
