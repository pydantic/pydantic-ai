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
from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkerConfig, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from pydantic_graph.exceptions import UnsupportedEventLoopError

from ...agent.abstract import AbstractAgent
from ...exceptions import AgentRunError, UserError
from ._agent import TemporalAgent  # pyright: ignore[reportDeprecated]
from ._durability import TemporalDurability
from ._logfire import LogfirePlugin
from ._payload_converter import PydanticAIPayloadConverter
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
    'PydanticAIPayloadConverter',
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
        return DataConverter(payload_converter_class=PydanticAIPayloadConverter)

    # Preserve genuine subclasses because replacing one could silently discard custom behavior. Authors
    # can inherit from `PydanticAIPayloadConverter` when they also want memoized adapter construction.
    if converter.payload_converter_class is not PydanticPayloadConverter and issubclass(
        converter.payload_converter_class, PydanticPayloadConverter
    ):
        return converter

    # If using a non-Pydantic payload converter, warn and replace just the payload converter class,
    # preserving any custom payload_codec or failure_converter_class.
    if converter.payload_converter_class not in (DefaultPayloadConverter, PydanticPayloadConverter):
        warnings.warn(
            'A non-Pydantic Temporal payload converter was used which has been replaced with '
            '`PydanticAIPayloadConverter`. To suppress this warning and retain memoized `TypeAdapter` construction, '
            'ensure your `payload_converter_class` inherits from `PydanticAIPayloadConverter`.'
        )

    return replace(converter, payload_converter_class=PydanticAIPayloadConverter)


def _workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
    if not runner:
        raise ValueError('No WorkflowRunner provided to the Pydantic AI plugin.')  # pragma: no cover

    if not isinstance(runner, SandboxedWorkflowRunner):
        return runner

    return replace(
        runner,
        restrictions=runner.restrictions.with_passthrough_modules(
            'pydantic_ai',
            'pydantic_graph',
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
            # Registering a per-toolset `child_workflow`-tagged tool-call workflow class
            # (`_function_toolset.py`) with a worker makes the sandbox re-import its defining module
            # to confirm the `workflow_failure_exception_types` classes resolve to the same objects
            # inside and outside the sandbox. That re-import transitively pulls in
            # `opentelemetry.context`, which reads `os.environ` at import time — restricted unless
            # passed through.
            'opentelemetry',
            # Decoding a `child_workflow` tool-call workflow argument (`CallToolParams`, a dataclass)
            # is the first schema Pydantic builds for a dataclass field *inside* the sandbox on some
            # code paths; that lazily imports `annotated_types` (a core pydantic dependency) after
            # the sandbox's initial workflow-module load, which the sandbox otherwise warns about —
            # and under `filterwarnings=error` that warning becomes a real decode failure, which
            # fails the workflow *task* (retried by Temporal) rather than the workflow itself.
            'annotated_types',
        ),
    )


def _temporal_workflow_name(workflow_class: type[Any]) -> str | None:
    """The registered Temporal workflow type name for a `@workflow.defn`-decorated class, if any."""
    defn = getattr(workflow_class, '__temporal_workflow_definition', None)
    return getattr(defn, 'name', None)


def _merge_temporal_workflows(existing: list[type[Any]], new: Sequence[type[Any]]) -> list[type[Any]]:
    """Append `new` onto `existing` in place, deduped by identity; raise on a colliding workflow name.

    Mutates `existing` rather than returning a copy so a caller iterating over the same list object
    while appending to it (as `PydanticAIPlugin.configure_worker` does below) keeps working.

    Temporal's own `Worker()` construction already rejects two entries for the same workflow name (the
    same protection activities get natively, scoped per `Worker`) — this gives the same protection
    earlier, with a message naming the collision, for the case that check is built to catch: two
    different `TemporalDurability` bindings whose agent name and toolset `id` happen to collide,
    combined onto the same worker (see the per-toolset workflow class built in `_function_toolset.py`).

    Deliberately scoped to *here* — the point classes are actually combined for a specific worker —
    rather than eagerly wherever `temporalize_function_toolset` runs: that would also flag benign
    cases, like two independently-bound agents that are never registered on the same worker together
    (e.g. an old and a new implementation of the same agent kept side by side, sharing a name on
    purpose, for replay testing across a capability migration).
    """
    names = {name: wf for wf in existing if (name := _temporal_workflow_name(wf)) is not None}
    for wf in new:
        if wf in existing:
            continue
        name = _temporal_workflow_name(wf)
        colliding = names.get(name) if name is not None else None
        if colliding is not None:
            raise UserError(
                f'Two different toolsets are both registered for child-workflow dispatch under the '
                f'workflow name {name!r} ({colliding.__qualname__!r} and {wf.__qualname__!r}). Give '
                'each agent or toolset a distinct name/`id` so their child-workflow dispatch names '
                'do not collide.'
            )
        existing.append(wf)
        if name is not None:
            names[name] = wf
    return existing


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
            # `UnsupportedEventLoopError` is raised by `pydantic_graph`'s sync entry points
            # (e.g. `Graph.run_sync()`), which don't go through the `pydantic_ai` wrapper that
            # would otherwise turn it into a `UserError`; without it those would hang the same way.
            workflow_failure_exception_types=[
                UserError,
                PydanticUserError,
                AgentRunError,
                UnsupportedEventLoopError,
            ],
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
                    # The same bound agent can be reachable via multiple plugins or also listed in
                    # user `workflows=[...]`, contributing the identical per-toolset workflow class
                    # object twice; dedupe by identity. Two *different* agents whose child-workflow
                    # dispatch name collides is a separate, real misconfiguration — see
                    # `_merge_temporal_workflows`.
                    workflows = _merge_temporal_workflows(workflows, durability.temporal_workflows)
                else:
                    raise TypeError(  # pragma: no cover
                        f'__pydantic_ai_agents__ items must be TemporalAgent or AbstractAgent, got {type(agent)}'  # type: ignore[reportUnknownVariableType]
                    )

        config['activities'] = activities
        config['workflows'] = workflows

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
        workflows: list[type[Any]] = []
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
            workflows = durability.temporal_workflows
        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='AgentPlugin',
            activities=activities,
            workflows=workflows,
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        config = super().configure_worker(config)
        # `SimplePlugin.configure_worker` (just called via `super()`) appends this plugin's own
        # `workflows=` to whatever earlier plugins already contributed, without deduplicating —
        # unlike activities, the Temporal `Worker` itself rejects two entries for the same
        # workflow name outright. The same bound agent can be registered through multiple plugins,
        # contributing identical per-toolset workflow class objects twice; dedupe by identity, and
        # raise if two *different* agents' collide on the registered workflow name (see
        # `_merge_temporal_workflows`) each time a plugin runs, so it's clean by the last one.
        workflows = config.get('workflows')  # type: ignore[reportUnknownMemberType]
        if workflows:
            config['workflows'] = _merge_temporal_workflows([], workflows)
        return config
