"""Toolset for the `DynamicWorkflow` capability.

Exposes the `run_workflow` tool: the model writes a Python orchestration script (run in a Monty
sandbox) that calls sub-agents as async functions and composes their results -- fan-out, chaining,
voting, loops -- in one step, or runs a saved workflow by name. With saving enabled it also
exposes `save_workflow`, which writes a script to the workflow library for later runs.
"""

from __future__ import annotations

import copy
import posixpath
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, TypeAdapter
from typing_extensions import NotRequired, Self, TypedDict

from pydantic_ai import AbstractToolset, RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool
from pydantic_ai.usage import UsageLimits
from pydantic_ai.workspaces import WorkspaceError

try:
    from pydantic_monty import (
        MontyCrashedError,
        MontyRuntimeError,
        MontySyntaxError,
        MontyTypingError,
        ResourceLimits,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for DynamicWorkflow. '
        'Install it with: uv add "pydantic-ai-harness[dynamic-workflow]"'
    ) from _import_error

from pydantic_ai_harness._monty_exec import in_temporal_workflow, is_sandbox_panic
from pydantic_ai_harness._workspace import raise_tool_failure, workspace_path
from pydantic_ai_harness.dynamic_workflow._catalog import (
    WorkflowAgent,
    build_type_check_stubs,
    index_workflow_agents,
    render_description,
    render_reveal,
    validate_workflow_agent,
)
from pydantic_ai_harness.dynamic_workflow._library import SavedWorkflow, WorkflowLibrary
from pydantic_ai_harness.dynamic_workflow._prelude import ARGS_NAME, HELPER_NAMES, render_prelude
from pydantic_ai_harness.dynamic_workflow._results import (
    budget_terminal_result,
    completed_retry_section,
    completed_workflow_result,
    worker_crash_result,
)
from pydantic_ai_harness.dynamic_workflow._run import CallBudget, Frame, WorkflowRun, in_workflow


class WorkflowResourceLimits(TypedDict, total=False):
    """Caps on the orchestration script's own sandbox resources (not sub-agent latency).

    A harness-owned view of the sandbox limits the capability supports, so the public API does
    not depend on the underlying sandbox's own types. Every field is optional; an omitted field
    keeps its backstop value.
    """

    max_duration_secs: float
    """Ceiling on the time the script spends executing sandbox code. Monty checks it per bytecode
    step, so time spent awaiting sub-agents does not count against it -- neither a single `await`
    nor a concurrent `asyncio.gather` batch, because during that wait the script is suspended on
    the host, not running sandbox code. There is no default cap. Set one to bound a pure-CPU
    `while True` loop, which would otherwise burn a core and block the event loop -- the one
    runaway the sub-agent budgets do not catch. Sleeping is not execution time either, so when set,
    the script may also sleep for up to this long in total."""

    max_memory: int
    """Maximum sandbox memory, in bytes."""


def _default_resource_limits() -> ResourceLimits:
    """Backstop sandbox limits; no duration cap -- see `WorkflowResourceLimits.max_duration_secs`."""
    return {
        'max_memory': 256 * 1024 * 1024,
    }


# The keys `WorkflowResourceLimits` accepts. A `total=False` TypedDict does not validate keys at
# runtime, so a typo (e.g. `max_durations_secs`) would otherwise merge through and be silently
# dropped -- quietly disabling the only guard against a pure-CPU `while True`. We reject unknowns.
_RESOURCE_LIMIT_KEYS = frozenset(WorkflowResourceLimits.__annotations__)


def _resolve_resource_limits(limits: WorkflowResourceLimits | Literal['unlimited'] | None) -> ResourceLimits:
    """Resolve the public `resource_limits` value to the limits handed to the sandbox.

    A partial mapping merges *onto* the backstop rather than replacing it. Full semantics:
    `DynamicWorkflow.resource_limits`.
    """
    if limits is None:
        return _default_resource_limits()
    if limits == 'unlimited':
        return {}
    unknown = set(limits) - _RESOURCE_LIMIT_KEYS
    if unknown:
        raise UserError(
            f'Unknown `resource_limits` key(s): {sorted(unknown)}. Valid keys are {sorted(_RESOURCE_LIMIT_KEYS)}.'
        )
    resolved = _default_resource_limits()
    if 'max_memory' in limits:
        resolved['max_memory'] = limits['max_memory']
    if 'max_duration_secs' in limits:
        resolved['max_feed_duration_secs'] = limits['max_duration_secs']
    return resolved


class _WorkflowArguments(TypedDict):
    code: Annotated[str, Field(description='The Python orchestration script to execute in the sandbox.')]


class _NamedWorkflowArguments(TypedDict):
    code: NotRequired[
        Annotated[
            str, Field(description='A Python orchestration script to execute in the sandbox. Pass this or `name`.')
        ]
    ]
    name: NotRequired[Annotated[str, Field(description='The name of a saved workflow to run. Pass this or `code`.')]]
    args: NotRequired[
        Annotated[dict[str, JsonValue], Field(description='Arguments for the script, bound to its `args` global.')]
    ]


class _SaveWorkflowArguments(TypedDict):
    name: Annotated[
        str, Field(description='File-stem name: 1-64 lowercase letters, digits, `-` and `_`, such as `triage-files`.')
    ]
    description: Annotated[str, Field(description='What the workflow does, shown when listing saved workflows.')]
    code: Annotated[
        str,
        Field(description='The script, as you would pass it to `run_workflow`, reading its inputs from `args`.'),
    ]
    when_to_use: NotRequired[Annotated[str, Field(description='When to reach for this workflow.')]]
    args: NotRequired[
        Annotated[dict[str, JsonValue], Field(description='JSON schema of the `args` the workflow expects.')]
    ]
    agents: NotRequired[Annotated[list[str], Field(description='Sub-agents the script calls.')]]
    returns: NotRequired[Annotated[str, Field(description='What the workflow returns.')]]
    overwrite: NotRequired[
        Annotated[bool, Field(description='Replace a saved workflow of the same name. Defaults to false.')]
    ]


def _tool_schema(arguments: type[object]) -> tuple[dict[str, Any], SchemaValidatorProt]:
    adapter = TypeAdapter(arguments)
    validator: SchemaValidatorProt = adapter.validator  # pyright: ignore[reportAssignmentType]
    return adapter.json_schema(), validator


_WORKFLOW_ARGS_SCHEMA = _tool_schema(_WorkflowArguments)
_NAMED_WORKFLOW_ARGS_SCHEMA = _tool_schema(_NamedWorkflowArguments)
_SAVE_WORKFLOW_ARGS_SCHEMA = _tool_schema(_SaveWorkflowArguments)
SAVE_TOOL_NAME = 'save_workflow'
_SAVE_DESCRIPTION = (
    'Save a workflow script to the workflow library, so this and later runs can run it by name with '
    '`run_workflow` or from a script with `await workflow(name, args)`. Save a script once it works, '
    'reading its inputs from the `args` global rather than hard-coding them.'
)
_NESTED_REFUSAL = (
    'Workflows do not nest: this sub-agent was invoked from a workflow and cannot start '
    'its own. Return your result to the orchestrating workflow instead.'
)


@dataclass(kw_only=True)
class DynamicWorkflowToolset(AbstractToolset[AgentDepsT]):
    """Toolset that runs sub-agent orchestration scripts in a Monty sandbox, and saves them."""

    agents: list[WorkflowAgent[AgentDepsT]]
    """Sub-agents callable from the orchestration script, each as an async function.

    `DynamicWorkflow.reveal()` is the only supported way to add a sub-agent mid-run.
    The toolset observes appends to this list as the reveal channel. Any entry that
    arrives invalid is a contract violation that raises."""

    tool_name: str = 'run_workflow'
    """Name of the tool exposed to the model."""

    max_agent_calls: int = 50
    """Maximum total sub-agent runs per agent run (an exact, host-enforced ceiling)."""

    max_retries: int = 3
    """Maximum retries for the `run_workflow` tool (syntax/runtime errors count as retries)."""

    forward_usage: bool = True
    """Share the parent run's `usage` accumulator with sub-agents. See
    `DynamicWorkflow.forward_usage` for what is and is not forwarded."""

    inherit_model: bool = False
    """Run every sub-agent with the parent run's resolved model instead of its constructed model.
    See `DynamicWorkflow.inherit_model` for when to use this."""

    sub_agent_usage_limits: UsageLimits | None = None
    """`UsageLimits` applied to every sub-agent run, replacing pydantic-ai's default.
    See `DynamicWorkflow.sub_agent_usage_limits` for the budgeting semantics."""

    resource_limits: WorkflowResourceLimits | Literal['unlimited'] | None = None
    """Sandbox limits guarding the orchestration script's own memory (not sub-agents).
    See `DynamicWorkflow.resource_limits` for the `None`/`'unlimited'`/partial-dict semantics."""

    toolset_id: str | None = None
    """Stable toolset id; defaults to the tool name."""

    owning_capability: AbstractCapability[AgentDepsT] | None = None
    """The `DynamicWorkflow` capability this toolset belongs to, when capability-provided.

    Used to resolve whether the capability is visible to the model on the current step (deferred
    and unloaded means hidden), by identity against the run's capability registry -- ids cannot be
    used for this, because an id-less capability is registered under a run-generated key and a
    wrapper capability is registered in place of what it wraps. `None` (a toolset used directly,
    outside any capability) is always visible."""

    default_agent: str | None = None
    """The sub-agent `agent()` runs when the script passes no `name`. See `DynamicWorkflow.default_agent`."""

    max_concurrent_agents: int = 16
    """Most sub-agents running at once in one `run_workflow` call, nested workflows included."""

    max_workflow_depth: int = 3
    """Most saved workflows running inside one another. See `DynamicWorkflow.max_workflow_depth`."""

    max_items_per_call: int = 4096
    """Most items one `parallel()` or `pipeline()` call accepts."""

    library: WorkflowLibrary | None = None
    """Saved workflows scripts can run by name; `None` turns saved workflows off.

    `save_workflow` adds to it, so a workflow saved in a run can be run later in that run.
    """

    save_directory: str | Path | None = None
    """Where `save_workflow` writes, in the run's workspace; `None` leaves the tool out.

    `DynamicWorkflow` sets this to its first `workflows` directory.

    The tool is also left out of a run whose workspace is missing or read-only, and when `library` is `None`.
    """

    # Per-run count of sub-agent calls; replaced on `for_run`.
    _budget: CallBudget = field(init=False, repr=False)

    # Sub-agents indexed by resolved sandbox name; seeded from `agents` in `__post_init__` and
    # extended in place as runtime appends to `agents` are revealed (`_fold_reveals`).
    _by_name: dict[str, WorkflowAgent[AgentDepsT]] = field(init=False, repr=False)

    # Tool description, frozen at run start. Rendered from the agents present when the run began
    # and never re-rendered after a reveal, so the description -- and thus the prompt-cache
    # prefix -- never changes mid-run.
    _description: str = field(init=False, repr=False)

    # Whether this run offers `save_workflow`: decided from the run's workspace in `for_run`.
    _can_save: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        for option in ('max_agent_calls', 'max_concurrent_agents', 'max_workflow_depth', 'max_items_per_call'):
            if getattr(self, option) < 1:
                raise UserError(f'DynamicWorkflow `{option}` must be at least 1.')
        _resolve_resource_limits(self.resource_limits)  # validate keys now, not at the first tool call
        if self.save_directory is not None and self.tool_name == SAVE_TOOL_NAME:
            raise UserError(f"DynamicWorkflow `tool_name` cannot be {SAVE_TOOL_NAME!r}, the saving tool's name.")
        self._budget = CallBudget(self.max_agent_calls)
        self._rebuild()
        if self.default_agent is not None and self.default_agent not in self._by_name:
            raise UserError(
                f'DynamicWorkflow `default_agent` {self.default_agent!r} is not a sub-agent; '
                f'choose one of: {", ".join(self._by_name)}.'
            )

    @property
    def _call_count(self) -> int:
        """Sub-agent calls made so far in this run."""
        return self._budget.used

    @property
    def _saved_workflows_enabled(self) -> bool:
        """Whether scripts can use saved workflows: `run_workflow(name=...)`, `workflow()` and `args`."""
        return self.library is not None

    def _omitted_helpers(self) -> set[str]:
        """Helpers left out of the scripts: shadowed by a sub-agent's name, or with nothing to do."""
        omitted = set(self._by_name) & HELPER_NAMES
        if not self._saved_workflows_enabled:
            omitted |= {'workflow', ARGS_NAME}
        return omitted

    def _rebuild(self) -> None:
        """Rebuild the name index and the frozen tool description from the current `agents`.

        Unusable entries raise `UserError`. `DynamicWorkflow.__post_init__` and
        `DynamicWorkflow.reveal()` validate entries eagerly, so this fails only when the
        lower-level toolset list was mutated outside those APIs.
        """
        self._by_name = index_workflow_agents(self.agents)
        self._description = render_description(
            self._by_name,
            max_agent_calls=self.max_agent_calls,
            max_concurrent_agents=self.max_concurrent_agents,
            max_items_per_call=self.max_items_per_call,
            default_agent=self.default_agent,
            omitted_helpers=self._omitted_helpers(),
        )

    @property
    def id(self) -> str | None:
        return self.toolset_id or self.tool_name

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Self:
        """Fresh instance per run so the sub-agent-call budget is per-run.

        Clone shallowly, keeping `agents` shared so reveals stay visible to this run,
        reset the per-run call budget, and rebuild the per-run index. Entries are
        validated eagerly by `DynamicWorkflow.__post_init__` and `DynamicWorkflow.reveal()`,
        so a strict rebuild here cannot fail unless this list was mutated outside those
        APIs. That is unsupported and fails fast.
        """
        clone = copy.copy(self)
        clone._budget = CallBudget(self.max_agent_calls)
        workspace = ctx.workspace
        clone._can_save = (
            self.library is not None
            and self.save_directory is not None
            and workspace.attached
            and not workspace.read_only
        )
        clone._rebuild()
        return clone

    def _fold_reveals(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fold sub-agents appended to `agents` since the run started into the name index.

        Diffs the live `agents` list against the names already known (`_by_name`, which holds the
        baseline plus anything revealed so far) and folds each newcomer in -- so `dispatch` resolves
        it -- enqueuing an announcement for the model. The frozen `_description` is untouched, so the
        cached prompt prefix is unaffected. Re-seeing an already-known entry (a baseline agent, or
        one revealed on an earlier step) is a no-op, identified by object identity so it is never
        mistaken for a name collision.

        A newcomer whose name is missing, invalid, or already taken by a different
        sub-agent is a contract violation and raises `UserError`.

        Synchronous and await-free between snapshotting `agents` and mutating `_by_name`, so a
        concurrently-running `dispatch` never observes a half-revealed agent -- the same
        await-free-critical-section reasoning that keeps `max_agent_calls` exact under fan-out.
        """
        for entry in tuple(self.agents):
            name = entry.resolved_name
            existing = self._by_name.get(name) if name else None
            if existing is entry:
                continue
            name = validate_workflow_agent(entry, set(self._by_name))
            self._by_name[name] = entry
            try:
                ctx.enqueue(render_reveal(name, self._by_name, self.tool_name))
            except UserError as exc:
                warnings.warn(
                    f'DynamicWorkflow revealed sub-agent {name!r}, but could not enqueue its announcement: '
                    f'{exc}. It is callable in this run, but the model will not see the reveal message.',
                    stacklevel=2,
                )

    def _visible_to_model(self, ctx: RunContext[AgentDepsT]) -> bool:
        """Whether this toolset's capability is visible to the model on this step.

        A deferred capability's toolset still gets `get_tools` calls while unloaded (its tool
        is indexed for `load_capability`/`search_tools`), but announcing a reveal then would
        leak the sub-agent signature into the conversation while the catalog is still hidden.

        The owning capability is resolved against the run's registry by identity, unwrapping
        `WrapperCapability` chains -- a wrapper over a leaf capability is registered in place of
        the leaf, and its `defer_loading`/registry id are what core keys loading on. Matching by
        id instead would misfire both ways: an id-less capability is registered under a
        run-generated key (so a lookup by tool name can hit an unrelated capability), and a
        deferred wrapper hides a non-deferred inner capability. No registry match (a toolset used
        directly, outside any capability) means always visible.
        """
        owner = self.owning_capability
        if owner is None:
            return True
        for capability_id, registered in ctx.capabilities.items():
            candidate: AbstractCapability[AgentDepsT] | None = registered
            while candidate is not None:
                if candidate is owner:
                    if registered.defer_loading is not True:
                        return True
                    return capability_id in ctx.loaded_capability_ids
                candidate = candidate.wrapped if isinstance(candidate, WrapperCapability) else None
        return True

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        # Skip reveal processing while the capability is deferred and unloaded: the newcomers
        # stay pending in `agents` and are folded in and announced on the first step after the
        # model loads the capability.
        if self._visible_to_model(ctx):
            self._fold_reveals(ctx)
        schema, validator = _NAMED_WORKFLOW_ARGS_SCHEMA if self._saved_workflows_enabled else _WORKFLOW_ARGS_SCHEMA
        tools = {
            self.tool_name: ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=self.tool_name,
                    description=self._description,
                    parameters_json_schema=schema,
                    metadata={'code_arg_name': 'code', 'code_arg_language': 'python'},
                    sequential=True,
                ),
                max_retries=self.max_retries,
                args_validator=validator,
            )
        }
        if self._can_save:
            schema, validator = _SAVE_WORKFLOW_ARGS_SCHEMA
            tools[SAVE_TOOL_NAME] = ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=SAVE_TOOL_NAME,
                    description=_SAVE_DESCRIPTION,
                    parameters_json_schema=schema,
                    metadata={'code_arg_name': 'code', 'code_arg_language': 'python'},
                    # Serialized, so two saves of one name in a step cannot both pass the collision check.
                    sequential=True,
                ),
                max_retries=self.max_retries,
                args_validator=validator,
            )
        return tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == SAVE_TOOL_NAME:
            return await self._save_workflow(tool_args, ctx)
        if in_workflow.get():
            return {'error': _NESTED_REFUSAL}

        code: str | None = tool_args.get('code')
        workflow_name: str | None = tool_args.get('name')
        args: dict[str, JsonValue] = tool_args.get('args') or {}
        if (code is None) == (workflow_name is None):
            raise ModelRetry('Pass exactly one of `code`, a script to run, or `name`, a saved workflow to run.')

        omitted = self._omitted_helpers()
        run = WorkflowRun(
            ctx=ctx,
            agents=self._by_name,
            library=self.library,
            budget=self._budget,
            default_agent=self.default_agent,
            max_concurrent_agents=self.max_concurrent_agents,
            max_workflow_depth=self.max_workflow_depth,
            forward_usage=self.forward_usage,
            inherit_model=self.inherit_model,
            sub_agent_usage_limits=self.sub_agent_usage_limits,
            prelude=render_prelude(shadowed=omitted, max_items=self.max_items_per_call),
            stubs=build_type_check_stubs(self._by_name, omitted_helpers=omitted),
            bind_args=ARGS_NAME not in omitted,
            limits=_resolve_resource_limits(self.resource_limits),
            in_temporal=in_temporal_workflow(),
        )
        frame = Frame()
        if workflow_name is not None:
            try:
                saved, args, frame = run.enter_saved(workflow_name, args, frame)
            except ValueError as error:
                raise ModelRetry(f'Cannot run saved workflow: {error}') from error
            code = saved.source
        assert code is not None

        capture = run.capture
        in_workflow_token = in_workflow.set(True)
        try:
            with run.workflow_span(frame, args) if frame.stack else nullcontext():
                session = await run.main_session()
                completed = await run.execute(session, code, frame=frame, args=args)
        except MontyTypingError as e:
            raise ModelRetry(f'Type error in workflow:\n{capture.prepend_to(e.display())}') from e
        except MontySyntaxError as e:  # pragma: no cover -- backstop; the type checker parses first
            raise ModelRetry(f'Syntax error in workflow:\n{capture.prepend_to(e.display())}') from e
        except MontyRuntimeError as e:
            if run.budget_exhausted:
                # The script may catch the budget error and fail later on something else, so
                # the flag -- not the displayed error -- proves this script hit the budget;
                # under gather, the displayed error may also be an independently surfaced
                # failure from the same batch.
                return budget_terminal_result(
                    max_agent_calls=self.max_agent_calls,
                    last_error=capture.prepend_to(e.display()),
                    completed_dispatches=run.completed,
                )
            raise ModelRetry(
                f'Runtime error in workflow:\n{capture.prepend_to(e.display())}{completed_retry_section(run.completed)}'
            ) from e
        except MontyCrashedError as e:
            # The worker died mid-script (e.g. resource exhaustion or request timeout);
            # the pool replaces it transparently. Completed sub-agent results are listed
            # so the retry can reuse them as plain values.
            return worker_crash_result(
                crash=e,
                budget_exhausted=run.budget_exhausted,
                max_agent_calls=self.max_agent_calls,
                completed_dispatches=run.completed,
            )
        except BaseException as e:
            # Convert a sandbox panic to a retry (see `is_sandbox_panic`);
            # anything else (CancelledError, ...) re-raises unchanged.
            if not is_sandbox_panic(e):
                raise
            if run.budget_exhausted:
                return budget_terminal_result(
                    max_agent_calls=self.max_agent_calls,
                    last_error='The workflow script aborted inside the sandbox after exhausting the sub-agent budget.',
                    completed_dispatches=run.completed,
                )
            raise ModelRetry(
                'The workflow script aborted inside the sandbox. Revise the script and try again.'
                f'{completed_retry_section(run.completed)}'
            ) from e
        finally:
            in_workflow.reset(in_workflow_token)
            await run.monty.close()

        # Monty lets workflow code catch host exceptions. Exhausting the budget remains
        # terminal even if the script catches that error and otherwise finishes normally.
        return completed_workflow_result(
            completed_output=completed.output,
            printed=capture.joined,
            budget_exhausted=run.budget_exhausted,
            max_agent_calls=self.max_agent_calls,
            completed_dispatches=run.completed,
        )

    async def _save_workflow(self, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT]) -> str:
        """Write a workflow to the library directory, refusing to replace one unless asked to."""
        assert self.library is not None and self.save_directory is not None
        agents: list[str] = tool_args.get('agents', [])
        if missing := [agent for agent in agents if agent not in self._by_name]:
            raise ModelRetry(
                f'Cannot save workflow: unknown sub-agents {", ".join(missing)}; available: {", ".join(self._by_name)}.'
            )
        try:
            workflow = SavedWorkflow.create(
                name=tool_args['name'],
                description=tool_args['description'],
                code=tool_args['code'],
                when_to_use=tool_args.get('when_to_use'),
                args=tool_args.get('args'),
                agents=agents,
                returns=tool_args.get('returns'),
            )
        except ValueError as error:
            raise ModelRetry(f'Cannot save workflow: {error}') from error
        workspace = ctx.workspace
        path = posixpath.join(await workspace.resolve(workspace_path(Path(self.save_directory))), f'{workflow.name}.py')
        try:
            if not tool_args.get('overwrite', False) and await workspace.exists(path):
                raise ModelRetry(
                    f'A workflow named {workflow.name!r} is already saved at {path}; '
                    'pass `overwrite: true` to replace it.'
                )
            await workspace.write_text(path, workflow.source)
        # `WorkspaceError` first: some are also `OSError`s, and `raise_tool_failure` knows which end the run.
        except WorkspaceError as error:
            raise_tool_failure(error)
        except OSError as error:
            raise ToolFailed(f'Cannot save workflow: {error}') from error
        self.library.workflows[workflow.name] = replace(workflow, path=path)
        return (
            f'Saved workflow {workflow.name!r} to {path}. Run it with `{self.tool_name}` by name, '
            f'or from a script with `await workflow({workflow.name!r}, args)`.'
        )
