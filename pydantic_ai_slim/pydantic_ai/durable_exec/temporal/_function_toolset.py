from __future__ import annotations

import asyncio
import itertools
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from temporalio import activity, workflow
from temporalio.exceptions import ApplicationError, FailureError
from temporalio.workflow import ActivityConfig, ChildWorkflowConfig

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DurableFunctionToolset,
    ToolConfig,
    unwrap_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._activity_execution import execute_activity, execute_child_workflow
from ._run_context import TemporalRunContext, deserialize_run_context
from ._toolset import (
    CallToolParams,
    call_tool_in_activity,
    heartbeating,
    resolve_tool_temporal_wrapping,
    tool_result_payload_errors,
)

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent


# Per-process-unique class attribute names so two bindings with colliding workflow type names never
# overwrite each other's module attribute (the sandbox resolves the class by its `__name__`).
_tool_call_workflow_counter = itertools.count(1)


class _ChildWorkflowClass(Protocol):
    """Static shape of a dynamically-built per-toolset child-workflow class, for the `cast` below.

    `type(class_name, (), {...})`'s 3-argument form only gives pyright a bare `type[_]`, with no
    visibility into the `run` method assigned into its namespace dict — this is a minimal stand-in
    so `registered_workflow.run` type-checks without a broader `cast(Any, ...)`.
    """

    run: Callable[[CallToolParams, Any], Awaitable[CallToolResult]]


def _require_toolset_id(toolset: FunctionToolset[Any]) -> str:
    """Return `toolset.id`, or raise if it's unset.

    The ID names the toolset's activity and, for `child_workflow`-tagged tools, its own per-toolset
    child workflow -- both need a stable, non-empty string to build a valid Temporal name from.
    """
    if toolset.id is None:
        raise UserError(
            f'{toolset.label} needs to have a unique `id` in order to be used with Temporal. '
            "The ID is used to name the toolset's activity and, for `child_workflow`-tagged tools, "
            'its own per-toolset child workflow.'
        )
    return toolset.id


async def _tool_for_call(
    toolset: FunctionToolset[AgentDepsT], params: CallToolParams, ctx: RunContext[AgentDepsT]
) -> ToolsetTool[AgentDepsT]:
    """Resolve the tool a `CallToolParams` refers to — shared by the activity and child-workflow paths."""
    try:
        if params.tool_def is not None:
            # Rebuild the tool from the definition the workflow prepared, so a tool's `prepare`
            # function isn't run a second time here against the durable unit's limited run context.
            return toolset.tool_for_tool_def(params.tool_def, ctx=ctx, original_name=params.original_name)
        # Only reachable for a call scheduled by a worker predating `tool_def` on these params;
        # re-prepare so in-flight executions still complete across the upgrade.
        return (await toolset.get_tools(ctx))[params.name]
    except KeyError as exc:
        raise UserError(
            f'Tool {params.name!r} not found in toolset {toolset.id!r}. '
            'Removing or renaming tools during an agent run is not supported with Temporal.'
        ) from exc


def temporalize_function_toolset(
    toolset: FunctionToolset[AgentDepsT],
    *,
    activity_name_prefix: str,
    activity_config: ActivityConfig,
    tool_activity_config: dict[str, ActivityConfig | Literal[False]],
    deps_type: type[AgentDepsT],
    run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    agent: AbstractAgent[AgentDepsT, Any] | None = None,
    child_workflow_config: ChildWorkflowConfig | None = None,
) -> DurableFunctionToolset[AgentDepsT]:
    toolset_id = _require_toolset_id(toolset)

    async def call_tool_activity(params: CallToolParams, deps: AgentDepsT) -> CallToolResult:
        async with heartbeating():
            ctx = deserialize_run_context(run_context_type, params.serialized_run_context, deps=deps, agent=agent)
            tool = await _tool_for_call(toolset, params, ctx)
            return await call_tool_in_activity(toolset, params.name, params.tool_args, ctx, tool)

    call_tool_activity.__annotations__['deps'] = deps_type
    registered_activity = activity.defn(name=f'{activity_name_prefix}__toolset__{toolset_id}__call_tool')(
        call_tool_activity
    )

    # Build one `@workflow.defn` class per toolset. The workflow *type name* is deterministic so
    # histories replay across worker restarts; the Python class name carries a per-process-unique
    # suffix so two colliding bindings on separate workers cannot overwrite each other's module
    # attribute (the sandbox resolves the class by `__name__`) -- a real risk, not a hypothetical
    # one: `pydantic_ai` is a sandbox passthrough module, so every worker's sandboxed runner
    # re-resolves the class against the *same* shared module namespace, not an isolated copy.
    # Requiring `toolset_id` above doesn't remove this: two different agents can deliberately (or
    # accidentally) share both a name and a toolset id, and Temporal's own duplicate-workflow-name
    # check never fires across separate `Worker`s, only within one.
    workflow_name = f'{activity_name_prefix}__toolset__{toolset_id}__call_tool_workflow'
    sanitized_toolset_id = re.sub(r'\W', '_', toolset_id)
    class_name = f'_ToolCallWorkflow_{sanitized_toolset_id}_{next(_tool_call_workflow_counter)}'

    async def _run(self, params: CallToolParams, deps: AgentDepsT) -> CallToolResult:
        # This workflow type is registered on the worker like any other, so any Temporal client with
        # permission to start workflows on this task queue could start it directly, supplying
        # arbitrary `CallToolParams` — bypassing the parent agent run, its tool preparation, and any
        # approval logic entirely. Rejecting non-child executions here is a cheap first check, not a
        # full authorization boundary: a caller can just as easily start their own throwaway parent
        # workflow and issue the same `execute_child_workflow` call from inside it, since the target
        # workflow type and `CallToolParams` shape are both public. The per-tool `child_workflow`
        # check below narrows the blast radius further — closing off tools never tagged for
        # `child_workflow` — but does not fully close this either, for the same reason. A worker
        # whose task queue is reachable by untrusted clients needs that boundary enforced by Temporal
        # itself (namespace/task-queue access control), not by this check.
        if workflow.info().parent is None:
            raise ApplicationError(
                f'{workflow.info().workflow_type!r} must be started as a child workflow of an agent run, '
                'not invoked directly.',
                type=UserError.__name__,
            )
        # No `workflow.execute_activity` anywhere in here: the tool's own function body runs
        # directly as workflow code, in-process, right now — it is NOT converted into an
        # activity. That's the entire point of `child_workflow`. If the tool body does
        # `await nested_agent.run(...)` and `nested_agent` carries its own `TemporalDurability`,
        # that nested run's *own* model/tool calls become ordinary Temporal activities scoped to
        # *this* child workflow's history, because `workflow.in_workflow()` is true in here.
        # A tool body that performs raw I/O directly (not through its own durable agent run)
        # would violate the workflow sandbox, the same constraint `metadata={'temporal': False}`
        # already imposes today.
        try:
            # `deps` is deserialized by Temporal's converter from the `deps_type` annotation set
            # below (the same idiom used for `call_tool_activity`), not via a separate validator.
            ctx = deserialize_run_context(
                run_context_type, params.serialized_run_context, deps=deps, agent=agent, unit_noun='child workflow'
            )
            tool = await _tool_for_call(toolset, params, ctx)
            # Re-verify against the tool's own current config, rather than trusting that reaching this
            # handler at all implies the caller went through `resolve_tool_config`'s normal `activity`
            # vs. `child_workflow` dispatch — direct starts of this workflow type are possible, so that
            # normal path isn't the only way to reach here. Rejects a request for a tool that was never
            # tagged for `child_workflow`, even if the request otherwise looks well-formed.
            wrapping = resolve_tool_temporal_wrapping(tool, params.name, tool_activity_config)
            if not (isinstance(wrapping, Mapping) and 'child_workflow' in wrapping):
                raise UserError(
                    f'Tool {params.name!r} is not configured to run as a `child_workflow`; refusing to run it as one.'
                )
            args = tool.args_validator.validate_python(params.tool_args)
            return await wrap_tool_call_result(toolset.call_tool(params.name, args, ctx, tool))
        except FailureError:
            # Already a proper Temporal failure (e.g. an `ActivityError`/`ChildWorkflowError` bubbling
            # up from a nested durable agent run inside this tool) — propagate as-is so this workflow
            # execution fails cleanly.
            raise
        except asyncio.CancelledError:
            # Cancellation of this child workflow (or of a nested durable call inside the tool) surfaces
            # here as `asyncio.CancelledError`. It must keep propagating as-is, not get converted to an
            # `ApplicationError` below: Temporal recognizes a workflow execution as *cancelled* (rather
            # than *failed*) only if this exception reaches the top uncaught, which is also why
            # `execute_child_workflow` re-raises it rather than swallowing it.
            raise
        except BaseException as exc:
            # An exception that escapes workflow code without being a Temporal failure (and isn't
            # covered by the worker's `workflow_failure_exception_types`) fails the workflow *task*,
            # which Temporal retries forever — a bug in the tool's own code would silently wedge the
            # child workflow instead of failing it. Convert to an `ApplicationError` (a `FailureError`)
            # so the child workflow execution fails cleanly and promptly instead.
            raise ApplicationError(str(exc), type=type(exc).__name__) from exc

    # `@workflow.run` rejects methods whose `__qualname__` contains `<locals>`, and
    # `@workflow.defn` requires the method's qualname prefix to match the class name. Rewriting it
    # here satisfies both SDK checks for a dynamically-built class.
    _run.__qualname__ = f'{class_name}.run'
    _run.__annotations__['deps'] = deps_type
    registered_workflow = cast(
        '_ChildWorkflowClass',
        workflow.defn(name=workflow_name)(
            type(class_name, (), {'__qualname__': class_name, 'run': workflow.run(_run)})
        ),
    )
    # Sandboxed runners re-resolve the class via `from {__module__} import {__name__}`; make the
    # class available under its sanitized name. Safe because `pydantic_ai` is a passthrough module
    # under `PydanticAIPlugin`.
    globals()[class_name] = registered_workflow

    def resolve_tool_config(tool: ToolsetTool[Any] | None, name: str) -> ToolConfig:
        config = resolve_tool_temporal_wrapping(tool, name, tool_activity_config)
        match config:
            case False:
                assert isinstance(tool, FunctionToolsetTool)
                if not tool.is_async:
                    raise UserError(
                        f'Temporal activity config for tool {name!r} has been explicitly set to `False` (activity disabled), '
                        'but non-async tools are run in threads which are not supported outside of an activity. Make the tool function async instead.'
                    )
            case {'child_workflow': _}:
                assert isinstance(tool, FunctionToolsetTool)
                if not tool.is_async:
                    raise UserError(
                        f'Temporal metadata for tool {name!r} configures it to run as a child workflow, '
                        'but non-async tools are run in threads which are not supported inside a workflow. Make the tool function async instead.'
                    )
            case _:
                pass
        # The resolved wrapping is always a plain dict (or `False`) at runtime; the
        # `{'child_workflow': ...}` shape is a `Mapping[str, Any]` like any `ActivityConfig`.
        return cast('ToolConfig', config)

    async def call_tool_operation(
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        params = CallToolParams(
            name=name,
            tool_args=tool_args,
            serialized_run_context=run_context_type.serialize_run_context(ctx),
            tool_def=tool.tool_def,
            # A `prepare` function can expose a tool under a different name than the toolset
            # holds it under; the activity or child workflow needs the latter to find the function to call.
            original_name=tool.original_name if isinstance(tool, FunctionToolsetTool) else None,
        )
        # `tool_result_payload_errors` covers both branches below: an over-limit result crosses the
        # same encoding boundary whether the durable unit is an activity or a child workflow, just
        # reported back wrapped in `ActivityError` or `ChildWorkflowError` respectively (see its
        # docstring in `_toolset.py`).
        with tool_result_payload_errors(name):
            match config:
                case {'child_workflow': per_tool_config}:
                    # Both components are deterministic and replay-stable: the workflow ID is fixed for
                    # the run and `tool_call_id` was assigned by the (replayed) model response, so a
                    # replay of the parent derives the same child ID and reattaches instead of starting
                    # a duplicate. Always user-overridable since it's spread first in the merge.
                    default_id = f'{workflow.info().workflow_id}--{ctx.tool_call_id}'
                    # `child_workflow_config` is the capability-level base from `TemporalDurability`;
                    # per-tool metadata always wins, same merge convention as the activity branch.
                    merged: ChildWorkflowConfig = {
                        'id': default_id,
                        # Default label for the Temporal UI, mirroring the activity branch's `summary`
                        # default below; always overridable via `child_workflow_config` or per-tool metadata.
                        'static_summary': f'call tool: {toolset.id}:{name}',
                        **(child_workflow_config or {}),
                        **cast('ChildWorkflowConfig', per_tool_config),
                    }
                    result = await execute_child_workflow(
                        registered_workflow.run,
                        args=[params, ctx.deps],
                        result_type=CallToolResult,
                        **merged,
                    )
                case _:
                    merged_config = cast(
                        'ActivityConfig',
                        {
                            'summary': f'call tool: {toolset.id}:{name}',
                            **activity_config,
                            **config,
                        },
                    )
                    result = await execute_activity(
                        activity=registered_activity,
                        args=[params, ctx.deps],
                        **merged_config,
                    )
        return unwrap_tool_call_result(result)

    return DurableFunctionToolset(
        toolset,
        in_durable_context=workflow.in_workflow,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=resolve_tool_config,
        lifecycle='enter-outside-durable',
        durable_registrations=[registered_activity],
        # The class is built per toolset, but harmless to list unconditionally (like
        # `registered_activity` above, whether or not any tool actually uses `child_workflow`);
        # the collection point in `TemporalDurability` deduplicates before handing the list to the
        # worker.
        durable_container_registrations=[registered_workflow],
        durable_config=activity_config,
    )


TemporalFunctionToolset = DurableFunctionToolset
