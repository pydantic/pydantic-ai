from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import TypeAdapter
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


_ChildWorkflowHandler = Callable[[CallToolParams, Any], Awaitable[CallToolResult]]
_child_workflow_handlers: dict[str, _ChildWorkflowHandler] = {}
"""Per-toolset child-workflow handlers, keyed by `CallToolParams.handler_key`.

Populated by `temporalize_function_toolset` at capability-bind time — always before any workflow
polls, since `TemporalDurability` must bind before the worker starts (`_check_bindable`). Module-level
because `_ToolCallWorkflow` has to be one shared class (see its docstring for why), so it can't close
over a specific toolset the way each toolset's own `call_tool_activity` closure does.
"""


@workflow.defn(name='pydantic_ai__tool_call_workflow')
class _ToolCallWorkflow:
    """Generic child workflow for any `child_workflow`-tagged tool call, across every toolset.

    One shared, module-level class rather than one built per toolset: `@workflow.run` explicitly
    rejects locally-scoped classes ("Local classes unsupported... we need to have the class globally
    referenceable by name"), so a class built inside `temporalize_function_toolset` can't be decorated
    at all. Dispatch to the right toolset happens via `params.handler_key` into `_child_workflow_handlers`
    instead of via which class was invoked.
    """

    @workflow.run
    async def run(self, params: CallToolParams, deps: Any) -> CallToolResult:
        assert params.handler_key is not None
        handler = _child_workflow_handlers.get(params.handler_key)
        if handler is None:  # pragma: no cover — only reachable if a worker is missing the toolset's capability
            raise ApplicationError(
                f'No child-workflow handler registered for {params.handler_key!r}. This worker may be '
                'missing the `TemporalDurability` capability (or a differently-configured one) for the '
                'toolset that scheduled this child workflow.',
                type=UserError.__name__,
            )
        return await handler(params, deps)


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
    async def call_tool_activity(params: CallToolParams, deps: AgentDepsT) -> CallToolResult:
        async with heartbeating():
            ctx = deserialize_run_context(run_context_type, params.serialized_run_context, deps=deps, agent=agent)
            tool = await _tool_for_call(toolset, params, ctx)
            return await call_tool_in_activity(toolset, params.name, params.tool_args, ctx, tool)

    call_tool_activity.__annotations__['deps'] = deps_type
    registered_activity = activity.defn(name=f'{activity_name_prefix}__toolset__{toolset.id}__call_tool')(
        call_tool_activity
    )

    # `_ToolCallWorkflow` is one shared class for the whole process (see its docstring for why), so
    # this toolset's own handler is registered into `_child_workflow_handlers` under a key unique to
    # it, rather than built as its own workflow class the way `call_tool_activity` is its own activity.
    child_workflow_handler_key = f'{activity_name_prefix}__toolset__{toolset.id}__call_tool_workflow'
    deps_type_adapter: TypeAdapter[AgentDepsT] | None = None

    async def call_tool_child_workflow(params: CallToolParams, raw_deps: Any) -> CallToolResult:
        # No `workflow.execute_activity` anywhere in here: the tool's own function body runs
        # directly as workflow code, in-process, right now — it is NOT converted into an
        # activity. That's the entire point of `child_workflow`. If the tool body does
        # `await nested_agent.run(...)` and `nested_agent` carries its own `TemporalDurability`,
        # that nested run's *own* model/tool calls become ordinary Temporal activities scoped to
        # *this* child workflow's history, because `workflow.in_workflow()` is true in here.
        # A tool body that performs raw I/O directly (not through its own durable agent run)
        # would violate the workflow sandbox, the same constraint `metadata={'temporal': False}`
        # already imposes today.
        nonlocal deps_type_adapter
        if deps_type_adapter is None:
            # Built lazily, on first actual child-workflow call, not at bind time: `deps` can't be
            # typed to `deps_type` via a wire annotation the way `call_tool_activity`'s is, since
            # `_ToolCallWorkflow.run`'s signature is shared by every toolset in the process —
            # Temporal's converter deserializes it generically instead, so this re-validates, the
            # same idiom `TemporalRunContext` already uses for `usage`/`usage_limits`. Building it
            # eagerly for every toolset (whether or not any tool uses `child_workflow`) would break
            # `deps_type`s that Pydantic can't build a schema for but that never actually need to
            # cross this boundary.
            deps_type_adapter = TypeAdapter(deps_type)
        deps = deps_type_adapter.validate_python(raw_deps)
        ctx = deserialize_run_context(
            run_context_type, params.serialized_run_context, deps=deps, agent=agent, unit_noun='child workflow'
        )
        try:
            tool = await _tool_for_call(toolset, params, ctx)
            args = tool.args_validator.validate_python(params.tool_args)
            return await wrap_tool_call_result(toolset.call_tool(params.name, args, ctx, tool))
        except FailureError:
            # Already a proper Temporal failure (e.g. an `ActivityError`/`ChildWorkflowError` bubbling
            # up from a nested durable agent run inside this tool) — propagate as-is so this workflow
            # execution fails cleanly.
            raise
        except BaseException as exc:
            # An exception that escapes workflow code without being a Temporal failure (and isn't
            # covered by the worker's `workflow_failure_exception_types`) fails the workflow *task*,
            # which Temporal retries forever — a bug in the tool's own code would silently wedge the
            # child workflow instead of failing it. Convert to an `ApplicationError` (a `FailureError`)
            # so the child workflow execution fails cleanly and promptly instead.
            raise ApplicationError(str(exc), type=type(exc).__name__) from exc

    _child_workflow_handlers[child_workflow_handler_key] = call_tool_child_workflow

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
            # Unused by the activity path; only `_ToolCallWorkflow.run` reads it, to dispatch to
            # this toolset's own handler (see `_child_workflow_handlers`'s module-level docstring).
            handler_key=child_workflow_handler_key,
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
                        **(child_workflow_config or {}),
                        **cast('ChildWorkflowConfig', per_tool_config),
                    }
                    result = await execute_child_workflow(
                        _ToolCallWorkflow.run,
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
        # `_ToolCallWorkflow` is the same shared class regardless of which toolset registers it —
        # harmless to list unconditionally (like `registered_activity` above, whether or not any
        # tool actually uses `child_workflow`); the collection point in `TemporalDurability`
        # deduplicates before handing the list to the worker.
        durable_container_registrations=[_ToolCallWorkflow],
        durable_config=activity_config,
    )


TemporalFunctionToolset = DurableFunctionToolset
