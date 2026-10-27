from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from pydantic_ai._deferred_capabilities import (
    CAPABILITY_LOAD_KIND,
    LoadCapabilityArgs,
    LoadCapabilityCallPart,
    LoadCapabilityReturn,
)
from pydantic_ai._enqueue import PendingMessageQueue
from pydantic_ai._instructions import resolve_sourced_instructions
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    CapabilityInstructionSource,
    InstructionId,
    InstructionPart,
    ModelResponse,
    ToolAvailabilityDeltaPart,
    ToolReturn,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.toolsets._instruction_collection import (
    collect_toolset_instructions,
)
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import AbstractCapability

LOAD_CAPABILITY_TOOL_NAME = 'load_capability'
LOAD_CAPABILITY_TOOL_DESCRIPTION = (
    'Load a listed capability whenever it is plausibly relevant to the task.'
    ' Loading makes the capability instructions and any tools it provides available.'
)
LOAD_CAPABILITY_ALREADY_ACTIVE_MESSAGE_TEMPLATE = (
    'Capability {capability_id!r} is already active. '
    'Use its existing instructions and any tools it provides; do not call `load_capability` for it again.'
)
# Resolving the instructions here as well would run dynamic instructions twice for one request, so the
# message doesn't depend on whether there are any.
LOAD_CAPABILITY_LOADED_MESSAGE_TEMPLATE = 'Capability {capability_id!r} loaded. Its instructions, if any, follow.'
LOAD_CAPABILITY_DUPLICATE_CALL_MESSAGE_TEMPLATE = (
    'Capability {capability_id!r} is already being loaded by an earlier `load_capability` call in this response; '
    'do not call `load_capability` for it again.'
)

_load_capability_args_ta = TypeAdapter(LoadCapabilityArgs)
_LOAD_CAPABILITY_SCHEMA = _load_capability_args_ta.json_schema()
_LOAD_CAPABILITY_SCHEMA['title'] = 'LoadCapabilityArgs'


@dataclass
class DeferredCapabilityLoaderToolset(WrapperToolset[AgentDepsT]):
    """Adds the framework-managed `load_capability` tool."""

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        all_tools = await self.wrapped.get_tools(ctx)

        if LOAD_CAPABILITY_TOOL_NAME in all_tools:
            raise UserError(
                f"Tool name '{LOAD_CAPABILITY_TOOL_NAME}' is reserved for deferred capability loading. "
                'Rename your tool to avoid conflicts.'
            )

        load_tool_def = ToolDefinition(
            name=LOAD_CAPABILITY_TOOL_NAME,
            description=LOAD_CAPABILITY_TOOL_DESCRIPTION,
            parameters_json_schema=_LOAD_CAPABILITY_SCHEMA,
            tool_kind=CAPABILITY_LOAD_KIND,
        )

        load_tool = ToolsetTool(
            toolset=self,
            tool_def=load_tool_def,
            max_retries=ctx.max_retries,
            args_validator=_load_capability_args_ta.validator,  # pyright: ignore[reportArgumentType]
        )

        result: dict[str, ToolsetTool[AgentDepsT]] = {LOAD_CAPABILITY_TOOL_NAME: load_tool}
        result.update(all_tools)
        return result

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if tool.tool_def.tool_kind == 'capability-load':
            return await self._load_capability(tool_args, ctx)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def _load_capability(
        self, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT]
    ) -> ToolReturn[LoadCapabilityReturn]:
        capability_id = tool_args['id']
        capability = ctx.capabilities.get(capability_id)
        if capability is None:
            raise ModelRetry(f'No capability found with id {capability_id!r}.')
        if capability_id in ctx.active_capability_ids:
            raise ModelRetry(LOAD_CAPABILITY_ALREADY_ACTIVE_MESSAGE_TEMPLATE.format(capability_id=capability_id))
        if capability.defer_loading is not True:
            # Inactive without being deferred itself, so it sits inside a deferred capability that isn't loaded yet.
            raise ModelRetry(_part_of_unloaded_group_message(ctx, capability_id))
        if _is_duplicate_load_in_response(ctx, capability_id):
            raise ModelRetry(LOAD_CAPABILITY_DUPLICATE_CALL_MESSAGE_TEMPLATE.format(capability_id=capability_id))

        # The return only confirms the load: the next request records the capability's instructions as an
        # `InstructionDeltaPart` for `capability:<id>`, which models receive in the system voice, the voice
        # they treat as authoritative over user and tool-output pressure. They're never in both places, which
        # made models treat them as tool output again. A realtime session has no channel for those parts, so
        # the instructions still ride the return there.
        message = LOAD_CAPABILITY_LOADED_MESSAGE_TEMPLATE.format(capability_id=capability_id)
        result: LoadCapabilityReturn = {'message': message}
        if ctx.realtime:
            parts = await resolve_loaded_capability_instructions(capability_id, self, ctx)
            if (text := InstructionPart.join([part for part in parts if part.content])) is not None:
                result['instructions'] = text
        tools = sorted(name for name, tool_def in ctx.tools.items() if tool_def.capability_id == capability_id)
        return ToolReturn(return_value=result, tools=tools or None)


async def resolve_loaded_capability_instructions(
    capability_id: str, toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> list[InstructionPart]:
    """Resolve the instruction parts a loaded capability delivers, each addressed beneath its block `capability:<id>`.

    That's the capability's own instructions, sourced through `_collect_instructions` as they would be if it
    were eager, followed by those of the toolsets it owns in `toolset`. A loaded capability is delivered as
    one block named after its registry id, which a capability without an `id` of its own also has, so every
    part is re-keyed beneath that block, keeping its `name`. The `load_capability` return and every later
    request resolve the same parts, so the block only changes when the instructions do.
    """
    capability = ctx.capabilities[capability_id]
    sourced = capability._collect_instructions()  # pyright: ignore[reportPrivateUsage]
    parts = await resolve_sourced_instructions(sourced, ctx)

    owned: list[CapabilityOwnedToolset[AgentDepsT]] = []

    def collect(ts: AbstractToolset[AgentDepsT]) -> None:
        if isinstance(ts, CapabilityOwnedToolset) and ts.capability is capability:
            owned.append(ts)

    toolset.apply(collect)
    for ts in owned:
        parts.extend(await collect_toolset_instructions(ts.wrapped, ctx))

    source = CapabilityInstructionSource(capability_id)
    return [replace(part, id=InstructionId(source, name=part.name)) for part in parts]


async def load_capability(capability: str | AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]) -> bool:
    """Load a deferred capability from code; see [`RunContext.load_capability`][pydantic_ai.tools.RunContext.load_capability]."""
    # Only while a tool call is handled, where the run normally continues with another model request
    # that the load can ride along with. After a model response with no tool calls, it would have to force
    # an extra model turn; loading before a run's first model request is tracked in #10089.
    if ctx.tool_call_id is None:
        raise UserError(
            '`ctx.load_capability()` can currently only be called while a tool call is being handled: '
            'from a tool function or a tool hook such as `after_tool_execute`. Loading before the first '
            'model request of a run is tracked in https://github.com/pydantic/pydantic-ai/issues/10089.'
        )
    tool_manager = ctx.tool_manager
    pending_messages = ctx.pending_messages
    # Durable engines swap the run's queue for a guard inside an activity, step, or task. Checked
    # first: a Temporal activity's context raises on most other fields, including `ctx.realtime`'s.
    if tool_manager is None or not isinstance(pending_messages, PendingMessageQueue):
        raise UserError(
            '`ctx.load_capability()` is not supported inside a durable execution activity, step, or task, '
            'whose recorded result is replayed without re-running your code. '
            'Load the capability from a tool hook such as `after_tool_execute` instead.'
        )
    if ctx.realtime:
        raise UserError('`ctx.load_capability()` is not supported in a realtime session.')
    tool_def = tool_manager.get_tool_def(ctx.tool_name) if ctx.tool_name is not None else None
    if tool_def is not None and tool_def.kind == 'output':
        raise UserError('`ctx.load_capability()` cannot be called from an output tool, which ends the run.')

    if isinstance(capability, str):
        capability_id = capability
    elif capability.id in ctx.capabilities:
        # By `id` rather than identity, so a per-run copy made by `for_run` still resolves.
        capability_id = capability.id
    else:
        # A capability without an explicit `id` is registered under a derived one.
        capability_id = next((key for key, cap in ctx.capabilities.items() if cap is capability), None)
        if capability_id is None:
            raise UserError(
                f'Capability {type(capability).__name__}(id={capability.id!r}) is not registered in this run.'
            )
    if capability_id not in ctx.capabilities:
        raise UserError(f'No capability with id {capability_id!r} is registered in this run.')

    if _is_active_or_loading(ctx, pending_messages, capability_id):
        return False
    if ctx.capabilities[capability_id].defer_loading is not True:
        # Inactive without being deferred itself, so it sits inside a deferred capability that isn't loaded yet.
        raise UserError(_part_of_unloaded_group_message(ctx, capability_id))

    # Recorded without a `load_capability` call, which the request may have no tool for: the same
    # `ToolAvailabilityDeltaPart` a model's load leaves after its return, naming the capability and revealing
    # its tools. The next request then states the capability's instructions, as after a model's load.
    tools = sorted(name for name, tool_def in ctx.tools.items() if tool_def.capability_id == capability_id)
    record = ToolAvailabilityDeltaPart(
        tools_added=tools, capabilities_added=[capability_id], tool_call_id=ctx.tool_call_id
    )
    # Delivered with the next model request; if this step ends the run instead (a tool call needs
    # approval or is deferred, or output is produced), the load is dropped rather than forcing one.
    pending_messages.append_capability_load(record)
    return True


def _part_of_unloaded_group_message(ctx: RunContext[Any], capability_id: str) -> str:
    """Say which unloaded deferred capability the inactive, non-deferred `capability_id` loads with."""
    capability = ctx.capabilities[capability_id]
    groups = [
        group_id
        for group_id, group in ctx.capabilities.items()
        if group.defer_loading is True
        and group_id not in ctx.loaded_capability_ids
        and any(inner is capability for inner in _beneath(group))
    ]
    names = ' or '.join(repr(group_id) for group_id in groups)
    return (
        f'Capability {capability_id!r} is part of deferred capability {names} and loads with it. Load {names} instead.'
    )


def _is_active_or_loading(ctx: RunContext[Any], queue: PendingMessageQueue, capability_id: str) -> bool:
    """Whether `capability_id` is active, or a load of it is queued or among the model's calls being executed."""
    if capability_id in ctx.active_capability_ids or any(
        capability_id in record.capabilities_added for record in queue.capability_loads
    ):
        return True
    response = next((message for message in reversed(ctx.messages) if isinstance(message, ModelResponse)), None)
    return response is not None and any(
        isinstance(part, LoadCapabilityCallPart) and part.capability_id == capability_id for part in response.parts
    )


def _is_duplicate_load_in_response(ctx: RunContext[Any], capability_id: str) -> bool:
    """Whether an earlier call in the response being executed already loads `capability_id`.

    A load only counts as loaded once its return reaches history at the end of the step, so sibling
    calls for the same id in one response can't see each other through `active_capability_ids`.
    Ownership is instead derived from the response itself — the last one in history, whose calls are
    the ones executing — in model call order, like `_prune_duplicate_tool_reveals` gives the first
    call to name a tool its reveal, so which call delivers the instructions doesn't depend on task
    scheduling.
    """
    response = next((message for message in reversed(ctx.messages) if isinstance(message, ModelResponse)), None)
    parts = response.parts if response is not None else ()
    owner_tool_call_id = next(
        (
            part.tool_call_id
            for part in parts
            if isinstance(part, LoadCapabilityCallPart) and part.capability_id == capability_id
        ),
        ctx.tool_call_id,
    )
    return owner_tool_call_id != ctx.tool_call_id


def _beneath(capability: AbstractCapability[Any]) -> list[AbstractCapability[Any]]:
    """What `capability` registers beneath itself: `apply` visits the capability first."""
    visited: list[AbstractCapability[Any]] = []
    capability.apply(visited.append)
    return visited[1:]
