from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from pydantic_ai import _utils
from pydantic_ai._deferred_capabilities import (
    LoadCapabilityArgs,
    LoadCapabilityCallPart,
    LoadCapabilityReturn,
    LoadCapabilityReturnPart,
)
from pydantic_ai._enqueue import PendingMessage, PendingMessageQueue
from pydantic_ai._instructions import resolve_sourced_instructions
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    InstructionPart,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ToolAvailabilityDeltaPart,
    ToolReturn,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.toolsets._instruction_collection import collect_toolset_instructions
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
            tool_kind='capability-load',
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
        if _is_duplicate_load_in_response(ctx, capability_id):
            raise ModelRetry(LOAD_CAPABILITY_DUPLICATE_CALL_MESSAGE_TEMPLATE.format(capability_id=capability_id))

        result, tools = await _build_capability_load(capability_id, capability, self, ctx)
        return ToolReturn(return_value=result, tools=tools or None)


async def load_capability(capability: str | AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]) -> bool:
    """Load a deferred capability from code; see [`RunContext.load_capability`][pydantic_ai.tools.RunContext.load_capability]."""
    # Only while a tool call is handled, where the run normally continues with another model request
    # that the load can ride along with; elsewhere it would have to force an extra model turn.
    if ctx.tool_call_id is None:
        raise UserError(
            '`ctx.load_capability()` can only be called while a tool call is being handled: '
            'from a tool function or a tool hook such as `after_tool_execute`.'
        )
    if ctx.realtime:
        raise UserError('`ctx.load_capability()` is not supported in a realtime session.')
    tool_manager = ctx.tool_manager
    pending_messages = ctx.pending_messages
    # Durable engines swap the run's queue for a guard inside an activity, step, or task.
    if tool_manager is None or not isinstance(pending_messages, PendingMessageQueue):
        raise UserError(
            '`ctx.load_capability()` is not supported inside a durable execution activity, step, or task, '
            'whose recorded result is replayed without re-running your code. '
            'Load the capability from a tool hook such as `after_tool_execute` instead.'
        )
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
    registered = ctx.capabilities.get(capability_id)
    if registered is None:
        raise UserError(f'No capability with id {capability_id!r} is registered in this run.')

    if _is_active_or_loading(ctx, pending_messages, capability_id):
        return False

    result, tools = await _build_capability_load(capability_id, registered, tool_manager.toolset, ctx)
    # Building awaits user code, during which a parallel tool may have queued the same load.
    if _is_load_pending(pending_messages, capability_id):
        return False

    # A capability counts as loaded because its `load_capability` exchange is in message history, so
    # recording the same exchange a model-initiated load produces is what makes this load take
    # effect from the next model request, and survive resumption and compaction.
    tool_call_id = _utils.generate_tool_call_id()
    request_parts: list[ModelRequestPart] = [LoadCapabilityReturnPart(content=result, tool_call_id=tool_call_id)]
    if tools:
        request_parts.append(ToolAvailabilityDeltaPart(tools_added=tools, tool_call_id=tool_call_id))
    pending = PendingMessage(
        messages=[
            ModelResponse(parts=[LoadCapabilityCallPart(args={'id': capability_id}, tool_call_id=tool_call_id)]),
            ModelRequest(parts=request_parts),
        ]
    )
    # Delivered with the next model request; if this step ends the run instead (a tool call needs
    # approval or is deferred, or output is produced), the load is dropped rather than forcing one.
    pending_messages.append_capability_load(pending)
    return True


def _is_active_or_loading(ctx: RunContext[Any], queue: PendingMessageQueue, capability_id: str) -> bool:
    """Whether `capability_id` is active, or a load of it is queued or among the model's calls being executed."""
    if capability_id in ctx.active_capability_ids or _is_load_pending(queue, capability_id):
        return True
    response = next((message for message in reversed(ctx.messages) if isinstance(message, ModelResponse)), None)
    return response is not None and any(
        isinstance(part, LoadCapabilityCallPart) and part.capability_id == capability_id for part in response.parts
    )


async def _build_capability_load(
    capability_id: str,
    capability: AbstractCapability[AgentDepsT],
    toolset: AbstractToolset[AgentDepsT],
    ctx: RunContext[AgentDepsT],
) -> tuple[LoadCapabilityReturn, list[str]]:
    """Build what loading a capability delivers: its instructions and the names of the tools it reveals.

    `toolset` is searched for the toolsets `capability` contributes, whose instructions are part of the load.
    """
    # Sourced through `_collect_instructions` rather than `get_instructions` so a loaded
    # capability's parts carry the same `capability:<id>` keys they would have had if the
    # capability were eager. `InstructionPart.join` below flattens the ids away today, because
    # a load delivers its instructions as tool-return text rather than as request parts — but
    # the identity is assigned in one place for both paths instead of two that can drift.
    parts = await resolve_sourced_instructions(
        capability._collect_instructions(),  # pyright: ignore[reportPrivateUsage]
        ctx,
    )
    parts.extend(await _collect_owned_toolset_instructions(capability, toolset, ctx))
    instructions_text = InstructionPart.join(parts)
    result: LoadCapabilityReturn = {'instructions': instructions_text} if instructions_text is not None else {}
    tools = sorted(name for name, tool_def in ctx.tools.items() if tool_def.capability_id == capability_id)
    return result, tools


async def _collect_owned_toolset_instructions(
    capability: AbstractCapability[AgentDepsT], toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> list[InstructionPart]:
    owned: list[CapabilityOwnedToolset[AgentDepsT]] = []

    def collect(ts: AbstractToolset[AgentDepsT]) -> None:
        if isinstance(ts, CapabilityOwnedToolset) and ts.capability is capability:
            owned.append(ts)

    toolset.apply(collect)

    parts: list[InstructionPart] = []
    for ts in owned:
        parts.extend(await collect_toolset_instructions(ts.wrapped, ctx))
    return parts


def _is_load_pending(queue: PendingMessageQueue, capability_id: str) -> bool:
    """Whether a load of `capability_id` was queued from code and hasn't reached message history yet."""
    return any(
        isinstance(part, LoadCapabilityCallPart) and part.capability_id == capability_id
        for pending in queue.capability_loads
        for message in pending.messages
        for part in message.parts
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
