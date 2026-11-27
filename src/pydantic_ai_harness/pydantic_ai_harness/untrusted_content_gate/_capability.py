"""Hold tool calls that act outside the run once the run has read content from outside sources."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, WrapToolExecuteHandler
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, SkipToolExecution, UserError
from pydantic_ai.messages import (
    CapabilityPart,
    CompactionPart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolReturnPart,
    ToolCallPart,
    ToolResultSource,
    ToolReturnPart,
)
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition, ToolDenied, ToolSelector, matches_tool_selector
from pydantic_ai_harness._judge import SerializableToolSelector
from pydantic_ai_harness._mcp import is_read_only

UntrustedContentAction = Literal['ask', 'block']
"""What happens to a gated call once the run has read untrusted content.

- `'ask'`: the call joins the run's deferred approval flow, with the gate's reason in its metadata.
- `'block'`: the call does not run, and the model is told why with a `ToolDenied` result.
"""

_KEY = 'untrusted_content_gate'

# `CodeMode`'s `run_code` and `ToolBridge`'s `tool_call` tool kinds, by id: importing them would make the gate need
# their extras. The calls they make are gated one by one, under the tools' own names. `tool_call` is never gated
# itself, since an approval of it would cover whichever tool the model named; `run_code` is gated only when
# `gated_tools` selects it, since a script can also write through Code Mode's `mount` or `os_access`.
_RUN_CODE_KIND_ID = 'code_mode.run_code'
_TOOL_CALL_KIND_ID = 'tool_bridge.tool_call'

# Tool results the framework wrote itself, rather than the tool, carry nothing from outside the run. A retry
# carries the tool's own text when the tool raised `ModelRetry`, such as an MCP server's error message.
_FRAMEWORK_OUTCOMES = frozenset({'denied', 'interrupted'})


class _Source(BaseModel):
    tool_name: str | None
    tool_call_id: str | None


class _Asked(BaseModel):
    reason: str
    tainted_by: list[_Source]


class _AskedMetadata(BaseModel):
    untrusted_content_gate: _Asked


@dataclass
class UntrustedContentGate(AbstractCapability[AgentDepsT]):
    """Hold calls to tools that act outside the run once the run has read content from outside sources.

    A run is tainted once its history holds a result written outside the run: a tool result (unless the tool
    is in `trusted_tools`), a provider-native tool result such as a web search, or a
    [`CapabilityPart`][pydantic_ai.messages.CapabilityPart] whose source has `'none'`
    [authority][pydantic_ai.messages.Authority], or the instructions of a capability lowered to `'none'`, or a
    compaction summary, which can carry what the parts it replaced said. From then on, a call to a gated tool
    asks a person first (`on_tainted='ask'`) or does not run (`'block'`). Calls to gated tools made before
    anything untrusted was read run normally.

    The gate does not try to detect prompt injection. It holds the calls an injection would need, whether or
    not a detector would have flagged the content, so it still holds when detection misses.

    A call is judged on the history before the model response that made it, plus any provider-native tool result
    in that response: a call made in the same response as a function tool's untrusted read was chosen before that
    read's result existed. A call made through [`ToolBridge`][pydantic_ai_harness.tool_bridge.ToolBridge]'s
    `tool_call` is judged the same way, under the bridged tool's name, and `tool_call` itself is not gated; its
    result is judged under that name too, when the bridged call ran in this run. A call no model response made,
    such as one a [`CodeMode`][pydantic_ai_harness.code_mode.CodeMode] script makes, is judged on the whole history
    and on every untrusted tool that has run in this run, including earlier calls in the same script. That record
    is kept in process, so a durable run that resumes mid-script rebuilds taint from history only.

    In `'ask'` mode the gate's reason goes in the approval request's metadata, under `'untrusted_content_gate'`
    (read it back with [`untrusted_content_reason`][pydantic_ai_harness.untrusted_content_gate.untrusted_content_reason]).
    When another capability asks about the same call, such as
    [`Permissions`][pydantic_ai_harness.permissions.Permissions] or a `requires_approval=True` tool, the gate
    adds its reason to that request instead of asking separately, so one answer covers both. An approval
    covers that one call; the run stays tainted and later gated calls ask again. A gated call to an external tool,
    which the application runs, reaches the application with the gate's reason in its metadata instead; `'block'`
    can't stop such a call, so it refuses a gated external tool before the run's first request.

    By default every tool not marked read-only is gated, where read-only means the tool's metadata carries
    `{'annotations': {'readOnlyHint': True}}`, as MCP servers publish it.
    """

    _: KW_ONLY

    on_tainted: UntrustedContentAction = 'ask'
    """Ask a person about gated calls in a tainted run, or block them."""

    gated_tools: ToolSelector[AgentDepsT] | None = None
    """Tools held in a tainted run. `None` (the default) holds every tool not marked read-only."""

    trusted_tools: ToolSelector[AgentDepsT] = ()
    """Tools whose results do not taint the run, such as a tool that reads the person's own files."""

    _ran: list[tuple[int, _Source]] = field(
        default_factory=list[tuple[int, _Source]], init=False, repr=False, compare=False
    )
    _trusted_by_name: dict[str, bool] = field(default_factory=dict[str, bool], init=False, repr=False, compare=False)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> UntrustedContentGate[AgentDepsT]:
        """A fresh instance per run, with its own record of the untrusted tools that ran."""
        return replace(self)

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Record an untrusted tool that ran, for calls no model response made and results compaction removed."""
        await self._record(ctx, call, tool_def)
        return result

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        """Refuse a gated external tool in `'block'` mode, which can't stop it: the application runs its calls."""
        if self.on_tainted == 'block':
            for tool_def in tool_defs:
                if tool_def.kind == 'external' and await self._gates(ctx, tool_def):
                    raise UserError(
                        f"`UntrustedContentGate(on_tainted='block')` can't stop calls to external tool "
                        f"{tool_def.name!r}, which your application runs. Use `on_tainted='ask'`, where the call "
                        "reaches your application with the gate's reason in its metadata, or leave the tool out of "
                        '`gated_tools`.'
                    )
        return tool_defs

    def get_ordering(self) -> CapabilityOrdering:
        """Run last among `before_tool_execute` hooks, so other capabilities' approval requests come first."""
        return CapabilityOrdering(position='innermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the stable name used in agent specs."""
        return 'UntrustedContentGate'

    @classmethod
    def from_spec(
        cls,
        *,
        on_tainted: UntrustedContentAction = 'ask',
        gated_tools: SerializableToolSelector | None = None,
        trusted_tools: SerializableToolSelector = (),
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> UntrustedContentGate[AgentDepsT]:
        """Build the serializable configuration, excluding predicate tool selectors."""
        return cls(
            on_tainted=on_tainted,
            gated_tools=gated_tools,
            trusted_tools=trusted_tools,
            id=id,
            description=description,
            defer_loading=defer_loading,
        )

    async def after_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Ask during validation about a call that may ask there before any execute hook runs.

        That is a `requires_approval=True` tool, or one with an `args_validator`, which can raise `ApprovalRequired`
        itself. The gate's request then replaces the tool's own, so an approval of it covers the gate too. An external
        tool's call reaches no execute hook, so the gate defers it here to the application, with its reason.
        """
        external = tool_def.kind == 'external'
        if ctx.tool_call_approved or not (external or _asks_at_validation(ctx, tool_def)):
            return args
        if self.on_tainted == 'block' and not external:
            return args
        asked = await self._decide(ctx, call=call, tool_def=tool_def)
        if asked is None:
            return args
        self._trace(ctx, call, asked)
        if external:
            # Core can't approve an external call inside the run, so the call goes to the application that runs it,
            # with the gate's reason for it to check (`prepare_tools` refuses this tool in `'block'` mode).
            raise CallDeferred(metadata=_with_reason(None, asked))
        raise ApprovalRequired(metadata=_with_reason(None, asked))

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Hold or block a gated call in a tainted run."""
        if ctx.tool_call_approved and self.on_tainted == 'ask':
            return args
        asked = await self._decide(ctx, call=call, tool_def=tool_def)
        if asked is None:
            return args
        self._trace(ctx, call, asked)
        if self.on_tainted == 'block':
            raise SkipToolExecution(ToolDenied(_block_message(call.tool_name)))
        raise ApprovalRequired(metadata=_with_reason(None, asked))

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Add the gate's reason to an approval request another capability raised for the same call.

        Also record an untrusted tool that failed or asked for a retry: its message can carry what it read, and a
        Code Mode script can catch the error and go on.
        """
        try:
            return await handler(args)
        except (CallDeferred, SkipToolExecution):
            raise
        except ApprovalRequired as error:
            if _has_reason(error.metadata):
                raise
            asked = await self._decide(ctx, call=call, tool_def=tool_def)
            if asked is None:
                raise
            self._trace(ctx, call, asked)
            if self.on_tainted == 'block':
                raise SkipToolExecution(ToolDenied(_block_message(call.tool_name))) from error
            raise ApprovalRequired(metadata=_with_reason(error.metadata, asked)) from error
        except Exception:
            await self._record(ctx, call, tool_def)
            raise

    async def _decide(
        self, ctx: RunContext[AgentDepsT], *, call: ToolCallPart, tool_def: ToolDefinition
    ) -> _Asked | None:
        """Why this call is held, or `None` to let it run. The one place the gate decides."""
        if not await self._gates(ctx, tool_def):
            return None
        sources = await self._untrusted_sources(ctx, call)
        if not sources:
            return None
        names = sorted({source.tool_name for source in sources if source.tool_name is not None})
        read = f' ({", ".join(f"`{name}`" for name in names)})' if names else ''
        reason = (
            f'This run has read content from outside sources{read}, and `{call.tool_name}` can act outside '
            'the run, so it waits for approval.'
        )
        return _Asked(reason=reason, tainted_by=sources)

    async def _gates(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> bool:
        if tool_def.tool_kind == _TOOL_CALL_KIND_ID:
            return False
        if self.gated_tools is None:
            return tool_def.tool_kind != _RUN_CODE_KIND_ID and not is_read_only(tool_def)
        return await matches_tool_selector(self.gated_tools, ctx, tool_def)

    async def _untrusted_sources(self, ctx: RunContext[AgentDepsT], call: ToolCallPart) -> list[_Source]:
        index = _making_response(ctx.messages, call.tool_call_id)
        if index is None:
            found = [source async for source in self._scan(ctx, ctx.messages)]
            ran = [source for _, source in self._ran]
        else:
            found = [source async for source in self._scan(ctx, ctx.messages[:index])]
            found += _native_reads(ctx.messages[index])
            # Tools that ran in this step answer the response that made the call, so the model chose the call
            # before their results existed. Earlier ones count even when compaction removed their results.
            ran = [source for step, source in self._ran if step < ctx.run_step]
        for source in ran:
            if source not in found:
                found.append(source)
        return found

    async def _scan(self, ctx: RunContext[AgentDepsT], messages: Sequence[ModelMessage]) -> AsyncIterator[_Source]:
        calls: dict[str, ToolCallPart] = {}
        for message in messages:
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    # The latest call with an ID is the one its next result answers: some models reuse short IDs.
                    calls[part.tool_call_id] = part
                source = await self._source_of(ctx, part, calls)
                if source is not None:
                    yield source

    async def _source_of(
        self, ctx: RunContext[AgentDepsT], part: ModelRequestPart | ModelResponsePart, calls: dict[str, ToolCallPart]
    ) -> _Source | None:
        """Where `part`'s content came from, when it came from outside the run; `None` when it did not."""
        if isinstance(part, NativeToolReturnPart):
            return _Source(tool_name=part.tool_name, tool_call_id=part.tool_call_id)
        if isinstance(part, ToolReturnPart):
            return await self._tool_return_source(ctx, part, calls) if _tool_wrote(part) else None
        if isinstance(part, CompactionPart):
            # A summary can carry what the tainting parts it replaced said, and doesn't say which they were.
            return _Source(tool_name=None, tool_call_id=None)
        if isinstance(part, InstructionDeltaPart):
            # The instructions of a capability lowered to `'none'`, such as an MCP server's.
            return _Source(tool_name=None, tool_call_id=None) if _untrusted_instructions(part) else None
        if not isinstance(part, CapabilityPart) or part.source.authority != 'none':
            return None
        source = part.source
        if not isinstance(source, ToolResultSource):
            return _Source(tool_name=None, tool_call_id=None)
        return await self._tool_source(ctx, source.tool_name, source.tool_call_id)

    async def _tool_return_source(
        self, ctx: RunContext[AgentDepsT], part: ToolReturnPart, calls: dict[str, ToolCallPart]
    ) -> _Source | None:
        """Where a result the tool wrote came from: under the bridged tool's name for `ToolBridge`'s `tool_call`."""
        if (bridged := self._bridged_name(part, calls)) is not None:
            trusted = self._trusted_by_name.get(bridged, False)
            return None if trusted else _Source(tool_name=bridged, tool_call_id=part.tool_call_id)
        return await self._tool_source(ctx, part.tool_name, part.tool_call_id)

    async def _tool_source(
        self, ctx: RunContext[AgentDepsT], tool_name: str, tool_call_id: str | None
    ) -> _Source | None:
        return None if await self._trusts(ctx, tool_name) else _Source(tool_name=tool_name, tool_call_id=tool_call_id)

    async def _record(self, ctx: RunContext[AgentDepsT], call: ToolCallPart, tool_def: ToolDefinition) -> None:
        if tool_def.tool_kind == _TOOL_CALL_KIND_ID:
            return  # The bridged call it ran was recorded under the bridged tool's name.
        trusted = await self._trusts(ctx, call.tool_name)
        self._trusted_by_name[call.tool_name] = trusted
        if not trusted:
            self._ran.append((ctx.run_step, _Source(tool_name=call.tool_name, tool_call_id=call.tool_call_id)))

    @staticmethod
    def _bridged_name(part: ToolReturnPart, calls: dict[str, ToolCallPart]) -> str | None:
        """The tool `ToolBridge`'s `tool_call` ran for this result, or `None` when `part` isn't `tool_call`'s.

        The result's own `tool_kind` says it is `tool_call`'s, so this holds inside a bridged call's nested tool
        manager and for history from earlier runs. The bridged tool's trust is known only when it ran in this run,
        which recorded it under its own name: a result from an earlier run taints.
        """
        call = calls.get(part.tool_call_id)
        if part.tool_kind != _TOOL_CALL_KIND_ID or call is None:
            return None
        name = call.args_as_dict().get('name')  # Malformed arguments read as `{}`.
        return name if isinstance(name, str) else part.tool_name

    async def _trusts(self, ctx: RunContext[AgentDepsT], tool_name: str) -> bool:
        """Whether `trusted_tools` selects the tool.

        A tool the tool manager can't resolve, such as a native tool seen from inside a bridged call, counts as
        trusted only when it ran in this run as a trusted tool; any other is untrusted.
        """
        tool_def = ctx.tool_manager.get_tool_def(tool_name) if ctx.tool_manager is not None else None
        if tool_def is None:
            return self._trusted_by_name.get(tool_name, False)
        return await matches_tool_selector(self.trusted_tools, ctx, tool_def)

    def _trace(self, ctx: RunContext[AgentDepsT], call: ToolCallPart, asked: _Asked) -> None:
        names = tuple(sorted({s.tool_name for s in asked.tainted_by if s.tool_name is not None}))
        attributes: dict[str, str | int | tuple[str, ...]] = {
            'untrusted_content_gate.action': self.on_tainted,
            'untrusted_content_gate.tool_name': call.tool_name,
            'untrusted_content_gate.source_count': len(asked.tainted_by),
            'untrusted_content_gate.source_tool_names': names,
        }
        ctx.tracer.start_span('untrusted content gate', attributes=attributes).end()


def _asks_at_validation(ctx: RunContext[Any], tool_def: ToolDefinition) -> bool:
    """Whether the call may ask for approval during validation: `requires_approval=True`, or an `args_validator`."""
    if tool_def.kind == 'unapproved':
        return True
    tools = ctx.tool_manager.tools if ctx.tool_manager is not None else None
    tool = tools.get(tool_def.name) if tools is not None else None
    return tool is not None and tool.args_validator_func is not None


def _tool_wrote(part: ToolReturnPart) -> bool:
    """Whether the tool, rather than the framework, wrote the result's content."""
    if part.outcome == 'retried':
        return part.retry is not None and part.retry.cause == 'model_retry'
    return part.outcome not in _FRAMEWORK_OUTCOMES


def _untrusted_instructions(part: InstructionDeltaPart) -> bool:
    return part.authority == 'none' and part.content is not None


def _making_response(messages: Sequence[ModelMessage], tool_call_id: str) -> int | None:
    """The index of the model response that made the call, or `None` when no response made it.

    Searched from the newest message back: some models reuse short call ids across turns, and an earlier response
    with the same id would hide everything read since.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelResponse) and any(
            isinstance(part, ToolCallPart) and part.tool_call_id == tool_call_id for part in message.parts
        ):
            return index
    return None


def _native_reads(response: ModelMessage) -> list[_Source]:
    """The provider-native tool results in the response that made a call.

    The provider ran them while the model wrote that response, so the model may have read them before making the
    call. Where they sit in the response doesn't say: some providers put them after every other part.
    """
    return [
        _Source(tool_name=part.tool_name, tool_call_id=part.tool_call_id)
        for part in response.parts
        if isinstance(part, NativeToolReturnPart)
    ]


def _has_reason(metadata: dict[str, Any] | None) -> bool:
    return metadata is not None and _KEY in metadata


def _with_reason(metadata: dict[str, Any] | None, asked: _Asked) -> dict[str, Any]:
    """`metadata` with the gate's reason added under its own key."""
    return {**(metadata or {}), _KEY: asked.model_dump()}


def _block_message(tool_name: str) -> str:
    return (
        f'`{tool_name}` was not run, because this run has read content from outside sources and calls that act '
        'outside the run are blocked after that. Continue with the parts of the task that do not need it.'
    )


def untrusted_content_reason(metadata: object) -> str | None:
    """The gate's reason in an approval request's metadata, or `None` when the gate did not ask about the call."""
    try:
        return _AskedMetadata.model_validate(metadata).untrusted_content_gate.reason
    except ValidationError:
        return None
