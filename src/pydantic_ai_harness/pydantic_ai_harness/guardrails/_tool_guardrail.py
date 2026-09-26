"""Tool guardrail capability.

`ToolGuardrail` is the third guardrail edge. `InputGuardrail` screens the user prompt
and `OutputGuardrail` screens the agent output, but neither sees a tool call:
`InputGuardrail` evaluates only the first model request, so everything a tool sends
out and everything a tool brings back passes unchecked. That matters because
tool results are where untrusted content enters an agent loop -- fetched pages,
file contents, MCP server responses.

`ToolGuardrail` inspects both sides of a tool call with the same
[`GuardrailResult`][pydantic_ai_harness.GuardrailResult] vocabulary the other two
guards use:

- `guard` sees the validated arguments before the tool runs.
- `result_guard` sees the tool's return value before it reaches the model.

Verdicts map onto Pydantic AI control flow rather than a parallel mechanism:
`block` uses [`SkipToolExecution`][pydantic_ai.exceptions.SkipToolExecution] so
the refusal becomes the tool result and the agent can recover, `retry` raises
[`ModelRetry`][pydantic_ai.exceptions.ModelRetry], and `approve` raises
[`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] so the call joins
the run's deferred approval flow.
"""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, WrapRunHandler
from pydantic_ai.exceptions import (
    ApprovalRequired,
    ModelRetry,
    SkipToolExecution,
    ToolFailedError,
    ToolRetryError,
    UserError,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from typing_extensions import assert_never

from pydantic_ai_harness.guardrails._shared import (
    GuardOutcome,
    evaluate,
    trace_approval,
    trace_block,
    trace_redaction,
)

if TYPE_CHECKING:
    from pydantic_ai.capabilities import WrapToolExecuteHandler
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition


_DEFAULT_ARGS_BLOCK_MESSAGE = 'Tool call blocked by tool guardrail.'
_DEFAULT_RESULT_BLOCK_MESSAGE = 'Tool result blocked by tool guardrail.'
_DEFAULT_RETRY_MESSAGE = 'Tool call rejected by tool guardrail.'

_ARGS_DIRECTION = 'tool args'
_RESULT_DIRECTION = 'tool result'


def _is_arguments(value: object) -> TypeGuard[Mapping[str, Any]]:
    """Whether a `replace` verdict carries something shaped like tool arguments.

    Shape and key type only. The values are **not** re-validated against the
    tool's schema: a guard that substitutes them is trusted to keep them valid,
    the same way a tool body is trusted with what it returns. Keys are checked
    because they become keyword arguments, where a non-string one raises a bare
    `TypeError` from the interpreter that names neither the tool nor the guard.
    """
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in cast('Mapping[object, object]', value))


@dataclass(frozen=True, kw_only=True)
class ToolCallInfo:
    """The tool call an argument guard inspects."""

    name: str
    """Name of the tool the model called."""

    args: Mapping[str, Any]
    """Validated arguments, after the tool's own schema validation."""

    tool_call_id: str
    """Identifier of this call, for correlating with the run's messages."""


@dataclass(frozen=True, kw_only=True)
class ToolResultInfo(ToolCallInfo):
    """The tool call and its return value, as a result guard sees them."""

    result: object
    """What the tool returned, before any other capability transforms it.

    A tool may return a [`ToolReturn`][pydantic_ai.messages.ToolReturn] wrapper
    rather than a bare value, in which case that wrapper is what arrives here.
    Stringifying it yields a repr and replacing it with a string discards its
    metadata. Use
    [`for_tool_result_text`][pydantic_ai_harness.guardrails.detectors.for_tool_result_text]
    when a text detector should inspect a string payload without losing the
    wrapper's content or metadata.

    This is the object the tool produced, not a copy: read it, and return
    `GuardrailResult.replace` to change it. Mutating it in place changes what the
    model sees while the verdict says `allow`, with no redaction span to show
    for it. The arguments a guard sees *are* copied, because they come from the
    tool's own schema and are therefore small and copyable; a return value is
    whatever the tool chose to return, which can be neither.
    """


ToolGuardrailFunc = (
    Callable[[ToolCallInfo], GuardOutcome | Awaitable[GuardOutcome]]
    | Callable[[RunContext[AgentDepsT], ToolCallInfo], GuardOutcome | Awaitable[GuardOutcome]]
)
"""Signature of the callable passed to `ToolGuardrail(guard=...)`.

The callable receives a [`ToolCallInfo`][pydantic_ai_harness.ToolCallInfo] and
returns `True` / `GuardrailResult`. It may optionally take a
[`RunContext`][pydantic_ai.tools.RunContext] as a first argument, and may be
sync or async. Raising an exception is treated as a hard failure and propagates
up to the caller.
"""

ToolResultGuardrailFunc = (
    Callable[[ToolResultInfo], GuardOutcome | Awaitable[GuardOutcome]]
    | Callable[[RunContext[AgentDepsT], ToolResultInfo], GuardOutcome | Awaitable[GuardOutcome]]
)
"""Signature of the callable passed to `ToolGuardrail(result_guard=...)`.

The callable receives a [`ToolResultInfo`][pydantic_ai_harness.ToolResultInfo],
whose `result` is the tool's return value unchanged -- no stringification, so a
structured result arrives as the object the tool produced.
"""


@dataclass
class ToolGuardrail(AbstractCapability[AgentDepsT]):
    """Validate tool arguments before execution and tool results after it.

    ```python
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai_harness.guardrails import GuardrailResult, ToolCallInfo, ToolGuardrail


    def no_writes_outside_workspace(call: ToolCallInfo) -> GuardrailResult:
        if call.name == 'write_file':
            # `resolve()` before the containment check: a prefix test on the raw
            # string accepts `/workspace/../etc/passwd`.
            target = Path(str(call.args['path'])).resolve()
            if not target.is_relative_to(Path('/workspace')):
                return GuardrailResult.block(f'{target} is outside the workspace.')
        return GuardrailResult.allow()


    agent = Agent('openai:gpt-5.4', capabilities=[ToolGuardrail(guard=no_writes_outside_workspace)])
    ```

    Verdicts, by stage:

    | Outcome | `guard` (arguments) | `result_guard` (result) |
    |---|---|---|
    | `allow` | run the tool | return the result unchanged |
    | `block` | skip execution; the refusal message becomes the tool result | the refusal message replaces the result |
    | `replace` | run the tool with substituted arguments (a mapping) | substitute a sanitized result |
    | `retry` | ask the model to redo the call | ask the model to redo the call |
    | `approve` | defer the call for human approval | not valid; raises `UserError` |

    `block` is the graceful path on both stages: the agent sees the refusal text
    where it expected a tool result and can choose another approach. To fail the
    run instead, raise [`ToolBlocked`][pydantic_ai_harness.ToolBlocked] (or any
    exception) from the guard.

    A `replace` verdict on the argument stage is trusted to match the tool's
    signature. Replacement keys reach the tool as keyword arguments, so one it
    does not accept raises a bare `TypeError` that names neither the tool nor
    the guard.

    Either guard may take a [`RunContext`][pydantic_ai.tools.RunContext] as a
    first parameter when it needs run state, such as `deps` for role-aware
    policy. The parameter is detected from the signature.

    Scope: output tools do not fire tool-execution hooks in Pydantic AI, so a
    guard never sees the call that produces the agent's structured output --
    use `OutputGuardrail` for that.

    Ordering: declares `position='innermost'`. Argument hooks run outermost
    first, so the guard sees the arguments every other capability has already
    finished modifying; result hooks run innermost first, so the guard sees the
    raw tool result before a capability such as `ToolOutputLimits`
    truncates or offloads it.
    """

    guard: ToolGuardrailFunc[AgentDepsT] | None = None
    """Callable that decides what to do with a tool call before it executes."""

    result_guard: ToolResultGuardrailFunc[AgentDepsT] | None = None
    """Callable that decides what to do with a tool result before the model sees it."""

    tools: Sequence[str] | None = None
    """Restrict both guards to these tool names. `None` guards every tool.

    An empty sequence guards nothing, which is the same as configuring no guard at all --
    write `None` when you mean every tool.

    A configured name that no offered tool matches warns after a successful run when
    `guard` or `result_guard` is set. Toolsets can vary between run steps, so absence
    from one step is not enough to diagnose a typo.
    """

    hidden: Sequence[str] = field(default_factory=tuple)
    """Tool names to withhold from the model entirely.

    Hiding differs from blocking: a hidden tool is dropped from the tool
    definitions sent to the model, so it costs no tokens and the model never
    tries to call it. A blocked tool stays visible and the model learns it was
    refused. Hiding is a static name list; for policy that depends on `deps` or
    the arguments, use `guard`.

    A name here that no offered tool answers to raises a `UserWarning` after the
    run. A misspelled entry leaves the tool on the wire and reads, at the call
    site, exactly like a tool that was hidden.
    """

    _seen_hidden: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    """`hidden` names that appeared during this run."""

    _seen_tools: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    """`tools` names that appeared during this run."""

    _prepared_tools: bool = field(default=False, init=False, repr=False, compare=False)
    """Whether this run reached function-tool preparation."""

    def __post_init__(self) -> None:
        """Reject a bare string where a collection of tool names is meant.

        `str` satisfies `Sequence[str]`, so pyright accepts `hidden='danger'`
        and the set built from it holds the six letters of the word. The tool
        stays on the wire and nothing reports it -- a safety control failing
        open. `tools='delete_all'` fails the other way, matching any tool whose
        name is a substring of it.
        """
        for field_name, value in (('tools', self.tools), ('hidden', self.hidden)):
            if isinstance(value, str):
                raise UserError(
                    f'ToolGuardrail.{field_name} takes a collection of tool names, not a single string. '
                    f'Pass [{value!r}] rather than {value!r}.'
                )

    def get_ordering(self) -> CapabilityOrdering:
        """Sit innermost: see the final arguments, and the tool result before other capabilities rewrite it."""
        return CapabilityOrdering(position='innermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Exclude a callable policy from YAML and JSON agent specifications."""
        return None

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ToolGuardrail[AgentDepsT]:
        """Start a fresh selector observation window for each run."""
        return replace(self)

    def _guards(self, tool_name: str) -> bool:
        """Whether this guard applies to `tool_name`."""
        return self.tools is None or tool_name in self.tools

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Drop the `hidden` tools from what the model is offered.

        Names are observed for a warning at the completed-run boundary. A toolset
        may legitimately offer a tool only on some steps, so first absence is not
        evidence of a typo.
        """
        self._prepared_tools = True
        offered = {tool_def.name for tool_def in tool_defs}
        self._seen_tools.update(set(self.tools or ()).intersection(offered))
        self._seen_hidden.update(set(self.hidden).intersection(offered))
        if not self.hidden:
            return tool_defs
        return [tool_def for tool_def in tool_defs if tool_def.name not in self.hidden]

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[object]:
        """Report unmatched selector names after a successful, prepared run."""
        result = await handler()
        self._warn_unmatched_names()
        return result

    def _warn_unmatched_names(self) -> None:
        """Warn when a configured selector never appeared in this run."""
        if not self._prepared_tools:
            return
        if self.tools and (self.guard is not None or self.result_guard is not None):
            for name in sorted(set(self.tools) - self._seen_tools):
                warnings.warn(
                    f'ToolGuardrail.tools names {name!r}, which no tool offered during this run is called. '
                    'Neither guard was applied to it.',
                    UserWarning,
                    stacklevel=3,
                )
        for name in sorted(set(self.hidden) - self._seen_hidden):
            warnings.warn(
                f'ToolGuardrail.hidden names {name!r}, which no tool offered during this run is called. '
                'The name stays available to the model.',
                UserWarning,
                stacklevel=3,
            )

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate `guard` against the validated arguments and act on its verdict."""
        if self.guard is None or not self._guards(call.tool_name):
            return args

        # A read-only view over a copy. The dict passed here is the one the tool is about to
        # be called with, so a guard mutating it would change the call while reporting
        # `allow`, bypassing both the `replace` contract and its span. The proxy alone stops
        # that only at the top level: `call.args['opts']['path'] = ...` reaches through it to
        # the real object. Validated arguments come from the tool's own schema, so they are
        # small and copyable; `replace` stays the one way to change a call, and the one that
        # gets a span.
        info = ToolCallInfo(name=call.tool_name, args=MappingProxyType(deepcopy(args)), tool_call_id=call.tool_call_id)
        verdict = await evaluate(self.guard, ctx, info)
        match verdict.action:
            case 'allow':
                return args
            case 'block':
                message = verdict.message or _DEFAULT_ARGS_BLOCK_MESSAGE
                trace_block(ctx, direction=_ARGS_DIRECTION, message=message, tool_name=call.tool_name)
                raise SkipToolExecution(message)
            case 'retry':
                raise ModelRetry(verdict.message or _DEFAULT_RETRY_MESSAGE)
            case 'approve':
                # A resumed run re-evaluates the guard, and a policy that asked for approval
                # once asks again. Honoring `tool_call_approved` -- the same flag
                # `ApprovalRequiredToolset` reads -- is what ends the round trip instead of
                # deferring the call forever.
                if ctx.tool_call_approved:
                    return args
                trace_approval(
                    ctx,
                    direction=_ARGS_DIRECTION,
                    tool_name=call.tool_name,
                    tool_call_id=call.tool_call_id,
                    args=args,
                )
                raise ApprovalRequired
            case 'replace':
                if not _is_arguments(verdict.replacement):
                    raise UserError(
                        'GuardrailResult.replace() for a tool argument guard must provide replacement arguments '
                        '(a mapping).'
                    )
                replacement = dict(verdict.replacement)
                trace_redaction(
                    ctx,
                    direction=_ARGS_DIRECTION,
                    original=args,
                    replacement=replacement,
                    tool_name=call.tool_name,
                )
                return replacement
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(verdict.action)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Screen the message of a tool that failed, which reaches the model like a result.

        A tool raising `ModelRetry` or `ToolFailed` never reaches `after_tool_execute`: core
        wraps both into control-flow exceptions and re-raises them past it
        (`tool_manager._run_execute_hooks`). Their text still lands in the conversation -- a
        failure as an ordinary `ToolReturnPart`, a retry as a `RetryPromptPart` -- so a
        `result_guard` that did not see them would leave the one text path it exists to
        screen unscreened whenever a tool errored.
        """
        if self.result_guard is None or not self._guards(call.tool_name):
            return await handler(args)
        try:
            return await handler(args)
        except ToolRetryError as error:
            message = await self._screen_failure(ctx, call=call, args=args, content=error.tool_retry.content)
            if message is None:
                raise
            raise ToolRetryError(replace(error.tool_retry, content=message)) from error
        except ToolFailedError as error:
            message = await self._screen_failure(ctx, call=call, args=args, content=error.tool_failed.content)
            if message is None:
                raise
            raise ToolFailedError(replace(error.tool_failed, content=message)) from error

    async def _screen_failure(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: dict[str, Any],
        content: object,
    ) -> str | None:
        """The text a failed tool should carry instead, or `None` to leave it alone.

        The failure keeps its own type either way: a failure screened into a refusal is still
        a failure, and turning it into a success would tell the model the call worked.
        """
        assert self.result_guard is not None
        info = ToolResultInfo(
            name=call.tool_name, args=MappingProxyType(deepcopy(args)), tool_call_id=call.tool_call_id, result=content
        )
        verdict = await evaluate(self.result_guard, ctx, info)
        match verdict.action:
            case 'allow':
                return None
            case 'retry':
                # The tool already failed, but `retry` still means retry: collapsing it into a
                # replacement message would hand the model a failed result where it was asked
                # to redo the call, and skip the run's retry accounting with it.
                raise ModelRetry(verdict.message or _DEFAULT_RETRY_MESSAGE)
            case 'block':
                message = verdict.message or _DEFAULT_RESULT_BLOCK_MESSAGE
                trace_block(ctx, direction=_RESULT_DIRECTION, message=message, tool_name=call.tool_name)
                return message
            case 'replace':
                trace_redaction(
                    ctx,
                    direction=_RESULT_DIRECTION,
                    original=content,
                    replacement=verdict.replacement,
                    tool_name=call.tool_name,
                )
                return str(verdict.replacement)
            case 'approve':
                raise UserError(
                    'A tool result guard cannot return GuardrailResult.approve() -- the tool has already run. '
                    'Return approve() from the argument guard instead.'
                )
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(verdict.action)

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Evaluate `result_guard` against the tool result and act on its verdict."""
        if self.result_guard is None or not self._guards(call.tool_name):
            return result

        info = ToolResultInfo(
            name=call.tool_name, args=MappingProxyType(deepcopy(args)), tool_call_id=call.tool_call_id, result=result
        )
        verdict = await evaluate(self.result_guard, ctx, info)
        match verdict.action:
            case 'allow':
                return result
            case 'block':
                message = verdict.message or _DEFAULT_RESULT_BLOCK_MESSAGE
                trace_block(ctx, direction=_RESULT_DIRECTION, message=message, tool_name=call.tool_name)
                return message
            case 'retry':
                raise ModelRetry(verdict.message or _DEFAULT_RETRY_MESSAGE)
            case 'approve':
                raise UserError(
                    'A tool result guard cannot return GuardrailResult.approve() -- the tool has already run. '
                    'Return approve() from the argument guard instead.'
                )
            case 'replace':
                trace_redaction(
                    ctx,
                    direction=_RESULT_DIRECTION,
                    original=result,
                    replacement=verdict.replacement,
                    tool_name=call.tool_name,
                )
                return verdict.replacement
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(verdict.action)
