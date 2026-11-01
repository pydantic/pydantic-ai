"""Tool plumbing shared by the code execution capabilities (`CodeMode` and `IPythonCodeMode`).

Kept free of any sandbox or kernel dependency, so each capability's optional extra stays its own.
"""

from __future__ import annotations

import keyword
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai import ToolDefinition
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, UsageLimitExceeded, UserError
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.messages import (
    ToolCallPart,
    ToolPartKind,
    ToolReturn,
    ToolReturnPart,
    UserContent,
    is_multi_modal_content,
)
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import AgentDepsT, ToolDenied
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai_harness.guardrails._capability import input_guardrails_allow

_FRAMEWORK_CONTROL_TOOL_KINDS = frozenset({'tool-search', 'capability-load'})
"""Kinds of the framework's control tools, which stay native to drive tool search and capability loading."""


def can_be_sandboxed(tool_def: ToolDefinition) -> bool:
    """Whether a tool may go inside `run_code` at all, whatever the selector says.

    Framework control tools (tool search, capability loading) drive protocol-level flows; a local fallback
    (`unless_native`) has to stay native so `Model.prepare_request` can drop it when the provider supports
    the native tool; and a tool that is itself a code-execution sandbox (e.g. DynamicWorkflow's
    `run_workflow`) is a peer of `run_code`, not something to fold inside it.
    """
    return (
        tool_def.tool_kind not in _FRAMEWORK_CONTROL_TOOL_KINDS
        and not tool_def.unless_native
        and not is_code_execution_tool(tool_def)
    )


_INVALID_IDENT_CHARS = re.compile(r'[^a-zA-Z0-9_]')


def is_code_execution_tool(tool_def: ToolDefinition) -> bool:
    """Whether a tool executes its string argument as a program when called.

    Such tools carry `code_arg_name` metadata -- the same marker instrumentation reads to render
    the argument as code. It covers script sandboxes (this `run_code`, DynamicWorkflow's
    `run_workflow`), shell surfaces that hand the argument to a shell (`Shell`'s
    `run_command`/`start_command`), and tools that import the
    argument as Python (`CapabilityCreation`'s `author_capability`). They must not be folded
    into `run_code`: nesting one code surface inside another would make the model write a script
    that passes a second script as a string literal. They stay native so the two code surfaces
    sit side by side. Tools whose string argument is data (file contents, an argv-style command
    parsed with `shlex`) are folded as usual.
    """
    return bool(tool_def.metadata and 'code_arg_name' in tool_def.metadata)


def sanitize_tool_name(name: str) -> str:
    """Turn a tool name into a valid Python identifier.

    Replaces hyphens, dots, and other non-identifier characters with underscores,
    prepends `_` if the result starts with a digit, appends `_` if it is a Python keyword.
    """
    sanitized = _INVALID_IDENT_CHARS.sub('_', name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f'_{sanitized}'
    if keyword.iskeyword(sanitized):
        sanitized = f'{sanitized}_'
    return sanitized or '_'


def partition_callable_tools(
    tools: Mapping[str, ToolsetTool[Any]], *, owner: str, reserved_name: str
) -> tuple[dict[str, ToolDefinition], dict[str, str]]:
    """Key the tool definitions code can call by a Python identifier.

    Tool names that are not valid Python identifiers (e.g. MCP tools with hyphens or dots like
    `get-weather`, `api.call`) are sanitized to underscored forms and mapped back to their original
    names for dispatch. When two names sanitize to the same function, the first keeps it and the
    other is hidden with a warning naming `owner`.

    Returns:
        A tuple of `(callable_defs, sanitized_to_original)`; the second only lists renamed tools.
    """
    callable_defs: dict[str, ToolDefinition] = {}
    sanitized_to_original: dict[str, str] = {}
    for name, tool in tools.items():
        td = tool.tool_def
        safe_name = sanitize_tool_name(name)
        if safe_name == reserved_name:
            raise UserError(
                f"Tool name '{name}' (sanitized to '{safe_name}') conflicts with the code mode "
                f'meta-tool. Rename your tool to avoid conflicts.'
            )
        if safe_name in callable_defs:
            existing = sanitized_to_original.get(safe_name, safe_name)
            warnings.warn(
                f'{owner}: tool {name!r} (sanitized to {safe_name!r}) collides '
                f'with {existing!r}; {name!r} will be hidden from the sandbox.',
                UserWarning,
                stacklevel=3,
            )
            continue
        if safe_name != name:
            sanitized_to_original[safe_name] = name
            td = replace(td, name=safe_name)
        callable_defs[safe_name] = td
    return callable_defs, sanitized_to_original


@dataclass(frozen=True, kw_only=True)
class NestedCallOutcome:
    """How one sandbox-dispatched tool call settled, before it is recorded against a `run_code` call.

    Exactly one of `error` or `content` is meaningful. Settling instead of raising lets a call that
    ran ahead of its snippet (speculation) hold a failure until the snippet asks for it.
    """

    content: Any = None
    """The plain tool return value, with a `ToolReturn` already unwrapped."""

    metadata: Any = None
    """`ToolReturn.metadata` when the tool returned a `ToolReturn`."""

    user_content: str | Sequence[UserContent] | None = None
    """`ToolReturn.content` when the tool returned a `ToolReturn`: model-facing content outside the sandbox."""

    error: Exception | None = None
    """The exception the sandbox sees at the call site."""

    denied_message: str | None = None
    """Set when a handler denied the call, so the history records `outcome='denied'`."""

    usage_limit_exceeded: UsageLimitExceeded | None = None
    """Set, along with `error`, when `tool_calls_limit` refused the call before it ran; it ends the run."""

    tool_kind: ToolPartKind | None = None
    """The called tool's `tool_kind`, so the history records the same typed parts a direct call would."""


async def run_nested_call(tool_manager: ToolManager[AgentDepsT], call_part: ToolCallPart) -> NestedCallOutcome:
    """Run one tool call dispatched from inside the sandbox through the nested `ToolManager`."""
    tool_def = tool_manager.get_tool_def(call_part.tool_name)
    tool_kind = tool_def.tool_kind if tool_def is not None else None
    ctx = tool_manager.ctx
    assert ctx is not None, 'the nested `ToolManager` is built with the run context'
    # Eager and speculative execution dispatch while the model is still streaming, possibly before a
    # parallel input guard has decided. The verdict comes first, as it does for a non-eager call.
    if not await input_guardrails_allow(ctx):
        return NestedCallOutcome(
            error=RuntimeError(
                f'Tool {call_part.tool_name!r} did not run: an input guardrail did not allow the prompt.'
            ),
            tool_kind=tool_kind,
        )
    usage_limits = ctx.usage_limits
    reserved = 0
    if usage_limits is not None and usage_limits.tool_calls_limit is not None:
        # This call counts once it succeeds, and so does the enclosing `run_code` call.
        try:
            usage_limits.check_before_tool_call(replace(ctx.usage, tool_calls=ctx.usage.tool_calls + 2))
        except UsageLimitExceeded as e:
            return NestedCallOutcome(error=e, usage_limit_exceeded=e, tool_kind=tool_kind)
        # A pending call counts toward the limit until it settles: calls started together with
        # `asyncio.gather` would otherwise all pass the check before any of them is counted. Core
        # counts the call itself once it succeeds, so the reservation is released however it ends.
        reserved = 1
        # usage-attribution: a temporary pending-call reservation, released in `finally` below
        ctx.usage.tool_calls += 1
    try:
        result = await tool_manager.handle_call(call_part, wrap_validation_errors=False)
    except (CallDeferred, ApprovalRequired) as e:
        # No handler resolved the deferral. The sandbox can't round-trip to the caller, so the
        # error propagates through Monty -> MontyRuntimeError -> ModelRetry.
        error = UserError(
            f'Tool {call_part.tool_name!r} raised {type(e).__name__} inside code mode, '
            'but no `HandleDeferredToolCalls` capability resolved it. Add a handler '
            'capability on the agent so deferred and approval-required calls can '
            'be resolved inline.'
        )
        error.__cause__ = e
        return NestedCallOutcome(error=error, tool_kind=tool_kind)
    except Exception as e:
        return NestedCallOutcome(error=e, tool_kind=tool_kind)
    finally:
        ctx.usage.tool_calls -= reserved

    if isinstance(result, ToolDenied):
        # Surfacing `ToolDenied` to the user's script would let it masquerade as a string tool
        # result, and the script cannot introspect the marker class inside Monty.
        return NestedCallOutcome(
            error=RuntimeError(f'Tool {call_part.tool_name!r} call denied: {result.message}'),
            denied_message=result.message,
            tool_kind=tool_kind,
        )

    if isinstance(result, ToolReturn):
        return NestedCallOutcome(
            content=result.return_value, metadata=result.metadata, user_content=result.content, tool_kind=tool_kind
        )
    return NestedCallOutcome(content=result, tool_kind=tool_kind)


def record_nested_call(
    call_part: ToolCallPart,
    outcome: NestedCallOutcome,
    *,
    calls: dict[str, ToolCallPart],
    returns: dict[str, ToolReturnPart],
    user_content: dict[str, list[UserContent]],
) -> None:
    """Record a nested call and how it settled with the typed parts a direct call would leave in history.

    A call that failed gets no return part unless a handler denied it. `user_content` gets the call's
    `ToolReturn.content`, which reaches the model on the enclosing `run_code` return, not inside the code.
    """
    tool_call_id = call_part.tool_call_id
    tool_kind = outcome.tool_kind
    calls[tool_call_id] = ToolCallPart.narrow_type(call_part, tool_kind=tool_kind)
    if outcome.denied_message is not None:
        returns[tool_call_id] = ToolReturnPart(
            tool_name=call_part.tool_name,
            content=outcome.denied_message,
            tool_call_id=tool_call_id,
            outcome='denied',
            tool_kind=tool_kind,
        )
    if outcome.error is not None:
        return
    returns[tool_call_id] = ToolReturnPart.narrow_type(
        ToolReturnPart(
            tool_name=call_part.tool_name,
            content=outcome.content,
            tool_call_id=tool_call_id,
            metadata=outcome.metadata,
            tool_kind=tool_kind,
        )
    )
    # Like core, empty content sends the model nothing.
    if outcome.user_content:
        content = outcome.user_content
        user_content[tool_call_id] = [content] if isinstance(content, str) else list(content)


def signatures_and_conflicts(
    callable_defs: dict[str, ToolDefinition],
) -> tuple[list[FunctionSignature], frozenset[str]]:
    """Extract FunctionSignatures and conflicting type names from tool definitions."""
    sigs: list[FunctionSignature] = []
    for td in callable_defs.values():
        assert td.function_signature is not None, f'function_signature missing for tool {td.name!r}'
        sigs.append(td.function_signature)
    return sigs, FunctionSignature.get_conflicting_type_names(sigs)


def contains_multimodal(value: Any) -> bool:
    """Check if a value is or directly contains multimodal content (images, audio, etc.)."""
    if is_multi_modal_content(value):
        return True
    if isinstance(value, list):
        return any(is_multi_modal_content(item) for item in value)  # pyright: ignore[reportUnknownVariableType]
    return False
