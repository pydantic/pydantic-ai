"""`PromptInjectionDefender` classifies local tool results for indirect prompt injection."""

from __future__ import annotations

import importlib.util
from collections.abc import Awaitable, Callable
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

import anyio.to_thread
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import TextContent, ToolCallPart, ToolReturn
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition, ToolSelector, matches_tool_selector
from pydantic_core import to_jsonable_python

try:
    from stackone_defender import DefenseResult, PromptDefense
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'stackone-defender is required for PromptInjectionDefender (Python 3.11 or newer). '
        'Install it with: uv add "pydantic-ai-harness[prompt-injection-defender]"'
    ) from _import_error

_DEFAULT_BLOCKED_MESSAGE = (
    'The result of `{tool_name}` was withheld: it matched prompt injection patterns '
    '(risk: {risk_level}). Do not retry this call; continue without this content and '
    'tell the user the tool result was blocked.'
)
"""Default replacement text for a withheld result. Instructs the model not to retry the call."""

_METADATA_KEY = 'prompt_injection'
"""Diagnostics key on a withheld result's `ToolReturn.metadata`."""

_ESCALATED_RISKS = ('high', 'critical')
"""Risk levels that indicate the defender escalated beyond its `'medium'` starting level."""


OnDetection = Callable[[RunContext[AgentDepsT], ToolCallPart, DefenseResult], None | Awaitable[None]]
"""Signature of the `on_detection` callback.

Called for each flagged verdict. A `ToolReturn` can produce separate verdicts for its
return value and additional content. May be sync or async. Raising fails the run.
"""


def _findings(verdict: DefenseResult) -> bool:
    """Whether the defender's pattern detection matched anything."""
    return bool(verdict.detections) or bool(verdict.fields_sanitized)


def _flagged(verdict: DefenseResult) -> bool:
    """Whether a verdict is worth reporting to `on_detection`.

    `risk_level` starts at `'medium'`, so only an escalated level counts on its own.
    """
    return (not verdict.allowed) or _findings(verdict) or verdict.risk_level in _ESCALATED_RISKS


def _diagnostics(verdict: DefenseResult) -> dict[str, object]:
    """A plain-JSON summary of a verdict, safe for durable snapshots."""
    return {
        'blocked': not verdict.allowed,
        'risk_level': verdict.risk_level,
        'detections': list(verdict.detections),
        'fields_sanitized': list(verdict.fields_sanitized),
        'patterns_by_field': {key: list(patterns) for key, patterns in verdict.patterns_by_field.items()},
        'tier2_score': verdict.tier2_score,
        'latency_ms': verdict.latency_ms,
    }


def _payload(value: object) -> object:
    """Put a bare string under Defender's standard `content` field."""
    return {'content': value} if isinstance(value, str) else to_jsonable_python(value, fallback=str)


@dataclass
class PromptInjectionDefender(AbstractCapability[AgentDepsT]):
    """Classify tool results for indirect prompt injection and withhold the risky ones.

    Tool results (emails, tickets, documents, MCP payloads) are a primary channel for
    indirect prompt injection: instructions planted in third-party data that redirect the
    agent. This capability classifies each locally executed tool result with
    `stackone-defender` after the tool returns. A result passes through unchanged unless
    the defense rejects it. With `block_high_risk=True`, the built-in defense rejects
    detected high or critical risk results, which are replaced with `blocked_message` so
    their content never reaches the model. Every flagged verdict is reported through
    `on_detection`, and a withheld result carries a diagnostics summary on
    `ToolReturn.metadata` (not visible to the model).

    Pass `semantic_detection=True` to add the local ML classifier, which is what catches
    injection in text under unrecognized fields (pattern detection inspects known risky
    fields, bare string results, and `ToolReturn` content). Pass a fully configured
    `defense` for anything beyond the defaults.

    Provider-native tools (for example hosted web search) run server-side and never transit
    the client, so they are not classified here.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness import PromptInjectionDefender

        agent = Agent(
            capabilities=[PromptInjectionDefender(block_high_risk=True)],
        )
        ```
    """

    defense: PromptDefense | None = None
    """An optional custom `stackone_defender.PromptDefense` to classify with.

    When omitted, the capability creates one from `block_high_risk` and
    `semantic_detection`. Supply one (via `create_prompt_defense(...)`) for custom
    thresholds, per-tool risky fields, or Tier 3.
    """

    _: KW_ONLY

    block_high_risk: bool | None = None
    """Ask the built-in defense to reject detected high or critical risk results.

    `None` keeps the library default (`False`: report only). Cannot be combined with
    `defense`; configure blocking on the `PromptDefense` instead.
    """

    semantic_detection: bool = False
    """Use StackOne Defender's local ML classifier in addition to pattern detection.

    Requires the `prompt-injection-defender-ml` extra. The model is preloaded at run start.
    Cannot be combined with `defense`; configure Tier 2 on the `PromptDefense` instead.
    """

    tool_filter: ToolSelector[AgentDepsT] = 'all'
    """Which tools this capability classifies. Non-matching tools always pass through."""

    on_detection: OnDetection[AgentDepsT] | None = None
    """Called for each verdict with detections, sanitization, rejection, or escalated risk."""

    blocked_message: str = _DEFAULT_BLOCKED_MESSAGE
    """Replacement text the model sees for a withheld result.

    May reference `{tool_name}` and `{risk_level}`; literal braces must be doubled.
    """

    _defense: PromptDefense = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            self.blocked_message.format(tool_name='tool', risk_level='high')
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            raise UserError(
                f'PromptInjectionDefender got an invalid `blocked_message` placeholder in '
                f'{self.blocked_message!r}: {error}. Only `{{tool_name}}` and `{{risk_level}}` are supported.'
            ) from error
        if self.defense is not None:
            if self.block_high_risk is not None:
                raise UserError(
                    'PromptInjectionDefender got both `defense` and `block_high_risk`; the option would have no '
                    'effect. Configure blocking on the supplied defense instead.'
                )
            if self.semantic_detection:
                raise UserError(
                    'PromptInjectionDefender got both `defense` and `semantic_detection`; the option would have no '
                    'effect. Configure Tier 2 on the supplied defense instead.'
                )
            self._defense = self.defense
        else:
            if self.semantic_detection and importlib.util.find_spec('onnxruntime') is None:
                raise UserError(
                    'PromptInjectionDefender requires ONNX Runtime when `semantic_detection=True`. '
                    'Install it with: uv add "pydantic-ai-harness[prompt-injection-defender-ml]"'
                )
            # Construct `PromptDefense` directly; the `create_prompt_defense` factory is untyped and fails pyright strict.
            self._defense = PromptDefense(
                # Scan large arrays in full; the default samples them, letting a tail injection reach the model.
                config={'traversal': {'skip_large_arrays': False}},
                block_high_risk=bool(self.block_high_risk),
                enable_tier2=self.semantic_detection,
            )

    def get_ordering(self) -> CapabilityOrdering:
        """Classify closest to tool execution, before other capabilities reshape the result."""
        return CapabilityOrdering(position='innermost')

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Preload the optional semantic classifier off the event loop."""
        if self.semantic_detection:
            await anyio.to_thread.run_sync(self._defense.warmup_tier2)

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Classify each model-visible part; withhold the whole result when any part is blocked."""
        if not await matches_tool_selector(self.tool_filter, ctx, tool_def):
            return result
        # Keep the original result; the `isinstance` check below narrows `result` to a type pyright rejects on return.
        unchanged = result
        values: list[object]
        if isinstance(result, ToolReturn):
            values = [result.return_value]
            if isinstance(result.content, str):
                # Tier 1 classifies strings under risky field names, so preserve the model-facing
                # `content` field when projecting this separate message part.
                values.append({'content': result.content})
            elif result.content is not None:
                text = '\n'.join(
                    item if isinstance(item, str) else item.content
                    for item in result.content
                    if isinstance(item, str | TextContent)
                )
                if text:
                    values.append({'content': text})
        else:
            values = [result]
        verdicts = [await self._defense.defend_tool_result_async(_payload(value), call.tool_name) for value in values]
        blocking = next((verdict for verdict in verdicts if not verdict.allowed), None)
        blocked: ToolReturn[str] | None = None
        if blocking is not None:
            message = self.blocked_message.format(tool_name=call.tool_name, risk_level=blocking.risk_level)
            blocked = ToolReturn(return_value=message, metadata={_METADATA_KEY: _diagnostics(blocking)})
        for verdict in verdicts:
            if _flagged(verdict):
                await self._notify(ctx, call, verdict)
        return unchanged if blocked is None else blocked

    async def _notify(self, ctx: RunContext[AgentDepsT], call: ToolCallPart, verdict: DefenseResult) -> None:
        if self.on_detection is None:
            return
        outcome = self.on_detection(ctx, call, verdict)
        if isinstance(outcome, Awaitable):
            await outcome
