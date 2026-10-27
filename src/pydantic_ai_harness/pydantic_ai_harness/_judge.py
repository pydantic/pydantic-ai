"""Shared plumbing for capabilities that ask a second model one question about a tool call.

`ToolCallJudge` and `ApprovalReviewer` build the same prompt, read the same provider confidence, and
claim their request on the run's usage the same way. They differ only in what an answer does.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal, TypeAlias

from pydantic import TypeAdapter, ValidationError

from pydantic_ai._framing import neutralize_tags
from pydantic_ai.messages import (
    ModelMessage,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryFeedbackPart,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]  # TODO(v3): remove RetryPromptPart
    SpeechPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import UsageLimits

JudgeAnswer: TypeAlias = Literal['yes', 'no', 'unsure']
"""The answer to a yes/no risk question about one tool call."""

SerializableToolSelector: TypeAlias = Literal['all'] | Sequence[str] | dict[str, Any]
"""The `ToolSelector` forms an agent spec can express. A predicate is code-only."""

# Same ~4 characters-per-token heuristic, transcript renderer, and window clamp as
# `pydantic_ai_harness.trajectory_judge`. Not shared with it because that renderer also tags the run
# id and splits the transcript; fold them together if a shared transcript helper lands.
_CHARS_PER_TOKEN = 4

# The `<` of every opening or closing `tool_call` or `conversation` tag, the two blocks this prompt is
# built from. Pydantic AI's own framing tags (`<system>` and the rest) are neutralized by
# `pydantic_ai._framing.neutralize_tags`, which doesn't know these two. Backslashes before the slash count,
# since the arguments are JSON, where `\/` is a standard escape for `/`. For the same reason a tag name
# followed by `"` or `\` counts as cut off: in JSON that is the end of the string or the start of an
# escape such as `\n`, either of which a reader may take as the tag's end.
_BLOCK_TAG_OPENER = re.compile(r'<(?=\s*(?:\\*/\s*)?(?:tool_call|conversation)(?=[\s/>"\\]|$))', re.IGNORECASE)

_CONFIDENCE_ADAPTER = TypeAdapter(dict[str, float])


def reported_confidence(provider_details: Mapping[str, object] | None) -> float | None:
    """Read single-answer confidence from model provider details, when available."""
    if provider_details is None:
        return None
    confidence = provider_details.get('confidence')
    try:
        parsed = _CONFIDENCE_ADAPTER.validate_python(confidence, strict=True)
    except ValidationError:
        return None
    value = parsed.get('response')
    return value if value is not None and 0 <= value <= 1 else None


def tool_call_prompt(
    messages: Sequence[ModelMessage],
    tool_name: str,
    args: dict[str, Any],
    *,
    include_conversation: bool,
    conversation_window: int,
) -> str:
    """Build the prompt: the call, and the conversation clamped to its most recent tokens when asked for.

    The arguments go in as JSON and the transcript as rendered, with nothing HTML-escaped, so `>`
    and `&` in a command reach the second model as they would reach a person. Only the `<` of an
    opening or closing framework tag is escaped: `tool_call`, `conversation`, and Pydantic AI's own
    framing tags such as `system`. So the call's arguments or a tool result can neither end their
    block early nor forge another block or the framework's voice.
    """
    payload = json.dumps({'tool_name': tool_name, 'arguments': args}, default=str, ensure_ascii=False)
    tool_call = f'<tool_call>\n{_neutralize_tags(payload)}\n</tool_call>'
    if not include_conversation:
        return tool_call
    # Neutralize before clamping, so escaping can't push the transcript past its window. A cut that
    # lands inside an escaped `&lt;` leaves no `<` behind, so it can't make a closing tag either.
    transcript = _neutralize_tags(_render_transcript(messages))
    max_chars = conversation_window * _CHARS_PER_TOKEN
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]
    return f'<conversation>\n{transcript}\n</conversation>\n\n{tool_call}'


def _neutralize_tags(text: str) -> str:
    r"""Escape the `<` of every opening or closing framework tag as `&lt;`, and leave everything else.

    That covers `tool_call` and `conversation`, including a closer with backslashes before its slash
    (`<\/tool_call>`), and every Pydantic AI framing tag.
    """
    return _BLOCK_TAG_OPENER.sub('&lt;', neutralize_tags(text))


def _prompt_text(content: str | Sequence[object]) -> str:
    """The text of a user prompt; non-text content (images, files) is omitted."""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for item in content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return ' '.join(texts)


def _render_transcript(messages: Sequence[ModelMessage]) -> str:
    """Render the conversation as model-readable lines.

    Instructions, system prompts, and thinking parts are omitted: the second model is shown what
    was asked, said, called, and returned, not the agent's configuration or private reasoning.
    """
    lines: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                text = _prompt_text(part.content)
                if text:
                    lines.append(f'user: {text}')
            elif isinstance(part, ToolReturnPart):
                lines.append(
                    f'retry ({part.tool_name}): {part.model_response_str(wrap_if_error=False)}'
                    if part.outcome == 'retried'
                    else f'tool {part.tool_name} returned: {part.model_response_str()}'
                )
            elif isinstance(part, NativeToolReturnPart):
                lines.append(f'native tool {part.tool_name} returned: {part.model_response_str()}')
            elif isinstance(part, RetryFeedbackPart):
                lines.append(f'retry (output): {part.model_response()}')
            # TODO(v3): remove RetryPromptPart
            elif isinstance(part, RetryPromptPart):  # pyright: ignore[reportDeprecated]
                # A hand-built history passed as `message_history` can still hold one; the model
                # translates it only when the request goes out.
                lines.append(f'retry ({part.tool_name or "output"}): {part.model_response()}')
            elif isinstance(part, TextPart):
                if part.content:
                    lines.append(f'assistant: {part.content}')
            elif isinstance(part, ToolCallPart):
                lines.append(f'assistant called tool {part.tool_name} with {part.args_as_json_str()}')
            elif isinstance(part, NativeToolCallPart):
                lines.append(f'assistant called native tool {part.tool_name} with {part.args_as_json_str()}')
            elif isinstance(part, SpeechPart) and part.transcript:
                lines.append(f'{part.speaker}: {part.transcript}')
    return '\n'.join(lines)


def claim_request(ctx: RunContext[Any]) -> None:
    """Claim one request on the shared usage before awaiting the second model, or refuse it.

    Sibling tool calls are executed concurrently and a nested run records its spend only once the
    response is in, so without a claim every judgement in a parallel batch passes preflight against
    the same `usage.requests` and the batch bills past the run's `request_limit`. Claiming here is
    atomic against siblings: the hook chain reaches this point without yielding the event loop.
    A claim the budget cannot fit raises the same `UsageLimitExceeded` the nested run's own
    preflight would have raised. The caller releases the claim once the nested run returns.
    Mirrors `pydantic_ai_harness.trajectory_judge`'s launch claim, in the shape a synchronously
    awaited judgement needs.
    """
    if ctx.usage_limits is not None:
        ctx.usage_limits.check_before_request(ctx.usage)
    ctx.usage.requests += 1


def claim_offset_limits(limits: UsageLimits | None) -> UsageLimits | None:
    """The run's limits with `request_limit` raised by the one request `claim_request` took.

    The claim already occupies a slot in the shared `usage.requests`, so against the unadjusted
    limit the nested run's own preflight would count its in-flight request twice and refuse a call
    the budget affords.
    """
    if limits is None or limits.request_limit is None:
        return limits
    return replace(limits, request_limit=limits.request_limit + 1)


def claimed_request_limits(limits: UsageLimits | None) -> UsageLimits:
    """The run's limits for a one-request nested run whose request `claim_request` already admitted.

    The claim is the request check, so `request_limit` is lifted and only the token limits apply. Use this
    instead of `claim_offset_limits` where judgments start while others are finishing, as through a
    concurrency gate. There the shared `usage.requests` counts other judgments' claims and their nested
    runs' own requests at once, so a preflight against the offset limit refuses requests the claims
    admitted, and the `UsageLimitExceeded` fails the host run. Only for a judge that makes exactly one
    request, with no retries, since nothing else bounds its requests. With no limits it lifts the request
    limit too, as the nested run would otherwise apply the default one to the shared usage.
    """
    return replace(limits or UsageLimits(), request_limit=None)
