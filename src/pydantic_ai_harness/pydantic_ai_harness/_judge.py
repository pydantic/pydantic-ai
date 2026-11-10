"""Shared plumbing for capabilities that ask a second model to judge an agent's tool calls or run.

`ToolCallJudge` and `ApprovalReviewer` build the same prompt, read the same provider confidence, and
claim their request on the run's usage the same way. They differ only in what an answer does.
`TrajectoryJudge` shares the transcript line rendering and the claim-offset limits.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import Any, Literal, TypeAlias

from opentelemetry.trace import Span
from pydantic import TypeAdapter, ValidationError

from pydantic_ai._framing import neutralize_tags
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequestPart,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryFeedbackPart,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]  # TODO(v3): remove RetryPromptPart
    SpeechPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness._message_text import CHARS_PER_TOKEN, user_prompt_text

JudgeAnswer: TypeAlias = Literal['yes', 'no', 'unsure']
"""The answer to a yes/no risk question about one tool call."""

SerializableToolSelector: TypeAlias = Literal['all'] | Sequence[str] | dict[str, Any]
"""The `ToolSelector` forms an agent spec can express. A predicate is code-only."""

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
    max_chars = conversation_window * CHARS_PER_TOKEN
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]
    return f'<conversation>\n{transcript}\n</conversation>\n\n{tool_call}'


def _neutralize_tags(text: str) -> str:
    r"""Escape the `<` of every opening or closing framework tag as `&lt;`, and leave everything else.

    That covers `tool_call` and `conversation`, including a closer with backslashes before its slash
    (`<\/tool_call>`), and every Pydantic AI framing tag.
    """
    return _BLOCK_TAG_OPENER.sub('&lt;', neutralize_tags(text))


# Same transcript renderer as `pydantic_ai_harness.trajectory_judge`. Not shared with it because
# that renderer also tags the run id and splits the transcript; fold them together if a shared
# transcript helper lands.
def _render_transcript(messages: Sequence[ModelMessage]) -> str:
    """Render the conversation as model-readable lines.

    Instructions, system prompts, and thinking parts are omitted: the second model is shown what
    was asked, said, called, and returned, not the agent's configuration or private reasoning.
    """
    lines: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                text = user_prompt_text(part)
                if text:
                    lines.append(f'user: {text}')
            elif (line := render_part(part)) is not None:
                lines.append(line)
    return '\n'.join(lines)


def render_part(part: ModelRequestPart | ModelResponsePart) -> str | None:
    """Render a part other than a user prompt as one transcript line, or `None` for a part the judge doesn't see.

    User prompts are left to the caller, which decides where their lines go.
    """
    if isinstance(part, ToolReturnPart):
        if part.outcome == 'retried':
            return f'retry ({part.tool_name}): {part.model_response_str(wrap_if_error=False)}'
        return f'tool {part.tool_name} returned: {part.model_response_str()}'
    if isinstance(part, NativeToolReturnPart):
        return f'native tool {part.tool_name} returned: {part.model_response_str()}'
    if isinstance(part, RetryFeedbackPart):
        return f'retry (output): {part.model_response()}'
    # TODO(v3): remove RetryPromptPart
    if isinstance(part, RetryPromptPart):  # pyright: ignore[reportDeprecated]
        # A hand-built history passed as `message_history` can still hold one; the model
        # translates it only when the request goes out.
        return f'retry ({part.tool_name or "output"}): {part.model_response()}'
    if isinstance(part, TextPart):
        return f'assistant: {part.content}' if part.content else None
    if isinstance(part, ToolCallPart):
        return f'assistant called tool {part.tool_name} with {part.args_as_json_str()}'
    if isinstance(part, NativeToolCallPart):
        return f'assistant called native tool {part.tool_name} with {part.args_as_json_str()}'
    if isinstance(part, SpeechPart) and part.transcript:
        return f'{part.speaker}: {part.transcript}'
    return None


def claim_request(ctx: RunContext[Any]) -> None:
    """Claim one request on the shared usage before awaiting the second model, or refuse it.

    A nested run records its spend only once the response is in, so without a claim every judgement
    in a parallel batch of tool calls passes preflight against the same `usage.requests` and the
    batch bills past the run's `request_limit`. The hook chain reaches this point without yielding
    the event loop, so the check and the claim are atomic against siblings. A claim the budget
    cannot fit raises the same `UsageLimitExceeded` the nested run's own preflight would have
    raised. The caller releases the claim once the nested run returns.
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


def judge_span(
    ctx: RunContext[Any], name: str, prefix: str, call: ToolCallPart, args: dict[str, Any]
) -> AbstractContextManager[Span]:
    """Open the span for judging one call; its arguments are recorded only when the run traces content."""
    attributes = {f'{prefix}.tool': call.tool_name, f'{prefix}.tool_call_id': call.tool_call_id}
    if ctx.trace_include_content:
        attributes[f'{prefix}.arguments'] = json.dumps(args, default=str)
    return ctx.tracer.start_as_current_span(name, attributes=attributes)


async def ask_claimed(
    ctx: RunContext[Any],
    span: Span,
    prefix: str,
    ask: Callable[[], Awaitable[tuple[JudgeAnswer, float | None]]],
) -> tuple[JudgeAnswer | None, float | None]:
    """Await `ask` under a `claim_request` claim and record its outcome on `span`.

    Returns the answer and confidence, or `(None, None)` when the claim or the model failed.
    """
    try:
        claim_request(ctx)
        try:
            answer, confidence = await ask()
        finally:
            ctx.usage.requests -= 1
    except Exception as error:
        if span.is_recording():
            span.set_attribute(f'{prefix}.model_result', 'error')
            span.set_attribute(f'{prefix}.error.type', type(error).__name__)
        return None, None
    if span.is_recording():
        span.set_attribute(f'{prefix}.model_result', answer)
    return answer, confidence
