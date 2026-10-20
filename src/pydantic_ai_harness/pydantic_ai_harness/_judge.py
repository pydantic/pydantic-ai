"""Shared plumbing for capabilities that ask a second model one question about a tool call.

`ToolCallJudge` and `ApprovalReviewer` build the same prompt, read the same provider confidence, and
claim their request on the run's usage the same way. They differ only in what an answer does.
"""

from __future__ import annotations

import json
import re
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pydantic import TypeAdapter, ValidationError

from pydantic_ai.messages import (
    ModelMessage,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SpeechPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import RunContext

if TYPE_CHECKING:
    from pydantic_ai.usage import UsageLimits

JudgeAnswer: TypeAlias = Literal['yes', 'no', 'unsure']
"""The answer to a yes/no risk question about one tool call."""

SerializableToolSelector: TypeAlias = Literal['all'] | Sequence[str] | dict[str, Any]
"""The `ToolSelector` forms an agent spec can express. A predicate is code-only."""

# Same ~4 characters-per-token heuristic, transcript renderer, and window clamp as
# `pydantic_ai_harness.trajectory_judge`. Not shared with it because that renderer also tags the run
# id and splits the transcript; fold them together if a shared transcript helper lands.
_CHARS_PER_TOKEN = 4

# TODO: switch to `pydantic_ai._framing.neutralize_closing_tag` once it lands
# (https://github.com/pydantic/pydantic-ai/pull/10547), adding `tool_call` and `conversation` to its tags.
# Until then this matches its semantics: only the `<` of a closing tag is escaped.
_CLOSING_BLOCK_TAG_OPENER = re.compile(r'<(?=\s*/\s*(?:tool_call|conversation)(?=[\s/>]))', re.IGNORECASE)

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
    and `&` in a command reach the second model as they would reach a person. Only the `<` of a
    closing `tool_call` or `conversation` tag is escaped, so the call's arguments or a tool result
    cannot end their block early and continue as something else.
    """
    payload = json.dumps({'tool_name': tool_name, 'arguments': args}, default=str, ensure_ascii=False)
    tool_call = f'<tool_call>\n{_neutralize_closing_tags(payload)}\n</tool_call>'
    if not include_conversation:
        return tool_call
    transcript = _render_transcript(messages)
    max_chars = conversation_window * _CHARS_PER_TOKEN
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]
    return f'<conversation>\n{_neutralize_closing_tags(transcript)}\n</conversation>\n\n{tool_call}'


def _neutralize_closing_tags(text: str) -> str:
    """Escape the `<` of every closing `tool_call` or `conversation` tag as `&lt;`, and leave everything else."""
    return _CLOSING_BLOCK_TAG_OPENER.sub('&lt;', text)


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
                lines.append(f'tool {part.tool_name} returned: {part.model_response_str()}')
            elif isinstance(part, NativeToolReturnPart):
                lines.append(f'native tool {part.tool_name} returned: {part.model_response_str()}')
            elif isinstance(part, RetryPromptPart):
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


@contextmanager
def claimed_request(ctx: RunContext[Any]) -> Generator[None]:
    """Hold one request on the shared usage while the second model is awaited, or refuse it.

    Sibling tool calls are executed concurrently and a nested run records its spend only once the
    response is in, so without a claim every judgement in a parallel batch passes preflight against
    the same `usage.requests` and the batch bills past the run's `request_limit`. Claiming here is
    atomic against siblings: the hook chain reaches this point without yielding the event loop.
    A claim the budget cannot fit raises the same `UsageLimitExceeded` the nested run's own
    preflight would have raised. The claim is released when the block exits, once the nested run
    has recorded its own request. Mirrors `pydantic_ai_harness.trajectory_judge`'s launch claim, in
    the shape a synchronously awaited judgement needs.
    """
    if ctx.usage_limits is not None:
        ctx.usage_limits.check_before_request(ctx.usage)
    ctx.usage.requests += 1
    try:
        yield
    finally:
        ctx.usage.requests -= 1


def claim_offset_limits(limits: UsageLimits | None) -> UsageLimits | None:
    """The run's limits with `request_limit` raised by the one request `claimed_request` holds.

    The claim already occupies a slot in the shared `usage.requests`, so against the unadjusted
    limit the nested run's own preflight would count its in-flight request twice and refuse a call
    the budget affords.
    """
    if limits is None or limits.request_limit is None:
        return limits
    return replace(limits, request_limit=limits.request_limit + 1)
