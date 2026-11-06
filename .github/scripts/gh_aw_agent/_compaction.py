"""History compaction: dedupe and trim first, then summarise the middle.

`ProcessHistory` re-applies the compacted list on every request. The workflow
prompt is the run's first user message, so `preserve_prompt=True` keeps it ahead
of the tail on every path: a history that starts with an assistant turn is
rejected by Z.AI (HTTP 400, code 1214).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Sequence

from opentelemetry.trace import Status, StatusCode, get_tracer

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import (
    CapabilityLoadDeltaPart,
    InstructionBaselinePart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponsePart,
    ModelSettingsDeltaPart,
    NativeToolCallPart,
    NativeToolSearchCallPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

MessagePart = ModelRequestPart | ModelResponsePart

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

COMPACTION_TRIGGER_CHARS = 400_000
COMPACTION_KEEP_RECENT = 10
TOOL_RESULT_HEAD_TAIL_CHARS = 4_000
TOOL_RESULT_TRIM_THRESHOLD = 10_000
COMPACTION_TRANSCRIPT_MAX_CHARS = 80_000
COMPACTION_TIMEOUT_SECS = 120
SYNTHETIC_SUMMARY_TAG = '[compacted history]'

COMPACTION_SUMMARY_INSTRUCTIONS = (
    'Summarise the agent transcript below for resumption in a fresh '
    'context window. Produce a structured brief, not free prose. Use this '
    'exact section layout, omitting any section that is empty:\n\n'
    '## Goal\n'
    'One short paragraph: what the agent was asked to do.\n\n'
    '## Files inspected\n'
    '- `<full/path>`: one-line note on what was found there.\n\n'
    '## Commands run\n'
    '- `<command>`: outcome in one line.\n\n'
    '## Errors encountered\n'
    'Verbatim error messages or unexpected behaviour, with the file or '
    'command that triggered each.\n\n'
    '## Decisions and approaches\n'
    '- Concrete decisions with reasoning. Include approaches already tried '
    'that did **not** work, so they are not re-attempted.\n\n'
    '## Open questions\n'
    '- Anything still unresolved.\n\n'
    '## Next step\n'
    'The single most likely next action.\n\n'
    'Preserve specifics (paths, identifiers, exact strings) over prose. '
    'Respond with text only — do not call any tools.'
)


def _part_text(part: MessagePart) -> str:
    if isinstance(part, (ToolCallPart, NativeToolCallPart, NativeToolSearchCallPart)):
        return f'{part.tool_name}({part.args_as_dict()!r})'
    if isinstance(part, ToolAvailabilityDeltaPart):
        return f'tool availability changed: tools_added={part.tools_added!r}'
    if isinstance(part, InstructionBaselinePart):
        return ''
    if isinstance(part, ModelSettingsDeltaPart):
        return f'model settings changed: {part.settings!r}'
    if isinstance(part, CapabilityLoadDeltaPart):
        return f'loaded capabilities changed: loaded={part.loaded!r}'
    return str(part.content)


def _render_messages_for_summary(messages: list[ModelMessage]) -> str:
    out: list[str] = []
    for message in messages:
        kind = 'user' if isinstance(message, ModelRequest) else 'assistant'
        for part in message.parts:
            out.append(f'[{kind}/{type(part).__name__}] {_part_text(part)[:1500]}')
    return '\n'.join(out)


def _history_size_chars(messages: list[ModelMessage]) -> int:
    return sum(len(_part_text(part)) for message in messages for part in message.parts)


def _head_tail(text: str, side: int) -> str:
    skipped = len(text) - side * 2
    return f'{text[:side]}\n…[trimmed {skipped} chars]…\n{text[-side:]}'


def _superseded_read_calls(messages: list[ModelMessage]) -> tuple[set[str], dict[str, str]]:
    """For each `read_file` call, key on (path, offset, limit); older calls with the same key are superseded."""
    label_by_call_id: dict[str, str] = {}
    latest_for_args: dict[tuple[str, object, object], str] = {}
    superseded: set[str] = set()
    for message in messages:
        for part in message.parts:
            if not (isinstance(part, ToolCallPart) and part.tool_name == 'read_file'):
                continue
            args = part.args_as_dict()
            if not isinstance(args, dict):
                continue
            path = args.get('path')
            if not isinstance(path, str):
                continue
            offset, limit = args.get('offset'), args.get('limit')
            label = path if not offset and limit is None else f'{path}[offset={offset!r}, limit={limit!r}]'
            label_by_call_id[part.tool_call_id] = label
            key = (path, offset, limit)
            prior = latest_for_args.get(key)
            if prior is not None:
                superseded.add(prior)
            latest_for_args[key] = part.tool_call_id
    return superseded, label_by_call_id


def _trim_tool_results(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Dedupe re-reads of the same file slice and head/tail-truncate oversized older tool returns."""
    if len(messages) <= COMPACTION_KEEP_RECENT:
        return messages
    superseded, label_by_call_id = _superseded_read_calls(messages)
    tail_start = len(messages) - COMPACTION_KEEP_RECENT
    out: list[ModelMessage] = []
    for index, message in enumerate(messages):
        if index >= tail_start:
            out.append(message)
            continue
        new_parts = [
            _rewrite_tool_result(part, superseded, label_by_call_id) if isinstance(part, ToolReturnPart) else part
            for part in message.parts
        ]
        out.append(dataclasses.replace(message, parts=new_parts) if new_parts != list(message.parts) else message)
    return out


def _rewrite_tool_result(
    part: ToolReturnPart, superseded: set[str], label_by_call_id: dict[str, str]
) -> ToolReturnPart:
    if part.tool_call_id in superseded:
        return dataclasses.replace(
            part, content=f'[superseded read: {label_by_call_id[part.tool_call_id]} — see later read with same args]'
        )
    content = str(part.content)
    if len(content) > TOOL_RESULT_TRIM_THRESHOLD:
        return dataclasses.replace(part, content=_head_tail(content, TOOL_RESULT_HEAD_TAIL_CHARS))
    return part


async def compact_history(
    ctx: RunContext[object], messages: list[ModelMessage], *, preserve_prompt: bool = True
) -> list[ModelMessage]:
    """Cheap trim first; LLM-summarise the middle as fallback if still over budget."""
    if len(messages) <= COMPACTION_KEEP_RECENT:
        return messages
    prompt = messages[0] if preserve_prompt else None
    trimmed = _trim_tool_results(messages)
    size = _history_size_chars(trimmed)
    if size < COMPACTION_TRIGGER_CHARS:
        return trimmed
    middle = trimmed[:-COMPACTION_KEEP_RECENT]
    tail = trimmed[-COMPACTION_KEEP_RECENT:]
    transcript = _render_messages_for_summary(middle)
    model = ctx.model
    if not isinstance(model, Model):
        logger.warning('compaction skipped: run model is not a concrete Model')
        return trimmed
    sub_usage = RunUsage()
    with _tracer.start_as_current_span('compact history') as span:
        try:
            result = await asyncio.wait_for(
                # `ctx.model` is an `AbstractModel`; `isinstance` proves a request-response model, but the
                # client generic cannot be inferred here and this repo forbids a cast.
                Agent[object, str](
                    model,  # pyright: ignore[reportUnknownArgumentType]
                    name='history-summariser',
                    instructions=COMPACTION_SUMMARY_INSTRUCTIONS,
                ).run(
                    f'Transcript to summarise:\n\n{transcript[:COMPACTION_TRANSCRIPT_MAX_CHARS]}',
                    usage_limits=UsageLimits(request_limit=2),
                    usage=sub_usage,
                ),
                timeout=COMPACTION_TIMEOUT_SECS,
            )
            summary = str(result.output or '').strip() or '(empty summary)'
        except Exception as exc:
            ctx.usage.incr(sub_usage)
            detail = f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__
            logger.warning('compaction summarisation failed (%s); falling back', detail, exc_info=True)
            span.set_attribute('gh_aw.compaction.outcome', 'timed_out' if isinstance(exc, TimeoutError) else 'failed')
            span.set_status(Status(StatusCode.ERROR, detail))
            return [*_prompt_prefix(prompt), *tail]
        ctx.usage.incr(sub_usage)
        middle_size = _history_size_chars(middle)
        if len(summary) >= middle_size:
            span.set_attribute('gh_aw.compaction.outcome', 'discarded')
            return [*_prompt_prefix(prompt), *tail]
        span.set_attribute('gh_aw.compaction.outcome', 'done')
    synthetic = ModelRequest(parts=[UserPromptPart(content=f'{SYNTHETIC_SUMMARY_TAG}\n{summary}')])
    return [*_prompt_prefix(prompt), synthetic, *tail]


def _prompt_prefix(prompt: ModelMessage | None) -> Sequence[ModelMessage]:
    return () if prompt is None else (prompt,)


def require_safe_output(output: str) -> str:
    """Send the model back to work when it ends the run before emitting safe output."""
    from ._budgets import in_delegate
    from ._config import is_review_run
    from ._safe_outputs import safe_output_status

    if in_delegate():
        return output
    output_pending, missing = safe_output_status()
    if missing:
        missing_ids = ', '.join(str(number) for number in missing)
        raise ModelRetry(
            'Record an attention decision for each missing candidate ID: '
            f'{missing_ids}. Keep the decisions already recorded and call '
            '`record_attention_decision` only for these missing IDs.'
        )
    if not output_pending:
        return output
    if is_review_run():
        raise ModelRetry(
            'Submit the formal pull-request review with `submit_pull_request_review`. '
            'A noop, inline comment, or missing-data/incomplete signal does not finish a review. '
            'If coverage is incomplete, submit a COMMENT review explaining the missing coverage; do not approve.'
        )
    raise ModelRetry(
        'You ended your turn without emitting a safe output, so nothing has been posted. '
        'Continue the task and finish by calling the safe-output tool it ends with, '
        'or `noop` if there is nothing to report.'
    )


__all__ = ['compact_history', 'require_safe_output']
