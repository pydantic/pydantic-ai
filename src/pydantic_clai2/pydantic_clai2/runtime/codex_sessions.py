"""Codex sessions: `sessions/YYYY/MM/DD/rollout-<time>-<session-id>.jsonl` under `CODEX_HOME` or `~/.codex`.

Each line is a `{type, payload}` record: `session_meta` first, then the `response_item`s the model saw,
`turn_context` with the model in use, and `compacted` where compaction replaced the history.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import BinaryContent, ImageUrl, ModelMessage, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.usage import RequestUsage
from pydantic_clai2.runtime.imported_history import (
    HEAD_LINES,
    Header,
    HistoryBuilder,
    JsonObject,
    number,
    obj,
    objects,
    records,
    text,
    timestamp,
    title_text,
)

PROVIDER = 'openai'

CONTEXT_PREFIXES = ('<environment_context>', '<user_instructions>', '<recommended_plugins>', '# AGENTS.md instructions')
"""User messages Codex adds itself: its environment and the project's instructions, which CLAI supplies its own way."""


def home() -> Path:
    """Codex's home directory."""
    return Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex')


def files(root: Path) -> list[Path]:
    """Every session transcript."""
    return list((root / 'sessions').rglob('rollout-*.jsonl'))


def find(root: Path, native_id: str) -> Path | None:
    """The transcript for a session ID, which ends its file name."""
    return next((root / 'sessions').rglob(f'rollout-*-{native_id}.jsonl'), None)


def titles(root: Path) -> dict[str, str]:
    """Session names from Codex's index, the latest name for each session winning."""
    try:
        return {
            text(entry.get('id')): text(entry.get('thread_name')) for entry in records(root / 'session_index.jsonl')
        }
    except OSError:
        # Names are optional: without the index, sessions keep their first prompt as the title.
        return {}


def header(path: Path) -> Header | None:
    """The session ID and directory from its metadata, and its first typed prompt."""
    meta: JsonObject = {}
    prompt = ''
    for record in records(path, limit=HEAD_LINES):
        payload = obj(record.get('payload'))
        if record.get('type') == 'session_meta':
            meta = meta or payload
        elif record.get('type') == 'response_item' and payload.get('role') == 'user':
            texts = (text(block.get('text')) for block in objects(payload.get('content')))
            prompt = prompt or next((t for t in texts if _typed(t)), '')
        if meta and prompt:
            return Header(native_id=text(meta.get('id')), cwd=text(meta.get('cwd')), title=title_text(prompt))
    return None


def messages(path: Path) -> list[ModelMessage]:
    """The history the session's model saw last, as Pydantic AI messages."""
    history = HistoryBuilder()
    model: str | None = None
    for record in records(path):
        kind = record.get('type')
        payload = obj(record.get('payload'))
        at = timestamp(record.get('timestamp'))
        if kind == 'turn_context':
            model = text(payload.get('model')) or model
        elif kind == 'response_item':
            _item(history, payload, at=at, model=model)
        elif kind == 'compacted':
            history.reset()
            for item in objects(payload.get('replacement_history')):
                _item(history, item, at=at, model=model)
            # Older versions record only the summary that replaced the history.
            history.prompt(text(payload.get('message')), at=at)
        elif kind == 'event_msg' and payload.get('type') == 'token_count':
            if usage := obj(obj(payload.get('info')).get('last_token_usage')):
                history.usage(
                    RequestUsage(
                        input_tokens=number(usage.get('input_tokens')),
                        cache_read_tokens=number(usage.get('cached_input_tokens')),
                        output_tokens=number(usage.get('output_tokens')),
                    )
                )
    return history.build()


def _typed(value: str) -> bool:
    """Whether user text was typed, rather than context Codex adds itself."""
    return bool(value) and not value.lstrip().startswith(CONTEXT_PREFIXES)


def _item(history: HistoryBuilder, item: JsonObject, *, at: datetime, model: str | None) -> None:
    kind = item.get('type')
    role = item.get('role')
    if kind == 'message' and role == 'user':
        # In order, so text keeps its place around the images it refers to.
        for block in objects(item.get('content')):
            if block.get('type') == 'input_text' and _typed(prompt := text(block.get('text'))):
                history.prompt(prompt, at=at)
            elif block.get('type') == 'input_image' and (url := text(block.get('image_url'))):
                history.prompt([BinaryContent.from_data_uri(url) if url.startswith('data:') else ImageUrl(url)], at=at)
    elif kind == 'message' and role == 'assistant':
        for block in objects(item.get('content')):
            if block.get('type') == 'output_text':
                history.respond(TextPart(text(block.get('text'))), at=at, model_name=model, provider_name=PROVIDER)
    elif kind == 'reasoning':
        # Only the summary is readable; the full reasoning is encrypted for Codex's own account.
        if summary := '\n\n'.join(text(part.get('text')) for part in objects(item.get('summary'))):
            history.respond(ThinkingPart(summary), at=at, model_name=model, provider_name=PROVIDER)
    elif kind in ('function_call', 'custom_tool_call'):
        args = text(item.get('arguments')) if kind == 'function_call' else {'input': text(item.get('input'))}
        call = ToolCallPart(text(item.get('name')), args, tool_call_id=text(item.get('call_id')))
        history.respond(call, at=at, model_name=model, provider_name=PROVIDER)
    elif kind in ('function_call_output', 'custom_tool_call_output'):
        output = item.get('output')
        content = output if isinstance(output, str) else '\n'.join(text(b.get('text')) for b in objects(output))
        history.result(text(item.get('call_id')), content, at=at)
