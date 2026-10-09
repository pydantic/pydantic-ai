"""Claude Code sessions: `projects/<project>/<session-id>.jsonl` under `CLAUDE_CONFIG_DIR` or `~/.claude`.

Each line is an entry linked to its parent by `parentUuid`. Rewinding starts a new branch from an
earlier entry, and compaction starts a new chain at its summary, so the conversation to continue is
the chain behind the last message, not every line in the file.
"""

from __future__ import annotations

import base64
import os
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
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

PROVIDER = 'anthropic'


def home() -> Path:
    """Claude Code's configuration directory."""
    return Path(os.environ.get('CLAUDE_CONFIG_DIR') or Path.home() / '.claude')


def files(root: Path) -> list[Path]:
    """Every main session transcript; sub-agent transcripts belong to their parent session."""
    return [path for path in (root / 'projects').glob('*/*.jsonl') if not path.name.startswith('agent-')]


def find(root: Path, native_id: str) -> Path | None:
    """The transcript for a session ID, in whichever project it was started."""
    return next((root / 'projects').glob(f'*/{native_id}.jsonl'), None)


def titles(root: Path) -> dict[str, str]:
    """Claude Code keeps no separate index of session names."""
    return {}


def header(path: Path) -> Header | None:
    """The directory and first prompt, preferring a summary title written before it."""
    cwd = prompt = summary = ''
    for entry in records(path, limit=HEAD_LINES):
        cwd = cwd or text(entry.get('cwd'))
        if entry.get('type') == 'summary':
            summary = summary or text(entry.get('summary'))
        elif entry.get('type') == 'user' and not entry.get('isMeta') and not entry.get('isSidechain'):
            prompt = prompt or _prompt_title(obj(entry.get('message')).get('content'))
        if cwd and prompt:
            return Header(native_id=path.stem, cwd=cwd, title=title_text(summary or prompt), named=bool(summary))
    return None


def messages(path: Path) -> list[ModelMessage]:
    """The conversation the session ended on, as Pydantic AI messages."""
    history = HistoryBuilder()
    for entry in _branch(list(records(path))):
        message = obj(entry.get('message'))
        at = timestamp(entry.get('timestamp'))
        if entry.get('type') == 'user':
            if not entry.get('isMeta'):
                _user(history, message.get('content'), at=at)
        elif text(message.get('model')) != '<synthetic>':
            # Synthetic replies are Claude Code's own notices, such as API errors, not model output.
            _assistant(history, message, at=at)
    return history.build()


def _prompt_title(content: object) -> str:
    """A typed prompt; command wrappers, caveats, and reminders start with a tag."""
    first = content if isinstance(content, str) else next((text(b.get('text')) for b in objects(content)), '')
    return '' if first.lstrip().startswith('<') else first


def _branch(entries: list[JsonObject]) -> list[JsonObject]:
    """User and assistant entries on the chain behind the last main-thread message, oldest first."""
    by_id = {text(entry.get('uuid')): entry for entry in entries}
    node = next(
        (e for e in reversed(entries) if e.get('type') in ('user', 'assistant') and not e.get('isSidechain')), None
    )
    chain: list[JsonObject] = []
    while node is not None:
        chain.append(node)
        # Popping each entry as it is visited also ends a malformed, cyclic chain.
        by_id.pop(text(node.get('uuid')), None)
        node = by_id.pop(text(node.get('parentUuid')), None)
    return [entry for entry in reversed(chain) if entry.get('type') in ('user', 'assistant')]


def _user(history: HistoryBuilder, content: object, *, at: datetime) -> None:
    if isinstance(content, str):
        history.prompt(content, at=at)
    for block in objects(content):
        kind = block.get('type')
        if kind == 'tool_result':
            result = block.get('content')
            output = result if isinstance(result, str) else '\n'.join(text(b.get('text')) for b in objects(result))
            history.result(text(block.get('tool_use_id')), output, at=at)
        elif kind == 'text':
            history.prompt(text(block.get('text')), at=at)
        elif kind == 'image' and (source := obj(block.get('source'))).get('type') == 'base64':
            data = base64.b64decode(text(source.get('data')))
            image = BinaryContent.narrow_type(BinaryContent(data, media_type=text(source.get('media_type'))))
            history.prompt([image], at=at)


def _assistant(history: HistoryBuilder, message: JsonObject, *, at: datetime) -> None:
    model = text(message.get('model')) or None
    for block in objects(message.get('content')):
        kind = block.get('type')
        part: ModelResponsePart
        if kind == 'text':
            part = TextPart(text(block.get('text')))
        elif kind == 'thinking':
            # Signed, so Claude can continue from it; other models receive the reasoning as text.
            signature = text(block.get('signature')) or None
            part = ThinkingPart(text(block.get('thinking')), signature=signature, provider_name=PROVIDER)
        elif kind == 'tool_use':
            part = ToolCallPart(text(block.get('name')), obj(block.get('input')), tool_call_id=text(block.get('id')))
        else:
            continue
        history.respond(part, at=at, model_name=model, provider_name=PROVIDER)
    if usage := obj(message.get('usage')):
        cache_read, cache_write = (
            number(usage.get('cache_read_input_tokens')),
            number(usage.get('cache_creation_input_tokens')),
        )
        history.usage(
            RequestUsage(
                input_tokens=number(usage.get('input_tokens')) + cache_read + cache_write,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                output_tokens=number(usage.get('output_tokens')),
            )
        )
