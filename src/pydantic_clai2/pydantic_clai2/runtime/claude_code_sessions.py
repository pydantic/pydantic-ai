"""Claude Code sessions: `projects/<project>/<session-id>.jsonl` under `CLAUDE_CONFIG_DIR` or `~/.claude`.

Each line is an entry linked to its parent by `parentUuid`. Rewinding starts a new branch from an
earlier entry, and compaction starts a new chain at its summary, so the conversation to continue is
the chain behind the last message, not every line in the file. Parallel tool calls fork that chain:
each result hangs off its own call, and only the last result continues it.

Besides the user's and the model's messages, the chain holds what Claude Code's harness added to the
conversation: `attachment` entries with the text it `rendered` for the model (files, reminders, and
messages queued while the model worked), and `isMeta` user messages (skill bodies, command output,
sub-agent reports). Each sub-agent keeps its own transcript in `<session-id>/subagents/`.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid5

from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnContent,
    UserContent,
)
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.subagents import DelegationTask
from pydantic_clai2.runtime.imported_history import (
    HEAD_LINES,
    Header,
    HistoryBuilder,
    JsonObject,
    imported_context,
    number,
    obj,
    objects,
    records,
    tail_records,
    text,
    timestamp,
    title_text,
)

PROVIDER = 'anthropic'
SOURCE = 'claude-code'
"""The `source` of the context imported from Claude Code."""

_TASK_NAMESPACE = UUID('0b6f3c52-8d1e-4c0a-9f7b-2e5d4a1c6b93')
"""Derives a CLAI task ID from a Claude Code session and sub-agent ID, so importing again finds the same task."""

_EPHEMERAL = frozenset({'total_tokens_reminder', 'silent_turn_reminder'})
"""Attachments that only nudged the request they came with: the tokens left, a prompt to update the user."""

_HAND_BACK = re.compile(
    r'<agent-message from="([^"]+)">\n\[Subagent hand-back\].*?The report follows:\n(.*)', re.DOTALL
)
"""A background sub-agent's final report, which Claude Code indents by two spaces under its own framing."""

_OPENING = (
    '{subject} was imported from Claude Code session {session_id}. Up to the note at its end, it ran in '
    "Claude Code's harness: the tool calls in it (such as Bash, Read, Edit, Write, Agent, SendMessage, Skill and "
    "ToolSearch), its `<system-reminder>` blocks and the other context it added are Claude Code's, and its "
    "sub-agents' reports appear as `<task_output>` blocks."
)
_SELF = 'self'
"""The agent name CLAI runs Claude Code's `general-purpose` sub-agents under: itself."""

_STOPPED = 'Stopped in Claude Code before it reported. Its history is saved: resume it to continue.'
"""The output of a sub-agent that never finished in Claude Code."""
_CLOSING = (
    'The conversation above was imported from Claude Code, and continues here in CLAI, a different harness. The '
    "tools, reminders and instructions above came from Claude Code's harness and no longer apply: use the tools and "
    'instructions you have now, whose names and arguments may differ.'
)


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
    """Claude Code keeps no separate index of session names: each transcript holds its own, read by `header`."""
    return {}


def header(path: Path) -> Header | None:
    """The directory and first prompt, titled with the session's name when it has one."""
    cwd = prompt = summary = ''
    for entry in records(path, limit=HEAD_LINES):
        cwd = cwd or text(entry.get('cwd'))
        if entry.get('type') == 'summary':
            # Older versions wrote a summary title ahead of the conversation.
            summary = summary or text(entry.get('summary'))
        elif entry.get('type') == 'user' and not entry.get('isMeta') and not entry.get('isSidechain'):
            prompt = prompt or _prompt_title(obj(entry.get('message')).get('content'))
        if cwd and prompt:
            name = _name(path) or summary
            return Header(native_id=path.stem, cwd=cwd, title=title_text(name or prompt), named=bool(name))
    return None


def messages(path: Path) -> list[ModelMessage]:
    """The conversation the session ended on, as Pydantic AI messages, framed by notes on the change of harness."""
    # Task IDs derive from the session alone, and a report names no conversation, so none is needed here.
    agents = _agents(path, conversation_id='')
    lines = [
        f'- Claude Code agent {agent_id} is task {task.id} ({_agent_type(task)}: {task.prompt})'
        for agent_id, task in agents.items()
    ]
    closing = _CLOSING + (
        "\n\nClaude Code's sub-agents were imported as tasks, by task ID. One whose report is not above stopped "
        'before it reported:\n' + '\n'.join(lines)
        if lines
        else ''
    )
    opening = _OPENING.format(subject='This conversation', session_id=path.stem)
    return _framed(_history(list(records(path)), agents), opening=opening, closing=closing)


def tasks(path: Path, conversation_id: str) -> list[DelegationTask]:
    """The session's sub-agents as finished CLAI tasks of `conversation_id`, with their own histories.

    One that reported is delivered already, in the parent's history; one that never finished is `cancelled`.
    """
    agents = _agents(path, conversation_id)
    found: list[DelegationTask] = []
    for agent_id, task in agents.items():
        entries = list(records(_agent_path(path, agent_id)))
        opening = _OPENING.format(subject=f'This sub-agent conversation ({agent_id})', session_id=path.stem)
        history = _history(entries, agents, main=False)
        output = _output(history)
        times = [timestamp(entry.get('timestamp')) for entry in entries if entry.get('timestamp')]
        prompt = next(
            (
                _prompt_text(obj(entry.get('message')).get('content'))
                for entry in entries
                if entry.get('type') == 'user' and not entry.get('isMeta')
            ),
            task.prompt,
        )
        if output is None:
            # Claude Code stopped it before it reported, when the session ended or was imported mid-run. Still
            # delivered, so opening the session starts no turn of its own; the closing note tells the parent.
            task = replace(task, outcome='cancelled')
        found.append(
            replace(
                task,
                prompt=prompt or task.prompt,
                messages=_framed(history, opening=opening, closing=_CLOSING),
                output=_STOPPED if output is None else output,
                started_at=times[0].timestamp() if times else task.started_at,
                finished_at=times[-1].timestamp() if times else None,
            )
        )
    return found


def _name(path: Path) -> str:
    """The name the user gave the session, else the one Claude Code generated, from the end of the transcript."""
    custom = generated = ''
    for entry in tail_records(path, containing=b'-title"'):
        if entry.get('type') == 'custom-title':
            custom = text(entry.get('customTitle')) or custom
        elif entry.get('type') == 'ai-title':
            generated = text(entry.get('aiTitle')) or generated
    return custom or generated


def _prompt_title(content: object) -> str:
    """A typed prompt; command wrappers, caveats, and reminders start with a tag."""
    first = content if isinstance(content, str) else next((text(b.get('text')) for b in objects(content)), '')
    return '' if first.lstrip().startswith('<') else first


def _prompt_text(content: object) -> str:
    return content if isinstance(content, str) else '\n'.join(text(b.get('text')) for b in objects(content))


def _agent_path(path: Path, agent_id: str) -> Path:
    return path.with_suffix('') / 'subagents' / f'agent-{agent_id}.jsonl'


def _task_id(session_id: str, agent_id: str) -> str:
    return uuid5(_TASK_NAMESPACE, f'{session_id}:{agent_id}').hex


def _agents(path: Path, conversation_id: str) -> dict[str, DelegationTask]:
    """The session's sub-agents by their Claude Code ID, as tasks without their histories, read from their metadata."""
    agents: dict[str, DelegationTask] = {}
    for meta_path in sorted((path.with_suffix('') / 'subagents').glob('agent-*.meta.json')):
        agent_id = meta_path.name.removeprefix('agent-').removesuffix('.meta.json')
        if not _agent_path(path, agent_id).is_file():
            continue
        try:
            meta = obj(json.loads(meta_path.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            meta = {}
        agent_type = text(meta.get('agentType')) or 'general-purpose'
        parent = text(meta.get('parentAgentId'))
        agents[agent_id] = DelegationTask(
            id=_task_id(path.stem, agent_id),
            run_id=uuid5(_TASK_NAMESPACE, f'{path.stem}:{agent_id}:run').hex,
            # CLAI runs `general-purpose` as itself, under the name its tasks are saved and resumed by.
            agent_name=_SELF if agent_type == 'general-purpose' else agent_type,
            prompt=text(meta.get('description')),
            conversation_id=conversation_id,
            parent_id=_task_id(path.stem, parent) if parent else None,
            background=True,
            # Claude Code's other agent types have no CLAI counterpart to resume them with.
            resumable=agent_type == 'general-purpose',
            status='finished',
            outcome='ok',
            delivered=True,
        )
    return agents


def _agent_type(task: DelegationTask) -> str:
    return 'general-purpose' if task.agent_name == _SELF else task.agent_name


def _output(history: list[ModelMessage]) -> str | None:
    """A sub-agent's report: what it handed back, else its final answer, or `None` when it never finished."""
    responses = [message for message in history if isinstance(message, ModelResponse)]
    for part in reversed(responses[-1].parts if responses else []):
        if isinstance(part, ToolCallPart) and part.tool_name == 'SubagentHandback':
            return text(part.args_as_dict().get('message'))
    # A final answer ends the history; anything after it is work the sub-agent had yet to answer.
    if responses and history[-1] is responses[-1] and not responses[-1].tool_calls:
        return '\n\n'.join(part.content for part in responses[-1].parts if isinstance(part, TextPart))
    return None


def _history(entries: list[JsonObject], agents: dict[str, DelegationTask], *, main: bool = True) -> list[ModelMessage]:
    history = HistoryBuilder()
    reports: dict[str, int] = {}
    for entry in _conversation(entries, main=main):
        message = obj(entry.get('message'))
        at = timestamp(entry.get('timestamp'))
        kind = entry.get('type')
        if kind == 'user' and (entry.get('isMeta') or entry.get('isCompactSummary')):
            _meta(history, entry, agents, reports, at=at)
        elif kind == 'user':
            _user(history, message.get('content'), at=at)
        elif kind == 'assistant' and text(message.get('model')) != '<synthetic>':
            # Synthetic replies are Claude Code's own notices, such as API errors, not model output.
            _assistant(history, message, at=at)
        elif kind == 'attachment':
            _attachment(history, entry, at=at)
    return history.build()


def _framed(history: list[ModelMessage], *, opening: str, closing: str) -> list[ModelMessage]:
    """`history` between the notes on the change of harness, or nothing when there is no history."""
    built = list(history)
    if not built:
        return []
    first = built[0]
    start = imported_context(opening, source=SOURCE, kind='import_start', at=first.timestamp or timestamp(None))
    end = imported_context(closing, source=SOURCE, kind='import_end', at=built[-1].timestamp or timestamp(None))
    if isinstance(first, ModelRequest):
        built[0] = replace(first, parts=[start, *first.parts])
    else:
        built.insert(0, ModelRequest(parts=[start], timestamp=start.timestamp))
    if isinstance(last := built[-1], ModelRequest):
        built[-1] = replace(last, parts=[*last.parts, end])
    else:
        built.append(ModelRequest(parts=[end], timestamp=end.timestamp))
    return built


def _conversation(entries: list[JsonObject], *, main: bool) -> list[JsonObject]:
    """The entries behind the last message, with the results of its parallel tool calls, in file order.

    In a main transcript the last message is the last one off a sidechain; a sub-agent's is all sidechain.
    """
    by_id = {text(entry.get('uuid')): entry for entry in entries}
    node = next(
        (
            e
            for e in reversed(entries)
            # An attachment can come last: a message queued while the model worked, which no reply has answered yet.
            if e.get('type') in ('user', 'assistant', 'attachment') and not (main and e.get('isSidechain'))
        ),
        None,
    )
    chain: set[str] = set()
    while node is not None:
        chain.add(text(node.get('uuid')))
        # Popping each entry as it is visited also ends a malformed, cyclic chain.
        by_id.pop(text(node.get('uuid')), None)
        node = by_id.pop(text(node.get('parentUuid')), None)
    on_chain = [entry for entry in entries if text(entry.get('uuid')) in chain]
    calls = {
        text(block.get('id'))
        for entry in on_chain
        if entry.get('type') == 'assistant'
        for block in objects(obj(entry.get('message')).get('content'))
        if block.get('type') == 'tool_use'
    }
    answered = {call_id for entry in on_chain for call_id in _result_ids(entry)}
    kept = set(chain)
    for entry in entries:
        ids = _result_ids(entry)
        if text(entry.get('uuid')) not in chain and text(entry.get('parentUuid')) in chain and ids:
            if ids <= calls - answered:
                kept.add(text(entry.get('uuid')))
                answered |= ids
    return [entry for entry in entries if text(entry.get('uuid')) in kept]


def _result_ids(entry: JsonObject) -> set[str]:
    if entry.get('type') != 'user':
        return set()
    content = obj(entry.get('message')).get('content')
    return {text(block.get('tool_use_id')) for block in objects(content) if block.get('type') == 'tool_result'}


def _content(block: JsonObject) -> UserContent | None:
    """A text, image or document block of a message or tool result."""
    kind = block.get('type')
    source = obj(block.get('source'))
    if kind == 'text':
        return text(block.get('text')) or None
    if kind == 'tool_reference':
        # A tool search's match, which the API expands into the tool's definition.
        return f'Tool loaded: {text(block.get("tool_name"))}'
    if kind not in ('image', 'document'):
        return None
    if source.get('type') == 'base64':
        data = base64.b64decode(text(source.get('data')))
        return BinaryContent.narrow_type(BinaryContent(data, media_type=text(source.get('media_type'))))
    if source.get('type') == 'url':
        return ImageUrl(text(source.get('url'))) if kind == 'image' else DocumentUrl(text(source.get('url')))
    if source.get('type') == 'text':
        return text(source.get('data')) or None
    return None


def _contents(content: object) -> list[UserContent]:
    if isinstance(content, str):
        return [content] if content else []
    return [item for block in objects(content) if (item := _content(block)) is not None]


def _user(history: HistoryBuilder, content: object, *, at: datetime) -> None:
    prompt: list[UserContent] = []
    for block in objects(content):
        if block.get('type') == 'tool_result':
            items = _contents(block.get('content'))
            result: ToolReturnContent = (
                '\n'.join(item for item in items if isinstance(item, str))
                if all(isinstance(item, str) for item in items)
                else items
            )
            outcome = 'failed' if block.get('is_error') else 'success'
            history.result(text(block.get('tool_use_id')), result, at=at, outcome=outcome)
        elif (item := _content(block)) is not None:
            prompt.append(item)
    if isinstance(content, str):
        prompt = [content]
    history.prompt(prompt[0] if len(prompt) == 1 and isinstance(prompt[0], str) else prompt, at=at)


def _meta(
    history: HistoryBuilder,
    entry: JsonObject,
    agents: dict[str, DelegationTask],
    reports: dict[str, int],
    *,
    at: datetime,
) -> None:
    """A message Claude Code wrote on the user's channel: a sub-agent's report, or other context it added."""
    items = _contents(obj(entry.get('message')).get('content'))
    body = '\n'.join(item for item in items if isinstance(item, str))
    files = [item for item in items if not isinstance(item, str)]
    if (match := _HAND_BACK.search(body)) and (task := agents.get(match.group(1))):
        report = '\n'.join(
            line.removeprefix('  ') for line in match.group(2).removesuffix('</agent-message>').splitlines()
        )
        reports[task.id] = generation = reports.get(task.id, 0) + 1
        part: ModelRequestPart = replace(
            replace(task, output=report.strip(), generation=generation).report(), timestamp=at
        )
    elif body or files:
        kind = 'compact_summary' if entry.get('isCompactSummary') else 'meta'
        part = imported_context(body, source=SOURCE, kind=kind, files=files, at=at)
    else:
        return
    history.request(part, at=at)


def _attachment(history: HistoryBuilder, entry: JsonObject, *, at: datetime) -> None:
    """What Claude Code attached to the next request: a message queued mid-turn, or the context it rendered."""
    attachment = obj(entry.get('attachment'))
    kind = text(attachment.get('type'))
    rendered = '\n\n'.join(text(item.get('content')) for item in objects(entry.get('rendered')))
    if kind == 'queued_command':
        prompt = attachment.get('prompt')
        if attachment.get('commandMode') == 'prompt' and obj(attachment.get('origin')).get('kind') != 'peer':
            # Something the user typed while the model worked.
            items = _contents(prompt)
            history.prompt(items[0] if len(items) == 1 and isinstance(items[0], str) else items, at=at)
            return
        # A background task's notification, or a message from another session.
        rendered = rendered or _prompt_text(prompt)
        kind = f'queued_command:{text(attachment.get("commandMode"))}'
    elif kind in _EPHEMERAL:
        return
    # Attachments with nothing rendered were Claude Code's own bookkeeping, never sent to the model.
    if rendered:
        history.request(imported_context(rendered, source=SOURCE, kind=kind, at=at), at=at)


def _assistant(history: HistoryBuilder, message: JsonObject, *, at: datetime) -> None:
    model = text(message.get('model')) or None
    response_id = text(message.get('id')) or None
    for block in objects(message.get('content')):
        kind = block.get('type')
        part: ModelResponsePart
        if kind == 'text':
            part = TextPart(text(block.get('text')))
        elif kind == 'thinking':
            # Signed, so Claude can continue from it; other models receive the reasoning as text.
            signature = text(block.get('signature')) or None
            part = ThinkingPart(text(block.get('thinking')), signature=signature, provider_name=PROVIDER)
        elif kind == 'redacted_thinking':
            part = ThinkingPart('', id='redacted_thinking', signature=text(block.get('data')), provider_name=PROVIDER)
        elif kind == 'tool_use':
            part = ToolCallPart(text(block.get('name')), obj(block.get('input')), tool_call_id=text(block.get('id')))
        else:
            continue
        history.respond(part, at=at, model_name=model, provider_name=PROVIDER, response_id=response_id)
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
