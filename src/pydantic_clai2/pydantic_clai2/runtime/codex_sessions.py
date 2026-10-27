"""Codex sessions: `sessions/YYYY/MM/DD/rollout-<time>-<session-id>.jsonl` under `CODEX_HOME` or `~/.codex`.

Each line is a `{type, payload}` record: `session_meta` first, then the `response_item`s the model saw,
`turn_context` with the model in use, and `compacted` where compaction replaced the history. Those
items include the context Codex's harness added: developer messages, and user messages such as
`<environment_context>` and AGENTS.md, which Codex tags with the kind of content they carry.

Codex encrypts some content for OpenAI's models only: reasoning, compaction summaries, and messages
between its agents. Reasoning and summaries are kept under the `openai-codex` provider, so a
ChatGPT-subscription model continuing the session reads them as Codex's did; agents' messages cannot
be sent again, and are marked where they were left out.

Each sub-agent is a session of its own, naming its parent in its `session_meta`; it is imported as
a task of the session it descends from, rather than listed as a session itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid5

from pydantic_ai.messages import (
    BinaryContent,
    CompactionPart,
    ImageUrl,
    ModelMessage,
    ModelRequestPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserContent,
)
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.subagents import DelegationTask, is_task_report
from pydantic_clai2.runtime.imported_history import (
    HEAD_LINES,
    Header,
    HistoryBuilder,
    JsonObject,
    final_answer,
    framed,
    imported_context,
    imported_kind,
    is_typed_prompt,
    number,
    obj,
    objects,
    records,
    text,
    timestamp,
    title_text,
)

PROVIDER = 'openai-codex'
"""Codex's model provider, signed in with a ChatGPT subscription, which can read what Codex's models encrypted."""

SOURCE = 'codex'
"""The `source` of the context imported from Codex."""

_SUMMARY_PREFIX = 'Another language model started to solve this problem'
"""How the summary of a compaction Codex ran itself starts."""

CONTEXT_PREFIXES = (
    '<environment_context>',
    '<user_instructions>',
    '<recommended_plugins>',
    '# AGENTS.md instructions',
    '<skill>',
    '<user_shell_command>',
    '<turn_aborted>',
    '<subagent_notification>',
    '<goal_context>',
    '<codex_internal_context',
    '<external_',
    '<agent_message_board_notification>',
    '<image_resize_notice>',
    '<user_verification_notice>',
    _SUMMARY_PREFIX,
)
"""How user messages Codex adds itself start, for versions that do not tag their content with its kind."""

_TASK_NAMESPACE = UUID('6a3e1f0c-2b7d-4e59-8c14-9d0b5f7a3e21')
"""Derives a CLAI task ID from a Codex session and sub-agent ID, so importing again finds the same task."""

_SELF = 'self'
"""The agent name CLAI runs Codex's default sub-agents under: itself."""

_ENCRYPTED = "[Encrypted by Codex for OpenAI's models, and left out.]"

_RETAINED_USER = 'compaction_retained:user'
"""The kind of a prompt compaction kept, which Codex counts as the turn it started."""
_RETAINED_AGENT_MESSAGE = 'compaction_retained:agent_message'
_RETAINED_CONTEXT = 'compaction_retained:context'
"""The kind of the context compaction kept on the user's channel, which starts no turn."""

_OPENING = (
    '{subject} was imported from Codex session {session_id}. Up to the note at its end, it ran in '
    "Codex's harness: the tool calls in it (such as exec, exec_command, apply_patch, spawn_agent and wait_agent), "
    'its developer instructions, `<environment_context>`, AGENTS.md and the other context it added are '
    "Codex's, and its sub-agents' final answers appear as `<task_output>` blocks."
)
_CLOSING = (
    'The conversation above was imported from Codex, and continues here in CLAI, a different harness. The '
    "tools, context and instructions above came from Codex's harness and no longer apply: use the tools and "
    'instructions you have now, whose names and arguments may differ.'
)
_COMPACTED = (
    'Codex compacted the conversation above at this point: from here on, its model saw only the summary that '
    'follows, which only OpenAI can read, and the messages after it.'
)
_COMPACTED_LOCALLY = (
    'Codex compacted the conversation above at this point: from here on, its model saw only the messages that '
    'follow, its summary among them.'
)
_STOPPED = 'Stopped in Codex before it answered. Its history is saved: resume it to continue.'
"""The output of a sub-agent that never finished in Codex."""


def home() -> Path:
    """Codex's home directory."""
    return Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex')


def files(root: Path) -> list[Path]:
    """Every session transcript, sub-agents' too, which `header` leaves out."""
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
    """The session ID and directory from its metadata, and its first typed prompt; `None` for a sub-agent."""
    meta: JsonObject = {}
    prompt = ''
    for record in records(path, limit=HEAD_LINES):
        payload = obj(record.get('payload'))
        if record.get('type') == 'session_meta':
            if not meta and _parent(payload):
                # A sub-agent's session is a task of its parent's.
                return None
            meta = meta or payload
        elif record.get('type') == 'response_item' and payload.get('role') == 'user':
            kinds = _kinds(payload)
            texts = (
                text(block.get('text'))
                for index, block in enumerate(objects(payload.get('content')))
                if _typed(text(block.get('text')), kinds[index] if index < len(kinds) else '')
            )
            prompt = prompt or next((t for t in texts if t), '')
        if meta and prompt:
            return Header(
                native_id=text(meta.get('id')),
                cwd=text(meta.get('cwd')),
                title=title_text(prompt),
                # `codex exec` runs a script's prompt, rather than a conversation to continue.
                listed=meta.get('source') != 'exec',
            )
    return None


def messages(path: Path) -> list[ModelMessage]:
    """The history the session's model saw, as Pydantic AI messages, framed by notes on the change of harness.

    History compacted away is kept ahead of the compaction, which a model of the `openai-codex` provider
    continues from, as Codex's did, and other models read past, as the summary is not readable to them.
    """
    # Task IDs derive from the session alone, and a report names no conversation, so none is needed here.
    session_id, agents = _agents(path, conversation_id='')
    lines = [f'- Codex sub-agent {agent.name} is task {agent.task.id}' for agent in agents.values()]
    closing = _CLOSING + (
        "\n\nCodex's sub-agents were imported as tasks, by task ID. One whose report is not above stopped "
        'before it answered:\n' + '\n'.join(lines)
        if lines
        else ''
    )
    opening = _OPENING.format(subject='This conversation', session_id=session_id)
    return framed(_history(list(records(path)), agents), source=SOURCE, opening=opening, closing=closing)


def tasks(path: Path, conversation_id: str) -> list[DelegationTask]:
    """The session's sub-agents, and theirs, as finished CLAI tasks of `conversation_id`, with their own histories.

    One that answered is delivered already, in its parent's history; one that never finished is `cancelled`.
    """
    session_id, agents = _agents(path, conversation_id)
    found: list[DelegationTask] = []
    for agent in agents.values():
        entries = list(records(agent.path))
        opening = _OPENING.format(subject=f'This sub-agent conversation ({agent.name})', session_id=session_id)
        history = _history(entries, agents)
        task = agent.task
        output = final_answer(history)
        times = [timestamp(entry.get('timestamp')) for entry in entries if entry.get('timestamp')]
        if output is None:
            # Codex stopped it before it answered, when the session ended or was imported mid-run. Still
            # delivered, so opening the session starts no turn of its own; the closing note tells the parent.
            task = replace(task, outcome='cancelled')
        found.append(
            replace(
                task,
                messages=framed(history, source=SOURCE, opening=opening, closing=_CLOSING),
                output=_STOPPED if output is None else output,
                started_at=times[0].timestamp() if times else task.started_at,
                finished_at=times[-1].timestamp() if times else None,
            )
        )
    return found


def _parent(meta: JsonObject) -> str:
    """The session a sub-agent's metadata names as its parent, or empty for a session of its own."""
    spawn = obj(obj(obj(meta.get('source')).get('subagent')).get('thread_spawn'))
    return text(meta.get('parent_thread_id')) or text(spawn.get('parent_thread_id'))


@dataclass(frozen=True, kw_only=True)
class _SubAgent:
    path: Path
    """Its own transcript."""
    address: str
    """The path Codex's agents address it by, such as `/root/research`, or empty for one they do not message."""
    task: DelegationTask
    """Its task, without its history."""

    @property
    def name(self) -> str:
        return self.task.prompt


def _agents(path: Path, conversation_id: str) -> tuple[str, dict[str, _SubAgent]]:
    """The session's ID, and the sub-agents descending from it by thread ID, read from their metadata."""
    session_id = ''
    for record in records(path, limit=1):
        session_id = text(obj(record.get('payload')).get('id'))
    sessions = next((parent for parent in path.parents if parent.name == 'sessions'), path.parent)
    children: dict[str, list[tuple[Path, JsonObject]]] = {}
    for candidate in sorted(sessions.rglob('rollout-*.jsonl')):
        # Named for when they started, so a session started before this one is no sub-agent of it.
        if candidate.name[:27] < path.name[:27]:
            continue
        try:
            first = list(records(candidate, limit=1))
        except OSError:
            # Another session, deleted or unreadable since the listing: it must not keep this one from importing.
            continue
        for record in first:
            meta = obj(record.get('payload'))
            if record.get('type') == 'session_meta' and (parent := _parent(meta)):
                children.setdefault(parent, []).append((candidate, meta))
    agents: dict[str, _SubAgent] = {}
    pending = [session_id]
    while pending:
        parent = pending.pop(0)
        for child_path, meta in children.get(parent, []):
            thread_id = text(meta.get('id'))
            if not thread_id or thread_id in agents or thread_id == session_id:
                continue
            pending.append(thread_id)
            subagent = obj(meta.get('source')).get('subagent')
            spawn = obj(obj(subagent).get('thread_spawn'))
            if spawn:
                role = text(meta.get('agent_role')) or text(spawn.get('agent_role'))
                name = _SELF if role in ('', 'default') else role
            else:
                # A review, or another kind of sub-agent CLAI has no counterpart for, keeps its kind as its name.
                name = text(subagent) or next(iter(obj(subagent)), 'subagent')
            nickname = text(meta.get('agent_nickname')) or text(spawn.get('agent_nickname'))
            address = text(meta.get('agent_path')) or text(spawn.get('agent_path'))
            # Codex encrypts the task it gives a sub-agent, so its address and nickname describe it.
            description = ' '.join(part for part in (address, f'({nickname})' if nickname else '') if part)
            agents[thread_id] = _SubAgent(
                path=child_path,
                address=address,
                task=DelegationTask(
                    id=uuid5(_TASK_NAMESPACE, f'{session_id}:{thread_id}').hex,
                    run_id=uuid5(_TASK_NAMESPACE, f'{session_id}:{thread_id}:run').hex,
                    agent_name=name,
                    prompt=description or f'{name} {thread_id}',
                    conversation_id=conversation_id,
                    parent_id=uuid5(_TASK_NAMESPACE, f'{session_id}:{parent}').hex if parent != session_id else None,
                    background=True,
                    resumable=name == _SELF,
                    status='finished',
                    outcome='ok',
                    delivered=True,
                ),
            )
    return session_id, agents


def _kinds(item: JsonObject) -> list[str]:
    """The kind of content Codex tagged each of a message's blocks with, such as `user.text`, where it did."""
    kinds = obj(item.get('internal_chat_message_metadata_passthrough')).get('content_item_kinds')
    return [text(kind) for kind in cast(list[object], kinds)] if isinstance(kinds, list) else []


def _typed(value: str, kind: str) -> bool:
    """Whether user text was typed, rather than context Codex adds itself."""
    if kind and kind != 'unknown':
        return kind.startswith('user.')
    return bool(value) and not value.lstrip().startswith(CONTEXT_PREFIXES)


def _context_kind(value: str, kind: str, *, role: str) -> str:
    """What context Codex added is, by its tag where Codex gave none, as `imported_context`'s `kind`."""
    if kind and kind != 'unknown':
        return kind
    stripped = value.lstrip()
    if stripped.startswith('<') and (tag := stripped[1:].split('>', 1)[0].split()):
        return tag[0]
    return 'agents_md.instructions' if stripped.startswith('# AGENTS.md') else role


def _image(url: str) -> UserContent:
    return BinaryContent.from_data_uri(url) if url.startswith('data:') else ImageUrl(url)


def _history(entries: list[JsonObject], agents: dict[str, _SubAgent]) -> list[ModelMessage]:
    history = HistoryBuilder()
    model: str | None = None
    by_path = {agent.address: agent.task for agent in agents.values() if agent.address}
    reports: dict[str, int] = {}
    for record in entries:
        kind = record.get('type')
        payload = obj(record.get('payload'))
        at = timestamp(record.get('timestamp'))
        if kind == 'turn_context':
            model = text(payload.get('model')) or model
        elif kind == 'response_item':
            if obj(record.get('metadata')).get('delivered_assistant_message'):
                # A record of what Codex showed the user, which its model never sees again.
                continue
            _item(history, payload, by_path, reports, at=at, model=model)
        elif kind == 'inter_agent_communication':
            # How older versions recorded a message between agents, sent to the model as `agent_message` is.
            _agent_message(history, _communication(payload), by_path, reports, at=at)
        elif kind == 'compacted':
            _compacted(history, payload, at=at, model=model)
        elif kind == 'event_msg' and payload.get('type') == 'thread_rolled_back':
            history.rollback(number(payload.get('num_turns')), starts=_turn_start, ahead=_turn_context)
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


def _turn_start(part: ModelRequestPart) -> bool:
    """Whether `part` starts a turn Codex can roll back: a prompt or a message between agents, compaction-kept too.

    A rollback right after a compaction drops the prompts it kept, rather than reaching back past it, as
    Codex's does; an encrypted summary stays, where Codex drops it with the last prompt it followed.
    """
    kind = imported_kind(part) or ''
    return (
        is_typed_prompt(part)
        or is_task_report(part)
        or kind.startswith('agent_message')
        or kind in (_RETAINED_USER, _RETAINED_AGENT_MESSAGE)
    )


def _turn_context(kind: str) -> bool:
    """Whether imported context of `kind` is what Codex added ahead of a turn, compaction-kept too."""
    return kind == _RETAINED_CONTEXT or (bool(kind) and not kind.startswith(('compaction', 'import_', 'agent_message')))


def _compacted(history: HistoryBuilder, payload: JsonObject, *, at: datetime, model: str | None) -> None:
    """Where Codex replaced the history: its summary, then the messages it kept, after the history it replaced."""
    replacement = objects(payload.get('replacement_history'))
    if not replacement:
        # Older versions kept the readable summary alone.
        if summary := text(payload.get('message')):
            history.request(imported_context(summary, source=SOURCE, kind='compaction_summary', at=at), at=at)
        return
    sealed = [
        item
        for item in replacement
        if item.get('type') in ('compaction', 'compaction_summary') and text(item.get('encrypted_content'))
    ]
    # Without an encrypted summary, Codex compacted the history itself, its summary a readable message.
    note = _COMPACTED if sealed else _COMPACTED_LOCALLY
    history.request(imported_context(note, source=SOURCE, kind='compaction', at=at), at=at)
    for item in sealed:
        # A boundary for the provider that can read it, which sends nothing before it; others skip it.
        details = {'type': 'compaction', 'encrypted_content': text(item.get('encrypted_content'))}
        part = CompactionPart(None, id=text(item.get('id')) or None, provider_name=PROVIDER, provider_details=details)
        history.respond(part, at=at, model_name=model, provider_name=PROVIDER)
    for item in replacement:
        # Kept from before the summary, as the history above holds them too: as context on the user's channel.
        content = _retained(item) if item not in sealed else []
        body = '\n\n'.join(entry for entry in content if isinstance(entry, str))
        files = [entry for entry in content if not isinstance(entry, str)]
        if body or files:
            kind = f'compaction_retained:{text(item.get("role")) or text(item.get("type"))}'
            kinds = _kinds(item)
            blocks = objects(item.get('content'))
            if kind == _RETAINED_USER and not any(
                _typed(t, kinds[index] if index < len(kinds) else '')
                for index, block in enumerate(blocks)
                if (t := text(block.get('text')))
            ):
                # A local compaction's summary, or Codex's own context, rather than a prompt it kept.
                kind = 'compaction_summary' if body.lstrip().startswith(_SUMMARY_PREFIX) else _RETAINED_CONTEXT
            history.request(imported_context(body, source=SOURCE, kind=kind, files=files, at=at), at=at)


def _retained(item: JsonObject) -> list[UserContent]:
    """The content of an item compaction kept: a message, a message between agents, or a tool's output."""
    kind = item.get('type')
    if kind == 'message':
        content: list[UserContent] = []
        for block in objects(item.get('content')):
            if block.get('type') == 'input_image' and (url := text(block.get('image_url'))):
                content.append(_image(url))
            elif body := text(block.get('text')):
                content.append(body)
        return content
    if kind == 'agent_message':
        return [_agent_text(item)]
    if kind in ('function_call_output', 'custom_tool_call_output'):
        output = _output(item.get('output'))
        return [output] if isinstance(output, str) else output
    return []


def _item(
    history: HistoryBuilder,
    item: JsonObject,
    by_path: dict[str, DelegationTask],
    reports: dict[str, int],
    *,
    at: datetime,
    model: str | None,
) -> None:
    kind = item.get('type')
    role = text(item.get('role'))
    item_id = text(item.get('id')) or None
    if kind == 'message' and role in ('user', 'developer'):
        _message(history, item, role=role, at=at)
    elif kind == 'message' and role == 'assistant':
        phase = text(item.get('phase'))
        for block in objects(item.get('content')):
            if block.get('type') == 'output_text' and (content := text(block.get('text'))):
                details = {'phase': phase} if phase else None
                part = TextPart(content, id=item_id, provider_name=PROVIDER, provider_details=details)
                history.respond(part, at=at, model_name=model, provider_name=PROVIDER)
    elif kind == 'reasoning' and (thinking := _reasoning(item)) is not None:
        history.respond(thinking, at=at, model_name=model, provider_name=PROVIDER)
    elif kind in ('function_call', 'custom_tool_call'):
        args = text(item.get('arguments')) if kind == 'function_call' else {'input': text(item.get('input'))}
        namespace = text(item.get('namespace'))
        call = ToolCallPart(
            text(item.get('name')),
            args,
            tool_call_id=text(item.get('call_id')),
            # A custom tool's call goes back as a function call, which OpenAI rejects with a custom call's ID.
            id=item_id if kind == 'function_call' else None,
            provider_name=PROVIDER,
            # As OpenAI's models return a namespaced tool's call, which they need back to accept it.
            provider_details={'namespace': namespace} if namespace else None,
        )
        history.respond(call, at=at, model_name=model, provider_name=PROVIDER)
    elif kind in ('function_call_output', 'custom_tool_call_output'):
        history.result(text(item.get('call_id')), _output(item.get('output')), at=at)
    elif kind == 'agent_message':
        _agent_message(history, item, by_path, reports, at=at)


def _message(history: HistoryBuilder, item: JsonObject, *, role: str, at: datetime) -> None:
    """A user message, or Codex's own context on the user's or developer's channel, in its order."""
    kinds = _kinds(item)
    prompt: list[UserContent] = []

    def typed() -> None:
        history.prompt(prompt[0] if len(prompt) == 1 and isinstance(prompt[0], str) else list(prompt), at=at)
        prompt.clear()

    for index, block in enumerate(objects(item.get('content'))):
        kind = kinds[index] if index < len(kinds) else ''
        body = text(block.get('text'))
        if block.get('type') == 'input_image' and (url := text(block.get('image_url'))):
            prompt.append(_image(url))
        elif not body:
            continue
        elif role == 'user' and _typed(body, kind):
            prompt.append(body)
        else:
            typed()
            context_kind = _context_kind(body, kind, role=role)
            history.request(imported_context(body, source=SOURCE, kind=context_kind, at=at), at=at)
    typed()


def _reasoning(item: JsonObject) -> ThinkingPart | None:
    """Reasoning as OpenAI's models return it: its readable summary, signed with what only they can read."""
    summary = '\n\n'.join(text(part.get('text')) for part in objects(item.get('summary')))
    raw = [t for part in objects(item.get('content')) if (t := text(part.get('text')))]
    signature = text(item.get('encrypted_content'))
    if not (summary or raw or signature):
        return None
    return ThinkingPart(
        summary,
        id=text(item.get('id')) or None,
        signature=signature or None,
        provider_name=PROVIDER,
        provider_details={'raw_content': raw} if raw else None,
    )


def _output(value: object) -> str | list[UserContent]:
    """A tool's output: text, or text and images in order."""
    if isinstance(value, str):
        return value
    items: list[UserContent] = []
    for block in objects(value):
        if block.get('type') == 'input_image' and (url := text(block.get('image_url'))):
            items.append(_image(url))
        elif block.get('type') == 'encrypted_content':
            items.append(_ENCRYPTED)
        elif body := text(block.get('text')):
            items.append(body)
    if all(isinstance(entry, str) for entry in items):
        return '\n'.join(entry for entry in items if isinstance(entry, str))
    return items


def _agent_message(
    history: HistoryBuilder,
    item: JsonObject,
    by_path: dict[str, DelegationTask],
    reports: dict[str, int],
    *,
    at: datetime,
) -> None:
    """A message between Codex's agents: a sub-agent's final answer becomes its task's report."""
    body = _agent_text(item)
    header, _, answer = body.partition('Payload:\n')
    task = by_path.get(text(item.get('author')))
    part: ModelRequestPart
    if task is not None and 'Message Type: FINAL_ANSWER' in header:
        reports[task.id] = generation = reports.get(task.id, 0) + 1
        part = replace(replace(task, output=answer.strip(), generation=generation).report(), timestamp=at)
    elif body:
        message_type = header.partition('Message Type: ')[2].partition('\n')[0].strip().lower()
        part = imported_context(body, source=SOURCE, kind=f'agent_message:{message_type or "message"}', at=at)
    else:
        return
    history.request(part, at=at)


def _agent_text(item: JsonObject) -> str:
    """A message between agents as the model reads it, marked where Codex encrypted it."""
    blocks = objects(item.get('content'))
    return ''.join(_ENCRYPTED if b.get('type') == 'encrypted_content' else text(b.get('text')) for b in blocks)


def _communication(payload: JsonObject) -> JsonObject:
    """An `inter_agent_communication` record as the `agent_message` Codex sends the model for it."""
    author, recipient = text(payload.get('author')), text(payload.get('recipient'))
    content: list[JsonObject] = [{'type': 'input_text', 'text': text(payload.get('content'))}]
    if encrypted := text(payload.get('encrypted_content')):
        message_type = 'NEW_TASK' if payload.get('trigger_turn') else 'MESSAGE'
        header = f'Message Type: {message_type}\nTask name: {recipient}\nSender: {author}\nPayload:\n'
        content = [
            {'type': 'input_text', 'text': header},
            {'type': 'encrypted_content', 'encrypted_content': encrypted},
        ]
    return {'type': 'agent_message', 'author': author, 'recipient': recipient, 'content': content}
