"""Sessions in Logfire (hackathon): "the trace is the storage", for conversation resume on another machine.

Every top-level run writes what it added to the conversation to a `clai2 session chunk` span under the run's
span: `clai2.session.payload` is the runs's new `ModelMessage`s as zlib-compressed JSON, base64-encoded, and the
other `clai2.session.*` attributes are its manifest (session ID, sequence number, size, SHA-256, parent). Binary
content is replaced by a placeholder. A chunk over 4 MiB is split across several spans (`part` of `parts`).

Resuming reads the chunks back with the query API and checks every one, failing loudly on a missing run, a
missing part, or a size or hash mismatch rather than continuing an incomplete conversation. It is a conversation
resume, never an exact one: binaries, tool state, and the workspace are not stored. Your own session continues
in place when this machine is up to date with it; someone else's, or one that moved on elsewhere, is forked,
and the fork records where it came from. Every member of the Logfire project can read these sessions.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import zlib
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit
from uuid import uuid4

import anyio.to_thread
import httpx
from opentelemetry.trace import TracerProvider
from pydantic import BaseModel, TypeAdapter, ValidationError

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.messages import (
    BinaryContent,
    FilePart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from pydantic_clai2.runtime.remote_sessions import RemoteListing, RemoteResume

SPAN_NAME = 'clai2 session chunk'
PART_CHARS = 4 * 1024 * 1024
"""Base64 characters per chunk span; a longer payload is split across spans."""
WINDOW = timedelta(days=30)
PRIVACY = 'Sessions are stored in {project} and visible to its members.'
SCOPE = 'pydantic-clai2.sessions'
_MAX_FORK_DEPTH = 8
_ID = re.compile(r'[0-9a-fA-F-]{8,64}')
_TRACE = re.compile(r"trace_id\s*=\s*'([0-9a-f]{32})'")
_DEPTH: ContextVar[int] = ContextVar('clai2_session_chunk_depth', default=0)


class SessionCorrupt(ValueError):
    """A stored session is incomplete or damaged, so resuming it would silently lose part of the conversation."""


# Writing


def _placeholder(content: BinaryContent) -> str:
    return f'[{content.media_type} attachment not stored in Logfire]'


def without_binary(messages: Sequence[ModelMessage]) -> tuple[list[ModelMessage], int]:
    """The messages with binary content replaced by a placeholder, and how many were replaced."""
    dropped = 0
    result: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            parts: list[Any] = []
            for part in message.parts:
                if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                    items = [_placeholder(item) if isinstance(item, BinaryContent) else item for item in part.content]
                    dropped += sum(isinstance(item, BinaryContent) for item in part.content)
                    part = replace(part, content=items)
                elif isinstance(part, ToolReturnPart) and isinstance(part.content, BinaryContent):
                    dropped += 1
                    part = replace(part, content=_placeholder(part.content))
                parts.append(part)
            result.append(replace(message, parts=parts))
        else:
            response_parts: list[Any] = []
            for part in message.parts:
                if isinstance(part, FilePart):
                    dropped += 1
                    part = TextPart(content=_placeholder(part.content))
                response_parts.append(part)
            result.append(replace(message, parts=response_parts))
    return result, dropped


def _json(messages: Sequence[ModelMessage]) -> bytes:
    return ModelMessagesTypeAdapter.dump_json(list(messages))


def history_sha(messages: Sequence[ModelMessage]) -> str:
    """Identifies a history, so the next chunk can carry only what was added to it."""
    return hashlib.sha256(_json(messages)).hexdigest()


@dataclass(frozen=True)
class Encoded:
    """One chunk's payload, split into the strings its spans carry."""

    parts: list[str]
    size: int
    """Bytes of compressed payload."""
    sha256: str
    """Of the compressed payload."""


def encode(messages: Sequence[ModelMessage]) -> Encoded:
    """Compress and base64-encode `messages`, split into span-sized parts."""
    compressed = zlib.compress(_json(messages), 6)
    text = base64.b64encode(compressed).decode()
    parts = [text[start : start + PART_CHARS] for start in range(0, len(text), PART_CHARS)] or ['']
    return Encoded(parts=parts, size=len(compressed), sha256=hashlib.sha256(compressed).hexdigest())


def decode(text: str, *, size: int, sha256: str, where: str) -> list[ModelMessage]:
    """The messages in one chunk; raises `SessionCorrupt` unless it is exactly what was written."""
    try:
        compressed = base64.b64decode(text, validate=True)
    except ValueError:
        raise SessionCorrupt(f'{where} is damaged in Logfire (not valid base64); not resuming.') from None
    if len(compressed) != size or hashlib.sha256(compressed).hexdigest() != sha256:
        raise SessionCorrupt(
            f'{where} is truncated or damaged in Logfire ({len(compressed):,} of {size:,} bytes, hash mismatch); '
            'not resuming an incomplete conversation.'
        )
    try:
        return ModelMessagesTypeAdapter.validate_json(zlib.decompress(compressed))
    except (zlib.error, ValidationError):
        raise SessionCorrupt(f'{where} could not be read back; not resuming.') from None


class ChunkState(BaseModel):
    """What this machine has written (or loaded) for one session, so the next chunk carries only the new part."""

    next_seq: int = 0
    count: int = 0
    """Messages the chunks so far cover."""
    prefix_sha: str = ''
    """`history_sha` of those messages; a history that no longer starts with them is written whole (`reset`)."""
    parent_id: str | None = None
    parent_seq: int | None = None
    parent_owner: str | None = None
    parent_trace: str | None = None
    forked_at: str | None = None


_STATES: TypeAdapter[dict[str, ChunkState]] = TypeAdapter(dict[str, ChunkState])


@dataclass
class ChunkStates:
    """Per-session `ChunkState`, kept in a JSON file beside CLAI's Logfire settings."""

    path: Path

    def get(self, session_id: str) -> ChunkState | None:
        return self._load().get(session_id)

    def set(self, session_id: str, state: ChunkState) -> None:
        states = self._load()
        states[session_id] = state
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(_STATES.dump_json(states, indent=2))

    def _load(self) -> dict[str, ChunkState]:
        try:
            return _STATES.validate_json(self.path.read_bytes())
        except (OSError, ValidationError):
            return {}


def first_prompt(messages: Sequence[ModelMessage]) -> str:
    """The first thing the user typed, for listings; at most 200 characters."""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    text = (
                        part.content
                        if isinstance(part.content, str)
                        else ' '.join(item for item in part.content if isinstance(item, str))
                    )
                    if text.strip():
                        return ' '.join(text.split())[:200]
    return ''


@dataclass(kw_only=True)
class SessionChunks(AbstractCapability[None]):
    """Writes each top-level run's addition to the conversation as `clai2 session chunk` spans."""

    tracer_provider: TracerProvider
    states: ChunkStates
    session_id: Callable[[], str | None]
    owner: Callable[[], str | None]
    attributes: Callable[[], dict[str, str]] = dict
    """Added to every chunk span, such as the repository."""
    warn: Callable[[str], None] = lambda message: None
    id: str | None = 'clai2_session_chunks'

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        depth = _DEPTH.set(_DEPTH.get() + 1)
        try:
            return await handler()
        finally:
            _DEPTH.reset(depth)

    async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        # Only the conversation's own runs: a sub-agent's run is part of a tool call, already in its result.
        if _DEPTH.get() == 1:
            try:
                # Off the event loop: it serializes the whole history and keeps state in a file.
                await anyio.to_thread.run_sync(self.write, result.all_messages())
            except Exception as error:  # noqa: BLE001 -- storing a session must never fail the user's turn
                self.warn(f'Could not store this turn in Logfire: {type(error).__name__}: {error}')
        return result

    def write(self, messages: Sequence[ModelMessage]) -> int | None:
        """Write what `messages` adds since the last chunk; returns the chunk's sequence number, if one was written."""
        session_id = self.session_id()
        if not session_id:
            return None
        history, dropped = without_binary(messages)
        state = self.states.get(session_id) or ChunkState()
        continues = 0 < state.count <= len(history) and history_sha(history[: state.count]) == state.prefix_sha
        if continues and state.count == len(history):
            return None
        delta = history[state.count :] if continues else history
        encoded = encode(delta)
        seq = state.next_seq
        attributes: dict[str, str | int | bool] = {
            **self.attributes(),
            'clai2.session.id': session_id,
            'clai2.session.seq': seq,
            'clai2.session.parts': len(encoded.parts),
            'clai2.session.bytes': encoded.size,
            'clai2.session.sha256': encoded.sha256,
            'clai2.session.binary_dropped': dropped,
            'clai2.session.reset': not continues,
            'clai2.session.messages': len(history),
            'clai2.session.first_prompt': first_prompt(history),
            **({'clai2.session.owner': owner} if (owner := self.owner()) else {}),
            **{
                f'clai2.session.{name}': value
                for name in ('parent_id', 'parent_seq', 'parent_owner', 'parent_trace', 'forked_at')
                if (value := getattr(state, name)) is not None
            },
        }
        tracer = self.tracer_provider.get_tracer(SCOPE)
        for index, part in enumerate(encoded.parts):
            with tracer.start_as_current_span(
                SPAN_NAME,
                attributes={
                    **attributes,
                    'clai2.session.part': index,
                    'clai2.session.payload': part,
                    'logfire.msg': f'clai2 session chunk {seq}'
                    + (f' (part {index + 1} of {len(encoded.parts)})' if len(encoded.parts) > 1 else ''),
                },
            ):
                pass
        self.states.set(
            session_id,
            state.model_copy(update={'next_seq': seq + 1, 'count': len(history), 'prefix_sha': history_sha(history)}),
        )
        return seq


# Reading


@dataclass
class LogfireQuery:
    """Logfire's query API (`/v1/query`) with a key that has `project:read_otlp`."""

    base_url: str
    key: str
    http: Callable[[], httpx.AsyncClient] = lambda: httpx.AsyncClient(timeout=httpx.Timeout(30, read=120))

    async def rows(self, sql: str, *, limit: int = 10_000, since: timedelta = WINDOW) -> list[dict[str, Any]]:
        start = (datetime.now(UTC) - since).isoformat()
        async with self.http() as http:
            response = await http.get(
                f'{self.base_url}/v1/query',
                params={'sql': sql, 'min_timestamp': start, 'limit': limit, 'json_rows': 'true'},
                headers={'Authorization': f'Bearer {self.key}', 'Accept': 'application/json'},
            )
        if response.status_code in (401, 403):
            raise PermissionError(
                'Your Logfire key cannot query sessions (it needs `project:read_otlp`). Sign in to Logfire again.'
            )
        response.raise_for_status()
        rows: object = response.json().get('rows', [])
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []  # pyright: ignore[reportUnknownVariableType]


def sql_quote(value: str) -> str:
    """`value` as a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def sql_attribute(name: str) -> str:
    """The SQL expression reading span attribute `name` as text."""
    return f"attributes->>'{name}'"


_MANIFEST = ', '.join(
    f'{sql_attribute(f"clai2.session.{name}")} AS {name}'
    for name in (
        'id',
        'seq',
        'part',
        'parts',
        'bytes',
        'sha256',
        'reset',
        'owner',
        'first_prompt',
        'messages',
        'parent_id',
        'parent_seq',
        'parent_owner',
    )
)


@dataclass(frozen=True)
class Loaded:
    """A session read back from Logfire and checked."""

    session_id: str
    messages: list[ModelMessage]
    last_seq: int
    owner: str | None
    trace_id: str
    first_prompt: str
    parent_id: str | None = None
    parent_owner: str | None = None


@dataclass(frozen=True)
class Listed:
    """One of your sessions stored in Logfire."""

    session_id: str
    runs: int
    last: datetime
    repo: str
    first_prompt: str

    def summary(self, workspace: str) -> ConversationSummary:
        """How `/sessions` and the `/resume` browser show it."""
        return ConversationSummary(
            id=self.session_id,
            workspace=workspace,
            updated_at=self.last,
            title=self.first_prompt[:80] or 'Session from Logfire',
            subtitle=' · '.join(part for part in ('Logfire · other machine', self.repo, _runs(self.runs)) if part),
            tags=('logfire',),
        )


def _timestamp(value: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return datetime.now(UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass
class LogfireSessions:
    """Lists, loads, and decides how to resume sessions stored in Logfire."""

    query: LogfireQuery
    states: ChunkStates
    owner: Callable[[], str | None]
    project: str
    _cache: dict[str, Loaded] = field(default_factory=dict[str, Loaded], init=False)

    async def listing(self, *, workspace: str) -> RemoteListing:
        """Your sessions in Logfire for `/sessions` and the `/resume` browser, or why they cannot be listed."""
        try:
            listed = await self.listed()
        except (PermissionError, ValueError, httpx.HTTPError) as error:
            return RemoteListing(entries=[], unavailable=f'Logfire sessions unavailable: {error}')
        return RemoteListing(entries=[session.summary(workspace) for session in listed])

    async def listed(self) -> list[Listed]:
        """Your sessions stored in Logfire in the last 30 days, newest first."""
        owner = self.owner()
        if not owner:
            raise ValueError('Sessions in Logfire are listed by your Logfire account email, which is unknown here.')
        rows = await self.query.rows(
            f'SELECT {sql_attribute("clai2.session.id")} AS id, {sql_attribute("clai2.session.first_prompt")} AS first_prompt, '
            f'{sql_attribute("clai2.repo_slug")} AS repo, start_timestamp FROM records '
            f'WHERE span_name = {sql_quote(SPAN_NAME)} AND {sql_attribute("clai2.session.owner")} = {sql_quote(owner)} '
            f"AND {sql_attribute('clai2.session.part')} = '0' ORDER BY start_timestamp DESC",
            limit=2_000,
        )
        sessions: dict[str, Listed] = {}
        for row in rows:
            session_id = str(row.get('id') or '')
            if not session_id:
                continue
            seen = sessions.get(session_id)
            if seen is None:
                sessions[session_id] = Listed(
                    session_id=session_id,
                    runs=1,
                    last=_timestamp(row.get('start_timestamp')),
                    repo=str(row.get('repo') or ''),
                    first_prompt=str(row.get('first_prompt') or ''),
                )
            else:
                prompt = seen.first_prompt or str(row.get('first_prompt') or '')
                sessions[session_id] = replace(seen, runs=seen.runs + 1, first_prompt=prompt)
        return list(sessions.values())

    async def session_for(self, reference: str) -> str | None:
        """The session ID a reference names: an ID, or a Logfire link to a trace that holds chunks."""
        if reference.startswith(('http://', 'https://')):
            trace_id = trace_from_link(reference)
            if trace_id is None:
                raise LookupError('That Logfire link names no trace; copy the trace link, or use the session ID.')
            rows = await self.query.rows(
                f'SELECT {sql_attribute("clai2.session.id")} AS id FROM records WHERE trace_id = {sql_quote(trace_id)} '
                f'AND span_name = {sql_quote(SPAN_NAME)} ORDER BY start_timestamp DESC',
                limit=1,
            )
            if not rows:
                raise LookupError(f'No clai2 session is stored in trace {trace_id} (or it is older than 30 days).')
            return str(rows[0]['id'])
        return reference if _ID.fullmatch(reference) else None

    async def load(self, session_id: str, *, upto: int | None = None, depth: int = 0) -> Loaded | None:
        """The session as stored, checked chunk by chunk; `None` when Logfire has none of it."""
        if depth > _MAX_FORK_DEPTH:
            raise SessionCorrupt(f'Session {session_id} is forked from too many sessions to resume.')
        bound = f' AND CAST({sql_attribute("clai2.session.seq")} AS BIGINT) <= {int(upto)}' if upto is not None else ''
        rows = await self.query.rows(
            f'SELECT {_MANIFEST}, {sql_attribute("clai2.session.payload")} AS payload, trace_id FROM records '
            f'WHERE span_name = {sql_quote(SPAN_NAME)} AND {sql_attribute("clai2.session.id")} = {sql_quote(session_id)}{bound}'
        )
        if not rows:
            return None
        chunks: dict[int, dict[int, dict[str, Any]]] = {}
        for row in rows:
            seq, part = int(row['seq']), int(row['part'])
            existing = chunks.setdefault(seq, {}).get(part)
            if existing is not None and existing['sha256'] != row['sha256']:
                raise SessionCorrupt(
                    f'Session {session_id} was continued in two places at run {seq}; resume one of them by its link.'
                )
            chunks[seq][part] = row
        last = max(chunks)
        missing = [seq for seq in range(last + 1) if seq not in chunks]
        if missing:
            raise SessionCorrupt(
                f'Session {session_id} is missing run {missing[0]} of {last + 1} in Logfire (expired or dropped); '
                'not resuming an incomplete conversation.'
            )
        messages: list[ModelMessage] = []
        first = chunks[0][min(chunks[0])]
        for seq in range(last + 1):
            parts = chunks[seq]
            head = parts[min(parts)]
            count = int(head['parts'])
            if sorted(parts) != list(range(count)):
                raise SessionCorrupt(
                    f'Run {seq} of session {session_id} is missing part {next(i for i in range(count) if i not in parts) + 1} '
                    f'of {count} in Logfire; not resuming.'
                )
            text = ''.join(str(parts[index]['payload'] or '') for index in range(count))
            added = decode(
                text, size=int(head['bytes']), sha256=str(head['sha256']), where=f'Run {seq} of session {session_id}'
            )
            if str(head['reset']).lower() == 'true':
                messages = added
            else:
                if seq == 0 and head.get('parent_id'):
                    parent = await self.load(str(head['parent_id']), upto=int(head['parent_seq']), depth=depth + 1)
                    if parent is None:
                        raise SessionCorrupt(
                            f'Session {session_id} is a fork of {head["parent_id"]}, which is gone from Logfire; '
                            'not resuming.'
                        )
                    messages = parent.messages
                messages = [*messages, *added]
        newest = chunks[last][min(chunks[last])]
        return Loaded(
            session_id=session_id,
            messages=messages,
            last_seq=last,
            owner=str(first['owner']) if first.get('owner') else None,
            trace_id=str(newest['trace_id']),
            first_prompt=str(first.get('first_prompt') or ''),
            parent_id=str(first['parent_id']) if first.get('parent_id') else None,
            parent_owner=str(first['parent_owner']) if first.get('parent_owner') else None,
        )

    async def resume(self, reference: str, *, local: bool) -> RemoteResume | None:
        """Continue your up-to-date session in place; fork someone else's, or yours that moved on elsewhere."""
        session_id = await self.session_for(reference)
        if session_id is None:
            return None
        loaded = await self.load(session_id)
        if loaded is None:
            if reference.startswith(('http://', 'https://')):
                raise LookupError(f'Session {session_id} is not in Logfire (or it is older than 30 days).')
            return None
        me = self.owner()
        state = self.states.get(session_id)
        remote_next = loaded.last_seq + 1
        title = loaded.first_prompt[:60] or 'Session from Logfire'
        if loaded.owner == me and (state is None or state.next_seq >= remote_next):
            if local and state is not None:
                return RemoteResume(conversation_id=session_id, messages=None)
            self.states.set(
                session_id,
                ChunkState(next_seq=remote_next, count=len(loaded.messages), prefix_sha=history_sha(loaded.messages)),
            )
            return RemoteResume(
                conversation_id=session_id,
                messages=loaded.messages,
                title=title,
                subtitle=f'Your session from Logfire ({self.project})',
                notice=(
                    f'Continuing your session from Logfire ({_runs(remote_next)}; conversation only, no files or '
                    'tool state).'
                ),
            )
        fork_id = str(uuid4())
        self.states.set(
            fork_id,
            ChunkState(
                count=len(loaded.messages),
                prefix_sha=history_sha(loaded.messages),
                parent_id=session_id,
                parent_seq=loaded.last_seq,
                parent_owner=loaded.owner,
                parent_trace=loaded.trace_id,
                forked_at=datetime.now(UTC).isoformat(),
            ),
        )
        whose = (
            f'your session {session_id[:8]}, which continued on another machine,'
            if loaded.owner == me
            else f"{loaded.owner or 'someone'}'s session {session_id[:8]}"
        )
        return RemoteResume(
            conversation_id=fork_id,
            messages=loaded.messages,
            title=f'Fork: {title}',
            subtitle=f'Fork of {session_id} after {_runs(remote_next)} ({loaded.owner or "unknown"})',
            notice=(
                f'Forked {whose} after {_runs(remote_next)} from Logfire. The original is untouched; your turns go '
                'to the fork (conversation only, no files or tool state).'
            ),
        )


def _runs(count: int) -> str:
    return f'{count} run' if count == 1 else f'{count} runs'


def trace_from_link(link: str) -> str | None:
    """The trace a Logfire link points at: `?q=trace_id='…'` or `?traceId=…`."""
    query = parse_qs(urlsplit(link).query)
    if trace := next(iter(query.get('traceId', [])), None):
        return trace.lower() if re.fullmatch(r'[0-9a-fA-F]{32}', trace) else None
    match = _TRACE.search(' '.join(query.get('q', [])))
    return match.group(1) if match else None


def trace_link(base_url: str, project: str, trace_id: str) -> str:
    """The Logfire link to one trace, as the Logfire UI writes it."""
    return f'{base_url.rstrip("/")}/{project}?q={quote(f"trace_id={sql_quote(trace_id)}")}&last=30d'


def privacy_notice(marker: Path, project: str) -> str | None:
    """The one-time privacy line, the first time this machine stores sessions in `project`."""
    seen: list[str] = []
    try:
        seen = json.loads(marker.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        pass
    if project in seen:
        return None
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps([*seen, project]), encoding='utf-8')
    except OSError:
        pass
    return PRIVACY.format(project=project)
