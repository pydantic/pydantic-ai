"""Shared reading for other agents' JSON Lines transcripts, and the history they become.

Claude Code and Codex write one JSON object per line. `HistoryBuilder` groups their parts into
alternating requests and responses; core's `repair_messages` then pairs every tool call with a result.

Content the other agent's harness added on the user's channel, rather than its user, is imported as
`imported_context`: tagged with an `origin` under the harness prefix, so compaction and the transcript
tell it apart from what the user typed, as they do a harness capability's content.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Literal, TypeGuard, cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnContent,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    repair_messages,
)
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.subagents import is_task_report

JsonObject = dict[str, object]

HEAD_LINES = 200
"""How many lines a listing reads from the top of a transcript for its directory and first prompt."""

TAIL_BYTES = 1 << 20
"""How much a listing reads from the end of a transcript for metadata the other agent appends as it goes."""

IMPORTED_CONTEXT_ORIGIN = 'pydantic-ai-harness.imported-context'
"""The `origin` of the `TextContent` that carries content another agent's harness added to its conversation.

Under the harness prefix, so compaction treats it as harness data rather than as something the user typed.
"""


@dataclass(frozen=True, kw_only=True)
class Header:
    """What the browser shows for a transcript, read from its first lines."""

    native_id: str
    """The session ID the other agent uses."""
    cwd: str
    title: str
    named: bool = False
    """Whether `title` is a name the other agent gave the session, rather than its first prompt."""
    listed: bool = True
    """Whether the browser lists the session; one it does not, such as a script's, can still be imported by ID."""


def records(path: Path, *, limit: int | None = None) -> Iterator[JsonObject]:
    """The file's JSON objects, skipping lines in a shape this version cannot parse."""
    with path.open(encoding='utf-8', errors='replace') as lines:
        for line in islice(lines, limit):
            try:
                value: object = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                yield cast(JsonObject, value)


def tail_records(path: Path, *, containing: bytes = b'', size: int = TAIL_BYTES) -> list[JsonObject]:
    """The JSON objects in the last `size` bytes of the file, skipping the line the cut lands in.

    Only lines `containing` the given bytes are parsed, so a caller after a few entries skips large payloads.
    """
    with path.open('rb') as file:
        start = max(file.seek(0, os.SEEK_END) - size, 0)
        file.seek(start)
        lines = file.read().split(b'\n')
    objects: list[JsonObject] = []
    for line in lines[1:] if start else lines:
        if containing not in line:
            continue
        try:
            value: object = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            objects.append(cast(JsonObject, value))
    return objects


def text(value: object) -> str:
    """A JSON string, or empty for anything else."""
    return value if isinstance(value, str) else ''


def number(value: object) -> int:
    """A JSON integer, or zero for anything else."""
    return value if isinstance(value, int) else 0


def obj(value: object) -> JsonObject:
    """A JSON object, or empty for anything else."""
    return cast(JsonObject, value) if isinstance(value, dict) else {}


def objects(value: object) -> list[JsonObject]:
    """The objects in a JSON array, or none for anything else."""
    if not isinstance(value, list):
        return []
    return [cast(JsonObject, item) for item in cast(list[object], value) if isinstance(item, dict)]


def timestamp(value: object) -> datetime:
    """An ISO 8601 time in UTC, or now when the transcript has none."""
    try:
        return datetime.fromisoformat(text(value)).astimezone(UTC)
    except ValueError:
        return datetime.now(UTC)


def title_text(value: str) -> str:
    """One line of printable text for a browser card."""
    return ' '.join(''.join(c for c in value if c.isprintable() or c.isspace()).split())[:100]


def imported_context(
    content: str, *, source: str, kind: str, files: Sequence[UserContent] = (), at: datetime
) -> UserPromptPart:
    """Content `source`'s harness added on the user's channel, such as a file it attached or a reminder, as it sent it.

    `kind` names what it was, such as an attachment type; like the source, it is metadata the model never sees.
    """
    metadata = {'origin': IMPORTED_CONTEXT_ORIGIN, 'source': source, 'kind': kind}
    return UserPromptPart([TextContent(content, metadata=metadata), *files], timestamp=at)


def framed(history: Sequence[ModelMessage], *, source: str, opening: str, closing: str) -> list[ModelMessage]:
    """`history` between notes on the change of harness, as `imported_context`, or nothing when there is no history."""
    built = list(history)
    if not built:
        return []
    first = built[0]
    start = imported_context(opening, source=source, kind='import_start', at=first.timestamp or timestamp(None))
    end = imported_context(closing, source=source, kind='import_end', at=built[-1].timestamp or timestamp(None))
    if isinstance(first, ModelRequest):
        built[0] = replace(first, parts=[start, *first.parts])
    else:
        built.insert(0, ModelRequest(parts=[start], timestamp=start.timestamp))
    if isinstance(last := built[-1], ModelRequest):
        built[-1] = replace(last, parts=[*last.parts, end])
    else:
        built.append(ModelRequest(parts=[end], timestamp=end.timestamp))
    return built


def final_answer(history: Sequence[ModelMessage]) -> str | None:
    """The text of the reply that ends `history`, or `None` when it ends on work not yet answered, or a call."""
    last = history[-1] if history else None
    if not isinstance(last, ModelResponse) or last.tool_calls:
        return None
    return '\n\n'.join(part.content for part in last.parts if isinstance(part, TextPart))


def is_typed_prompt(part: ModelRequestPart) -> TypeGuard[UserPromptPart]:
    """Whether `part` is a prompt the user typed, not a task's report or context another agent's harness added."""
    return isinstance(part, UserPromptPart) and not is_task_report(part) and not is_imported_context(part)


def is_imported_context(part: ModelRequestPart) -> bool:
    """Whether `part` is `imported_context`, rather than something the user typed."""
    return imported_kind(part) is not None


def imported_kind(part: ModelRequestPart) -> str | None:
    """The `kind` of `imported_context`, such as an attachment type, or `None` for any other part."""
    if not isinstance(part, UserPromptPart) or isinstance(part.content, str) or not part.content:
        return None
    first = part.content[0]
    raw: object = first.metadata if isinstance(first, TextContent) else None
    metadata = cast(dict[str, object], raw) if isinstance(raw, dict) else {}
    return text(metadata.get('kind')) if metadata.get('origin') == IMPORTED_CONTEXT_ORIGIN else None


@dataclass(kw_only=True)
class _Request:
    parts: list[ModelRequestPart]
    timestamp: datetime


@dataclass(kw_only=True)
class _Response:
    parts: list[ModelResponsePart]
    timestamp: datetime
    model_name: str | None
    provider_name: str
    response_id: str | None = None
    usage: RequestUsage = field(default_factory=RequestUsage)


class HistoryBuilder:
    """Parts in transcript order, grouped into alternating requests and responses."""

    def __init__(self) -> None:
        self._turns: list[_Request | _Response] = []
        self._calls: dict[str, str] = {}

    def prompt(self, content: str | Sequence[UserContent], *, at: datetime) -> None:
        """A user message, skipped when it is empty."""
        if content:
            self.request(UserPromptPart(content, timestamp=at), at=at)

    def result(
        self,
        call_id: str,
        content: ToolReturnContent,
        *,
        at: datetime,
        outcome: Literal['success', 'failed'] = 'success',
    ) -> None:
        """A tool result, skipped when no earlier call made it, such as one compacted away."""
        if (name := self._calls.get(call_id)) is not None:
            part = ToolReturnPart(tool_name=name, content=content, tool_call_id=call_id, timestamp=at, outcome=outcome)
            self.request(part, at=at)

    def respond(
        self,
        part: ModelResponsePart,
        *,
        at: datetime,
        model_name: str | None,
        provider_name: str,
        response_id: str | None = None,
    ) -> None:
        """A part of the model's reply, joining the reply in progress unless `response_id` says it is another one."""
        last = self._turns[-1] if self._turns else None
        if not isinstance(last, _Response) or (response_id is not None and last.response_id not in (None, response_id)):
            last = _Response(
                parts=[], timestamp=at, model_name=model_name, provider_name=provider_name, response_id=response_id
            )
            self._turns.append(last)
        last.response_id = last.response_id or response_id
        last.parts.append(part)
        if isinstance(part, ToolCallPart):
            self._calls[part.tool_call_id] = part.tool_name

    def usage(self, usage: RequestUsage) -> None:
        """The token usage of the latest reply's model request."""
        response = next((turn for turn in reversed(self._turns) if isinstance(turn, _Response)), None)
        if response is not None:
            response.usage = usage

    def rollback(self, turns: int, *, starts: Callable[[ModelRequestPart], bool], ahead: Callable[[str], bool]) -> None:
        """Drop the last `turns` turns, each begun by a part that `starts` one.

        The context the harness added `ahead` of a turn, by its `kind`, goes with it, except ahead of the
        first turn: that is the session's own, which rolling back every turn keeps.
        """
        found = [
            (index, turn, position)
            for index, turn in enumerate(self._turns)
            if isinstance(turn, _Request)
            for position, part in enumerate(turn.parts)
            if starts(part)
        ]
        if turns <= 0 or not found:
            return
        index, request, position = found[-min(turns, len(found))]
        while turns < len(found) and position and ahead(imported_kind(request.parts[position - 1]) or ''):
            position -= 1
        # Results of calls from before the turn stay: they answer the reply the turn followed.
        del self._turns[index + 1 :]
        del request.parts[position:]
        if not request.parts:
            self._turns.pop()

    def reset(self) -> None:
        """Forget everything so far, as compaction replaces the history the model sees."""
        self._turns.clear()
        self._calls.clear()

    def build(self) -> list[ModelMessage]:
        """Messages any model accepts: every tool call answered, results ahead of user text."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=turn.parts, timestamp=turn.timestamp)
            if isinstance(turn, _Request)
            else ModelResponse(
                parts=turn.parts,
                timestamp=turn.timestamp,
                model_name=turn.model_name,
                provider_name=turn.provider_name,
                provider_response_id=turn.response_id,
                usage=turn.usage,
            )
            for turn in self._turns
        ]
        return repair_messages(messages)

    def request(self, part: ModelRequestPart, *, at: datetime) -> None:
        """A part of the next request, joining the request in progress."""
        last = self._turns[-1] if self._turns else None
        if not isinstance(last, _Request):
            last = _Request(parts=[], timestamp=at)
            self._turns.append(last)
        last.parts.append(part)
