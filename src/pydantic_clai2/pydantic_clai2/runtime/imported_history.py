"""Shared reading for other agents' JSON Lines transcripts, and the history they become.

Claude Code and Codex write one JSON object per line. `HistoryBuilder` groups their parts into
alternating requests and responses; core's `repair_messages` then pairs every tool call with a result.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    repair_messages,
)
from pydantic_ai.usage import RequestUsage

JsonObject = dict[str, object]

HEAD_LINES = 200
"""How many lines a listing reads from the top of a transcript for its directory and first prompt."""


@dataclass(frozen=True, kw_only=True)
class Header:
    """What the browser shows for a transcript, read from its first lines."""

    native_id: str
    """The session ID the other agent uses."""
    cwd: str
    title: str
    named: bool = False
    """Whether `title` is a name the other agent gave the session, rather than its first prompt."""


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
    usage: RequestUsage = field(default_factory=RequestUsage)


class HistoryBuilder:
    """Parts in transcript order, grouped into alternating requests and responses."""

    def __init__(self) -> None:
        self._turns: list[_Request | _Response] = []
        self._calls: dict[str, str] = {}

    def prompt(self, content: str | Sequence[UserContent], *, at: datetime) -> None:
        """A user message, skipped when it is empty."""
        if content:
            self._request(UserPromptPart(content, timestamp=at), at=at)

    def result(self, call_id: str, content: str, *, at: datetime) -> None:
        """A tool result, skipped when no earlier call made it, such as one compacted away."""
        if (name := self._calls.get(call_id)) is not None:
            self._request(ToolReturnPart(tool_name=name, content=content, tool_call_id=call_id, timestamp=at), at=at)

    def respond(self, part: ModelResponsePart, *, at: datetime, model_name: str | None, provider_name: str) -> None:
        """A part of the model's reply, joining the reply in progress."""
        last = self._turns[-1] if self._turns else None
        if not isinstance(last, _Response):
            last = _Response(parts=[], timestamp=at, model_name=model_name, provider_name=provider_name)
            self._turns.append(last)
        last.parts.append(part)
        if isinstance(part, ToolCallPart):
            self._calls[part.tool_call_id] = part.tool_name

    def usage(self, usage: RequestUsage) -> None:
        """The token usage of the latest reply's model request."""
        response = next((turn for turn in reversed(self._turns) if isinstance(turn, _Response)), None)
        if response is not None:
            response.usage = usage

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
                usage=turn.usage,
            )
            for turn in self._turns
        ]
        return repair_messages(messages)

    def _request(self, part: ModelRequestPart, *, at: datetime) -> None:
        last = self._turns[-1] if self._turns else None
        if not isinstance(last, _Request):
            last = _Request(parts=[], timestamp=at)
            self._turns.append(last)
        last.parts.append(part)
