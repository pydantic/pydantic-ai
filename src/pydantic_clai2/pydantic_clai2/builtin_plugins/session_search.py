"""Search your past sessions (hackathon): this machine's and the ones stored in Logfire, as one corpus.

Harness `ConversationSearch` ranks the history; this module is its `HistorySource`. Each saved session is one
"run" named by its session ID, so a result points at the session `/resume ID` reopens. The corpus is this user's
own: this machine's session store, plus the Logfire sessions whose owner is this user. Sessions another person
shared are not searched.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness import ConversationSearch
from pydantic_ai_harness.step_persistence import RunRecord
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2.builtin_plugins.logfire_sessions import LogfireSessions, SessionCorrupt

_CONVERSATION = 'mine'
_CACHE_SECONDS = 60.0
INSTRUCTIONS = (
    "`search_conversation_history` searches the user's own past sessions, from this machine and from Logfire, "
    'not only this one. Each result names its session as `run: <session id>`: when the user wants to go back to '
    'one, tell them to run `/resume <session id>`.'
)


@dataclass
class SessionsHistorySource:
    """This machine's saved sessions and your Logfire sessions, one run per session."""

    local: SqliteConversationStore | None
    logfire: LogfireSessions | None
    limit: int = 200
    _loaded: dict[str, tuple[float, list[ModelMessage]]] = field(
        default_factory=dict[str, tuple[float, list[ModelMessage]]], init=False
    )

    async def list_runs(self) -> list[RunRecord]:
        runs: dict[str, RunRecord] = {}
        if self.local is not None:
            for summary in await self.local.listing(limit=self.limit):
                runs[summary.id] = RunRecord(
                    run_id=summary.id,
                    conversation_id=_CONVERSATION,
                    metadata={'title': summary.title, 'where': 'this machine'},
                    started_at=summary.updated_at,
                )
        if self.logfire is not None:
            try:
                listed = await self.logfire.listed()
            except Exception:  # noqa: BLE001 -- Logfire unreachable: search this machine's sessions alone
                listed = []
            for session in listed:
                runs.setdefault(
                    session.session_id,
                    RunRecord(
                        run_id=session.session_id,
                        conversation_id=_CONVERSATION,
                        metadata={'title': session.first_prompt[:80], 'where': 'Logfire'},
                        started_at=session.last,
                    ),
                )
        return sorted(runs.values(), key=lambda run: run.started_at)

    async def run_history(self, *, run_id: str) -> list[ModelMessage]:
        if self.local is not None:
            try:
                return (await self.local.get(conversation_id=run_id)).messages
            except LookupError:
                pass
        if self.logfire is None:
            return []
        cached = self._loaded.get(run_id)
        if cached is not None and time.monotonic() - cached[0] < _CACHE_SECONDS:
            return cached[1]
        try:
            loaded = await self.logfire.load(run_id)
        except (SessionCorrupt, PermissionError, OSError):
            return []  # A damaged session is left out of search; resuming it says why.
        messages = loaded.messages if loaded is not None else []
        self._loaded[run_id] = (time.monotonic(), messages)
        return messages


def session_search(
    local: SqliteConversationStore | None, logfire: LogfireSessions | None
) -> Sequence[AbstractCapability[None]]:
    """`search_conversation_history` over your sessions, and the instruction that says how to reopen one."""
    return (
        ConversationSearch[None](
            SessionsHistorySource(local=local, logfire=logfire), scope='all', id='clai2_session_search'
        ),
        Capability[None](instructions=INSTRUCTIONS, id='clai2_session_search_hint'),
    )
