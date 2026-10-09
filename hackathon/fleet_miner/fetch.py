"""Pull what users typed into clai2 out of a Logfire project, or out of a fixture file."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from logfire.query_client import AsyncLogfireQueryClient
from pydantic import TypeAdapter

from .models import UserPrompt

# clai2's `observability` plugin records every submitted prompt as a `prompt submitted` log under the
# `CLAI session` root span, which carries `user.email` (see `pydantic_clai2.ui.telemetry`). Only typed
# prompts count: slash commands (`kind != 'prompt'`) and discarded queued prompts are not user intent.
# Track A tags its own test sessions; they are not anyone's real usage.
NOT_TEST = """coalesce(r.attributes->>'clai2.test', 'false') NOT IN ('true', 'True', '1')
  AND coalesce(r.attributes->>'clai2.team', '') != 'test'"""

_SESSIONS = """
    SELECT trace_id,
           max(attributes->>'user.email') AS user_email,
           max(attributes->>'agent_session_id') AS session_id
    FROM records
    WHERE span_name = 'CLAI session'
    GROUP BY trace_id
"""

# Current clai2 puts `user.email`, `clai2.team` and `agent_session_id` on every record and span itself (as baggage).
# Older records only have them on the `CLAI session` root, which lands once a session ends, and builds before
# 2026-10-05 have no root at all: fall back to the machine as the user and the process as the session.
IDENTITY = """coalesce(r.attributes->>'user.email', s.user_email) AS user_email,
       r.attributes->>'clai2.team' AS team,
       r.attributes->>'clai2.repo_slug' AS repo_slug,
       r.otel_resource_attributes->>'host.name' AS host,
       coalesce(r.attributes->>'agent_session_id', s.session_id,
                'process:' || (r.otel_resource_attributes->>'service.instance.id')) AS session_id"""

PROMPTS_SQL = f"""
SELECT r.trace_id, r.span_id, r.start_timestamp, r.attributes->>'prompt' AS prompt, r.attributes->>'route' AS route,
       {IDENTITY}
FROM records r
LEFT JOIN ({_SESSIONS}) s ON r.trace_id = s.trace_id
WHERE r.span_name = 'prompt submitted'
  AND r.attributes->>'prompt' IS NOT NULL
  AND coalesce(r.attributes->>'kind', 'prompt') = 'prompt'
  AND coalesce(r.attributes->>'route', '') != 'discarded queued'
  -- Records from before clai2 tagged sources are typed by definition: the UI only records what was submitted.
  AND coalesce(r.attributes->>'clai2.prompt.source', 'typed') = 'typed'
  AND {NOT_TEST}
ORDER BY r.start_timestamp
"""

# Fallback for clients that ran with `ui_events` off: the user's text parts on agent run spans. Each run's
# `pydantic_ai.all_messages` repeats the conversation so far, so texts are deduplicated per trace below.
AGENT_RUNS_SQL = f"""
SELECT r.trace_id, r.span_id, r.start_timestamp, r.attributes->>'pydantic_ai.all_messages' AS messages,
       {IDENTITY}
FROM records r
LEFT JOIN ({_SESSIONS}) s ON r.trace_id = s.trace_id
WHERE r.service_name = 'pydantic-clai2'
  AND r.attributes->>'pydantic_ai.all_messages' IS NOT NULL
  AND {{source_filter}}
  AND {NOT_TEST}
ORDER BY r.start_timestamp
"""

# Runs clai2 tagged as typed are always mined. Untagged (older) runs may hold plugin-, headless- or subagent-dispatched
# prompts that cannot be told apart from typed ones, so they are only mined on request.
_TYPED_RUNS = "r.attributes->>'clai2.prompt.source' = 'typed'"
_TYPED_OR_UNTAGGED_RUNS = "coalesce(r.attributes->>'clai2.prompt.source', 'typed') = 'typed'"

# Text clai2 itself puts in user messages, which is not something the user typed.
_INJECTED = ('The user ran a local shell command', '<system-reminder>', 'Summary of the conversation so far')

_prompts_adapter = TypeAdapter(list[UserPrompt])


OVERLAP = timedelta(minutes=30)
"""Re-read this far behind the watermark: spans can land late, and duplicates are dropped by span id."""


class PromptStore:
    """What earlier runs already fetched, so each run only queries records newer than a per-source watermark."""

    def __init__(self, path: Path | None = None):
        self.path = path
        data = json.loads(path.read_text()) if path and path.exists() else {}
        self.watermarks: dict[str, datetime] = {
            k: datetime.fromisoformat(v) for k, v in data.get('watermarks', {}).items()
        }
        self.prompts: dict[str, UserPrompt] = {
            p.span_id: p for p in _prompts_adapter.validate_python(data.get('prompts', []))
        }

    def since(self, source: str, window_start: datetime) -> datetime:
        watermark = self.watermarks.get(source)
        return max(window_start, watermark - OVERLAP) if watermark else window_start

    def add(self, source: str, prompts: list[UserPrompt]) -> int:
        new = [p for p in prompts if p.span_id not in self.prompts]
        for p in prompts:
            self.prompts.setdefault(p.span_id, p)
        if prompts:
            latest = max(p.timestamp for p in prompts)
            self.watermarks[source] = max(latest, self.watermarks.get(source, latest))
        return len(new)

    def save(self, window_start: datetime) -> None:
        if self.path is None:
            return
        kept = [p for p in self.prompts.values() if p.timestamp >= window_start]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    'watermarks': {k: v.isoformat() for k, v in self.watermarks.items()},
                    'prompts': _prompts_adapter.dump_python(kept, mode='json'),
                }
            )
        )


async def fetch_prompts(
    read_token: str,
    *,
    base_url: str,
    since: datetime,
    include_untagged_agent_runs: bool = False,
    store: PromptStore | None = None,
) -> list[UserPrompt]:
    """Typed prompts in the window. With a `store`, only records newer than its watermarks are queried."""
    store = store or PromptStore()
    async with AsyncLogfireQueryClient(read_token, base_url=base_url, timeout=120) as client:
        rows = (await client.query_json_rows(PROMPTS_SQL, min_timestamp=store.since('prompts', since), limit=10_000))[
            'rows'
        ]
        store.add('prompts', [_from_prompt_row(row) for row in rows])
        source_filter = _TYPED_OR_UNTAGGED_RUNS if include_untagged_agent_runs else _TYPED_RUNS
        sql = AGENT_RUNS_SQL.format(source_filter=source_filter)
        source = 'agent_runs_untagged' if include_untagged_agent_runs else 'agent_runs'
        run_rows = (await client.query_json_rows(sql, min_timestamp=store.since(source, since), limit=2_000))['rows']
        seen = {(p.session_id or p.trace_id, p.text) for p in store.prompts.values()}
        store.add(source, _from_agent_rows(run_rows, seen=seen))
        # Identity can arrive after the prompt (a session root lands when the session ends): retry the unknown ones.
        unknown = {p.span_id for p in store.prompts.values() if not p.user and p.timestamp >= since}
        if unknown:
            found = await _span_users(client, unknown, since)
            for span_id, user in found.items():
                if not user.startswith('host:'):
                    store.prompts[span_id].user = user
        # Prompts stored before the miner read `route` (mid-run steering): look them up once.
        unrouted = {
            p.span_id
            for p in store.prompts.values()
            if p.route is None and p.source == 'prompt_submitted' and p.timestamp >= since
        }
        if unrouted:
            routes = await _span_routes(client, unrouted, since)
            for span_id in unrouted:
                store.prompts[span_id].route = routes.get(span_id) or ''
    prompts = [p.model_copy() for p in store.prompts.values() if p.timestamp >= since]
    store.save(since)
    return _resolve_users(sorted(prompts, key=lambda p: p.timestamp))


async def fetch_span_users(read_token: str, *, base_url: str, span_ids: set[str], since: datetime) -> dict[str, str]:
    """Who is behind each of these spans (evidence from earlier runs, outside this run's prompts)."""
    if not span_ids:
        return {}
    async with AsyncLogfireQueryClient(read_token, base_url=base_url, timeout=120) as client:
        return await _span_users(client, span_ids, since)


async def _span_users(client: AsyncLogfireQueryClient, span_ids: set[str], since: datetime) -> dict[str, str]:
    in_list = ', '.join(f"'{s}'" for s in sorted(span_ids) if s.isalnum())
    sql = f"""
SELECT r.span_id, {IDENTITY}
FROM records r
LEFT JOIN ({_SESSIONS}) s ON r.trace_id = s.trace_id
WHERE r.span_id IN ({in_list})
"""
    rows = (await client.query_json_rows(sql, min_timestamp=since, limit=10_000))['rows']
    return {r['span_id']: r.get('user_email') or f'host:{r.get("host")}' for r in rows}


async def _span_routes(client: AsyncLogfireQueryClient, span_ids: set[str], since: datetime) -> dict[str, str]:
    in_list = ', '.join(f"'{s}'" for s in sorted(span_ids) if s.isalnum())
    sql = f"SELECT r.span_id, r.attributes->>'route' AS route FROM records r WHERE r.span_id IN ({in_list})"
    rows = (await client.query_json_rows(sql, min_timestamp=since, limit=10_000))['rows']
    return {r['span_id']: r['route'] for r in rows if r.get('route')}


def _resolve_users(prompts: list[UserPrompt]) -> list[UserPrompt]:
    """Name each prompt's user by email, learning which machine is whose from prompts that know both."""
    emails = {p.host: p.user for p in prompts if p.user and p.host}
    for p in prompts:
        p.user = p.user or emails.get(p.host) or (f'host:{p.host}' if p.host else None)
    return prompts


def _from_prompt_row(row: dict[str, Any]) -> UserPrompt:
    return UserPrompt(
        trace_id=row['trace_id'],
        span_id=row['span_id'],
        timestamp=row['start_timestamp'],
        text=row['prompt'],
        user=row.get('user_email'),
        host=row.get('host'),
        session_id=row.get('session_id'),
        team=row.get('team'),
        repo_slug=row.get('repo_slug'),
        route=row.get('route') or '',
    )


def _from_agent_rows(rows: list[dict[str, Any]], *, seen: set[tuple[str, str]]) -> list[UserPrompt]:
    """The prompt that started each run: its latest user text, not the whole history `all_messages` repeats.

    Reading every user message of every run would re-process the conversation on each turn (quadratic in its length).
    """
    prompts: list[UserPrompt] = []
    for row in rows:
        messages = row['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)
        key = row.get('session_id') or row['trace_id']
        text = _latest_user_text(messages or [])
        if text is None or (key, text) in seen:
            continue
        seen.add((key, text))
        prompts.append(
            UserPrompt(
                trace_id=row['trace_id'],
                span_id=row['span_id'],
                timestamp=row['start_timestamp'],
                text=text,
                user=row.get('user_email'),
                host=row.get('host'),
                session_id=row.get('session_id'),
                team=row.get('team'),
                repo_slug=row.get('repo_slug'),
                source='agent_run',
            )
        )
    return prompts


def _latest_user_text(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get('role') != 'user':
            continue
        texts = [
            p['content']
            for p in message.get('parts', [])
            if p.get('type') == 'text' and isinstance(p.get('content'), str) and not p['content'].startswith(_INJECTED)
        ]
        if texts:
            return '\n'.join(texts)
    return None


def load_fixture(path: Path) -> list[UserPrompt]:
    return _prompts_adapter.validate_json(path.read_bytes())


def save_fixture(path: Path, prompts: list[UserPrompt]) -> None:
    path.write_bytes(_prompts_adapter.dump_json(prompts, indent=2))
