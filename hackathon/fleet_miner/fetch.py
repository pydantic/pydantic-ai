"""Pull what users typed into clai2 out of a Logfire project, or out of a fixture file."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from logfire.query_client import AsyncLogfireQueryClient
from pydantic import TypeAdapter

from .models import UserPrompt

# clai2's `observability` plugin records every submitted prompt as a `prompt submitted` log under the
# `CLAI session` root span, which carries `user.email` (see `pydantic_clai2.ui.telemetry`). Only typed
# prompts count: slash commands (`kind != 'prompt'`) and discarded queued prompts are not user intent.
_SESSIONS = """
    SELECT trace_id,
           max(attributes->>'user.email') AS user_email,
           max(attributes->>'agent_session_id') AS session_id
    FROM records
    WHERE span_name = 'CLAI session'
    GROUP BY trace_id
"""

# The session root only lands once a session ends, and clai2 builds before 2026-10-05 have none: fall back to the
# machine as the user and the process as the session, so open and older sessions still count.
_IDENTITY = """s.user_email, r.otel_resource_attributes->>'host.name' AS host,
       coalesce(s.session_id, 'process:' || (r.otel_resource_attributes->>'service.instance.id')) AS session_id"""

PROMPTS_SQL = f"""
SELECT r.trace_id, r.span_id, r.start_timestamp, r.attributes->>'prompt' AS prompt,
       {_IDENTITY}
FROM records r
LEFT JOIN ({_SESSIONS}) s ON r.trace_id = s.trace_id
WHERE r.span_name = 'prompt submitted'
  AND r.attributes->>'prompt' IS NOT NULL
  AND coalesce(r.attributes->>'kind', 'prompt') = 'prompt'
  AND coalesce(r.attributes->>'route', '') != 'discarded queued'
  -- Records from before clai2 tagged sources are typed by definition: the UI only records what was submitted.
  AND coalesce(r.attributes->>'clai2.prompt.source', 'typed') = 'typed'
ORDER BY r.start_timestamp
"""

# Fallback for clients that ran with `ui_events` off: the user's text parts on agent run spans. Each run's
# `pydantic_ai.all_messages` repeats the conversation so far, so texts are deduplicated per trace below.
AGENT_RUNS_SQL = f"""
SELECT r.trace_id, r.span_id, r.start_timestamp, r.attributes->>'pydantic_ai.all_messages' AS messages,
       {_IDENTITY}
FROM records r
LEFT JOIN ({_SESSIONS}) s ON r.trace_id = s.trace_id
WHERE r.service_name = 'pydantic-clai2'
  AND r.attributes->>'pydantic_ai.all_messages' IS NOT NULL
  AND {{source_filter}}
ORDER BY r.start_timestamp
"""

# Runs clai2 tagged as typed are always mined. Untagged (older) runs may hold plugin-, headless- or subagent-dispatched
# prompts that cannot be told apart from typed ones, so they are only mined on request.
_TYPED_RUNS = "r.attributes->>'clai2.prompt.source' = 'typed'"
_TYPED_OR_UNTAGGED_RUNS = "coalesce(r.attributes->>'clai2.prompt.source', 'typed') = 'typed'"

# Text clai2 itself puts in user messages, which is not something the user typed.
_INJECTED = ('The user ran a local shell command', '<system-reminder>', 'Summary of the conversation so far')

_prompts_adapter = TypeAdapter(list[UserPrompt])


async def fetch_prompts(
    read_token: str, *, base_url: str, since: datetime, include_untagged_agent_runs: bool = False
) -> list[UserPrompt]:
    async with AsyncLogfireQueryClient(read_token, base_url=base_url, timeout=120) as client:
        rows = (await client.query_json_rows(PROMPTS_SQL, min_timestamp=since, limit=10_000))['rows']
        prompts = [_from_prompt_row(row) for row in rows]
        source_filter = _TYPED_OR_UNTAGGED_RUNS if include_untagged_agent_runs else _TYPED_RUNS
        sql = AGENT_RUNS_SQL.format(source_filter=source_filter)
        run_rows = (await client.query_json_rows(sql, min_timestamp=since, limit=2_000))['rows']
        prompts += _from_agent_rows(run_rows, seen={(p.session_id or p.trace_id, p.text) for p in prompts})
    return _resolve_users(prompts)


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
    )


def _from_agent_rows(rows: list[dict[str, Any]], *, seen: set[tuple[str, str]]) -> list[UserPrompt]:
    prompts: list[UserPrompt] = []
    for row in rows:
        messages = row['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)
        key = row.get('session_id') or row['trace_id']
        for message in messages or []:
            if message.get('role') != 'user':
                continue
            for part in message.get('parts', []):
                text = part.get('content') if part.get('type') == 'text' else None
                if not isinstance(text, str) or text.startswith(_INJECTED) or (key, text) in seen:
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
                        source='agent_run',
                    )
                )
    return prompts


def load_fixture(path: Path) -> list[UserPrompt]:
    return _prompts_adapter.validate_json(path.read_bytes())


def save_fixture(path: Path, prompts: list[UserPrompt]) -> None:
    path.write_bytes(_prompts_adapter.dump_json(prompts, indent=2))
