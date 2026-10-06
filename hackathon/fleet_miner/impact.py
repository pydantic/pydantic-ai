"""The "it worked" numbers: what changed for each accepted proposal since it was accepted."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from logfire.query_client import AsyncLogfireQueryClient

from .fetch import NOT_TEST
from .models import Impact, Proposal, UserPrompt
from .policy import ToolCall, matching_segment


def _days(start: datetime, end: datetime) -> float:
    return round(max((end - start).total_seconds(), 0) / 86_400, 2)


def _per_day(count: int, days: float) -> float | None:
    # Under an hour of data says nothing per day; report the count alone.
    return round(count / days, 2) if days >= 1 / 24 else None


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


async def compute_impacts(
    accepted: Sequence[Proposal],
    *,
    prompts: list[UserPrompt],
    pattern_spans: dict[str, set[str]],
    calls: list[ToolCall],
    window_start: datetime,
    now: datetime,
    read_token: str | None,
    base_url: str,
) -> dict[str, Impact]:
    """Impact per accepted proposal id.

    Skills and instructions: prompts in this window that belong to the proposal's pattern (the clusters this run
    and earlier runs mapped to its id, so the same matching as clustering), before vs after `accepted_at`, plus the
    developers whose runs reported the item active. Policy: matching tool calls before vs after, plus the rule's
    `policy decision` records since acceptance by outcome.
    """
    by_span = {p.span_id: p for p in prompts}
    impacts: dict[str, Impact] = {}
    async with _Telemetry(read_token, base_url) if read_token else _Null() as client:
        for proposal in accepted:
            accepted_at = proposal.accepted_at
            if accepted_at is None:
                continue
            days_before, days_after = _days(window_start, accepted_at), _days(accepted_at, now)
            if proposal.kind == 'policy' and proposal.rule and proposal.rule.match.command:
                glob = proposal.rule.match.command
                times = [c.timestamp for c in calls if c.command and matching_segment(c.command, glob)]
                decisions, decision_users = await _decisions(client, proposal.rule.name, accepted_at)
                extra: dict[str, object] = {'decisions': decisions, 'decision_users': decision_users}
            else:
                times = [by_span[s].timestamp for s in pattern_spans.get(proposal.id, set()) if s in by_span]
                extra = {'users_with_item': await _users_with_item(client, proposal.name, accepted_at)}
            before = sum(t < accepted_at for t in times)
            after = len(times) - before
            before_rate, after_rate = _per_day(before, days_before), _per_day(after, days_after)
            if proposal.kind != 'policy' and before_rate is not None and after_rate is not None:
                extra['follow_up_prompts_avoided_estimate'] = max(0, round((before_rate - after_rate) * days_after))
            impacts[proposal.id] = Impact(
                computed_at=now,
                accepted_at=accepted_at,
                days_before=days_before,
                days_after=days_after,
                before_count=before,
                after_count=after,
                before_per_day=before_rate,
                after_per_day=after_rate,
                **extra,  # pyright: ignore[reportArgumentType]
            )
    return impacts


async def _users_with_item(client: _Telemetry | _Null, name: str, since: datetime) -> int | None:
    sql = f"""
SELECT count(DISTINCT coalesce(r.attributes->>'user.email', r.otel_resource_attributes->>'host.name')) AS users
FROM records r
WHERE r.attributes->>'clai2.fleet.active' LIKE {_quoted(f'%{name}%')}
  AND {NOT_TEST}
"""
    rows = await client.rows(sql, since)
    return int(rows[0]['users']) if rows else None


async def _decisions(
    client: _Telemetry | _Null, rule: str, since: datetime
) -> tuple[dict[str, int] | None, int | None]:
    sql = f"""
SELECT r.attributes->>'clai2.policy.outcome' AS outcome, count(*) AS n,
       count(DISTINCT coalesce(r.attributes->>'user.email', r.otel_resource_attributes->>'host.name')) AS users
FROM records r
WHERE r.attributes->>'clai2.policy.rule' = {_quoted(rule)}
  AND {NOT_TEST}
GROUP BY 1
"""
    rows = await client.rows(sql, since)
    if rows is None:
        return None, None
    return {str(r['outcome']): int(r['n']) for r in rows}, sum(int(r['users']) for r in rows)


class _Null:
    """Stand-in when there is no read token (fixture runs): no telemetry, so no numbers rather than zeros."""

    async def __aenter__(self) -> _Null:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def rows(self, sql: str, since: datetime) -> None:
        return None


class _Telemetry:
    def __init__(self, read_token: str, base_url: str):
        self._client = AsyncLogfireQueryClient(read_token, base_url=base_url, timeout=120)

    async def __aenter__(self) -> _Telemetry:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.__aexit__(None, None, None)

    async def rows(self, sql: str, since: datetime) -> list[dict[str, object]]:
        return (await self._client.query_json_rows(sql, min_timestamp=since, limit=1_000))['rows']
