"""Fleet miner CLI: `uv run --env-file .env python -m hackathon.fleet_miner --help`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import logfire

from . import extract as extract_mod, fetch, impact as impact_mod, patterns as patterns_mod, policy as policy_mod
from .models import Impact, Proposal, ProposalsDoc, UserPrompt, Window, daily_trend, pseudonymize, span_of
from .llm_cache import CACHE_DIR, USAGE
from .variables import CONTROL_VARIABLE, VARIABLE, VariablesClient

HERE = Path(__file__).parent
RECLUSTER_EVERY = timedelta(hours=24)
EXTRACT_VERSION = 1
VERBOSE = True


def say(*values: object) -> None:
    """Progress output; `--watch` turns it off and prints one line per cycle instead."""
    if VERBOSE:
        print(*values)


def _min_users(value: str) -> int:
    n = int(value)
    if n < 2:
        raise argparse.ArgumentTypeError('at least 2: a suggestion never comes from one person')
    return n


def _since(value: str) -> datetime:
    if match := re.fullmatch(r'(\d+)([hd])', value):
        amount, unit = int(match[1]), match[2]
        return datetime.now(UTC) - (timedelta(hours=amount) if unit == 'h' else timedelta(days=amount))
    return datetime.fromisoformat(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='fleet_miner', description=__doc__)
    parser.add_argument('--since', type=_since, default='7d', help='window start: 24h, 7d or an ISO timestamp')
    parser.add_argument(
        '--min-users',
        type=_min_users,
        default=3,
        help='distinct users a pattern needs (at least 2: never suggest from one person)',
    )
    parser.add_argument('--fixture', type=Path, help='read prompts from this JSON instead of Logfire')
    parser.add_argument('--save-fixture', type=Path, help='also save the fetched prompts here')
    parser.add_argument(
        '--agent-runs',
        action='store_true',
        help='also mine user text on agent runs from before clai2 tagged prompt sources (may include agent-dispatched prompts)',
    )
    parser.add_argument('--dry-run', action='store_true', help='print, write nothing')
    parser.add_argument(
        '--reuse-clusters', action='store_true', help='re-draft from the cached groups, skip clustering'
    )
    parser.add_argument(
        '--only', nargs='+', choices=['prompts', 'policy'], default=['prompts', 'policy'], help='what to mine'
    )
    parser.add_argument(
        '--drop-unreviewed-policy',
        action='store_true',
        help='remove policy proposals nobody accepted or dismissed yet (e.g. after changing how policy ids are made)',
    )
    parser.add_argument(
        '--dismiss', action='append', default=[], metavar='ID=REASON', help='dismiss a pending or stale proposal'
    )
    parser.add_argument('--drop', action='append', default=[], help='remove this proposal id from the live document')
    parser.add_argument('--out', type=Path, help='write the proposals document to this file')
    parser.add_argument(
        '--base-url', default=os.environ.get('LOGFIRE_CLAI2_BASE_URL', 'https://logfire-eu.pydantic.info')
    )
    gates = parser.add_argument_group('gates for skills and instructions (counted over verified prompts)')
    gates.add_argument('--min-prompts', type=int, default=3, help='verified prompts a rule needs')
    gates.add_argument('--min-sessions', type=int, default=2, help='sessions those prompts come from')
    gates.add_argument('--min-days', type=int, default=2, help='distinct days, unless --min-sessions-one-day is met')
    gates.add_argument(
        '--min-sessions-one-day', type=int, default=3, help='sessions that make evidence from a single day enough'
    )
    gates.add_argument('--min-confidence', type=float, default=0.8, help="validation confidence (braindump's high)")
    gates.add_argument('--min-coherence', type=float, default=0.7, help='validation cluster coherence')
    gates.add_argument('--max-pending', type=int, default=5, help='pending skills and instructions at most')
    risk = parser.add_argument_group('gates for policy findings')
    risk.add_argument('--min-policy-sessions', type=int, default=2, help='sessions a flagged risky action needs')
    risk.add_argument('--max-pending-policy', type=int, default=3, help='pending policy suggestions at most')
    parser.add_argument('--extract-model', default='gateway/anthropic:claude-sonnet-5-5')
    parser.add_argument('--pattern-model', default='gateway/anthropic:claude-sonnet-5-5')
    parser.add_argument('--recluster', action='store_true', help='cluster everything from scratch (default: daily)')
    parser.add_argument('--watch', type=float, metavar='MINUTES', help='re-run every N minutes, one line per cycle')
    parser.add_argument(
        '--cache', type=Path, help='extraction cache file (default: one per extraction model under .cache/)'
    )
    return parser


async def main(args: argparse.Namespace) -> tuple[str, int]:
    """One mining run: returns a one-line summary and the number of new suggestions."""
    USAGE.reset()
    if args.fixture and not args.dry_run:
        # Fixture output must never reach the live variable.
        raise SystemExit('--fixture runs are offline: add --dry-run (and --out to keep the result).')
    now = datetime.now(UTC)
    if args.fixture:
        prompts = fetch.load_fixture(args.fixture)
    else:
        prompts = await fetch.fetch_prompts(
            os.environ.get('LOGFIRE_CLAI2_READ_TOKEN') or os.environ['LOGFIRE_CLAI2_API_KEY'],
            base_url=args.base_url,
            since=args.since,
            include_untagged_agent_runs=args.agent_runs,
            store=fetch.PromptStore(CACHE_DIR / 'prompts.json'),
        )
        if args.save_fixture:
            fetch.save_fixture(args.save_fixture, prompts)
    users = {p.user for p in prompts}
    say(f'{len(prompts)} prompts from {len(users)} users in {len({p.trace_id for p in prompts})} sessions')

    api_key = os.environ.get('LOGFIRE_CLAI2_API_KEY')
    client = VariablesClient(api_key, base_url=args.base_url) if api_key else None
    existing_doc = await client.read() if client else None
    existing = existing_doc.proposals if existing_doc else []

    def keep(p: Proposal) -> bool:
        unreviewed_policy = p.kind == 'policy' and p.status in ('pending', 'stale')
        return p.id not in args.drop and not (args.drop_unreviewed_policy and unreviewed_policy)

    existing = [p for p in existing if keep(p)]
    mine_prompts, mine_policy = 'prompts' in args.only, 'policy' in args.only and not args.fixture
    start = min((p.timestamp for p in prompts), default=args.since if isinstance(args.since, datetime) else now)
    # Tool calls first: they are policy's input, and the agent's actions before a prompt are context for extraction.
    calls = await _fetch_calls(args) if not args.fixture else []
    drafted, found, stale_reasons = (
        await _mine_prompts(args, prompts, existing, calls, window=(start, now)) if mine_prompts else ([], [], {})
    )
    identifiers = patterns_mod.personal_identifiers(prompts + _calls_as_prompts(calls))
    if mine_policy:
        # Once someone accepted or dismissed a rule for a risk category, don't re-propose that category with
        # another action (ids are `policy-<category>-<action>`).
        reviewed = {
            p.id.rsplit('-', 1)[0] for p in existing if p.kind == 'policy' and p.status in ('accepted', 'dismissed')
        }
        result = await policy_mod.mine_policy(
            calls,
            prompts=prompts,
            model=args.pattern_model,
            gates=policy_mod.PolicyGates(
                min_users=args.min_users,
                min_sessions=args.min_policy_sessions,
                max_pending=args.max_pending_policy,
            ),
            identifiers=identifiers,
            window=(start, now),
            classes_path=CACHE_DIR / 'call_classes.json',
        )
        say('\nRisky tool calls (oversight):')
        for key, stats in result.oversight.items():
            say(f'   {key}: {stats}')
        drafted += [
            p
            for p in result.proposals
            if p.rule is None or p.id.rsplit('-', 1)[0] not in reviewed or any(e.id == p.id for e in existing)
        ]
        # Policy ids carry the action (`policy-<category>-<action>`); the reasons are per category or per id.
        for p in existing:
            if p.kind == 'policy':
                category = p.id.rsplit('-', 1)[0]
                if reason := result.stale_reasons.get(p.id) or result.stale_reasons.get(category):
                    stale_reasons[p.id] = reason
    users_by_span = {p.span_id: p.user or p.span_id for p in prompts} | {c.span_id: c.user for c in calls}
    # Evidence of earlier proposals can point at spans this run didn't fetch: look those up, so developer numbers
    # stay truthful ("2 developers", not one number per unknown span).
    if not args.fixture and existing:
        earlier = [e for p in existing for e in p.evidence if e.span_id not in users_by_span]
        if earlier:
            users_by_span |= await fetch.fetch_span_users(
                os.environ.get('LOGFIRE_CLAI2_READ_TOKEN') or os.environ['LOGFIRE_CLAI2_API_KEY'],
                base_url=args.base_url,
                span_ids={e.span_id for e in earlier},
                since=min(e.timestamp for e in earlier) - timedelta(hours=1),
            )
    # Unify machines with emails where any record links them, as for the prompts themselves.
    emails_by_host = {p.host: p.user for p in prompts if p.host and p.user and not p.user.startswith('host:')}
    users_by_span = {
        span: emails_by_host.get(user.removeprefix('host:'), user) if user.startswith('host:') else user
        for span, user in users_by_span.items()
    }
    dismissals = dict(item.split('=', 1) for item in args.dismiss)
    pattern_spans = patterns_mod.load_prior_spans()
    impacts = await impact_mod.compute_impacts(
        [p for p in existing if p.status == 'accepted'],
        prompts=prompts,
        pattern_spans=pattern_spans,
        calls=calls,
        window_start=start,
        now=now,
        read_token=None if args.fixture else (os.environ.get('LOGFIRE_CLAI2_READ_TOKEN') or api_key),
        base_url=args.base_url,
    )
    finish = Finisher(users_by_span, identifiers, dismissals, impacts)
    stale_kinds = ({'skill', 'instruction'} if mine_prompts else set()) | ({'policy'} if mine_policy else set())
    merged, actions = patterns_mod.merge(existing, drafted, stale_kinds=stale_kinds, stale_reasons=stale_reasons)
    for proposal in drafted:
        state = 'emerging' if proposal.emerging else actions[proposal.id]
        say(f'\n--- [{state}] {proposal.kind} `{proposal.name}` -> {proposal.suggested_tier}')
        say(proposal.description)
        say(proposal.text)
        if proposal.value_reason:
            say(f'Why: {proposal.value_reason}')
    for id_, action in actions.items():
        if action == 'stale':
            say(f'\n--- [stale] `{id_}`: {stale_reasons.get(id_, "no longer found")}')

    doc = finish(
        ProposalsDoc(generated_at=now, window=Window(start=start, end=now), min_users=args.min_users, proposals=merged)
    )
    if args.out:
        args.out.write_text(doc.model_dump_json(indent=2))
        say(f'\nWrote {args.out}')
    new_count = sum(a == 'new' for a in actions.values())
    summary = (
        f'{len(prompts)} prompts, {len(users)} users, {len(found)} clusters, '
        f'{sum(p.kind != "policy" and not p.emerging for p in drafted)} suggested '
        f'(+{sum(p.kind != "policy" and p.emerging for p in drafted)} emerging), {len(calls)} tool calls, '
        f'{new_count} new suggestion(s); {USAGE}'
    )
    if args.dry_run:
        return f'{summary}; dry run', new_count
    if client is None:
        say(f'\nNo LOGFIRE_CLAI2_API_KEY: not writing `{VARIABLE}`.')
        return f'{summary}; no API key', new_count

    def build(current: ProposalsDoc | None) -> ProposalsDoc:
        # Merge onto what is live right now: statuses written while we were mining win.
        fresh, _ = patterns_mod.merge(
            [p for p in (current.proposals if current else []) if keep(p)],
            drafted,
            stale_kinds=stale_kinds,
            stale_reasons=stale_reasons,
        )
        return finish(doc.model_copy(update={'proposals': fresh}))

    async with client:
        written, wrote = await client.update(build, unchanged=same_content)
    statuses = {s: sum(p.status == s for p in written.proposals) for s in ('pending', 'stale', 'accepted', 'dismissed')}
    status_text = ' '.join(f'{k}={v}' for k, v in statuses.items())
    result = f'{summary}; {status_text}; ' + ('WROTE live doc (verified)' if wrote else 'unchanged, not written')
    say(f'\n{result}')
    return result, new_count


async def _mine_prompts(
    args: argparse.Namespace,
    prompts: list[UserPrompt],
    existing: list[Proposal],
    calls: list[policy_mod.ToolCall],
    *,
    window: tuple[datetime, datetime],
) -> tuple[list[Proposal], list[patterns_mod.Pattern], dict[str, str]]:
    """Extract, cluster, validate, gate, rank and draft. Also returns `status_reason`s for pending ids that fail."""
    # The version is bumped whenever the extraction shape changes, so cached extractions are redone.
    model_key = re.sub(r'\W+', '_', args.extract_model)
    cache_path = args.cache or CACHE_DIR / f'extract-v{EXTRACT_VERSION}-{model_key}.json'
    extractions = await extract_mod.extract_rules(
        prompts, model=args.extract_model, cache=extract_mod.ExtractionCache(cache_path), calls=calls
    )
    say(f'Extraction: {extract_mod.summary(extractions)}')
    clusters_path = patterns_mod.CLUSTERS_PATH
    age = datetime.now(UTC).timestamp() - clusters_path.stat().st_mtime if clusters_path.exists() else None
    if args.reuse_clusters:
        found = patterns_mod.load_patterns(prompts)
    elif args.recluster or age is None or age > RECLUSTER_EVERY.total_seconds():
        # Full re-clustering (first run, on request, or daily): groups can merge and split as the fleet evolves.
        found = await patterns_mod.find_patterns(prompts, extractions, model=args.pattern_model, existing=existing)
        patterns_mod.assign_stable_ids(
            found,
            existing,
            prior_spans=patterns_mod.load_prior_spans(),
            single_rule_spans=patterns_mod.single_rule_spans(extractions),
        )
        patterns_mod.save_clustered_spans(set(patterns_mod.items_of(prompts, extractions)))
        patterns_mod.save_patterns(found)
    else:
        cached = patterns_mod.load_patterns(prompts)
        found = await patterns_mod.update_patterns(
            prompts, extractions, cached, model=args.pattern_model, existing=existing
        )
        patterns_mod.save_patterns(found)
    gates = patterns_mod.Gates(
        min_users=args.min_users,
        min_prompts=args.min_prompts,
        min_sessions=args.min_sessions,
        min_days=args.min_days,
        min_sessions_one_day=args.min_sessions_one_day,
        min_confidence=args.min_confidence,
        min_coherence=args.min_coherence,
        max_pending=args.max_pending,
    )
    candidates = await patterns_mod.validate_patterns(found, extractions, model=args.pattern_model, gates=gates)
    # Impact and stable ids follow the verified prompts where a cluster was validated.
    patterns_mod.record_pattern_spans([c.pattern for c in candidates])
    reviewed = {p.id for p in existing if p.status in ('accepted', 'dismissed')}
    pending, emerging = patterns_mod.rank_and_cap(candidates, gates, reviewed=reviewed)
    say('\nClusters (verified developers / sessions / prompts / corrections, score):')
    order = {id(c): n for n, c in enumerate([*pending, *emerging])}
    for c in sorted(candidates, key=lambda c: order.get(id(c), len(order))):
        p = c.pattern
        state = (
            'PENDING '
            if c in pending
            else 'emerging'
            if c in emerging
            else 'reviewed'
            if c.failure is None
            else '        '
        )
        say(
            f' {state} {len(p.users)}u {len(p.sessions)}s {len(p.prompts)}p {c.corrections}c {c.score:.2f}  {p.id}: '
            f'{c.validation.rule if c.validation and c.validation.rule else p.pattern}'
        )
        say(f'          {c.failure or (c.validation.value_reason if c.validation else "")}')
    stale_reasons = {c.pattern.id: c.failure for c in candidates if c.failure}
    clustered = {c.pattern.id for c in candidates}
    prior_spans = patterns_mod.load_prior_spans()
    for p in existing:
        if p.kind in ('skill', 'instruction') and p.status in ('pending', 'stale') and p.id not in clustered:
            spans = {span_of(i) for i in prior_spans.get(p.id, set())} | {e.span_id for e in p.evidence}
            stale_reasons[p.id] = patterns_mod.unmatched_reason(spans, extractions)
    drafted = await patterns_mod.draft_proposals(
        [*pending, *emerging],
        model=args.pattern_model,
        emerging={c.pattern.id for c in emerging},
        window_teams={p.team for p in prompts if p.team},
        window_repos={p.repo_slug for p in prompts if p.repo_slug},
    )
    by_id = {c.pattern.id: c.pattern for c in [*pending, *emerging]}
    for proposal in drafted:
        events = ((u.timestamp, u.user or u.span_id) for u in by_id[proposal.id].prompts)
        proposal.trend = daily_trend(events, *window)
    return drafted, found, stale_reasons


async def _fetch_calls(args: argparse.Namespace) -> list[policy_mod.ToolCall]:
    calls = await policy_mod.fetch_tool_calls(
        os.environ.get('LOGFIRE_CLAI2_READ_TOKEN') or os.environ['LOGFIRE_CLAI2_API_KEY'],
        base_url=args.base_url,
        since=args.since,
        store_path=CACHE_DIR / 'tool_calls.json',
    )
    commands = sum(1 for c in calls if c.command)
    say(f'\n{len(calls)} tool calls ({commands} shell commands) from {len({c.user for c in calls})} users')
    for group in policy_mod.risk_groups(calls):
        say(f'   risk {group.key}: {len(group.calls)} calls, {len(group.users)} users')
    return calls


def _calls_as_prompts(calls: list[policy_mod.ToolCall]) -> list[UserPrompt]:
    """Tool calls as identity carriers, so handles that show up inside commands count as identifiers too."""
    return [
        UserPrompt(trace_id=c.trace_id, span_id=c.span_id, timestamp=c.timestamp, text=c.command or '', user=c.user)
        for c in calls
    ]


@dataclass
class Finisher:
    """Make the document safe to hand to every clai2 process, and attach the numbers that are only known now."""

    users_by_span: dict[str, str]
    identifiers: set[str]
    dismissals: dict[str, str]
    impacts: dict[str, Impact]

    def __call__(self, doc: ProposalsDoc) -> ProposalsDoc:
        proposals: list[Proposal] = []
        for p in doc.proposals:
            update: dict[str, object] = {
                f: policy_mod.mask_identifiers(getattr(p, f), self.identifiers)
                for f in ('name', 'description', 'text', 'rationale', 'pattern')
            }
            if p.id in self.dismissals and p.status in ('pending', 'stale'):
                update |= {'status': 'dismissed', 'status_reason': self.dismissals[p.id]}
            for f in ('scope_reason', 'value_reason', 'status_reason', 'suggested_instruction'):
                if value := getattr(p, f):
                    update[f] = policy_mod.mask_identifiers(value, self.identifiers)
            if p.status == 'accepted' and p.id in self.impacts:
                update['impact'] = self.impacts[p.id]
            proposals.append(p.model_copy(update=update, deep=True))
        doc = doc.model_copy(update={'proposals': proposals})
        pseudonymize(doc, self.users_by_span)
        return doc


def same_content(a: ProposalsDoc, b: ProposalsDoc) -> bool:
    """Equal apart from when it was computed: the timestamps every run changes."""

    def stable(doc: ProposalsDoc) -> str:
        data = doc.model_dump(mode='json', exclude={'generated_at': True, 'window': {'end'}})
        for p in data['proposals']:
            if p.get('impact'):
                p['impact'].pop('computed_at', None)
                p['impact'].pop('days_after', None)
                p['impact'].pop('after_per_day', None)
                p['impact'].pop('days_before', None)
                p['impact'].pop('before_per_day', None)
                p['impact'].pop('follow_up_prompts_avoided_estimate', None)
        return json.dumps(data, sort_keys=True)

    return stable(a) == stable(b)


POLL = timedelta(seconds=30)


def _ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def watch(args: argparse.Namespace) -> None:
    """Mine every `--watch` minutes, and right away whenever the UI's Run now button sets `requested_at`.

    The control variable is `{requested_at, requested_by}` (written by the UI) plus what this loop writes back:
    `{started_at, finished_at, status: running|done|error, message}`. The UI's fields are always preserved.
    """
    global VERBOSE
    VERBOSE = False
    api_key = os.environ['LOGFIRE_CLAI2_API_KEY']
    control = VariablesClient(api_key, base_url=args.base_url)
    description = 'Fleet miner control for clai2 (hackathon): Run now requests from the UI, run status from the miner.'

    async def set_status(**fields: object) -> None:
        def build(current: dict[str, object] | None) -> dict[str, object]:
            base: dict[str, object] = {'requested_at': None, 'requested_by': None}
            return (
                base | (current or {}) | {k: v.isoformat() if isinstance(v, datetime) else v for k, v in fields.items()}
            )

        try:
            await control.update_json(CONTROL_VARIABLE, build, description=description)
        except Exception as exc:  # a status write (e.g. a 503 from Logfire) must not stop the demo loop
            print(f'{datetime.now(UTC):%H:%M:%S} status write error: {type(exc).__name__}: {exc}'[:200], flush=True)

    if await control.read_json(CONTROL_VARIABLE) is None:
        print(f'creating `{CONTROL_VARIABLE}`', flush=True)
        await set_status(started_at=None, finished_at=None, status=None, message=None)

    last_started: datetime | None = None
    next_scheduled = datetime.now(UTC)
    while True:
        now = datetime.now(UTC)
        try:
            requested = _ts((await control.read_json(CONTROL_VARIABLE) or {}).get('requested_at'))
        except Exception as exc:  # a flaky poll must not stop the demo loop
            print(f'{now:%H:%M:%S} poll error: {type(exc).__name__}: {exc}'[:200], flush=True)
            requested = None
        by_request = requested is not None and (last_started is None or requested > last_started)
        if by_request or now >= next_scheduled:
            last_started, next_scheduled = now, now + timedelta(minutes=args.watch)
            args.since = _since(args.since_text)
            await set_status(started_at=now, finished_at=None, status='running', message=None)
            try:
                line, new = await main(args)
                await set_status(
                    finished_at=datetime.now(UTC),
                    status='done',
                    message=f'{new} new suggestion{"" if new == 1 else "s"}',
                )
            except Exception as exc:  # keep the loop alive; the next cycle retries
                line = f'error: {type(exc).__name__}: {exc}'[:300]
                await set_status(finished_at=datetime.now(UTC), status='error', message=line)
            print(f'{now:%H:%M:%S} [{"run now" if by_request else "scheduled"}] {line}', flush=True)
        await asyncio.sleep(POLL.total_seconds())


if __name__ == '__main__':
    logfire.configure(send_to_logfire='if-token-present', service_name='fleet-miner', console=False)
    logfire.instrument_pydantic_ai()
    parsed = _parser().parse_args()
    if parsed.watch:
        parsed.since_text = sys.argv[sys.argv.index('--since') + 1] if '--since' in sys.argv else '7d'
        asyncio.run(watch(parsed))
    else:
        print(asyncio.run(main(parsed))[0])
