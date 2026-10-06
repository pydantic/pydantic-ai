"""Fleet miner CLI: `uv run --env-file .env python -m hackathon.fleet_miner --help`."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import logfire

from . import facets as facets_mod, fetch, patterns as patterns_mod, policy as policy_mod
from .models import Proposal, ProposalsDoc, UserPrompt, Window, pseudonymize
from .variables import VARIABLE, VariablesClient

HERE = Path(__file__).parent


def _since(value: str) -> datetime:
    if match := re.fullmatch(r'(\d+)([hd])', value):
        amount, unit = int(match[1]), match[2]
        return datetime.now(UTC) - (timedelta(hours=amount) if unit == 'h' else timedelta(days=amount))
    return datetime.fromisoformat(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='fleet_miner', description=__doc__)
    parser.add_argument('--since', type=_since, default='7d', help='window start: 24h, 7d or an ISO timestamp')
    parser.add_argument('--min-users', type=int, default=3, help='distinct users a pattern needs')
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
    parser.add_argument('--facet-model', default='gateway/anthropic:claude-sonnet-5-5')
    parser.add_argument('--pattern-model', default='gateway/anthropic:claude-sonnet-5-5')
    parser.add_argument('--cache', type=Path, help='facet cache file (default: one per facet model under .cache/)')
    return parser


async def main(args: argparse.Namespace) -> None:
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
        )
        if args.save_fixture:
            fetch.save_fixture(args.save_fixture, prompts)
    users = {p.user for p in prompts}
    print(f'{len(prompts)} prompts from {len(users)} users in {len({p.trace_id for p in prompts})} sessions')

    api_key = os.environ.get('LOGFIRE_CLAI2_API_KEY')
    client = VariablesClient(api_key, base_url=args.base_url) if api_key else None
    existing_doc = await client.read() if client else None
    existing = existing_doc.proposals if existing_doc else []

    def keep(p: Proposal) -> bool:
        unreviewed_policy = p.kind == 'policy' and p.status in ('pending', 'stale')
        return p.id not in args.drop and not (args.drop_unreviewed_policy and unreviewed_policy)

    existing = [p for p in existing if keep(p)]
    mine_prompts, mine_policy = 'prompts' in args.only, 'policy' in args.only and not args.fixture
    drafted = await _mine_prompts(args, prompts, existing) if mine_prompts else []
    calls = await _fetch_calls(args) if mine_policy else []
    identifiers = patterns_mod.personal_identifiers(prompts + _calls_as_prompts(calls))
    if mine_policy:
        # Once someone accepted or dismissed a rule for a risk category, don't re-propose that category with
        # another action (ids are `policy-<category>-<action>`).
        reviewed = {
            p.id.rsplit('-', 1)[0] for p in existing if p.kind == 'policy' and p.status in ('accepted', 'dismissed')
        }
        drafted += [
            p
            for p in await policy_mod.mine_policy(
                calls, model=args.pattern_model, min_users=args.min_users, identifiers=identifiers
            )
            if p.rule is None or p.id.rsplit('-', 1)[0] not in reviewed or any(e.id == p.id for e in existing)
        ]
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
    stale_kinds = ({'skill', 'instruction'} if mine_prompts else set()) | ({'policy'} if mine_policy else set())
    merged, actions = patterns_mod.merge(existing, drafted, stale_kinds=stale_kinds)
    for proposal in drafted:
        print(f'\n--- [{actions[proposal.id]}] {proposal.kind} `{proposal.name}` -> {proposal.suggested_tier}')
        print(proposal.description)
        print(proposal.text)
        if proposal.kind == 'policy':
            print(proposal.rationale)
    for id_, action in actions.items():
        if action == 'stale':
            print(f'\n--- [stale] `{id_}` no longer qualifies')

    start = min((p.timestamp for p in prompts), default=args.since if isinstance(args.since, datetime) else now)
    doc = _finalize(
        ProposalsDoc(generated_at=now, window=Window(start=start, end=now), min_users=args.min_users, proposals=merged),
        users_by_span,
        identifiers,
        dismissals,
    )
    if args.out:
        args.out.write_text(doc.model_dump_json(indent=2))
        print(f'\nWrote {args.out}')
    if args.dry_run:
        return
    if client is None:
        print(f'\nNo LOGFIRE_CLAI2_API_KEY: not writing `{VARIABLE}`.')
        return

    def build(current: ProposalsDoc | None) -> ProposalsDoc:
        # Merge onto what is live right now: statuses written while we were mining win.
        fresh, _ = patterns_mod.merge(
            [p for p in (current.proposals if current else []) if keep(p)], drafted, stale_kinds=stale_kinds
        )
        return _finalize(doc.model_copy(update={'proposals': fresh}), users_by_span, identifiers, dismissals)

    async with client:
        written = await client.update(build)
    statuses = {s: sum(p.status == s for p in written.proposals) for s in ('pending', 'stale', 'accepted', 'dismissed')}
    print(f'\nWrote and verified `{VARIABLE}`: {statuses}')


async def _mine_prompts(
    args: argparse.Namespace, prompts: list[UserPrompt], existing: list[Proposal]
) -> list[Proposal]:
    cache_path = args.cache or HERE / '.cache' / f'facets-{re.sub(r"\W+", "_", args.facet_model)}.json'
    facets = (
        {}
        if args.reuse_clusters
        else await facets_mod.extract_facets(prompts, model=args.facet_model, cache=facets_mod.FacetCache(cache_path))
    )
    if facets:
        print(f'{sum(1 for f in facets.values() if f.intent)} prompts carry a reusable intent')
    clusters_path = HERE / '.cache' / 'clusters.json'
    if args.reuse_clusters:
        found = patterns_mod.load_patterns(clusters_path, prompts)
    else:
        found = await patterns_mod.find_patterns(prompts, facets, model=args.pattern_model, existing=existing)
        patterns_mod.assign_stable_ids(found, existing, prior_spans=patterns_mod.load_prior_spans(clusters_path))
        patterns_mod.save_patterns(clusters_path, found)
    print('\nPatterns (distinct users / sessions / score):')
    for p in found:
        mark = '*' if len(p.users) >= args.min_users else ' '
        print(f' {mark} {len(p.users)}u {len(p.sessions)}s {p.score:.2f}  {p.id}: {p.pattern}')
    qualifying = [p for p in found if len(p.users) >= args.min_users]
    if not qualifying:
        print(f'\nNo pattern reached {args.min_users} distinct users.')
        return []
    return await patterns_mod.draft_proposals(qualifying, model=args.pattern_model)


async def _fetch_calls(args: argparse.Namespace) -> list[policy_mod.ToolCall]:
    calls = await policy_mod.fetch_tool_calls(
        os.environ.get('LOGFIRE_CLAI2_READ_TOKEN') or os.environ['LOGFIRE_CLAI2_API_KEY'],
        base_url=args.base_url,
        since=args.since,
    )
    commands = sum(1 for c in calls if c.command)
    print(f'\n{len(calls)} tool calls ({commands} shell commands) from {len({c.user for c in calls})} users')
    for group in policy_mod.risk_groups(calls):
        print(f'   risk {group.key}: {len(group.calls)} calls, {len(group.users)} users')
    return calls


def _calls_as_prompts(calls: list[policy_mod.ToolCall]) -> list[UserPrompt]:
    """Tool calls as identity carriers, so handles that show up inside commands count as identifiers too."""
    return [
        UserPrompt(trace_id=c.trace_id, span_id=c.span_id, timestamp=c.timestamp, text=c.command or '', user=c.user)
        for c in calls
    ]


def _finalize(
    doc: ProposalsDoc, users_by_span: dict[str, str], identifiers: set[str], dismissals: dict[str, str]
) -> ProposalsDoc:
    """Make the document safe to hand to every clai2 process: numbered developers, no identifiers in any text."""
    proposals: list[Proposal] = []
    for p in doc.proposals:
        update: dict[str, object] = {
            f: policy_mod.mask_identifiers(getattr(p, f), identifiers)
            for f in ('name', 'description', 'text', 'rationale', 'pattern')
        }
        if p.id in dismissals and p.status in ('pending', 'stale'):
            update |= {'status': 'dismissed', 'status_reason': dismissals[p.id]}
        proposals.append(p.model_copy(update=update, deep=True))
    doc = doc.model_copy(update={'proposals': proposals})
    pseudonymize(doc, users_by_span)
    return doc


if __name__ == '__main__':
    logfire.configure(send_to_logfire='if-token-present', service_name='fleet-miner', console=False)
    logfire.instrument_pydantic_ai()
    asyncio.run(main(_parser().parse_args()))
