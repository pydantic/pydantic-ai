"""Fleet miner CLI: `uv run --env-file .env python -m hackathon.fleet_miner --help`."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import logfire

from . import facets as facets_mod, fetch, patterns as patterns_mod
from .models import ProposalsDoc, Window
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
    parser.add_argument('--agent-runs', action='store_true', help='also mine user text on agent run spans')
    parser.add_argument('--dry-run', action='store_true', help='print, write nothing')
    parser.add_argument(
        '--include-proposals', type=Path, help='also merge in the proposals of this document (e.g. a fixture run)'
    )
    parser.add_argument('--out', type=Path, help='write the proposals document to this file')
    parser.add_argument('--base-url', default=os.environ.get('LOGFIRE_CLAI2_BASE_URL', 'https://logfire-eu.pydantic.info'))
    parser.add_argument('--facet-model', default='gateway/anthropic:claude-sonnet-5-5')
    parser.add_argument('--pattern-model', default='gateway/anthropic:claude-sonnet-5-5')
    parser.add_argument('--cache', type=Path, help='facet cache file (default: one per facet model under .cache/)')
    return parser


async def main(args: argparse.Namespace) -> None:
    now = datetime.now(UTC)
    if args.fixture:
        prompts = fetch.load_fixture(args.fixture)
    else:
        prompts = await fetch.fetch_prompts(
            os.environ.get('LOGFIRE_CLAI2_READ_TOKEN') or os.environ['LOGFIRE_CLAI2_API_KEY'],
            base_url=args.base_url,
            since=args.since,
            include_agent_runs=args.agent_runs,
        )
        if args.save_fixture:
            fetch.save_fixture(args.save_fixture, prompts)
    users = {p.user for p in prompts}
    print(f'{len(prompts)} prompts from {len(users)} users in {len({p.trace_id for p in prompts})} sessions')

    api_key = os.environ.get('LOGFIRE_CLAI2_API_KEY')
    client = VariablesClient(api_key, base_url=args.base_url) if api_key else None
    existing_doc = await client.read() if client else None
    existing = existing_doc.proposals if existing_doc else []

    cache_path = args.cache or HERE / '.cache' / f'facets-{re.sub(r"\W+", "_", args.facet_model)}.json'
    facets = await facets_mod.extract_facets(prompts, model=args.facet_model, cache=facets_mod.FacetCache(cache_path))
    print(f'{sum(1 for f in facets.values() if f.intent)} prompts carry a reusable intent')

    found = await patterns_mod.find_patterns(prompts, facets, model=args.pattern_model, existing=existing)
    print('\nPatterns (distinct users / sessions / score):')
    for p in found:
        mark = '*' if len(p.users) >= args.min_users else ' '
        print(f' {mark} {len(p.users)}u {len(p.sessions)}s {p.score:.2f}  {p.id}: {p.pattern}')

    qualifying = [p for p in found if len(p.users) >= args.min_users]
    if not qualifying:
        print(f'\nNo pattern reached {args.min_users} distinct users.')
    drafted = await patterns_mod.draft_proposals(qualifying, model=args.pattern_model) if qualifying else []
    included = ProposalsDoc.model_validate_json(args.include_proposals.read_bytes()).proposals if args.include_proposals else []
    merged, actions = patterns_mod.merge(existing, drafted + included)
    for proposal in drafted:
        print(f'\n--- [{actions[proposal.id]}] {proposal.kind} `{proposal.name}` -> {proposal.suggested_tier}')
        print(proposal.description)
        print(proposal.text)

    start = min((p.timestamp for p in prompts), default=args.since if isinstance(args.since, datetime) else now)
    doc = ProposalsDoc(generated_at=now, window=Window(start=start, end=now), proposals=merged)
    if args.out:
        args.out.write_text(doc.model_dump_json(indent=2))
        print(f'\nWrote {args.out}')
    if args.dry_run:
        return
    if client is None:
        print(f'\nNo LOGFIRE_CLAI2_API_KEY: not writing `{VARIABLE}`.')
        return
    async with client:
        await client.write(doc, exists=existing_doc is not None)
    print(f'\nWrote `{VARIABLE}` ({sum(a != "skipped" for a in actions.values())} new or updated proposals).')


if __name__ == '__main__':
    logfire.configure(send_to_logfire='if-token-present', service_name='fleet-miner', console=False)
    logfire.instrument_pydantic_ai()
    asyncio.run(main(_parser().parse_args()))
