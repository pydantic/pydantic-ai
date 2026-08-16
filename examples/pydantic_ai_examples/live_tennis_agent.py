"""Example of a Pydantic AI agent that answers questions about professional tennis.

In this case the idea is a "live tennis" agent — the user can ask what's happening on
the ATP/WTA tours right now, and the agent uses the [Live Tennis
API](https://livetennisapi.com) to look up matches in progress (including who is
serving and whether there's a break point), upcoming fixtures, and player rankings.

This shows how to wrap a plain REST API as Pydantic AI tools using `httpx`: each tool
makes one HTTP call and returns a trimmed-down JSON summary so the LLM sees only the
fields it needs.

The API's free tier (30 requests/minute, 100/day, no card required) is enough to run
this example; set `LIVETENNIS_API_KEY` to your key.

Run with:

    uv run -m pydantic_ai_examples.live_tennis_agent
"""

from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import logfire
from httpx import AsyncClient

from pydantic_ai import Agent, RunContext

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

BASE_URL = 'https://api.livetennisapi.com/api/public/v1'


@dataclass
class Deps:
    client: AsyncClient


tennis_agent = Agent(
    'openai:gpt-5-mini',
    instructions=(
        'You are a tennis assistant. Use the tools to look up live matches, '
        'upcoming fixtures and player rankings, then answer concisely. '
        "If there are no live matches, say so and check what's coming up instead."
    ),
    deps_type=Deps,
    retries=2,
)


def _player_label(player: dict[str, Any]) -> str:
    """Format a player as 'Name (#ranking)', or just the name if unranked."""
    name: str = player['name']
    ranking: int | None = player.get('ranking')
    return f'{name} (#{ranking})' if ranking is not None else name


def _is_break_point(score: dict[str, Any]) -> bool:
    """Whether the returner is one point from breaking serve (never in a tiebreak)."""
    server = score.get('server')
    points: list[str | None] = score.get('points') or []
    if score.get('is_tiebreak') or server not in (1, 2) or len(points) != 2:
        return False
    server_points, returner_points = points[server - 1], points[2 - server]
    return returner_points == 'AD' or (
        returner_points == '40' and server_points not in ('40', 'AD')
    )


def _summarize_match(match: dict[str, Any]) -> dict[str, Any]:
    """Trim a match object down to the fields the LLM needs."""
    p1, p2 = match['players']['p1'], match['players']['p2']
    summary: dict[str, Any] = {
        'match_id': match['id'],
        'tournament': match['tournament'],
        'tour': match['tour'],
        'round': match['round'],
        'players': [_player_label(p1), _player_label(p2)],
    }
    score = match.get('score')
    if score is not None:
        # games is [games_p1, games_p2], each a per-set list; points are tennis
        # strings like '0', '15', '40', 'AD'.
        summary['sets'] = score['sets']
        summary['games'] = score['games']
        summary['points'] = score['points']
        server = score.get('server')
        if server in (1, 2):
            summary['serving'] = _player_label(p1 if server == 1 else p2)
        if score.get('is_tiebreak'):
            summary['tiebreak'] = True
        elif _is_break_point(score):
            summary['break_point'] = True
    return summary


@tennis_agent.tool
async def get_live_matches(
    ctx: RunContext[Deps], tour: str | None = None
) -> list[dict[str, Any]]:
    """Get tennis matches that are in progress right now, with their current score.

    Each match includes who is serving and whether the returner has a break point.

    Args:
        ctx: The context.
        tour: Optionally restrict to one tour: "atp", "wta", "challenger", "itf"
            or "juniors". Omit for all tours.
    """
    params: dict[str, Any] = {'status': 'live'}
    if tour is not None:
        params['tour'] = tour
    r = await ctx.deps.client.get(f'{BASE_URL}/matches', params=params)
    r.raise_for_status()
    return [_summarize_match(match) for match in r.json()['data']]


@tennis_agent.tool
async def get_upcoming_fixtures(
    ctx: RunContext[Deps], tour: str | None = None, limit: int = 10
) -> list[dict[str, Any]]:
    """Get upcoming scheduled fixtures, earliest first.

    Args:
        ctx: The context.
        tour: Optionally restrict to one tour: "atp", "wta", "challenger", "itf"
            or "juniors". Omit for all tours.
        limit: Maximum number of fixtures to return.
    """
    params: dict[str, Any] = {'limit': limit}
    if tour is not None:
        params['tour'] = tour
    r = await ctx.deps.client.get(f'{BASE_URL}/fixtures', params=params)
    r.raise_for_status()
    return [
        {
            'tournament': fixture['tournament'],
            'round': fixture['round'],
            'players': [fixture['player1_name'], fixture['player2_name']],
            # Scheduled start in UTC; null until the order of play assigns a time.
            'start_time': fixture['start_time'],
        }
        for fixture in r.json()['data']
    ]


@tennis_agent.tool
async def search_players(ctx: RunContext[Deps], name: str) -> list[dict[str, Any]]:
    """Search players by name and return their current ranking.

    Args:
        ctx: The context.
        name: Full or partial player name, e.g. "Alcaraz".
    """
    r = await ctx.deps.client.get(
        f'{BASE_URL}/players', params={'search': name, 'limit': 5}
    )
    r.raise_for_status()
    return [
        {
            'player_id': player['id'],
            'name': player['name'],
            'country': player['country'],
            'ranking': player['ranking'],
            'ranking_points': player['ranking_points'],
            'ranking_movement': player['ranking_movement'],
        }
        for player in r.json()['data']
    ]


async def main():
    api_key = os.environ.get('LIVETENNIS_API_KEY')
    if not api_key:
        print(
            'Set LIVETENNIS_API_KEY to run this example — the free tier '
            '(30 requests/minute, 100/day, no card required) is enough: '
            'https://livetennisapi.com/subscribe/free'
        )
        return

    headers = {'Authorization': f'Bearer {api_key}'}
    async with AsyncClient(headers=headers) as client:
        logfire.instrument_httpx(client, capture_all=True)
        deps = Deps(client=client)
        result = await tennis_agent.run(
            'Is there any ATP or WTA tennis on right now? If so, who is serving '
            'and are there any break points? If not, what are the next few '
            "fixtures, and what is Carlos Alcaraz's current ranking?",
            deps=deps,
        )
        print('Response:', result.output)


if __name__ == '__main__':
    asyncio.run(main())
