"""How `/team` and the status row summarize a team: its members' states, plans, shutdowns and spend."""

from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.teams import Team, TeamMember
from pydantic_clai2.runtime.tasks import display_name
from pydantic_clai2.ui.rendering.usage_report import format_cost


def usage_text(usage: RunUsage, separator: str = ' · ') -> str:
    """Requests, tokens, and the cost when the models' prices are known: `3 req · 12.5k tok · $0.0123`."""
    tokens = usage.input_tokens + usage.output_tokens
    parts = [f'{usage.requests} req', f'{tokens / 1000:.1f}k tok' if tokens >= 1000 else f'{tokens} tok']
    if usage.cost is not None:
        parts.append(format_cost(usage.cost))
    return separator.join(parts)


def member_line(member: TeamMember) -> str:
    """One `/team` row: name and type, model, state, held item, usage, and a shutdown or plan waiting for review."""
    parts = [
        f'{member.name} ({display_name(member.agent_name)})',
        member.model or 'session model',
        member.state,
    ]
    if member.current_task:
        parts.append(f'#{member.current_task}')
    parts.append(usage_text(member.usage))
    if member.shutdown == 'requested':
        parts.append('shutting down')
    elif member.shutdown == 'done':
        parts.append('shut down')
    line = ' · '.join(parts)
    if member.plan is not None:
        plan = ' '.join(member.plan.plan.split())
        line += f'\n  plan waiting for approval: {plan[:200]}{"…" if len(plan) > 200 else ""}'
    return line


def team_segment(team: Team[object], conversation_id: str) -> str:
    """The status row's team figures: who works and waits, and the whole team's spend, lead included."""
    members = team.owner.members(conversation_id)
    if not members:
        return ''
    queued = sum(record.queued for record in members)
    working = sum(record.status == 'running' for record in members) - queued
    # The status row paints ASCII only.
    waiting = f', {queued} queued' if queued else ''
    return f'team {working} working{waiting}, {usage_text(team.usage(conversation_id), ", ")}'
