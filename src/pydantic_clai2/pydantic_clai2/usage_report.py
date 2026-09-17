"""Session tokens and cost, derived from the retained messages every time they are asked for.

Nothing is stored separately: `Session.clear()` resets the numbers because it drops the
messages. Prices come from core, which runs genai-prices on every response it produces and
stores the result in `ModelResponse.usage.cost` (`None` when the model or provider has no
price data). Following core's `RunUsage` convention, sums skip unpriced responses and stay
`None` only when nothing could be priced.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart, ToolReturnPart
from pydantic_ai.usage import RunUsage
from rich.console import Console
from rich.table import Table

from . import theme

_UNKNOWN = 'unknown'


@dataclass(kw_only=True)
class SessionUsage:
    """One `RunUsage` per turn plus the retained-history total; `unpriced` names the models without price data."""

    turns: list[RunUsage]
    total: RunUsage
    unpriced: tuple[str, ...]


def session_usage(messages: Sequence[ModelMessage]) -> SessionUsage:
    """Group responses into turns; a turn starts at a request that carries no tool return or retry.

    A prompt cancelled before any response leaves only its request behind; such turns are dropped.
    """
    turns: list[RunUsage] = []
    unpriced: dict[str, None] = {}
    for message in messages:
        if isinstance(message, ModelRequest):
            if not any(isinstance(part, (ToolReturnPart, RetryPromptPart)) for part in message.parts):
                turns.append(RunUsage())
            continue
        if not turns:
            turns.append(RunUsage())
        turns[-1].incr(message.usage)
        turns[-1].requests += 1
        if message.usage.cost is None:
            unpriced.setdefault(message.model_name or _UNKNOWN, None)
    turns = [turn for turn in turns if turn.requests]
    total = RunUsage()
    for turn in turns:
        total.incr(turn)
    return SessionUsage(turns=turns, total=total, unpriced=tuple(unpriced))


def format_cost(cost: Decimal) -> str:
    """Compact dollars, `$0.0123`; a nonzero amount that rounds away is shown as `<$0.0001` rather than zero."""
    if 0 < cost < Decimal('0.0001'):
        return '<$0.0001'
    return f'${cost:.4f}'


def cost_line(usage: SessionUsage) -> str:
    """The `/cost` line: retained-history total cost and tokens, with the unpriced models named once."""
    if not usage.turns:
        return 'No usage yet.'
    total = usage.total
    cost = _UNKNOWN if total.cost is None else format_cost(total.cost)
    requests = f'{total.requests:,} request' + ('' if total.requests == 1 else 's')
    line = (
        f'Retained history cost {cost}: {total.input_tokens + total.output_tokens:,} tokens'
        f' ({total.input_tokens:,} in, {total.output_tokens:,} out) over {requests}'
    )
    if usage.unpriced:
        line += f'. No price data for {", ".join(usage.unpriced)}; those responses are not counted in costs.'
    return line


def usage_table(usage: SessionUsage) -> Table:
    """Per-turn rows and a totals row. Cache columns appear only when a turn used the cache."""
    cached = any(turn.cache_read_tokens or turn.cache_write_tokens for turn in usage.turns)
    table = Table(header_style=theme.ACCENT, border_style=theme.MUTED)
    for name in ('Turn', 'Requests', 'Input', *(('Cache read', 'Cache write') if cached else ()), 'Output', 'Cost'):
        table.add_column(name, justify='right')
    for index, turn in enumerate(usage.turns, start=1):
        table.add_row(str(index), *_cells(turn, cached=cached))
    table.add_section()
    table.add_row('Total', *_cells(usage.total, cached=cached), style=theme.ACCENT)
    return table


def _cells(usage: RunUsage, *, cached: bool) -> list[str]:
    cells = [f'{usage.requests:,}', f'{usage.input_tokens:,}']
    if cached:
        cells += [f'{usage.cache_read_tokens:,}', f'{usage.cache_write_tokens:,}']
    cells += [f'{usage.output_tokens:,}', _UNKNOWN if usage.cost is None else format_cost(usage.cost)]
    return cells


def usage_command(messages: Sequence[ModelMessage], *, console: Console) -> str:
    """`/usage`: print the table, then return the cost line so the shell prints it underneath."""
    usage = session_usage(messages)
    if usage.turns:
        console.print(usage_table(usage))
    return cost_line(usage)
