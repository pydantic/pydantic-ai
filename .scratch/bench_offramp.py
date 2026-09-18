"""Does adding an off-ramp question move the answers to the others?

Runs the real `TypeSafeModel` mapping over the 120 labelled tickets in four shapes, twice each, so
instability between passes can be told apart from a shift between shapes:

  baseline            urgent + area
  reversed options    urgent + area, with area's options in reverse order (order sensitivity)
  action last         urgent + area + action (the off-ramp Choice, asked after the fields)
  action first        action + urgent + area (asked before them)
"""

from __future__ import annotations

import asyncio
import sys
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from cases import CASES  # noqa: E402

URGENT = 'Does this need a reply within the hour?'
AREA = 'Which team owns it?'
ACTION = 'What should happen next?'


class Action(str, Enum):
    answer = 'answer'
    """The fields above can be filled from the ticket alone."""
    escalate_to_human = 'escalate_to_human'
    """A person has to read this before anyone replies."""
    search_docs = 'search_docs'
    """The answer is in the product documentation and should be looked up."""
    refund = 'refund'
    """Money has to be moved back to the customer."""


class Baseline(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description=URGENT)
    area: Literal['billing', 'account', 'bug', 'other'] = Field(description=AREA)


class Reversed(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description=URGENT)
    area: Literal['other', 'bug', 'account', 'billing'] = Field(description=AREA)


class ActionLast(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description=URGENT)
    area: Literal['billing', 'account', 'bug', 'other'] = Field(description=AREA)
    action: Action = Field(description=ACTION)


class ActionFirst(BaseModel):
    """Triage a support ticket."""

    action: Action = Field(description=ACTION)
    urgent: bool = Field(description=URGENT)
    area: Literal['billing', 'account', 'bug', 'other'] = Field(description=AREA)


SHAPES: dict[str, type[BaseModel]] = {
    'baseline': Baseline,
    'reversed options': Reversed,
    'action last': ActionLast,
    'action first': ActionFirst,
}
PASSES = 2


async def run_shape(model: type[BaseModel]) -> list[list[BaseModel]]:
    agent = Agent('typesafe:jev-latest', output_type=model)
    sem = asyncio.Semaphore(8)

    async def one(text: str) -> BaseModel:
        async with sem:
            return (await agent.run(text)).output

    return [await asyncio.gather(*(one(text) for text, _, _ in CASES)) for _ in range(PASSES)]


async def main() -> None:
    results = {label: await run_shape(model) for label, model in SHAPES.items()}
    n = len(CASES)
    print(f'\n{"shape":<18} {"urgent":>8} {"area":>8}   {"flips between passes":>22}   action distribution (pass 1)')
    for label, passes in results.items():
        urgent = sum(getattr(o, 'urgent') == want for o, (_, want, _) in zip(passes[0], CASES))
        area = sum(getattr(o, 'area') == want for o, (_, _, want) in zip(passes[0], CASES))
        flips = sum(
            (getattr(a, 'urgent'), getattr(a, 'area')) != (getattr(b, 'urgent'), getattr(b, 'area'))
            for a, b in zip(passes[0], passes[1])
        )
        dist = ''
        if hasattr(passes[0][0], 'action'):
            counts: dict[str, int] = {}
            for o in passes[0]:
                counts[getattr(o, 'action').value] = counts.get(getattr(o, 'action').value, 0) + 1
            dist = '  '.join(f'{k}={v}' for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f'{label:<18} {urgent:>4}/{n}  {area:>4}/{n}   {flips:>22}   {dist}')

    base = results['baseline'][0]
    print('\nfield-level disagreement with baseline (pass 1 vs pass 1):')
    for label, passes in results.items():
        if label == 'baseline':
            continue
        du = sum(getattr(a, 'urgent') != getattr(b, 'urgent') for a, b in zip(base, passes[0]))
        da = sum(getattr(a, 'area') != getattr(b, 'area') for a, b in zip(base, passes[0]))
        print(f'  {label:<18} urgent differs on {du:>3}   area differs on {da:>3}')

    print('\nbaseline pass-1 misses (text | got | want):')
    for o, (text, want_u, want_a) in zip(base, CASES):
        if (getattr(o, 'urgent'), getattr(o, 'area')) != (want_u, want_a):
            print(f'  {text[:70]:<70} | {getattr(o, "urgent")!s:<5} {getattr(o, "area"):<8} | {want_u!s:<5} {want_a}')


if __name__ == '__main__':
    asyncio.run(main())
