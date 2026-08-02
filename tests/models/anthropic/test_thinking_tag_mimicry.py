"""Live evidence for [#5869](https://github.com/pydantic/pydantic-ai/issues/5869): Anthropic models copy
`<thinking>` tags they find in assistant history into their own user-visible answers.

These are recorded observations of *model* behavior, not assertions about our mapping — the histories are
built from plain `TextPart`s shaped exactly the way `AnthropicModel._map_message` renders a `ThinkingPart`
that can't ride the native reasoning channel. That keeps them valid as evidence no matter how the mapper
is changed.

What the cassettes show:

* the leak is driven by *cumulative format precedent*, not one occurrence — 1, 2, 4 and 5 prior
  reasoning-shaped turns never triggered it, six did, matching the reporter's "90+ turns, self-sustaining"
* annotating the tag suppresses the copying on `claude-sonnet-4-6` but **not** on `claude-opus-4-5`, which
  copies the annotation verbatim — attribute included — onto reasoning it just produced itself

Because the behavior is stochastic, each cassette pins one recorded instance of a rate measured over
repeated live runs. The rates come from a 134-call sweep at 6 prior turns with no system prompt: bare
`6/9` sonnet and `4/4` opus, annotated `0/9` sonnet and `4/4` opus, and `0/5` sonnet / `0/4` opus for the
no-tag control. Recording matched those rates — the `sonnet-bare-leaks` cassette took three attempts to
catch the leak, every other case recorded its outcome first try.
"""

from __future__ import annotations as _annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from ..._inline_snapshot import snapshot
from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel

if TYPE_CHECKING:
    from pydantic_ai.models.anthropic import AnthropicModel

    ANTHROPIC_MODEL_FIXTURE = Callable[..., AnthropicModel]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

# Six reasoning-shaped prior turns: a deliberation block followed by a terse answer. This is the shortest
# history that reproduced the leak; the same puzzles at 5 turns did not.
_PRIOR_TURNS: list[tuple[str, str, str]] = [
    (
        'if it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?',
        'Classic rate trap. 5 machines / 5 minutes / 5 widgets means each machine makes 1 widget in 5 minutes. '
        'So 100 machines each make 1 widget in 5 minutes -> 100 widgets in 5 minutes. The intuitive but wrong '
        'answer is 100 minutes. I should give the number and one line of why.',
        '5 minutes. Each machine makes one widget every 5 minutes, so 100 machines make 100 widgets in that '
        'same 5 minutes — the rate per machine never changed.',
    ),
    (
        'a bat and a ball cost $1.10 together. the bat costs $1.00 more than the ball. how much is the ball?',
        'Let ball = b. Then bat = b + 1.00. Sum: 2b + 1.00 = 1.10, so 2b = 0.10, b = 0.05. The reflexive wrong '
        'answer is $0.10, which would make the bat $1.10 and the total $1.20.',
        '$0.05. If the ball were $0.10 the bat would be $1.10 and the total $1.20 — too much. At $0.05 the bat '
        'is $1.05 and the total is exactly $1.10.',
    ),
    (
        'i have a 7-litre jug and a 3-litre jug. how do i measure exactly 5 litres?',
        'Need 5 from 7 and 3. Fill 7 (7,0). Pour into 3 (4,3). Empty 3 (4,0). Pour 3 of the 4 across (1,3). '
        'Empty 3 (1,0). Move the 1 (0,1). Fill 7 (7,1). Pour into 3 until full — needs 2 — (5,3). Five litres '
        'in the big jug. Six steps.',
        'Fill the 7 → pour into the 3 (4 left) → empty the 3 → pour the 4 across (1 left) → empty the 3 → move '
        'the 1 into the 3 → refill the 7 → top up the 3 (takes only 2). You now have exactly 5 litres in the '
        '7-litre jug.',
    ),
    (
        'a lily pad doubles every day and covers the lake on day 48. what day is it half covered?',
        'Doubling backwards one step from full: day 47. The trap answer is day 24, which conflates doubling '
        'with linear growth.',
        'Day 47. It doubles daily, so the day before it covers the whole lake it covers exactly half — halving '
        'once, not halving the timeline.',
    ),
    (
        'three friends pay $30 for a room, the clerk refunds $5, the bellhop keeps $2. where is the missing dollar?',
        'There is no missing dollar — the question adds the wrong things. They paid $27 total, and that $27 '
        "already contains the bellhop's $2 plus the $25 the hotel kept. Adding the $2 to the $27 double-counts "
        'it. The correct sum is 25 + 2 + 3 = 30.',
        'Nothing is missing — the question adds the wrong columns. The $27 they paid already includes the '
        "bellhop's $2. The right accounting is $25 (hotel) + $2 (bellhop) + $3 (refunded) = $30.",
    ),
    (
        'a train leaves at 60mph and another at 40mph toward each other 200 miles apart. when do they meet?',
        'Closing speed is 100 mph, distance 200 miles, so 2 hours. Straightforward, no trap here.',
        '2 hours. They close the gap at 60 + 40 = 100 mph, and 200 / 100 = 2.',
    ),
]

# A puzzle the model has to reason through in text, so mimicry has somewhere to show up.
_FOLLOWUP = (
    'ok here is one more: a snail is at the bottom of a 30-foot well. each day it climbs 3 feet and each '
    'night it slides back 2. how many days until it gets out? explain your reasoning.'
)

_THINKING_TAG = re.compile(r'</?thinking\b[^>]*>')


def _history(wrapper: str | None) -> list[ModelMessage]:
    """Six prior turns whose reasoning is wrapped in `wrapper`, or omitted entirely when it is `None`."""
    messages: list[ModelMessage] = []
    for question, reasoning, answer in _PRIOR_TURNS:
        messages.append(ModelRequest(parts=[UserPromptPart(content=question)]))
        parts: list[TextPart] = []
        if wrapper is not None:
            parts.append(TextPart(content=f'<thinking{wrapper}>\n{reasoning}\n</thinking>'))
        parts.append(TextPart(content=answer))
        messages.append(ModelResponse(parts=parts))
    return messages


@dataclass
class Case:
    id: str
    model_name: str
    wrapper: str | None
    expected_tags: list[str]
    """The `<thinking>` tags the model emitted in its own user-visible answer."""


CASES = [
    Case(
        # The #5869 report, live: bare tags, 6/9 across repeated runs.
        'sonnet-bare-leaks',
        'claude-sonnet-4-6',
        '',
        snapshot(['<thinking>', '</thinking>']),
    ),
    Case(
        # An annotated tag suppresses the copying on sonnet: 0/9.
        'sonnet-annotated-clean',
        'claude-sonnet-4-6',
        ' note="carried over from earlier in this conversation"',
        snapshot([]),
    ),
    Case(
        # Opus copies the annotated tag anyway: 4/4, same as bare. The annotation is read as part of the
        # format to imitate, not as a signal that the block isn't the model's own.
        'opus-annotated-leaks',
        'claude-opus-4-5',
        ' note="carried over from earlier in this conversation"',
        snapshot(['<thinking note="carried over from earlier in this conversation">', '</thinking>']),
    ),
    Case(
        # And it copies a provenance attribute wholesale, so opus's own reasoning reaches the user labelled
        # as another provider's: 4/4.
        'opus-attributed-leaks',
        'claude-opus-4-5',
        ' by="openai"',
        snapshot(['<thinking by="openai">', '</thinking>']),
    ),
    Case(
        # Control: with no tag anywhere in history opus never emits one, so the copying above is precedent,
        # not house style.
        'opus-no-tags-control',
        'claude-opus-4-5',
        None,
        snapshot([]),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_thinking_tag_mimicry(case: Case, allow_model_requests: None, anthropic_model: ANTHROPIC_MODEL_FIXTURE):
    """A `<thinking>` tag replayed in assistant history is copied into the model's user-visible answer."""
    agent = Agent(anthropic_model(case.model_name))
    result = await agent.run(_FOLLOWUP, message_history=_history(case.wrapper))

    assert _THINKING_TAG.findall(result.output) == case.expected_tags
