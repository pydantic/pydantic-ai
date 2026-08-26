"""Live evidence for [#5869](https://github.com/pydantic/pydantic-ai/issues/5869): Anthropic models copy
`<thinking>` tags they find in assistant history into their own user-visible answers.

These are recorded observations of *model* behavior, not assertions about our mapping — the histories are
built from plain text, a `TextPart` for the assistant channel and a `UserPromptPart` for the user one,
shaped exactly the way `AnthropicModel._map_message` renders a `ThinkingPart` that can't ride the native
reasoning channel. That keeps them valid as evidence no matter how the mapper is changed.

What the cassettes show:

* the leak is driven by *cumulative format precedent*, not one occurrence — 1, 2, 4 and 5 prior
  reasoning-shaped turns never triggered it, six did, matching the reporter's "90+ turns, self-sustaining"
* annotating the tag suppresses the copying on `claude-sonnet-4-6` but **not** on `claude-opus-4-5`, which
  copies the annotation verbatim — attribute included — onto reasoning it just produced itself
* the *channel* is what settles it: the identical attributed block moved into a `user` turn stops the
  copying on the model that copies every assistant-turn variant, which is the design this ships

Because the behavior is stochastic, each cassette pins one recorded instance of a rate measured over
repeated live runs. The rates come from a 134-call sweep at 6 prior turns with no system prompt: bare
`6/9` sonnet and `4/4` opus, annotated `0/9` sonnet and `4/4` opus, and `0/5` sonnet / `0/4` opus for the
no-tag control, and `0/16` for the same attributed block moved into the user turn. Recording matched
those rates — the `sonnet-bare-leaks` cassette took three attempts to catch the leak, every other case
recorded its outcome first try.
"""

from __future__ import annotations as _annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from ..._inline_snapshot import snapshot
from ...conftest import try_import
from ..conftest import AnthropicModelFactory, RequestCapture

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel

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

_THINKING_TAG = re.compile(r'</?(?:assistant_)?thinking\b[^>]*>')


def _history(case: Case) -> list[ModelMessage]:
    """Six prior turns whose reasoning rides `case.block` in `case.channel`, or is omitted entirely."""
    messages: list[ModelMessage] = []
    for question, reasoning, answer in _PRIOR_TURNS:
        messages.append(ModelRequest(parts=[UserPromptPart(content=question)]))
        rendered = None if case.block is None else case.block.format(reasoning=reasoning)
        if rendered is not None and case.channel == 'user':
            messages.append(ModelRequest(parts=[UserPromptPart(content=rendered)]))
        parts: list[TextPart] = []
        if rendered is not None and case.channel == 'assistant':
            parts.append(TextPart(content=rendered))
        parts.append(TextPart(content=answer))
        messages.append(ModelResponse(parts=parts))
    return messages


@dataclass
class Case:
    id: str
    model_name: str
    block: str | None
    """How each prior turn's reasoning is wrapped, with a `{reasoning}` placeholder; `None` omits it."""
    channel: Literal['assistant', 'user']
    """The turn the block rides in. `'user'` is the shape Pydantic AI ships for this family."""
    expected_tags: list[str]
    """The thinking tags the model emitted in its own user-visible answer."""
    sent_tags: set[tuple[str, str]]
    """Every (role, tag) pair the recorded request carried, read off the wire rather than the cassette."""
    sent_length: int
    """Length of the serialized outbound messages, so an edit to the history can't replay green."""


_BARE = '<thinking>\n{reasoning}\n</thinking>'
_NOTED = '<thinking note="carried over from earlier in this conversation">\n{reasoning}\n</thinking>'
_ATTRIBUTED = '<thinking by="openai">\n{reasoning}\n</thinking>'
# The shape Pydantic AI ships for this family: the same reasoning, attributed, in a user turn.
_CARRIED = '<assistant_thinking by="openai">\n{reasoning}\n</assistant_thinking>'


CASES = [
    Case(
        # The report in https://github.com/pydantic/pydantic-ai/issues/5869, live: bare tags, 6/9 across
        # repeated runs.
        'sonnet-bare-leaks',
        'claude-sonnet-4-6',
        _BARE,
        'assistant',
        snapshot(['<thinking>', '</thinking>']),
        snapshot({('assistant', '</thinking>'), ('assistant', '<thinking>')}),
        snapshot(3971),
    ),
    Case(
        # An annotated tag suppresses the copying on sonnet: 0/9.
        'sonnet-annotated-clean',
        'claude-sonnet-4-6',
        _NOTED,
        'assistant',
        snapshot([]),
        snapshot(
            {
                ('assistant', '</thinking>'),
                ('assistant', '<thinking note="carried over from earlier in this conversation">'),
            }
        ),
        snapshot(4307),
    ),
    Case(
        # Opus copies the annotated tag anyway: 4/4, same as bare. The annotation is read as part of the
        # format to imitate, not as a signal that the block isn't the model's own.
        'opus-annotated-leaks',
        'claude-opus-4-5',
        _NOTED,
        'assistant',
        snapshot(['<thinking note="carried over from earlier in this conversation">', '</thinking>']),
        snapshot(
            {
                ('assistant', '</thinking>'),
                ('assistant', '<thinking note="carried over from earlier in this conversation">'),
            }
        ),
        snapshot(4307),
    ),
    Case(
        # And it copies a provenance attribute wholesale, so opus's own reasoning reaches the user labelled
        # as another provider's: 4/4.
        'opus-attributed-leaks',
        'claude-opus-4-5',
        _ATTRIBUTED,
        'assistant',
        snapshot(['<thinking by="openai">', '</thinking>']),
        snapshot({('assistant', '</thinking>'), ('assistant', '<thinking by="openai">')}),
        snapshot(4055),
    ),
    Case(
        # The shipped design, against `opus-attributed-leaks` as its control: same model, same attributed
        # wrapper, same reasoning — only the channel differs, and the copying stops. Recorded once here;
        # the rate behind it is 0/16 across the sweep that chose this placement, against 8/8 for the same
        # content in the assistant turn.
        'opus-carried-in-user-turn-clean',
        'claude-opus-4-5',
        _CARRIED,
        'user',
        snapshot([]),
        snapshot({('user', '</assistant_thinking>'), ('user', '<assistant_thinking by="openai">')}),
        snapshot(4175),
    ),
    Case(
        # Control: with no tag anywhere in history opus never emits one, so the copying above is precedent,
        # not house style.
        'opus-no-tags-control',
        'claude-opus-4-5',
        None,
        'assistant',
        snapshot([]),
        snapshot(set()),
        snapshot(2513),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_thinking_tag_mimicry(
    case: Case,
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
):
    """A thinking tag replayed in an *assistant* turn is copied into the answer the user reads; in a
    *user* turn it is not."""
    model: AnthropicModel = anthropic_model(case.model_name, capture=True)
    agent = Agent(model)
    result = await agent.run(_FOLLOWUP, message_history=_history(case))

    assert _THINKING_TAG.findall(result.output) == case.expected_tags

    # These cassettes match on method and URI only, so an edited history replays the old answer and stays
    # green while the recording stops being evidence for it. Reading the request off the wire pins what
    # each recorded answer is evidence *for*: the wrapper, the turn it rides in, and the history's size.
    sent: list[dict[str, Any]] = request_capture.body('/v1/messages')['messages']
    assert {
        (message['role'], tag)
        for message in sent
        for block in message['content']
        for tag in _THINKING_TAG.findall(block['text'])
    } == case.sent_tags
    assert len(json.dumps(sent)) == case.sent_length
