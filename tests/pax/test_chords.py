"""Each terminal's bytes for an Enter chord reach the action `tests/pax/chords.py` records for them."""

from __future__ import annotations

import pytest

from tests.pax.chords import CHORDS, Action, Chord
from tests.pax.test_live_prompt import editor


@pytest.mark.parametrize(
    'chord',
    [pytest.param(chord, id=f'{chord.terminal}-{chord.mode}-{chord.chord}') for chord in CHORDS if chord.sends],
)
async def test_the_bytes_a_terminal_sends_reach_their_action(chord: Chord) -> None:
    """Type `a`, the chord, then `b` and Enter: the draft shows whether the chord submitted, broke a line or steered."""
    steered: list[str] = []

    def steer(text: str) -> bool:
        steered.append(text)
        return True

    async with editor() as (live, pipe, _):
        live.steer = steer
        live.ctrl_enter_steers = chord.terminal == 'windows-terminal'
        pipe.send_text(f'a{chord.sends}b\r')
        first = await live.read()
    reached: dict[str, Action] = {'a': 'submit', 'a\nb': 'newline'}
    if steered:
        assert (steered, first) == (['a'], 'b')
        assert chord.action == 'steer'
    else:
        assert reached.get(first) == chord.action, f'the editor read {first!r}'


def test_a_chord_a_terminal_keeps_has_another_way() -> None:
    """A chord the terminal takes for itself leaves its action to another chord on that terminal."""
    reachable = {(chord.terminal, chord.action) for chord in CHORDS if chord.sends}
    kept = [chord for chord in CHORDS if chord.sends is None]
    assert kept, 'the table records at least one chord a terminal keeps'
    assert [chord for chord in kept if (chord.terminal, chord.action) not in reachable] == []
