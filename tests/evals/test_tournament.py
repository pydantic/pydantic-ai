from __future__ import annotations as _annotations

import pytest

from pydantic_ai.settings import ModelSettings
from pydantic_evals.tournament import EvalGame, EvalPlayer, evaluation_agent

MODEL_SETTINGS = ModelSettings(
    temperature=0.0,  # Model needs to be deterministic for VCR recording to work.
    timeout=300,
)

def test_evalplayer() -> None:
    """
    Test the EvalPlayer class.
    """

    player = EvalPlayer(
        idx=42,
        item='toasted rice & miso caramel ice cream',
    )
    assert player.idx == 42
    assert player.item == 'toasted rice & miso caramel ice cream'


@pytest.mark.asyncio
async def test_evalgame(ice_cream_players: list[EvalPlayer]) -> None:
    """
    Test the EvalGame class.
    """
 
    game = EvalGame(criterion='Which of the two ice cream flavours A or B is more creative?')
    assert game.criterion == 'Which of the two ice cream flavours A or B is more creative?'

    result = await game.run(
        players=(ice_cream_players[0], ice_cream_players[4]),
        agent=evaluation_agent,
        model_settings=MODEL_SETTINGS,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(r, int) for r in result)
    assert result[0] == 4  # Toasted rice & miso caramel ice cream flavour is more creative.
