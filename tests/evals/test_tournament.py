from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.settings import ModelSettings
    from pydantic_evals.tournament import EVALUATION_INSTRUCTIONS, EvalGame, EvalPlayer, GameResult

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'),
    pytest.mark.anyio,
]

MODEL_SETTINGS = ModelSettings(
    temperature=0.0,  # Model needs to be deterministic for VCR recording to work.
    timeout=300,
)


@pytest.fixture
def evaluation_agent() -> Agent[None, GameResult]:
    """Create a test evaluation agent for tournament games."""
    return Agent(
        model=OpenAIChatModel(
            model_name='qwen2.5:72b',
            provider=OpenAIProvider(base_url='http://localhost:11434/v1'),
        ),
        output_type=GameResult,
        system_prompt=EVALUATION_INSTRUCTIONS,
        retries=5,
        instrument=True,
    )

@pytest.fixture
def ice_cream_players() -> list[EvalPlayer]:
    """Provide a list of EvalPlayer instances with ice cream flavours."""
    return [
        EvalPlayer(idx=0, item='vanilla'),
        EvalPlayer(idx=1, item='chocolate'),
        EvalPlayer(idx=2, item='strawberry'),
        EvalPlayer(idx=3, item='peach'),
        EvalPlayer(idx=4, item='toasted rice & miso caramel ice cream'),
    ]



def test_evalplayer() -> None:
    """Test the EvalPlayer class."""
    player = EvalPlayer(
        idx=42,
        item='toasted rice & miso caramel ice cream',
    )
    assert player.idx == 42
    assert player.item == 'toasted rice & miso caramel ice cream'


@pytest.mark.vcr
async def test_evalgame(ice_cream_players: list[EvalPlayer], evaluation_agent: Agent[None, GameResult], allow_model_requests: None) -> None:
    """Test the EvalGame class."""

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
    assert result[0] in {0, 4} and result[1] in {0, 4}
    assert result[0] != result[1]
    assert result[0] == 4  # Toasted rice & miso caramel ice cream flavour is more creative.