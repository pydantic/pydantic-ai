from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.settings import ModelSettings
    from pydantic_evals.dataset import Case, Dataset
    from pydantic_evals.tournament import (
        EVALUATION_INSTRUCTIONS,
        EvalGame,
        EvalPlayer,
        EvalTournament,
        GameResult,
        adaptive_uncertainty_strategy,
        random_sampling_strategy,
        round_robin_strategy,
    )

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
def query_agent() -> Agent[None, GameResult]:
    """Create a test evaluation agent for generating search queries."""
    return Agent(
        model=OpenAIChatModel(
            model_name='qwen2.5:72b',
            provider=OpenAIProvider(base_url='http://localhost:11434/v1'),
        ),
        output_type=str,
        system_prompt='Please generate a concise web search query for the given research topic. Reply with ONLY the query string. Do NOT use quotes.',
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


@pytest.fixture
def ice_cream_game() -> EvalGame:
    """Provide an EvalGame instance for ice cream flavour comparison."""
    return EvalGame(criterion='Which of the two ice cream flavours A or B is more creative?')


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


@pytest.mark.vcr()
async def test_random_sampling_strategy(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame, evaluation_agent: Agent[None, GameResult], allow_model_requests: None) -> None:
    """Test the random sampling tournament strategy."""
    players_with_scores = await random_sampling_strategy(
        players=ice_cream_players,
        game=ice_cream_game,
        agent=evaluation_agent,
        model_settings=MODEL_SETTINGS,
        fraction_of_games=0.3,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, 'score')
        assert isinstance(player.score, float)
        assert player.score is not None


@pytest.mark.vcr()
async def test_round_robin_strategy(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame, evaluation_agent: Agent[None, GameResult], allow_model_requests: None) -> None:
    """Test the round robin tournament strategy."""
    players_with_scores = await round_robin_strategy(
        players=ice_cream_players,
        game=ice_cream_game,
        agent=evaluation_agent,
        model_settings=MODEL_SETTINGS,
        number_of_rounds=1,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, 'score')
        assert isinstance(player.score, float)
        assert player.score is not None


@pytest.mark.vcr()
async def test_adaptive_uncertainty_strategy(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame, evaluation_agent: Agent[None, GameResult], allow_model_requests: None) -> None:
    """Test the adaptive uncertainty tournament strategy."""
    players_with_scores = await adaptive_uncertainty_strategy(
        players=ice_cream_players,
        game=ice_cream_game,
        agent=evaluation_agent,
        model_settings=MODEL_SETTINGS,
        max_standard_deviation=1.0,
        alpha=0.01,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, 'score')
        assert isinstance(player.score, float)
        assert player.score is not None


@pytest.mark.vcr()
async def test_evaltournament(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame, evaluation_agent: Agent[None, GameResult], allow_model_requests: None) -> None:
    """Test the EvalTournament class."""
    tournament = EvalTournament(players=ice_cream_players, game=ice_cream_game)

    assert len(tournament.players) == len(ice_cream_players)
    assert tournament.game.criterion == ice_cream_game.criterion

    # Test player retrieval
    player = tournament.get_player_by_idx(1)
    assert player is not None
    assert player.item == ice_cream_players[1].item

    # Test the default strategy
    players_with_scores = await tournament.run(
        agent=evaluation_agent,
        model_settings=MODEL_SETTINGS,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, 'score')
        assert isinstance(player.score, float)
        assert player.score is not None

    # Test the random sampling strategy
    players_with_scores = await tournament.run(
        agent=evaluation_agent,
        model_settings=MODEL_SETTINGS,
        strategy=random_sampling_strategy,
        fraction_of_games=0.3,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, 'score')
        assert isinstance(player.score, float)
        assert player.score is not None


@pytest.mark.vcr()
async def test_evaltournament_usecase(tmp_path: Path, query_agent: Agent[None, str], evaluation_agent: Agent[None, GameResult], allow_model_requests: None) -> None:
    """Use case for EvalTournament, EvalGame and EvalPlayer classes.

    The code demonstrates how the evaluation framework can be used in practice. It is not intended as test for individual components.
    In this use case, we are provided with a list of topics. The objective is to generate creative web search queries for these topics.
    We have a baseline implementation in the `main` branch and a novel implementation in some `feature` branch. In this simple example,
    the implementations differ merely in the prompt (`prompt_baseline` vs. `prompt_novel`) and temperature. We want to check whether the
    novel implementation does indeed generate more creative queries.

    The use case proceeds in three steps:
    (1) We generate an evaluation `Dataset` containing the topics.
    (2) We run the baseline implementation and store the generated queries in the dataset. This code could be run as part of the
        CI/CD pipeline whenever the `main` branch changes.
    (3) We run the novel implementation, score both baseline and novel queries in one go using a Bradley-Terry tournament,
        and check whether the scores have improved.
    """

    # Path to store the evaluation dataset
    path_out = tmp_path / 'dataset.json'

    # (1) Generate Cases and serialise them

    topics = [
        'pangolin trafficking networks',
        'molecular gastronomy',
        'dark kitchen economics',
        'nano-medicine delivery systems',
        'Streisand effect dynamics',
        'social cooling phenomenon',
        'Anne Brorhilke',
        'bioconcrete self-healing',
        'bacteriophage therapy revival',
    ]

    cases: list[Case[dict[str, str], type[None], Any]] = []
    for idx, topic in enumerate(topics):
        case = Case(
            name=f'case_{idx:03d}',
            inputs={'topic': topic},
        )
        cases.append(case)
    dataset: Dataset[dict[str, str], type[None], Any] = Dataset[dict[str, str], type[None], Any](cases=cases)
    dataset.to_file(path_out)

    # (2) Generate base line model outputs

    dataset = Dataset[dict[str, str], type[None], Any].from_file(path_out)
    cases_new: list[Case[dict[str, str], type[None], Any]] = []
    for case in dataset.cases:
        prompt_baseline = f"Please generate a query for the research topic: <TOPIC>{case.inputs['topic']}</TOPIC>"
        async with query_agent:
            result = await query_agent.run(
                user_prompt=prompt_baseline,
                model_settings=MODEL_SETTINGS,
            )

        case_new = Case(
            name=case.name,
            inputs={'topic': case.inputs['topic'], 'query': result.output},
        )
        cases_new.append(case_new)
    dataset_new: Dataset[dict[str, str], type[None], Any] = Dataset[dict[str, str], type[None], Any](cases=cases_new)
    dataset_new.to_file(path_out)

    # (3) Generate novel model outputs and score them

    dataset = Dataset[dict[str, str], type[None], Any].from_file(path_out)
    players: list[EvalPlayer] = []
    for idx, case in enumerate(dataset.cases):
        prompt_novel = (
            f"Please generate a very creative search query for the research topic: <TOPIC>{case.inputs['topic']}</TOPIC>\n"
            "The query should show genuine originality and interest in the topic. AVOID any generic or formulaic phrases."
        )
        async with query_agent:
            result = await query_agent.run(
                user_prompt=prompt_novel,
                model_settings=ModelSettings(
                    temperature=1.0,
                    timeout=300,
                ),
            )

        player_baseline = EvalPlayer(idx=idx, item=case.inputs['query'])
        player_novel = EvalPlayer(idx=idx + len(dataset.cases), item=result.output)
        players.append(player_baseline)
        players.append(player_novel)

    # Run the Bradley-Terry tournament to score both baseline and novel queries
    game = EvalGame(criterion='Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?')
    tournament = EvalTournament(players=players, game=game)
    await tournament.run(
        agent=evaluation_agent,
        model_settings=MODEL_SETTINGS,
    )

    # Average score for both baseline and novel queries
    scores_baseline = [tournament.get_player_by_idx(idx=i).score or 0.0 for i in range(len(dataset.cases))]
    scores_novel = [tournament.get_player_by_idx(idx=i + len(dataset.cases)).score or 0.0 for i in range(len(dataset.cases))]
    # Not every novel query will have scored higher than the baseline case. But on average the novel queries should have improved scores.
    assert np.mean(scores_novel) > np.mean(scores_baseline)
