"""`Researcher` under the durable execution engines, checked at construction time."""

import pytest

# `Researcher` builds `WebFetch(local=True)`, which needs the `web-fetch` extra.
pytest.importorskip('markdownify')

try:
    from pydantic_ai.durable_exec.dbos import DBOSDurability
    from pydantic_ai.durable_exec.prefect import PrefectDurability
    from pydantic_ai.durable_exec.temporal import TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio, prefect, and dbos are not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import SubAgent
from pydantic_ai_harness.researcher import Researcher


def test_researcher_constructs_under_temporal_durability() -> None:
    Agent(TestModel(), name='researcher_temporal', capabilities=[Researcher(), TemporalDurability()])


def test_researcher_constructs_under_prefect_durability() -> None:
    Agent(TestModel(), name='researcher_prefect', capabilities=[Researcher(), PrefectDurability()])


def test_researcher_constructs_under_dbos_durability() -> None:
    Agent(TestModel(), name='researcher_dbos', capabilities=[Researcher(), DBOSDurability()])


def _researcher(subagent_name: str) -> Researcher[None]:
    sub_researcher = SubAgent(Agent(name=subagent_name, description='Research a focused sub-question on the web'))
    return Researcher(subagents=[sub_researcher])


def test_two_researchers_pin_combine_semantics() -> None:
    """Two researchers share the `researcher_instructions` id; same-id capabilities collide by design."""
    with pytest.raises(UserError, match='researcher_instructions'):
        Agent(  # pyright: ignore[reportCallIssue, reportArgumentType]
            TestModel(),
            name='researcher_twice',
            capabilities=[_researcher('researcher_a'), _researcher('researcher_b')],
        )
