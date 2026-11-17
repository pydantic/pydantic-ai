from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.logfire import AgentControl

from ._helpers import Publish


async def test_nameless_agent_normalizes_agent_name() -> None:
    capability = AgentControl()
    agent = Agent(TestModel(), name=' Checkout Assistant #2 ', capabilities=[capability])
    await agent.run('hello')
    assert [
        variable.name
        for variable in capability._variables_by_agent.values()  # pyright: ignore[reportPrivateUsage]
    ] == ['agent__checkout_assistant_2']


async def test_hyphenated_agent_name() -> None:
    capability = AgentControl()
    await Agent(TestModel(), name='pydanty-explorer', capabilities=[capability]).run('hello')
    assert [
        variable.name
        for variable in capability._variables_by_agent.values()  # pyright: ignore[reportPrivateUsage]
    ] == ['agent__pydanty_explorer']


def test_an_inferred_agent_name_is_refused_at_construction() -> None:
    """An agent with no `name` of its own is refused, not silently given the inferred one.

    Pydantic AI fills `Agent.name` in on the first run from the variable the agent was assigned to,
    which would make renaming a local silently repoint the agent at a different managed config. The
    capability is bound (`for_agent`) at construction, before that inference runs, so it can see the
    difference and say so there -- at the line the user would have to change anyway.
    """
    with pytest.raises(UserError, match='would infer one from the variable'):
        Agent(TestModel(), capabilities=[AgentControl()])


def test_explicit_name_rules_unchanged() -> None:
    variable = AgentControl('Checkout-Agent')._variable  # pyright: ignore[reportPrivateUsage]
    assert variable is not None and variable.name == 'agent__Checkout_Agent'
    with pytest.raises(ValueError, match='invalid variable name'):
        AgentControl('Checkout Agent')


async def test_nameless_sources_model_for_model_less_agent(publish: Publish) -> None:
    # A nameless capability can't source the model statically (there is no agent yet), so it hands
    # back a selector Pydantic AI evaluates once it has a `ModelSelectionContext`. That selector
    # derives `agent__solo` from the agent's name and drives a model-less agent.
    publish('solo', {'model': 'test'})
    capability = AgentControl(label='production')
    result = await Agent(None, name='solo', capabilities=[capability]).run('hello')
    assert result.output.startswith('success')
    assert [variable.name for variable in capability._variables_by_agent.values()] == [  # pyright: ignore[reportPrivateUsage]
        'agent__solo'
    ]


async def test_nameless_model_less_agent_without_managed_model_raises() -> None:
    agent = Agent(None, name='unpublished', capabilities=[AgentControl()])
    with pytest.raises(UserError, match='no model to run'):
        await agent.run('hello')


async def test_shared_nameless_capability_derives_a_variable_per_agent() -> None:
    # One capability instance can back several agents -- `SubAgents.shared_capabilities` hands the same
    # object to each -- so each has to read its own `agent__<name>`. A single cache would serve every
    # later agent whichever variable the first one to run happened to build.
    capability = AgentControl()
    await Agent(TestModel(), name='first_agent', capabilities=[capability]).run('hello')
    await Agent(TestModel(), name='second_agent', capabilities=[capability]).run('hello')
    assert sorted(variable.name for variable in capability._variables_by_agent.values()) == [  # pyright: ignore[reportPrivateUsage]
        'agent__first_agent',
        'agent__second_agent',
    ]


async def test_nameless_with_underivable_agent_name_raises() -> None:
    # Normalization is lossy by design, but an empty result is not two agents reading alike -- it is
    # every such agent landing on the bare prefix, naming no agent at all.
    agent = Agent(TestModel(), name='!!!', capabilities=[AgentControl()])
    with pytest.raises(UserError, match='has no letters, digits, or underscores'):
        await agent.run('hello')


async def test_failed_model_selection_leaves_no_resolution_behind() -> None:
    # `wrap_run` consumes the handoff, and a selection that raises never reaches it. Setting it before
    # the model decision would leave this run's instructions, settings and tool overrides in the
    # context for whatever runs next.
    capability = AgentControl()
    with pytest.raises(UserError, match='has no model to run'):
        await Agent(None, name='stranded', capabilities=[capability]).run('hello')
    assert capability._selection_resolved.get() is None  # pyright: ignore[reportPrivateUsage]


def test_a_prefixed_name_warns_at_the_call_site() -> None:
    with pytest.warns(UserWarning, match='added automatically') as caught:
        variable = AgentControl('agent__checkout')._variable  # pyright: ignore[reportPrivateUsage]
    assert variable is not None and variable.name == 'agent__checkout'
    assert caught[0].filename == __file__


async def test_an_agent_whose_name_was_removed_is_refused_at_run_time() -> None:
    agent = Agent(TestModel(), name='checkout', capabilities=[AgentControl()])
    agent.name = None
    with pytest.raises(UserError, match='this agent has none'):
        await agent.run('hello', infer_name=False)
