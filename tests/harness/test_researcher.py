import pytest

pytest.importorskip('ddgs')
pytest.importorskip('markdownify')

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, WebFetch, WebSearch
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import WebFetchTool, WebSearchTool

from pydantic_ai_harness.researcher import DEFAULT_RESEARCHER_INSTRUCTIONS, Researcher, researcher_agent
from pydantic_ai_harness.subagents import SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits


def test_researcher_constructs_agent() -> None:
    agent = Agent(TestModel(), capabilities=[Researcher()])

    assert isinstance(agent, Agent)


def test_researcher_agent_is_model_less_and_composed() -> None:
    assert isinstance(researcher_agent, Agent)
    assert researcher_agent.model is None
    assert researcher_agent.name == 'researcher'
    assert any(isinstance(capability, WebFetch) for capability in researcher_agent.root_capability.capabilities)


def test_researcher_unknown_export() -> None:
    import pydantic_ai_harness.researcher

    with pytest.raises(AttributeError, match='has no attribute'):
        pydantic_ai_harness.researcher.__getattr__('missing')


def test_researcher_members_are_transparent() -> None:
    researcher = Researcher()

    assert [type(capability) for capability in researcher.capabilities] == [
        Capability,
        WebSearch,
        WebFetch,
        SubAgents,
        ToolOutputLimits,
    ]
    capability = next(capability for capability in researcher.capabilities if isinstance(capability, Capability))
    assert capability.get_instructions() == [DEFAULT_RESEARCHER_INSTRUCTIONS]
    web_search = next(capability for capability in researcher.capabilities if isinstance(capability, WebSearch))
    assert isinstance(web_search.native, WebSearchTool)
    assert web_search.local is not None
    web_fetch = next(capability for capability in researcher.capabilities if isinstance(capability, WebFetch))
    assert isinstance(web_fetch.native, WebFetchTool)
    assert web_fetch.local is not None
    subagents = next(capability for capability in researcher.capabilities if isinstance(capability, SubAgents))
    delegate = subagents.agents[0].agent
    assert delegate.name == 'researcher'
    assert (
        delegate.description
        == 'Research a focused sub-question on the web and report back with findings and source links'
    )
    delegate_capabilities = delegate.root_capability.capabilities
    delegate_search = next(capability for capability in delegate_capabilities if isinstance(capability, WebSearch))
    delegate_fetch = next(capability for capability in delegate_capabilities if isinstance(capability, WebFetch))
    assert delegate_search.local is not None
    assert delegate_fetch.local is not None
    assert any(isinstance(capability, ToolOutputLimits) for capability in delegate_capabilities)


def test_researcher_threads_instructions() -> None:
    researcher = Researcher(instructions='Custom instructions')

    capability = next(capability for capability in researcher.capabilities if isinstance(capability, Capability))
    assert capability.get_instructions() == ['Custom instructions']


def test_researcher_none_disables_instructions() -> None:
    researcher = Researcher(instructions=None)

    assert not any(isinstance(capability, Capability) for capability in researcher.capabilities)


def test_researcher_empty_subagents_disables_delegation() -> None:
    researcher = Researcher(subagents=[])

    assert not any(isinstance(capability, SubAgents) for capability in researcher.capabilities)


def test_researcher_for_agent_preserves_subclass() -> None:
    researcher = Researcher()
    bound = researcher.for_agent(Agent(TestModel()))

    assert isinstance(bound, Researcher)
