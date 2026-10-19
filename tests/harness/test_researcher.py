import importlib
from typing import NoReturn

import pytest

pytest.importorskip('ddgs')
pytest.importorskip('markdownify')

import pydantic_ai.capabilities.local_workspace
import pydantic_ai_harness.researcher
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, LocalWorkspace, WebFetch, WebSearch
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import WebFetchTool, WebSearchTool
from pydantic_ai_harness.researcher import DEFAULT_RESEARCHER_INSTRUCTIONS, Researcher, researcher_agent
from pydantic_ai_harness.subagents import SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits


def test_researcher_constructs_agent() -> None:
    agent = Agent(TestModel(), capabilities=[Researcher()])

    assert isinstance(agent, Agent)


def test_researcher_run_needs_a_workspace() -> None:
    # Spills go to the run's workspace, so a run without one fails at its start, naming `Researcher`.
    agent = Agent(TestModel(), capabilities=[Researcher()])

    with pytest.raises(UserError, match='`Researcher` needs a workspace'):
        agent.run_sync('go')


def test_researcher_agent_is_model_less_and_composed() -> None:
    assert isinstance(researcher_agent, Agent)
    assert researcher_agent.model is None
    assert researcher_agent.name == 'researcher'
    assert any(isinstance(capability, WebFetch) for capability in researcher_agent.root_capability.capabilities)


def _posix_only(*_: object, **__: object) -> NoReturn:
    raise NotImplementedError('LocalWorkspaceBackend requires a POSIX host.')


def test_researcher_agent_runs_on_a_non_posix_host(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without `LocalWorkspace` (POSIX only), the web-only agent spills to a host temp directory instead.
    import pydantic_ai_harness.researcher._agent as module

    monkeypatch.setattr(pydantic_ai.capabilities.local_workspace, 'LocalWorkspaceBackend', _posix_only)
    try:
        agent = importlib.reload(module).researcher_agent
    finally:
        monkeypatch.undo()
        importlib.reload(module)
    assert not any(isinstance(item, LocalWorkspace) for item in agent.root_capability.capabilities)
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')]))
    assert agent.run_sync('go', model=model).output == 'done'


def test_researcher_unknown_export() -> None:

    with pytest.raises(AttributeError, match='has no attribute'):
        pydantic_ai_harness.researcher.__getattr__('missing')


def test_researcher_members_are_transparent() -> None:
    researcher = Researcher()

    assert [type(capability).__name__ for capability in researcher.capabilities] == [
        'RequireWorkspace',
        'Capability',
        'WebSearch',
        'WebFetch',
        'SubAgents',
        'ToolOutputLimits',
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
