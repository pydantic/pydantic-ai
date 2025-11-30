from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.builtin_tools import AbstractBuiltinTool, WebSearchTool, WebSearchUserLocation
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings


@dataclass
class UserContext:
    location: str | None


async def prepared_web_search(ctx: RunContext[UserContext]) -> WebSearchTool | None:
    if not ctx.deps.location:
        return None

    return WebSearchTool(
        search_context_size='medium',
        user_location=WebSearchUserLocation(city=ctx.deps.location),
    )


class InspectToolsModel(Model):
    def __init__(self):
        self.captured_tools: list[AbstractBuiltinTool] = []

    @property
    def model_name(self) -> str:
        return 'inspect-tools-model'

    @property
    def system(self) -> str:
        return 'test'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.captured_tools = model_request_parameters.builtin_tools
        return ModelResponse(parts=[TextPart('OK')])


async def test_dynamic_builtin_tool_configured():
    model = InspectToolsModel()
    assert model.system == 'test'
    agent = Agent(model, builtin_tools=[prepared_web_search], deps_type=UserContext)

    user_context = UserContext(location='London')
    await agent.run('Hello', deps=user_context)

    tools = model.captured_tools
    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, WebSearchTool)
    assert tool.user_location is not None
    assert tool.user_location.get('city') == 'London'
    assert tool.search_context_size == 'medium'


async def test_dynamic_builtin_tool_omitted():
    model = InspectToolsModel()
    agent = Agent(model, builtin_tools=[prepared_web_search], deps_type=UserContext)

    user_context = UserContext(location=None)
    await agent.run('Hello', deps=user_context)

    tools = model.captured_tools
    assert len(tools) == 0


async def test_mixed_static_and_dynamic_builtin_tools():
    model = InspectToolsModel()

    static_tool = WebSearchTool(search_context_size='low')
    agent = Agent(model, builtin_tools=[static_tool, prepared_web_search], deps_type=UserContext)

    # Case 1: Dynamic tool returns None
    await agent.run('Hello', deps=UserContext(location=None))
    assert len(model.captured_tools) == 1
    assert model.captured_tools[0] == static_tool

    # Case 2: Dynamic tool returns a tool
    await agent.run('Hello', deps=UserContext(location='Paris'))
    assert len(model.captured_tools) == 2
    assert model.captured_tools[0] == static_tool
    dynamic_tool = model.captured_tools[1]
    assert isinstance(dynamic_tool, WebSearchTool)
    assert dynamic_tool.user_location is not None
    assert dynamic_tool.user_location.get('city') == 'Paris'


def sync_dynamic_tool(ctx: RunContext[UserContext]) -> WebSearchTool:
    """Verify that synchronous functions work."""
    return WebSearchTool(search_context_size='low')


async def test_sync_dynamic_tool():
    model = InspectToolsModel()
    agent = Agent(model, builtin_tools=[sync_dynamic_tool], deps_type=UserContext)

    await agent.run('Hello', deps=UserContext(location='London'))

    tools = model.captured_tools
    assert len(tools) == 1
    assert isinstance(tools[0], WebSearchTool)
    assert tools[0].search_context_size == 'low'


async def test_dynamic_tool_in_run_call():
    """Verify dynamic tools can be passed to agent.run()."""
    model = InspectToolsModel()
    agent = Agent(model, deps_type=UserContext)

    await agent.run('Hello', deps=UserContext(location='Berlin'), builtin_tools=[prepared_web_search])

    tools = model.captured_tools
    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, WebSearchTool)
    assert tool.user_location is not None
    assert tool.user_location.get('city') == 'Berlin'
