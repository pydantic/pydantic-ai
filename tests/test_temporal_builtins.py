from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, ModelSettings, TextPart, WebSearchTool
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.models import Model, ModelRequestParameters

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        PydanticAIWorkflow,
        TemporalAgent,
    )
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

with workflow.unsafe.imports_passed_through():
    import annotated_types  # pyright: ignore[reportUnusedImport] # noqa: F401

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.xdist_group(name='temporal'),
]

TEMPORAL_PORT = 7243
TASK_QUEUE = 'pydantic-ai-agent-task-queue'

Agent.instrument_all()


class _BuiltinToolModel(Model):
    """Test model that supports specific builtin tools."""

    SUPPORTED_TOOLS: frozenset[type[AbstractBuiltinTool]] = frozenset()

    def __init__(self, *, response_text: str, model_name: str) -> None:
        self._response_text = response_text
        self._model_name = model_name
        self.last_model_request_parameters: ModelRequestParameters | None = None
        super().__init__()

    @classmethod
    def supported_builtin_tools(cls) -> frozenset[type[AbstractBuiltinTool]]:
        return cls.SUPPORTED_TOOLS

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return 'test'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        self.last_model_request_parameters = model_request_parameters
        return ModelResponse(parts=[TextPart(self._response_text)], model_name=self._model_name)


class _WebSearchOnlyModel(_BuiltinToolModel):
    SUPPORTED_TOOLS = frozenset({WebSearchTool})


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        ui=True,
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
    ) as env:
        yield env


@pytest.fixture
async def client(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(
        f'localhost:{TEMPORAL_PORT}',
        plugins=[PydanticAIPlugin()],
    )


# Model that does NOT support any builtin tools (used as default)
no_builtin_support_model = _BuiltinToolModel(response_text='no builtin support', model_name='no-builtin-test')

# Model that DOES support WebSearchTool (registered as alternate model)
web_search_builtin_model = _WebSearchOnlyModel(response_text='web search response', model_name='web-search-test')

# Agent initialized with model that doesn't support builtins, but has builtin tools configured
agent = Agent(
    no_builtin_support_model,
    builtin_tools=[WebSearchTool()],
)

# TemporalAgent registers an alternate model that DOES support builtins
temporal_agent = TemporalAgent(
    agent,
    name='builtins_in_workflow',
    models={'web_search': web_search_builtin_model},
)


@workflow.defn
class BuiltinsInWorkflow(PydanticAIWorkflow):
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await temporal_agent.run(prompt, model=model_id)
        return result.output


async def test_builtins_in_workflow_with_runtime_model_override(allow_model_requests: None, client: Client):
    """Test that builtin tools work when agent is initialized with a non-supporting model
    but run with a model that does support builtins."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BuiltinsInWorkflow],
        plugins=[AgentPlugin(temporal_agent)],
    ):
        # Run with the model that supports WebSearchTool
        result = await client.execute_workflow(
            BuiltinsInWorkflow.run,
            args=['search for something', 'web_search'],
            id='BuiltinsInWorkflow',
            task_queue=TASK_QUEUE,
        )
        assert result == 'web search response'

    # Verify the web search model received the WebSearchTool in its request parameters
    assert isinstance(web_search_builtin_model.last_model_request_parameters, ModelRequestParameters)
    assert web_search_builtin_model.last_model_request_parameters.builtin_tools
    assert isinstance(web_search_builtin_model.last_model_request_parameters.builtin_tools[0], WebSearchTool)
