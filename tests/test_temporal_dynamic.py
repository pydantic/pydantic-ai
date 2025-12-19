from datetime import timedelta

import pytest
from temporalio import workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.workflow import ActivityConfig

from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, TemporalAgent, TemporalFunctionToolset
from pydantic_ai.models.test import TestModel


# 1. Define Tool
def echo(x: str) -> str:
    return f'echo: {x}'


# 2. Create Toolset with specific ID
# Using an explicit ID allows us to reference it later if needed,
# though here we pass the toolset object directly.
toolset = FunctionToolset(tools=[echo], id='my_tools')

# 3. Wrap Toolset for Temporal (DouweM pattern)
wrapped_toolset = TemporalFunctionToolset(
    toolset,
    activity_name_prefix='shared_tools',
    activity_config=ActivityConfig(start_to_close_timeout=timedelta(minutes=1)),
    tool_activity_config={},
    deps_type=type(None),
)

# 4. Create base agent for model activity registration
# This agent's activities will be registered in the Worker
# We use a known name "test_agent" so the dynamic agent can share it.
base_model = TestModel()
base_agent = Agent(base_model, name='test_agent')
base_temporal_agent = TemporalAgent(base_agent)


# 5. Define Workflow
@workflow.defn
class DynamicToolWorkflow:
    @workflow.run
    async def run(self, user_prompt: str) -> str:
        # Create agent dynamically within the workflow
        # Note: We are using TestModel which mocks LLM behavior.
        # We explicitly tell TestModel to call the 'echo' tool.
        model = TestModel(call_tools=['echo'])

        # We reuse the name "test_agent" so that the model activities
        # (which are registered under that name) can be found.
        # TEST: Revert to run-time passing
        agent = Agent(
            model,
            name='test_agent',
        )

        temporal_agent = TemporalAgent(agent)

        # Pass wrapped toolset at runtime
        result = await temporal_agent.run(user_prompt, toolsets=[wrapped_toolset])
        return result.output


# 6. Test
pytestmark = pytest.mark.anyio


async def test_dynamic_tool_registration():
    """Test passing a `TemporalWrapperToolset` at runtime to a `TemporalAgent` within a workflow."""
    env = await WorkflowEnvironment.start_local()  # type: ignore[reportUnknownMemberType]
    async with env:
        async with Worker(
            env.client,
            task_queue='test-queue',
            workflows=[DynamicToolWorkflow],
            # Register activities from both base agent and shared toolset
            activities=[
                *base_temporal_agent.temporal_activities,
                *wrapped_toolset.temporal_activities,
            ],
            plugins=[PydanticAIPlugin()],
        ):
            result = await env.client.execute_workflow(
                DynamicToolWorkflow.run,
                args=['test prompt'],
                id='test-workflow-run',
                task_queue='test-queue',
            )

            # Verify tool was called successfully
            # TestModel generates random args, so we just verify echo was called
            # "echo" is the tool return value format: "echo: {arg}"
            assert 'echo' in result
            assert 'echo:' in result
