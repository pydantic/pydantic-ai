"""Test A2A with dependency injection via deps_factory."""

import anyio
import httpx
import pytest
from asgi_lifespan import LifespanManager
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, TextPart as TextPartMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from .conftest import try_import

with try_import() as imports_successful:
    from fasta2a.client import A2AClient
    from fasta2a.schema import Message, TextPart, Task, is_task

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='fasta2a not installed'),
    pytest.mark.anyio,
]


async def test_a2a_with_deps_factory():
    """Test that deps_factory enables agents with dependencies to work with A2A."""

    # 1. Define a simple dependency class
    @dataclass
    class Deps:
        user_name: str
        multiplier: int = 2

    # 2. Create a model that returns output based on deps
    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # This function doesn't have access to deps, so it just returns a placeholder
        if info.output_tools:
            # Return a simple string result using the output tool
            args_json = '{"response": "Result computed with deps"}'
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])
        else:
            # No output tools, just return text
            return ModelResponse(parts=[TextPartMessage(content='Result computed with deps')])

    model = FunctionModel(model_function)
    agent = Agent(model=model, deps_type=Deps, output_type=str)

    # 3. Add a system prompt that uses deps
    @agent.system_prompt
    def add_user_info(ctx: RunContext[Deps]) -> str:
        return f"The user's name is {ctx.deps.user_name} with multiplier {ctx.deps.multiplier}"

    # 4. Create deps_factory that reads from task metadata
    def create_deps(task: Task) -> Deps:
        metadata = task.get('metadata', {})
        return Deps(user_name=metadata.get('user_name', 'DefaultUser'), multiplier=metadata.get('multiplier', 2))

    # 5. Create A2A app with deps_factory
    app = agent.to_a2a(deps_factory=create_deps)

    # 6. Test the full flow
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            # Send task with metadata
            message = Message(role='user', parts=[TextPart(text='Process this', kind='text')])
            response = await a2a_client.send_message(message=message, metadata={'user_name': 'Alice', 'multiplier': 5})
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert is_task(result), 'Expected Task response'
            task_id = result['id']

            # Wait for task completion
            task = None
            for _ in range(10):  # Max 10 attempts
                response = await a2a_client.get_task(task_id)
                if 'result' in response:
                    task = response['result']
                    if task['status']['state'] in ('completed', 'failed'):
                        break
                await anyio.sleep(0.1)

            # Verify the result
            assert task is not None
            if task['status']['state'] == 'failed':
                print(f'Task failed. Full task: {task}')
            assert task['status']['state'] == 'completed'
            assert 'artifacts' in task
            artifacts = task['artifacts']
            assert len(artifacts) == 1
            part = artifacts[0]['parts'][0]
            assert part['kind'] == 'text'
            assert part['text'] == 'Result computed with deps'


async def test_a2a_without_deps_factory():
    """Test that agents without deps still work when no deps_factory is provided."""

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.output_tools:
            args_json = '{"response": "Hello from agent"}'
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])
        else:
            return ModelResponse(parts=[TextPartMessage(content='Hello from agent')])

    model = FunctionModel(model_function)
    # Agent with no deps_type
    agent = Agent(model=model, output_type=str)

    # Create A2A app without deps_factory
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(role='user', parts=[TextPart(text='Hello', kind='text')])
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert is_task(result), 'Expected Task response'
            task_id = result['id']

            # Wait for completion
            task = None
            for _ in range(10):
                response = await a2a_client.get_task(task_id)
                if 'result' in response:
                    task = response['result']
                    if task['status']['state'] == 'completed':
                        break
                await anyio.sleep(0.1)

            assert task is not None
            assert task['status']['state'] == 'completed'
            assert 'artifacts' in task
            part = task['artifacts'][0]['parts'][0]
            assert part['kind'] == 'text'
            assert part['text'] == 'Hello from agent'
