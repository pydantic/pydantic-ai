# /// script
# dependencies = [
#   "temporalio",
#   "logfire",
# ]
# ///
import asyncio
import random
from datetime import timedelta

from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.runtime import OpenTelemetryConfig, Runtime, TelemetryConfig
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.temporal import (
        TemporalSettings,
        initialize_temporal,
        temporalize_agent,
    )
    from pydantic_ai.toolsets.function import FunctionToolset

    initialize_temporal()

    temporal_settings = TemporalSettings(start_to_close_timeout=timedelta(seconds=60))

    def get_uv_index(location: str) -> int:
        return 3

    toolset = FunctionToolset(tools=[get_uv_index])
    mcp_server = MCPServerStdio(
        'python',
        ['-m', 'tests.mcp_server'],
        timeout=20,
    )

    model = OpenAIModel('gpt-4o')
    # my_agent = Agent(model=model, instructions='be helpful', toolsets=[toolset, mcp_server])
    my_agent = Agent(model=model, instructions='be helpful', tools=[get_uv_index], toolsets=[mcp_server])
    activities = temporalize_agent(my_agent, temporal_settings)


def init_runtime_with_telemetry() -> Runtime:
    # import logfire

    # logfire.configure(send_to_logfire=True, service_version='0.0.1', console=False)
    # logfire.instrument_pydantic_ai()
    # logfire.instrument_httpx(capture_all=True)

    # Setup SDK metrics to OTel endpoint
    return Runtime(telemetry=TelemetryConfig(metrics=OpenTelemetryConfig(url='http://localhost:4318')))


# Basic workflow that logs and invokes an activity
@workflow.defn
class MyAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await my_agent.run(prompt)).output


async def main():
    client = await Client.connect(
        'localhost:7233',
        interceptors=[TracingInterceptor()],
        data_converter=pydantic_data_converter,
        runtime=init_runtime_with_telemetry(),
    )

    async with Worker(
        client,
        task_queue='my-agent-task-queue',
        workflows=[MyAgentWorkflow],
        activities=activities,
    ):
        result = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
            MyAgentWorkflow.run,
            'what is 2 plus the UV Index in Mexico City? and what is the product name?',
            id=f'my-agent-workflow-id-{random.random()}',
            task_queue='my-agent-task-queue',
        )
        print(f'Result: {result!r}')


if __name__ == '__main__':
    asyncio.run(main())
