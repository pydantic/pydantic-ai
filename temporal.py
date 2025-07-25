# /// script
# dependencies = [
#   "temporalio",
#   "logfire",
# ]
# ///
import asyncio
import random
from collections.abc import AsyncIterable
from datetime import timedelta

import logfire
from opentelemetry import trace
from temporalio import workflow
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.runtime import OpenTelemetryConfig, Runtime, TelemetryConfig
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import AgentStreamEvent, HandleResponseEvent
from pydantic_ai.temporal import (
    TemporalSettings,
    initialize_temporal,
    temporalize_agent,
)
from pydantic_ai.toolsets import FunctionToolset

initialize_temporal()


class Deps(TypedDict):
    country: str


def get_country(ctx: RunContext[Deps]) -> str:
    return ctx.deps['country']


toolset = FunctionToolset[Deps](tools=[get_country], id='country')
mcp_server = MCPServerStdio(
    'python',
    ['-m', 'tests.mcp_server'],
    timeout=20,
    id='test',
)


async def event_stream_handler(
    ctx: RunContext[Deps],
    stream: AsyncIterable[AgentStreamEvent | HandleResponseEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info(f'{event=}')


my_agent = Agent(
    'openai:gpt-4o',
    toolsets=[toolset, mcp_server],
    event_stream_handler=event_stream_handler,
    deps_type=Deps,
)

temporal_settings = TemporalSettings(
    start_to_close_timeout=timedelta(seconds=60),
    tool_settings={  # TODO: Allow default temporal settings to be set for all activities in a toolset
        'country': {
            'get_country': TemporalSettings(start_to_close_timeout=timedelta(seconds=110)),
        },
    },
)
activities = temporalize_agent(my_agent, temporal_settings)


TASK_QUEUE = 'pydantic-ai-agent-task-queue'


@workflow.defn
class MyAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> str:
        result = await my_agent.run(prompt, deps=deps)
        return result.output


async def main():
    def init_runtime_with_telemetry() -> Runtime:
        logfire.configure(console=False)
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx(capture_all=True)

        # Setup SDK metrics to OTel endpoint
        return Runtime(telemetry=TelemetryConfig(metrics=OpenTelemetryConfig(url='http://localhost:4318')))

    client = await Client.connect(
        'localhost:7233',
        interceptors=[  # TODO: Use ClientPlugin.configure_client for this
            TracingInterceptor(trace.get_tracer('temporal'))
        ],
        data_converter=pydantic_data_converter,  # TODO: Use ClientPlugin.configure_client for this
        runtime=init_runtime_with_telemetry(),  # TODO: Use ClientPlugin.connect_service_client for this
    )

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MyAgentWorkflow],
        activities=activities,
        workflow_runner=SandboxedWorkflowRunner(  # TODO: Use WorkerPlugin.configure_worker for this, see https://github.com/temporalio/sdk-python/blob/da6616a93e9ee5170842bb5a056e2383e18d07c6/tests/test_plugins.py#L71
            restrictions=SandboxRestrictions.default.with_passthrough_modules(
                'pydantic_ai',
                'logfire',  # TODO: Only if module available?
                # Imported inside `logfire._internal.json_encoder` when running `logfire.info` inside an activity with attributes to serialize
                'attrs',
                # Imported inside `logfire._internal.json_schema` when running `logfire.info` inside an activity with attributes to serialize
                'numpy',  # TODO: Only if module available?
                'pandas',  # TODO: Only if module available?
            ),
        ),
    ):
        output = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
            MyAgentWorkflow.run,
            args=[
                'what is the capital of the capital of the country? and what is the product name?',
                Deps(country='Mexico'),
            ],
            id=f'my-agent-workflow-id-{random.random()}',
            task_queue=TASK_QUEUE,
        )
        print(output)


if __name__ == '__main__':
    asyncio.run(main())
