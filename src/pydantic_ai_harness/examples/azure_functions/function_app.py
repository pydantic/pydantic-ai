"""Preview: native Functions orchestration and a timer using the SDK client.

The timer starts a run when the local host starts, without HTTP endpoints or model
credentials. Replace TestModel and choose a production trigger before deploying.
"""

import logging
from collections.abc import Generator

import azure.durable_functions as df
import azure.functions as func
from durabletask import task

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable
from pydantic_ai_harness.azure_durable_task.functions import register_activities

app = df.DFApp()
agent = Agent(
    TestModel(custom_output_text='Hosted durably'),
    name='support',
    capabilities=[AzureDurableTaskDurability()],
)
durability = AzureDurableTaskDurability.from_agent(agent)
assert durability is not None
register_activities(app, durability)


def support(context: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, str]:
    """Run the agent inside a user-written native orchestration."""
    result = yield from run_durable(context, lambda: agent.run(prompt))
    return result.output


app.orchestration_trigger(context_name='context')(support)


@app.timer_trigger(schedule='0 0 0 * * *', arg_name='timer', run_on_startup=True, use_monitor=False)
@app.durable_client_input(client_name='client')
async def start(timer: func.TimerRequest, client: df.DurableFunctionsClient) -> None:
    """Use the native client in application code, not a harness client wrapper."""
    instance_id = await client.schedule_new_orchestration('support', input='Hello')
    state = await client.wait_for_orchestration_completion(instance_id, timeout=60)
    if state is None:
        raise TimeoutError(f'Orchestration {instance_id!r} did not complete.')
    state.raise_if_failed()
    logging.info('Agent output: %s', state.serialized_output)
