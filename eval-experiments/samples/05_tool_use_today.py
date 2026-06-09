"""WITHOUT easy_evals -- assert the agent actually CALLED a tool.

You must: configure OpenTelemetry, instrument the agent, and know the exact span
name ('running tool') and attribute ('gen_ai.tool.name') to query.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from pydantic_ai import Agent
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import HasMatchingSpan

from fake_models import tool_agent

trace.set_tracer_provider(TracerProvider())  # without this, no spans are recorded
Agent.instrument_all()                        # without this, the agent emits nothing

agent = tool_agent()


async def task(question: str) -> str:
    return (await agent.run(question)).output


dataset = Dataset(
    name='weather',
    cases=[
        Case(
            name='weather',
            inputs='What is the weather in Paris?',
            evaluators=(
                HasMatchingSpan(
                    query={'name_equals': 'running tool', 'has_attributes': {'gen_ai.tool.name': 'get_weather'}},
                    evaluation_name='calls_get_weather',
                ),
            ),
        ),
    ],
)

if __name__ == '__main__':
    dataset.evaluate_sync(task).print(include_input=True, include_output=True)
