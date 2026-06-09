"""WITHOUT easy_evals -- push/pull a Logfire-hosted dataset, grade with a managed variable.

Needs logfire>=4.35 and a LOGFIRE_TOKEN to run live.
"""

import logfire
from logfire.experimental.api_client import LogfireAPIClient

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from fake_models import qa_agent

agent = qa_agent()
rubric = logfire.var('prompt__haiku_rubric', default='is a haiku').get(label='production').value

dataset = Dataset(
    name='qa-regression',
    cases=[Case(name='haiku', inputs='Write a haiku about the sea.',
                evaluators=(LLMJudge(rubric=f'The output {rubric}.'),))],
)


async def task(question: str) -> str:
    return (await agent.run(question)).output


if __name__ == '__main__':
    with LogfireAPIClient() as client:
        client.push_dataset(dataset, name='qa-regression')
        fetched = client.get_dataset('qa-regression', input_type=str, output_type=str, metadata_type=type(None))
    assert isinstance(fetched, Dataset)
    fetched.evaluate_sync(task).print()
