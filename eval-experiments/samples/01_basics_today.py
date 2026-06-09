"""WITHOUT easy_evals -- basic Q&A correctness with today's pydantic_evals API.

Note the ceremony: wrap the agent in a task, build Case + Dataset, and pick an
evaluator (Contains, because the agent answers in full sentences -- EqualsExpected
would fail).
"""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, IsInstance

from fake_models import qa_agent

agent = qa_agent()


async def task(question: str) -> str:
    result = await agent.run(question)
    return result.output


dataset = Dataset(
    name='qa',
    cases=[
        Case(name='france', inputs='What is the capital of France?', expected_output='Paris',
             evaluators=(Contains(value='Paris'),)),
        Case(name='japan', inputs='What is the capital of Japan?', expected_output='Tokyo',
             evaluators=(Contains(value='Tokyo'),)),
    ],
    evaluators=[IsInstance(type_name='str')],
)

if __name__ == '__main__':
    dataset.evaluate_sync(task).print(include_input=True, include_output=True)
