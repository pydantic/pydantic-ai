"""WITHOUT easy_evals -- run each case 3x (flaky LLMs) and diff against a baseline."""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains

from fake_models import qa_agent

agent = qa_agent()


async def task(question: str) -> str:
    return (await agent.run(question)).output


dataset = Dataset(
    name='qa',
    cases=[Case(name='france', inputs='What is the capital of France?', evaluators=(Contains(value='Paris'),))],
)

if __name__ == '__main__':
    baseline = dataset.evaluate_sync(task, repeat=3)
    latest = dataset.evaluate_sync(task, repeat=3)
    latest.print(baseline=baseline, include_input=True)
