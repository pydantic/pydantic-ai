"""WITHOUT easy_evals -- subjective quality via LLM-as-judge."""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from fake_models import judge_model, qa_agent

agent = qa_agent()
judge = judge_model()


async def task(question: str) -> str:
    return (await agent.run(question)).output


dataset = Dataset(
    name='haiku',
    cases=[
        Case(name='haiku', inputs='Write a haiku about the sea.',
             evaluators=(LLMJudge(rubric='The output is a haiku (three lines).', model=judge),)),
    ],
)

if __name__ == '__main__':
    dataset.evaluate_sync(task).print(include_input=True, include_output=True)
