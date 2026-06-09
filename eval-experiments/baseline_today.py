"""BASELINE: evaluating an agent with pydantic-evals *as it exists today*.

Goal: a beginner wants to check that their Q&A agent gives the right answers,
and that a 'write a haiku' request actually produces a haiku.

Run: uv run python eval-experiments/baseline_today.py
"""

from __future__ import annotations

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, IsInstance, LLMJudge

from fake_models import judge_model, qa_agent

agent = qa_agent()


# 1. You have to wrap the agent in a task function with the right signature.
async def task(question: str) -> str:
    result = await agent.run(question)
    return result.output


# 2. You have to learn Case + Dataset + which evaluator to pick.
#    Note: EqualsExpected() would FAIL here because the agent answers in full
#    sentences ("The capital of France is Paris."), so a beginner has to know
#    to reach for Contains instead -- a classic first-time gotcha.
dataset = Dataset(
    name='qa',
    cases=[
        Case(
            name='france',
            inputs='What is the capital of France?',
            expected_output='Paris',
            evaluators=(Contains(value='Paris'),),
        ),
        Case(
            name='japan',
            inputs='What is the capital of Japan?',
            expected_output='Tokyo',
            evaluators=(Contains(value='Tokyo'),),
        ),
        Case(
            name='haiku',
            inputs='Write a haiku about the sea.',
            evaluators=(
                LLMJudge(
                    rubric='The output is a haiku (three lines) about the sea',
                    model=judge_model(),
                ),
            ),
        ),
    ],
    evaluators=[IsInstance(type_name='str')],
)


if __name__ == '__main__':
    report = dataset.evaluate_sync(task)
    report.print(include_input=True, include_output=True, include_durations=False)
