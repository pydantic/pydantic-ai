"""WITHOUT easy_evals -- several checks on one case.

pydantic_evals has no built-in "does not contain" or "max words", so you hand-write
custom Evaluator classes for them.
"""

from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, Evaluator, EvaluatorContext

from fake_models import qa_agent


@dataclass
class NotContains(Evaluator[object, object, object]):
    """Pass if `value` is absent from the output."""

    value: str

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        return self.value.lower() not in str(ctx.output).lower()


@dataclass
class MaxWords(Evaluator[object, object, object]):
    """Pass if the output has at most `limit` words."""

    limit: int

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        return len(str(ctx.output).split()) <= self.limit


agent = qa_agent()


async def task(question: str) -> str:
    return (await agent.run(question)).output


dataset = Dataset(
    name='qa',
    cases=[
        Case(name='japan', inputs='What is the capital of Japan?',
             evaluators=(Contains(value='Tokyo'), NotContains(value='Paris'), MaxWords(limit=20))),
    ],
)

if __name__ == '__main__':
    dataset.evaluate_sync(task).print(include_input=True, include_output=True)
