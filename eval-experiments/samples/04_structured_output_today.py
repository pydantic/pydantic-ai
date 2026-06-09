"""WITHOUT easy_evals -- check fields of a structured (Pydantic) output.

No built-in partial-field matcher, so you write one.
"""

from dataclasses import dataclass, field

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from fake_models import person_agent


@dataclass
class HasFields(Evaluator[object, object, object]):
    """Pass if the output has the given attribute values."""

    fields: dict[str, object] = field(default_factory=dict)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        return all(getattr(ctx.output, key, None) == value for key, value in self.fields.items())


agent = person_agent()


async def task(question: str) -> object:
    return (await agent.run(question)).output


dataset = Dataset(
    name='people',
    cases=[
        Case(name='ada', inputs='Make a person named Ada aged 36',
             evaluators=(HasFields(fields={'name': 'Ada', 'age': 36}),)),
    ],
)

if __name__ == '__main__':
    dataset.evaluate_sync(task).print(include_input=True)
