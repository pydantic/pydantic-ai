"""Regressions are typed and run in CI, offline.

Same types as the agent, same harness as CI: dataset -> evaluators -> report.
"""
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, EqualsExpected


async def model(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('shout', {'text': 'hello'})])
    return ModelResponse(parts=[TextPart('HELLO WORLD')])


agent = Agent(FunctionModel(model))


@agent.tool
def shout(ctx: RunContext[None], text: str) -> str:
    return text.upper()


dataset = Dataset(
    name='shout',
    cases=[Case(name='hello', inputs='hello', expected_output='HELLO WORLD')],
    evaluators=[EqualsExpected(), Contains(value='HELLO', case_sensitive=True)],
)


def run_case(text: str) -> str:
    return str(agent.run_sync(text).output)


def main() -> None:
    report = dataset.evaluate_sync(run_case)
    averages = report.averages()
    print(f'assertions passed: {averages.assertions * 100:.0f}%')
    report.print()
    assert averages.assertions == 1.0


if __name__ == '__main__':
    main()
