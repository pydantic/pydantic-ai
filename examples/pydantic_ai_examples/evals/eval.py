from pydantic_evals.dataset import EvaluatorContext
from pydantic_evals.evaluators import Evaluator
from pydantic_evals.evaluators.common import IsInstance, LLMJudge
from pydantic_evals.evaluators.llm_as_a_judge import GradingOutput, judge_input_output

from .agent import infer_time_range
from .models import TimeRangeDataset, TimeRangeInputs, TimeRangeResponse


async def judge_time_range_case(
    inputs: TimeRangeInputs, output: TimeRangeResponse
) -> GradingOutput:
    """Judge the output of a time range inference agent based on a rubric."""
    rubric = (
        'The output should be a reasonable time range to select for the given inputs, or an error '
        'message if no good time range could be selected. Pick a score between 0 and 1 to represent how confident '
        'you are that the selected time range was what the user intended, or that an error message was '
        'an appropriate response.'
    )
    return await judge_input_output(inputs, output, rubric)


async def main():
    """TODO: Task: Convert this pydantic_evals.demo package into docs."""
    from pathlib import Path

    import logfire

    logfire.configure(
        send_to_logfire='if-token-present', console=logfire.ConsoleOptions(verbose=True)
    )
    dataset = TimeRangeDataset.from_file(
        Path(__file__).parent / 'test_cases.yaml',
        custom_evaluator_types=(IsInstance, LLMJudge),
    )

    class MyEvaluator(Evaluator[TimeRangeInputs, TimeRangeResponse]):
        async def evaluate(
            self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeResponse]
        ):
            result = await judge_time_range_case(inputs=ctx.inputs, output=ctx.output)
            return {
                'is_reasonable': 'yes' if result.pass_ else 'no',
                'accuracy': result.score,
            }

    dataset.add_evaluator(MyEvaluator())

    report = await dataset.evaluate(infer_time_range, max_concurrency=10)
    report.print(include_input=True, include_output=True)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
