import asyncio
from pathlib import Path
from types import NoneType

from pydantic_ai_examples.evals.models import TimeRangeInputs, TimeRangeResponse
from pydantic_evals import Dataset
from pydantic_evals.generation import generate_dataset


async def main():
    dataset = await generate_dataset(
        dataset_type=Dataset[TimeRangeInputs, TimeRangeResponse, NoneType],
        model='openai:o1',  # Use a smarter model since this is a more complex task that is only run once
        n_examples=10,
        extra_instructions="""
        Generate a dataset of test cases for the time range inference agent.

        Include a variety of inputs that might be given to the agent, including some where the only
        reasonable response is a `TimeRangeBuilderError`, and some where a `TimeRangeBuilderSuccess` is
        expected. Make use of the `IsInstance` evaluator to ensure that the inputs and outputs are of the appropriate
        type.

        When appropriate, use the `LLMJudge` evaluator to provide a more precise description of the time range the
        agent should have inferred. In particular, it's good if the example user inputs are somewhat ambiguous, to
        reflect realistic (difficult-to-handle) user questions, but the LLMJudge evaluator can help ensure that the
        agent's output is still judged based on precisely what the desired behavior is even for somewhat ambiguous
        user questions. You do not need to include LLMJudge evaluations for all cases (in particular, for cases where
        the expected output is unambiguous from the user's question), but you should include at least one or two
        examples that do benefit from an LLMJudge evaluation (and include it).

        To be clear, the LLMJudge rubrics should be concise and reflect only information that is NOT ALREADY PRESENT
        in the user prompt for the example.

        Leave the model and include_input arguments to LLMJudge as their default values (null).

        Also add a dataset-wide LLMJudge evaluator to ensure that the 'explanation' or 'error_message' fields are
        appropriate to be displayed to the user (e.g., written in second person, etc.).
        """,
    )

    dataset.to_file(
        Path(__file__).parent / 'datasets' / 'time_range_v1.yaml',
        fmt='yaml',
    )


if __name__ == '__main__':
    asyncio.run(main())
