from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from typing_extensions import TypeVar

from pydantic_ai import Agent, models
from pydantic_evals import Dataset
from pydantic_evals.evaluators.common import IsInstance, LlmJudge
from pydantic_evals.evaluators.evaluator import Evaluator

InputsT = TypeVar('InputsT', default=Any)
OutputT = TypeVar('OutputT', default=Any)
MetadataT = TypeVar('MetadataT', default=Any)


async def generate_dataset(
    *,
    path: Path | str | None = None,
    inputs_type: type[InputsT],
    output_type: type[OutputT],
    metadata_type: type[MetadataT],
    custom_evaluators: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    model: models.Model | models.KnownModelName = 'gpt-4o',
    n_examples: int = 3,
    extra_instructions: str | None = None,
) -> Dataset[InputsT, OutputT, MetadataT]:
    """Use an LLM to generate a dataset of test cases, each consisting of input, expected output, and metadata."""
    dataset_type = Dataset[inputs_type, output_type, metadata_type]
    result_schema = dataset_type.model_json_schema_with_evaluators(custom_evaluators)
    # TODO: Make it so pydantic_ai can be given just a JSON schema for the result, so we can simplify this system prompt
    agent = Agent(
        model,
        system_prompt=(
            f'Generate an object that is in compliance with this JSON schema:\n{result_schema}\n\n'
            f'Include {n_examples} example cases.'
            ' You must not include any characters in your response before the opening { of the JSON object, or after the closing }.'
        ),
        result_type=str,
        retries=1,
    )

    raw_response = (await agent.run(extra_instructions or 'Please generate the object.')).data
    try:
        result = dataset_type.from_text(raw_response, fmt='json', custom_evaluators=custom_evaluators)
    except ValidationError as e:
        print(f'Raw response from model:\n{raw_response}')
        raise e
    if path is not None:
        result.to_file(path, custom_evaluators=custom_evaluators)
    return result


if __name__ == '__main__':

    async def main():
        """Usage example."""
        from pydantic_evals.demo.time_range.models import TimeRangeInputs, TimeRangeResponse

        custom_evaluators = (LlmJudge, IsInstance)
        await generate_dataset(
            path='test_cases.yaml',
            inputs_type=TimeRangeInputs,
            output_type=TimeRangeResponse,  # type: ignore  # need to support TypeForm
            metadata_type=dict[str, Any],
            custom_evaluators=custom_evaluators,
            extra_instructions=(
                'Include some case-specific evaluators, including some llm_judge calls that might not be'
                ' expected to pass for a naive implementation (note llm_judge can see both the case inputs and outputs).'
                ' Do not use the `model` or `include_input` kwargs to llm_judge.'
            ),
        )

    import asyncio

    asyncio.run(main())
