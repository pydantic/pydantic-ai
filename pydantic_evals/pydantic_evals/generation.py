"""Utilities for generating example datasets for pydantic_evals.

This module provides functions for generating sample datasets for testing and examples,
using LLMs to create realistic test data with proper structure.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from typing_extensions import TypeVar

from pydantic_ai import Agent, models
from pydantic_ai._utils import strip_markdown_fences
from pydantic_evals import Dataset
from pydantic_evals.evaluators.evaluator import Evaluator

__all__ = ('generate_dataset',)

InputsT = TypeVar('InputsT', default=Any)
"""Generic type for the inputs to the task being evaluated."""

OutputT = TypeVar('OutputT', default=Any)
"""Generic type for the expected output of the task being evaluated."""

MetadataT = TypeVar('MetadataT', default=Any)
"""Generic type for the metadata associated with the task being evaluated."""


async def generate_dataset(
    *,
    dataset_type: type[Dataset[InputsT, OutputT, MetadataT]],
    path: Path | str | None = None,
    custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    model: models.Model | models.KnownModelName = 'openai:gpt-4o',
    n_examples: int = 3,
    extra_instructions: str | None = None,
) -> Dataset[InputsT, OutputT, MetadataT]:
    """Use an LLM to generate a dataset of test cases, each consisting of input, expected output, and metadata.

    This function creates a properly structured dataset with the specified input, output, and metadata types.
    It uses an LLM to attempt to generate realistic test cases that conform to the types' schemas.

    Args:
        path: Optional path to save the generated dataset. If provided, the dataset will be saved to this location.
        dataset_type: The type of dataset to generate, with the desired input, output, and metadata types.
        custom_evaluator_types: Optional sequence of custom evaluator classes to include in the schema.
        model: The Pydantic AI model to use for generation. Defaults to 'gpt-4o'.
        n_examples: Number of examples to generate. Defaults to 3.
        extra_instructions: Optional additional instructions to provide to the LLM.

    Returns:
        A properly structured Dataset object with generated test cases.

    Raises:
        ValidationError: If the LLM's response cannot be parsed as a valid dataset.
    """
    output_schema = dataset_type.model_json_schema_with_evaluators(custom_evaluator_types)

    # TODO: Use `output_type=StructuredDict(output_schema)` (and `from_dict` below) once https://github.com/pydantic/pydantic/issues/12145
    # is fixed and `StructuredDict` no longer needs to use `InlineDefsJsonSchemaTransformer`.
    agent = Agent(
        model,
        system_prompt=(
            f'Generate an object that is in compliance with this JSON schema:\n{output_schema}\n\n'
            f'Include {n_examples} example cases.'
            ' You must not include any characters in your response before the opening { of the JSON object, or after the closing }.'
        ),
        output_type=str,
        retries=1,
    )

    result = await agent.run(extra_instructions or 'Please generate the object.')
    output = strip_markdown_fences(result.output)
    try:
        result = dataset_type.from_text(output, fmt='json', custom_evaluator_types=custom_evaluator_types)
    except ValidationError as e:  # pragma: no cover
        print(f'Raw response from model:\n{result.output}')
        raise e
    if path is not None:
        result.to_file(path, custom_evaluator_types=custom_evaluator_types)  # pragma: no cover
    return result
