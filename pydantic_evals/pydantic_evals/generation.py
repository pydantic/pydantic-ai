"""Utilities for generating example datasets for pydantic_evals.

This module provides functions for generating sample datasets for testing and examples,
using LLMs to create realistic test data with proper structure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError
from typing_extensions import TypeVar

from pydantic_ai import Agent, models
from pydantic_ai._utils import strip_markdown_fences
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.evaluator import Evaluator

__all__ = ('generate_dataset', 'generate_dataset_for_agent')

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


async def generate_dataset_for_agent(
    agent: Agent[Any, OutputT],
    *,
    inputs_type: type[InputsT] = str,
    metadata_type: type[MetadataT] | None = None,
    path: Path | str | None = None,
    model: models.Model | models.KnownModelName = 'openai:gpt-4o',
    n_examples: int = 3,
    extra_instructions: str | None = None,
) -> Dataset[InputsT, OutputT, MetadataT]:
    """Generate evaluation cases by running inputs through a target agent.

    Generates diverse inputs and metadata using an LLM, then runs them through the agent
    to produce realistic expected outputs for evaluation.

    Args:
        agent: Pydantic AI agent to extract outputs from.
        inputs_type: Type of inputs the agent expects. Defaults to str.
        metadata_type: Type for metadata. Defaults to None (uses NoneType).
        path: Optional path to save the generated dataset.
        model: Pydantic AI model to use for generation. Defaults to 'gpt-4o'.
        n_examples: Number of examples to generate. Defaults to 3.
        extra_instructions: Optional additional instructions for the LLM.

    Returns:
        A properly structured Dataset object with generated test cases.

    Raises:
        ValidationError: If the LLM's response cannot be parsed.
    """
    # Get output schema with proper type handling
    # Check if it's a Pydantic model class (not an instance) before calling model_json_schema
    output_schema: str
    if isinstance(agent.output_type, type) and issubclass(agent.output_type, BaseModel):
        output_schema = str(agent.output_type.model_json_schema())
    else:
        # For other types (str, custom output specs, etc.), just use string representation
        output_schema = str(agent.output_type)  # type: ignore[arg-type]

    # Get inputs schema with proper type handling
    inputs_schema: str
    if isinstance(inputs_type, type) and issubclass(inputs_type, BaseModel):
        inputs_schema = str(inputs_type.model_json_schema())
    else:
        inputs_schema = str(inputs_type)

    generation_prompt = (
        f'Generate {n_examples} test case inputs for an agent.\n\n'
        f'The agent accepts inputs of type: {inputs_schema}\n'
        f'The agent produces outputs of type: {output_schema}\n\n'
        f'Return a JSON array of objects with "name" (optional string), "inputs" (matching the input type), '
        f'and "metadata" (optional, any additional context).\n'
        f'You must not include any characters in your response before the opening [ of the JSON array, or after the closing ].'
        + (f'\n\n{extra_instructions}' if extra_instructions else '')
    )

    gen_agent = Agent(
        model,
        system_prompt=generation_prompt,
        output_type=str,
        retries=1,
    )

    result = await gen_agent.run('Please generate the test case inputs and metadata.')
    output = strip_markdown_fences(result.output).strip()

    try:
        if not output:
            raise ValueError('Empty output after stripping markdown fences')

        # Additional cleanup in case strip_markdown_fences didn't catch everything
        # Remove markdown code blocks with optional language identifier
        output = re.sub(r'^```(?:json)?\s*\n?', '', output)
        output = re.sub(r'\n?```\s*$', '', output)
        output = output.strip()

        inputs_metadata: list[dict[str, Any]] = json.loads(output)
        cases: list[Case[InputsT, OutputT, MetadataT]] = []

        for i, item in enumerate(inputs_metadata):
            agent_result = await agent.run(item['inputs'])
            cases.append(
                Case(
                    name=item.get('name', f'case-{i}'),
                    inputs=cast(InputsT, item['inputs']),
                    expected_output=agent_result.output,
                    metadata=cast(MetadataT, item.get('metadata')),
                )
            )

        result_dataset: Dataset[InputsT, OutputT, MetadataT] = Dataset(cases=cases, evaluators=[])

    except (json.JSONDecodeError, KeyError, ValidationError) as e:  # pragma: no cover
        print(f'Raw response from model:\n{result.output}\n')
        print(f'After stripping markdown fences:\n{output}')
        raise e

    if path is not None:
        result_dataset.to_file(path)

    return result_dataset
