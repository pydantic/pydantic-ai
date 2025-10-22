# Dataset Management

Create, save, load, and generate evaluation datasets.

## Creating Datasets

### From Code

Define datasets directly in Python:

```python
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected, IsInstance

dataset = Dataset[str, str, Any](
    name='my_eval_suite',
    cases=[
        Case(
            name='test_1',
            inputs='input 1',
            expected_output='output 1',
        ),
        Case(
            name='test_2',
            inputs='input 2',
            expected_output='output 2',
        ),
    ],
    evaluators=[
        IsInstance(type_name='str'),
        EqualsExpected(),
    ],
)
```

### Adding Cases Dynamically

```python
from typing import Any

from pydantic_evals import Dataset
from pydantic_evals.evaluators import IsInstance

dataset = Dataset[str, str, Any](cases=[], evaluators=[])

# Add cases one at a time
dataset.add_case(
    name='dynamic_case',
    inputs='test input',
    expected_output='test output',
)

# Add evaluators
dataset.add_evaluator(IsInstance(type_name='str'))
```

## Saving Datasets

!!! info "Detailed Serialization Guide"
    For complete details on serialization formats, JSON schema generation, and custom evaluators, see [Dataset Serialization](dataset-serialization.md).

### Save to YAML

```python
from typing import Any

from pydantic_evals import Case, Dataset

dataset = Dataset[str, str, Any](cases=[Case(name='test', inputs='example')])
dataset.to_file('my_dataset.yaml')

# Also saves schema file: my_dataset_schema.json
```

Output (`my_dataset.yaml`):

```yaml
# yaml-language-server: $schema=my_dataset_schema.json
name: my_eval_suite
cases:
- name: test_1
  inputs: input 1
  expected_output: output 1
  evaluators:
  - EqualsExpected
- name: test_2
  inputs: input 2
  expected_output: output 2
  evaluators:
  - EqualsExpected
evaluators:
- IsInstance: str
```

### Save to JSON

```python
from typing import Any

from pydantic_evals import Case, Dataset

dataset = Dataset[str, str, Any](cases=[Case(name='test', inputs='example')])
dataset.to_file('my_dataset.json')

# Also saves schema file: my_dataset_schema.json
```

### Custom Schema Path

```python
from pathlib import Path
from typing import Any

from pydantic_evals import Case, Dataset

dataset = Dataset[str, str, Any](cases=[Case(name='test', inputs='example')])

# Custom schema location
Path('data').mkdir(exist_ok=True)
Path('data/schemas').mkdir(parents=True, exist_ok=True)
dataset.to_file(
    'data/my_dataset.yaml',
    schema_path='schemas/my_schema.json',
)

# No schema file
dataset.to_file('my_dataset.yaml', schema_path=None)
```

## Loading Datasets

### From YAML/JSON

```python {test="skip"}
from typing import Any

from pydantic_evals import Dataset

# Infers format from extension
dataset = Dataset[str, str, Any].from_file('my_dataset.yaml')
dataset = Dataset[str, str, Any].from_file('my_dataset.json')

# Explicit format for non-standard extensions
dataset = Dataset[str, str, Any].from_file('data.txt', fmt='yaml')
```

### From String

```python
from typing import Any

from pydantic_evals import Dataset

yaml_content = """
cases:
- name: test
  inputs: hello
  expected_output: HELLO
evaluators:
- EqualsExpected
"""

dataset = Dataset[str, str, Any].from_text(yaml_content, fmt='yaml')
```

### From Dict

```python
from typing import Any

from pydantic_evals import Dataset

data = {
    'cases': [
        {
            'name': 'test',
            'inputs': 'hello',
            'expected_output': 'HELLO',
        },
    ],
    'evaluators': [{'EqualsExpected': {}}],
}

dataset = Dataset[str, str, Any].from_dict(data)
```

### With Custom Evaluators

When loading datasets that use custom evaluators, you must pass them to `from_file()`:

```python {test="skip"}
from dataclasses import dataclass
from typing import Any

from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class MyCustomEvaluator(Evaluator):
    threshold: float = 0.5

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


# Load with custom evaluator registry
dataset = Dataset[str, str, Any].from_file(
    'my_dataset.yaml',
    custom_evaluator_types=[MyCustomEvaluator],
)
```

For complete details on serialization with custom evaluators, see [Dataset Serialization](dataset-serialization.md).

## Generating Datasets

Pydantic Evals allows you to generate test datasets using LLMs with [`generate_dataset`][pydantic_evals.generation.generate_dataset].

Datasets can be generated in either JSON or YAML format, in both cases a JSON schema file is generated alongside the dataset and referenced in the dataset, so you should get type checking and auto-completion in your editor.

```python {title="generate_dataset_example.py"}
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from pydantic_evals import Dataset
from pydantic_evals.generation import generate_dataset


class QuestionInputs(BaseModel, use_attribute_docstrings=True):  # (1)!
    """Model for question inputs."""

    question: str
    """A question to answer"""
    context: str | None = None
    """Optional context for the question"""


class AnswerOutput(BaseModel, use_attribute_docstrings=True):  # (2)!
    """Model for expected answer outputs."""

    answer: str
    """The answer to the question"""
    confidence: float = Field(ge=0, le=1)
    """Confidence level (0-1)"""


class MetadataType(BaseModel, use_attribute_docstrings=True):  # (3)!
    """Metadata model for test cases."""

    difficulty: str
    """Difficulty level (easy, medium, hard)"""
    category: str
    """Question category"""


async def main():
    dataset = await generate_dataset(  # (4)!
        dataset_type=Dataset[QuestionInputs, AnswerOutput, MetadataType],
        n_examples=2,
        extra_instructions="""
        Generate question-answer pairs about world capitals and landmarks.
        Make sure to include both easy and challenging questions.
        """,
    )
    output_file = Path('questions_cases.yaml')
    dataset.to_file(output_file)  # (5)!
    print(output_file.read_text())
    """
    # yaml-language-server: $schema=questions_cases_schema.json
    name: null
    cases:
    - name: Easy Capital Question
      inputs:
        question: What is the capital of France?
        context: null
      metadata:
        difficulty: easy
        category: Geography
      expected_output:
        answer: Paris
        confidence: 0.95
      evaluators:
      - EqualsExpected
    - name: Challenging Landmark Question
      inputs:
        question: Which world-famous landmark is located on the banks of the Seine River?
        context: null
      metadata:
        difficulty: hard
        category: Landmarks
      expected_output:
        answer: Eiffel Tower
        confidence: 0.9
      evaluators:
      - EqualsExpected
    evaluators: []
    """
```

1. Define the schema for the inputs to the task.
2. Define the schema for the expected outputs of the task.
3. Define the schema for the metadata of the test cases.
4. Call [`generate_dataset`][pydantic_evals.generation.generate_dataset] to create a [`Dataset`][pydantic_evals.Dataset] with 2 cases confirming to the schema.
5. Save the dataset to a YAML file, this will also write `questions_cases_schema.json` with the schema JSON schema for `questions_cases.yaml` to make editing easier. The magic `yaml-language-server` comment is supported by at least vscode, jetbrains/pycharm (more details [here](https://github.com/redhat-developer/yaml-language-server#using-inlined-schema)).

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main(answer))` to run `main`)_

You can also write datasets as JSON files:

```python {title="generate_dataset_example_json.py" requires="generate_dataset_example.py"}
from pathlib import Path

from pydantic_evals import Dataset
from pydantic_evals.generation import generate_dataset

from generate_dataset_example import AnswerOutput, MetadataType, QuestionInputs


async def main():
    dataset = await generate_dataset(  # (1)!
        dataset_type=Dataset[QuestionInputs, AnswerOutput, MetadataType],
        n_examples=2,
        extra_instructions="""
        Generate question-answer pairs about world capitals and landmarks.
        Make sure to include both easy and challenging questions.
        """,
    )
    output_file = Path('questions_cases.json')
    dataset.to_file(output_file)  # (2)!
    print(output_file.read_text())
    """
    {
      "$schema": "questions_cases_schema.json",
      "name": null,
      "cases": [
        {
          "name": "Easy Capital Question",
          "inputs": {
            "question": "What is the capital of France?",
            "context": null
          },
          "metadata": {
            "difficulty": "easy",
            "category": "Geography"
          },
          "expected_output": {
            "answer": "Paris",
            "confidence": 0.95
          },
          "evaluators": [
            "EqualsExpected"
          ]
        },
        {
          "name": "Challenging Landmark Question",
          "inputs": {
            "question": "Which world-famous landmark is located on the banks of the Seine River?",
            "context": null
          },
          "metadata": {
            "difficulty": "hard",
            "category": "Landmarks"
          },
          "expected_output": {
            "answer": "Eiffel Tower",
            "confidence": 0.9
          },
          "evaluators": [
            "EqualsExpected"
          ]
        }
      ],
      "evaluators": []
    }
    """
```

1. Generate the [`Dataset`][pydantic_evals.Dataset] exactly as above.
2. Save the dataset to a JSON file, this will also write `questions_cases_schema.json` with th JSON schema for `questions_cases.json`. This time the `$schema` key is included in the JSON file to define the schema for IDEs to use while you edit the file, there's no formal spec for this, but it works in vscode and pycharm and is discussed at length in [json-schema-org/json-schema-spec#828](https://github.com/json-schema-org/json-schema-spec/issues/828).

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main(answer))` to run `main`)_

## Type-Safe Datasets

Use generic type parameters for type safety:

```python
from typing_extensions import TypedDict

from pydantic_evals import Case, Dataset


class MyInput(TypedDict):
    query: str
    max_results: int


class MyOutput(TypedDict):
    results: list[str]


class MyMetadata(TypedDict):
    category: str


# Type-safe dataset
dataset: Dataset[MyInput, MyOutput, MyMetadata] = Dataset(
    cases=[
        Case(
            name='test',
            inputs={'query': 'test', 'max_results': 10},
            expected_output={'results': ['a', 'b']},
            metadata={'category': 'search'},
        ),
    ],
)
```

## Schema Generation

Generate JSON Schema for IDE support:

```python
from typing import Any

from pydantic_evals import Case, Dataset

dataset = Dataset[str, str, Any](cases=[Case(name='test', inputs='example')])

# Save with schema
dataset.to_file('my_dataset.yaml')  # Creates my_dataset_schema.json

# Schema enables:
# - Autocomplete in VS Code/PyCharm
# - Validation while editing
# - Inline documentation
```

Manual schema generation:

```python
import json
from dataclasses import dataclass
from typing import Any

from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class MyCustomEvaluator(Evaluator):
    threshold: float = 0.5

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


schema = Dataset[str, str, Any].model_json_schema_with_evaluators(
    custom_evaluator_types=[MyCustomEvaluator],
)
print(json.dumps(schema, indent=2)[:66] + '...')
"""
{
  "$defs": {
    "Case": {
      "additionalProperties": false,
...
"""
```

## Best Practices

### 1. Use Clear Names

```python
from pydantic_evals import Case

# Good
Case(name='uppercase_basic_ascii', inputs='hello')
Case(name='uppercase_unicode_emoji', inputs='hello 😀')
Case(name='uppercase_empty_string', inputs='')

# Bad
Case(name='test1', inputs='hello')
Case(name='test2', inputs='world')
Case(name='test3', inputs='foo')
```

### 2. Organize by Difficulty

```python
from pydantic_evals import Case, Dataset

dataset = Dataset(
    cases=[
        Case(name='easy_1', inputs='test', metadata={'difficulty': 'easy'}),
        Case(name='easy_2', inputs='test2', metadata={'difficulty': 'easy'}),
        Case(name='medium_1', inputs='test3', metadata={'difficulty': 'medium'}),
        Case(name='hard_1', inputs='test4', metadata={'difficulty': 'hard'}),
    ],
)
```

### 3. Start Small, Grow Gradually

```python
from pydantic_evals import Case, Dataset

# Start with representative cases
dataset = Dataset(
    cases=[
        Case(name='happy_path', inputs='test'),
        Case(name='edge_case', inputs=''),
        Case(name='error_case', inputs='invalid'),
    ],
)

# Add more as you find issues
dataset.add_case(name='newly_discovered_edge_case', inputs='edge')
```

### 4. Use Metadata Extensively

```python
from pydantic_evals import Case

Case(
    name='complex_query',
    inputs='solve integral of x^2',
    metadata={
        'difficulty': 'hard',
        'category': 'math',
        'subcategory': 'calculus',
        'source': 'textbook_page_123',
        'added_date': '2024-03-20',
        'issue_ref': 'GH-456',
    },
)
```

### 5. Separate Datasets by Purpose

```python
from typing import Any

from pydantic_evals import Case, Dataset

# First create some test datasets
for name in ['smoke_tests', 'comprehensive_tests', 'regression_tests']:
    test_dataset = Dataset[str, Any, Any](cases=[Case(name='test', inputs='example')])
    test_dataset.to_file(f'{name}.yaml')

# Smoke tests (fast, critical paths)
smoke_tests = Dataset[str, Any, Any].from_file('smoke_tests.yaml')

# Comprehensive tests (slow, thorough)
comprehensive = Dataset[str, Any, Any].from_file('comprehensive_tests.yaml')

# Regression tests (specific bugs)
regression = Dataset[str, Any, Any].from_file('regression_tests.yaml')
```

## Next Steps

- **[Dataset Serialization](dataset-serialization.md)** - In-depth guide to saving and loading datasets
- **[Generating Datasets](#generating-datasets)** - Use LLMs to generate test cases
- **[Examples: Simple Validation](../examples/simple-validation.md)** - Practical examples
