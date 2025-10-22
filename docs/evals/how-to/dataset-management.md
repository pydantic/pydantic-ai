# Dataset Management

Create, save, load, and generate evaluation datasets.

## Creating Datasets

### From Code

Define datasets directly in Python:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected, IsInstance

dataset = Dataset(
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
from pydantic_evals import Dataset
from pydantic_evals.evaluators import IsInstance

dataset = Dataset(cases=[], evaluators=[])

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

### Save to YAML

```python
import warnings

from pydantic_evals import Case, Dataset

dataset = Dataset(cases=[Case(name='test', inputs='example')])
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
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
import warnings

from pydantic_evals import Case, Dataset

dataset = Dataset(cases=[Case(name='test', inputs='example')])
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
    dataset.to_file('my_dataset.json')

# Also saves schema file: my_dataset_schema.json
```

### Custom Schema Path

```python
import warnings
from pathlib import Path

from pydantic_evals import Case, Dataset

dataset = Dataset(cases=[Case(name='test', inputs='example')])

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
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

```python
import warnings

from pydantic_evals import Case, Dataset

# First create a file to load
test_dataset = Dataset(cases=[Case(name='test', inputs='example')])
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
    test_dataset.to_file('my_dataset.yaml')
    test_dataset.to_file('my_dataset.json')

# Infers format from extension
dataset = Dataset.from_file('my_dataset.yaml')
dataset = Dataset.from_file('my_dataset.json')

# Explicit format
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
    test_dataset.to_file('data.txt', fmt='yaml')
dataset = Dataset.from_file('data.txt', fmt='yaml')
```

### From String

```python
from pydantic_evals import Dataset

yaml_content = """
cases:
- name: test
  inputs: hello
  expected_output: HELLO
evaluators:
- EqualsExpected
"""

dataset = Dataset.from_text(yaml_content, fmt='yaml')
```

### From Dict

```python
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

dataset = Dataset.from_dict(data)
```

### With Custom Evaluators

```python
import warnings
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class MyCustomEvaluator(Evaluator):
    threshold: float = 0.5

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


# First create a file to load
test_dataset = Dataset(cases=[Case(name='test', inputs='example')])
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
    test_dataset.to_file('my_dataset.yaml')

# Load with custom evaluator registry
dataset = Dataset.from_file(
    'my_dataset.yaml',
    custom_evaluator_types=[MyCustomEvaluator],
)
```

## Generating Datasets

Use an LLM to generate test cases:

```python {test="skip"}
from pydantic import BaseModel

from pydantic_evals import Dataset
from pydantic_evals.generation import generate_dataset


class QuestionInput(BaseModel):
    question: str
    context: str | None = None


class AnswerOutput(BaseModel):
    answer: str
    confidence: float


class TestMetadata(BaseModel):
    difficulty: str
    category: str


async def main():
    # Generate dataset
    dataset = await generate_dataset(
        dataset_type=Dataset[QuestionInput, AnswerOutput, TestMetadata],
        n_examples=10,
        extra_instructions='''
            Generate questions about world capitals.
            Include both easy and hard questions.
            Provide context where helpful.
        ''',
        model='openai:gpt-4o',  # Optional, defaults to gpt-4o
    )

    # Save generated dataset
    dataset.to_file('generated_cases.yaml')
```

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
import warnings

from pydantic_evals import Case, Dataset

dataset = Dataset(cases=[Case(name='test', inputs='example')])

# Save with schema
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
    dataset.to_file('my_dataset.yaml')  # Creates my_dataset_schema.json

# Schema enables:
# - Autocomplete in VS Code/PyCharm
# - Validation while editing
# - Inline documentation
```

Manual schema generation:

```python
import json
import warnings
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class MyCustomEvaluator(Evaluator):
    threshold: float = 0.5

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


dataset = Dataset(cases=[Case(name='test', inputs='example')])

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
    schema = dataset.model_json_schema_with_evaluators(
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

## Versioning Datasets

### Git Integration

```python
import warnings
from pathlib import Path

from pydantic_evals import Case, Dataset

version = 'v1'
dataset = Dataset(cases=[Case(name='test', inputs='example')])

# Save dataset with descriptive name
Path(f'eval_datasets/{version}').mkdir(parents=True, exist_ok=True)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
    dataset.to_file(f'eval_datasets/{version}/test_suite.yaml')

# Commit to version control
# git add eval_datasets/v1/test_suite.yaml
# git commit -m "Add evaluation dataset v1"
```

### Track Dataset Changes

```python
import warnings
from datetime import datetime

from pydantic_evals import Case, Dataset

# Include version info in name
dataset = Dataset(
    name=f'my_eval_suite_v1_{datetime.now().strftime("%Y%m%d")}',
    cases=[Case(name='test', inputs='example')],
)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
    dataset.to_file('eval_suite_v1_20250320.yaml')
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
import warnings

from pydantic_evals import Case, Dataset

# First create some test datasets
for name in ['smoke_tests', 'comprehensive_tests', 'regression_tests']:
    test_dataset = Dataset(cases=[Case(name='test', inputs='example')])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
        test_dataset.to_file(f'{name}.yaml')

# Smoke tests (fast, critical paths)
smoke_tests = Dataset.from_file('smoke_tests.yaml')

# Comprehensive tests (slow, thorough)
comprehensive = Dataset.from_file('comprehensive_tests.yaml')

# Regression tests (specific bugs)
regression = Dataset.from_file('regression_tests.yaml')
```

## Working with Large Datasets

### Lazy Loading

```python
import warnings

from pydantic_evals import Case, Dataset

# First create a test dataset
test_dataset = Dataset(
    cases=[
        Case(name='test1', inputs='example1', metadata={'category': 'critical'}),
        Case(name='test2', inputs='example2', metadata={'category': 'normal'}),
    ],
)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
    test_dataset.to_file('huge_dataset.yaml')

# Don't load all at once
dataset = Dataset.from_file('huge_dataset.yaml')


def task(inputs: str) -> str:
    return f'result for {inputs}'


# Filter cases
subset = Dataset(
    cases=[c for c in dataset.cases if c.metadata and c.metadata.get('category') == 'critical'],
    evaluators=dataset.evaluators,
)

report = subset.evaluate_sync(task)
```

### Parallel Processing

```python
import warnings

from pydantic_evals import Case, Dataset

# First create a test dataset
test_dataset = Dataset(cases=[Case(name='test', inputs='example')])
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
    test_dataset.to_file('large_dataset.yaml')


def task(inputs: str) -> str:
    return f'result for {inputs}'


# Process in batches
dataset = Dataset.from_file('large_dataset.yaml')

# Control concurrency
report = dataset.evaluate_sync(
    task,
    max_concurrency=10,  # Limit concurrent execution
)
```

### Sampling

```python
import random
import warnings

from pydantic_evals import Case, Dataset

# First create a test dataset
test_dataset = Dataset(cases=[Case(name=f'test{i}', inputs=f'example{i}') for i in range(10)])
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='Could not determine the generic parameters')
    test_dataset.to_file('large_dataset.yaml')


def task(inputs: str) -> str:
    return f'result for {inputs}'


# Random sample
dataset = Dataset.from_file('large_dataset.yaml')
sample = Dataset(
    cases=random.sample(dataset.cases, k=min(100, len(dataset.cases))),
    evaluators=dataset.evaluators,
)

report = sample.evaluate_sync(task)
```

## Next Steps

- **[Generating Datasets](../../evals.md#generating-test-datasets)** - Use LLMs to generate cases
- **[Examples: Simple Validation](../examples/simple-validation.md)** - Proof of concept
