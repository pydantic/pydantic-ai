# Dataset Serialization

Learn how to save and load datasets in different formats, with support for custom evaluators and IDE integration.

## Overview

Pydantic Evals supports serializing datasets to files in two formats:

- **YAML** (`.yaml`, `.yml`) - Human-readable, great for version control
- **JSON** (`.json`) - Structured, machine-readable

Both formats support:
- Automatic JSON schema generation for IDE autocomplete and validation
- Custom evaluator serialization/deserialization
- Type-safe loading with generic parameters

## YAML Format

YAML is the recommended format for most use cases due to its readability and compact syntax.

### Basic Example

```python
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected, IsInstance

# Create a dataset with typed parameters
dataset = Dataset[str, str, Any](
    name='my_tests',
    cases=[
        Case(
            name='test_1',
            inputs='hello',
            expected_output='HELLO',
        ),
    ],
    evaluators=[
        IsInstance(type_name='str'),
        EqualsExpected(),
    ],
)

# Save to YAML
dataset.to_file('my_tests.yaml')
```

This creates two files:

1. **`my_tests.yaml`** - The dataset
2. **`my_tests_schema.json`** - JSON schema for IDE support

### YAML Output

```yaml
# yaml-language-server: $schema=my_tests_schema.json
name: my_tests
cases:
- name: test_1
  inputs: hello
  expected_output: HELLO
evaluators:
- IsInstance: str
- EqualsExpected
```

### JSON Schema for IDEs

The first line references the schema file:

```yaml
# yaml-language-server: $schema=my_tests_schema.json
```

This enables:
- ✅ **Autocomplete** in VS Code, PyCharm, and other editors
- ✅ **Inline validation** while editing
- ✅ **Documentation tooltips** for fields
- ✅ **Error highlighting** for invalid data

!!! note "Editor Support"
    The `yaml-language-server` comment is supported by:

    - VS Code (with YAML extension)
    - JetBrains IDEs (PyCharm, IntelliJ, etc.)
    - Most editors with YAML language server support

    See the [YAML Language Server docs](https://github.com/redhat-developer/yaml-language-server#using-inlined-schema) for more details.

### Loading from YAML

```python
from pathlib import Path
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected, IsInstance

# First create and save the dataset
Path('my_tests.yaml').parent.mkdir(exist_ok=True)
dataset = Dataset[str, str, Any](
    name='my_tests',
    cases=[Case(name='test_1', inputs='hello', expected_output='HELLO')],
    evaluators=[IsInstance(type_name='str'), EqualsExpected()],
)
dataset.to_file('my_tests.yaml')

# Load the dataset with type parameters
dataset = Dataset[str, str, Any].from_file('my_tests.yaml')


def my_task(text: str) -> str:
    return text.upper()


# Run evaluation
report = dataset.evaluate_sync(my_task)
```

## JSON Format

JSON format is useful for programmatic generation or when strict structure is required.

### Basic Example

```python
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

dataset = Dataset[str, str, Any](
    name='my_tests',
    cases=[
        Case(name='test_1', inputs='hello', expected_output='HELLO'),
    ],
    evaluators=[EqualsExpected()],
)

# Save to JSON
dataset.to_file('my_tests.json')
```

### JSON Output

```json
{
  "$schema": "my_tests_schema.json",
  "name": "my_tests",
  "cases": [
    {
      "name": "test_1",
      "inputs": "hello",
      "expected_output": "HELLO"
    }
  ],
  "evaluators": [
    "EqualsExpected"
  ]
}
```

The `$schema` key at the top enables IDE support similar to YAML.

### Loading from JSON

```python
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

# First create and save the dataset
dataset = Dataset[str, str, Any](
    name='my_tests',
    cases=[Case(name='test_1', inputs='hello', expected_output='HELLO')],
    evaluators=[EqualsExpected()],
)
dataset.to_file('my_tests.json')

# Load from JSON
dataset = Dataset[str, str, Any].from_file('my_tests.json')
```

## Schema Generation

### Automatic Schema Creation

By default, `to_file()` creates a JSON schema file alongside your dataset:

```python
from typing import Any

from pydantic_evals import Case, Dataset

dataset = Dataset[str, str, Any](cases=[Case(inputs='test')])

# Creates both my_tests.yaml AND my_tests_schema.json
dataset.to_file('my_tests.yaml')
```

### Custom Schema Location

```python
from pathlib import Path
from typing import Any

from pydantic_evals import Case, Dataset

dataset = Dataset[str, str, Any](cases=[Case(inputs='test')])

# Create directories
Path('data').mkdir(exist_ok=True)

# Custom schema filename (relative to dataset file location)
dataset.to_file(
    'data/my_tests.yaml',
    schema_path='my_schema.json',
)

# No schema file
dataset.to_file('my_tests.yaml', schema_path=None)
```

### Schema Path Templates

Use `{stem}` to reference the dataset filename:

```python
from typing import Any

from pydantic_evals import Case, Dataset

dataset = Dataset[str, str, Any](cases=[Case(inputs='test')])

# Creates: my_tests.yaml and my_tests.schema.json
dataset.to_file(
    'my_tests.yaml',
    schema_path='{stem}.schema.json',
)
```

### Manual Schema Generation

Generate a schema without saving the dataset:

```python
import json
from typing import Any

from pydantic_evals import Dataset

# Get schema as dictionary for a specific dataset type
schema = Dataset[str, str, Any].model_json_schema_with_evaluators()

# Save manually
with open('custom_schema.json', 'w') as f:
    json.dump(schema, f, indent=2)
```

## Custom Evaluators

Custom evaluators require special handling during serialization and deserialization.

### Requirements

Custom evaluators must:

1. Be decorated with `@dataclass`
2. Inherit from `Evaluator`
3. Be passed to both `to_file()` and `from_file()`

### Complete Example

```python
from dataclasses import dataclass
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class CustomThreshold(Evaluator):
    """Check if output length exceeds a threshold."""

    min_length: int
    max_length: int = 100

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        length = len(str(ctx.output))
        return self.min_length <= length <= self.max_length


# Create dataset with custom evaluator
dataset = Dataset[str, str, Any](
    cases=[
        Case(
            name='test_length',
            inputs='example',
            expected_output='long result',
            evaluators=[
                CustomThreshold(min_length=5, max_length=20),
            ],
        ),
    ],
)

# Save with custom evaluator types
dataset.to_file(
    'dataset.yaml',
    custom_evaluator_types=[CustomThreshold],
)
```

### Saved YAML

```yaml
# yaml-language-server: $schema=dataset_schema.json
cases:
- name: test_length
  inputs: example
  expected_output: long result
  evaluators:
  - CustomThreshold:
      min_length: 5
      max_length: 20
```

### Loading with Custom Evaluators

```python
from dataclasses import dataclass
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class CustomThreshold(Evaluator):
    """Check if output length exceeds a threshold."""

    min_length: int
    max_length: int = 100

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        length = len(str(ctx.output))
        return self.min_length <= length <= self.max_length


# First create and save the dataset
dataset = Dataset[str, str, Any](
    cases=[
        Case(
            name='test_length',
            inputs='example',
            expected_output='long result',
            evaluators=[CustomThreshold(min_length=5, max_length=20)],
        ),
    ],
)
dataset.to_file('dataset.yaml', custom_evaluator_types=[CustomThreshold])

# Load with custom evaluator registry
dataset = Dataset[str, str, Any].from_file(
    'dataset.yaml',
    custom_evaluator_types=[CustomThreshold],
)
```

!!! warning "Important"
    You must pass `custom_evaluator_types` to **both** `to_file()` and `from_file()`.

    - `to_file()`: Includes the evaluator in the JSON schema
    - `from_file()`: Registers the evaluator for deserialization

## Evaluator Serialization Formats

Evaluators can be serialized in three forms:

### 1. Name Only (No Parameters)

```yaml
evaluators:
- EqualsExpected
- IsInstance: str  # Using default parameter
```

### 2. Single Parameter (Short Form)

```yaml
evaluators:
- IsInstance: str
- Contains: "required text"
- MaxDuration: 2.0
```

### 3. Multiple Parameters (Dict Form)

```yaml
evaluators:
- CustomThreshold:
    min_length: 5
    max_length: 20
- LLMJudge:
    rubric: "Response is accurate"
    model: "openai:gpt-5"
    include_input: true
```

## Format Comparison

| Feature | YAML | JSON |
|---------|------|------|
| Human readable | ✅ Excellent | ⚠️ Good |
| Comments | ✅ Yes | ❌ No |
| Compact | ✅ Yes | ⚠️ Verbose |
| Machine parsing | ✅ Good | ✅ Excellent |
| IDE support | ✅ Yes | ✅ Yes |
| Version control | ✅ Clean diffs | ⚠️ Noisy diffs |

**Recommendation**: Use YAML for most cases, JSON for programmatic generation.

## Advanced: Evaluator Serialization Name

Customize how your evaluator appears in serialized files:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class VeryLongDescriptiveEvaluatorName(Evaluator):
    @classmethod
    def get_serialization_name(cls) -> str:
        return 'ShortName'

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True
```

In YAML:

```yaml
evaluators:
- ShortName  # Instead of VeryLongDescriptiveEvaluatorName
```

## Troubleshooting

### Schema Not Found in IDE

**Problem**: YAML file doesn't show autocomplete

**Solutions**:

1. **Check the schema path** in the first line of YAML:
   ```yaml
   # yaml-language-server: $schema=correct_schema_name.json
   ```

2. **Verify schema file exists** in the same directory

3. **Restart the language server** in your IDE

4. **Install YAML extension** (VS Code: "YAML" by Red Hat)

### Custom Evaluator Not Found

**Problem**: `ValueError: Unknown evaluator name: 'CustomEvaluator'`

**Solution**: Pass `custom_evaluator_types` when loading:

```python
from dataclasses import dataclass
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class CustomEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


# First create and save with custom evaluator
dataset = Dataset[str, str, Any](
    cases=[Case(inputs='test', evaluators=[CustomEvaluator()])],
)
dataset.to_file('tests.yaml', custom_evaluator_types=[CustomEvaluator])

# Load with custom evaluator types
dataset = Dataset[str, str, Any].from_file(
    'tests.yaml',
    custom_evaluator_types=[CustomEvaluator],  # Required!
)
```

### Format Inference Failed

**Problem**: `ValueError: Cannot infer format from extension`

**Solution**: Specify format explicitly:

```python
from typing import Any

from pydantic_evals import Case, Dataset

dataset = Dataset[str, str, Any](cases=[Case(inputs='test')])

# Explicit format for unusual extensions
dataset.to_file('data.txt', fmt='yaml')
dataset_loaded = Dataset[str, str, Any].from_file('data.txt', fmt='yaml')
```

### Schema Generation Error

**Problem**: Custom evaluator causes schema generation to fail

**Solution**: Ensure evaluator is a proper dataclass:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


# ✅ Correct
@dataclass
class MyEvaluator(Evaluator):
    value: int

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


# ❌ Wrong: Missing @dataclass
class BadEvaluator(Evaluator):
    def __init__(self, value: int):
        self.value = value

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True
```

## Next Steps

- **[Dataset Management](dataset-management.md)** - Creating and organizing datasets
- **[Custom Evaluators](../evaluators/custom.md)** - Write custom evaluation logic
- **[Core Concepts](../core-concepts.md)** - Understand the data model
