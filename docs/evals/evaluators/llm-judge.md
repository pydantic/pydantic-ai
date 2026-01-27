# LLM Judge Deep Dive

The [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] evaluator uses an LLM to assess subjective qualities of outputs based on a rubric.

## When to Use LLM-as-a-Judge

LLM judges are ideal for evaluating qualities that require understanding and judgment:

**Good Use Cases:**

- Factual accuracy
- Helpfulness and relevance
- Tone and style compliance
- Completeness of responses
- Following complex instructions
- RAG groundedness (does the answer use provided context?)
- Citation accuracy

**Poor Use Cases:**

- Format validation (use [`IsInstance`][pydantic_evals.evaluators.IsInstance] instead)
- Exact matching (use [`EqualsExpected`][pydantic_evals.evaluators.EqualsExpected])
- Performance checks (use [`MaxDuration`][pydantic_evals.evaluators.MaxDuration])
- Deterministic logic (write a custom evaluator)

## Basic Usage

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

dataset = Dataset(
    cases=[Case(inputs='test')],
    evaluators=[
        LLMJudge(rubric='Response is factually accurate'),
    ],
)
```

## Configuration Options

### Rubric

The `rubric` is your evaluation criteria. Be specific and clear:

**Bad rubrics (vague):**
```python
from pydantic_evals.evaluators import LLMJudge

LLMJudge(rubric='Good response')  # Too vague
LLMJudge(rubric='Check quality')  # What aspect of quality?
```

**Good rubrics (specific):**
```python
from pydantic_evals.evaluators import LLMJudge

LLMJudge(rubric='Response directly answers the user question without hallucination')
LLMJudge(rubric='Response uses formal, professional language appropriate for business communication')
LLMJudge(rubric='All factual claims in the response are supported by the provided context')
```

### Including Context

Control what information the judge sees:

```python
from pydantic_evals.evaluators import LLMJudge

# Output only (default)
LLMJudge(rubric='Response is polite')

# Output + Input
LLMJudge(
    rubric='Response accurately answers the input question',
    include_input=True,
)

# Output + Input + Expected Output
LLMJudge(
    rubric='Response is semantically equivalent to the expected output',
    include_input=True,
    include_expected_output=True,
)
```

**Example:**
```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

dataset = Dataset(
    cases=[
        Case(
            inputs='What is 2+2?',
            expected_output='4',
        ),
    ],
    evaluators=[
        # This judge sees: output + inputs + expected_output
        LLMJudge(
            rubric='Response provides the same answer as expected, possibly with explanation',
            include_input=True,
            include_expected_output=True,
        ),
    ],
)
```

### Model Selection

Choose the judge model based on cost/quality tradeoffs:

```python
from pydantic_evals.evaluators import LLMJudge

# Default: GPT-4o (good balance)
LLMJudge(rubric='...')

# Anthropic Claude (alternative default)
LLMJudge(
    rubric='...',
    model='anthropic:claude-sonnet-4-5',
)

# Cheaper option for simple checks
LLMJudge(
    rubric='Response contains profanity',
    model='openai:gpt-5-mini',
)

# Premium option for nuanced evaluation
LLMJudge(
    rubric='Response demonstrates deep understanding of quantum mechanics',
    model='anthropic:claude-opus-4-20250514',
)
```

### Model Settings

Customize model behavior:

```python
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import LLMJudge

LLMJudge(
    rubric='...',
    model_settings=ModelSettings(
        temperature=0.0,  # Deterministic evaluation
        max_tokens=100,  # Shorter responses
    ),
)
```

## Output Modes

### Assertion Only (Default)

Returns pass/fail with reason:

```python
from pydantic_evals.evaluators import LLMJudge

LLMJudge(rubric='Response is accurate')
# Returns: {'LLMJudge_pass': EvaluationReason(value=True, reason='...')}
```

In reports:
```
┃ Assertions ┃
┃ ✔          ┃
```

### Score Only

Returns a numeric score (0.0 to 1.0):

```python
from pydantic_evals.evaluators import LLMJudge

LLMJudge(
    rubric='Response quality',
    score={'include_reason': True},
    assertion=False,
)
# Returns: {'LLMJudge_score': EvaluationReason(value=0.85, reason='...')}
```

In reports:
```
┃ Scores             ┃
┃ LLMJudge_score: 0.85 ┃
```

### Both Score and Assertion

```python
from pydantic_evals.evaluators import LLMJudge

LLMJudge(
    rubric='Response quality',
    score={'include_reason': True},
    assertion={'include_reason': True},
)
# Returns: {
#     'LLMJudge_score': EvaluationReason(value=0.85, reason='...'),
#     'LLMJudge_pass': EvaluationReason(value=True, reason='...'),
# }
```

### Custom Names

```python
from pydantic_evals.evaluators import LLMJudge

LLMJudge(
    rubric='Response is factually accurate',
    assertion={
        'evaluation_name': 'accuracy',
        'include_reason': True,
    },
)
# Returns: {'accuracy': EvaluationReason(value=True, reason='...')}
```

In reports:
```
┃ Assertions ┃
┃ accuracy: ✔ ┃
```

## Practical Examples

### RAG Evaluation

Evaluate whether a RAG system uses provided context:

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge


@dataclass
class RAGInput:
    question: str
    context: str


dataset = Dataset(
    cases=[
        Case(
            inputs=RAGInput(
                question='What is the capital of France?',
                context='France is a country in Europe. Its capital is Paris.',
            ),
        ),
    ],
    evaluators=[
        LLMJudge(
            rubric='Response answers the question using only information from the provided context',
            include_input=True,
            assertion={'evaluation_name': 'grounded', 'include_reason': True},
        ),
        LLMJudge(
            rubric='Response cites specific quotes or facts from the context',
            include_input=True,
            assertion={'evaluation_name': 'uses_citations', 'include_reason': True},
        ),
    ],
)
```

### Recipe Generation with Case-Specific Rubrics

This example shows how to use both dataset-level and case-specific evaluators:

```python {title="recipe_evaluation.py" test="skip"}
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from pydantic_ai import Agent, format_as_xml
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge


class CustomerOrder(BaseModel):
    dish_name: str
    dietary_restriction: str | None = None


class Recipe(BaseModel):
    ingredients: list[str]
    steps: list[str]


recipe_agent = Agent(
    'openai:gpt-5-mini',
    output_type=Recipe,
    instructions=(
        'Generate a recipe to cook the dish that meets the dietary restrictions.'
    ),
)


async def transform_recipe(customer_order: CustomerOrder) -> Recipe:
    r = await recipe_agent.run(format_as_xml(customer_order))
    return r.output


recipe_dataset = Dataset[CustomerOrder, Recipe, Any](
    cases=[
        Case(
            name='vegetarian_recipe',
            inputs=CustomerOrder(
                dish_name='Spaghetti Bolognese', dietary_restriction='vegetarian'
            ),
            expected_output=None,
            metadata={'focus': 'vegetarian'},
            evaluators=(  # (1)!
                LLMJudge(
                    rubric='Recipe should not contain meat or animal products',
                ),
            ),
        ),
        Case(
            name='gluten_free_recipe',
            inputs=CustomerOrder(
                dish_name='Chocolate Cake', dietary_restriction='gluten-free'
            ),
            expected_output=None,
            metadata={'focus': 'gluten-free'},
            evaluators=(  # (2)!
                LLMJudge(
                    rubric='Recipe should not contain gluten or wheat products',
                ),
            ),
        ),
    ],
    evaluators=[  # (3)!
        IsInstance(type_name='Recipe'),
        LLMJudge(
            rubric='Recipe should have clear steps and relevant ingredients',
            include_input=True,
            model='anthropic:claude-sonnet-4-5',
        ),
    ],
)


report = recipe_dataset.evaluate_sync(transform_recipe)
print(report)
"""
     Evaluation Summary: transform_recipe
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID            ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ vegetarian_recipe  │ ✔✔✔        │    38.1s │
├────────────────────┼────────────┼──────────┤
│ gluten_free_recipe │ ✔✔✔        │    22.4s │
├────────────────────┼────────────┼──────────┤
│ Averages           │ 100.0% ✔   │    30.3s │
└────────────────────┴────────────┴──────────┘
"""
```

1. Case-specific evaluator - only runs for the vegetarian recipe case
2. Case-specific evaluator - only runs for the gluten-free recipe case
3. Dataset-level evaluators - run for all cases

### Multi-Aspect Evaluation

Use multiple judges for different quality dimensions:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

dataset = Dataset(
    cases=[Case(inputs='test')],
    evaluators=[
        # Accuracy
        LLMJudge(
            rubric='Response is factually accurate',
            include_input=True,
            assertion={'evaluation_name': 'accurate'},
        ),

        # Helpfulness
        LLMJudge(
            rubric='Response is helpful and actionable',
            include_input=True,
            score={'evaluation_name': 'helpfulness'},
            assertion=False,
        ),

        # Tone
        LLMJudge(
            rubric='Response uses professional, respectful language',
            assertion={'evaluation_name': 'professional_tone'},
        ),

        # Safety
        LLMJudge(
            rubric='Response contains no harmful, biased, or inappropriate content',
            assertion={'evaluation_name': 'safe'},
        ),
    ],
)
```

### Comparative Evaluation

Compare output against expected output:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

dataset = Dataset(
    cases=[
        Case(
            name='translation',
            inputs='Hello world',
            expected_output='Bonjour le monde',
        ),
    ],
    evaluators=[
        LLMJudge(
            rubric='Response is semantically equivalent to the expected output',
            include_input=True,
            include_expected_output=True,
            score={'evaluation_name': 'semantic_similarity'},
            assertion={'evaluation_name': 'correct_meaning'},
        ),
    ],
)
```

## Best Practices

### 1. Be Specific in Rubrics

**Bad:**
```python
from pydantic_evals.evaluators import LLMJudge

LLMJudge(rubric='Good answer')
```

**Better:**
```python
from pydantic_evals.evaluators import LLMJudge

LLMJudge(rubric='Response accurately answers the question without hallucinating facts')
```

**Best:**
```python
from pydantic_evals.evaluators import LLMJudge

LLMJudge(
    rubric='''
    Response must:
    1. Directly answer the question asked
    2. Use only information from the provided context
    3. Cite specific passages from the context
    4. Acknowledge if information is insufficient
    ''',
    include_input=True,
)
```

### 2. Use Multiple Judges

Don't always try to evaluate everything with one rubric:

```python
from pydantic_evals.evaluators import LLMJudge

# Instead of this:
LLMJudge(rubric='Response is good, accurate, helpful, and safe')

# Do this:
evaluators = [
    LLMJudge(rubric='Response is factually accurate'),
    LLMJudge(rubric='Response is helpful and actionable'),
    LLMJudge(rubric='Response is safe and appropriate'),
]
```

### 3. Combine with Deterministic Checks

Don't use LLM evaluation for checks that can be done deterministically:

```python
from pydantic_evals.evaluators import Contains, IsInstance, LLMJudge

evaluators = [
    IsInstance(type_name='str'),
    Contains(value='required_section'),
    LLMJudge(rubric='Response quality is high'),
]
```

### 4. Use Temperature 0 for Consistency

```python
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import LLMJudge

LLMJudge(
    rubric='...',
    model_settings=ModelSettings(temperature=0.0),
)
```


## Limitations

### Non-Determinism

LLM judges are not deterministic. The same output may receive different scores across runs.

**Mitigation:**

- Use `temperature=0.0` for more consistency
- Run multiple evaluations and average
- Use retry strategies for flaky evaluations

### Cost

LLM judges make API calls, which cost money and time.

**Mitigation:**

- Use cheaper models for simple checks (`gpt-5-mini`)
- Run deterministic checks first to fail fast
- Cache results when possible
- Limit evaluation to changed cases

### Model Biases

LLM judges inherit biases from their training data.

**Mitigation:**

- Use multiple judge models and compare
- Review evaluation reasons, not just scores
- Validate judges against human-labeled test sets
- Be aware of known biases (length bias, style preferences)

### Context Limits

Judges have token limits for inputs.

**Mitigation:**

- Truncate long inputs/outputs intelligently
- Use focused rubrics that don't require full context
- Consider chunked evaluation for very long content

## Debugging LLM Judges

### View Reasons

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge


def my_task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(
    cases=[Case(inputs='test')],
    evaluators=[LLMJudge(rubric='Response is clear')],
)
report = dataset.evaluate_sync(my_task)
report.print(include_reasons=True)
"""
     Evaluation Summary: my_task
┏━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID  ┃ Assertions  ┃ Duration ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Case 1   │ LLMJudge: ✔ │     10ms │
│          │   Reason: - │          │
│          │             │          │
│          │             │          │
├──────────┼─────────────┼──────────┤
│ Averages │ 100.0% ✔    │     10ms │
└──────────┴─────────────┴──────────┘
"""
```

Output:
```
┃ Assertions              ┃
┃ accuracy: ✔            ┃
┃   Reason: The response │
┃   correctly states...  │
```

### Access Programmatically

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge


def my_task(inputs: str) -> str:
    return f'Result: {inputs}'


dataset = Dataset(
    cases=[Case(inputs='test')],
    evaluators=[LLMJudge(rubric='Response is clear')],
)
report = dataset.evaluate_sync(my_task)
for case in report.cases:
    for name, result in case.assertions.items():
        print(f'{name}: {result.value}')
        #> LLMJudge: True
        if result.reason:
            print(f'  Reason: {result.reason}')
            #>   Reason: -
```

### Compare Judges

Test the same cases with different judge models:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge


def my_task(inputs: str) -> str:
    return f'Result: {inputs}'


judges = [
    LLMJudge(rubric='Response is clear', model='openai:gpt-5'),
    LLMJudge(rubric='Response is clear', model='anthropic:claude-sonnet-4-5'),
    LLMJudge(rubric='Response is clear', model='openai:gpt-5-mini'),
]

for judge in judges:
    dataset = Dataset(cases=[Case(inputs='test')], evaluators=[judge])
    report = dataset.evaluate_sync(my_task)
    # Compare results
```

## Advanced: Custom Judge Models

Set a default judge model for all `LLMJudge` evaluators:

```python
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model

# Set default to Claude
set_default_judge_model('anthropic:claude-sonnet-4-5')

# Now all LLMJudge instances use Claude by default
LLMJudge(rubric='...')  # Uses Claude
```

## Next Steps

- **[Custom Evaluators](custom.md)** - Write custom evaluation logic
- **[Built-in Evaluators](built-in.md)** - Complete evaluator reference
