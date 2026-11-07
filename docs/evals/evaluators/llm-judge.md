---
description: "Use an LLM as a judge in Pydantic Evals: grade outputs against a rubric with LLMJudge, get pass/fail or scores, pick the judge model and write good rubrics."
---

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
    name='factual_accuracy',
    cases=[Case(inputs='test')],
    evaluators=[
        LLMJudge(rubric='Response is factually accurate'),
    ],
)
```

## Configuration Options

### Rubric

The `rubric` defines the evaluation criteria. Be specific and clear:

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
    name='math_check',
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

# Uses the default judge model
LLMJudge(rubric='...')

# Anthropic Claude (alternative default)
LLMJudge(
    rubric='...',
    model='{{ anthropic:medium }}',
)

# Lightweight option for simple checks
LLMJudge(
    rubric='Response contains profanity',
    model='{{ openai:small }}',
)

# Most capable option for nuanced evaluation
LLMJudge(
    rubric='Response demonstrates deep understanding of quantum mechanics',
    model='{{ openai:large }}',
)
```

When the judge model cannot generate text, like a [decision model](../../models/decision.md) such as [TypeSafe's Jev](../../models/typesafe.md), `LLMJudge` asks it only for the pass/fail verdict. The result has no reason rather than a fabricated one, and its score is `1.0` for pass or `0.0` for fail. A model reports this through [`supports_text_output=False`][pydantic_ai.profiles.ModelProfile.supports_text_output] on its profile.

The verdict is one yes-or-no question naming only the sections the prompt carries, so a rubric judged without an expected output is not asked to weigh one.

### Model Settings

Customize model behavior:

```python
from pydantic_ai import ModelSettings
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
    name='rag_evaluation',
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
    '{{ openai:small }}',
    output_type=Recipe,
    instructions=(
        'Generate a recipe to cook the dish that meets the dietary restrictions.'
    ),
)


async def transform_recipe(customer_order: CustomerOrder) -> Recipe:
    r = await recipe_agent.run(format_as_xml(customer_order))
    return r.output


recipe_dataset = Dataset[CustomerOrder, Recipe, Any](
    name='recipe_evaluation',
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
            model='{{ anthropic:medium }}',
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
    name='multi_aspect',
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
    name='comparative_eval',
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

## Validating a Judge Against Human Labels

A judge can disagree with your own reading of the rubric. Before a judge gates anything, such as a release check or an [online evaluation](../online-evaluation.md) alert, measure how often it agrees with people:

1. Collect a sample of outputs your application produced, with both good and bad examples.
2. Have a person apply the rubric to each output, and record the verdict in the case metadata.
3. Run the judge as a case evaluator over the stored outputs, using a task that returns each output unchanged, so the judge sees exactly what the person labeled.
4. Compare the judge's verdicts with the human verdicts in a [report evaluator](report-evaluators.md).

[`ConfusionMatrixEvaluator`][pydantic_evals.evaluators.ConfusionMatrixEvaluator] reads predictions from the output, metadata or labels, not from assertions, so a short custom report evaluator tabulates the judge's pass/fail assertion instead:

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge, ReportEvaluator, ReportEvaluatorContext
from pydantic_evals.reporting.analyses import ConfusionMatrix, ReportAnalysis, ScalarResult

# Outputs your application produced, each with the verdict of a person applying the rubric
labeled_outputs = [
    ('Your refund for order 12345 has been issued.', 'pass'),
    ('Refunds are handled by our billing team.', 'fail'),
]


def replay(output: str) -> str:
    return output


@dataclass
class JudgeAgreement(ReportEvaluator):
    assertion_name: str

    def evaluate(self, ctx: ReportEvaluatorContext) -> list[ReportAnalysis]:
        verdicts = ['fail', 'pass']
        matrix = [[0, 0], [0, 0]]
        for case in ctx.report.cases:
            judged = case.assertions.get(self.assertion_name)
            human_verdict = (case.metadata or {}).get('human_verdict')
            if judged is None or human_verdict not in verdicts:  # the judge failed, or the case is unlabeled
                continue
            matrix[verdicts.index(human_verdict)][1 if judged.value else 0] += 1
        judged_cases = sum(map(sum, matrix))
        agreed = matrix[0][0] + matrix[1][1]
        return [
            ConfusionMatrix(
                title='Human verdict (rows) vs. judge verdict (columns)',
                class_labels=verdicts,
                matrix=matrix,
            ),
            ScalarResult(
                title='Judge agreement',
                value=agreed / judged_cases * 100 if judged_cases else 0,
                unit='%',
            ),
        ]


dataset = Dataset(
    name='refund_judge_validation',
    cases=[
        Case(inputs=output, metadata={'human_verdict': verdict})
        for output, verdict in labeled_outputs
    ],
    evaluators=[
        LLMJudge(
            rubric='The response confirms that the refund was issued and quotes the order ID.',
            model='{{ openai:small }}',
            assertion={'evaluation_name': 'judge_pass', 'include_reason': True},
        ),
    ],
    report_evaluators=[JudgeAgreement(assertion_name='judge_pass')],
)
report = dataset.evaluate_sync(replay)
```

The analyses are on `report.analyses`, and in [Logfire](report-evaluators.md#viewing-analyses-in-logfire) when it's configured. For a gate, watch the matrix's `fail` row, `pass` column: outputs a person failed that the judge let through. Read the judge's reasons on the cases it got wrong, refine the rubric or switch the judge model, and re-run until the agreement is acceptable. Keep the labeled dataset, and re-run it whenever you change the rubric or the judge model.

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
from pydantic_ai import ModelSettings
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

- Use lightweight models for simple checks (`{{ openai:small.name }}`)
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
    name='debug_reasons',
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
    name='programmatic_access',
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
    LLMJudge(rubric='Response is clear', model='{{ openai:medium }}'),
    LLMJudge(rubric='Response is clear', model='{{ anthropic:medium }}'),
    LLMJudge(rubric='Response is clear', model='{{ openai:small }}'),
]

for judge in judges:
    dataset = Dataset(name='judge_comparison', cases=[Case(inputs='test')], evaluators=[judge])
    report = dataset.evaluate_sync(my_task)
    # Compare results
```

## Advanced: Custom Judge Models

Set a default judge model for all `LLMJudge` evaluators:

```python
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model

# Set default to Claude
set_default_judge_model('{{ anthropic:medium }}')

# Now all LLMJudge instances use Claude by default
LLMJudge(rubric='...')  # Uses Claude
```

## Next Steps

- **[Custom Evaluators](custom.md)** - Write custom evaluation logic
- **[Native Evaluators](built-in.md)** - Complete evaluator reference
- **[LLM-as-a-Judge: A Practical Guide with Pydantic Evals](https://pydantic.dev/articles/llm-as-a-judge)** - Case-specific judges and writing good rubrics, on the Pydantic blog
