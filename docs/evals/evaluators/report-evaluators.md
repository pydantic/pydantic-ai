# Report Evaluators

Report evaluators analyze entire experiment results rather than individual cases. Use them to compute
experiment-wide statistics like confusion matrices, precision-recall curves, accuracy scores, or
custom summary tables.

## How Report Evaluators Work

Regular [evaluators](overview.md) run once per case and assess individual outputs.
Report evaluators run once per experiment _after_ all cases have been evaluated,
receiving the full [`EvaluationReport`][pydantic_evals.reporting.EvaluationReport] as input.

```
Cases executed → Case evaluators run → Report evaluators run → Final report
```

Results from report evaluators are stored as **analyses** on the report and, when Logfire is
configured, are attached to the experiment span as structured attributes for visualization.

## Using Report Evaluators

Pass report evaluators to `Dataset` via the `report_evaluators` parameter:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ConfusionMatrixEvaluator


def my_classifier(text: str) -> str:
    text = text.lower()
    if 'cat' in text or 'meow' in text:
        return 'cat'
    elif 'dog' in text or 'bark' in text:
        return 'dog'
    return 'unknown'


dataset = Dataset(
    cases=[
        Case(name='cat', inputs='The cat goes meow', expected_output='cat'),
        Case(name='dog', inputs='The dog barks', expected_output='dog'),
    ],
    report_evaluators=[
        ConfusionMatrixEvaluator(
            predicted_from='output',
            expected_from='expected_output',
            title='Animal Classification',
        ),
    ],
)

report = dataset.evaluate_sync(my_classifier)
# report.analyses contains the ConfusionMatrix result
```

## Built-in Report Evaluators

### ConfusionMatrixEvaluator

Builds a confusion matrix comparing predicted vs expected labels across all cases.

```python
from pydantic_evals.evaluators import ConfusionMatrixEvaluator

ConfusionMatrixEvaluator(
    predicted_from='output',
    expected_from='expected_output',
    title='My Confusion Matrix',
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predicted_from` | `'expected_output' \| 'output' \| 'metadata' \| 'labels'` | `'output'` | Source for predicted values |
| `predicted_key` | `str \| None` | `None` | Key to extract when using `metadata` or `labels` |
| `expected_from` | `'expected_output' \| 'output' \| 'metadata' \| 'labels'` | `'expected_output'` | Source for expected/true values |
| `expected_key` | `str \| None` | `None` | Key to extract when using `metadata` or `labels` |
| `title` | `str` | `'Confusion Matrix'` | Title shown in reports |

**Returns:** [`ConfusionMatrix`][pydantic_evals.reporting.analyses.ConfusionMatrix]

**Data Sources:**

- `'output'` — the task's actual output (converted to string)
- `'expected_output'` — the case's expected output (converted to string)
- `'metadata'` — a value from the case's metadata dict (requires `key`)
- `'labels'` — a label result from a case-level evaluator (requires `key`)

**Example — classification with expected outputs:**

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ConfusionMatrixEvaluator

dataset = Dataset(
    cases=[
        Case(inputs='meow', expected_output='cat'),
        Case(inputs='woof', expected_output='dog'),
        Case(inputs='chirp', expected_output='bird'),
    ],
    report_evaluators=[
        ConfusionMatrixEvaluator(
            predicted_from='output',
            expected_from='expected_output',
        ),
    ],
)
```

**Example — using evaluator labels:**

If a case-level evaluator produces a label like `predicted_class`, you can reference it:

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    ConfusionMatrixEvaluator,
    Evaluator,
    EvaluatorContext,
)


@dataclass
class ClassifyOutput(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, str]:
        # Classify the output into a category
        return {'predicted_class': categorize(ctx.output)}


def categorize(output: str) -> str:
    return 'positive' if 'good' in output.lower() else 'negative'


dataset = Dataset(
    cases=[Case(inputs='test', expected_output='positive')],
    evaluators=[ClassifyOutput()],
    report_evaluators=[
        ConfusionMatrixEvaluator(
            predicted_from='labels',
            predicted_key='predicted_class',
            expected_from='expected_output',
        ),
    ],
)
```

---

### PrecisionRecallEvaluator

Computes a precision-recall curve with AUC (area under the curve) from numeric scores
and binary ground-truth labels.

```python
from pydantic_evals.evaluators import PrecisionRecallEvaluator

PrecisionRecallEvaluator(
    score_from='scores',
    score_key='confidence',
    positive_from='assertions',
    positive_key='is_correct',
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_key` | `str` | _(required)_ | Key in scores or metrics dict |
| `positive_from` | `'expected_output' \| 'assertions' \| 'labels'` | _(required)_ | Source for ground-truth binary labels |
| `positive_key` | `str \| None` | `None` | Key in assertions or labels dict |
| `score_from` | `'scores' \| 'metrics'` | `'scores'` | Source for numeric scores |
| `title` | `str` | `'Precision-Recall Curve'` | Title shown in reports |
| `n_thresholds` | `int` | `100` | Number of threshold points on the curve |

**Returns:** [`PrecisionRecall`][pydantic_evals.reporting.analyses.PrecisionRecall] + [`ScalarResult`][pydantic_evals.reporting.analyses.ScalarResult] (AUC)

The AUC is computed at full resolution (using every unique score as a threshold) for accuracy,
then the curve points are downsampled to `n_thresholds` for display. The AUC is returned both
on the curve (for chart rendering) and as a separate `ScalarResult` for querying and sorting.

**Score Sources:**

- `'scores'` — a numeric score from a case-level evaluator (looked up by `score_key`)
- `'metrics'` — a custom metric set during task execution (looked up by `score_key`)

**Positive Sources:**

- `'assertions'` — a boolean assertion from a case-level evaluator (looked up by `positive_key`)
- `'labels'` — a label result cast to boolean (looked up by `positive_key`)
- `'expected_output'` — the case's expected output cast to boolean

**Example:**

```python
from dataclasses import dataclass
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    PrecisionRecallEvaluator,
)


@dataclass
class ConfidenceEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, Any]:
        confidence = calculate_confidence(ctx.output)
        return {
            'confidence': confidence,      # numeric score
            'is_correct': ctx.output == ctx.expected_output,  # boolean assertion
        }


def calculate_confidence(output: str) -> float:
    return 0.85  # placeholder


dataset = Dataset(
    cases=[
        Case(inputs='test 1', expected_output='cat'),
        Case(inputs='test 2', expected_output='dog'),
    ],
    evaluators=[ConfidenceEvaluator()],
    report_evaluators=[
        PrecisionRecallEvaluator(
            score_from='scores',
            score_key='confidence',
            positive_from='assertions',
            positive_key='is_correct',
        ),
    ],
)
```

---

### ROCAUCEvaluator

Computes an ROC (Receiver Operating Characteristic) curve and AUC from numeric scores
and binary ground-truth labels. The ROC curve plots the True Positive Rate against the
False Positive Rate at various threshold values, with a dashed random-baseline diagonal
for reference.

```python
from pydantic_evals.evaluators import ROCAUCEvaluator

ROCAUCEvaluator(
    score_key='confidence',
    positive_from='assertions',
    positive_key='is_correct',
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_key` | `str` | _(required)_ | Key in scores or metrics dict |
| `positive_from` | `'expected_output' \| 'assertions' \| 'labels'` | _(required)_ | Source for ground-truth binary labels |
| `positive_key` | `str \| None` | `None` | Key in assertions or labels dict |
| `score_from` | `'scores' \| 'metrics'` | `'scores'` | Source for numeric scores |
| `title` | `str` | `'ROC Curve'` | Title shown in reports |
| `n_thresholds` | `int` | `100` | Number of threshold points on the curve |

**Returns:** [`LinePlot`][pydantic_evals.reporting.analyses.LinePlot] + [`ScalarResult`][pydantic_evals.reporting.analyses.ScalarResult] (AUC)

The AUC is computed at full resolution. The chart includes a dashed "Random" baseline
diagonal from (0, 0) to (1, 1) for visual comparison.

**Score and Positive Sources:** Same as [`PrecisionRecallEvaluator`](#precisionrecallevaluator).

---

### KolmogorovSmirnovEvaluator

Computes a Kolmogorov-Smirnov plot and KS statistic from numeric scores and binary
ground-truth labels. The KS plot shows the empirical CDFs (cumulative distribution functions)
of the score distribution for positive and negative cases. The KS statistic is the maximum
vertical distance between the two CDFs — higher values indicate better class separation.

```python
from pydantic_evals.evaluators import KolmogorovSmirnovEvaluator

KolmogorovSmirnovEvaluator(
    score_key='confidence',
    positive_from='assertions',
    positive_key='is_correct',
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_key` | `str` | _(required)_ | Key in scores or metrics dict |
| `positive_from` | `'expected_output' \| 'assertions' \| 'labels'` | _(required)_ | Source for ground-truth binary labels |
| `positive_key` | `str \| None` | `None` | Key in assertions or labels dict |
| `score_from` | `'scores' \| 'metrics'` | `'scores'` | Source for numeric scores |
| `title` | `str` | `'KS Plot'` | Title shown in reports |
| `n_thresholds` | `int` | `100` | Number of threshold points on the curve |

**Returns:** [`LinePlot`][pydantic_evals.reporting.analyses.LinePlot] + [`ScalarResult`][pydantic_evals.reporting.analyses.ScalarResult] (KS Statistic)

**Score and Positive Sources:** Same as [`PrecisionRecallEvaluator`](#precisionrecallevaluator).

---

## Custom Report Evaluators

Write custom report evaluators by inheriting from [`ReportEvaluator`][pydantic_evals.evaluators.ReportEvaluator]
and implementing the `evaluate` method:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import ReportEvaluator, ReportEvaluatorContext
from pydantic_evals.reporting.analyses import ScalarResult


@dataclass
class AccuracyEvaluator(ReportEvaluator):
    """Computes overall accuracy as a scalar metric."""

    def evaluate(self, ctx: ReportEvaluatorContext) -> ScalarResult:
        cases = ctx.report.cases
        if not cases:
            return ScalarResult(title='Accuracy', value=0.0, unit='%')

        correct = sum(
            1 for case in cases
            if case.output == case.expected_output
        )
        accuracy = correct / len(cases) * 100
        return ScalarResult(title='Accuracy', value=accuracy, unit='%')
```

### ReportEvaluatorContext

The context passed to `evaluate()` contains:

- `ctx.name` — the experiment name
- `ctx.report` — the full [`EvaluationReport`][pydantic_evals.reporting.EvaluationReport] with all case results
- `ctx.experiment_metadata` — optional experiment-level metadata dict

Through `ctx.report.cases`, you can access each case's inputs, outputs, expected outputs, scores,
labels, assertions, metrics, and attributes.

### Return Types

Report evaluators must return a `ReportAnalysis` or a `list[ReportAnalysis]`. The available
analysis types are:

#### ScalarResult

A single numeric statistic:

```python
from pydantic_evals.reporting.analyses import ScalarResult

ScalarResult(
    title='Accuracy',
    value=93.3,
    unit='%',
    description='Percentage of correctly classified cases.',
)
```

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Display name |
| `value` | `float \| int` | The numeric value |
| `unit` | `str \| None` | Optional unit label (e.g., `'%'`, `'ms'`) |
| `description` | `str \| None` | Optional longer description |

---

#### TableResult

A generic table of data:

```python
from pydantic_evals.reporting.analyses import TableResult

TableResult(
    title='Per-Class Metrics',
    columns=['Class', 'Precision', 'Recall', 'F1'],
    rows=[
        ['cat', 0.95, 0.90, 0.924],
        ['dog', 0.88, 0.92, 0.899],
    ],
    description='Precision, recall, and F1 per class.',
)
```

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Display name |
| `columns` | `list[str]` | Column headers |
| `rows` | `list[list[str \| int \| float \| bool \| None]]` | Row data |
| `description` | `str \| None` | Optional longer description |

---

#### ConfusionMatrix

A confusion matrix (typically produced by `ConfusionMatrixEvaluator`, but can be constructed directly):

```python
from pydantic_evals.reporting.analyses import ConfusionMatrix

ConfusionMatrix(
    title='Sentiment',
    class_labels=['positive', 'negative', 'neutral'],
    matrix=[
        [45, 3, 2],   # expected=positive
        [5, 40, 5],   # expected=negative
        [1, 2, 47],   # expected=neutral
    ],
)
```

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Display name |
| `class_labels` | `list[str]` | Ordered labels for both axes |
| `matrix` | `list[list[int]]` | `matrix[expected][predicted]` = count |
| `description` | `str \| None` | Optional longer description |

---

#### PrecisionRecall

Precision-recall curve data (typically produced by `PrecisionRecallEvaluator`):

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Display name |
| `curves` | `list[PrecisionRecallCurve]` | One or more curves |
| `description` | `str \| None` | Optional longer description |

Each `PrecisionRecallCurve` contains a `name`, a list of `PrecisionRecallPoint`s (with `threshold`,
`precision`, `recall`), and an optional `auc` value.

---

#### LinePlot

A generic XY line chart with labeled axes, supporting multiple curves. Use this for ROC curves,
KS plots, calibration curves, or any custom line chart:

```python
from pydantic_evals.reporting.analyses import LinePlot, LinePlotCurve, LinePlotPoint

LinePlot(
    title='ROC Curve',
    x_label='False Positive Rate',
    y_label='True Positive Rate',
    x_range=(0, 1),
    y_range=(0, 1),
    curves=[
        LinePlotCurve(
            name='Model (AUC: 0.95)',
            points=[LinePlotPoint(x=0.0, y=0.0), LinePlotPoint(x=0.1, y=0.8), LinePlotPoint(x=1.0, y=1.0)],
        ),
        LinePlotCurve(
            name='Random',
            points=[LinePlotPoint(x=0, y=0), LinePlotPoint(x=1, y=1)],
            style='dashed',
        ),
    ],
)
```

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Display name |
| `x_label` | `str` | Label for the x-axis |
| `y_label` | `str` | Label for the y-axis |
| `x_range` | `tuple[float, float] \| None` | Optional fixed range for x-axis |
| `y_range` | `tuple[float, float] \| None` | Optional fixed range for y-axis |
| `curves` | `list[LinePlotCurve]` | One or more curves to plot |
| `description` | `str \| None` | Optional longer description |

Each `LinePlotCurve` contains a `name`, a list of `LinePlotPoint`s (with `x`, `y`),
an optional `style` (`'solid'` or `'dashed'`), and an optional `step` interpolation
mode (`'start'`, `'middle'`, or `'end'`) for step functions like empirical CDFs.

`LinePlot` is the recommended return type for custom curve-based evaluators — any evaluator
that returns a `LinePlot` will be rendered as a line chart in the Logfire UI without requiring
any frontend changes.

### Returning Multiple Analyses

A single report evaluator can return multiple analyses by returning a list:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import ReportEvaluator, ReportEvaluatorContext
from pydantic_evals.reporting.analyses import ReportAnalysis, ScalarResult, TableResult


@dataclass
class ClassificationSummary(ReportEvaluator):
    """Produces both a scalar accuracy and a per-class metrics table."""

    def evaluate(self, ctx: ReportEvaluatorContext) -> list[ReportAnalysis]:
        cases = ctx.report.cases
        if not cases:
            return []

        labels = sorted({str(c.expected_output) for c in cases if c.expected_output})

        # Scalar: overall accuracy
        correct = sum(1 for c in cases if c.output == c.expected_output)
        accuracy = ScalarResult(
            title='Accuracy', value=correct / len(cases) * 100, unit='%'
        )

        # Table: per-class breakdown
        rows = []
        for label in labels:
            tp = sum(1 for c in cases if str(c.output) == label and str(c.expected_output) == label)
            fp = sum(1 for c in cases if str(c.output) == label and str(c.expected_output) != label)
            fn = sum(1 for c in cases if str(c.output) != label and str(c.expected_output) == label)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            rows.append([label, round(p, 3), round(r, 3), round(f1, 3)])

        table = TableResult(
            title='Per-Class Metrics',
            columns=['Class', 'Precision', 'Recall', 'F1'],
            rows=rows,
        )

        return [accuracy, table]
```

### Async Report Evaluators

Report evaluators support async `evaluate` methods, handled automatically via `evaluate_async`:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import ReportEvaluator, ReportEvaluatorContext
from pydantic_evals.reporting.analyses import ScalarResult


@dataclass
class AsyncAccuracy(ReportEvaluator):
    async def evaluate(self, ctx: ReportEvaluatorContext) -> ScalarResult:
        # Can use async I/O here (e.g., call an external API)
        cases = ctx.report.cases
        correct = sum(1 for c in cases if c.output == c.expected_output)
        return ScalarResult(
            title='Accuracy',
            value=correct / len(cases) * 100 if cases else 0.0,
            unit='%',
        )
```

## Serialization

Report evaluators are serialized to and from YAML/JSON dataset files using the same format as
case-level evaluators. This means datasets with report evaluators can be fully round-tripped
through file serialization.

**Example YAML dataset with report evaluators:**

```yaml
# yaml-language-server: $schema=./test_cases_schema.json
name: classifier_eval
cases:
  - name: cat_test
    inputs: The cat meows
    expected_output: cat
  - name: dog_test
    inputs: The dog barks
    expected_output: dog
report_evaluators:
  - ConfusionMatrixEvaluator
  - PrecisionRecallEvaluator:
      score_key: confidence
      positive_from: assertions
      positive_key: is_correct
```

Built-in report evaluators (`ConfusionMatrixEvaluator`, `PrecisionRecallEvaluator`,
`ROCAUCEvaluator`, `KolmogorovSmirnovEvaluator`) are recognized automatically. For custom report evaluators, pass them via `custom_report_evaluator_types`:

```python {test="skip" lint="skip"}
from pydantic_evals import Dataset

dataset = Dataset[str, str, None].from_file(
    'test_cases.yaml',
    custom_report_evaluator_types=[MyCustomReportEvaluator],
)
```

Similarly, when saving a dataset with custom report evaluators, pass them to `to_file` so the
JSON schema includes them:

```python {test="skip" lint="skip"}
dataset.to_file(
    'test_cases.yaml',
    custom_report_evaluator_types=[MyCustomReportEvaluator],
)
```

## Viewing Analyses in Logfire

When [Logfire is configured](../how-to/logfire-integration.md), analyses are automatically attached
to the experiment span as the `logfire.experiment.analyses` attribute. The Logfire UI renders them
as interactive visualizations:

- **Confusion matrices** are displayed as heatmaps
- **Precision-recall curves** are rendered as line charts with AUC in the legend
- **Line plots** (ROC curves, KS plots, etc.) are rendered as line charts with configurable axes
- **Scalar results** are shown as labeled values
- **Tables** are rendered as formatted data tables

When comparing multiple experiments in the Logfire Evals view, analyses of the same type are
displayed side by side for easy comparison.

## Complete Example

A full example combining case-level evaluators with report evaluators:

```python
from dataclasses import dataclass
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    ConfusionMatrixEvaluator,
    Evaluator,
    EvaluatorContext,
    KolmogorovSmirnovEvaluator,
    PrecisionRecallEvaluator,
    ReportEvaluator,
    ReportEvaluatorContext,
    ROCAUCEvaluator,
)
from pydantic_evals.reporting.analyses import ScalarResult


def my_classifier(text: str) -> str:
    text = text.lower()
    if 'cat' in text or 'meow' in text:
        return 'cat'
    elif 'dog' in text or 'bark' in text:
        return 'dog'
    elif 'bird' in text or 'chirp' in text:
        return 'bird'
    return 'unknown'


# Case-level evaluator: runs per case
@dataclass
class ConfidenceEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, Any]:
        confidence = compute_confidence(ctx.output, ctx.inputs)
        is_correct = ctx.output == ctx.expected_output
        return {
            'confidence': confidence,
            'is_correct': is_correct,
        }


def compute_confidence(output: str, inputs: str) -> float:
    return 0.85  # placeholder


# Report-level evaluator: runs once over the full report
@dataclass
class AccuracyEvaluator(ReportEvaluator):
    def evaluate(self, ctx: ReportEvaluatorContext) -> ScalarResult:
        cases = ctx.report.cases
        correct = sum(1 for c in cases if c.output == c.expected_output)
        return ScalarResult(
            title='Accuracy',
            value=correct / len(cases) * 100 if cases else 0.0,
            unit='%',
        )


dataset = Dataset(
    cases=[
        Case(inputs='The cat meows', expected_output='cat'),
        Case(inputs='The dog barks', expected_output='dog'),
        Case(inputs='A bird chirps', expected_output='bird'),
    ],
    evaluators=[ConfidenceEvaluator()],
    report_evaluators=[
        ConfusionMatrixEvaluator(
            predicted_from='output',
            expected_from='expected_output',
            title='Animal Classification',
        ),
        PrecisionRecallEvaluator(
            score_from='scores',
            score_key='confidence',
            positive_from='assertions',
            positive_key='is_correct',
        ),
        ROCAUCEvaluator(
            score_from='scores',
            score_key='confidence',
            positive_from='assertions',
            positive_key='is_correct',
        ),
        KolmogorovSmirnovEvaluator(
            score_from='scores',
            score_key='confidence',
            positive_from='assertions',
            positive_key='is_correct',
        ),
        AccuracyEvaluator(),
    ],
)

report = dataset.evaluate_sync(my_classifier)

# Access analyses programmatically
for analysis in report.analyses:
    print(f'{analysis.type}: {analysis.title}')
    #> confusion_matrix: Animal Classification
    #> precision_recall: Precision-Recall Curve
    #> scalar: Precision-Recall Curve AUC
    #> line_plot: ROC Curve
    #> scalar: ROC Curve AUC
    #> line_plot: KS Plot
    #> scalar: KS Statistic
    #> scalar: Accuracy
```

## Next Steps

- **[Built-in Evaluators](built-in.md)** — Case-level evaluator reference
- **[Custom Evaluators](custom.md)** — Writing case-level evaluators
- **[Logfire Integration](../how-to/logfire-integration.md)** — Viewing analyses in the Logfire UI
