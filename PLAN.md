# PLAN: Batteries-Included Agentic Evaluators for pydantic-evals

## Problem

Users onboarding with Logfire frequently ask how to evaluate agentic behaviors: tool call selection, task completion, trajectory quality. The existing `HasMatchingSpan` evaluator can accomplish this, but requires users to manually compose `SpanQuery` dicts, understand span naming conventions, and build evaluation logic from low-level primitives.

Competitors (DeepEval, Google ADK, LangChain `agentevals`, Arize) offer more plug-and-play agentic evaluation metrics. Pydantic AI has the data (rich OTEL traces) but lacks the convenient evaluation layer that turns that data into actionable quality signals.

## Proposal

Add 4 deterministic `Evaluator` subclasses to `pydantic-evals` that work with existing OTEL spans emitted by Pydantic AI agents. No changes required on the agent/instrumentation side.

### Evaluators

All evaluators live in a new file `pydantic_evals/pydantic_evals/evaluators/agentic.py` and are re-exported from `pydantic_evals.evaluators`.

#### 1. `ToolCorrectness`

Checks whether the agent called the expected tools (multiset membership).

```python
from pydantic_evals.evaluators import ToolCorrectness

ToolCorrectness(
    expected_tools=['search_database', 'format_response'],
    allow_extra=True,    # whether extra tool calls are OK
)
```

**Returns:** `EvaluationReason(value=bool, reason=...)` with diagnostic info on missing/unexpected tools.

**How it works:** Extracts tool call names from `SpanTree` via shared helper. Uses `Counter` (multiset) comparison, not set, so `['search', 'search']` correctly requires two calls. If `allow_extra=False`, also checks that no unexpected tools were called.

#### 2. `TrajectoryMatch`

Scores how well the agent's execution sequence matches an expected trajectory.

```python
from pydantic_evals.evaluators import TrajectoryMatch

TrajectoryMatch(
    expected_trajectory=['retrieve_context', 'rerank', 'generate_response'],
    order='in_order',  # 'exact' | 'in_order' | 'any_order'
)
```

**Returns:** `EvaluationReason(value=float, reason=...)` with 0.0-1.0 score and explanation.

**How it works:** Extracts ordered tool call names. Scoring by mode:
- `'exact'`: 1.0 if sequences are identical, 0.0 otherwise
- `'in_order'`: F1 score combining precision (LCS/actual) and recall (LCS/expected). This penalizes both missing steps AND extra wrong steps.
- `'any_order'`: Multiset intersection size / expected set size

#### 3. `ArgumentCorrectness`

Checks whether a specific tool call received correct arguments.

```python
from pydantic_evals.evaluators import ArgumentCorrectness

ArgumentCorrectness(
    tool_name='search_database',
    expected_arguments={'query': 'user question', 'limit': 10},
    match_mode='subset',  # 'exact' | 'subset'
    occurrence='first',   # 'first' | 'last' | int
)
```

**Returns:** `EvaluationReason(value=bool, reason=...)` with details on which arguments differed.

**How it works:** Finds the tool span for the named tool, parses the JSON `arguments` attribute, compares against `expected_arguments`. `'exact'` requires deep equality. `'subset'` checks all keys in `expected_arguments` are present with matching values. When `include_content=False`, arguments are unavailable and the evaluator returns `EvaluationReason(value=False, reason='Tool arguments not available in span (include_content may be disabled)')`.

#### 4. `StepEfficiency`

Checks whether the agent completed within a reasonable step budget.

```python
from pydantic_evals.evaluators import StepEfficiency

StepEfficiency(
    max_tool_calls=5,
    max_model_requests=3,
)
```

**Returns:** `dict[str, EvaluationReason]` with one entry per specified threshold. Uses `ctx.metrics['requests']` for model request count (already extracted by the eval runner) and span counting for tool calls.

### Shared Private Helper

```python
@dataclass
class ToolCallInfo:
    name: str
    arguments: str | None  # JSON string, None when include_content=False
    result: str | None     # JSON string, None when include_content=False
    duration: timedelta

def _extract_tool_calls(span_tree: SpanTree) -> list[ToolCallInfo]:
    """Extract ordered tool call info from span tree.

    Detects tool spans by name pattern + gen_ai.tool.name attribute:
    - v2: span name == "running tool"
    - v3+: span name starts with "execute_tool "

    Discriminates output-function spans from real tool calls:
    - v2: separate span name "running output function"
    - v3+: logfire.msg attribute starts with "running output function:"

    Returns tool calls sorted by start_timestamp.
    """
```

Span name constants are duplicated locally (4-6 strings) rather than importing from `pydantic_ai._instrumentation`, since that's a private module.

### Error Handling

- **No OTEL configured:** Evaluators catch `SpanTreeRecordingError` and return `EvaluationReason(value=False, reason='No tool call spans found in trace. Ensure logfire is configured.')`
- **Zero tool spans:** Return appropriate failure reason, not raise
- **JSON parse failure (ArgumentCorrectness):** Catch and return reason string
- **Missing ctx.metrics keys (StepEfficiency):** Fall back to span counting

### File Changes

```
pydantic_evals/pydantic_evals/evaluators/
    __init__.py          # Add new exports + __all__ entries
    agentic.py           # NEW — all agentic evaluators + shared helper
    common.py            # Add to DEFAULT_EVALUATORS tuple
```

```
docs/evals/evaluators/
    agentic.md           # NEW — cookbook with usage recipes
    built-in.md          # Update evaluator table
    overview.md          # Update evaluator list
```

### Documentation

New page `docs/evals/evaluators/agentic.md` following the structure of `span-based.md`, with 4 recipes:
1. RAG agent eval (search + rerank + generate)
2. Multi-tool agent eval (multiple tools, order matters)
3. Customer support agent eval (efficiency + arg correctness)
4. Task completion with LLMJudge (configuring LLMJudge with trajectory context from `_extract_tool_calls`)

Also update `built-in.md` and `overview.md` to include the new evaluators in their tables.

**Important caveat to document:** These evaluators cover locally-executed tools only. Provider-native built-in tools (executed server-side) do not produce local spans and won't appear in evaluations.

## Design Decisions

1. **Separate file (`agentic.py`)** rather than extending `common.py`: agentic evaluators share a private helper that doesn't belong in `common.py`, and it's easier to review as a self-contained addition.

2. **Private helper, not public `AgentTrace`:** Instrumentation versions (v2/v3+) are still evolving. A public trace API would freeze an unstable interface. The private helper can be refactored freely.

3. **No `TaskCompletion` evaluator:** Instead, a cookbook recipe shows how to configure `LLMJudge` with trajectory context. Less API surface, same user value.

4. **F1 scoring for `TrajectoryMatch`** (not recall-only): LCS/expected alone doesn't penalize extra wrong steps. F1 combining precision (LCS/actual) and recall (LCS/expected) properly penalizes both missing and extraneous steps.

5. **`EvaluationReason` returns everywhere** (not bare bool/float): Trace-based evaluators need to explain mismatches. "Missing tools: search_db" is much more useful than `False`.

6. **Multiset (Counter) comparisons** for `ToolCorrectness`: `['search', 'search']` should require two calls, not just one.

7. **Detect by span name + `gen_ai.tool.name` attribute**, not by tool argument attributes: Tool argument/result attributes are omitted when `include_content=False`, but `gen_ai.tool.name` and span names are always present.

## What Already Exists

| Existing code | How this proposal uses it |
|---|---|
| `HasMatchingSpan` (`common.py:269`) | Pattern template for `ctx.span_tree` access |
| `SpanTree.find()`, `SpanNode` attributes | Used by `_extract_tool_calls()` for traversal |
| `EvaluationReason` (`evaluator.py:33`) | Return type for all new evaluators |
| `LLMJudge` (`common.py:198`) | Reused in cookbook recipe for task completion |
| `DEFAULT_EVALUATORS` (`common.py:283`) | Extended with new evaluators |
| `_extract_span_tree_metrics()` (`dataset.py:986`) | StepEfficiency uses `ctx.metrics['requests']` |
| `pydantic_ai` imports in `common.py` | Cross-package dependency already established |

## Not in Scope

- `AgentTrace` public class (defer until instrumentation stabilizes)
- Trace-to-contract generator (depends on stable trace abstraction)
- `ArgumentCorrectness` semantic mode (LLM-as-judge for arg comparison)
- `StepEfficiency` max_retries (no reliable retry span signal)
- Provider-native built-in tools evaluation
- Logfire dashboard attributes for eval result rendering

## Usage Example

```python
import logfire
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    ToolCorrectness,
    TrajectoryMatch,
    StepEfficiency,
)

logfire.configure(send_to_logfire='if-token-present')

dataset = Dataset(
    name='rag_agent_evals',
    cases=[
        Case(
            name='knowledge_base_query',
            inputs='What is the return policy?',
            expected_output='The return policy allows returns within 30 days...',
        ),
    ],
    evaluators=[
        ToolCorrectness(
            expected_tools=['search_kb', 'generate_response'],
        ),
        TrajectoryMatch(
            expected_trajectory=['search_kb', 'rerank', 'generate_response'],
            order='in_order',
        ),
        StepEfficiency(max_tool_calls=5, max_model_requests=3),
    ],
)
```

## Open Questions for Maintainers

1. **File location:** Proposed `agentic.py` in evaluators directory. Would you prefer a different module name or extending `common.py`?
2. **Scope of first PR:** All 4 evaluators share the same helper and test infrastructure. Prefer one PR or split?
3. **Instrumentation version override:** Should evaluators accept an explicit `instrumentation_version` parameter for edge cases, or is auto-detection sufficient?
