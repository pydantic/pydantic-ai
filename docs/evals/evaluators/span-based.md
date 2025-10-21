# Span-Based Evaluation

Evaluate AI system behavior by analyzing OpenTelemetry spans captured during execution.

!!! note "Requires Logfire"
    Span-based evaluation requires `logfire` to be installed and configured:
    ```bash
    pip install 'pydantic-evals[logfire]'
    ```

## Overview

When you configure logfire (`logfire.configure()`), Pydantic Evals captures all OpenTelemetry spans generated during task execution. You can then evaluate based on:

- **Which tools were called** - Verify expected behavior
- **Code paths executed** - Ensure specific functions ran
- **Timing characteristics** - Check individual operation latency
- **Error conditions** - Detect specific failures

This is powerful for evaluating **how** your AI system works, not just **what** it produces.

## Basic Usage

```python
import logfire

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import HasMatchingSpan

# Configure logfire to capture spans
logfire.configure(send_to_logfire='if-token-present')

dataset = Dataset(
    cases=[Case(inputs='test')],
    evaluators=[
        # Check that database was queried
        HasMatchingSpan(
            query={'name_contains': 'database_query'},
            evaluation_name='used_database',
        ),
    ],
)
```

## HasMatchingSpan Evaluator

The [`HasMatchingSpan`][pydantic_evals.evaluators.HasMatchingSpan] evaluator checks if any span matches a query:

```python
from pydantic_evals.evaluators import HasMatchingSpan

HasMatchingSpan(
    query={'name_contains': 'test'},
    evaluation_name='span_check',
)
```

**Returns:** `bool` - `True` if any span matches the query

## SpanQuery Reference

A [`SpanQuery`][pydantic_evals.otel.SpanQuery] is a dictionary with query conditions:

### Name Conditions

Match spans by name:

```python
# Exact name match
{'name_equals': 'search_database'}

# Contains substring
{'name_contains': 'tool_call'}

# Regex pattern
{'name_matches_regex': r'llm_call_\d+'}
```

### Attribute Conditions

Match spans with specific attributes:

```python
# Has specific attribute values
{'has_attributes': {'operation': 'search', 'status': 'success'}}

# Has attribute keys (any value)
{'has_attribute_keys': ['user_id', 'request_id']}
```

### Duration Conditions

Match based on execution time:

```python
from datetime import timedelta

# Minimum duration
{'min_duration': 1.0}  # seconds
{'min_duration': timedelta(seconds=1)}

# Maximum duration
{'max_duration': 5.0}  # seconds
{'max_duration': timedelta(seconds=5)}

# Range
{'min_duration': 0.5, 'max_duration': 2.0}
```

### Logical Operators

Combine conditions:

```python
# NOT
{'not_': {'name_contains': 'error'}}

# AND (all must match)
{'and_': [
    {'name_contains': 'tool'},
    {'max_duration': 1.0},
]}

# OR (any must match)
{'or_': [
    {'name_equals': 'search'},
    {'name_equals': 'query'},
]}
```

### Child/Descendant Conditions

Query relationships between spans:

```python
# Count direct children
{'min_child_count': 1}
{'max_child_count': 5}

# Some child matches query
{'some_child_has': {'name_contains': 'retry'}}

# All children match query
{'all_children_have': {'max_duration': 0.5}}

# No children match query
{'no_child_has': {'has_attributes': {'error': True}}}

# Descendant queries (recursive)
{'min_descendant_count': 5}
{'some_descendant_has': {'name_contains': 'api_call'}}
```

### Ancestor/Depth Conditions

Query span hierarchy:

```python
# Depth (root spans have depth 0)
{'min_depth': 1}  # Not a root span
{'max_depth': 2}  # At most 2 levels deep

# Ancestor queries
{'some_ancestor_has': {'name_equals': 'agent_run'}}
{'all_ancestors_have': {'max_duration': 10.0}}
{'no_ancestor_has': {'has_attributes': {'error': True}}}
```

### Stop Recursing

Control recursive queries:

```python
{
    'some_descendant_has': {'name_contains': 'expensive'},
    'stop_recursing_when': {'name_equals': 'boundary'},
}
# Only search descendants until hitting a span named 'boundary'
```

## Practical Examples

### Verify Tool Usage

Check that specific tools were called:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import HasMatchingSpan

dataset = Dataset(
    cases=[Case(inputs='test')],
    evaluators=[
        # Must call search tool
        HasMatchingSpan(
            query={'name_contains': 'search_tool'},
            evaluation_name='used_search',
        ),

        # Must NOT call dangerous tool
        HasMatchingSpan(
            query={'not_': {'name_contains': 'delete_database'}},
            evaluation_name='safe_execution',
        ),
    ],
)
```

### Check Multiple Tools

Verify a sequence of operations:

```python
from pydantic_evals.evaluators import HasMatchingSpan

evaluators = [
    HasMatchingSpan(
        query={'name_contains': 'retrieve_context'},
        evaluation_name='retrieved_context',
    ),
    HasMatchingSpan(
        query={'name_contains': 'generate_response'},
        evaluation_name='generated_response',
    ),
    HasMatchingSpan(
        query={'and_': [
            {'name_contains': 'cite'},
            {'has_attribute_keys': ['source_id']},
        ]},
        evaluation_name='added_citations',
    ),
]
```

### Performance Assertions

Ensure operations meet latency requirements:

```python
from pydantic_evals.evaluators import HasMatchingSpan

evaluators = [
    # Database queries should be fast
    HasMatchingSpan(
        query={'and_': [
            {'name_contains': 'database'},
            {'max_duration': 0.1},  # 100ms max
        ]},
        evaluation_name='fast_db_queries',
    ),

    # Overall should complete quickly
    HasMatchingSpan(
        query={'and_': [
            {'name_equals': 'task_execution'},
            {'max_duration': 2.0},
        ]},
        evaluation_name='within_sla',
    ),
]
```

### Error Detection

Check for error conditions:

```python
from pydantic_evals.evaluators import HasMatchingSpan

evaluators = [
    # No errors occurred
    HasMatchingSpan(
        query={'not_': {'has_attributes': {'error': True}}},
        evaluation_name='no_errors',
    ),

    # Retries happened
    HasMatchingSpan(
        query={'name_contains': 'retry'},
        evaluation_name='had_retries',
    ),

    # Fallback was used
    HasMatchingSpan(
        query={'name_contains': 'fallback_model'},
        evaluation_name='used_fallback',
    ),
]
```

### Complex Behavioral Checks

Verify sophisticated behavior patterns:

```python
from pydantic_evals.evaluators import HasMatchingSpan

evaluators = [
    # Agent delegated to sub-agent
    HasMatchingSpan(
        query={'and_': [
            {'name_contains': 'agent'},
            {'some_child_has': {'name_contains': 'delegate'}},
        ]},
        evaluation_name='used_delegation',
    ),

    # Made multiple LLM calls with retries
    HasMatchingSpan(
        query={'and_': [
            {'name_contains': 'llm_call'},
            {'some_descendant_has': {'name_contains': 'retry'}},
            {'min_descendant_count': 3},
        ]},
        evaluation_name='retry_pattern',
    ),
]
```

## Custom Evaluators with SpanTree

For more complex span analysis, write custom evaluators:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class CustomSpanCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | int]:
        span_tree = ctx.span_tree

        # Find specific spans
        llm_spans = span_tree.find(lambda node: 'llm' in node.name)
        tool_spans = span_tree.find(lambda node: 'tool' in node.name)

        # Calculate metrics
        total_llm_time = sum(
            span.duration.total_seconds() for span in llm_spans
        )

        return {
            'used_llm': len(llm_spans) > 0,
            'used_tools': len(tool_spans) > 0,
            'tool_count': len(tool_spans),
            'llm_fast': total_llm_time < 2.0,
        }
```

### SpanTree API

The [`SpanTree`][pydantic_evals.otel.SpanTree] provides methods for span analysis:

```python
from pydantic_evals.otel import SpanTree


# Example API (requires span_tree from context)
def example_api(span_tree: SpanTree) -> None:
    span_tree.find(lambda n: True)  # Find all matching nodes
    span_tree.any({'name_contains': 'test'})  # Check if any span matches
    span_tree.all({'name_contains': 'test'})  # Check if all spans match
    span_tree.count({'name_contains': 'test'})  # Count matching spans

    # Iteration
    for node in span_tree:
        print(node.name, node.duration, node.attributes)
```

### SpanNode Properties

Each [`SpanNode`][pydantic_evals.otel.SpanNode] has:

```python
from pydantic_evals.otel import SpanNode


# Example properties (requires node from context)
def example_properties(node: SpanNode) -> None:
    _ = node.name  # Span name
    _ = node.duration  # timedelta
    _ = node.attributes  # dict[str, AttributeValue]
    _ = node.start_timestamp  # datetime
    _ = node.end_timestamp  # datetime
    _ = node.children  # list[SpanNode]
    _ = node.descendants  # list[SpanNode] (recursive)
    _ = node.ancestors  # list[SpanNode]
    _ = node.parent  # SpanNode | None
```

## Debugging Span Queries

### View Spans in Logfire

If you're sending data to Logfire, you can view all spans in the web UI to understand the trace structure.

### Print Span Tree

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class DebugSpans(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        for node in ctx.span_tree:
            print(f"{'  ' * len(node.ancestors)}{node.name} ({node.duration})")
        return True
```

### Query Testing

Test queries incrementally:

```python
from pydantic_evals.evaluators import HasMatchingSpan

# Start simple
query = {'name_contains': 'tool'}

# Add conditions gradually
query = {'and_': [
    {'name_contains': 'tool'},
    {'max_duration': 1.0},
]}

# Test in evaluator
HasMatchingSpan(query=query, evaluation_name='test')
```

## Use Cases

### RAG System Verification

Verify retrieval-augmented generation workflow:

```python
from pydantic_evals.evaluators import HasMatchingSpan

evaluators = [
    # Retrieved documents
    HasMatchingSpan(
        query={'name_contains': 'vector_search'},
        evaluation_name='retrieved_docs',
    ),

    # Reranked results
    HasMatchingSpan(
        query={'name_contains': 'rerank'},
        evaluation_name='reranked_results',
    ),

    # Generated with context
    HasMatchingSpan(
        query={'and_': [
            {'name_contains': 'generate'},
            {'has_attribute_keys': ['context_ids']},
        ]},
        evaluation_name='used_context',
    ),
]
```

### Multi-Agent Systems

Verify agent coordination:

```python
from pydantic_evals.evaluators import HasMatchingSpan

evaluators = [
    # Master agent ran
    HasMatchingSpan(
        query={'name_equals': 'master_agent'},
        evaluation_name='master_ran',
    ),

    # Delegated to specialist
    HasMatchingSpan(
        query={'and_': [
            {'name_contains': 'specialist_agent'},
            {'some_ancestor_has': {'name_equals': 'master_agent'}},
        ]},
        evaluation_name='delegated_correctly',
    ),

    # No circular delegation
    HasMatchingSpan(
        query={'not_': {'and_': [
            {'name_contains': 'agent'},
            {'some_descendant_has': {'name_contains': 'agent'}},
            {'some_ancestor_has': {'name_contains': 'agent'}},
        ]}},
        evaluation_name='no_circular_delegation',
    ),
]
```

### Tool Usage Patterns

Verify intelligent tool selection:

```python
from pydantic_evals.evaluators import HasMatchingSpan

evaluators = [
    # Used search before answering
    HasMatchingSpan(
        query={'and_': [
            {'name_contains': 'search'},
            {'some_ancestor_has': {'name_contains': 'answer'}},
        ]},
        evaluation_name='searched_before_answering',
    ),

    # Limited tool calls (no loops)
    HasMatchingSpan(
        query={'and_': [
            {'name_contains': 'tool'},
            {'max_child_count': 5},
        ]},
        evaluation_name='reasonable_tool_usage',
    ),
]
```

## Best Practices

1. **Start Simple**: Begin with basic name queries, add complexity as needed
2. **Use Descriptive Names**: Name your spans well in your application code
3. **Test Queries**: Verify queries work before running full evaluations
4. **Combine with Other Evaluators**: Use span checks alongside output validation
5. **Document Expectations**: Comment why specific spans should/shouldn't exist

## Next Steps

- **[Logfire Integration](../how-to/logfire-integration.md)** - Set up Logfire for span capture
- **[Custom Evaluators](custom.md)** - Write advanced span analysis
- **[Examples: Agent Behavior](../examples/agent-behavior.md)** - Multi-agent span evaluation
