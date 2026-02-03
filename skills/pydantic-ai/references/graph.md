# Graph Reference

Source: `pydantic_graph/pydantic_graph/`, `pydantic_ai_slim/pydantic_ai/_agent_graph.py`

## Overview

PydanticAI uses a graph-based execution engine for agent runs. The graph engine is also available
as a standalone library (`pydantic_graph`) for building custom multi-step workflows.

## Core Concepts

### Graph

A directed graph of nodes that defines a workflow:

```python
from pydantic_graph import Graph, BaseNode, End, GraphRunContext

class MyState:
    count: int = 0

class StepOne(BaseNode[MyState]):
    async def run(self, ctx: GraphRunContext[MyState]) -> StepTwo | End[str]:
        ctx.state.count += 1
        if ctx.state.count >= 3:
            return End(f'Done after {ctx.state.count} steps')
        return StepTwo()

class StepTwo(BaseNode[MyState]):
    async def run(self, ctx: GraphRunContext[MyState]) -> StepOne:
        ctx.state.count += 1
        return StepOne()

graph = Graph(nodes=[StepOne, StepTwo])
```

### BaseNode

Abstract base class for graph nodes. Each node's `run()` method returns the next node or `End`.

```python
class MyNode(BaseNode[StateType, DepsType, EndType]):
    async def run(
        self, ctx: GraphRunContext[StateType, DepsType]
    ) -> OtherNode | End[EndType]:
        ...
```

### End

Signals the graph run is complete, carrying a result value:

```python
from pydantic_graph import End

return End('final result')
```

### GraphRunContext

Context available during node execution:

```python
ctx.state    # The shared state object
ctx.deps     # Dependencies (if any)
```

## Agent Graph Nodes

PydanticAI agents internally use these graph nodes:

```
UserPromptNode → ModelRequestNode → CallToolsNode → (loop or End)
```

| Node | Description |
|------|-------------|
| `UserPromptNode` | Processes the user prompt and prepares the request |
| `ModelRequestNode` | Sends the request to the model |
| `CallToolsNode` | Executes any tool calls from the model response |

These are accessible via:

```python
from pydantic_ai import UserPromptNode, ModelRequestNode, CallToolsNode
```

## Running a Graph

```python
result = await graph.run(StepOne(), state=MyState())
print(result.output)  # The End value
```

## State Management

State is shared across all nodes via `GraphRunContext.state`. Use a dataclass or any mutable object:

```python
from dataclasses import dataclass, field

@dataclass
class WorkflowState:
    messages: list[str] = field(default_factory=list)
    step_count: int = 0
```

## Mermaid Diagrams

Generate visual diagrams of graph structure:

```python
mermaid_code = graph.mermaid_code()
print(mermaid_code)
```

## When to Use Graphs

- **Simple agent**: Use `Agent.run()` directly — no graph needed.
- **Multi-agent pipeline**: Use `iter()` to coordinate between agents.
- **Complex workflows**: Use `pydantic_graph.Graph` with custom nodes for branching, looping, and state management.

## Tracing Graph Execution

With Logfire instrumentation, graph execution is traced with spans for each node, showing:

- Node transitions and decision points
- State changes through the workflow
- Time spent in each node

This is essential for debugging complex multi-step workflows where it's unclear which path was taken.

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `Graph` | `pydantic_graph.Graph` | Graph definition |
| `BaseNode` | `pydantic_graph.BaseNode` | Abstract node |
| `End` | `pydantic_graph.End` | End signal with result |
| `GraphRunContext` | `pydantic_graph.GraphRunContext` | Node execution context |
| `UserPromptNode` | `pydantic_ai.UserPromptNode` | Agent graph: user prompt |
| `ModelRequestNode` | `pydantic_ai.ModelRequestNode` | Agent graph: model request |
| `CallToolsNode` | `pydantic_ai.CallToolsNode` | Agent graph: tool execution |
