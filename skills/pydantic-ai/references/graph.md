# Graph Reference

Source: `pydantic_graph/pydantic_graph/`, `pydantic_ai_slim/pydantic_ai/_agent_graph.py`

## Overview

PydanticAI uses a graph-based execution engine for agent runs. The graph engine is also available
as a standalone library (`pydantic_graph`) for building custom multi-step workflows.

Install: `pip/uv-add pydantic-graph`

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

### BaseNode — Generic Parameters

Nodes are generic in three type parameters:

```python
class MyNode(BaseNode[StateT, DepsT, RunEndT]):
    ...
```

| Parameter | Type Variable | Default | Description |
|-----------|---------------|---------|-------------|
| `StateT` | State type | `None` | Shared state across nodes |
| `DepsT` | Dependencies type | `None` | Dependency injection |
| `RunEndT` | End result type | `Never` | Type returned by `End()` |

### Node Structure

Nodes are typically dataclasses with:

1. **Fields** — parameters required when instantiating the node
2. **`run()` method** — business logic returning next node or `End`
3. **Return type annotation** — determines outgoing edges

```python
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, GraphRunContext


@dataclass
class ProcessItem(BaseNode[MyState]):
    item_id: int  # Node parameter

    async def run(self, ctx: GraphRunContext[MyState]) -> ValidateItem | End[str]:
        # Business logic
        item = await fetch_item(self.item_id)
        if item.valid:
            return ValidateItem(item=item)
        return End(f'Item {self.item_id} invalid')
```

### Intermediate Node (No End)

A node that always transitions to another node:

```python
@dataclass
class FetchData(BaseNode[MyState]):  # No RunEndT = Never (can't end)
    query: str

    async def run(self, ctx: GraphRunContext[MyState]) -> ProcessData:
        ctx.state.data = await fetch(self.query)
        return ProcessData()
```

### End Node (Can Terminate)

A node that can end the graph run:

```python
@dataclass
class FinalStep(BaseNode[MyState, None, str]):  # RunEndT = str
    async def run(self, ctx: GraphRunContext[MyState]) -> End[str]:
        return End(f'Completed with {ctx.state.count} items')
```

### Branching Node

A node with multiple possible next states:

```python
@dataclass
class Router(BaseNode[MyState, None, str]):
    async def run(
        self, ctx: GraphRunContext[MyState]
    ) -> PathA | PathB | End[str]:
        if ctx.state.should_retry:
            return PathA()
        elif ctx.state.is_done:
            return End('success')
        else:
            return PathB()
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

## Graph with Dependencies

Pass dependencies to nodes:

```python
from dataclasses import dataclass

from pydantic_graph import BaseNode, Graph, GraphRunContext


@dataclass
class MyDeps:
    api_client: ApiClient


@dataclass
class FetchNode(BaseNode[MyState, MyDeps]):  # StateT, DepsT
    async def run(self, ctx: GraphRunContext[MyState, MyDeps]) -> ProcessNode:
        data = await ctx.deps.api_client.fetch()
        ctx.state.data = data
        return ProcessNode()


graph = Graph(nodes=[FetchNode, ProcessNode])

# Run with deps
result = await graph.run(FetchNode(), state=MyState(), deps=MyDeps(api_client=client))
```

## Graph Generic Parameters

```python
graph = Graph[StateT, DepsT, RunEndT](nodes=[...])
```

| Parameter | Description |
|-----------|-------------|
| `StateT` | State type shared across nodes |
| `DepsT` | Dependencies passed to nodes |
| `RunEndT` | Return type of the graph run |

## Handling Agent Graph Nodes with iter()

When using `agent.iter()`, handle different node types:

```python
from pydantic_ai import Agent, CallToolsNode, ModelRequestNode, UserPromptNode

agent = Agent('openai:gpt-5')


def is_user_prompt_node(node) -> bool:
    return isinstance(node, UserPromptNode)


def is_model_request_node(node) -> bool:
    return isinstance(node, ModelRequestNode)


def is_call_tools_node(node) -> bool:
    return isinstance(node, CallToolsNode)


async with agent.iter('prompt') as agent_run:
    async for node in agent_run:
        if is_user_prompt_node(node):
            print('Processing user prompt...')
        elif is_model_request_node(node):
            print('Waiting for model response...')
        elif is_call_tools_node(node):
            print(f'Executing tools: {[tc.tool_name for tc in node.tool_calls]}')
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `Graph` | `pydantic_graph.Graph` | Graph definition |
| `BaseNode` | `pydantic_graph.BaseNode` | Abstract node |
| `End` | `pydantic_graph.End` | End signal with result |
| `GraphRunContext` | `pydantic_graph.GraphRunContext` | Node execution context |
| `StateT` | `pydantic_graph.nodes.StateT` | State type variable |
| `DepsT` | `pydantic_graph.nodes.DepsT` | Dependencies type variable |
| `RunEndT` | `pydantic_graph.nodes.RunEndT` | End result type variable |
| `UserPromptNode` | `pydantic_ai.UserPromptNode` | Agent graph: user prompt |
| `ModelRequestNode` | `pydantic_ai.ModelRequestNode` | Agent graph: model request |
| `CallToolsNode` | `pydantic_ai.CallToolsNode` | Agent graph: tool execution |
