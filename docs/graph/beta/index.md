# Beta Graph API

!!! warning "Beta API"
    This is the new beta graph API. It provides enhanced capabilities for parallel execution, conditional branching, and complex workflows.
The original graph API is still available (and compatible of interop with the new beta API) and is documented in the [main graph documentation](../../graph.md).

## Overview

The beta graph API in `pydantic-graph` provides a powerful builder pattern for constructing parallel execution graphs with:

- **Step nodes** for executing async functions
- **Decision nodes** for conditional branching
- **Spread operations** for parallel processing of iterables
- **Broadcast operations** for sending the same data to multiple parallel paths
- **Join nodes and Reducers** for aggregating results from parallel execution

This API is designed for advanced workflows where you want declarative control over parallelism, routing, and data aggregation.

## Installation

The beta graph API is included with `pydantic-graph`:

```bash
pip install pydantic-graph
```

Or as part of `pydantic-ai`:

```bash
pip install pydantic-ai
```

## Quick Start

Here's a simple example to get you started:

```python {title="simple_counter.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext


@dataclass
class CounterState:
    """State for tracking a counter value."""

    value: int = 0


async def main():
    # Create a graph builder with state and output types
    g = GraphBuilder(state_type=CounterState, output_type=int)

    # Define steps using the decorator
    @g.step
    async def increment(ctx: StepContext[CounterState, None, None]) -> int:
        """Increment the counter and return its value."""
        ctx.state.value += 1
        return ctx.state.value

    @g.step
    async def double_it(ctx: StepContext[CounterState, None, int]) -> int:
        """Double the input value."""
        return ctx.inputs * 2

    # Add edges connecting the nodes
    g.add(
        g.edge_from(g.start_node).to(increment),
        g.edge_from(increment).to(double_it),
        g.edge_from(double_it).to(g.end_node),
    )

    # Build and run the graph
    graph = g.build()
    state = CounterState()
    result = await graph.run(state=state)
    print(f'Result: {result}')
    #> Result: 2
    print(f'Final state: {state.value}')
    #> Final state: 1
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Key Concepts

### GraphBuilder

The [`GraphBuilder`][pydantic_graph.beta.graph_builder.GraphBuilder] is the main entry point for constructing graphs. It's generic over:

- `StateT` - The type of mutable state shared across all nodes
- `DepsT` - The type of dependencies injected into nodes
- `InputT` - The type of initial input to the graph
- `OutputT` - The type of final output from the graph

### Steps

Steps are async functions decorated with [`@g.step`][pydantic_graph.beta.graph_builder.GraphBuilder.step] that define the actual work to be done in each node. They receive a [`StepContext`][pydantic_graph.beta.step.StepContext] with access to:

- `ctx.state` - The mutable graph state
- `ctx.deps` - Injected dependencies
- `ctx.inputs` - Input data for this step

### Edges

Edges define the connections between nodes. The builder provides multiple ways to create edges:

- [`g.add()`][pydantic_graph.beta.graph_builder.GraphBuilder.add] - Add one or more edge paths
- [`g.add_edge()`][pydantic_graph.beta.graph_builder.GraphBuilder.add_edge] - Add a simple edge between two nodes
- [`g.edge_from()`][pydantic_graph.beta.graph_builder.GraphBuilder.edge_from] - Start building a complex edge path

### Start and End Nodes

Every graph has:

- [`g.start_node`][pydantic_graph.beta.graph_builder.GraphBuilder.start_node] - The entry point receiving initial inputs
- [`g.end_node`][pydantic_graph.beta.graph_builder.GraphBuilder.end_node] - The exit point producing final outputs

## A More Complex Example

Here's an example showcasing parallel execution with a map operation:

```python {title="parallel_processing.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append


@dataclass
class ProcessingState:
    """State for tracking processing metrics."""

    items_processed: int = 0


async def main():
    g = GraphBuilder(
        state_type=ProcessingState,
        input_type=list[int],
        output_type=list[int],
    )

    @g.step
    async def square(ctx: StepContext[ProcessingState, None, int]) -> int:
        """Square a number and track that we processed it."""
        ctx.state.items_processed += 1
        return ctx.inputs * ctx.inputs

    # Create a join to collect results
    collect_results = g.join(reduce_list_append, initial_factory=list[int])

    # Build the graph with map operation
    g.add(
        g.edge_from(g.start_node).map().to(square),
        g.edge_from(square).to(collect_results),
        g.edge_from(collect_results).to(g.end_node),
    )

    graph = g.build()
    state = ProcessingState()
    result = await graph.run(state=state, inputs=[1, 2, 3, 4, 5])

    print(f'Results: {sorted(result)}')
    #> Results: [1, 4, 9, 16, 25]
    print(f'Items processed: {state.items_processed}')
    #> Items processed: 5
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

In this example:

1. The start node receives a list of integers
2. The `.map()` operation fans out each item to a separate parallel execution of the `square` step
3. All results are collected back together using [`reduce_list_append`][pydantic_graph.beta.join.reduce_list_append]
4. The joined results flow to the end node

## Next Steps

Explore the detailed documentation for each feature:

- [**Steps**](steps.md) - Learn about step nodes and execution contexts
- [**Joins**](joins.md) - Understand join nodes and reducer patterns
- [**Decisions**](decisions.md) - Implement conditional branching
- [**Parallel Execution**](parallel.md) - Master broadcasting and mapping

## Advanced Execution Control

Beyond the basic [`graph.run()`][pydantic_graph.beta.graph.Graph.run] method, the beta API provides fine-grained control over graph execution.

### Step-by-Step Execution

Use [`graph.iter()`][pydantic_graph.beta.graph.Graph.iter] to execute the graph one step at a time:

```python {title="step_by_step.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext


@dataclass
class CounterState:
    value: int = 0


async def main():
    g = GraphBuilder(state_type=CounterState, output_type=int)

    @g.step
    async def increment(ctx: StepContext[CounterState, None, None]) -> int:
        ctx.state.value += 1
        return ctx.state.value

    @g.step
    async def double_it(ctx: StepContext[CounterState, None, int]) -> int:
        return ctx.inputs * 2

    g.add(
        g.edge_from(g.start_node).to(increment),
        g.edge_from(increment).to(double_it),
        g.edge_from(double_it).to(g.end_node),
    )

    graph = g.build()
    state = CounterState()

    # Use iter() for step-by-step execution
    async with graph.iter(state=state) as graph_run:
        print(f'Initial state: {state.value}')
        #> Initial state: 0

        # Advance execution step by step
        async for event in graph_run:
            print(f'{state.value=} | {event=}')
            #> state.value=0 | event=[GraphTask(node_id='increment', inputs=None)]
            #> state.value=1 | event=[GraphTask(node_id='double_it', inputs=1)]
            #> state.value=1 | event=[GraphTask(node_id='__end__', inputs=2)]
            #> state.value=1 | event=EndMarker(_value=2)
            if graph_run.output is not None:
                print(f'Final output: {graph_run.output}')
                #> Final output: 2
                break
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

The [`GraphRun`][pydantic_graph.beta.graph.GraphRun] object provides:

- **Async iteration**: Iterate through execution events
- **`next_task` property**: Inspect upcoming tasks
- **`output` property**: Check if the graph has completed and get the final output
- **`next()` method**: Manually advance execution with optional value injection

### Visualizing Graphs

Generate Mermaid diagrams of your graph structure using [`graph.render()`][pydantic_graph.beta.graph.Graph.render]:

```python {title="visualize_graph.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext


@dataclass
class SimpleState:
    pass


g = GraphBuilder(state_type=SimpleState, output_type=str)

@g.step
async def step_a(ctx: StepContext[SimpleState, None, None]) -> int:
    return 10

@g.step
async def step_b(ctx: StepContext[SimpleState, None, int]) -> str:
    return f'Result: {ctx.inputs}'

g.add(
    g.edge_from(g.start_node).to(step_a),
    g.edge_from(step_a).to(step_b),
    g.edge_from(step_b).to(g.end_node),
)

graph = g.build()

# Generate a Mermaid diagram
mermaid_diagram = graph.render(title='My Graph', direction='LR')
print(mermaid_diagram)
"""
---
title: My Graph
---
stateDiagram-v2
  direction LR
  step_a
  step_b

  [*] --> step_a
  step_a --> step_b
  step_b --> [*]
"""
```

The rendered diagram can be displayed in documentation, notebooks, or any tool that supports Mermaid syntax.

## Comparison with Original API

The original graph API (documented in the [main graph page](../../graph.md)) uses a class-based approach with [`BaseNode`][pydantic_graph.nodes.BaseNode] subclasses. The beta API uses a builder pattern with decorated functions, which provides:

**Advantages:**
- More concise syntax for simple workflows
- Explicit control over parallelism with map/broadcast
- Built-in reducers for common aggregation patterns
- Easier to visualize complex data flows

**Trade-offs:**
- Requires understanding of builder patterns
- Less object-oriented, more functional style

Both APIs are fully supported and can even be integrated together when needed.

## Persistence and Resumability

!!! info "No Native Persistence"
    Unlike the [original Graph API](../../graph.md#state-persistence), the beta graph API does not include built-in state persistence. This is due to the complexity of achieving consistent snapshotting with parallel execution.

For workflows that need to preserve progress across failures, restarts, or long-running operations, use one of the supported [durable execution](../../durable_execution/overview.md) solutions:

- [Temporal](../../durable_execution/temporal.md)
- [DBOS](../../durable_execution/dbos.md)
- [Prefect](../../durable_execution/prefect.md)

These integrations provide production-grade reliability with full support for streaming and MCP, handling the complexities of checkpointing parallel execution automatically.
