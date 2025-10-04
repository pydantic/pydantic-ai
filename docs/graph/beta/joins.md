# Joins and Reducers

Join nodes synchronize and aggregate data from parallel execution paths. They use **Reducers** to combine multiple inputs into a single output.

## Overview

When you use [parallel execution](parallel.md) (broadcasting or mapping), you often need to collect and combine the results. Join nodes serve this purpose by:

1. Waiting for all parallel tasks to complete
2. Aggregating their outputs using a [`Reducer`][pydantic_graph.beta.join.Reducer]
3. Passing the aggregated result to the next node

## Creating Joins

Create a join using [`g.join()`][pydantic_graph.beta.graph_builder.GraphBuilder.join] with a reducer type:

```python {title="basic_join.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, ListReducer, StepContext


@dataclass
class SimpleState:
    pass


g = GraphBuilder(state_type=SimpleState, output_type=list[int])

@g.step
async def generate_numbers(ctx: StepContext[SimpleState, None, None]) -> list[int]:
    return [1, 2, 3, 4, 5]

@g.step
async def square(ctx: StepContext[SimpleState, None, int]) -> int:
    return ctx.inputs * ctx.inputs

# Create a join to collect all squared values
collect = g.join(ListReducer[int])

g.add(
    g.edge_from(g.start_node).to(generate_numbers),
    g.edge_from(generate_numbers).map().to(square),
    g.edge_from(square).to(collect),
    g.edge_from(collect).to(g.end_node),
)

graph = g.build()

async def main():
    result = await graph.run(state=SimpleState())
    print(sorted(result))
    #> [1, 4, 9, 16, 25]
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Built-in Reducers

Pydantic Graph provides several common reducer types out of the box:

### ListReducer

[`ListReducer`][pydantic_graph.beta.join.ListReducer] collects all inputs into a list:

```python {title="list_reducer.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, ListReducer, StepContext


@dataclass
class SimpleState:
    pass


async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=list[str])

    @g.step
    async def generate(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return [10, 20, 30]

    @g.step
    async def to_string(ctx: StepContext[SimpleState, None, int]) -> str:
        return f'value-{ctx.inputs}'

    collect = g.join(ListReducer[str])

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(to_string),
        g.edge_from(to_string).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    print(sorted(result))
    #> ['value-10', 'value-20', 'value-30']
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

### DictReducer

[`DictReducer`][pydantic_graph.beta.join.DictReducer] merges dictionaries together:

```python {title="dict_reducer.py"}
from dataclasses import dataclass

from pydantic_graph.beta import DictReducer, GraphBuilder, StepContext


@dataclass
class SimpleState:
    pass


async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=dict[str, int])

    @g.step
    async def generate_keys(ctx: StepContext[SimpleState, None, None]) -> list[str]:
        return ['apple', 'banana', 'cherry']

    @g.step
    async def create_entry(ctx: StepContext[SimpleState, None, str]) -> dict[str, int]:
        return {ctx.inputs: len(ctx.inputs)}

    merge = g.join(DictReducer[str, int])

    g.add(
        g.edge_from(g.start_node).to(generate_keys),
        g.edge_from(generate_keys).map().to(create_entry),
        g.edge_from(create_entry).to(merge),
        g.edge_from(merge).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    result = {k: result[k] for k in sorted(result)}  # force deterministic ordering
    print(result)
    #> {'apple': 5, 'banana': 6, 'cherry': 6}
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

### NullReducer

[`NullReducer`][pydantic_graph.beta.join.NullReducer] discards all inputs and returns `None`. Useful when you only care about side effects:

```python {title="null_reducer.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, NullReducer, StepContext


@dataclass
class CounterState:
    total: int = 0


async def main():
    g = GraphBuilder(state_type=CounterState, output_type=int)

    @g.step
    async def generate(ctx: StepContext[CounterState, None, None]) -> list[int]:
        return [1, 2, 3, 4, 5]

    @g.step
    async def accumulate(ctx: StepContext[CounterState, None, int]) -> int:
        ctx.state.total += ctx.inputs
        return ctx.inputs

    # We don't care about the outputs, only the side effect on state
    ignore = g.join(NullReducer)

    @g.step
    async def get_total(ctx: StepContext[CounterState, None, None]) -> int:
        return ctx.state.total

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(accumulate),
        g.edge_from(accumulate).to(ignore),
        g.edge_from(ignore).to(get_total),
        g.edge_from(get_total).to(g.end_node),
    )

    graph = g.build()
    state = CounterState()
    result = await graph.run(state=state)
    print(result)
    #> 15
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Custom Reducers

Create custom reducers by subclassing [`Reducer`][pydantic_graph.beta.join.Reducer]:

```python {title="custom_reducer.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, Reducer, StepContext


@dataclass
class SimpleState:
    pass


@dataclass(init=False)
class SumReducer(Reducer[SimpleState, None, int, int]):
    """Reducer that sums all input values."""

    total: int = 0

    def reduce(self, ctx: StepContext[SimpleState, None, int]) -> None:
        """Called for each input - accumulate the sum."""
        self.total += ctx.inputs

    def finalize(self, ctx: StepContext[SimpleState, None, None]) -> int:
        """Called after all inputs - return the final result."""
        return self.total


async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step
    async def generate(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return [5, 10, 15, 20]

    @g.step
    async def identity(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs

    sum_join = g.join(SumReducer)

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(identity),
        g.edge_from(identity).to(sum_join),
        g.edge_from(sum_join).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    print(result)
    #> 50
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

### Reducer Lifecycle

Reducers have two key methods:

1. **`reduce(ctx)`** - Called for each input from parallel paths. Use this to accumulate data.
2. **`finalize(ctx)`** - Called once after all inputs are received. Return the final aggregated value.

## Reducers with State Access

Reducers can access and modify the graph state:

```python {title="stateful_reducer.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, Reducer, StepContext


@dataclass
class MetricsState:
    items_processed: int = 0
    sum_total: int = 0


@dataclass(init=False)
class MetricsReducer(Reducer[MetricsState, None, int, dict[str, int]]):
    """Reducer that tracks processing metrics in state."""

    count: int = 0
    total: int = 0

    def reduce(self, ctx: StepContext[MetricsState, None, int]) -> None:
        self.count += 1
        self.total += ctx.inputs
        ctx.state.items_processed += 1
        ctx.state.sum_total += ctx.inputs

    def finalize(self, ctx: StepContext[MetricsState, None, None]) -> dict[str, int]:
        return {
            'count': self.count,
            'total': self.total,
        }


async def main():
    g = GraphBuilder(state_type=MetricsState, output_type=dict[str, int])

    @g.step
    async def generate(ctx: StepContext[MetricsState, None, None]) -> list[int]:
        return [10, 20, 30, 40]

    @g.step
    async def process(ctx: StepContext[MetricsState, None, int]) -> int:
        return ctx.inputs * 2

    metrics = g.join(MetricsReducer)

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(process),
        g.edge_from(process).to(metrics),
        g.edge_from(metrics).to(g.end_node),
    )

    graph = g.build()
    state = MetricsState()
    result = await graph.run(state=state)

    print(f'Result: {result}')
    #> Result: {'count': 4, 'total': 200}
    print(f'State items_processed: {state.items_processed}')
    #> State items_processed: 4
    print(f'State sum_total: {state.sum_total}')
    #> State sum_total: 200
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Multiple Joins

A graph can have multiple independent joins:

```python {title="multiple_joins.py"}
from dataclasses import dataclass, field

from pydantic_graph.beta import GraphBuilder, ListReducer, StepContext


@dataclass
class MultiState:
    results: dict[str, list[int]] = field(default_factory=dict)


async def main():
    g = GraphBuilder(state_type=MultiState, output_type=dict[str, list[int]])

    @g.step
    async def source_a(ctx: StepContext[MultiState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def source_b(ctx: StepContext[MultiState, None, None]) -> list[int]:
        return [10, 20]

    @g.step
    async def process_a(ctx: StepContext[MultiState, None, int]) -> int:
        return ctx.inputs * 2

    @g.step
    async def process_b(ctx: StepContext[MultiState, None, int]) -> int:
        return ctx.inputs * 3

    join_a = g.join(ListReducer[int], node_id='join_a')
    join_b = g.join(ListReducer[int], node_id='join_b')

    @g.step
    async def store_a(ctx: StepContext[MultiState, None, list[int]]) -> None:
        ctx.state.results['a'] = ctx.inputs

    @g.step
    async def store_b(ctx: StepContext[MultiState, None, list[int]]) -> None:
        ctx.state.results['b'] = ctx.inputs

    @g.step
    async def combine(ctx: StepContext[MultiState, None, None]) -> dict[str, list[int]]:
        return ctx.state.results

    g.add(
        g.edge_from(g.start_node).to(source_a, source_b),
        g.edge_from(source_a).map().to(process_a),
        g.edge_from(source_b).map().to(process_b),
        g.edge_from(process_a).to(join_a),
        g.edge_from(process_b).to(join_b),
        g.edge_from(join_a).to(store_a),
        g.edge_from(join_b).to(store_b),
        g.edge_from(store_a, store_b).to(combine),
        g.edge_from(combine).to(g.end_node),
    )

    graph = g.build()
    state = MultiState()
    result = await graph.run(state=state)

    print(f"Group A: {sorted(result['a'])}")
    #> Group A: [2, 4, 6]
    print(f"Group B: {sorted(result['b'])}")
    #> Group B: [30, 60]
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Customizing Join Nodes

### Custom Node IDs

Like steps, joins can have custom IDs:

```python {title="join_custom_id.py" requires="basic_join.py"}
from basic_join import ListReducer, g

my_join = g.join(ListReducer[int], node_id='my_custom_join_id')
```

## How Joins Work

Internally, the graph tracks which "fork" each parallel task belongs to. A join:

1. Identifies its parent fork (the fork that created the parallel paths)
2. Waits for all tasks from that fork to reach the join
3. Calls `reduce()` for each incoming value
4. Calls `finalize()` once all values are received
5. Passes the finalized result to downstream nodes

This ensures proper synchronization even with nested parallel operations.

## Next Steps

- Learn about [parallel execution](parallel.md) with broadcasting and mapping
- Explore [conditional branching](decisions.md) with decision nodes
- See the [API reference][pydantic_graph.beta.join] for complete reducer documentation
