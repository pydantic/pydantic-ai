from mypyc.primitives.float_ops import int_to_float_op

# Joins and Reducers

Join nodes synchronize and aggregate data from parallel execution paths. They use **Reducers** to combine multiple inputs into a single output.

## Overview

When you use [parallel execution](parallel.md) (broadcasting or mapping), you often need to collect and combine the results. Join nodes serve this purpose by:

1. Waiting for all parallel tasks to complete
2. Aggregating their outputs using a [`ReducerFunction`][pydantic_graph.beta.join.ReducerFunction]
3. Passing the aggregated result to the next node

## Creating Joins

Create a join using `GraphBuilder.join` with a reducer function and initial value or factory:

```python {title="basic_join.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append


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
collect = g.join(reduce_list_append, initial_factory=list[int])

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

### `reduce_list_append`

[`reduce_list_append`][pydantic_graph.beta.join.reduce_list_append] collects all inputs into a list:

```python {title="list_reducer.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append


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

    collect = g.join(reduce_list_append, initial_factory=list[str])

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

### `reduce_dict_update`

[`reduce_dict_update`][pydantic_graph.beta.join.reduce_dict_update] merges dictionaries together:

```python {title="dict_reducer.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_dict_update


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

    merge = g.join(reduce_dict_update, initial_factory=dict[str, int])

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

### `reduce_null`

[`reduce_null`][pydantic_graph.beta.join.reduce_null] discards all inputs and returns `None`. Useful when you only care about side effects:

```python {title="null_reducer.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_null


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
    ignore = g.join(reduce_null, initial=None)

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

Create custom reducers by defining a [`ReducerFunction`][pydantic_graph.beta.join.ReducerFunction]:

```python {title="custom_reducer.py"}

from pydantic_graph.beta import GraphBuilder, StepContext


def reduce_sum(current: int, inputs: int) -> int:
    """A reducer that sums numbers."""
    return current + inputs


async def main():
    g = GraphBuilder(output_type=int)

    @g.step
    async def generate(ctx: StepContext[None, None, None]) -> list[int]:
        return [5, 10, 15, 20]

    @g.step
    async def identity(ctx: StepContext[None, None, int]) -> int:
        return ctx.inputs

    sum_join = g.join(reduce_sum, initial=0)

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(identity),
        g.edge_from(identity).to(sum_join),
        g.edge_from(sum_join).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run()
    print(result)
    #> 50
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Reducers with State Access

Reducers can access and modify the graph state:

```python {title="stateful_reducer.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import ReducerContext


@dataclass
class MetricsState:
    total_count: int = 0
    total_sum: int = 0


@dataclass
class ReducedMetrics:
    count: int = 0
    sum: int = 0


def reduce_metrics_sum(ctx: ReducerContext[MetricsState, None], current: ReducedMetrics, inputs: int) -> ReducedMetrics:
    ctx.state.total_count += 1
    ctx.state.total_sum += inputs
    return ReducedMetrics(count=current.count + 1, sum=current.sum + inputs)

def reduce_metrics_max(current: ReducedMetrics, inputs: ReducedMetrics) -> ReducedMetrics:
    return ReducedMetrics(count=max(current.count, inputs.count), sum=max(current.sum, inputs.sum))


async def main():
    g = GraphBuilder(state_type=MetricsState, output_type=dict[str, int])

    @g.step
    async def generate(ctx: StepContext[object, None, None]) -> list[int]:
        return [1, 3, 5, 7, 9, 10, 20, 30, 40]

    @g.step
    async def process_even(ctx: StepContext[MetricsState, None, int]) -> int:
        return ctx.inputs * 2

    @g.step
    async def process_odd(ctx: StepContext[MetricsState, None, int]) -> int:
        return ctx.inputs * 3

    metrics_even = g.join(reduce_metrics_sum, initial_factory=ReducedMetrics, node_id='metrics_even')
    metrics_odd = g.join(reduce_metrics_sum, initial_factory=ReducedMetrics, node_id='metrics_odd')
    metrics_max = g.join(reduce_metrics_max, initial_factory=ReducedMetrics, node_id='metrics_max')

    g.add(
        g.edge_from(g.start_node).to(generate),
        # Send even and odd numbers to their respective `process` steps
        g.edge_from(generate).map().to(
            g.decision()
            .branch(g.match(int, matches=lambda x: x % 2 == 0).label('even').to(process_even))
            .branch(g.match(int, matches=lambda x: x % 2 == 1).label('odd').to(process_odd))
        ),
        # Reduce metrics for even and odd numbers separately
        g.edge_from(process_even).to(metrics_even),
        g.edge_from(process_odd).to(metrics_odd),
        # Aggregate the max values for each field
        g.edge_from(metrics_even).to(metrics_max),
        g.edge_from(metrics_odd).to(metrics_max),
        # Finish the graph run with the final reduced value
        g.edge_from(metrics_max).to(g.end_node),
    )

    graph = g.build()
    state = MetricsState()
    result = await graph.run(state=state)

    print(f'Result: {result}')
    #> Result: ReducedMetrics(count=5, sum=200)
    # > Result: {'count': 4, 'total': 200}
    print(f'State total_count: {state.total_count}')
    #> State total_count: 9
    # > State items_processed: 4
    print(f'State total_sum: {state.total_sum}')
    #> State total_sum: 275
    # > State sum_total: 200
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Multiple Joins

A graph can have multiple independent joins:

```python {title="multiple_joins.py"}
from dataclasses import dataclass, field

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append


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

    join_a = g.join(reduce_list_append, initial_factory=list[int], node_id='join_a')
    join_b = g.join(reduce_list_append, initial_factory=list[int], node_id='join_b')

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
from pydantic_graph.beta.join import reduce_list_append

from basic_join import g

my_join = g.join(reduce_list_append, initial_factory=list[int], node_id='my_custom_join_id')
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
