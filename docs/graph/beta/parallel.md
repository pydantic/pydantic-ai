# Parallel Execution

The beta graph API provides two powerful mechanisms for parallel execution: **broadcasting** and **mapping**.

## Overview

- **Broadcasting** - Send the same data to multiple parallel paths
- **Spreading** - Fan out items from an iterable to parallel paths

Both create "forks" in the execution graph that can later be synchronized with [join nodes](joins.md).

## Broadcasting

Broadcasting sends identical data to multiple destinations simultaneously:

```python {title="basic_broadcast.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, ListReducer, StepContext


@dataclass
class SimpleState:
    pass


async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=list[int])

    @g.step
    async def source(ctx: StepContext[SimpleState, None, None]) -> int:
        return 10

    @g.step
    async def add_one(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs + 1

    @g.step
    async def add_two(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs + 2

    @g.step
    async def add_three(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs + 3

    collect = g.join(ListReducer[int])

    # Broadcasting: send the value from source to all three steps
    g.add(
        g.edge_from(g.start_node).to(source),
        g.edge_from(source).to(add_one, add_two, add_three),
        g.edge_from(add_one, add_two, add_three).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    print(sorted(result))
    #> [11, 12, 13]
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

All three steps receive the same input value (`10`) and execute in parallel.

## Spreading

Spreading fans out elements from an iterable, processing each element in parallel:

```python {title="basic_map.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, ListReducer, StepContext


@dataclass
class SimpleState:
    pass


async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=list[int])

    @g.step
    async def generate_list(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return [1, 2, 3, 4, 5]

    @g.step
    async def square(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs * ctx.inputs

    collect = g.join(ListReducer[int])

    # Spreading: each item in the list gets its own parallel execution
    g.add(
        g.edge_from(g.start_node).to(generate_list),
        g.edge_from(generate_list).map().to(square),
        g.edge_from(square).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    print(sorted(result))
    #> [1, 4, 9, 16, 25]
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

### Using `add_mapping_edge()`

The convenience method [`add_mapping_edge()`][pydantic_graph.beta.graph_builder.GraphBuilder.add_mapping_edge] provides a simpler syntax:

```python {title="mapping_convenience.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, ListReducer, StepContext, Reducer


@dataclass
class SimpleState:
    pass


async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=list[str])

    @g.step
    async def generate_numbers(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return [10, 20, 30]

    @g.step
    async def stringify(ctx: StepContext[SimpleState, None, int]) -> str:
        return f'Value: {ctx.inputs}'

    collect = g.join(ListReducer[str])

    g.add(g.edge_from(g.start_node).to(generate_numbers))
    g.add_mapping_edge(generate_numbers, stringify)
    g.add(
        g.edge_from(stringify).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    print(sorted(result))
    #> ['Value: 10', 'Value: 20', 'Value: 30']
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Empty Iterables

When mapping an empty iterable, you can specify a `downstream_join_id` to ensure the join still executes:

```python {title="empty_map.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, ListReducer, StepContext


@dataclass
class SimpleState:
    pass


async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=list[int])

    @g.step
    async def generate_empty(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return []

    @g.step
    async def double(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs * 2

    collect = g.join(ListReducer[int])

    g.add(g.edge_from(g.start_node).to(generate_empty))
    g.add_mapping_edge(generate_empty, double, downstream_join_id=collect.id)
    g.add(
        g.edge_from(double).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    print(result)
    #> []
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Nested Parallel Operations

You can nest broadcasts and maps for complex parallel patterns:

### Spread then Broadcast

```python {title="map_then_broadcast.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, ListReducer, StepContext


@dataclass
class SimpleState:
    pass


async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=list[int])

    @g.step
    async def generate_list(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return [10, 20]

    @g.step
    async def add_one(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs + 1

    @g.step
    async def add_two(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs + 2

    collect = g.join(ListReducer[int])

    g.add(
        g.edge_from(g.start_node).to(generate_list),
        # Spread the list, then broadcast each item to both steps
        g.edge_from(generate_list).map().to(add_one, add_two),
        g.edge_from(add_one, add_two).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    print(sorted(result))
    #> [11, 12, 21, 22]
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

The result contains:
- From 10: `10+1=11` and `10+2=12`
- From 20: `20+1=21` and `20+2=22`

### Multiple Sequential Spreads

```python {title="sequential_maps.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, ListReducer, StepContext


@dataclass
class SimpleState:
    pass


async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=list[str])

    @g.step
    async def generate_pairs(ctx: StepContext[SimpleState, None, None]) -> list[tuple[int, int]]:
        return [(1, 2), (3, 4)]

    @g.step
    async def unpack_pair(ctx: StepContext[SimpleState, None, tuple[int, int]]) -> list[int]:
        return [ctx.inputs[0], ctx.inputs[1]]

    @g.step
    async def stringify(ctx: StepContext[SimpleState, None, int]) -> str:
        return f'num:{ctx.inputs}'

    collect = g.join(ListReducer[str])

    g.add(
        g.edge_from(g.start_node).to(generate_pairs),
        # First map: one task per tuple
        g.edge_from(generate_pairs).map().to(unpack_pair),
        # Second map: one task per number in each tuple
        g.edge_from(unpack_pair).map().to(stringify),
        g.edge_from(stringify).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    print(sorted(result))
    #> ['num:1', 'num:2', 'num:3', 'num:4']
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Edge Labels

Add labels to parallel edges for better documentation:

```python {title="labeled_parallel.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, ListReducer, StepContext


@dataclass
class SimpleState:
    pass


async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=list[str])

    @g.step
    async def generate(ctx: StepContext[SimpleState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def process(ctx: StepContext[SimpleState, None, int]) -> str:
        return f'item-{ctx.inputs}'

    collect = g.join(ListReducer[str])

    g.add(g.edge_from(g.start_node).to(generate))
    g.add_mapping_edge(
        generate,
        process,
        pre_map_label='before map',
        post_map_label='after map',
    )
    g.add(
        g.edge_from(process).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    print(sorted(result))
    #> ['item-1', 'item-2', 'item-3']
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## State Sharing in Parallel Execution

All parallel tasks share the same graph state. Be careful with mutations:

```python {title="parallel_state.py"}
from dataclasses import dataclass, field

from pydantic_graph.beta import GraphBuilder, ListReducer, StepContext


@dataclass
class CounterState:
    values: list[int] = field(default_factory=list)


async def main():
    g = GraphBuilder(state_type=CounterState, output_type=list[int])

    @g.step
    async def generate(ctx: StepContext[CounterState, None, None]) -> list[int]:
        return [1, 2, 3]

    @g.step
    async def track_and_square(ctx: StepContext[CounterState, None, int]) -> int:
        # All parallel tasks mutate the same state
        ctx.state.values.append(ctx.inputs)
        return ctx.inputs * ctx.inputs

    collect = g.join(ListReducer[int])

    g.add(
        g.edge_from(g.start_node).to(generate),
        g.edge_from(generate).map().to(track_and_square),
        g.edge_from(track_and_square).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    state = CounterState()
    result = await graph.run(state=state)

    print(f'Squared: {sorted(result)}')
    #> Squared: [1, 4, 9]
    print(f'Tracked: {sorted(state.values)}')
    #> Tracked: [1, 2, 3]
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Next Steps

- Learn about [join nodes](joins.md) for aggregating parallel results
- Explore [conditional branching](decisions.md) with decision nodes
- See the [steps documentation](steps.md) for more on step execution
