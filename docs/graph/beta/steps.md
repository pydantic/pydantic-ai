# Steps

Steps are the fundamental units of work in a graph. They're async functions that receive a [`StepContext`][pydantic_graph.beta.step.StepContext] and return a value.

## Creating Steps

Steps are created using the [`@g.step`][pydantic_graph.beta.graph_builder.GraphBuilder.step] decorator on the [`GraphBuilder`][pydantic_graph.beta.graph_builder.GraphBuilder]:

```python {title="basic_step.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext


@dataclass
class MyState:
    counter: int = 0


async def main():
    g = GraphBuilder(state_type=MyState, output_type=int)

    @g.step
    async def increment(ctx: StepContext[MyState, None, None]) -> int:
        ctx.state.counter += 1
        return ctx.state.counter

    g.add(
        g.edge_from(g.start_node).to(increment),
        g.edge_from(increment).to(g.end_node),
    )

    graph = g.build()
    state = MyState()
    result = await graph.run(state=state)
    print(result)
    #> 1
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Step Context

Every step function receives a [`StepContext`][pydantic_graph.beta.step.StepContext] as its first parameter. The context provides access to:

- `ctx.state` - The mutable graph state (type: `StateT`)
- `ctx.deps` - Injected dependencies (type: `DepsT`)
- `ctx.inputs` - Input data for this step (type: `InputT`)

### Accessing State

State is shared across all steps in a graph and can be freely mutated:

```python {title="state_access.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext


@dataclass
class AppState:
    messages: list[str]


async def main():
    g = GraphBuilder(state_type=AppState, output_type=list[str])

    @g.step
    async def add_hello(ctx: StepContext[AppState, None, None]) -> None:
        ctx.state.messages.append('Hello')

    @g.step
    async def add_world(ctx: StepContext[AppState, None, None]) -> None:
        ctx.state.messages.append('World')

    @g.step
    async def get_messages(ctx: StepContext[AppState, None, None]) -> list[str]:
        return ctx.state.messages

    g.add(
        g.edge_from(g.start_node).to(add_hello),
        g.edge_from(add_hello).to(add_world),
        g.edge_from(add_world).to(get_messages),
        g.edge_from(get_messages).to(g.end_node),
    )

    graph = g.build()
    state = AppState(messages=[])
    result = await graph.run(state=state)
    print(result)
    #> ['Hello', 'World']
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

### Working with Inputs

Steps can receive and transform input data:

```python {title="step_inputs.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext


@dataclass
class SimpleState:
    pass


async def main():
    g = GraphBuilder(
        state_type=SimpleState,
        input_type=int,
        output_type=str,
    )

    @g.step
    async def double_it(ctx: StepContext[SimpleState, None, int]) -> int:
        """Double the input value."""
        return ctx.inputs * 2

    @g.step
    async def stringify(ctx: StepContext[SimpleState, None, int]) -> str:
        """Convert to a formatted string."""
        return f'Result: {ctx.inputs}'

    g.add(
        g.edge_from(g.start_node).to(double_it),
        g.edge_from(double_it).to(stringify),
        g.edge_from(stringify).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState(), inputs=21)
    print(result)
    #> Result: 42
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Dependency Injection

Steps can access injected dependencies through `ctx.deps`:

```python {title="dependencies.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext


@dataclass
class AppState:
    pass


@dataclass
class AppDeps:
    """Dependencies injected into the graph."""

    multiplier: int


async def main():
    g = GraphBuilder(
        state_type=AppState,
        deps_type=AppDeps,
        input_type=int,
        output_type=int,
    )

    @g.step
    async def multiply(ctx: StepContext[AppState, AppDeps, int]) -> int:
        """Multiply input by the injected multiplier."""
        return ctx.inputs * ctx.deps.multiplier

    g.add(
        g.edge_from(g.start_node).to(multiply),
        g.edge_from(multiply).to(g.end_node),
    )

    graph = g.build()
    deps = AppDeps(multiplier=10)
    result = await graph.run(state=AppState(), deps=deps, inputs=5)
    print(result)
    #> 50
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Customizing Steps

### Custom Node IDs

By default, step node IDs are inferred from the function name. You can override this:

```python {title="custom_id.py" requires="basic_step.py"}
from basic_step import MyState, g

@g.step(node_id='my_custom_id')
async def my_step(ctx: StepContext[MyState, None, None]) -> int:
    return 42

# The node ID is now 'my_custom_id' instead of 'my_step'
```

### Human-Readable Labels

Labels provide documentation for diagram generation:

```python {title="labels.py" requires="basic_step.py"}
from basic_step import MyState, g

@g.step(label='Increment the counter')
async def increment(ctx: StepContext[MyState, None, None]) -> int:
    ctx.state.counter += 1
    return ctx.state.counter

# Access the label programmatically
print(increment.label)
#> Increment the counter
```

## Sequential Steps

Multiple steps can be chained sequentially:

```python {title="sequential.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext


@dataclass
class MathState:
    operations: list[str]


async def main():
    g = GraphBuilder(
        state_type=MathState,
        input_type=int,
        output_type=int,
    )

    @g.step
    async def add_five(ctx: StepContext[MathState, None, int]) -> int:
        ctx.state.operations.append('add 5')
        return ctx.inputs + 5

    @g.step
    async def multiply_by_two(ctx: StepContext[MathState, None, int]) -> int:
        ctx.state.operations.append('multiply by 2')
        return ctx.inputs * 2

    @g.step
    async def subtract_three(ctx: StepContext[MathState, None, int]) -> int:
        ctx.state.operations.append('subtract 3')
        return ctx.inputs - 3

    # Connect steps sequentially
    g.add(
        g.edge_from(g.start_node).to(add_five),
        g.edge_from(add_five).to(multiply_by_two),
        g.edge_from(multiply_by_two).to(subtract_three),
        g.edge_from(subtract_three).to(g.end_node),
    )

    graph = g.build()
    state = MathState(operations=[])
    result = await graph.run(state=state, inputs=10)

    print(f'Result: {result}')
    #> Result: 27
    print(f'Operations: {state.operations}')
    #> Operations: ['add 5', 'multiply by 2', 'subtract 3']
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

The computation is: `(10 + 5) * 2 - 3 = 27`

## Edge Building Convenience Methods

The builder provides helper methods for common edge patterns:

### Simple Edges with `add_edge()`

```python {title="add_edge_example.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext


@dataclass
class SimpleState:
    pass


async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=int)

    @g.step
    async def step_a(ctx: StepContext[SimpleState, None, None]) -> int:
        return 10

    @g.step
    async def step_b(ctx: StepContext[SimpleState, None, int]) -> int:
        return ctx.inputs + 5

    # Using add_edge() for simple connections
    g.add_edge(g.start_node, step_a)
    g.add_edge(step_a, step_b, label='from a to b')
    g.add_edge(step_b, g.end_node)

    graph = g.build()
    result = await graph.run(state=SimpleState())
    print(result)
    #> 15
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Type Safety

The beta graph API provides strong type checking through generics. Type parameters on [`StepContext`][pydantic_graph.beta.step.StepContext] ensure:

- State access is properly typed
- Dependencies are correctly typed
- Input/output types match across edges

```python
# Type checker will catch mismatches
@g.step
async def expects_int(ctx: StepContext[MyState, None, int]) -> str:
    return str(ctx.inputs)

@g.step
async def returns_str(ctx: StepContext[MyState, None, None]) -> str:
    return 'hello'

# This would be a type error - expects_int needs int input, but returns_str outputs str
# g.add(g.edge_from(returns_str).to(expects_int))  # Type error!
```

## Next Steps

- Learn about [parallel execution](parallel.md) with broadcasting and spreading
- Understand [join nodes](joins.md) for aggregating parallel results
- Explore [conditional branching](decisions.md) with decision nodes
