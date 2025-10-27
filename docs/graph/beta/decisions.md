# Decision Nodes

Decision nodes enable conditional branching in your graph based on the type or value of data flowing through it.

## Overview

A decision node evaluates incoming data and routes it to different branches based on:

- Type matching (using `isinstance`)
- Literal value matching
- Custom predicate functions

The first matching branch is taken, similar to pattern matching or `if-elif-else` chains.

## Creating Decisions

Use [`g.decision()`][pydantic_graph.beta.graph_builder.GraphBuilder.decision] to create a decision node, then add branches with [`g.match()`][pydantic_graph.beta.graph_builder.GraphBuilder.match]:

```python {title="simple_decision.py"}
from dataclasses import dataclass
from typing import Literal

from pydantic_graph.beta import GraphBuilder, StepContext, TypeExpression


@dataclass
class DecisionState:
    path_taken: str | None = None


async def main():
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def choose_path(ctx: StepContext[DecisionState, None, None]) -> Literal['left', 'right']:
        return 'left'

    @g.step
    async def left_path(ctx: StepContext[DecisionState, None, object]) -> str:
        ctx.state.path_taken = 'left'
        return 'Went left'

    @g.step
    async def right_path(ctx: StepContext[DecisionState, None, object]) -> str:
        ctx.state.path_taken = 'right'
        return 'Went right'

    g.add(
        g.edge_from(g.start_node).to(choose_path),
        g.edge_from(choose_path).to(
            g.decision()
            .branch(g.match(TypeExpression[Literal['left']]).to(left_path))
            .branch(g.match(TypeExpression[Literal['right']]).to(right_path))
        ),
        g.edge_from(left_path, right_path).to(g.end_node),
    )

    graph = g.build()
    state = DecisionState()
    result = await graph.run(state=state)
    print(result)
    #> Went left
    print(state.path_taken)
    #> left
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Type Matching

Match by type using regular Python types:

```python {title="type_matching.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext


@dataclass
class DecisionState:
    pass


async def main():
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def return_int(ctx: StepContext[DecisionState, None, None]) -> int:
        return 42

    @g.step
    async def handle_int(ctx: StepContext[DecisionState, None, int]) -> str:
        return f'Got int: {ctx.inputs}'

    @g.step
    async def handle_str(ctx: StepContext[DecisionState, None, str]) -> str:
        return f'Got str: {ctx.inputs}'

    g.add(
        g.edge_from(g.start_node).to(return_int),
        g.edge_from(return_int).to(
            g.decision()
            .branch(g.match(int).to(handle_int))
            .branch(g.match(str).to(handle_str))
        ),
        g.edge_from(handle_int, handle_str).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    print(result)
    #> Got int: 42
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

### Matching Union Types

For more complex type expressions like unions, you need to use [`TypeExpression`][pydantic_graph.beta.util.TypeExpression] because Python's type system doesn't allow union types to be used directly as runtime values:

```python {title="union_type_matching.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext, TypeExpression


@dataclass
class DecisionState:
    pass


async def main():
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def return_value(ctx: StepContext[DecisionState, None, None]) -> int | str:
        """Returns either an int or a str."""
        return 42

    @g.step
    async def handle_number(ctx: StepContext[DecisionState, None, int | float]) -> str:
        return f'Got number: {ctx.inputs}'

    @g.step
    async def handle_text(ctx: StepContext[DecisionState, None, str]) -> str:
        return f'Got text: {ctx.inputs}'

    g.add(
        g.edge_from(g.start_node).to(return_value),
        g.edge_from(return_value).to(
            g.decision()
            # Use TypeExpression for union types
            .branch(g.match(TypeExpression[int | float]).to(handle_number))
            .branch(g.match(str).to(handle_text))
        ),
        g.edge_from(handle_number, handle_text).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    print(result)
    #> Got number: 42
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

!!! note
    [`TypeExpression`][pydantic_graph.beta.util.TypeExpression] is only necessary for complex type expressions like unions (`int | str`), `Literal`, and other type forms that aren't valid as runtime `type` objects. For simple types like `int`, `str`, or custom classes, you can pass them directly to `g.match()`.

    The `TypeForm` class introduced in [PEP 747](https://peps.python.org/pep-0747/) should eventually eliminate the need for this workaround.


## Custom Matchers

Provide custom matching logic with the `matches` parameter:

```python {title="custom_matcher.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext, TypeExpression


@dataclass
class DecisionState:
    pass


async def main():
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def return_number(ctx: StepContext[DecisionState, None, None]) -> int:
        return 7

    @g.step
    async def even_path(ctx: StepContext[DecisionState, None, int]) -> str:
        return f'{ctx.inputs} is even'

    @g.step
    async def odd_path(ctx: StepContext[DecisionState, None, int]) -> str:
        return f'{ctx.inputs} is odd'

    g.add(
        g.edge_from(g.start_node).to(return_number),
        g.edge_from(return_number).to(
            g.decision()
            .branch(g.match(TypeExpression[int], matches=lambda x: x % 2 == 0).to(even_path))
            .branch(g.match(TypeExpression[int], matches=lambda x: x % 2 == 1).to(odd_path))
        ),
        g.edge_from(even_path, odd_path).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    print(result)
    #> 7 is odd
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Branch Priority

Branches are evaluated in the order they're added. The first matching branch is taken:

```python {title="branch_priority.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext, TypeExpression


@dataclass
class DecisionState:
    pass


async def main():
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def return_value(ctx: StepContext[DecisionState, None, None]) -> int:
        return 10

    @g.step
    async def branch_a(ctx: StepContext[DecisionState, None, int]) -> str:
        return 'Branch A'

    @g.step
    async def branch_b(ctx: StepContext[DecisionState, None, int]) -> str:
        return 'Branch B'

    g.add(
        g.edge_from(g.start_node).to(return_value),
        g.edge_from(return_value).to(
            g.decision()
            .branch(g.match(TypeExpression[int], matches=lambda x: x >= 5).to(branch_a))
            .branch(g.match(TypeExpression[int], matches=lambda x: x >= 0).to(branch_b))
        ),
        g.edge_from(branch_a, branch_b).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    print(result)
    #> Branch A
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

Both branches could match `10`, but Branch A is first, so it's taken.

## Catch-All Branches

Use `object` or `Any` to create a catch-all branch:

```python {title="catch_all.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext, TypeExpression


@dataclass
class DecisionState:
    pass


async def main():
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def return_value(ctx: StepContext[DecisionState, None, None]) -> int:
        return 100

    @g.step
    async def catch_all(ctx: StepContext[DecisionState, None, object]) -> str:
        return f'Caught: {ctx.inputs}'

    g.add(
        g.edge_from(g.start_node).to(return_value),
        g.edge_from(return_value).to(g.decision().branch(g.match(TypeExpression[object]).to(catch_all))),
        g.edge_from(catch_all).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    print(result)
    #> Caught: 100
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Nested Decisions

Decisions can be nested for complex conditional logic:

```python {title="nested_decisions.py"}
from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext, TypeExpression


@dataclass
class DecisionState:
    pass


async def main():
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def get_number(ctx: StepContext[DecisionState, None, None]) -> int:
        return 15

    @g.step
    async def is_positive(ctx: StepContext[DecisionState, None, int]) -> int:
        return ctx.inputs

    @g.step
    async def is_negative(ctx: StepContext[DecisionState, None, int]) -> str:
        return 'Negative'

    @g.step
    async def small_positive(ctx: StepContext[DecisionState, None, int]) -> str:
        return 'Small positive'

    @g.step
    async def large_positive(ctx: StepContext[DecisionState, None, int]) -> str:
        return 'Large positive'

    g.add(
        g.edge_from(g.start_node).to(get_number),
        g.edge_from(get_number).to(
            g.decision()
            .branch(g.match(TypeExpression[int], matches=lambda x: x > 0).to(is_positive))
            .branch(g.match(TypeExpression[int], matches=lambda x: x <= 0).to(is_negative))
        ),
        g.edge_from(is_positive).to(
            g.decision()
            .branch(g.match(TypeExpression[int], matches=lambda x: x < 10).to(small_positive))
            .branch(g.match(TypeExpression[int], matches=lambda x: x >= 10).to(large_positive))
        ),
        g.edge_from(is_negative, small_positive, large_positive).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    print(result)
    #> Large positive
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Branching with Labels

Add labels to branches for documentation and diagram generation:

```python {title="labeled_branches.py"}
from dataclasses import dataclass
from typing import Literal

from pydantic_graph.beta import GraphBuilder, StepContext, TypeExpression


@dataclass
class DecisionState:
    pass


async def main():
    g = GraphBuilder(state_type=DecisionState, output_type=str)

    @g.step
    async def choose(ctx: StepContext[DecisionState, None, None]) -> Literal['a', 'b']:
        return 'a'

    @g.step
    async def path_a(ctx: StepContext[DecisionState, None, object]) -> str:
        return 'Path A'

    @g.step
    async def path_b(ctx: StepContext[DecisionState, None, object]) -> str:
        return 'Path B'

    g.add(
        g.edge_from(g.start_node).to(choose),
        g.edge_from(choose).to(
            g.decision()
            .branch(g.match(TypeExpression[Literal['a']]).label('Take path A').to(path_a))
            .branch(g.match(TypeExpression[Literal['b']]).label('Take path B').to(path_b))
        ),
        g.edge_from(path_a, path_b).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=DecisionState())
    print(result)
    #> Path A
```

_(This example is complete, it can be run "as is" — you'll need to add `import asyncio; asyncio.run(main())` to run `main`)_

## Next Steps

- Learn about [parallel execution](parallel.md) with broadcasting and mapping
- Understand [join nodes](joins.md) for aggregating parallel results
- See the [API reference][pydantic_graph.beta.decision] for complete decision documentation
