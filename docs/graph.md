# Graphs

!!! danger "Don't use a nail gun unless you need a nail gun"
    If PydanticAI [agents](agents.md) are a hammer, and [multi-agent workflows](multi-agent-applications.md) are a sledgehammer, then graphs are a nail gun:

    * sure, nail guns look cooler than hammers
    * but nail guns take a lot more setup than hammers
    * and nail guns don't make you a better builder, they make you a builder with a nail gun
    * Lastly, (and at the risk of torturing this metaphor), if you're a fan of medieval tools like mallets and untyped Python, you probably won't like nail guns or our approach to graphs. (But then again, if you're not a fan of type hints in Python, you've probably already bounced off PydanticAI to use one of the toy agent frameworks — good luck, and feel free to borrow my sledgehammer when you realize you need it)

    In short, graphs are a powerful tool, but they're not the right tool for every job. Please consider other [multi-agent approaches](multi-agent-applications.md) before proceeding.

    If you're not confident a graph-based approach is a good idea, it might be unnecessary.

Graphs and finite state machines (FSMs) are a powerful abstraction to model, execute, control and visualize complex workflows.

Alongside PydanticAI, we've developed `pydantic-graph` — an async graph and state machine library for Python where nodes and edges are defined using type hints.

While this library is developed as part of PydanticAI; it has no dependency on `pydantic-ai` and can be considered as a pure  graph-based state machine library. You may find it useful whether or not you're using PydanticAI or even building with GenAI.

`pydantic-graph` is designed for advanced users and makes heavy use of Python generics and types hints. It is not designed to be as beginner-friendly as PydanticAI.

!!! note "Very Early beta"
    Graph support was [introduced](https://github.com/pydantic/pydantic-ai/pull/528) in v0.0.19 and is in very earlier beta. The API is subject to change. The documentation is incomplete. The implementation is incomplete.

## Installation

`pydantic-graph` is a required dependency of `pydantic-ai`, and an optional dependency of `pydantic-ai-slim`, see [installation instructions](install.md#slim-install) for more information. You can also install it directly:

```bash
pip/uv-add pydantic-graph
```

## Graph Types

`pydantic-graph` made up of a few key components:

### GraphRunContext

[`GraphRunContext`][pydantic_graph.nodes.GraphRunContext] — The context for the graph run, similar to PydanticAI's [`RunContext`][pydantic_ai.tools.RunContext]. This holds graph dependencies and is passed to nodes when they're run.

### End

[`End`][pydantic_graph.nodes.End] — return value to indicate the graph run should end.

`End` is generic in the graph return type of the graph it's used in, [`RunEndT`][pydantic_graph.nodes.RunEndT].

### Nodes

Subclasses of [`BaseNode`][pydantic_graph.nodes.BaseNode] define nodes for execution in the graph.

Nodes, which are generally [`dataclass`es][dataclasses.dataclass], generally consist of:

* fields containing any parameters required/optional when calling the node
* the business logic to execute the node, in the [`run`][pydantic_graph.nodes.BaseNode.run] method
* return annotations of the [`run`][pydantic_graph.nodes.BaseNode.run] method, which are read by `pydantic-graph` to determine the outgoing edges of the node

Nodes are generic in:

* **deps**, which must have the same type as the deps of the graph they're included in, [`DepsT`][pydantic_graph.nodes.DepsT] has a default of `None`, so if you're not using deps you can omit this generic parameter, see [dependency injection](#dependency-injection) for more information
* **graph return type** — this only applies if the node returns [`End`][pydantic_graph.nodes.End]. [`RunEndT`][pydantic_graph.nodes.RunEndT] has a default of [`Never`][typing.Never], so this generic parameter can be omitted if the node doesn't return [`End`][pydantic_graph.nodes.End], but must be included if it does.

Here's an example of a start or intermediate node in a graph — it can't end the run as it doesn't return [`End`][pydantic_graph.nodes.End]:

```py {title="intermediate_node.py" noqa="F821" test="skip"}
from dataclasses import dataclass

from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class MyNode(BaseNode):  # (1)!
    foo: int  # (2)!

    async def run(
        self,
        ctx: GraphRunContext,  # (3)!
    ) -> AnotherNode:  # (4)!
        ...
        return AnotherNode()
```

1. This graph doesn't take deps, so the [`DepsT`][pydantic_graph.nodes.DepsT] generic is omitted (and defaults to `None`). This node can't end the run, so the [`RunEndT`][pydantic_graph.nodes.RunEndT] generic parameter is omitted (and therefore defaults to [`Never`][typing.Never]). So in this case, both generics can be omitted when inheriting from [`BaseNode`][pydantic_graph.nodes.BaseNode].
2. `MyNode` is a dataclass and has a single field `foo`, an `int`.
3. The [`run`][pydantic_graph.nodes.BaseNode.run] method takes a [`GraphRunContext`][pydantic_graph.nodes.GraphRunContext] parameter.
4. The return type of the [`run`][pydantic_graph.nodes.BaseNode.run] method is `AnotherNode` (not shown); this is used to determine the outgoing edges of the node.

We could extend `MyNode` to optionally end the run if `foo` is divisible by 5:

```py {title="intermediate_or_end_node.py" hl_lines="7 13 15" noqa="F821" test="skip"}
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext


@dataclass
class MyNode(BaseNode[None, int]):  # (1)!
    foo: int

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> AnotherNode | End[int]:  # (2)!
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return AnotherNode()
```

1. We parameterize the node with the graph return type (`int` in this case). Because generic parameters are positional-only, we have to include `None` as the first parameter, representing the deps type.
2. The return type of the `run` method is now a union of `AnotherNode` and [`End[int]`][pydantic_graph.nodes.End], this allows the node to end the run if `foo` is divisible by 5.

### Graph

[`Graph`][pydantic_graph.graph.Graph] — this is the execution graph itself, made up of a set of [node classes](#nodes) (i.e., `BaseNode` subclasses).

`Graph` is generic in:

* **deps** the deps type of the graph, [`DepsT`][pydantic_graph.nodes.DepsT]
* **graph return type** the return type of the graph run, [`RunEndT`][pydantic_graph.nodes.RunEndT]

Here's an example of a simple graph:

```py {title="graph_example.py" py="3.10"}
from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class DivisibleBy5(BaseNode[None, int]):  # (1)!
    foo: int

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> Increment | End[int]:
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return Increment(self.foo)


@dataclass
class Increment(BaseNode):  # (2)!
    foo: int

    async def run(self, ctx: GraphRunContext) -> DivisibleBy5:
        return DivisibleBy5(self.foo + 1)


fives_graph = Graph(nodes=[DivisibleBy5, Increment])  # (3)!
result, history = fives_graph.run_sync(DivisibleBy5(4))  # (4)!
print(result)
#> 5
# the full history is quite verbose (see below), so we'll just print the summary
print([item.data for item in history])
#> [DivisibleBy5(foo=4), Increment(foo=4), DivisibleBy5(foo=5), End(data=5)]
```

1. The `DivisibleBy5` node is parameterized with `None` for the state param and `None` for the deps param as this graph doesn't use state or deps, and `int` as it can end the run.
2. The `Increment` node doesn't return `End`, so the `RunEndT` generic parameter is omitted, state can also be omitted as the graph doesn't use state.
3. The graph is created with a sequence of nodes.
4. The graph is run synchronously with [`run_sync`][pydantic_graph.graph.Graph.run_sync] the initial state `None` and the start node `DivisibleBy5(4)` are passed as arguments.

_(This example is complete, it can be run "as is" with Python 3.10+)_

A [mermaid diagram](#mermaid-diagrams) for this graph can be generated with the following code:

```py {title="graph_example_diagram.py" py="3.10"}
from graph_example import DivisibleBy5, fives_graph

fives_graph.mermaid_code(start_node=DivisibleBy5)
```

```mermaid
---
title: fives_graph
---
stateDiagram-v2
  [*] --> DivisibleBy5
  DivisibleBy5 --> Increment
  DivisibleBy5 --> [*]
  Increment --> DivisibleBy5
```

In order to visualize a graph within a `jupyter-notebook`, `IPython.display` needs to be used:

```python {title="jupyter_display_mermaid.py"  test="skip"}
from graph_example import DivisibleBy5, fives_graph
from IPython.display import Image, display

display(Image(fives_graph.mermaid_image(start_node=DivisibleBy5)))
```

## Stateful Graphs

In `pydantic-graph`, the "state" of the finite state machine is encapsulated in the node instances themselves. This is in contrast with some other state machine libraries where the state exists as a separate object external to the nodes.

Note that in many scenarios, the state of the overall system may contain data well beyond the necessary inputs and outputs for the execution of any individual node. In these cases, the solution is to include a consistent `state` attribute on all nodes of the graph, and pass mutated versions of this state as an input to each subsequent node.

If you think of Graphs as an automotive production line, then this state is analogous to the vehicle being passed along the line and built up by each node as the graph is run.

In the future, we intend to extend `pydantic-graph` to provide persistence by recording the result of running each node; see [#695](https://github.com/pydantic/pydantic-ai/issues/695).

!!! note "Removed State Generic"
    Prior to `pydantic-graph` v0.23 graphs, had another generic parameter, `StateT`, which could be accessed via [`GraphRunContext`][pydantic_graph.nodes.GraphRunContext]. To simplify graphs and make their state easier to persist, this parameter has been removed. If you need a consistent "state" object, you should make it as a parameter to each of your nodes as shown below.

Here's an example of a graph which represents a vending machine where the user may insert coins and select a product to purchase.

```python {title="vending_machine.py" py="3.10"}
from __future__ import annotations

from dataclasses import dataclass

from rich.prompt import Prompt

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class MachineState:  # (1)!
    user_balance: float = 0.0
    product: str | None = None


@dataclass
class InsertCoin(BaseNode):  # (3)!
    state: MachineState

    async def run(self, ctx: GraphRunContext) -> CoinsInserted:  # (16)!
        return CoinsInserted(self.state, float(Prompt.ask('Insert coins')))  # (4)!


@dataclass
class CoinsInserted(BaseNode):
    state: MachineState
    amount: float  # (5)!

    async def run(self, ctx: GraphRunContext) -> SelectProduct | Purchase:  # (17)!
        self.state.user_balance += self.amount  # (6)!
        if self.state.product is not None:  # (7)!
            return Purchase(self.state, self.state.product)
        else:
            return SelectProduct(self.state)


@dataclass
class SelectProduct(BaseNode):
    state: MachineState

    async def run(self, ctx: GraphRunContext) -> Purchase:
        return Purchase(self.state, Prompt.ask('Select product'))


PRODUCT_PRICES = {  # (2)!
    'water': 1.25,
    'soda': 1.50,
    'crisps': 1.75,
    'chocolate': 2.00,
}


@dataclass
class Purchase(BaseNode[None, None]):  # (18)!
    state: MachineState
    product: str

    async def run(self, ctx: GraphRunContext) -> End | InsertCoin | SelectProduct:
        if price := PRODUCT_PRICES.get(self.product):  # (8)!
            self.state.product = self.product  # (9)!
            if self.state.user_balance >= price:  # (10)!
                self.state.user_balance -= price
                return End(None)
            else:
                diff = price - self.state.user_balance
                print(f'Not enough money for {self.product}, need {diff:0.2f} more')
                #> Not enough money for crisps, need 0.75 more
                return InsertCoin(self.state)  # (11)!
        else:
            print(f'No such product: {self.product}, try again')
            return SelectProduct(self.state)  # (12)!


vending_machine_graph = Graph(  # (13)!
    nodes=[InsertCoin, CoinsInserted, SelectProduct, Purchase]
)


async def main():
    state = MachineState()  # (14)!
    await vending_machine_graph.run(InsertCoin(state))  # (15)!
    print(f'purchase successful item={state.product} change={state.user_balance:0.2f}')
    #> purchase successful item=crisps change=0.25
```

1. The state of the vending machine is defined as a dataclass with the user's balance and the product they've selected, if any.
2. A dictionary of products mapped to prices.
3. The `InsertCoin` node, [`BaseNode`][pydantic_graph.nodes.BaseNode] is parameterized with `MachineState` as that's the state used in this graph.
4. The `InsertCoin` node prompts the user to insert coins. We keep things simple by just entering a monetary amount as a float. Before you start thinking this is a toy too since it's using [rich's `Prompt.ask`][rich.prompt.PromptBase.ask] within nodes, see [below](#custom-control-flow) for how control flow can be managed when nodes require external input.
5. The `CoinsInserted` node; again this is a [`dataclass`][dataclasses.dataclass], in this case with one field `amount`, thus nodes calling `CoinsInserted` must provide an amount.
6. Update the user's balance with the amount inserted.
7. If the user has already selected a product, go to `Purchase`, otherwise go to `SelectProduct`.
8. In the `Purchase` node, look up the price of the product if the user entered a valid product.
9. If the user did enter a valid product, set the product in the state so we don't revisit `SelectProduct`.
10. If the balance is enough to purchase the product, adjust the balance to reflect the purchase and return [`End`][pydantic_graph.nodes.End] to end the graph. We're not using the run return type, so we call `End` with `None`.
11. If the balance is insufficient, to go `InsertCoin` to prompt the user to insert more coins.
12. If the product is invalid, go to `SelectProduct` to prompt the user to select a product again.
13. The graph is created by passing a list of nodes to [`Graph`][pydantic_graph.graph.Graph]. Order of nodes is not important, but will alter how [diagrams](#mermaid-diagrams) are displayed.
14. Initialize the state. This will be passed to the graph run and mutated as the graph runs.
15. Run the graph with the initial state. Since the graph can be run from any node, we must pass the start node — in this case, `InsertCoin`. [`Graph.run`][pydantic_graph.graph.Graph.run] returns a tuple of the return value (`None`) in this case, and the [history][pydantic_graph.state.HistoryStep] of the graph run.
16. The return type of the node's [`run`][pydantic_graph.nodes.BaseNode.run] method is important as it is used to determine the outgoing edges of the node. This information in turn is used to render [mermaid diagrams](#mermaid-diagrams) and is enforced at runtime to detect misbehavior as soon as possible.
17. The return type of `CoinsInserted`'s [`run`][pydantic_graph.nodes.BaseNode.run] method is a union, meaning multiple outgoing edges are possible.
18. Unlike other nodes, `Purchase` can end the run, so the [`RunEndT`][pydantic_graph.nodes.RunEndT] generic parameter must be set. In this case it's `None` since the graph run return type is `None`.

_(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add `asyncio.run(main())` to run `main`)_

A [mermaid diagram](#mermaid-diagrams) for this graph can be generated with the following code:

```py {title="vending_machine_diagram.py" py="3.10"}
from vending_machine import InsertCoin, vending_machine_graph

vending_machine_graph.mermaid_code(start_node=InsertCoin)
```

The diagram generated by the above code is:

```mermaid
---
title: vending_machine_graph
---
stateDiagram-v2
  [*] --> InsertCoin
  InsertCoin --> CoinsInserted
  CoinsInserted --> SelectProduct
  CoinsInserted --> Purchase
  SelectProduct --> Purchase
  Purchase --> InsertCoin
  Purchase --> SelectProduct
  Purchase --> [*]
```

See [below](#mermaid-diagrams) for more information on generating diagrams.

## GenAI Example

So far we haven't shown an example of a Graph that actually uses PydanticAI or GenAI at all.

In this example, one agent generates a welcome email to a user and the other agent provides feedback on the email.

This graph has a very simple structure:

```mermaid
---
title: feedback_graph
---
stateDiagram-v2
  [*] --> WriteEmail
  WriteEmail --> Feedback
  Feedback --> WriteEmail
  Feedback --> [*]
```

```python {title="genai_email_feedback.py" py="3.10"}
from __future__ import annotations as _annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, EmailStr

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]


@dataclass
class Email:
    subject: str
    body: str


@dataclass
class State:
    user: User
    write_agent_messages: list[ModelMessage] = field(default_factory=list)


email_writer_agent = Agent(
    'google-vertex:gemini-1.5-pro',
    result_type=Email,
    system_prompt='Write a welcome email to our tech blog.',
)


@dataclass
class WriteEmail(BaseNode):
    state: State
    email_feedback: str | None = None

    async def run(self, ctx: GraphRunContext) -> Feedback:
        if self.email_feedback:
            prompt = (
                f'Rewrite the email for the user:\n'
                f'{format_as_xml(self.state.user)}\n'
                f'Feedback: {self.email_feedback}'
            )
        else:
            prompt = (
                f'Write a welcome email for the user:\n'
                f'{format_as_xml(self.state.user)}'
            )

        result = await email_writer_agent.run(
            prompt,
            message_history=self.state.write_agent_messages,
        )
        self.state.write_agent_messages += result.all_messages()
        return Feedback(self.state, result.data)


class EmailRequiresWrite(BaseModel):
    feedback: str


class EmailOk(BaseModel):
    pass


feedback_agent = Agent[None, EmailRequiresWrite | EmailOk](
    'openai:gpt-4o',
    result_type=EmailRequiresWrite | EmailOk,  # type: ignore
    system_prompt=(
        'Review the email and provide feedback, email must reference the users specific interests.'
    ),
)


@dataclass
class Feedback(BaseNode[None, Email]):
    state: State
    email: Email

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> WriteEmail | End[Email]:
        prompt = format_as_xml({'user': self.state.user, 'email': self.email})
        result = await feedback_agent.run(prompt)
        if isinstance(result.data, EmailRequiresWrite):
            return WriteEmail(self.state, email_feedback=result.data.feedback)
        else:
            return End(self.email)


async def main():
    user = User(
        name='John Doe',
        email='john.joe@example.com',
        interests=['Haskel', 'Lisp', 'Fortran'],
    )
    state = State(user)
    feedback_graph = Graph(nodes=(WriteEmail, Feedback))
    email, _ = await feedback_graph.run(WriteEmail(state))
    print(email)
    """
    Email(
        subject='Welcome to our tech blog!',
        body='Hello John, Welcome to our tech blog! ...',
    )
    """
```

_(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add `asyncio.run(main())` to run `main`)_

## Custom Control Flow

In many real-world applications, Graphs cannot run uninterrupted from start to finish — they might require external input, or run over an extended period of time such that a single process cannot execute the entire graph run from start to finish without interruption.

In these scenarios the [`next`][pydantic_graph.graph.Graph.next] method can be used to run the graph one node at a time.

In this example, an AI asks the user a question, the user provides an answer, the AI evaluates the answer and ends if the user got it right or asks another question if they got it wrong.

??? example "`ai_q_and_a_graph.py` — `question_graph` definition"
    ```python {title="ai_q_and_a_graph.py" noqa="I001" py="3.10"}
    from __future__ import annotations as _annotations

    from dataclasses import dataclass, field

    from pydantic_graph import BaseNode, End, Graph, GraphRunContext

    from pydantic_ai import Agent
    from pydantic_ai.format_as_xml import format_as_xml
    from pydantic_ai.messages import ModelMessage

    ask_agent = Agent('openai:gpt-4o', result_type=str)


    @dataclass
    class QuestionState:
        question: str | None = None
        ask_agent_messages: list[ModelMessage] = field(default_factory=list)
        evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


    @dataclass
    class Ask(BaseNode):
        state: QuestionState = field(repr=False)

        async def run(self, ctx: GraphRunContext) -> Answer:
            result = await ask_agent.run(
                'Ask a simple question with a single correct answer.',
                message_history=self.state.ask_agent_messages,
            )
            self.state.ask_agent_messages += result.all_messages()
            self.state.question = result.data
            return Answer(self.state, result.data)


    @dataclass
    class Answer(BaseNode):
        state: QuestionState = field(repr=False)
        question: str
        answer: str | None = None

        async def run(self, ctx: GraphRunContext) -> Evaluate:
            assert self.answer is not None
            return Evaluate(self.state, self.answer)


    @dataclass
    class EvaluationResult:
        correct: bool
        comment: str


    evaluate_agent = Agent(
        'openai:gpt-4o',
        result_type=EvaluationResult,
        system_prompt='Given a question and answer, evaluate if the answer is correct.',
    )


    @dataclass
    class Evaluate(BaseNode):
        state: QuestionState = field(repr=False)
        answer: str

        async def run(
            self,
            ctx: GraphRunContext,
        ) -> End[str] | Reprimand:
            assert self.state.question is not None
            result = await evaluate_agent.run(
                format_as_xml({'question': self.state.question, 'answer': self.answer}),
                message_history=self.state.evaluate_agent_messages,
            )
            self.state.evaluate_agent_messages += result.all_messages()
            if result.data.correct:
                return End(result.data.comment)
            else:
                return Reprimand(self.state, result.data.comment)


    @dataclass
    class Reprimand(BaseNode):
        state: QuestionState = field(repr=False)
        comment: str

        async def run(self, ctx: GraphRunContext) -> Ask:
            print(f'Comment: {self.comment}')
            self.state.question = None
            return Ask(self.state)


    question_graph = Graph(nodes=(Ask, Answer, Evaluate, Reprimand))
    ```

    _(This example is complete, it can be run "as is" with Python 3.10+)_


```python {title="ai_q_and_a_run.py" noqa="I001" py="3.10"}
from rich.prompt import Prompt

from pydantic_graph import End, HistoryStep

from ai_q_and_a_graph import Ask, question_graph, QuestionState, Answer


async def main():
    state = QuestionState()  # (1)!
    node = Ask(state)  # (2)!
    history: list[HistoryStep[QuestionState]] = []  # (3)!
    while True:
        node = await question_graph.next(node, history)  # (4)!
        if isinstance(node, Answer):
            node.answer = Prompt.ask(node.question)  # (5)!
        elif isinstance(node, End):  # (6)!
            print(f'Correct answer! {node.data}')
            #> Correct answer! Well done, 1 + 1 = 2
            print([e.data for e in history])
            """
            [
                Ask(),
                Answer(question='What is the capital of France?', answer='Vichy'),
                Evaluate(answer='Vichy'),
                Reprimand(comment='Vichy is no longer the capital of France.'),
                Ask(),
                Answer(question='what is 1 + 1?', answer='2'),
                Evaluate(answer='2'),
            ]
            """
            return
        # otherwise just continue
```

1. Create the state object which will be mutated by [`next`][pydantic_graph.graph.Graph.next].
2. The start node is `Ask` but will be updated by [`next`][pydantic_graph.graph.Graph.next] as the graph runs.
3. The history of the graph run is stored in a list of [`HistoryStep`][pydantic_graph.state.HistoryStep] objects. Again [`next`][pydantic_graph.graph.Graph.next] will update this list in place.
4. [Run][pydantic_graph.graph.Graph.next] the graph one node at a time, updating the state, current node and history as the graph runs.
5. If the current node is an `Answer` node, prompt the user for an answer.
6. Since we're using [`next`][pydantic_graph.graph.Graph.next] we have to manually check for an [`End`][pydantic_graph.nodes.End] and exit the loop if we get one.

_(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add `asyncio.run(main())` to run `main`)_

A [mermaid diagram](#mermaid-diagrams) for this graph can be generated with the following code:

```py {title="ai_q_and_a_diagram.py" py="3.10"}
from ai_q_and_a_graph import Ask, question_graph

question_graph.mermaid_code(start_node=Ask)
```

```mermaid
---
title: question_graph
---
stateDiagram-v2
  [*] --> Ask
  Ask --> Answer
  Answer --> Evaluate
  Evaluate --> Reprimand
  Evaluate --> [*]
  Reprimand --> Ask
```

You maybe have noticed that although this examples transfers control flow out of the graph run, we're still using [rich's `Prompt.ask`][rich.prompt.PromptBase.ask] to get user input, with the process hanging while we wait for the user to enter a response. For an example of genuine out-of-process control flow, see the [question graph example](examples/question-graph.md).

## Dependency Injection

As with PydanticAI, `pydantic-graph` supports dependency injection via a generic parameter on [`Graph`][pydantic_graph.graph.Graph] and [`BaseNode`][pydantic_graph.nodes.BaseNode], and the [`GraphRunContext.deps`][pydantic_graph.nodes.GraphRunContext.deps] fields.

As an example of dependency injection, let's modify the `DivisibleBy5` example [above](#graph) to use a [`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor] to run the compute load in a separate process (this is a contrived example, `ProcessPoolExecutor` wouldn't actually improve performance in this example):

```py {title="deps_example.py" py="3.10" test="skip" hl_lines="4 10-12 35-38 49-50"}
from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class GraphDeps:
    executor: ProcessPoolExecutor


@dataclass
class DivisibleBy5(BaseNode[GraphDeps, int]):
    foo: int

    async def run(
        self,
        ctx: GraphRunContext[GraphDeps],
    ) -> Increment | End[int]:
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return Increment(self.foo)


@dataclass
class Increment(BaseNode):
    foo: int

    async def run(self, ctx: GraphRunContext[GraphDeps]) -> DivisibleBy5:
        loop = asyncio.get_running_loop()
        compute_result = await loop.run_in_executor(
            ctx.deps.executor,
            self.compute,
        )
        return DivisibleBy5(compute_result)

    def compute(self) -> int:
        return self.foo + 1


fives_graph = Graph(nodes=[DivisibleBy5, Increment])


async def main():
    with ProcessPoolExecutor() as executor:
        deps = GraphDeps(executor)
        result, history = await fives_graph.run(DivisibleBy5(3), deps=deps)
    print(result)
    #> 5
    # the full history is quite verbose (see below), so we'll just print the summary
    print([item.data for item in history])
    """
    [
        DivisibleBy5(foo=3),
        Increment(foo=3),
        DivisibleBy5(foo=4),
        Increment(foo=4),
        DivisibleBy5(foo=5),
        End(data=5),
    ]
    """
```

_(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add `asyncio.run(main())` to run `main`)_

## Mermaid Diagrams

Pydantic Graph can generate [mermaid](https://mermaid.js.org/) [`stateDiagram-v2`](https://mermaid.js.org/syntax/stateDiagram.html) diagrams for graphs, as shown above.

These diagrams can be generated with:

* [`Graph.mermaid_code`][pydantic_graph.graph.Graph.mermaid_code] to generate the mermaid code for a graph
* [`Graph.mermaid_image`][pydantic_graph.graph.Graph.mermaid_image] to generate an image of the graph using [mermaid.ink](https://mermaid.ink/)
* [`Graph.mermaid_save`][pydantic_graph.graph.Graph.mermaid_save] to generate an image of the graph using [mermaid.ink](https://mermaid.ink/) and save it to a file

Beyond the diagrams shown above, you can also customize mermaid diagrams with the following options:

* [`Edge`][pydantic_graph.nodes.Edge] allows you to apply a label to an edge
* [`BaseNode.docstring_notes`][pydantic_graph.nodes.BaseNode.docstring_notes] and [`BaseNode.get_note`][pydantic_graph.nodes.BaseNode.get_note] allows you to add notes to nodes
* The [`highlighted_nodes`][pydantic_graph.graph.Graph.mermaid_code] parameter allows you to highlight specific node(s) in the diagram

Putting that together, we can edit the last [`ai_q_and_a_graph.py`](#custom-control-flow) example to:

* add labels to some edges
* add a note to the `Ask` node
* highlight the `Answer` node
* save the diagram as a `PNG` image to file

```python {title="ai_q_and_a_graph_extra.py" test="skip" lint="skip" hl_lines="2 4 10-11 14 26 31"}
...
from typing import Annotated

from pydantic_graph import BaseNode, End, GraphRunContext, Edge

...

@dataclass
class Ask(BaseNode):
    """Generate question using GPT-4o."""
    docstring_notes = True
    async def run(
        self, ctx: GraphRunContext
    ) -> Annotated[Answer, Edge(label='Ask the question')]:
        ...

...

@dataclass
class Evaluate(BaseNode):
    answer: str

    async def run(
            self,
            ctx: GraphRunContext,
    ) -> Annotated[End[str], Edge(label='success')] | Reprimand:
        ...

...

question_graph.mermaid_save('image.png', highlighted_nodes=[Answer])
```

_(This example is not complete and cannot be run directly)_

Would generate and image that looks like this:

```mermaid
---
title: question_graph
---
stateDiagram-v2
  Ask --> Answer: Ask the question
  note right of Ask
    Judge the answer.
    Decide on next step.
  end note
  Answer --> Evaluate
  Evaluate --> Reprimand
  Evaluate --> [*]: success
  Reprimand --> Ask

classDef highlighted fill:#fdff32
class Answer highlighted
```

### Setting Direction of the State Diagram

You can specify the direction of the state diagram using one of the following values:

- `'TB'`: Top to bottom, the diagram flows vertically from top to bottom.
- `'LR'`: Left to right, the diagram flows horizontally from left to right.
- `'RL'`: Right to left, the diagram flows horizontally from right to left.
- `'BT'`: Bottom to top, the diagram flows vertically from bottom to top.

Here is an example of how to do this using 'Left to Right' (LR) instead of the default 'Top to Bottom' (TB)
```py {title="vending_machine_diagram.py" py="3.10"}
from vending_machine import InsertCoin, vending_machine_graph

vending_machine_graph.mermaid_code(start_node=InsertCoin, direction='LR')
```

```mermaid
---
title: vending_machine_graph
---
stateDiagram-v2
  direction LR
  [*] --> InsertCoin
  InsertCoin --> CoinsInserted
  CoinsInserted --> SelectProduct
  CoinsInserted --> Purchase
  SelectProduct --> Purchase
  Purchase --> InsertCoin
  Purchase --> SelectProduct
  Purchase --> [*]
```
