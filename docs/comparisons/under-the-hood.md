# The numbers people quote about us

A few numbers circulate about Pydantic AI. They describe real mechanisms. This is what those
mechanisms do once the agent is in an application.

## "Creating an agent is much slower"

Agno constructs an agent with less work at `Agent(...)`. Pydantic AI resolves the model, checks
config and credentials, and builds tool schemas there. That runs once per process, at startup.

A bad model name fails at that call, not on the first customer request:

```python {title="config_fails_here.py"}
from pydantic_ai import Agent

try:
    Agent('does-not-exist:gpt-4')
except Exception as exc:
    print(f'{type(exc).__name__}: {str(exc)[:120]}')
    """
    UserError: Unknown model: does-not-exist:gpt-4. Did you mean 'openai-chat:gpt-4'?
    """
```

Skip that work and construction is faster; the same typo lands on a live request. Either way the
number is small next to a model call.

## "It's four lines to a working agent"

Ours is about that long too. Four lines get a reply. Secrets the model shouldn't see, a spend cap, a
stop that leaves you the conversation, evals, and crash recovery are extra in every framework,
including this one.

## "Checkpointed at every step"

LangGraph writes the graph state at every step. That is how time travel and forking work; we don't
have those. The checkpoint grows with what you carry, and crash recovery means writing the agent as
a graph.

Here the run is a coroutine. [Durability is a capability](../durable_execution/overview.md) on the
same [`Agent`][pydantic_ai.Agent], usually Temporal, DBOS, or Prefect, which is the engine the rest
of the application already runs.

## "Memory that just works"

Automatic compression is extra model calls, whether or not they show up in the usage you look at.
Ours is history processors and dependencies you attach. If memory is the product, [Mastra](vs-mastra.md)
and [Agno](vs-agno.md) ship more of it.
