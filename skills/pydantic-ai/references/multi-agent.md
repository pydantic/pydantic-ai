# Multi-Agent Applications Reference

Source: `pydantic_ai_slim/pydantic_ai/agent/`

## Complexity Levels

1. **Single agent** — standard agent usage
2. **Agent delegation** — agents call other agents via tools
3. **Programmatic hand-off** — application code orchestrates agents
4. **Graph-based control** — state machine for complex flows
5. **Deep agents** — autonomous planning, file ops, sandboxed execution

## Agent Delegation

An agent calls another agent from within a tool:

```python {title="agent_delegation.py"}
from pydantic_ai import Agent, RunContext, UsageLimits

# Parent agent
selector_agent = Agent(
    'openai:gpt-5',
    instructions='Use joke_factory to get jokes, then pick the best one.',
)

# Delegate agent
generator_agent = Agent('google-gla:gemini-2.5-flash', output_type=list[str])


@selector_agent.tool
async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:
    """Generate jokes using the delegate agent."""
    result = await generator_agent.run(
        f'Generate {count} jokes.',
        usage=ctx.usage,  # Pass usage for combined tracking
    )
    return result.output


result = selector_agent.run_sync(
    'Tell me a joke.',
    usage_limits=UsageLimits(request_limit=5, total_tokens_limit=500),
)
print(result.output)
print(result.usage())  # Combined usage from both agents
```

### Key Points

- Pass `ctx.usage` to delegate for combined usage tracking
- Different models can be used per agent
- `UsageLimits` applies across all delegated calls
- Agents are stateless — no need to include in dependencies

### Delegation with Shared Dependencies

```python {title="delegation_with_deps.py"}
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, RunContext


@dataclass
class Deps:
    http_client: httpx.AsyncClient
    api_key: str


parent_agent = Agent('openai:gpt-5', deps_type=Deps)
child_agent = Agent('google-gla:gemini-2.5-flash', deps_type=Deps, output_type=list[str])


@parent_agent.tool
async def delegate_task(ctx: RunContext[Deps], query: str) -> list[str]:
    result = await child_agent.run(
        query,
        deps=ctx.deps,   # Pass dependencies
        usage=ctx.usage,
    )
    return result.output


@child_agent.tool
async def fetch_data(ctx: RunContext[Deps], endpoint: str) -> str:
    response = await ctx.deps.http_client.get(
        endpoint,
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )
    return response.text
```

## Programmatic Hand-off

Application code orchestrates multiple agents in sequence:

```python {title="programmatic_handoff.py" test="skip"}
from pydantic import BaseModel

from pydantic_ai import Agent, RunUsage, UsageLimits


class FlightDetails(BaseModel):
    flight_number: str


class SeatPreference(BaseModel):
    row: int
    seat: str


class Failed(BaseModel):
    """Unable to find a satisfactory choice."""


flight_agent = Agent[None, FlightDetails | Failed](
    'openai:gpt-5',
    output_type=FlightDetails | Failed,  # type: ignore
)

seat_agent = Agent[None, SeatPreference | Failed](
    'openai:gpt-5',
    output_type=SeatPreference | Failed,  # type: ignore
)

usage_limits = UsageLimits(request_limit=15)


async def book_flight():
    usage = RunUsage()

    # First agent: find flight
    result = await flight_agent.run(
        'Find a flight from NYC to LA',
        usage=usage,
        usage_limits=usage_limits,
    )
    if isinstance(result.output, Failed):
        return 'Could not find flight'

    flight = result.output

    # Second agent: choose seat
    result = await seat_agent.run(
        f'Choose a window seat for flight {flight.flight_number}',
        usage=usage,
        usage_limits=usage_limits,
    )
    if isinstance(result.output, Failed):
        return f'Booked {flight.flight_number}, no seat preference'

    seat = result.output
    return f'Booked {flight.flight_number}, seat {seat.row}{seat.seat}'
```

## Output Functions for Hand-off

Use output functions for permanent hand-off (no return to parent):

```python
from pydantic_ai import Agent


def route_to_specialist(topic: str) -> str:
    """Route to a specialist agent based on topic."""
    if topic == 'billing':
        return billing_agent.run_sync('Handle billing query').output
    return support_agent.run_sync('Handle general query').output


agent = Agent(
    'openai:gpt-5',
    output_type=str,
    output_functions=[route_to_specialist],
)
```

## Deep Agents

Combine patterns for autonomous agents:

| Capability | Implementation |
|------------|----------------|
| Planning & progress | Task management toolsets |
| File operations | File ops toolsets |
| Task delegation | Agent delegation (above) |
| Code execution | Sandboxed execution toolsets |
| Context management | Message history summarization |
| Human-in-the-loop | Tool approval workflows |
| Durable execution | Temporal, DBOS, Prefect |

Community package: [`pydantic-deep`](https://github.com/vstorm-co/pydantic-deepagents)

## Observability

Multi-agent systems benefit from tracing:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

# Traces show:
# - Which agent handled which part
# - Delegation decisions and timing
# - Token usage per agent
# - Tool calls within each agent
```

## Key Patterns

| Pattern | Use When |
|---------|----------|
| Delegation via tools | Child returns to parent |
| Output functions | Permanent hand-off |
| Programmatic hand-off | App logic between agents |
| Graph control | Complex state machines |

## See Also

- [graph.md](graph.md) — Graph-based state machines
- [dependencies.md](dependencies.md) — Sharing dependencies
- [output.md](output.md) — Output functions
- [observability.md](observability.md) — Logfire tracing
