# Multi-Agent Applications Reference

Source: `pydantic_ai_slim/pydantic_ai/agent/`

## Architecture Patterns

### Agent Delegation (Tool-based)

```
┌─────────────────────────────────────────────────┐
│              Parent Agent                        │
│  ┌─────────────────────────────────────────┐    │
│  │ Instructions: "Use tools to complete..." │    │
│  └─────────────────────────────────────────┘    │
│                      │                           │
│         ┌────────────┼────────────┐             │
│         ▼            ▼            ▼             │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│    │ tool_a  │ │ tool_b  │ │delegate │         │
│    └─────────┘ └─────────┘ └────┬────┘         │
│                                  │              │
└──────────────────────────────────│──────────────┘
                                   │
                                   ▼
                          ┌───────────────┐
                          │ Child Agent   │
                          │  (returns to  │
                          │   parent)     │
                          └───────────────┘
```

### Router Pattern

```
                    User Request
                          │
                          ▼
                 ┌────────────────┐
                 │  Router Agent  │  Classifies request
                 └───────┬────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │  Billing   │ │ Technical  │ │  General   │
   │   Agent    │ │   Agent    │ │   Agent    │
   └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
                        ▼
                    Response
```

### Programmatic Hand-off

```
    Application Code
          │
    ┌─────┴─────┐
    ▼           │
┌────────┐      │
│Agent 1 │──────┤  Result 1
└────────┘      │
                ▼
          ┌────────┐
          │Agent 2 │──────┐  Result 2
          └────────┘      │
                          ▼
                    ┌────────┐
                    │Agent 3 │  Final Result
                    └────────┘
```

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

Autonomous agents that can plan, execute, and verify complex tasks:

| Capability | Implementation |
|------------|----------------|
| Planning & progress | Task management toolsets |
| File operations | File ops toolsets |
| Task delegation | Agent delegation (above) |
| Code execution | Sandboxed execution toolsets |
| Context management | Message history summarization |
| Human-in-the-loop | Tool approval workflows |
| Durable execution | Temporal, DBOS, Prefect |

### Building a Deep Agent

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import ApprovalRequiredToolset, FilteredToolset

# 1. Core agent with basic capabilities
agent = Agent(
    'openai:gpt-5',
    instructions='''
    You are an autonomous assistant that can:
    - Read and write files
    - Execute code in a sandbox
    - Search the web
    - Delegate to specialist agents

    Always verify your work before reporting completion.
    ''',
)

# 2. Add file operations (with approval for writes)
file_toolset = FileOperationsToolset()
safe_file_toolset = file_toolset.approval_required(
    lambda ctx, td, args: td.name.startswith('write_')
)

# 3. Add code execution (sandboxed)
code_toolset = SandboxedCodeExecutionToolset()

# 4. Combine with history processing for long conversations
from pydantic_ai.history_processors import SummarizingHistoryProcessor

agent = Agent(
    'openai:gpt-5',
    toolsets=[safe_file_toolset, code_toolset],
    history_processors=[SummarizingHistoryProcessor(max_tokens=8000)],
)
```

### Deep Agent Checklist

- [ ] **Planning**: Agent can break tasks into steps
- [ ] **Verification**: Agent validates its own output
- [ ] **Recovery**: Agent handles errors gracefully
- [ ] **Safety**: Dangerous operations require approval
- [ ] **Context**: Long conversations are summarized
- [ ] **Observability**: Full tracing enabled

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

## Agentic Workflow Patterns

### Router/Triage Agent Pattern

A router agent classifies requests and delegates to specialists:

```python {title="router_pattern.py" test="skip"}
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext


class RouteDecision(BaseModel):
    topic: str  # 'billing', 'technical', 'general'
    confidence: float


router_agent = Agent('openai:gpt-5', output_type=RouteDecision)
billing_agent = Agent('openai:gpt-5', instructions='You handle billing questions.')
technical_agent = Agent('openai:gpt-5', instructions='You handle technical questions.')


async def handle_request(user_query: str) -> str:
    # Router decides which specialist
    route = await router_agent.run(f'Classify this query: {user_query}')

    # Delegate to specialist
    if route.output.topic == 'billing':
        result = await billing_agent.run(user_query)
    elif route.output.topic == 'technical':
        result = await technical_agent.run(user_query)
    else:
        result = await Agent('openai:gpt-5').run(user_query)

    return result.output
```

### Reflection/Self-Critique Pattern

An agent checks and improves its own output:

```python {title="reflection_pattern.py" test="skip"}
from pydantic import BaseModel

from pydantic_ai import Agent


class ReviewResult(BaseModel):
    is_good: bool
    feedback: str


generator_agent = Agent('openai:gpt-5')
reviewer_agent = Agent('openai:gpt-5', output_type=ReviewResult)


async def generate_with_reflection(prompt: str, max_iterations: int = 3) -> str:
    result = await generator_agent.run(prompt)
    output = result.output

    for _ in range(max_iterations):
        # Self-critique
        review = await reviewer_agent.run(
            f'Review this output for quality and accuracy:\n\n{output}'
        )

        if review.output.is_good:
            break

        # Improve based on feedback
        result = await generator_agent.run(
            f'Improve this based on feedback:\n\nOriginal: {output}\n\nFeedback: {review.output.feedback}'
        )
        output = result.output

    return output
```

### Plan-Execute-Verify Pattern

Break complex tasks into steps with verification:

```python {title="plan_execute_pattern.py" test="skip"}
from pydantic import BaseModel

from pydantic_ai import Agent


class Plan(BaseModel):
    steps: list[str]


class StepResult(BaseModel):
    success: bool
    output: str


planner_agent = Agent('openai:gpt-5', output_type=Plan)
executor_agent = Agent('openai:gpt-5', output_type=StepResult)
verifier_agent = Agent('openai:gpt-5', output_type=bool)


async def plan_and_execute(task: str) -> str:
    # Plan
    plan = await planner_agent.run(f'Create a step-by-step plan for: {task}')

    results = []
    for step in plan.output.steps:
        # Execute
        result = await executor_agent.run(f'Execute this step: {step}')
        results.append(result.output)

        # Verify
        verification = await verifier_agent.run(
            f'Did this step succeed?\nStep: {step}\nResult: {result.output}'
        )
        if not verification.output:
            # Handle failure (retry, rollback, etc.)
            break

    return '\n'.join([r.output for r in results])
```

### Combining Patterns with Usage Limits

```python
from pydantic_ai import RunUsage, UsageLimits

usage = RunUsage()
limits = UsageLimits(request_limit=20, total_tokens_limit=10000)

# All agents share usage tracking
result1 = await agent1.run(prompt, usage=usage, usage_limits=limits)
result2 = await agent2.run(prompt, usage=usage, usage_limits=limits)

# Check combined usage
print(f'Total tokens used: {usage.total_tokens}')
```

## Key Patterns

| Pattern | Use When |
|---------|----------|
| Delegation via tools | Child returns to parent |
| Output functions | Permanent hand-off |
| Programmatic hand-off | App logic between agents |
| Graph control | Complex state machines |
| Router/Triage | Classify and route to specialists |
| Reflection | Self-improvement on output |
| Plan-Execute-Verify | Multi-step with validation |

## See Also

- [graph.md](graph.md) — Graph-based state machines
- [dependencies.md](dependencies.md) — Sharing dependencies
- [output.md](output.md) — Output functions
- [observability.md](observability.md) — Logfire tracing
- [messages.md](messages.md) — History processors for long conversations
