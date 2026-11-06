# Teams

`Team` checks the work of a team of agents with gates: when a member is about to go idle, when a task is added to the team's board, and when a task is marked completed. Use it when a lead agent and its members should all follow the same rules, such as "end with a summary" or "run the tests before completing a task", and a gate that rejects should send the agent back to work with feedback.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/teams/)

Install the OpenAI provider before running this example:

uv:

```bash
uv add "pydantic-ai-slim[openai]" pydantic-ai-harness
```

pip:

```bash
pip install "pydantic-ai-slim[openai]" pydantic-ai-harness
```

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness import Team
from pydantic_ai_harness.teams import TeammateIdleEvent, TeamTaskCompletingEvent, TeamTaskCreatingEvent

gates = Hooks()


@gates.on.event(TeammateIdleEvent)
async def require_summary(ctx: RunContext[None], event: TeammateIdleEvent) -> None:
    if 'Summary:' not in event.final_output:
        event.keep_working('End with a line starting "Summary:" that says what you changed.')


@gates.on.event(TeamTaskCreatingEvent)
async def require_done_criteria(ctx: RunContext[None], event: TeamTaskCreatingEvent) -> None:
    if 'done when' not in event.item.content.lower():
        event.reject('Say when the task is done, for example "... done when the tests pass".')


@gates.on.event(TeamTaskCompletingEvent)
async def require_tests(ctx: RunContext[None], event: TeamTaskCompletingEvent) -> None:
    ran_tests = any(
        isinstance(part, ToolCallPart) and part.tool_name == 'run_tests'
        for message in ctx.messages
        for part in message.parts
    )
    if not ran_tests:
        event.reject('Run the tests before completing a task.')


team = Team(gates=[gates])

lead = Agent('openai:gpt-5.6-sol', capabilities=[team])
member = Agent('openai:gpt-5.6-sol')


async def run_member(name: str, prompt: str) -> str:
    result = await member.run(prompt, capabilities=[team.member_capability(name)])
    return result.output
```

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Gates

A gate is any capability whose event listeners decide one of these events. Each is dispatched immediately, so the decision is made before the action goes ahead. Several gates can reject the same event; their feedback is passed on together, in the order they ran.

| Event | Emitted when | Fields | Decision | What a rejection does |
|---|---|---|---|---|
| `TeammateIdleEvent` | A member's run produced its final output and is about to end | `member`, `final_output`, `truncated` | `keep_working(feedback)` | The feedback becomes the member's next user message (enqueued with `asap`), so the run continues |
| `TeamTaskCreatingEvent` | A tool call adds an item to the board | `item` | `reject(feedback)` | The tool call becomes a tool retry carrying the feedback; the item is not added |
| `TeamTaskCompletingEvent` | A tool call marks an item completed | `item`, `owner` | `reject(feedback)` | The tool call becomes a tool retry carrying the feedback; the item keeps its status |

The events are in the `teams` family (`TEAMS_EVENTS`). `item` is a `PlanItem` from [Planning](../planning/). `final_output` is the member's output as text, cut at `MAX_EVENT_TEXT_CHARS` (4096, shared with [Subagents](../subagents/)) with `truncated` set.

## Where gates run

- **The lead's run** gets the gates from `Team` itself.
- **Each member's run** gets them from `team.member_capability(name)`, which also adds the idle gate for that member. Pass it to the member's run, the way `SubAgents(shared_capabilities=[...])` shares a capability with its delegates. The same gate instances serve the lead and every member, so a task the lead adds is checked exactly like one a member adds.
- **Changes made outside any run are not gated.** The user, or the application acting for them, is the authority the gates protect, so board edits and messages it makes directly do not go through them.

## Gating board tools

A board tool calls `check_task_creating(ctx, item)` before it stores a new item, and `check_task_completing(ctx, item, owner=name)` before it stores a `completed` status. Each emits its event and raises `ModelRetry` with the gates' feedback when one rejects:

```python
from pydantic_ai import RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai_harness.planning import PlanItem
from pydantic_ai_harness.teams import check_task_creating

board: list[PlanItem] = []


async def add_task(ctx: RunContext[None], content: str) -> str:
    """Add a task to the team board."""
    item = PlanItem(content=content)
    await check_task_creating(ctx, item)
    board.append(item)
    return f'Added task {item.id}.'


board_tools = Capability(tools=[add_task])
```

The tool has to be contributed by a capability (here `Capability(tools=...)`), because Pydantic AI accepts capability events only from capability hooks and capability-contributed tools.

## How the idle gate decides

The idle gate runs in a member's `after_node_run` when the run reaches its end. Content enqueued there with `asap` is turned into another model request by Pydantic AI's end-of-run drain, which is what keeps the member working.

- It does not fire while messages are queued for the run (from a background tool, a report, or another capability's `after_node_run`), because the run continues anyway. It is ordered outermost, so it sees what other capabilities enqueue in the same step.
- It does not fire when the run pauses for deferred tools or ends with a `PendingModelResponse`: the member is waiting, not idle.
- It fires again every time the member is about to go idle, so a gate that keeps rejecting keeps the member running until a usage limit stops it. Give member runs `UsageLimits`.
- `run_stream()` skips `after_node_run` for the final streamed response, so the idle gate needs the member to run with `run()`, `iter()` or `run_stream_events()`.

Listeners see every `TeammateIdleEvent`. A stream consumer (`event_stream_handler`, `run_stream_events()`) sees one only when the gate kept the member working: Pydantic AI does not deliver capability events emitted in `after_node_run` of the step that ends the run to the event stream.

## Telemetry

Each rejection records a zero-duration `team gate rejected` span on the run's tracer, with `team.gate` (`teammate_idle`, `task_creating` or `task_completing`), `team.feedback_count`, `team.member` (the idle member or the completing owner) and `team.item_id`. The feedback text is added as `team.feedback` only when the run's instrumentation includes content. Gates that let the action go ahead record nothing beyond Pydantic AI's own spans.
