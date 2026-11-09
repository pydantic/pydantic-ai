---
title: Teams
description: "A team of Pydantic AI agents: a lead that spawns named teammates, each on its own model, a shared task board with atomic claims, messages between them, and gates that send work back with feedback."
---

# Teams

`Team` runs a team of agents in one conversation: a lead that spawns named, long-lived teammates, a shared board they claim tasks from, messages between them, and gates that check their work. Each teammate can run on its own model, so the lead can plan on a large model while its teammates work on smaller, faster ones.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/teams/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

`Team` composes two primitives: a [`DelegationTasks`](subagents.md#team-members) owner, which runs the teammates, and a [`ClaimablePlanStore`](planning.md#shared-boards-with-claims), which holds the board. The application wires them up once:

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import Team
from pydantic_ai_harness.planning import InMemoryClaimablePlanStore
from pydantic_ai_harness.subagents import DelegationTask, DelegationTaskEvent, DelegationTasks

LEAD_MODEL = 'test'  # The lead's model ID.
MODELS = {'test': 'test'}  # How the application resolves a teammate's model ID.


async def run_member(record: DelegationTask) -> str:
    """Run one turn of a teammate: on its pinned model, or else on the lead's."""
    member = team.member_run(record, self_agent=lead)
    result = await member.run(model=MODELS[member.model] if member.model else LEAD_MODEL)
    return str(result.output)


async def observe(update: DelegationTaskEvent) -> None:
    await team.observe(update)


owner = DelegationTasks(member_runner=run_member, observer=observe)
team = Team(
    owner,
    board=InMemoryClaimablePlanStore(),
    models={'self': 'test'},  # The model a teammate of each agent type runs on unless its spawn names one.
    member_usage_limits=UsageLimits(request_limit=50),
)
owner.claims = team.claims
lead = Agent(LEAD_MODEL, capabilities=[team])


async def main() -> None:
    async with owner.opened():
        with owner.bind():
            await lead.run('Split the migration into tasks, and start two teammates on them.', conversation_id='c')
```

- **`run_member`** runs each turn of a teammate, the first and every wake, with `team.member_run`. That adds the team to the teammate's run, and `member.model` is the model ID the teammate is pinned to. A teammate never borrows the run that started it, so one reloaded after a restart can still be woken.
- **`observe`** keeps the roster's state current from the owner's events.
- **`owner.claims = team.claims`** lets the owner hand a failed or stopped teammate's board items back to `pending`.
- **The lead's run** happens inside `owner.bind()`, with a `conversation_id`. A conversation has one team.

## Tools

The lead and the teammates get different tools. A teammate's own sub-agents, and any other delegation, get none: teams don't nest.

| Tool | Lead | Teammate | What it does |
|---|---|---|---|
| `spawn_teammate(name, prompt, agent='self', model=None)` | ✓ | | Starts a named teammate in the background with its first task. It reports back when it goes idle |
| `board_add(content, depends_on=None)` | ✓ | ✓ | Adds an item to the board, `blocked` until the items it depends on are completed |
| `board_list()` | ✓ | ✓ | Lists every item with its status, owner and dependencies |
| `claim_task(item_id=None)` | | ✓ | Takes an item, or the next available one, atomically |
| `assign_task(item_id, teammate)` | ✓ | | Hands an item to a teammate, and messages it, so an idle teammate wakes to work on it |
| `finish_task(item_id, status='completed')` | ✓ | ✓ | Completes, cancels, or hands back (`pending`) an item in progress; the lead may finish anyone's. Completing an item unblocks its dependents in the same atomic step |
| `send_message(to, text)` | ✓ | ✓ | Messages a teammate by name, or the lead as `lead`. An idle teammate is woken with it |
| `shutdown_teammate(teammate, reason='')` | ✓ | | Asks a teammate to wrap up, then stops it when that turn ends; the items it still holds go back to the board |
| `request_plan_approval(plan)` | | ✓ | Submits a plan, and waits until it is approved or rejected with feedback |
| `review_plan(teammate, approved, feedback='')` | ✓ | | Approves or rejects the plan a teammate waits on; only with `plan_reviewer='lead'`, the default |

A teammate goes idle when its turn ends, and its final output reaches the lead's inbox. The lead's active run reads it when its current work is done. With `DelegationTasks(lead_waker=...)`, an idle lead gets one automated turn for a batch of reports, rather than one per teammate. Messages and reports from another agent arrive framed as data, never as the user's words, and can't grant the user's approval.

The tool names are fixed, so an agent has one team: two `Team`s on one agent over different owners or boards raise. The same `Team` on the agent and again for a run is one team. `Team` replaces the `TeamMessages` capability the owner adds to a teammate's run on its own, so don't bind both. Another capability that also offers `send_message` collides with `Team` on the tool name; wrap one of them in `PrefixTools` to keep both.

## Lifecycle

A teammate is `working` while a turn of it runs, `idle` once its last turn succeeded, `stopped` when the user stopped it or it was shut down, and `failed` when its last turn failed. A message wakes an idle or failed teammate; only the user's message wakes a stopped one.

### Queued turns

With `DelegationTasks(max_active=...)`, at most that many teammates run at once. A teammate started or woken beyond it is `queued`, and queued teammates start in the order they were queued, as turns free up. `owner.max_active` can be changed while the team runs: raising it starts queued teammates at once.

### Shutting a teammate down

The lead's `shutdown_teammate`, or `await team.shutdown(conversation_id, name, reason='')` for the user, asks a teammate to wrap up: it gets the request as a message, which wakes it if idle, and its instructions tell it to finish or hand back the item it holds, then end its turn. When that turn ends, the teammate is shut down:

- the board items it still holds go back to `pending`, each with a `BoardItemChangedEvent` to the observer;
- its report reaches the lead, as any turn's does;
- it is `stopped`, with `TeamMember.shutdown == 'done'`, which tells it apart from a teammate the user stopped (`shutdown is None`). Only the user's message wakes it again.

While it wraps up, `TeamMember.shutdown` is `'requested'`. A teammate that would get no turn to read the request, because it is queued or the user stopped it, is shut down at once. A wrap-up that takes longer than `Team(shutdown_grace=...)` seconds (120 by default) is stopped, and so is one a process exit interrupts.

### Plan approval

A teammate can submit a plan with `request_plan_approval(plan)` and wait for the answer before it acts, for example when its prompt says to plan first. With `Team(plan_reviewer='lead')`, the default, the lead gets the plan as a message and answers with `review_plan`; with `plan_reviewer='user'`, only the application does, through `await team.approve(conversation_id, name, feedback='')` or `await team.reject(conversation_id, name, feedback)`. The user can review either way. A rejection needs feedback, and the teammate revises its plan and submits it again.

The plan waiting for review is saved on the teammate's record and shown as `TeamMember.plan`, so it survives a restart. If the teammate's run ends while it waits, as on a restart, its review arrives as a message that wakes it: the user's own words for the user's review, or a message from the lead.

## Usage and budget

The team adds up what its runs use as they go: each teammate's turns on `TeamMember.usage`, saved with its record, and the lead's runs while it leads the team on `TeamSnapshot.lead_usage`, saved with the lead's inbox. `TeamSnapshot.usage`, or `team.usage(conversation_id)`, is the whole team's total, with each teammate counted on its own model. Each is a `RunUsage`: requests, tokens, and `cost` when the models' prices are known.

`Team(usage_limits=UsageLimits(...))` caps the whole team, the lead and every teammate together, and is checked before each of their model requests; a request over it fails the run that would make it with `UsageLimitExceeded`. `member_usage_limits` caps each teammate's turn on its own.

## Models

Each teammate runs on its own model, which it keeps for its whole life:

1. the `model` its spawn names, through `spawn_teammate(model=...)` or `team.spawn(..., model=...)`;
2. else `models[agent_type]`, the default for its agent type (`self` for the lead's own agent, or a name from the owner's `agents`; aliases work too);
3. else the lead's model, or, for a specialist whose agent has a model of its own, that model.

The model ID is saved on the teammate's `DelegationTask`, so it survives idle, wake and a restart, and the roster shows it. The application's runner resolves it, as it resolves any model ID. The lead's instructions list the agent types with their default models.

## Board

The board is a [`ClaimablePlanStore`](planning.md#shared-boards-with-claims) per conversation: pass one store, or `board=lambda conversation_id: ...` for one per conversation. Use a store that several processes can share, such as `FileClaimablePlanStore` or `SqliteClaimablePlanStore`, when teammates may run in more than one. Claims are atomic, so two teammates racing for one item get one winner.

Every board change a tool makes is emitted into that run's event stream as a `BoardItemChangedEvent`, with the item before and after from the same atomic store operation. A teammate's events reach the owner's observer as `DelegationTaskEvent.event`, like any task's.

## Events, observer and snapshot

| Event | Fields | Emitted when |
|---|---|---|
| `TeammateSpawnedEvent` | `member`, `prompt`, `truncated` | A teammate starts |
| `TeammateStateEvent` | `member`, `previous` | A teammate is queued, starts or finishes a turn, is stopped, or is asked to shut down or shut down |
| `TeamMessageEvent` | `message` | A message is sent |
| `PlanApprovalRequestedEvent` | `member`, `request_id`, `plan`, `truncated` | A teammate submits a plan |
| `PlanReviewedEvent` | `member`, `request_id`, `approved`, `feedback`, `reviewer`, `truncated` | A plan is approved or rejected |
| `BoardItemChangedEvent` | `item`, `previous` | A board item changes; `previous` is `None` for a new one |

A change a tool call makes is emitted into the run that made it. A change made outside any run goes to `Team(observer=...)` instead: teammates' state, claims the owner hands back, and what the application does through `team.spawn`, `team.send` (the user's message to a teammate), `team.stop`, `team.shutdown`, `team.approve` and `team.reject`. A UI starts from `await team.snapshot(conversation_id)`, a `TeamSnapshot` of the roster (`TeamMember`: name, agent type, model, state, current task, shutdown, plan waiting for review, and usage), the board, the latest messages and the team's usage, then reduces the events.

## Gates

A gate is any capability whose event listeners decide one of these events. Pass gates as `Team(gates=[...])`: the same instances check the lead's run and every teammate's, so a task the lead adds is checked exactly like one a teammate adds. Several gates can reject the same event; their feedback is passed on together, in the order they ran.

```python
from pydantic_ai import RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ToolCallPart
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
```

| Event | Emitted when | Fields | Decision | What a rejection does |
|---|---|---|---|---|
| `TeammateIdleEvent` | A teammate's run produced its final output and is about to end | `member`, `final_output`, `truncated` | `keep_working(feedback)` | The feedback becomes the teammate's next user message (enqueued with `asap`), so the run continues |
| `TeamTaskCreatingEvent` | `board_add` is about to add an item | `item` | `reject(feedback)` | The tool call becomes a tool retry carrying the feedback; the item is not added |
| `TeamTaskCompletingEvent` | `finish_task` is about to complete an item | `item`, `owner` | `reject(feedback)` | The tool call becomes a tool retry carrying the feedback; the item keeps its status |

The events are in the `teams` family (`TEAMS_EVENTS`) and are dispatched immediately, so the decision is made before the action goes ahead. `item` is an `OwnedPlanItem`, with the `owner` that holds it; the completing event's `owner` is who completes it, a teammate's name or `lead`. `final_output` is the teammate's output as text, cut at `MAX_EVENT_TEXT_CHARS` (4096, shared with [Subagents](subagents.md)) with `truncated` set.

Changes the application makes outside any run, such as the user's own board edits, are not gated: the user is the authority the gates protect. A board tool of your own can call `check_task_creating(ctx, item)` and `check_task_completing(ctx, item, owner=name)` before it writes, and is gated the same way.

### How the idle gate decides

The idle gate runs in a teammate's `after_node_run` when the run reaches its end. Content enqueued there with `asap` is turned into another model request by Pydantic AI's end-of-run drain, which is what keeps the teammate working.

- It does not fire while messages are queued for the run (from a background tool, a report, or another capability's `after_node_run`), because the run continues anyway. It is ordered outermost, so it sees what other capabilities enqueue in the same step.
- It does not fire when the run pauses for deferred tools or ends with a `PendingModelResponse`: the teammate is waiting, not idle.
- It fires again every time the teammate is about to go idle, so a gate that keeps rejecting keeps the teammate running until a usage limit stops it. Set `Team(member_usage_limits=...)`: a turn over it fails, and the teammate is not woken again until a new message arrives.
- `run_stream()` skips `after_node_run` for the final streamed response, so teammates run with `run()`, `iter()` or `run_stream_events()`, as `MemberRun.run` does.

Listeners see every `TeammateIdleEvent`. A stream consumer (`event_stream_handler`, `run_stream_events()`) sees one only when the gate kept the teammate working: Pydantic AI does not deliver capability events emitted in `after_node_run` of the step that ends the run to the event stream.

## Telemetry

Each gate rejection records a zero-duration `team gate rejected` span on the run's tracer, with `team.gate` (`teammate_idle`, `task_creating` or `task_completing`), `team.feedback_count`, `team.member` (the idle teammate, or who completes the item) and `team.item_id`. The feedback text is added as `team.feedback` only when the run's instrumentation includes content. Board, spawn and message tools record nothing beyond Pydantic AI's own tool spans.
