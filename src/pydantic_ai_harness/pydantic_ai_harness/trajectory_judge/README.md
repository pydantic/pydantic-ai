# Trajectory Judge

> [!NOTE]
> The verdict types are not re-exported at the top level -- import them from the submodule:
>
> ```python
> from pydantic_ai_harness import TrajectoryJudge
> from pydantic_ai_harness.trajectory_judge import AllGood, Steer
> ```
>
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

Watch a live agent run with a second model, and steer it back on course mid-run.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/trajectory_judge/)

## The problem

Long-horizon runs drift. Instructions fade, unsupported claims compound, and the agent wanders away from the goal. Guardrails catch policy violations at fixed boundaries, and evals score the run after it ends, when recovery costs a whole re-run. Neither watches the trajectory *as it unfolds* and intervenes while a correction is still cheap.

## The solution

`TrajectoryJudge` reviews the run's recent trajectory with a second model on a cadence: every `every` model requests, the most recent `window` tokens of the conversation (user messages, assistant messages, tool calls, and tool results, rendered as a transcript) go to the judge model for evaluation. The evaluation runs concurrently with the agent, so the run is never blocked waiting on a judge.

The judge delivers exactly one verdict per evaluation, as its output type:

- `AllGood` -- the run is on track; nothing happens.
- `Steer(message=...)` -- the run needs correction; the message is enqueued into the running conversation ([`RunContext.enqueue`](https://ai.pydantic.dev/message-history/#enqueueing-messages), `'asap'` priority) and delivered on the next model request, attributed to the judge: `Steering from trajectory judge 'hallucination-check': ...`.

Steering is the judge's *output*, not a tool it calls: the judge knows it gets one final verdict per evaluation, rather than being tempted to steer repeatedly mid-thought.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import TrajectoryJudge

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[
        TrajectoryJudge(
            model='anthropic:claude-haiku-4-5',
            instructions='Flag claims that lack evidence from files the agent actually read.',
            every=20,
        )
    ],
)

result = agent.run_sync('Fix the flaky checkout test and add a regression test.')
print(result.output)
```

## Cadence and window

- `every` counts model requests within the run; the judge evaluates on each multiple.
- `window` bounds what each evaluation sees: the transcript is clamped to its most recent `window` tokens (estimated at ~4 characters per token), so per-evaluation cost stays bounded no matter how long the run gets.
- At most one evaluation per judge is in flight at a time. A cadence tick that finds the previous evaluation still running is skipped, so a slow judge falls behind rather than piling up concurrent calls.
- An evaluation still in flight when the run ends is cancelled: its steering would have nowhere to go.

## Several judges

Each judge is its own capability instance; add one per concern. They schedule and evaluate independently, and each steering message carries its own attribution (`name`, the judge agent's `name`, or `'trajectory-judge'`).

```python
from pydantic_ai import Agent
from pydantic_ai_harness import TrajectoryJudge

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[
        TrajectoryJudge(
            model='anthropic:claude-haiku-4-5',
            name='hallucination-check',
            instructions='Flag claims that lack evidence from files the agent actually read.',
            every=20,
        ),
        TrajectoryJudge(
            model='google:gemini-3.7-flash',
            name='scope-creep',
            instructions='Flag work that was not asked for in the original request.',
            every=10,
        ),
    ],
)
```

## Advanced: bring your own judge agent

For anything beyond a model and a review focus (model settings, toolsets, fallback models, custom instructions), pass a full `Agent` instead of piling knobs onto the capability. Every evaluation runs it with `output_type=[AllGood, Steer]`, so the verdict contract is enforced at the run boundary whatever output type the agent was configured with, and an existing agent can be reused as-is only when it is dependency-free; judge dependencies are not passed to evaluations. The one constraint: the judge agent must not have output validators, which are incompatible with a per-run `output_type`.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import TrajectoryJudge
from pydantic_ai_harness.trajectory_judge import AllGood, Steer

judge = Agent(
    'openai:gpt-5.6-luna',
    name='security-risk',
    instructions=(
        'You review an AI agent trajectory for security risks: exposed secrets, unsafe '
        'file access, and unexpected egress. Return all-good, or steer with a specific warning.'
    ),
    output_type=[AllGood, Steer],
)

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[TrajectoryJudge(agent=judge, every=10)],
)
```

`agent` is mutually exclusive with `model`/`instructions`: the passed agent owns its own instructions.

## Cost and failure semantics

- The judge's model usage is threaded onto the run's `usage` and respects the run's `usage_limits`: each launch claims one request on the shared usage before the evaluation starts, so the parent's next request and concurrent judges account for in-flight evaluations and the shared request limit cannot be exceeded. A launch the request budget cannot fit skips the tick, like one that finds an evaluation still in flight.
- An evaluation failure is raised on the run at the next cadence tick or at run end; judge failures are never silently dropped. If you need a judge to degrade instead, give it a fallback model through `agent` (for example a `FallbackModel`): resilience policy belongs to the judge agent, not to fields on the capability.
- A judged run inside a durable workflow or flow (Temporal, DBOS, Prefect) is rejected with `UserError` before the first model request: the evaluation is launched from a capability hook in orchestration context, so its model calls would not be checkpointed and could repeat on replay. A durable-capable agent run outside its workflow or flow is unaffected. Run judged work outside durable execution.

## Observability

`on_verdict` is called with each verdict after it is processed (after any steering has been enqueued):

```python
from pydantic_ai_harness import TrajectoryJudge

TrajectoryJudge(
    model='anthropic:claude-haiku-4-5',
    every=20,
    on_verdict=lambda verdict: print(f'judge verdict: {verdict}'),
)
```

## Not spec-serializable

`TrajectoryJudge.get_serialization_name()` returns `None`: the capability may hold a live `Agent` instance and a callback, which cannot be serialized to an agent spec.
