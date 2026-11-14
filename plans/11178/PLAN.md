# Investigate durable agent-graph delays

Related issue: https://github.com/pydantic/pydantic-ai/issues/11178

This is a proposal for maintainer review, not an implementation or a claim that recovery has been reproduced.

## Current evidence

- The graph routes its delays through `set_agent_graph_sleep`; without a registration it uses `anyio.sleep`.
- Temporal and DBOS register engine-owned sleeps. Prefect and the four Harness engines named in the issue do not.
- The current Prefect guide explicitly documents plain polling waits being repeated on a flow retry.
- PR #11932 introduces another consumer of graph delays and explicitly retains the Prefect limitation. This proposal does not duplicate that retry capability.

Sources: [graph hook](https://github.com/pydantic/pydantic-ai/blob/59ae8e10f605993f448683397a48e7dc778f9a6c/pydantic_ai_slim/pydantic_ai/_agent_graph.py#L234), [DBOS adapter](https://github.com/pydantic/pydantic-ai/blob/59ae8e10f605993f448683397a48e7dc778f9a6c/pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_utils.py), [Prefect guide](https://github.com/pydantic/pydantic-ai/blob/59ae8e10f605993f448683397a48e7dc778f9a6c/docs/durable_execution/prefect.md#L308), [retry PR](https://github.com/pydantic/pydantic-ai/pull/11932).

## Proposed first step: establish recovery behavior

Use a small public-agent scenario that completes a model operation, reaches a graph-owned polling delay, and resumes after the engine interrupts or retries its workflow. Use the real engine recovery harness and controllable engine time where available, rather than asserting only that a mocked sleep was called.

Record these cases separately:

1. A completed delay followed by recovery: is the completed wait repeated?
2. An interrupted pending delay: does recovery honor the original wake-up deadline, or start the full delay again?
3. Recovery after the original deadline: is another full wait imposed?
4. Confirm that already-recorded model operations are replayed rather than requested again.

Start with Absurd's existing Postgres recovery fixtures and a `FunctionModel`/`BatchModelRequests` scenario, rather than depending on unmerged #11932 or live model credentials. Compare with Temporal or DBOS as a control. Measure only behavior each engine promises; a missing registration alone is not proof of a defect.

## Engine-specific investigation

| Engine | Boundary to verify before proposing a change |
| --- | --- |
| Prefect | The guide documents no native durable sleep. Determine whether to retain and pin that limitation or design a separate persisted-deadline approach; do not call a cached plain-sleep task a durable pending timer. |
| Absurd | Investigate its task-context durable sleep and how a stable operation name is allocated across replay. |
| pgtask | Investigate named task sleeps and occurrence accounting so repeated equal delays do not collapse into one checkpoint. |
| AWS Lambda | Investigate durable waits through the existing handler-thread bridge and the SDK's local runner. Resolve its minimum one-second wait for subsecond graph delays; do not call a synchronous durable context directly from the async agent loop. |
| Render | Establish whether the supported SDK implements a durable timer before offering an adapter; an unimplemented SDK sleep is not a usable primitive. |

SDK references: [Absurd](https://github.com/earendil-works/absurd/blob/main/sdks/python/src/absurd_sdk/__init__.py), [pgtask](https://github.com/Kludex/pgtask/blob/main/sdks/python/python/pgtask/client.py), [AWS Lambda](https://github.com/aws/aws-durable-execution-sdk-python/blob/main/packages/aws-durable-execution-sdk-python/src/aws_durable_execution_sdk_python/context.py), [Render](https://github.com/render-oss/sdk/blob/main/python/render/workflows/durable_context.py).

Run one engine's investigation first, then expand only where the evidence warrants it. Report unavailable services or SDK limitations without claiming recovery was tested.

## Boundaries for a later implementation

- Obtain maintainer agreement and issue assignment before runtime changes.
- Fix only an engine with a demonstrated unwanted recovery result and an agreed contract.
- Reuse the graph's existing sleep hook; avoid introducing a new cross-engine timer abstraction.
- Keep ordinary agents, execution inside durable units, and nested runs on their existing behavior.
- Test registration cleanup after exceptions, task cancellation and concurrent runs.
- Keep persisted operation names and sequence order stable, or agree an explicit in-flight workflow migration.
- Split engine changes into follow-ups when they require different SDK contracts or deployment services.

## Decisions requested

1. Is the desired timer contract to skip completed waits and preserve an outstanding wait's original deadline?
2. Which engine should be investigated first, and is its recovery environment available to contributors?
3. For Prefect and Render, should the documented limitation remain until an engine-supported timer exists?

No runtime tests or engine recovery experiments have been executed for this plan.
