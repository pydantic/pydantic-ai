# Brief: Fix OTel Span Nesting After wrap_node_run Changes

## Problem

The `agent.run()` method was changed from `async for node in agent_run` to manual `next()` driving so that `wrap_node_run` capability hooks fire. This changed when spans are created during the agent run, causing:

1. Span ordering changed (e.g. `get_tools` appears later)
2. Span nesting depth changed (no "running tools" parent wrapper)
3. Child spans missing or at wrong nesting level in test_dbos.py and test_temporal.py snapshots

The `wrap_node_run` hook MUST work for `agent.run()` — that's non-negotiable. But the span structure must match the original.

## Key Files

- `pydantic_ai_slim/pydantic_ai/agent/abstract.py` lines 282-293 — where `run()` drives iteration
- `pydantic_ai_slim/pydantic_ai/run.py` — `AgentRun.next()` and `__anext__`
- `tests/test_dbos.py` around line 314 — lost child spans
- `tests/test_temporal.py` around line 414 — lost spans
- `tests/test_logfire.py` — logfire span snapshots

## Approach to Investigate

1. Check if we can manually set parent span IDs when calling `next()` to maintain the same span hierarchy as `async for`
2. Check if the instrumentation layer (InstrumentedModel, OTel spans) uses the current async context to determine span parents — if so, `next()` running in a different async context than `async for` would explain the nesting change
3. Consider wrapping the `next()` call in the same span context that `async for` would use
4. Look at `use_span()` and how run spans are created in `Agent.iter()` — the span should parent all child spans regardless of iteration method
5. Compare span creation in the `async for` path vs `next()` path step by step

## Constraints

- `wrap_node_run` must fire for `agent.run()`
- Span structure must match main branch
- `async for` iteration (bare `agent.iter()`) doesn't need `wrap_node_run` (documented limitation)
- Backward compatible — no public API changes to span structure

## Reference

- PR #4640 on pydantic/pydantic-ai
- Commit 8979125e introduced the `run()` change from `async for` to `next()`
- `git diff main..capabilities -- tests/test_dbos.py tests/test_temporal.py tests/test_logfire.py` shows the span changes
