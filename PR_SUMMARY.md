- Closes #6092

This fixes durable runtime `event_stream_handler` propagation for both DBOS and Prefect, while also avoiding unnecessary durable model-wrapper rebuilds on ordinary runs.

Without this change:

1. Per-run `event_stream_handler=` does not reliably reach the durable model wrapper used inside execution (`DBOSModel` / `PrefectModel`).
2. On the DBOS side, rebuilding `DBOSModel(...)` when no runtime handler is active can trigger duplicate DBOS step-registration warnings for durable model step names registered from the agent name, e.g. `<agent-name>__model.request` and `<agent-name>__model.request_stream`.

### What Changed

- `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_agent.py`
  - preserve the effective per-run DBOS event-stream handler using DBOS-local run-scoped state
  - reuse `self._model` by default inside `_dbos_overrides(...)`
  - only rebuild a fresh `DBOSModel(...)` when a runtime event-stream handler is actually in play
- `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_agent.py`
  - preserve the effective per-run Prefect event-stream handler using Prefect-local run-scoped state
  - reuse `self._model` by default inside `_prefect_overrides(...)`
  - only rebuild a fresh `PrefectModel(...)` when a runtime event-stream handler is actually in play
- `tests/test_dbos.py`
  - add a deterministic regression test for runtime `event_stream_handler` propagation in DBOS workflows
  - verify that streamed events are actually observed through a per-run handler
- `tests/test_prefect.py`
  - add the analogous deterministic Prefect regression for runtime `event_stream_handler` propagation
  - note: the file is still blanket-skipped under pytest today, so runnable confirmation currently comes from direct Prefect repro plus static checks

### Validation

```bash
uv run --group dev --extra dbos pytest tests/test_dbos.py -k "runtime_event_stream_handler or runtime_external_toolset"
```

DBOS regression tests passed. In addition, the Prefect behavior was confirmed with a direct standalone flow repro before/after the `_agent.py` patch, since the in-suite pytest path is still blocked by the current file-wide skip.

### Checklist

- [ ] Any **AI generated code** has been reviewed line-by-line by the human PR author, who stands by it.
- [ ] No **breaking changes** in accordance with the [version policy](https://github.com/pydantic/pydantic-ai/blob/main/docs/version-policy.md).
- [ ] **PR title** is fit for the [release changelog](https://github.com/pydantic/pydantic-ai/releases).
