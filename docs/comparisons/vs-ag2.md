# Pydantic AI vs AG2

AG2 1.0 is a rewrite. The AutoGen module is gone (`import autogen` fails). What replaced it is close
to us: typed `Agent`, `AgentSpec`, `Inject`/`Depends`, `Task.cancel()`, `TestConfig`.

The fork is durability. Theirs is a `Task` with a checkpoint store and no extra infrastructure. Ours
is an engine you already run, wrapping the same agent.

## Side by side

| | AG2 1.0.4 | Pydantic AI 2.42 |
|---|---|---|
| Durability | `Task(checkpoint_store=..., resume_from=...)` | Temporal, DBOS, Prefect, Restate, Kitaru, Airflow |
| Stop | `Task.cancel()` | `CancellationToken` / `ctx.cancel()` |
| Agent as data | `AgentSpec` | `AgentSpec` → YAML; templates checked at `Agent.from_spec` |
| Dependencies | `Inject` / `Depends` | `deps_type` plus `RunContext` |
| Test offline | `TestConfig` | `TestModel` / `FunctionModel` |
| Protocols | A2A in-tree (extra for the SDK); ACP extra | ACP in the harness (experimental) |

## FAQ

**AutoGen-era code?** It won't run on AG2 1.x either. Compare at the rewrite, don't assume.

**Compatible specs?** No. Same name, different shape. Don't call `Agent.from_spec` on theirs.

---

*ag2 1.0.4, installed. `Agent`, `AgentSpec`, `Task`, `Inject`, `Depends`, `TestConfig` import from
`ag2`. `ConversableAgent` / `UserProxyAgent` / `GroupChat` are absent. A full `Task.run()` was not
completed. Pydantic AI 2.42.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
