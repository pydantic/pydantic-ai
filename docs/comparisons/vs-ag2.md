# Pydantic AI vs AG2

AG2 1.0 is a rewrite. The AutoGen module is gone (`import autogen` fails). What replaced it is close
to us: typed `Agent`, `AgentSpec`, `Inject`/`Depends`, `Task.cancel()`, `TestConfig`.

The fork is durability. Theirs is a `Task` with a checkpoint store and no extra infrastructure. Ours
is the same agent, inside Temporal, DBOS, or Prefect.

## Side by side

| | AG2 | Pydantic AI |
|---|---|---|
| Durability | `Task(checkpoint_store=..., resume_from=...)` | The same agent, inside Temporal, DBOS, or Prefect |
| Stop | `Task.cancel()` | A stop signal; you get the messages back |
| Agent as data | `AgentSpec` | YAML you load; templates checked when you construct |
| Dependencies | `Inject` / `Depends` | A typed object your tools read; the model never sees it |
| Test offline | `TestConfig` | A fake model you script; no API key |
| Protocols | A2A in-tree (extra for the SDK); ACP extra | ACP in the harness (experimental) |

## FAQ

**Can I ship an agent as YAML?** Yes. Templates are checked when you construct, so a typo fails before
a customer hits it.

**If the process dies?** The same [`Agent`][pydantic_ai.Agent], inside Temporal, DBOS, or Prefect.
