# Pydantic AI vs CrewAI

CrewAI is a team: roles, tasks, a process mode. Pydantic AI has no crew. Multi-agent work is Python:
call, branch, `asyncio.gather`.

A sequence of specialists is a good fit for theirs. A branch, a join, a retry on one arm only is
ordinary `async` on ours, and there is no cancel method on `Crew`.

## Side by side

| | CrewAI | Pydantic AI |
|---|---|---|
| Shape of work | Roles, tasks, `Process` | Async Python |
| Stop | No method on `Crew`; streaming `aclose()` | A stop signal; you get the messages back |
| Budgets | `max_rpm` / `max_iter` per agent | Per run, `cost_limit` after each response |
| Crash recovery | `Crew.from_checkpoint` | The same agent, inside Temporal, DBOS, or Prefect |
| Memory / knowledge | On `Agent` and `Crew` | You wire it |
| Trusted state | Closures / config | A typed object your tools read; the model never sees it |
| Test offline | `crewai test` runs a live model | A fake model you script; no API key |

## FAQ

**How do I do multi-agent without a crew?** An agent as a tool, a router, or `asyncio.gather`. A
retry on one arm is ordinary `async`.

**Can one of those agents be a coding agent?** Yes. [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/)
on that [`Agent`][pydantic_ai.Agent]. The others stay ordinary Python.
