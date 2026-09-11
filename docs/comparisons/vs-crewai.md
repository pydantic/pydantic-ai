# Pydantic AI vs CrewAI

CrewAI is a team: roles, tasks, a process mode. Pydantic AI has no crew. Multi-agent work is Python:
call, branch, `asyncio.gather`.

A sequence of specialists is a good fit for theirs. A branch, a join, a retry on one arm only is
ordinary `async` on ours, and there is no cancel method on `Crew`.

## Side by side

| | CrewAI | Pydantic AI |
|---|---|---|
| Shape of work | Roles, tasks, `Process` | Async Python |
| Stop | None on `Crew` (`cancel` lives on A2A tasks) | A stop signal; you get the messages back |
| Budgets | `max_rpm` / `max_iter` per agent | Per run, `cost_limit` after each response |
| Crash recovery | `Crew.from_checkpoint` | The same agent, inside Temporal, DBOS, or Prefect |
| Memory / knowledge | On `Agent` and `Crew` | You wire it |
| Trusted state | Closures / config | A typed object your tools read; the model never sees it |
| Test offline | `Crew.test` runs a live `eval_llm` | A fake model you script; no API key |

## FAQ

**Drop-in?** No. Tools carry. Roles become functions.

**A Crew class?** No. An agent as a tool, a router, or `gather`.
