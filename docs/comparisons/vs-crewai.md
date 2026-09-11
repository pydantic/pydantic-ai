# Pydantic AI vs CrewAI

CrewAI is a team: roles, tasks, a process mode. Pydantic AI has no crew. Multi-agent work is Python:
call, branch, `asyncio.gather`.

A sequence of specialists is a good fit for theirs. A branch, a join, a retry on one arm only is
ordinary `async` on ours, and there is no cancel method on `Crew`.

## Side by side

| | CrewAI 1.15.21 | Pydantic AI 2.42 |
|---|---|---|
| Shape of work | Roles, tasks, process mode | Async Python |
| Stop | None on `Crew` | `CancellationToken` / `ctx.cancel()` |
| Budgets | Per agent | Per run, `cost_limit` after each response |
| Crash recovery | Built-in checkpoints | Six engines wrap the agent |
| Memory / knowledge | Built in | You wire it |
| Test offline | Live model | `TestModel` / `FunctionModel` |

## FAQ

**Drop-in?** No. Tools carry. Roles become functions.

**A Crew class?** No. An agent as a tool, a router, or `gather`.

---

*crewai 1.15.21, Pydantic AI 2.42. No `cancel` on `Crew`. Runtime with a live model was not run.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
