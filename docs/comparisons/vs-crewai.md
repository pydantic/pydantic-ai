# Pydantic AI vs CrewAI

CrewAI is a team: roles, tasks, a process mode. Pydantic AI has no crew. Multi-agent work is Python:
call, branch, `asyncio.gather`.

A sequence of specialists is a good fit for theirs. A branch, a join, a retry on one arm only is
ordinary `async` on ours, and there is no cancel method on `Crew`.

## Side by side

| | CrewAI 1.15.21 | Pydantic AI 2.42 |
|---|---|---|
| Shape of work | Roles, tasks, `Process` | Async Python |
| Stop | None on `Crew` (`cancel` lives on A2A tasks) | `CancellationToken` / `ctx.cancel()` |
| Budgets | `max_rpm` / `max_iter` per agent | Per run, `cost_limit` after each response |
| Crash recovery | `Crew.from_checkpoint` | Six engines wrap the agent |
| Memory / knowledge | On `Agent` and `Crew` | You wire it |
| Trusted state | Closures / config | `deps_type` plus `RunContext` |
| Test offline | `Crew.test` runs a live `eval_llm` | `TestModel` / `FunctionModel` |

## FAQ

**Drop-in?** No. Tools carry. Roles become functions.

**A Crew class?** No. An agent as a tool, a router, or `gather`.

---

*crewai 1.15.21, installed. No `cancel`/`stop`/`abort` on `Crew`. The only `def cancel` in the
package is `a2a/utils/task.py`. `Crew.test` constructs an LLM and calls `kickoff`. Pydantic AI 2.42.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
