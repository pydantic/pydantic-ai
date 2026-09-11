# Pydantic AI vs Pi

Pi is a finished coding agent you can also embed
([`@earendil-works/pi-coding-agent`](https://www.npmjs.com/package/@earendil-works/pi-coding-agent)).
TypeScript, skills, compaction, no sandbox of its own (run it in a container; they say so).

Ours is [pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness)
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/): the same pieces, on a Python agent you can
take apart. Want Pi's behaviour this afternoon, take Pi. Want a gate Pi didn't ship, that's a
capability here and a fork there.

The `Coder()` snippet is on the [Claude page](vs-claude-agent-sdk.md): same object, your process.

## Side by side

| | Pi 0.85.1 | Pydantic AI + harness 2.42 |
|---|---|---|
| Language | TypeScript | Python |
| What you get | A working coding agent | Parts, including `Coder()` |
| Change it | Config, or fork | Swap a capability |
| Isolation | Container you provide | Monty / Modal, plus your container |
| Drive it | NDJSON over stdio | A function call |

## FAQ

**Is the harness Claude Code / Pi?** No. It's the parts. Want a finished product, use one.

---

*Pi 0.85.1 from its docs; embeddable core not run. Pydantic AI 2.42.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
