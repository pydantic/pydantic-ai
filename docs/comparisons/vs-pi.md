# Pydantic AI vs Pi

Pi is a TypeScript coding agent you can also embed
([`@earendil-works/pi-coding-agent`](https://www.npmjs.com/package/@earendil-works/pi-coding-agent)).
`createAgentSession()` returns a session whose `.agent` is
[`@earendil-works/pi-agent-core`](https://www.npmjs.com/package/@earendil-works/pi-agent-core)'s
`Agent`. You change it with extensions, skills, and [pi
packages](https://pi.dev/docs/latest/packages) (`pi.registerTool()`, `pi install`). Their docs say
you should not need to fork.

Ours is a typed Python [`Agent`][pydantic_ai.Agent].
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) is capabilities on that same object, so
`deps_type`, `output_type`, `TestModel`, and a durability engine attach without becoming a different
product.

Want Pi this afternoon, take Pi. Want that coding agent inside a Python app that already has types,
tests, and Temporal, that's the harness.

## Side by side

| | Pi 0.85.1 | Pydantic AI + harness 2.42 |
|---|---|---|
| Language | TypeScript | Python |
| What you get | CLI + embeddable `Agent` / `createAgentSession` | A typed `Agent`; `Coder()` is a capability |
| Change it | Extensions, skills, `pi install` | Swap a capability |
| In core | `read`, `bash`, `edit`, `write` | `Coder()` includes planning and sub-agents |
| Sub-agents, plan, MCP | Examples and packages, not core | Planning and SubAgents in `Coder()`; MCP is a toolset |
| Isolation | None built in; container/VM you add | Host by default in `Coder()`; `ModalSandbox` / Monty / a container for untrusted work |
| Trusted state | Closures on tools | `deps_type` plus `RunContext` |
| Crash recovery | Session tree on disk | Six engines wrap the same agent |
| Structured output | A terminating tool you write | `output_type` |
| Test offline | Stub `streamFn` | `TestModel` / `FunctionModel` |

The `Coder()` snippet is on the [Claude page](vs-claude-agent-sdk.md). Pi's SDK is also in-process;
that page's fight is the `claude` subprocess, not this one.

## FAQ

**Do I have to fork Pi to add a tool?** No. An extension calls `pi.registerTool()`, or you
`pi install` a package.

**Is `Coder()` Pi?** No. It's the parts, on a Python agent. More assembly, more of the rest of
Pydantic AI attached.

**Can I embed Pi?** Yes. `createAgentSession()` in Node, or RPC JSONL if the host isn't Node.

---

*Pi `@earendil-works/pi-coding-agent` and `@earendil-works/pi-agent-core` 0.85.1, installed.
`new Agent({ streamFn })` ran a stub turn. `createAgentSession({ sessionManager: SessionManager.inMemory() })`
constructed; `session.agent` was `Agent`; default tools were `read`, `bash`, `edit`, `write`. No
built-in sandbox: [security.md](https://pi.dev/docs/latest/security). Pydantic AI 2.42.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
