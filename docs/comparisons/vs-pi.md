# Pydantic AI vs Pi

Pi is a TypeScript coding agent you can embed without forking: extensions, skills, and `pi install`
packages. Pydantic AI is a typed Python [`Agent`][pydantic_ai.Agent].
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) is a capability on that same object, so
types, tests, and Temporal attach without becoming a different product.

## Side by side

| | Pi | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| What you get | CLI + embeddable `Agent` / `createAgentSession` | A typed `Agent`; `Coder()` is a capability |
| Change it | Extensions, skills, `pi install` | Swap a capability |
| In core | `read`, `bash`, `edit`, `write` | `Coder()` includes planning and sub-agents |
| Sub-agents, plan, MCP | Examples and packages, not core | Planning and SubAgents in `Coder()`; MCP is a toolset |
| Isolation | None built in; container/VM you add | Host by default in `Coder()`; Modal / Monty / a container for untrusted work |
| Trusted state | Closures on tools | A typed object your tools read; the model never sees it |
| Crash recovery | Session tree on disk | The same agent, inside Temporal, DBOS, or Prefect |
| Structured output | A terminating tool you write | A type on the agent |
| Test offline | Stub `streamFn` | A fake model you script; no API key |

The `Coder()` snippet is on the [Claude page](vs-claude-agent-sdk.md). Pi's SDK is also in-process;
that page's fight is the `claude` subprocess, not this one.

## FAQ

**Do I have to fork Pi to add a tool?** No. An extension calls `pi.registerTool()`, or you
`pi install` a package.

**Is `Coder()` Pi?** No. It's the parts, on a Python agent. More assembly, more of the rest of
Pydantic AI attached.

**Can I embed Pi?** Yes. `createAgentSession()` in Node, or RPC JSONL if the host isn't Node.
