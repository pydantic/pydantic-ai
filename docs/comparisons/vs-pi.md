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
| Sub-agents, plan, MCP | Examples and packages, not core | Planning and SubAgents in `Coder()`; MCP is a capability |
| Isolation | None built in; container/VM you add | `FileSystem` rooted at a workspace; `Shell` allowlist is a guardrail; Modal / a container for untrusted work |
| Trusted state | Closures on tools | A typed object your tools read; the model never sees it |
| Crash recovery | Session tree on disk | The same agent, inside Temporal, DBOS, or Prefect |
| Structured output | A terminating tool you write | A type on the agent |
| Test offline | Stub `streamFn` | A fake model you script; no API key |

## FAQ

**Can I code like Pi?** Yes. [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) on a typed
[`Agent`][pydantic_ai.Agent]:

```python {title="coder_like_pi.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder()])
```

Files, shell, planning, and sub-agents. Take it apart if you want the blocks.

**Can that coding agent live in the Python app I already have?** Yes. Types, tests, and Temporal stay
attached.
