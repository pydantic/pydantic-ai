# Pydantic AI vs Claude Agent SDK

The Claude Agent SDK gives you Claude Code from Python. That's not a figure of speech: the library
finds the `claude` binary on your machine and runs it as a child process, then talks to it over its
own protocol. Everything the CLI can do, editing files, running commands, skills, subagents, hooks,
checkpoints, session forking, MCP servers, the permission prompts, you get, because it is the same
program.

If you want something Claude-Code-shaped, that is the shortest path there is. Version 0.2.152 has
picked up useful things since we last looked: `max_budget_usd` caps what a run may spend, and
permission handling, session resumption and hook events are all configurable.

Pydantic AI runs the loop in your own process, on any model, with your own Python functions as tools.
These are different tools. The way to choose is which side of one line your project sits on.

## Whose process is it

Configuring the Claude SDK means describing an agent in data. Tools are strings
(`allowed_tools=['Read', 'Glob']`). Subagents are dictionaries. Hooks are JSON events. That's a
reasonable interface to a program running elsewhere, and it's the only interface available, because
your Python isn't where the loop lives.

In Pydantic AI the loop is in your process, so a tool is a function you wrote, with your types, your
imports, and a debugger that stops inside it. Your tools can hold a database connection. Your tests
can run without launching a CLI. `TestModel` and `FunctionModel` script the model's behaviour
deterministically, and `ALLOW_MODEL_REQUESTS = False` turns any accidental real API call into an
error. Testing a Claude SDK agent means launching the CLI and letting it talk to Anthropic, which is
an integration test whether you wanted one or not.

Stopping splits the same way. `ClaudeSDKClient.interrupt()` sends a control request over the
transport, in streaming mode only, from outside the run. Ours is a `CancellationToken` or
`ctx.cancel()` from inside a tool, and the run ends in `RunCancelled` carrying the history.

## Sessions remember conversations; engines remember work

The Claude SDK's continuity story is sessions: resume by id, fork one, rewind to a checkpoint. That's
good for a conversation you want to pick back up.

It's not crash recovery. If the process dies halfway through a long run, the session tells you what
was said, not which of the six things the agent was doing had finished. Pydantic AI runs can be
wrapped by a durable engine (Temporal, DBOS, Prefect, Restate, Kitaru, or Airflow) which restarts the
work rather than the transcript, and the wrapping doesn't change the agent.

## Side by side

| | Claude Agent SDK 0.2.152 | Pydantic AI 2.42 |
|---|---|---|
| Where the loop runs | The `claude` CLI, as a child process (`SubprocessCLITransport`) | Your process |
| Models | Anthropic | Any provider, with `FallbackModel` for failover |
| Tools | Named in strings; the CLI owns them | Your Python functions, with types and validation |
| Subagents and hooks | Configuration dictionaries and JSON events | Capabilities and typed hooks in your code |
| Trusted state | Nothing typed; configuration and environment | `deps_type`, read by tools, invisible to the model |
| Coding-agent features | Everything Claude Code has, immediately | Composable pieces in the harness: filesystem, shell, `CodeMode`, subagents, skills, memory |
| Spend limits | `max_budget_usd` for the run, enforced by the CLI | `UsageLimits` on requests, tool calls and tokens, checked before the next call; `cost_limit` when pricing data is available |
| Stopping a run | `ClaudeSDKClient.interrupt()`, streaming mode only, from outside the run | `CancellationToken`, `ctx.cancel()` from inside a tool, `RunCancelled` carrying resumable history |
| Continuity | Sessions: resume, fork, rewind | Message history you own and store |
| Crash recovery | Not the same thing as a session | Six engines wrap the agent object |
| Testing offline | Launch the CLI; it's an integration test | `TestModel` and `FunctionModel`, no network |

## FAQ

**Can I use Claude models with Pydantic AI?**
Yes, directly, including the Anthropic-hosted tools. This isn't about which model you use.

**Can Pydantic AI build a coding agent?**
Yes, through [pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness), which supplies a
filesystem, a shell, subagents, skills, memory, compaction, and `CodeMode` as pieces you assemble.
It's a library of parts, not a finished product, so it's more work than pointing the Claude SDK
at a directory, and more yours afterwards.

**Is one more secure?**
They protect different things. The Claude SDK isolates by process and has a mature permission
interface. Pydantic AI keeps trusted state out of the model's reach entirely, pauses on any tool marked
`requires_approval=True`, and runs model-written code in a sandbox through the harness.

---

*Checked against claude-agent-sdk 0.2.152 and Pydantic AI 2.42 on 2026-09-10. `interrupt()` was read in
`client.py` and `_internal/query.py`, where it sends an `interrupt` control request over the transport rather
than signalling the process. The subprocess behaviour, `SubprocessCLITransport`, and the `ClaudeAgentOptions`
fields, including `max_budget_usd`, come from reading the installed package. We recheck this page's version
pins and behaviour claims each time Pydantic AI ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
