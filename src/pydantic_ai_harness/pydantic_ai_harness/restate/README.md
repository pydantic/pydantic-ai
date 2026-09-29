# Restate Durability

`RestateDurability` makes an agent run durable on [Restate](https://restate.dev), a
durable-execution engine (Python SDK `restate-sdk`). Attach the capability and call `agent.run()`
inside a Restate service handler: every model request, MCP call, and function tool call is
journaled into a Restate run step (`ctx.run_typed(...)`), so a handler that crashes or is retried
mid-run replays from the journal instead of repeating the work.

Restate Durability is a released, non-experimental capability. Pydantic AI Harness is still on 0.x
releases, so the API may change between minor releases. See the repository
[version policy](https://pydantic.dev/docs/ai/version-policy/).

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/restate/)

## Installation

```bash
uv add "pydantic-ai-harness[restate]" "pydantic-ai-slim[openai]"
```

## Quick start

Construct the agent with the capability and call it from a Restate service handler. The agent needs
a `name`; it prefixes every step name.

```python {test="skip"}
import restate
from pydantic_ai import Agent
from pydantic_ai_harness import RestateDurability

agent = Agent('openai:gpt-5', name='analyst', capabilities=[RestateDurability()])
analyst = restate.Service('analyst')


@analyst.handler()
async def analyse(ctx: restate.Context, prompt: str) -> str:
    result = await agent.run(prompt)
    return result.output


app = restate.app([analyst])
```

The handler runs with a Restate `Context` active, which is how the capability knows to journal. Call
`agent.run()` (async) from the async handler.

## What gets journaled, and what replay means

Each of these is wrapped in its own `ctx.run_typed(...)`, so once it completes its result is
journaled and served from the journal on replay rather than being recomputed:

- **model requests** -- `{name}__model.request`, `.request_stream`, `.compact_messages`, and
  `.cancel_suspended_response`
  (the request's model id is appended as a `.{model_id}` suffix when it differs from the agent's
  default);
- **function tool calls** -- `{name}__function_toolset__{id}.call_tool:{tool_name}`;
- **MCP I/O** -- `{name}__mcp_server__{id}.get_tools`, `.get_instructions`, and `.call_tool`;
- **dynamic toolset resolution and calls** -- `{name}__dynamic_toolset__{id}.get_tools` and
  `.call_tool:{tool_name}`. A `DynamicToolset` built at construction time is resolved inside the
  step (its factory runs there, not in handler code), so its resolution and inner tool calls are
  journaled and not re-run on recovery;
- **event-stream handler calls** -- `{name}__event_stream_handler`, when an `event_stream_handler`
  is set.
- **capability and workspace operations** -- methods marked with `@durable_operation` and calls on
  `ctx.workspace` are journaled through the shared durability base.

Replay means: after a crash or retry, Restate re-runs the handler from the top. Plain Python in the
handler body runs again, but each journaled step returns its stored result instead of re-issuing the
model request or re-calling the tool. A `ModelRetry`, `ApprovalRequired`, or `CallDeferred` raised by
a tool crosses the journal as a serialized value (with its metadata preserved), so on replay the same
outcome is reproduced without re-running the tool.

Restate's journal identity is positional (encounter order); the step name is a label for
observability, not the identity. A replay reaches the steps in the same order and lines each entry up
with its recorded result, so the handler's step sequence must stay stable across a resume.

## Constraints

- The agent needs a `name` (or pass `name=` to `RestateDurability`); it prefixes every step name.
- Leaf toolsets that execute their own tools (function toolsets, MCP servers) need a unique `id`,
  which identifies their steps within the handler. A `DynamicToolset` also needs an `id`; it is
  supported when built at construction time, but cannot be added per-run via `run(toolsets=...)`.
- A journaled tool's return value is written to the journal as JSON bytes, so it must be
  JSON-serializable. Structured returns such as `ToolReturn` and `BinaryContent` are encoded by
  Pydantic first, so they round-trip.
- The executing toolsets are fixed when the agent is constructed. Passing an executing toolset
  per-run via `run(toolsets=...)` inside a handler raises a `UserError`, because a runtime toolset
  would re-run its side effects on recovery. Non-executing toolsets such as `ExternalToolset` are
  allowed at runtime.
- Tool calls run one at a time inside a handler. A journal entry's identity is its encounter order,
  so concurrently scheduled tool calls could claim each other's entries on replay. Outside a Restate
  context the agent keeps its configured parallelism.
- A step is journaled after it runs, so a crash between a tool's side effect and its journal entry
  re-runs the tool on recovery: keep tool side effects idempotent.
- Streaming inside a handler is a replay, not a live wire: the model stream is consumed and captured
  inside the step, and the run-side stream replays the captured events.
- `ctx.enqueue()` is not available inside a journaled tool, because a replay serves the recorded step
  output and would drop the enqueued messages. Enqueue from handler-level code instead.
- Each step is attempted at most three times by default. Set `max_attempts` on
  `RestateDurability` to another positive integer, or set it to `None` to opt in to Restate's
  invocation-level retry policy. Provider-client and Pydantic AI retries are separate and can
  multiply the total attempts.

## Per-tool opt-out

A function tool can opt out of journaling with `metadata={'restate': False}`, which runs it inline in
the handler rather than in a step. Reach for this for a cheap, side-effect-free tool where a journal
entry would add more overhead than it saves.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness import RestateDurability

tools = FunctionToolset(id='math')


@tools.tool_plain(metadata={'restate': False})
def add(a: int, b: int) -> int:
    return a + b


agent = Agent('openai:gpt-5', name='calc', toolsets=[tools], capabilities=[RestateDurability()])
```

`False` is the only supported value for the `restate` metadata key. Every step is journaled through
one shared set of run options, so a mapping (`metadata={'restate': {...}}`) has nothing to apply and
raises a `UserError` rather than being dropped. The opt-out and this rule apply to a tool from a
dynamic toolset as well: `False` runs it inline (its listing stays journaled), and a mapping raises.

MCP tools cannot opt out: they perform I/O and so are always journaled. Setting
`metadata={'restate': False}` on an MCP tool raises a `UserError`.

## Composition with other capabilities

`RestateDurability` orders itself innermost, so any other capability's contribution to a model
request is already applied inside the journaled step. Attach it alongside other capabilities as
usual.

The capability is built on Pydantic AI's shared durable-execution base, so durable capability
operations and workspace calls use the same Restate steps as model and tool operations. MCP
sessions connect lazily inside the first step that needs them, then remain open for the run.
`RestateDurability` emits no additional spans because core instrumentation already records the
agent, model, and tool operations it journals.

## Restate SDK integration

Pydantic AI also documents [Restate's own `RestateAgent` integration](https://ai.pydantic.dev/capabilities/durable_execution/restate/).
Use that adapter when you want the integration maintained by the Restate SDK. Use
`RestateDurability` when you want a Harness capability built on Pydantic AI's shared durability
backend, including dynamic toolsets, event handlers, durable capability operations, and workspace
calls. Do not attach both integrations to the same agent.
