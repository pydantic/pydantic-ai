# Runtime and Extension

How a harness agent is persisted, made durable, configured, extended, and served. This covers saving and
resuming runs (`StepPersistence`), AWS Lambda durable functions (`AWSLambdaDurability`), harness
capabilities under core durable execution, Logfire-managed instructions (`ManagedPrompt`), agent-written
capabilities (`CapabilityCreation`), loading harness capabilities from YAML/JSON specs, serving an agent
to editors over ACP, and running one as a GitHub Agentic Workflow.

## Choose

| I want to ... | Use |
|---|---|
| Resume, continue, or fork a run from saved history; audit tool side effects after a crash | `StepPersistence` |
| Survive worker crashes with automatic replay (Temporal, DBOS, Prefect) | core durability capability; most harness capabilities work inside it |
| Checkpoint every model/tool step on AWS Lambda durable functions | `AWSLambdaDurability` |
| Edit, version, and roll out the system prompt from Logfire without redeploying | `ManagedPrompt` |
| Let the agent write new capabilities that load on the next run | `CapabilityCreation` |
| Define the agent in YAML/JSON with harness capabilities | `Agent.from_file(..., custom_capability_types=[...])` |
| Use the agent inside Zed or another ACP editor | `pydantic_ai_harness.experimental.acp` |
| Run the agent headless on GitHub issues/PRs/schedules | the gh-aw `pydantic-ai` engine |

## StepPersistence

Records an append-only event log, full-history snapshots at settled tool-cycle boundaries, and a ledger
of tool calls (`started`/`completed`/`failed`). It stores message history only: it does not restore
capability state, graph-node state, or in-flight streams, and it does not replay anything by itself.
You continue a run by passing a snapshot's messages to `Agent.run(message_history=...)`.

```python
import asyncio

from pydantic_ai import Agent

from pydantic_ai_harness import StepPersistence
from pydantic_ai_harness.step_persistence import InMemoryStepStore, continue_run

store = InMemoryStepStore()
agent = Agent(
    'test', capabilities=[StepPersistence(store=store, agent_name='librarian')]
)


async def main():
    await agent.run('Find the bug', conversation_id='conv-1')
    run_id = (await store.list_runs(conversation_id='conv-1'))[-1].run_id
    history = await continue_run(store, run_id=run_id)
    await agent.run('Now fix it', message_history=history, conversation_id='conv-1')
    runs = await store.list_runs(conversation_id='conv-1')
    print(len(runs), runs[-1].agent_name)
    #> 2 librarian


asyncio.run(main())
```

Key parameters: `store` (default `InMemoryStepStore()`), `agent_name` (prefix for the derived run id),
`run_id` (explicit, single-use), `parent_run_id` (auto-filled when a tool of one persisted agent runs
another in-process), `metadata: dict[str, str]`, and `capture_frontier=False` (also checkpoint the prompt
and the model's proposed tool calls before execution, so a first-request failure or a kill mid-tool-cycle
keeps them).

Stores, all in `pydantic_ai_harness.step_persistence`, all async, all accepting
`max_snapshots_per_run=None` (unbounded by default):

- `InMemoryStepStore()`: process-local; tests.
- `FileStepStore(directory)`: `<directory>/<run_id>/` with JSON/JSONL files.
- `SqliteStepStore(database='runs.db')`, or `connection=` a `sqlite3.Connection` opened with
  `check_same_thread=False`.
- `MongoStepStore(client=AsyncMongoClient | db_url=..., database=...)`: needs `uv add
  "pydantic-ai-harness[mongodb]"`. Exactly one of `client`/`db_url`, and `database` is required. With
  `db_url` the store owns the client, so call `await store.aclose()`. It creates indexes on its first
  write, so the user needs index-creation privileges.

File, SQLite, and Mongo stores move `BinaryContent` and any text part of 64 KiB or more to a
`MediaStore` (disk, same DB, or same Mongo client by default). Pass `media_store=None` to keep them
inline, or `media_store=S3MediaStore(...)` from `pydantic_ai_harness.media`.

Reading back:

- `continue_run(store, run_id=...)` / `fork_run(...)` return `list[ModelMessage]` from the latest
  `complete` snapshot, and raise `LookupError` if there is none. `fork_run` returns the same data; start
  it under a new conversation.
- `include_interrupted=True` also considers snapshots taken mid-tool-cycle. Check
  `await store.list_unresolved_tool_effects(run_id=...)` first: a `started` effect with no terminal
  record may or may not have happened.
- `store.list_runs(parent_run_id=..., conversation_id=...)` is sorted oldest first. `list_events`,
  `latest_snapshot`, and `recovery.inspect_recovery(store=..., run_id=...)` inspect a run.
- In a tool that writes external state, call
  `await annotate_tool_effect(store, ctx, idempotency_key=..., effect_summary=...)` so an orchestrator
  can tell whether a replay is safe.

What exists after a crash: each finished tool cycle saves a `complete` snapshot that already includes
the tool returns, so a crash after it resumes from there. If a tool raises and fails the run,
`on_run_error` also saves an `interrupted` snapshot with the open tool calls. A process kill mid-cycle
runs no hook, so only the last `complete` snapshot exists, plus an `interrupted` one if
`capture_frontier=True`. The default read skips `interrupted` snapshots.

```python {test="skip" lint="skip"}
async def resume(run_id: str):
    unresolved = await store.list_unresolved_tool_effects(run_id=run_id)
    for effect in unresolved:  # status 'started': may or may not have happened
        print(effect.tool_name, effect.tool_call_id, effect.idempotency_key)
    # Take the interrupted point only when no call is in doubt: resuming from it
    # re-runs its open calls or closes them out with synthesized returns.
    history = await continue_run(store, run_id=run_id, include_interrupted=not unresolved)
    return await agent.run('Continue.', message_history=history)
```

Gotchas:

- Shapes a test sees: when a streamed model request fails, the saved history (default read included)
  ends with an empty or partial-text `ModelResponse`. With `Planning`, a scripted model receives the
  last user prompt as `[text, CachePoint(...)]` plus a plan-reminder `UserPromptPart` (neither is
  stored). Assert on the parts you need, not exact message shapes.
- `run_id` is per `Agent.run`. Reusing one instance with the same explicit `run_id` raises `ValueError`.
  Group turns with `conversation_id=` on `Agent.run`.
- A derived `run_id` longer than 200 characters raises `ValueError`, so keep `agent_name` short.
  `FileStepStore` only accepts ids matching `[A-Za-z0-9_.-]{1,200}`.
- Events are never pruned. Snapshots are pruned only with `max_snapshots_per_run`, and pruning never
  deletes media blobs.
- Put `StepPersistence` before capabilities whose `after_run` rewrites history. After-hooks run in
  reverse order.
- The default `id` is `'step_persistence'`, and store writes are durable operations, so it works
  alongside a durability capability. Pass `id=` only for a second instance.
- Spec form: `{'StepPersistence': {'backend': 'memory' | 'file' | 'sqlite', 'directory': ..., 'database':
  ..., 'max_snapshots_per_run': ...}}`. Any other `backend` raises `ValueError`. Mongo is Python-only.

## Harness capabilities under core durable execution

Most harness capabilities run under core `TemporalDurability`, `DBOSDurability`, and
`PrefectDurability` (`pydantic_ai.durable_exec.*`), but support is per capability:

- `Agent(...)` with `TemporalDurability` raises `UserError` ("Toolsets that are 'leaves' ... need to
  have a unique id") for `AskUser`, `CapabilityCreation`, `ExaSearch`, `ExaAgent`, `YouSearch`,
  `YouResearch`, `Researcher`, `PydanticAIDocs`, `LocalStack`, `Macroscope`, and `BrowserUse` (either
  `session_scope`): their toolsets carry
  no stable id, and passing `id=` to the capability does not reach the toolset.
- `PlaywrightBrowser` raises `UserError` at `Agent(...)` with any durability capability (a live
  browser cannot survive replay); `TrajectoryJudge` raises at run start inside a durable workflow.
- `CodeMode` runs but skips its `speculate=` early tool launches. `BackgroundTools`, guardrails,
  `SpendLimits`, `Memory`, `Planning`, `SubAgents`, the sandboxes, and the hosted integrations build
  under Temporal.

Attach the workspace, the harness capabilities, and the durability capability at construction, and
give the agent a `name`:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.temporal import TemporalDurability

from pydantic_ai_harness.coder import Coder

agent = Agent(
    'anthropic:claude-opus-5-5',
    name='coder',
    capabilities=[LocalWorkspace('.'), Coder(), TemporalDurability()],
)
```

Rules that apply to every harness capability:

- Every leaf toolset needs a stable, unique `id`. The capabilities not listed above ship fixed or
  derived ids; two instances of one class need distinct `id=`.
- Per-run inputs (tokens, user ids) come from `deps`, never globals: tools and `auth` functions can
  execute on another worker.
- Inside a durable workflow or flow, `agent.run(..., capabilities=...)` raises `UserError` for any
  capability except a few core ones such as `Instrumentation`, and Temporal also refuses toolsets
  added at run time (for example `SubAgents` with `agent_folders`). Load authored capabilities at
  construction; they take effect on the next worker start.
- Removing or adding a capability changes the replay shape. Drain in-flight workflows first.

The per-engine feature matrix for workspace tools (commands, background jobs, live events, delegation,
timeouts) is in `CODING-AND-WORKSPACES.md` and the durable execution docs page.

## AWSLambdaDurability

Checkpoints every model request, function tool call, MCP call, and dynamic-toolset resolution as an
AWS Lambda durable step. A resumed invocation replays completed steps from the log.

```bash
uv add "pydantic-ai-harness[aws-lambda]" "pydantic-ai-slim[bedrock]"   # Python 3.11+
```

```python {test="skip"}
from typing import Any

from aws_durable_execution_sdk_python import DurableContext, durable_execution
from pydantic_ai import Agent

from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, durable_agent_handler

agent = Agent(
    'bedrock:us.amazon.nova-pro-v1:0',
    name='support',
    capabilities=[AWSLambdaDurability()],
)


@durable_execution
@durable_agent_handler
async def handler(event: dict[str, Any], context: DurableContext) -> str:
    result = await agent.run(str(event['prompt']))
    return result.output
```

Parameters: `AWSLambdaDurability(*, models=None, event_stream_handler=None, name=None, step_config=None)`.
`models` maps ids to extra models for `agent.run(model='<id>')`. `step_config` sets the base
`retry_strategy`/`step_semantics`/`serdes` for all steps. Per-tool `metadata={'aws_lambda': {...}}`
overrides it key by key, and `metadata={'aws_lambda': False}` opts a function tool out (MCP tools cannot
opt out).

Gotchas:

- A run is durable only inside `durable_agent_handler` or `run_durable(lambda: agent.run(...),
  context=context)`. `agent.run_sync(...)` or your own `asyncio.run` runs normally but checkpoints
  nothing, and raises no error to say so.
- `@durable_execution` must be the outermost decorator. The reverse order raises `UserError`.
  `run_durable` cannot be called from inside a running event loop.
- The agent needs a `name` (or `name=`), and every toolset needs a unique `id`. Both are checked at
  `Agent(...)`.
- Steps are at-least-once, and the SDK retries six times by default. For a tool that must not repeat,
  set both `StepSemantics.AT_MOST_ONCE_PER_RETRY` and `retry_strategy=RetryPresets.none()`. Step
  retries stack with Pydantic AI and provider-client retries, so disable one side.
- Tool calls run sequentially inside a durable handler. Nested durable runs are rejected.
- Changing the tools, MCP servers, model, or `event_stream_handler` breaks in-flight executions. Deploy
  a new published version.
- `ctx.enqueue()` is unavailable inside a durable step. Do not detach work with `asyncio.create_task()`.
- Budget: 3,000 operations and 100 MB of checkpointed state. Return references, not blobs.

## ManagedPrompt

Resolves a Logfire-managed prompt once per run and uses it as the agent's instructions, with the label
and version attached as baggage to every span of the run.

```bash
uv add "pydantic-ai-harness[logfire]"
```

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness import ManagedPrompt


def echo_instructions(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(info.instructions or '')])


agent = Agent(
    FunctionModel(echo_instructions),
    capabilities=[
        ManagedPrompt('support_agent', default='You are support.', label='production')
    ],
)
print(agent.run_sync('hi').output)
#> You are support.
```

Parameters: `name` (positional; declared as the variable `prompt__<name>`, with hyphens turned into
underscores), or a prebuilt `logfire.variables.Variable`; `default` (required with a name); `label`;
`targeting_key` and `attributes` (static or `ctx -> value`); `render_template=False` (Handlebars against
`deps`; needs `pydantic-ai-slim[spec]`); `logfire_instance`. Call `logfire.configure()` in your app to
get remote values. Until then, and whenever no remote value is published, `default` is used.

Gotchas: pin `label='production'` to keep the provider prompt cache warm, because rollouts across
labels split it. A label change mid-run applies from the next run. The capability orders itself
outermost, wrapping `Instrumentation`. `ManagedPrompt.resolved` exposes the current run's
`ResolvedVariable`, and is `None` outside a run.

## CapabilityCreation

Gives the model `author_capability(name, code)`, `list_authored_capabilities()`, and
`disable_authored_capability(name)`. Authored code is written to `<directory>/<name>.py`, imported,
and validated (exactly one `AbstractCapability` subclass that constructs with no arguments). It becomes
active on the **next** run, and only if you pass it in:

```python {test="skip"}
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

from pydantic_ai_harness import CapabilityCreation

creation = CapabilityCreation(directory=Path('.authored'))
agent = Agent('anthropic:claude-opus-5-5', capabilities=[LocalWorkspace('.'), creation])



async def main():
    history = None
    for prompt in ['Add a capability that logs tool calls.', 'Now refactor utils.py.']:
        result = await agent.run(
            prompt, message_history=history, capabilities=creation.store.load_active()
        )
        history = result.all_messages()
```

Gotchas: a run raises `UserError` unless its workspace is a writable `LocalWorkspace`, because
authored code executes in your process and would bypass a sandbox. Only use it where you would run the
model's code yourself. `load_active()` skips entries that fail to load. Under durable execution
`capabilities=` on a run is refused, and `CapabilityCreation` itself does not build under
`TemporalDurability` (see the durable execution section). `guidance=''` drops the
built-in system-prompt guidance. `CapabilityCreation` is not spec-serializable.

## Harness capabilities in agent specs

Core resolves spec capability names from a closed built-in registry. A harness class loads only when
you pass it in `custom_capability_types`. There is no auto-registration, and no harness-wide list to
pass. The spec name is the class name unless the class overrides `get_serialization_name()`. A name
not in the list raises `ValueError` naming the valid choices.

```python
from pydantic_ai import Agent

from pydantic_ai_harness import Planning, StepPersistence

spec = {
    'model': 'test',
    'capabilities': [
        'Planning',
        {'StepPersistence': {'backend': 'memory', 'agent_name': 'triage'}},
    ],
}
agent = Agent.from_spec(spec, custom_capability_types=[Planning, StepPersistence])
```

`Agent.from_file('agent.yaml', custom_capability_types=[...])` works the same way (install
`pydantic-ai-slim[spec]` for YAML). Capability entries take the forms `Name`, `{Name: positional_arg}`,
or `{Name: {kwargs}}`, and each class's `from_spec` builds the instance. Opted out (they hold callables
or live agents, and raise `ValueError` if listed): `AWSLambdaDurability`, `CapabilityCreation`,
`DynamicWorkflow`, `InputGuardrail`, `OutputGuardrail`, `SubAgents`, `SystemReminders`,
`ToolGuardrail`, `TrajectoryJudge`. `AskUser` has a spec name but needs an `answerer` callable, so in
practice it is passed in code too. Pass those instances with the `capabilities=` keyword of
`Agent.from_spec`/`Agent.from_file`, which adds them to the spec's list. Keep secrets out of spec files
and let capabilities read their env vars.

A spec whose `model` is a provider string (`'openai:gpt-5'`) needs that provider's API key when the
agent is built. In tests, pass `defer_model_check=True` and run under `agent.override(model='test')`.

## ACP server (experimental)

Serves any `Agent` over the Agent Client Protocol on stdio, so Zed and other ACP clients can drive it
with streamed text, diffs, tool approval, and sessions. It lives under
`pydantic_ai_harness.experimental` and may change or be removed in any release, without deprecation.
Importing it emits `HarnessExperimentalWarning`.

```bash
uv add "pydantic-ai-harness[acp,anthropic]"
```

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.workspaces import LocalWorkspaceBackend

from pydantic_ai_harness.experimental.acp import (
    AcpSession,
    AcpSessionConfig,
    InMemorySessionStore,
    run_acp_stdio_sync,
)
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-opus-5-5')


def session_config(session: AcpSession) -> AcpSessionConfig[None]:
    return AcpSessionConfig(
        deps=None,
        capabilities=[FileSystem[None](), Shell[None]()],
        workspace=LocalWorkspaceBackend(session.cwd),
    )


if __name__ == '__main__':
    run_acp_stdio_sync(
        agent, session_config=session_config, session_store=InMemorySessionStore()
    )
```

`AcpSessionConfig` requires only `deps` (`None` for no deps); `capabilities`, `toolsets`, and
`workspace` default to `None`, so an agent without file or shell tools needs no workspace.

Gotchas: root tools at `session.cwd` through `session_config`, not a static `FileSystem`. Tools marked
`requires_approval=True` become client approval prompts. `acp_filesystem(session)` and
`acp_terminal(session)` return editor-backed tools, or `None` when the client does not offer them, so use
`acp_filesystem(session) or FileSystem[None]()`. If a client sends MCP servers and there is no
`session_config` to connect them, the session is rejected. Prompt content is text-only unless you pass
`prompt_capabilities`. `models=[...]` exposes a model picker. Use async `run_acp_stdio` inside a running
loop. Silence the warning with `warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)`
(importing `pydantic_ai_harness.experimental` itself does not warn).

## GitHub Agentic Workflows

The gh-aw `pydantic-ai` engine runs a Pydantic AI agent in GitHub Actions on issues, PRs, or a schedule.
In the workflow `.md`, import `pydantic/pydantic-ai-harness/gh-aw/pydantic.md@main`, set
`engine: {id: pydantic-ai, model: openai/gpt-5}` (`provider/model` is required), and point
`engine.env.PAI_AGENT` at `module:variable` (for example `my_agent:agent`), at
`pydantic_ai_harness.researcher:researcher_agent`, or at a `.yml`/`.json` spec. Omit `PAI_AGENT` to run
`Coder`. Then run `gh aw compile` and commit the `.lock.yml` with it. Gotchas: leave the model off the
`Agent`, because the engine's `-m` overrides it. Writes go through safe outputs (for example the
`safeoutputs_add_comment` tool). Install extra dependencies with a workflow `steps:` `pip install --user`.
The provider key is a repository secret, never `engine.env`. A spec agent **cannot** use harness
capabilities, because the CLI passes no `custom_capability_types`. Use a Python module for those.

## See also

- https://pydantic.dev/docs/ai/harness/step-persistence/
- https://pydantic.dev/docs/ai/harness/durable-execution/
- https://pydantic.dev/docs/ai/harness/aws-lambda/
- https://pydantic.dev/docs/ai/harness/managed-prompt/
- https://pydantic.dev/docs/ai/harness/capability-creation/
- https://pydantic.dev/docs/ai/harness/acp/
- https://pydantic.dev/docs/ai/harness/gh-aw/
- https://pydantic.dev/docs/ai/agent-spec/
- https://pydantic.dev/docs/ai/durable_execution/overview/
