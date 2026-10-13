# Deep Agents Mapping

Use this reference when the slice calls `create_deep_agent`, or a vendored equivalent, or depends on Deep Agents middleware, backends, profiles, skills, memory, subagents, permissions, sandboxes, or hosts. Deep Agents is an opinionated harness over the LangChain agent loop and the LangGraph runtime; Pydantic AI Harness is the matching capability library over the Pydantic AI agent loop. Map behavior to behavior, then confirm every candidate against the locked source and target versions. Rows that concern the plain agent loop, tools, structured output, state, or streaming are in [Concept Mapping](CONCEPT-MAPPING.md); this file covers what the harness adds on top.

## Contents

- [What the harness bundles](#what-the-harness-bundles)
- [Construction, models, and middleware](#construction-models-and-middleware)
- [Context and execution](#context-and-execution)
- [Orchestration and recovery](#orchestration-and-recovery)
- [Hosts and protocols](#hosts-and-protocols)
- [A tested Harness composition](#a-tested-harness-composition)
- [Migration traps](#migration-traps)

## What the harness bundles

Translating constructor arguments one by one misses the harness. Snapshot the resolved agent instead: the effective prompt, the model-visible tool list, the middleware order on the main agent and on every subagent, the state schema, and the runtime configuration. In current `deepagents` releases the defaults are:

- **Tools:** `ls`, `read_file`, `write_file`, `edit_file`, `glob`, and `grep`; `execute` only when the resolved backend supports shell execution; `task` when inline subagents or the general-purpose child are enabled. `write_todos` exists only before 0.7 or with an explicitly configured todo middleware.
- **Prompt:** the caller's `system_prompt`, then the harness profile's `base_system_prompt`, then its `system_prompt_suffix`, with skills, memory, and filesystem middleware injecting more text per request.
- **Middleware order:** skills, filesystem, subagents, summarization, patch-tool-calls, async subagents; then caller middleware; then profile `extra_middleware`, tool exclusion, provider prompt caching, memory, and human-in-the-loop. Profiles can exclude entries by class or name.
- **Profiles:** `HarnessProfile` shapes prompt layers, tool descriptions and exclusions, excluded middleware, and the general-purpose child; `ProviderProfile` shapes model construction. A Pydantic AI model profile configures provider behavior only; it never configures the Harness composition, so profile-driven differences must be reproduced explicitly.
- **Subagent stacks:** each declarative child gets its own filesystem, skills, memory, and exclusion middleware; a compiled child is whatever the caller built; an `AsyncSubAgent` is a remote LangGraph deployment. A behavior seen only during delegated work may come from any of these.

Record which of these the application actually exercises. An unused default is `not-applicable`, not a migration task.

## Construction, models, and middleware

| Deep Agents contract | Pydantic AI candidate | Required check |
| --- | --- | --- |
| `create_deep_agent` | `Agent(...)` with explicit instructions, tools, toolsets, capabilities, output type, and settings | Snapshot the effective prompt, tool schemas, profile behavior, result, errors, and limits. |
| LangChain `BaseChatModel` | a Pydantic AI provider model or model string | Do not pass the LangChain model through. Translate transport, endpoint, authentication, provider settings, model profiles, retries, and limits. Construct and validate every configured model branch. |
| harness and provider profiles | explicit agent configuration plus Pydantic AI model settings and profiles | Trace provider/model lookup and merge order. Preserve prompt layers, tool descriptions and exclusions, middleware changes, general-purpose-child settings, and model-construction defaults. |
| graph `.invoke()`, `.ainvoke()`, `.stream()`, `.astream()`, or event callbacks | `run_sync`, `run`, `run_stream`, `run_stream_events`, `iter`, and `event_stream_handler` | Use `run_stream` for final-output streaming; use `run_stream_events` or `iter` when the contract needs the complete tool/event lifecycle. Preserve sync, async, result, message, error, and event shapes at the application boundary. |
| `context_schema` | `deps_type` and `RunContext` | Dependencies are runtime resources and identity, not checkpointed state. |
| custom `state_schema` and reducers | capability-owned run state, an application repository, or `pydantic_graph` | Classify each field's lifecycle and merge semantics. |
| custom or replaced middleware | a core capability, `Hooks`, a wrapper toolset, or a focused Harness capability | Match hook timing, ordering, name-based replacement or exclusion, request mutation, retry, and failure behavior on main and child stacks. |
| `response_format` | Pydantic `output_type` and an explicit output mode | Test invalid output, final-tool behavior, and streaming. |
| graph cache and provider prompt caching | application caching plus the selected Pydantic AI provider behavior | Separate graph-step caching from provider prompt caching; verify keys, scope, invalidation, cached content, usage, and replay behavior. |
| LangSmith, Langfuse, or other callbacks | retain the existing system, or use OpenTelemetry or Logfire after agreement | Compare correlation, dashboards, evaluations, retention, export, and privacy. Preserve conversation, run, child, tool-call, and external-job identities separately. |

Adapt LangChain `@tool` and `StructuredTool` objects to typed Pydantic functions or toolsets. Preserve their names, schemas, return values, and errors rather than passing framework objects through unchanged; the transitional wrappers in [Concept Mapping](CONCEPT-MAPPING.md#transitional-bridges) apply here too.

For MCP, use `pydantic_ai.capabilities.MCP` when provider-native or client-side selection is needed; use `pydantic_ai.mcp.MCPToolset` for client-side MCP connections. Preserve transport, authentication, tool filtering, structured content, connection and session lifetime, elicitation, sampling, retries, and tracing at the integration boundary.

## Context and execution

Import each capability from its owning public module, for example `Skills` from `pydantic_ai_harness.skills` and `RepoContext` from `pydantic_ai_harness.repo_context`; do not use compatibility modules left behind by a rename.

| Deep Agents feature | Harness candidate | Important difference |
| --- | --- | --- |
| coding-agent defaults | `Coder(workspace)` | Compare its documented composition first. It is an opinionated local coding harness, not a guarantee of Deep Agents parity. Use individual capabilities when a component must change. |
| todo planning | `Planning` | Add it only when the source actually exposes `write_todos` or planning is an accepted addition. Harness has its own schema, tools, reminder injection, events, and stores. |
| local workspace files | `FileSystem` | It is a rooted local filesystem toolset, not Deep Agents' virtual `BackendProtocol`. Compare the installed tool schemas and limits: Deep Agents file tools use `file_path`, Harness uses `path`, and backend file-size limits need a separate guard. |
| filesystem `permissions` | a wrapper toolset or capability, deferred approval, and backing-service authorization | Preserve operation classes, first-match rule order, unmatched-call behavior, allow/deny/interrupt results, route-relative paths, and child overrides. Deep Agents interrupt rules require a checkpointer; persist and restore the target approval continuation too. The rules cover built-in file tools, not arbitrary tools, direct backend access, or sandbox commands. |
| host or local shell execution | `Shell` | It runs host subprocesses. Its command filtering and path checks cannot reproduce a sandbox backend's OS isolation. |
| remote isolated execution (Modal, Daytona, Runloop, Vercel, LangSmith sandboxes) | `ModalSandbox`, or the existing sandbox client in typed deps behind narrow tools | Match lease, filesystem, network, credential, reconnect, timeout, and cleanup contracts. The sandbox service, not a tool filter, is the isolation boundary. |
| interpreter and programmatic tool calling | `CodeMode`, `DynamicWorkflow`, or an application interpreter | Deep Agents' QuickJS REPL persists state per thread by default, dispatches `task(...)`, and calls `tools.*` only from an allowlist. Harness uses Monty and different tool schemas. Match state lifetime, structured results, resource limits, and authority; interpreter bridges can bypass ordinary tool-approval paths unless guarded explicitly. |
| multimodal input, file reads, or tool results | Pydantic AI user content and `ToolReturn`; `pydantic_ai_harness.media` stores and walkers for persistence offload | Keep model-facing content in Pydantic AI types. With `StepPersistence`, configure `media_store` on its `FileStepStore`, `SqliteStepStore`, or `MongoStepStore`; for another serializer, use `externalize_media` and `restore_media` directly. Preserve order, MIME support, provider filtering, references, serialization, and restoration. |
| `skills=[...]` and `SKILL.md` catalogs | `Skills` | Pass the parent skill-library directory whose immediate children are `<name>/SKILL.md` packages. Harness scans it at construction and keeps that snapshot. Normalize flat catalogs into this shape; do not replace it with eager prompt concatenation. Harness loads instructions only: preserve bundled files, scripts, writable catalogs, and behavioral frontmatter with a focused service only when those contracts matter. |
| `AGENTS.md` and repository instructions | `RepoContext` | Configure `workspace_dir`, `home_dir`, `filenames`, `autoload_instructions`, `expose_inventory_tool`, `inventory_tool_name`, `asset_roots`, `nested_traversal`, and `nested_inject` from observed discovery and injection behavior. Compose it with `FileSystem` when file traversal should inject nested context. It does not replace skill activation or writable memory. Never promote untrusted repository text to system priority accidentally. |
| `memory=[...]` and writable long-term memory | `Memory` | Check namespace, store, injection role and bound, search, concurrency, retention, and deletion. Deep Agents `memory=[...]` injects backend-relative files: use explicit instructions for static content, `RepoContext` for repository-owned context, and `Memory` or an application store for model-written notes. If callers use file tools at paths such as `/memory/...`, preserve those names with an adapter or intentionally change callers and prompts. |
| retrieval or RAG | existing retrieval services exposed as typed tools | Retrieval is application data access, not Deep Agents memory or a filesystem backend. Preserve ingestion, chunking, filters, tenancy, ranking, metadata, citations, result bounds, latency, and failure shapes. |
| `SummarizationMiddleware` | Harness compaction capabilities, often `TieredCompaction`; provider-native compaction from core where the provider supports it | Harness rewrites Pydantic AI message history. Match thresholds, summary prompt and role, tool-pair validity, receipts, usage, and recovery. |
| oversized tool-result offload and message eviction | `ToolOutputLimits` with matching bands and store, or the retained backend artifact path | Use `ToolOutputLimits(store=...)` only when its overflow-store contract matches; otherwise retain or adapt the backend artifact path. Human-message eviction needs separate handling. Verify thresholds, lossless versus lossy behavior, receipts, read-back, failure fallback, store lifetime, and serialization. |
| prompt-caching middleware | provider cache controls plus stable capability composition; `WarnOnCacheBusts` to detect prefix collapse | Compare exact cache markers and the effect of plans, memories, and compaction on the cached prefix. |
| `PatchToolCallsMiddleware` | provider-valid history capture and an explicit continuation policy | Pydantic AI repairs dangling tool calls before a request; test malformed tails, orphan results, pending approvals, client-supplied history, and whether repair hides a real side effect. |

For pluggable or composite backends (`StateBackend`, `StoreBackend`, `FilesystemBackend`, `CompositeBackend`, sandbox backends), expose the existing storage or sandbox service through a focused toolset or capability. Preserve runtime-injected backend factories, route prefixes, path, tenancy, transaction, sandbox ownership, and lifecycle contracts instead of forcing them through local `FileSystem`. Preserve each route's storage lifetime and ownership: do not replace an ephemeral `StateBackend` route with persistent filesystem or store state.

Enforce authorization and isolation at the backing service. Treat prompt instructions, globs, path checks, and shell allowlists as defense in depth.

## Orchestration and recovery

Import `SubAgent` and `SubAgents` from `pydantic_ai_harness.subagents`. Harness can load its Markdown agent format from configured folders, but it does not consume a Deep Agents YAML roster. Parse and validate retained application configuration into explicit `Agent` and `SubAgent` objects, or intentionally convert it: YAML `system_prompt` becomes `Agent.instructions` or the Markdown body, `description` remains the description, and `tools` become explicit toolsets or a `tool_resolver`. Set the child model explicitly or through `AgentOverride`; a `model` key in Harness Markdown is parsed but not applied.

| Deep Agents feature | Pydantic AI and Harness candidate | Important difference |
| --- | --- | --- |
| synchronous named children and the general-purpose child | `SubAgents`, or a parent tool that runs a typed child `Agent` | Inspect the automatic general-purpose child and its disablement, isolated versus forked context, declarative versus compiled children, tools, skills, permissions, approvals, state, structured results, and streaming inheritance. Harness does not transfer mutable parent capability state implicitly; use explicit dependencies, `shared_capabilities`, or an application repository, then test child write visibility. |
| model-written fan-out or chaining (QuickJS `task(...)`) | `DynamicWorkflow` | This is a redesign, not an API port: Deep Agents generates JavaScript for QuickJS, Harness generates Python for Monty and calls child agents from `run_workflow`. Verify structured results, interpreter and thread state, nesting, and tool boundaries. Neither is a durable background task service. |
| `AsyncSubAgent` background work | an application queue/worker plus narrow start, list, check, update, and cancel tools | Preserve graph ID, transport and authentication, task/thread/run identity, pending state, stale reads, steering, cancellation, and result durability. Harness synchronous delegation and generated workflows are not background-task services. |
| `RubricMiddleware` grading loop | an explicit evaluator/revision loop or focused capability | Preserve the invocation rubric, grader model and tools, revision feedback, iteration limit, terminal verdict, callbacks, and progress events. `TrajectoryJudge` provides cadence-based mid-run steering, so use it only when that different lifecycle is accepted. |
| `interrupt_on` and approval | `DeferredToolRequests`, `DeferredToolResults`, `ToolApproved`, and `ToolDenied` | Include `DeferredToolRequests` in `output_type`. When returned, persist `result.all_messages()`, resolve the request with `build_results(...)`, then call `run(message_history=..., deferred_tool_results=...)`. Use `HandleDeferredToolCalls` only for an in-process handler whose lifecycle is sufficient. The LangGraph `Command(resume=...)` protocol is not reproduced; see [Workaround Recipes](WORKAROUND-RECIPES.md#conversational-interrupts-approval-and-durable-resume) for the durable record. |
| provider-valid run snapshots | `StepPersistence` | It records step events, continuable snapshots, lineage, and tool effects; it is not arbitrary LangGraph graph-state checkpointing. |
| durable model and tool execution | `TemporalDurability`, `DBOSDurability`, `PrefectDurability`, `AWSLambdaDurability`, or another matching integration | Select by the deployment contract and install its extra. Wire the integration's workflow or handler boundary, not only its capability; verify crash and redeploy recovery, replay, timers, signals, and side-effect idempotency in that runtime. |
| retries, fallbacks, and call limits | Pydantic AI tool/output retries, provider transport retries, `FallbackModel`, `UsageLimits`, Harness `SpendLimits`, and application or durable-workflow policy | Classify expected versus unexpected errors and retry scope. Match backoff, fallback order, tool-error visibility, counters, terminal propagation, and side-effect idempotency. |
| LangGraph and Deep Agents event streaming | Pydantic AI streaming APIs plus capability event handlers | Translate the application's selected modes and projections. Namespaced graphs and subagent handles with child identity, status, messages, tool calls, nested work, and output are not preserved automatically. |

Explicit designs for gaps:

- **Forked child context or compiled LangGraph children:** pass selected context explicitly, rebuild the child with Pydantic AI primitives, or retain it behind a typed tool boundary during migration.
- **Background subagents:** keep the queue and worker in the host. Give the agent narrow start, list, inspect, steer, and cancel tools with tenant checks, idempotency keys, bounded retries, and durable results.
- **Approvals:** test approve, deny, changed arguments, stale decisions, authorization changes, crash before and after execution, replay, and side-effect idempotency.
- **Checkpoints:** separate message continuation, capability state, plans, files, approvals, domain state, and side effects. Do not replace a LangGraph checkpointer keyed by `thread_id` with `StepPersistence` without designing thread identity, `message_history` continuation, and ownership of the remaining state.
- **Streaming:** verify lineage, ordering, backpressure, redaction, child and tool events, error termination, and final-output timing.

## Hosts and protocols

Migrate the underlying agent separately, then preserve the external contract with an adapter.

| Source surface | Target direction | Required check |
| --- | --- | --- |
| Deep Agents Code or another CLI host | `Coder` or narrower Harness capabilities behind the existing command boundary | Inventory config and environment precedence, hooks and plugins, skills and memory roots, MCP servers, sandbox selection, approval modes, sessions, interactive and headless output, exit status, cancellation, and updates. Do not treat CLI behavior as `create_deep_agent` behavior. |
| frontend or AG-UI client | Pydantic AI UI adapters plus an application event/state adapter | Preserve coordinator messages, subagent projections, todo and custom state, tool-call lifecycle, interrupts, thread identity, reconnect behavior, files, sandbox artifacts, and final-result timing. |
| Agent Client Protocol | the experimental Harness ACP adapter when its session contract matches, otherwise an application adapter | Verify protocol negotiation, stdio lifecycle, sessions, content blocks, tool presentation, filesystem and terminal ownership, approvals, cancellation, history, errors, and client capability fallbacks. |
| A2A endpoint or client | an application server/client adapter around the migrated agent | Preserve the protocol version, agent card, methods, task and context identity, history, streaming, file parts, authentication, cancellation, tracing, and conformance behavior. |
| managed deployment | the selected host and application services | Preserve API and stream schemas, thread/run/store semantics, queues, schedules, tenancy, authentication, secrets, regional and data-retention rules, observability, retries, rollout, and recovery. |

If the target lacks a protocol feature, retain the existing adapter or propose a focused adapter built on public Pydantic AI run, message, event, and deferred-tool primitives. Do not bury a wire-level behavior change inside the agent migration.

## A tested Harness composition

Use this smoke test when a local coding-agent composition is plausible; `Coder(workspace)` is shorter when its documented defaults match the source. Run it in the exact target environment: it proves that imports and public constructor signatures resolve against the locked versions, not that behavior matches the source application.

```python {noqa="I001"}
# ruff: noqa: I001
import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.compaction import SlidingWindowCompaction
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.skills import Skills
from pydantic_ai_harness.subagents import SubAgent, SubAgents


async def stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    yield 'Workspace ready.'


worker = Agent(TestModel(call_tools=[]), name='researcher', description='Research a bounded question.')

with TemporaryDirectory() as workspace:
    root = Path(workspace)
    skill = root / 'skills' / 'inspect-workspace'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text(
        '---\nname: inspect-workspace\ndescription: Inspect a workspace.\n---\nUse read-only tools.',
        encoding='utf-8',
    )
    migrated = Agent(
        FunctionModel(stream_function=stream),
        output_type=str,
        instructions='Work only inside the configured workspace.',
        capabilities=[
            Planning(),
            FileSystem(root, read_only=True),
            RepoContext(workspace_dir=root),
            Skills(root / 'skills'),
            SubAgents(agents=[SubAgent(worker, max_calls=1)], agent_folders=None),
            SlidingWindowCompaction(max_messages=20, keep_messages=10),
        ],
    )
    result = asyncio.run(migrated.run('Describe the available workspace tools.'))
    assert isinstance(result.output, str)
    assert result.all_messages()
```

Use `TestModel(call_tools=[])` for construction-only checks and a `FunctionModel` with scripted tool calls when a test claims tool behavior. Event-aware capabilities such as `RepoContext` can make `Agent.run()` use the streamed model path, so provide `stream_function` and execute the complete composition once rather than inferring the path from the model stub.

## Migration traps

- Treating a `deepagents` import as out of scope, or as a reason to hand the migration to another skill, instead of mapping the bundled features it actually uses.
- Translating `create_deep_agent(...)` arguments without snapshotting the profile-resolved prompt, tools, and middleware.
- Passing a LangChain `BaseChatModel` or LangChain tools through unchanged.
- Replacing a sandbox backend with local `Shell` or `FileSystem`, or treating path checks and allowlists as the isolation boundary.
- Turning `/memory/...` file paths into Harness `Memory` without preserving the tool names, paths, and injection role callers depend on.
- Replacing a LangGraph checkpointer with `StepPersistence` or `message_history` alone.
- Assuming `write_todos`, the general-purpose child, or summarization exists on the source without checking the installed version and profile.
- Reproducing `AsyncSubAgent` with synchronous `SubAgents` or `DynamicWorkflow`.

Primary sources: [Deep Agents architecture](https://github.com/langchain-ai/deepagents/blob/main/libs/ARCHITECTURE.md), [`create_deep_agent` source](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py), [profiles](https://docs.langchain.com/oss/python/deepagents/profiles), [backends](https://docs.langchain.com/oss/python/deepagents/backends), [permissions](https://docs.langchain.com/oss/python/deepagents/permissions), [sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes), [subagents](https://docs.langchain.com/oss/python/deepagents/subagents), [async subagents](https://docs.langchain.com/oss/python/deepagents/async-subagents), [human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop), [Deep Agents Code](https://docs.langchain.com/oss/deepagents/code/overview), [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/), [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/), [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/), and [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/).
