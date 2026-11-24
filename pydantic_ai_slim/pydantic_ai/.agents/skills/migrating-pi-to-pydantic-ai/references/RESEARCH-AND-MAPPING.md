# Research and concept mapping

Use this reference conditionally after tracing the source path. It is a decision guide, not a requirement to reproduce the whole Pi host.

## Primary documentation

Inspect installed versions before migrating because Pi, Pydantic AI, and Harness continue to evolve.

- Pi: [README](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/README.md), [extensions](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/extensions.md), [SDK](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/sdk.md), [sessions](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/sessions.md), [compaction](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/compaction.md), and [skills](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/skills.md).
- Pydantic AI: [agents](https://pydantic.dev/docs/ai/core-concepts/agent/), [capabilities](https://pydantic.dev/docs/ai/capabilities/overview/), [custom capabilities](https://pydantic.dev/docs/ai/capabilities/custom/), [hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/), [message history](https://pydantic.dev/docs/ai/core-concepts/message-history/), and [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/).
- Pydantic AI Harness: [overview](https://pydantic.dev/docs/ai/harness/), [Coder](https://pydantic.dev/docs/ai/harness/coder/), [skills](https://pydantic.dev/docs/ai/harness/skills/), [guardrails](https://pydantic.dev/docs/ai/harness/guardrails/), [subagents](https://pydantic.dev/docs/ai/harness/subagents/), and [step persistence](https://pydantic.dev/docs/ai/harness/step-persistence/).

Pi is TypeScript and Pydantic AI is Python. Capture existing RPC, event, provider, and persisted-record boundaries with language-neutral fixtures and preserve field names with explicit adapters where required.

## Extension-to-capability decision table

| Pi extension responsibility | Target owner and likely seam | What must be proved |
|---|---|---|
| `registerTool()` with instructions and related policy | **Core:** typed function, `Tool`, toolset, or declarative `Capability`; **Harness:** existing focused capability when semantics match | Tool name/schema, guidance, validation, progress/result/error shape, cancellation, parallelism, output limits, and effects. |
| Reusable extension behavior spanning tools, instructions, and hooks | **Core:** custom `AbstractCapability`, optionally composed from smaller capabilities | Run binding, hook ordering, emitted typed capability events, state lifetime, errors, and composition with sibling capabilities. |
| `before_agent_start` system-prompt injection | **Core:** static/dynamic instructions or a custom capability hook; **Harness:** `SystemReminders` for matching cache-safe reinjection | Exact firing frequency, trusted data interpolation, cache impact, and whether injected content persists in history. |
| `context` message rewriting or custom compaction | **Core:** `ProcessHistory`; **Core/Harness:** a matching compaction strategy; **Application:** persisted summary/branch policy | Which messages are removed/mutated, summary boundary, retries, token thresholds, restore behavior, and visibility to callers. |
| `tool_call` mutation/blocking | **Core:** tool preparation or toolset wrapper for schema/args; **Harness:** `ToolGuardrail`; **Core:** deferred approval for protected effects | Pre-validation vs post-validation timing, handler order, mutation revalidation, block reason, termination, authorization, and side effects. |
| `tool_result` modification | **Core:** toolset wrapper or custom capability around the tool lifecycle; **Harness:** tool guardrail/output-limit capability where matching | Result content/details/error/usage, order, retries, persistence, stream events, and terminal behavior. |
| Model/message/tool lifecycle listeners | **Core:** `Hooks`, `ProcessEventStream`, `@on_event`, or capability events; **Application:** observability/event adapter | Event set and order, sync/async timing, listener errors, stream activation, cancellation, and public serialization. |
| `registerCommand`, shortcuts, flags, dialogs, widgets, renderers, editor/footer/theme changes | **Application/interface host**, such as a custom CLI, web UI, ACP host, or retained Pi frontend | Command routing, input/output, keyboard behavior, mode fallbacks, accessibility, cancellation, and terminal snapshots where contractual. |
| `appendEntry`, labels, session name, branch/tree navigation | **Application:** session/log/navigation model; **Core:** normalized messages only; **Harness:** `StepPersistence` only for matching run continuation | Append-only/tree behavior, stable IDs, branches, labels, abandoned history, extension data, restart, and migration of existing JSONL. |
| `sendMessage`, `sendUserMessage`, steering, follow-up queues | **Core:** explicit run messages, `RunContext.enqueue()` from in-run tools/capabilities, or `AgentRun.enqueue()` from external drivers where lifecycle matches; **Application:** queue/session control | Delivery point, trigger behavior, expansion, concurrent input, retries/compaction interaction, order, and cancellation. |
| `registerProvider`, model catalog/auth, payload/header hooks, custom streaming | **Core/model integration or Application transport**, not a generic capability by default | Credential precedence, refresh, model selection, request/response wire fixtures, usage, errors, abort, tool calls, and streaming. |
| Resource discovery and reload | **Application:** configuration/discovery/reload; **Harness:** construct selected capability/`Skills` instances per run/process | Search roots, precedence, trust, snapshots vs hot reload, diagnostics, and behavior for removed/changed resources. |
| Pi package manifest and `pi install` | **Distribution/Application:** Python package exposing capabilities plus optional host adapters | Installed assets, dependency/extras boundary, enable/disable configuration, trust, version pin/update behavior, and wheel contents. |
| Project trust | **Application security policy** | Which files/code/settings load before and after trust, persistence of decisions, non-interactive behavior, and confirmation that trust is not a sandbox. |
| Shared extension event bus | **Application event bus** for host coordination; **Core capability events** only for typed agent-run coordination | Names, payloads, scope, listener order/errors, restart behavior, and whether events enter model/UI streams. |

## Agent and runtime ownership map

| Observed Pi behavior | Target owner and likely seam | Focused proof |
|---|---|---|
| `createAgentSession().prompt()` and final assistant message | **Core:** reusable `Agent.run()`; **Application:** result adapter | Assert public input/output/error, messages, usage/cost, tool effects, and model selection. |
| Pi's default read/write/edit/bash coding loop | **Harness:** `Coder` only when Coder's exact six-tool composition and unrestricted host shell match the required behavior; otherwise compose `FileSystem`, `Shell`, `RepoContext`, context controls, and custom tools | Compare exact tool schemas, edit semantics, cwd, ignores/protected paths, command allowlisting or rejection, truncation, process lifetime, environment, and cancellation. |
| Pi `AgentSession` event subscription | **Core:** `run_stream_events()`, `iter()`, agent/capability/custom events; **Application:** event adapter | Golden-test only consumed event types/fields, order, reconstruction, usage, terminal detection, errors, and cancellation. |
| Pi session messages | **Core:** serialized `ModelMessage` history; **Application:** session ownership/store | Continue from real persisted records in a fresh process and test incomplete tool calls and cross-session isolation. |
| JSONL session tree, fork/clone/switch, compaction/branch entries, model/thinking changes | **Application:** explicit session format and navigation; **Harness `StepPersistence`:** only when agent run resume/fork semantics fit | Test every consumed record type, branch restoration, compaction context, stable cursors, crash recovery, and old-record migration. |
| Automatic retry and overflow compaction | **Core/provider retry settings plus Core/Harness compaction and application loop policy** | Assert triggering error classes, delay/limits, cancellation, failed-message handling, retry count, and context after compaction. |
| Skills and `/skill:name` | **Harness:** `Skills` for configured `SKILL.md` instructions; **Application:** command/discovery layer and resources/scripts | Assert explicit library roots, selected catalog, instruction loading, missing resources, reload behavior, trust, and behavioral frontmatter differences. |
| Dynamic tool loading via `setActiveTools()` | **Core:** `ToolSearch`, deferred `Capability`, or per-step tool preparation | Assert initial catalog, selection, next-request availability, removals, provider-native/fallback protocol, and prompt-cache effects. |
| Plan-mode extension | **Harness:** `Planning` when the model-owned plan lifecycle fits; **Application:** command/widget; otherwise custom capability/state | Assert plan ownership, updates, reminders, persistence, user edits, UI, and behavior across compaction/restart. |
| Subagent extension or spawned Pi process | **Harness:** `SubAgents` for model-directed isolated delegation; **Application:** subprocess orchestration when process/session semantics matter | Assert task input, tools/capabilities, history isolation, model choice, result/events, budgets, cancellation, errors, and recursion. |
| Permission/protected-path extension | **Harness:** guardrail and/or **Core:** approval; **Application:** trusted policy and approver UI | Exercise allow/block/approve/deny, symlink/path cases, identity, audit, and exactly-once protected effect. |
| Sandbox or SSH extension | **Harness:** `FileSystem`, `Shell`, or `ModalSandbox`; **Application:** selected remote/sandbox backend | Exercise filesystem/shell operations, cwd, network, credentials, timeouts, cleanup, and real containment against escapes. |
| Pi RPC or JSON mode | **Application:** preserve protocol with a Python adapter or intentionally adopt a Pydantic AI UI protocol | Run the same language-neutral request/event fixtures, including errors, mid-run input, abort, command listing, and session operations. |
| Pi TUI | **Application/interface:** retain Pi temporarily, build a host, use `clai`, ACP, web chat, or another adapter according to observed needs | Test only user-facing behavior in scope; capability parity does not prove terminal UI parity. |
| Pi skills/prompts/extensions/packages as one installable package | **Python distribution:** capabilities, skill assets, and host integration can ship together but remain separate runtime owners | Inspect wheel, import selected capabilities, discover skill assets explicitly, and test host startup without dev dependencies. |
| Telemetry and extension logging | **Core:** instrumentation/OpenTelemetry; optional Logfire; **Application:** existing backend and logs | Assert correlation, content/privacy, required spans/events, nested usage, retention, dashboards, and alerts before switching. |

## Common plugin ports

- **Custom tool extension:** start with a typed tool. Use a declarative `Capability` only when the tool and its instructions should travel together; subclass only if hooks or events are also required.
- **Permission gate:** separate policy evaluation, approval pause, and approver UI. Usually this is a guardrail or tool wrapper plus core deferred approval plus an application interface—not one monolithic capability.
- **Plan mode:** use Harness `Planning` for model-owned plans, while slash commands and widgets remain interface code.
- **Subagent package:** use Harness `SubAgents` only when Pi's subprocess/session behavior is not itself contractual.
- **Coding/sandbox package:** compare `Coder` first, then compose smaller filesystem/shell/repo-context capabilities. Add a real sandbox independently when untrusted execution is in scope.
- **Provider package:** implement the model/provider transport and keep login/catalog UI in the host; do not route provider payload rewriting through an agent capability unless it is genuinely per-run policy.
