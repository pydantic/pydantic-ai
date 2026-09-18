# Semantic Gaps

Use this reference only when the active slice reaches one of these boundaries. For each gap, state the source behavior, target behavior, user-visible impact, chosen owner, and executable proof.

## Sessions are several contracts

An ADK `Session` holds an event log and mutable state, while a `SessionService` applies state deltas and records events containing artifact-version deltas. State prefixes define session, user, application, and temporary lifetimes. `MemoryService` is a separate searchable cross-session archive, and artifacts are separately versioned binary data.

Pydantic AI [`message_history`](https://pydantic.dev/docs/ai/core-concepts/message-history/) is normalized model conversation context. It does not implement ADK state scopes, memory ingestion/search, artifact versioning, event IDs, or session CRUD. Split those responsibilities, preserve current stores where possible, and test tenant isolation and reload behavior.

ADK session [rewind](https://adk.dev/sessions/session/rewind/) restores only session-scoped state and artifacts, retains an audit log, and does not undo external side effects. Do not describe history truncation or a graph retry as rewind parity.

## Workflow execution and resumability

ADK 2.x has graph workflows, dynamic workflows, and older prebuilt sequential/parallel/loop agents. Similar diagrams do not prove the same scheduling, branch isolation, failure, retry, or output behavior. Use plain async Python for simple deterministic code; use `pydantic_graph` when an explicit typed graph remains useful.

ADK [resume](https://adk.dev/runtime/resume/) records completed node/tool events, skips recorded work, and can run tools more than once. Pydantic AI message replay is not workflow resume. When restart is required, choose a supported [durable integration](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) or a deliberately scoped persistence capability. Test interruption at each effect boundary and make non-repeatable effects idempotent.

Parallel ADK branches isolate conversation history but may share session state. Do not replace that with `asyncio.gather()` until tests define input visibility, write conflicts, event interleaving, cancellation, and partial failure.

## Callbacks and plugins

ADK callbacks are configured per agent; plugins apply runner-wide. Callback lists and plugin lists run in registration order. A non-`None` plugin result skips later plugins and the corresponding agent callback. Exceptions in `on_agent_error_callback` and `on_run_error_callback` are best-effort notifications; exceptions in other plugin callbacks fail the run.

Pydantic AI [`Hooks`](https://pydantic.dev/docs/ai/core-concepts/hooks/) compose through capability ordering and use explicit exceptions such as `SkipModelRequest`, `SkipToolExecution`, or recovery hooks. Consolidate order-sensitive ADK middleware into one owner when necessary. Test order, mutation visibility, short-circuit output, error recovery, and cleanup rather than translating callback names mechanically.

## Agent transfer and delegated tasks

ADK 2.x [collaboration behavior](https://adk.dev/workflows/collaboration/) depends on `mode`; 1.x `sub_agents` use `chat` behavior. In `chat`, `transfer_to_agent` moves control within the same session and conversation, and the target can answer the user until another transfer. In `task`, the child uses an isolated session branch, may ask the user for clarification, and returns control through `finish_task`; `single_turn` takes no user interaction, returns immediately, and may run in parallel. Harness `SubAgents` gives a child a self-contained task with its own message history and returns its string output to the parent. Use core hand-off patterns for `chat`; consider a typed agent tool or Harness `SubAgents` for `task` or `single_turn` only after testing history, return control, interaction, concurrency, and final-answer ownership.

## Human input, approval, and authentication

ADK workflow `RequestInput`, tool confirmation, and tool authentication all pause execution, but they protect different contracts. A text answer is not authorization. A confirmation is not an identity proof.

Use Pydantic AI [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) for protected tool approval or external execution. When the decision is available during the same call, use `HandleDeferredToolCalls` so the run can continue inline. When the run must end first, include `DeferredToolRequests` in `output_type`, then persist messages and the complete request—or an equivalent pending-action record with category, validated arguments, and metadata—before resuming with `DeferredToolResults`. Keep authenticated principals, credentials, and policy services in trusted dependencies/application code. Prove approve, deny, timeout, duplicate response, restart, and no-side-effect-before-approval behavior as applicable.

## Events and streaming

ADK's `Event` is simultaneously a stream item, persisted session record, content carrier, action delta, workflow signal, and correlation record. Partial events are normally forwarded without applying their state deltas. Each non-partial event is appended independently and applies its own state delta at that point; this does not wait for a final-response event. Saving an artifact persists it first, then records the returned version in the current event's `artifact_delta`, so a later failure can leave an artifact without a corresponding persisted event.

Pydantic AI exposes output streaming and typed lifecycle events, but its event classes and chunk boundaries are not ADK's wire protocol. Keep an adapter when UI or API clients consume ADK event fields such as `invocation_id`, `author`, `branch`, `actions`, or final-response detection. Test ordered serialized events and cancellation, not only the concatenated final text.

## Skills and execution environments

ADK `SkillToolset` can load skill instructions, resources, and scripts, and may run scripts through a configured executor. Harness [`Skills`](https://pydantic.dev/docs/ai/harness/skills/) loads `SKILL.md` instructions only; it does not expose bundled resources or execute scripts. Preserve resource/script behavior with reviewed application tools or record a gap.

ADK code executors range from unsafe local execution to managed sandboxes. Harness `FileSystem` and `Shell` provide workspace policy, not process isolation. Use an isolated execution environment such as [`ModalSandbox`](https://pydantic.dev/docs/ai/harness/modal-sandbox/) when containment is part of the contract, and test filesystem/network/credential boundaries.

## Provider-native and platform features

Gemini tools, context caching, live audio, Agent Engine services, ADK deployment commands, A2A, and evaluation formats have provider or platform semantics beyond an ordinary agent loop. Prefer supported Pydantic AI provider-native tools and realtime APIs only after probing the installed versions. Retain existing platform adapters or declare a gap where event, credential, deployment, or managed-service behavior cannot be preserved.
