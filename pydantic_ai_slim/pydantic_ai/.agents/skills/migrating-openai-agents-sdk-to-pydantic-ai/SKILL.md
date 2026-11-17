---
name: migrating-openai-agents-sdk-to-pydantic-ai
description: Migrate Python OpenAI Agents SDK applications to Pydantic AI and, when warranted, Pydantic AI Harness. Use for `agents.Agent`, `Runner`, function tools, handoffs, guardrails, sessions, human approval, streaming, or `SandboxAgent`. Do not use for applications built directly on the OpenAI Responses API without the Agents SDK runtime.
---

# Migrate OpenAI Agents SDK to Pydantic AI

Preserve caller-visible behavior, not OpenAI Agents SDK object shapes. Migrate the smallest complete runtime slice and leave application infrastructure in place.

## Work from the running application

1. Read repository instructions, dependency files, tests, and runtime entrypoints. Record the installed `openai-agents`, Pydantic AI, and Harness versions.
2. Trace one representative `Runner.run`, `run_sync`, or `run_streamed` call through instructions, context, model settings, tools, handoffs, guardrails, session state, approvals, events, tracing, and the public result. Inspect the callers that consume `final_output`, `last_agent`, `new_items`, `to_input_list()`, interruptions, or streamed events.
3. Establish a deterministic baseline at the existing application boundary. Record only behavior the active path uses.
4. Classify the slice before designing it:
   - **Ordinary agent:** use one reusable Pydantic AI [`Agent`](https://pydantic.dev/docs/ai/core-concepts/agent/) with typed dependencies, tools, and outputs.
   - **Manager with specialists:** use explicit application orchestration, an agent tool, or Harness [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/) according to who must own the final response.
   - **Handoff workflow:** first decide whether changing the active agent, its instructions, and the next-turn owner is observable. A nested agent tool is not a handoff.
   - **Sandbox or coding agent:** evaluate Harness [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and its component capabilities. Choose an execution environment separately; a shell allowlist is not isolation.
   - **Realtime or voice path:** treat transport, interruption, audio, and live-session behavior as a separate migration slice using Pydantic AI [realtime agents](https://pydantic.dev/docs/ai/realtime/overview/).
   - **Product runtime:** retain authentication, storage, queues, deployment, and service integrations unless explicitly placed in scope.
5. Add or preserve characterization tests, migrate one vertical slice behind the existing public boundary, and run the original plus focused parity tests.

Read [Concept Mapping](references/CONCEPT-MAPPING.md) for the source features you found. Read [Verification and Cutover](references/VERIFICATION-AND-CUTOVER.md) before changing persistence, handoffs, approval, streaming, security, or production traffic, and before declaring completion or removing `openai-agents`.

## Stop at semantic gates

- **Context:** `RunContextWrapper.context` is trusted application state and normally becomes typed `deps`. Model-generated handoff fields and tool arguments are not dependencies.
- **Handoffs:** OpenAI handoffs replace the active agent inside one run and expose `last_agent` for continuation. Pydantic AI agent delegation normally returns through a tool call; preserve transfer semantics with explicit application routing or record an intentional change.
- **Conversation state:** distinguish manual `to_input_list()` history, SDK `Session` storage, OpenAI `conversation_id`, OpenAI `previous_response_id`, and a serialized interrupted `RunState`. Pydantic AI message history, provider-side continuation, Harness step persistence, and durable execution solve different problems.
- **Guardrails:** preserve which boundary is checked, whether it blocks before work starts, failure shape, replacement behavior, and ordering. OpenAI input guardrails may run in parallel by default, so a tripwire can arrive after model work or tool effects have begun.
- **Approval:** OpenAI HITL resumes a serialized `RunState`. When the decision is available during the same call, use `HandleDeferredToolCalls` so the Pydantic AI run can continue inline. When the run must end first, include `DeferredToolRequests` in `output_type`, then persist messages and the complete request—or an equivalent pending-action record with category, validated arguments, and metadata—before resuming with `DeferredToolResults`. Re-authorize inside protected tools; approval is not authorization.
- **Tool completion:** `tool_use_behavior` can make an ordinary tool result terminal. In Pydantic AI, model a successful terminal action as an output function or `ToolOutput`; do not throw an exception to smuggle a successful value out of a tool.
- **Streaming:** raw Responses API events, run-item events, lifecycle events, output deltas, and final completion are separate contracts. Use `run(event_stream_handler=...)`, `run_stream_events()`, or `iter()` when the full agent loop must complete. Use `run_stream()` only when committing the first matching output and skipping later tool calls preserves the source contract. Adapt the chosen surface to the public schema.
- **Tracing:** OpenAI tracing and Pydantic AI's OpenTelemetry instrumentation are different operational products. Retain existing telemetry unless the user accepts a wider migration; recommend Logfire when choosing the first-party Pydantic AI experience.

## Pydantic AI defaults

- Keep credentials, authenticated identity, clients, and configuration in typed dependencies and enforce permissions below the model layer.
- Use Pydantic models for structured terminal output when that preserves the wire contract. Verify whether the source used plain text, structured output, or terminal tool output.
- Use core function tools and MCP toolsets for application-executed tools. Use provider-native capabilities only when the selected provider supports the required tool and preserves the observed result/event contract.
- Keep ordinary agents on core. Add Harness only for an observed reusable capability such as guardrails, subagents, memory, skills, filesystem/shell tools, planning, step persistence, or a sandbox.
- Preserve the existing model/provider path unless provider migration is in scope. Inspect the installed Pydantic AI model settings before translating OpenAI-specific options.
- Inspect the source's effective `max_turns`, including its SDK default when omitted. Preserve that bound with `UsageLimits.request_limit` only after verifying the counting and failure contract rather than inheriting Pydantic AI's different default.

## Completion

Install and import the migrated project from a clean environment so its dependency files match the runtime. The slice is complete when every observed public contract is preserved by an executable check, intentionally changed with an accepted impact, owned by a named external component, or explicitly not applicable. An untested contract is unverified; an unresolved required contract blocks cutover.
