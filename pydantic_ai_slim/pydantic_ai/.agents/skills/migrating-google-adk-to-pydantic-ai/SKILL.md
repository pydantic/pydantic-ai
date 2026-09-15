---
name: migrating-google-adk-to-pydantic-ai
description: Migrate Python Google Agent Development Kit (ADK) applications to Pydantic AI. Use for `LlmAgent`, `Runner`, sessions, state, memory, tools, callbacks, plugins, graph or dynamic workflows, resumability, events, artifacts, MCP, and ADK deployment boundaries.
---

# Migrate Google ADK to Pydantic AI

Preserve caller-visible behavior, not ADK's class tree. Migrate the smallest complete request path and leave product infrastructure in the application.

## Work from the running application

1. Read repository instructions, dependencies, tests, and runtime entrypoints. Record the installed Google ADK, Pydantic AI, and optional `pydantic-ai-harness` versions.
2. Trace one real request from `Runner.run_async()` or the deployed endpoint through the root agent, instructions, collaboration mode, model calls, tools, workflow nodes, callbacks/plugins, session service, emitted events, state/artifact deltas, and final response. Record only behavior that path uses.
3. Separate the contracts before choosing targets:
   - `Session.events` used as model context;
   - session-, user-, app-, and invocation-scoped state;
   - searchable long-term memory;
   - workflow/node checkpoints and resume IDs;
   - versioned artifacts and external side effects.
4. Classify the active slice:
   - **Ordinary `LlmAgent`:** normally one reusable [`Agent`](https://pydantic.dev/docs/ai/core-concepts/agent/) with typed dependencies, tools, and output.
   - **ADK graph or dynamic workflow:** keep simple deterministic control flow in plain async Python; use [`pydantic_graph`](https://pydantic.dev/docs/ai/graph/graph/) when explicit typed nodes, branching, or graph inspection remain valuable.
   - **Multi-agent delegation:** inspect ADK's collaboration `mode` in Python 2.x; 1.x `sub_agents` use `chat` behavior. Use core [multi-agent patterns](https://pydantic.dev/docs/ai/guides/multi-agent-applications/) for `chat` transfer semantics. Harness [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/) can fit `task` or `single_turn` only when their isolated context, return, interaction, and concurrency behavior match.
   - **Product runtime:** retain auth, session/state stores, artifact stores, queues, transport, A2A endpoints, evaluation, observability, and deployment unless they are explicitly in scope.
5. Add deterministic characterization tests, then migrate one vertical slice behind its existing caller boundary.
6. Run the original tests and focused parity tests. Mark unexercised behavior `unverified`; similar names are not equivalence evidence.

Read [Concept Mapping](references/CONCEPT-MAPPING.md) for the features the slice uses. Read [Semantic Gaps](references/SEMANTIC-GAPS.md) for workflows, state, resume, confirmation, callbacks/plugins, event streams, skills, or execution environments. Read [Verification and Cutover](references/VERIFICATION-AND-CUTOVER.md) before removing ADK or changing production traffic.

## High-risk gates

- Do not pass ADK `Session` objects or mutable state dictionaries through model-chosen tool arguments. Put authenticated identity and service clients in typed dependencies; persist product and workflow state through application-owned stores.
- [`message_history`](https://pydantic.dev/docs/ai/core-concepts/message-history/) continues model context. It does not replace ADK session state, memory, artifacts, event records, node checkpoints, or invocation resume.
- ADK resumability replays recorded node/tool results and can execute tools more than once. Choose a [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) design explicitly and prove restart plus idempotency for side effects.
- Keep conversational input, approval, and authorization distinct. Map tool confirmation to [deferred tool approval](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/); keep identity and access checks in trusted application code.
- Preserve callback/plugin ordering and short-circuit rules deliberately. Pydantic AI hooks have their own capability ordering and exception-based skip/recovery semantics; a list of lookalike hooks is not parity.
- ADK partial events are delivered without applying state deltas; each non-partial event applies its delta when appended. Artifact writes happen during the artifact operation, before the current event records the returned version. If callers consume ADK event fields or final-event detection, retain a boundary adapter and test the exact stream and persistence order.
- Use Harness only for an observed reusable capability. Ordinary agents need core only, and a command allowlist is not an OS security boundary.

## Pydantic AI defaults

- Map `instruction` to `instructions`; use a `RunContext` instructions function when it depends on trusted runtime data.
- Map function tools to typed Pydantic AI tools. Preserve tool names, descriptions, validation behavior, error shape, retries, confirmation, and concurrency only where callers rely on them.
- Map `output_schema` to `output_type` when the terminal output contract is structured. Preserve an existing wire adapter if changing response shape would widen the migration.
- Persist [`ModelMessage`](https://pydantic.dev/docs/ai/core-concepts/message-history/) histories separately from workflow and product state. Reuse the application's current stores before adding a new persistence subsystem.
- Use [`Hooks`](https://pydantic.dev/docs/ai/core-concepts/hooks/) for local lifecycle interception or a custom capability for reusable policy that also owns tools, instructions, settings, or events.
- Keep the active model/provider unless changing it is requested. Use provider-prefixed Pydantic AI model IDs and verify provider-specific tools, settings, streaming, and realtime behavior against installed APIs.

## Completion

The slice is complete only when every observed caller contract is `preserved` by executable evidence, explicitly accepted as `changed`, or `not applicable`. Treat `unverified` and `blocked` contracts as unfinished. Do not remove `google-adk` while a retained runtime, session migration, deployment command, evaluation, or compatibility path still imports or invokes it.
