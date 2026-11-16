---
name: migrating-claude-agent-sdk-to-pydantic-ai
description: Migrate Python applications from the Claude Agent SDK to Pydantic AI and, only when needed, Pydantic AI Harness. Use when source code imports `claude_agent_sdk` or relies on Claude Code's agent loop, sessions, built-in tools, hooks, permissions, skills, or subagents. Do not use for the Anthropic Messages SDK or Claude Managed Agents.
---

# Migrate Claude Agent SDK to Pydantic AI

Preserve observable behavior, not the Claude Code process or API shape. Migrate the smallest complete caller path and keep application infrastructure in place.

## Trace the source before choosing a target

1. Read repository instructions, dependencies, tests, and the runtime entrypoint. Record the installed Claude Agent SDK, bundled Claude Code CLI, Pydantic AI, and Harness versions.
2. Trace one real request from `query()` or `ClaudeSDKClient` through prompts, tools, events, results, state, persistence, approvals, and side effects that callers use. Establish a focused baseline or characterization test.
3. Separate these source contracts when present:
   - fresh `query()` calls, live `ClaudeSDKClient` continuation, disk/external-store resume, transcript forks, and file rewind;
   - tool availability, automatic approval, policy callbacks, and OS/container isolation;
   - completed messages, raw token deltas, lifecycle events, and terminal results;
   - conversation history, workflow checkpoints, model-owned plans, long-term memory, and workspace state;
   - model-directed subagents from application-owned loops, queues, retries, and schedulers.
4. Record each observed contract, its owner, semantic difference, and executable proof. An unobserved Claude Code feature is not migration scope.

Read [Research and concept mapping](references/RESEARCH-AND-MAPPING.md) for the detected source features. Read [Verification and cutover](references/VERIFICATION-AND-CUTOVER.md) before implementation.

## Choose the smallest target

- **Core:** use `pydantic_ai.Agent` for model calls, the agent loop, typed dependencies, tools, outputs, normalized messages, generic hooks, streaming, MCP, approvals, usage limits, instrumentation, and durable-runtime integrations.
- **Harness:** add `pydantic-ai-harness` only for observed reusable policy such as coding filesystem/shell tools, repository context, planning, memory, model-directed subagents, model-agnostic context management, or Agent Skills. Harness capabilities compose through the core agent loop; Harness is not a second runtime.
- **Application:** retain authentication, service clients, databases, queues, schedulers, deployment, product state, session lookup, tenant policy, and existing transports unless the requested slice includes them.
- **Gap:** name behavior that no supported public seam preserves, explain its impact, and test a bounded adapter. Do not create a lookalike `ClaudeSDKClient` or emulate Claude transcript files merely to hide a difference.

The normal migration is one reusable `Agent`, application services supplied through typed dependencies, ordinary Pydantic AI tools, a typed output when the caller expects structure, and application-owned storage of `result.all_messages()` when later turns need context. Preserve the existing public request/response/event boundary with a small adapter while callers migrate.

## Apply high-risk gates

- `allowed_tools` in the Claude Agent SDK means auto-approval, not exclusive availability and not isolation. Map availability, approval, policy, and sandboxing independently.
- Pydantic AI `message_history` explicitly replays normalized messages. `conversation_id` correlates runs; it is not a Claude session-resume token. Prove restart and fork behavior rather than renaming IDs.
- Pydantic AI approval defers a validated tool call. The application still owns authenticated authorization, identity, and audit, and the protected effect must not run before approval.
- Map hooks by firing point, inputs, ordering, mutation, blocking, and error behavior. Claude hook matchers may run concurrently. Pydantic AI runs `before_*` hooks in capability order, `after_*` hooks in reverse order, and nests `wrap_*` hooks with the first capability outermost. Require a golden trace for any hook-visible contract.
- Map consumer intent for streaming. Pydantic AI output streaming, run-event streaming, and graph iteration are different surfaces; none promises the Claude SDK message taxonomy.
- Use Harness `SubAgents` only for model-directed delegation. Keep deterministic orchestration and verification loops in application code.
- Harness `Skills` supplies on-demand `SKILL.md` instructions; verify resources and scripts separately instead of assuming Claude Code skill or plugin loading parity.
- Harness command controls are best-effort policy, not an OS security boundary. For untrusted execution, use `ModalSandbox` where it fits or retain application-owned container, VM, or sandbox isolation.
- File rewind, Claude Code prompt presets, plugin packaging, raw stream-event identity, and dynamic MCP control are gaps unless the migrated caller actually needs them and a public target seam is proven.

## Implement and prove one vertical slice

1. Preserve the supported caller boundary and replace only the agent-owned internals.
2. Start core-only. Add the narrowest Harness capabilities only after a source contract requires them.
3. Test inputs, typed outputs, errors, event order, tool arguments/results, approval decisions, state across turns, and side effects at that boundary. Use deterministic models and fake application services for offline tests; add a focused recorded/live test only when provider behavior is the contract.
4. For resume, durability, approvals, concurrent tools, or external effects, test interruption/restart, lineage, authorization, and idempotency at the exact boundary promised by the source.
5. Remove `claude-agent-sdk` only after no retained path imports it, spawns Claude Code, reads its transcripts, or consumes its event/result types.

Explain any consequential semantic change before implementing it: state the source behavior, target behavior, caller impact, recommended choice, and remaining risk.

## Completion

Apply the completion criterion in [Verification and cutover](references/VERIFICATION-AND-CUTOVER.md). Label evidence from fakes, recordings, and live providers accurately.
