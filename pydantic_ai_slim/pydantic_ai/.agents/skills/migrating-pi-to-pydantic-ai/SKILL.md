---
name: migrating-pi-to-pydantic-ai
description: Migrate TypeScript Pi coding-agent applications, extensions, or packages to Python with Pydantic AI and Pydantic AI Harness. Use when source code imports `@earendil-works/pi-*`, calls `createAgentSession`, registers Pi extensions, or relies on Pi tools, hooks, skills, sessions, compaction, providers, TUI, RPC, or packages.
---

# Migrate Pi to Pydantic AI

Preserve observable behavior, not Pi's TypeScript API. Migrate the smallest complete caller path and keep product/host infrastructure in the application.

## Trace the source before choosing a target

1. Read repository instructions, `package.json`, Pi settings, tests, extension/package manifests, and runtime entrypoints. Record the installed Pi, Pydantic AI, and Harness versions.
2. Trace one real request from the Pi CLI, RPC boundary, or `createAgentSession().prompt()` through resource discovery, system instructions, tools, extensions, model requests, messages, compaction/retry, events, session entries, UI/RPC output, and side effects. Establish a focused baseline or characterization test.
3. Inventory every active extension factory and package resource. For each `pi.on(...)`, `registerTool`, `registerCommand`, provider registration, renderer/UI contribution, skill, prompt, and persisted entry, record its firing point, trusted inputs, mutations, output, state owner, and caller-visible effect.
4. Separate these contracts when present:
   - model messages, append-only session entries/tree branches, compaction summaries, and extension state;
   - model-chosen tool input, trusted host context, project trust, approval, authorization, and OS isolation;
   - token deltas, tool lifecycle, Pi extension events, capability events, RPC events, and terminal rendering;
   - agent-loop behavior, CLI/TUI host behavior, provider transport, package distribution, and external side effects.
5. Record each observed contract, its owner, semantic difference, and executable proof. An installed but inactive Pi package feature is not migration scope.

Read [Research and concept mapping](references/RESEARCH-AND-MAPPING.md) for the detected features. Read [Verification and cutover](references/VERIFICATION-AND-CUTOVER.md) before implementation.

## Treat extensions as capability candidates

A Pi extension is the closest source concept to a Pydantic AI capability, but it is broader. An extension can combine agent behavior with terminal UI, commands, provider registration, resource discovery, and host lifecycle. **Port responsibilities, not the extension file:**

1. Map reusable model-facing behavior—tools, instructions, model settings, history/event processing, guardrails, and agent-loop hooks—to an existing Core or Harness capability when its lifecycle matches.
2. Bundle related instructions and tools in core `Capability`; subclass `AbstractCapability` only for reusable behavior that needs lifecycle hooks, adaptive models/settings, native tools, or typed capability events.
3. Keep CLI commands, flags, keyboard shortcuts, TUI renderers/dialogs, session selection, project trust, package installation, and provider credential setup in the Python application or interface adapter. They are not agent capabilities merely because a Pi extension owns them.
4. Split mixed extensions at that seam. A permission extension may become a `ToolGuardrail` or approval capability plus an application-owned approver UI; a coding package may become `Coder` plus host configuration; a provider extension remains a model/provider integration.
5. Preserve order only where evidence shows it matters. Pi handler load order and Pydantic AI capability/hook ordering are different contracts.

## Choose the smallest target

- **Core:** use `pydantic_ai.Agent` for the agent loop, typed dependencies, tools/toolsets, outputs, normalized messages, capabilities/hooks, streaming, approvals, usage limits, instrumentation, MCP, and provider/model integration.
- **Harness:** use `Coder` for Pi's ordinary coding tools only when its exact composition fits. Add focused capabilities such as `FileSystem`, `Shell`, `Skills`, `Planning`, `SubAgents`, guardrails, compaction, tool-output limits, or step persistence only for observed behavior.
- **Application/interface:** retain or replace CLI/TUI/RPC, auth, project trust, settings, provider login/catalogs, session browsing, package management, deployment, and transport deliberately.
- **Graph:** use plain async Python or `pydantic_graph` for deterministic workflows; do not encode them in prompts or subagents.
- **Gap:** name behavior with no supported public seam, explain its impact, and test a bounded adapter. Do not clone Pi's extension bus or JSONL format just to claim parity.

The normal embedded migration is one reusable `Agent`, `Coder` or a smaller capability composition, application services in typed dependencies, an application-owned message/session store, and a thin adapter at the existing RPC or UI boundary.

## Apply high-risk gates

- `ctx`, project trust, credentials, session managers, and service handles are trusted host context. Put equivalents in typed dependencies or application services, never model-chosen tool arguments.
- Pi session JSONL is an append-only branching host log containing more than model context. `result.all_messages()` preserves model history, not tree navigation, labels, extension entries, compaction records, model changes, queues, or abandoned branches. Choose and test conversion, a compatibility store, or an accepted fresh start.
- Pi compaction, context interception, steering/follow-up queues, retries, and branch summaries have specific timing. Select matching core/Harness seams independently and test ordering; generic message history or summarization is not automatic parity.
- Map `tool_call` permission gates by effect. Use guardrails for validation/block/redaction and deferred tools for an action that must await approval. Keep identity, authorization, UI, audit, persistence, and idempotency in the application.
- Pi extensions run arbitrary TypeScript with host permissions. A Pydantic AI capability also runs application code; neither is a sandbox. Use an OS/container/cloud isolation boundary for untrusted commands or code.
- Harness `Skills` loads configured `SKILL.md` instructions on demand but does not reproduce Pi's discovery roots, resource files, scripts, reload, package installation, or behavioral frontmatter. Preserve those separately when observed.
- Use `SubAgents` only for model-directed isolated tasks. Keep deterministic orchestration and Pi subprocess/tmux/package-specific semantics in application code.
- Map dynamic tool activation to core `ToolSearch` or on-demand capabilities only after testing load timing, schemas, prompt/cache changes, and provider fallback behavior.
- Pydantic AI run events and capability events do not reproduce Pi's extension event bus, RPC protocol, or TUI render lifecycle. Adapt only stable fields consumers use.
- Pi provider extensions and payload hooks may alter authentication, catalogs, wire payloads, headers, and streaming. Implement them at the Pydantic AI model/provider or application transport layer and run provider contract tests; do not hide them in a generic capability.

## Implement and prove one vertical slice

1. Preserve the supported CLI, RPC, SDK, job, or UI boundary and replace only agent-owned internals.
2. Start with core plus the smallest coding capabilities. Add broader Harness behavior only after a traced contract requires it.
3. For each source extension, document the split: capability behavior, application/interface behavior, retained integration, and gap. Test each side through its real boundary.
4. Use language-neutral fixtures for RPC/events and persisted records. Test tool arguments/results, errors, event order, session continuation, UI decisions, and side effects. Use deterministic models offline; add focused recorded/live provider tests only for provider behavior.
5. Remove Pi packages, settings, Node runtime, or extension adapters only after no retained path needs them.

Explain consequential semantic changes before implementing them: state the Pi behavior, target behavior, caller impact, recommended choice, and remaining risk.

## Completion

Apply the completion criterion in [Verification and cutover](references/VERIFICATION-AND-CUTOVER.md). Label fake, recording, live-provider, terminal, restart, and sandbox evidence accurately.
