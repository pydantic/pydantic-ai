---
name: migrating-deep-agents-to-pydantic-ai-harness
description: Migrate Python LangChain Deep Agents applications to Pydantic AI and Pydantic AI Harness. Use when the source uses the upstream `deepagents` package or demonstrably reproduces its middleware, backends, skills, memory, subagents, or sandbox contracts. Use `migrating-langchain-to-pydantic-ai` for plain LangChain, LangGraph, or LCEL migrations and `pydantic-ai-harness` for greenfield Harness usage.
---

# Migrate Deep Agents to Pydantic AI Harness

Preserve the application's observed contracts, not the shape of `create_deep_agent`. Compose Pydantic AI primitives and Harness capabilities; keep infrastructure with the application that already owns it.

## Establish the source contract

1. Resolve the `deepagents` import origin before applying these mappings. Route a local module or project-owned `create_deep_agent` to the ordinary LangChain migration only when its implementation has ordinary LangChain or LangGraph semantics; vendored implementations that reproduce Deep Agents contracts remain in scope. Read repository instructions, manifests, lockfiles, tests, runtime entrypoints, and the resolved source. Record exact Deep Agents, LangChain, LangGraph, Pydantic AI, and Harness versions.
   If the locked source is unavailable, reproduce its environment without changing the target lockfile. If that is not possible, mark affected behavior unverified and do not claim parity.
2. Trace one representative request through prompt and model profile, tools and integrations, middleware and permissions, backend and execution, skills and memory, delegation, state, checkpoints and approvals, retries and limits, events, side effects, and the public boundary. Inspect every caller of that boundary.
3. Run the cheapest deterministic baseline. Record evidence separately from the outcome: `source-inspected`, `probe-observed`, or `regression-tested`.
4. Load only the mapping needed for the detected source features:
   - [Core Migration](references/CORE-MIGRATION.md) for construction, models and profiles, prompts, tools and MCP, middleware, state, outputs, caching, or tracing.
   - [Context and Execution](references/CONTEXT-AND-EXECUTION.md) for coding-agent defaults, planning, files, permissions, backends, sandboxes, interpreters, media, skills, memory, retrieval, or context limits.
   - [Orchestration and Recovery](references/ORCHESTRATION-AND-RECOVERY.md) for synchronous, dynamic, or background subagents; grading loops; approvals; checkpoints; fault tolerance; durable execution; or streaming.
   - [Hosts and Protocols](references/HOSTS-AND-PROTOCOLS.md) only when migrating Deep Agents Code, a frontend, ACP, A2A, or a deployment boundary.

## Build the Pydantic AI target

- Start with one reusable `Agent`. Pydantic AI owns the model/provider boundary, normalized messages, agent loop, typed dependencies and outputs, tools and toolsets, run APIs, usage limits, and generic hooks.
- Put authenticated identity, service clients, and configuration in `deps_type` and `RunContext`; expose only model-chosen inputs through typed tools. Choose `instructions` versus `system_prompt` from the required history behavior, and consume `result.output` rather than leaking framework messages through the application boundary.
- Add Harness capabilities one at a time for observed optional behavior. A capability composes instructions, toolsets, hooks, and model settings onto the agent; it does not replace application infrastructure or Pydantic AI runtime semantics.
- Select `run_sync`, `run`, `run_stream`, `run_stream_events`, `iter`, deferred results, and message history from the caller's actual lifecycle. Preserve the old wire shape with a boundary adapter while callers migrate.
- Before coding, inspect the locked `pydantic_ai` and `pydantic_ai_harness` public exports, docs, and source signatures for the selected provider, `Agent`, tools/toolsets, capabilities, run and deferred APIs, and persistence integration. Import capabilities from their owning public submodules, reject deprecation warnings, and run an import-and-construction smoke test in the exact target environment. Record the interpreter and resolved `pydantic_ai` and `pydantic_ai_harness` module paths with the result. Add only the required extras; do not pass LangChain objects through or copy remembered examples.
- Keep queues, tenancy, artifact storage, remote sandbox lifecycle, durable domain state, deployment, and external side-effect recovery in application services.
- Classify orchestration as model-controlled delegation, a fixed application workflow, or a background worker before selecting `SubAgents` or `DynamicWorkflow`; keep fixed sequencing and worker queues in application code.
- If public Pydantic AI primitives cannot implement required generic runtime semantics correctly, propose the core primitive instead of recreating the runtime in Harness.

When there is no direct equivalent, do not stop at "unsupported." State the source contract and impact, then recommend either an existing composition, a narrow adapter or application service, a new Harness capability built from public primitives, or a Pydantic AI core change. Recommend one option and name its residual risk; ask before choosing when the options materially change behavior, architecture, public API, or scope.

## Migrate and verify

1. Port one vertical slice behind the existing application boundary: typed dependencies, one tool family, output, and one representative request.
2. Preserve source request, result, error, event, and continuation shapes with a small adapter while callers migrate.
3. Run the original suite together with focused characterization, parity, and evaluation checks before expanding to the next capability. Exercise the supported `Agent` boundary in the same interpreter and installed checkout the target will use; do not accept results from another environment. Read [Tested Composition](references/TESTED-COMPOSITION.md) only when a local coding-agent composition is useful.
4. Read [Validation and Cutover](references/VALIDATION.md) before publishing examples or claiming cutover; for a simple slice, use only the applicable checks.
5. Cut over only when each in-scope contract is regression-tested, intentionally changed with agreement, or explicitly not applicable. Report anything else as unverified.

Match validation to observed contracts. Focused characterization and parity tests are sufficient for a stateless slice; persistence, approvals, concurrency, streaming, and external side effects require the ledger and applicable restart and idempotency checks.
