---
name: migrating-langchain-to-pydantic-ai
description: Migrate Python LangChain, LangGraph, or Deep Agents applications to Pydantic AI and, when the source uses harness features, Pydantic AI Harness. Use for LangChain agents, chains, LCEL, direct LangGraph graphs, persistence, interrupts, streaming, and `create_deep_agent` projects with planning, filesystem or sandbox backends, skills, memory, subagents, permissions, approvals, or Deep Agents Code hosts.
---

# Migrate LangChain, LangGraph, and Deep Agents to Pydantic AI

Preserve behavior, not framework shape. Migrate the smallest behaviorally complete slice and leave application infrastructure outside that slice unchanged. LangChain, LangGraph, and Deep Agents are one source ecosystem; Pydantic AI, `pydantic_graph`, and Pydantic AI Harness are the matching target ecosystem, and this skill maps across all of it.

## Work from the running application

1. Read repository instructions, dependency files, tests, and the actual runtime entrypoints. Identify the installed LangChain, LangGraph, Deep Agents, Pydantic AI, and Harness versions.
2. Trace one representative request through prompts, retrieval, model and tool calls, state, persistence, interrupts, emitted events, tracing/metrics callbacks, and the public result. Inspect every caller and sibling endpoint that consumes the migrated component; a narrow implementation slice can still have several public contracts. Include keyword parameter names and the sync, async, callback, and streaming forms callers actually use. Record only contracts those paths actually use.
3. Run the cheapest useful baseline. When the migration surface is broad or unclear, search dependency files and source for `langchain`, `langgraph`, `langsmith`, and `deepagents`, then confirm findings against imports, factories, and call sites. Resolve where `create_deep_agent` comes from: the upstream package and a vendored copy that reproduces its contracts both count as Deep Agents.
4. Classify the slice before choosing a target:
   - **Chain or LCEL pipeline:** keep deterministic retrieval and transformation in plain Python; use a Pydantic AI agent only where a model/tool loop adds value.
   - **LangChain agent:** normally use one reusable `pydantic_ai.Agent` with typed dependencies, tools, and outputs.
   - **Direct LangGraph workflow:** use plain async Python for simple fixed control flow, or `pydantic_graph` when explicit typed nodes and branching remain useful. Treat persistence as a separate design decision.
   - **Deep Agents harness:** `create_deep_agent` bundles file tools, shell execution, subagent delegation, summarization, skills, memory, permissions, prompt caching, and approval middleware on top of the LangChain agent loop, and a harness profile can add or hide more. Migrate the loop with the LangChain mappings, then map each bundled feature the slice actually uses to a Harness capability, a core primitive, or an application service with [Deep Agents Mapping](references/DEEP-AGENTS-MAPPING.md). Inventory implicit defaults the application code never mentions.
   - **Product runtime:** retain queues, configured database backends, sandboxes, auth, schedulers, webhooks, tracing, and transport adapters unless the user placed them in scope. Extend an existing application seam before creating a parallel persistence or provider subsystem.
5. Add or preserve deterministic characterization tests, then migrate one vertical slice behind the existing public boundary.
6. Run the original tests and focused parity tests. Classify each observed contract by its evidence; never describe the migration as one-to-one merely because the happy path or trace shape looks similar.

Read [Concept Mapping](references/CONCEPT-MAPPING.md) for the detected source features. Read [Deep Agents Mapping](references/DEEP-AGENTS-MAPPING.md) whenever `deepagents` is in the slice. Read [Semantic Gaps](references/SEMANTIC-GAPS.md) only for state, middleware, retries, approval, concurrency, streaming, or other behavior where similar-looking APIs may differ. Use [Workaround Recipes](references/WORKAROUND-RECIPES.md) after a concrete gap is identified, not as a mandatory checklist. Read [Logfire Verification](references/LOGFIRE-VERIFICATION.md) when adding observability, comparing source and target runs, or debugging a semantic difference. Read [Verification and Cutover](references/VERIFICATION-AND-CUTOVER.md) before a production cutover.

## Explain semantic differences

When an observed source contract has no direct equivalent, explain it to the user before making a consequential design choice. State the source behavior, how the proposed Pydantic AI design differs, the user-visible or operational impact, and the available choices. Recommend one option and name its residual risk. Keep this proportional: do not turn ordinary import or naming changes into semantic warnings.

Do not stop at "unsupported." When nothing maps directly, recommend an existing core or Harness composition, a narrow adapter or application service, a new capability built from public primitives, or a core change, in that order of preference. Ask before choosing only when the options materially change behavior, architecture, public API, or scope.

## Match rigor to risk

- For a stateless chain or ordinary agent port, focused characterization tests and a short residual-risk note are enough. Do not require a semantic-gap register or durability exercise for behavior the source does not have.
- For middleware, structured output transport, retrieval, tool retries, or streaming, probe the affected contract against the installed versions.
- For checkpointed graphs, interrupts, approvals, durable execution, concurrent fan-out, or external side effects, create a migration ledger. Separate dependencies, messages, workflow state, checkpoint state, and long-term memory. Fit those owners into the repository's existing backend-selection and service interfaces where possible. Test restart, replay, correlation, authorization, and idempotency only to the extent the source promises them.
- A `deepagents` dependency alone changes nothing. When the active slice calls `create_deep_agent` or depends on its planning, filesystem, sandbox, skills, memory, subagent, permission, or deployment contracts, those are in-scope features with Harness or application owners, not a reason to stop. Files, sandboxes, permissions, subagents, and background work carry the same ledger and restart obligations as checkpointed graphs.

## Pydantic AI defaults

- Put authenticated identity, service clients, and configuration in typed dependencies, never model-chosen tool arguments.
- Strengthen observed unstable seams, not the whole application: parameterize the agent's dependency and output types, and validate terminal choices, persisted workflow records, and framework adapters. Preserve stable public wire shapes and do not invent types for paths outside the migrated slice.
- Preserve public request, response, error, and event shapes with a small adapter while callers migrate.
- Keep retrieval, storage, provider, and transport integrations in place when they are outside the requested slice. Transitional LangChain integrations are acceptable when named and bounded.
- Use Pydantic models for terminal structured output when that preserves the contract; retain an existing parser when changing the wire contract would expand the migration.
- Do not force an `Agent` onto deterministic LCEL or `pydantic_graph` onto every `StateGraph`.
- Inspect the installed Pydantic AI API before choosing model classes, provider transports, hooks, streaming methods, or durable integrations.
- Add `pydantic-ai-harness` only for observed harness behavior: planning, workspace files, shell or sandboxed execution, Agent Skills, memory notebooks, repository context, model-directed subagents, model-agnostic compaction, tool-output limits, guardrails, spend limits, or step persistence. Add capabilities one at a time, import each from its owning public submodule, treat `pydantic_ai_harness.experimental.*` as version-sensitive, and run an import-and-construction smoke test in the target environment. Harness composes onto the core agent loop; it is not a second runtime and does not replace application infrastructure.
- When adding Pydantic AI, prefer a currently supported stable release. Use the newest compatible release unless that would expand the migration through an unrelated major/runtime upgrade; explain and pin any exception. Resolve the whole project from a clean environment and run an import probe because an existing environment can hide incompatible transitive versions. Prefer `pydantic-ai-slim` with only the required provider and integration extras when the dependency surface is bounded, and use the full distribution when its broader integrations are actually needed. Do not pin an older release merely to match a remembered example.
- If the source already uses LangSmith, Langfuse, or another observability system, do not replace it silently. Explain that Logfire is the first-party Pydantic AI integration and normally provides the most direct agent, model, tool, retry, error, usage, and timing experience. Contrast that with the continuity of retaining the current system, including its dashboards, alerts, evaluations, retention, and export pipeline; recommend a choice and obtain agreement before switching. Offer Logfire at application startup when no tracing system exists or the user chooses it, make content capture an explicit privacy decision, and keep executable contract tests as the authority for parity.

## Completion

The slice is complete when every observed contract is either preserved by an executable check, intentionally changed by an accepted decision, or explicitly not applicable. An untested contract is `unverified`, not equivalent; an unresolved requested contract is unfinished work, not completion evidence. Constrain the slice or ask the user to accept the deferral. Remove LangChain, LangGraph, or Deep Agents dependencies only after no retained path needs them.
