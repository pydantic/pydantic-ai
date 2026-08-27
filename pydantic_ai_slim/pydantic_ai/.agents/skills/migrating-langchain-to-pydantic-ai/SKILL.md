---
name: migrating-langchain-to-pydantic-ai
description: Migrate Python LangChain or LangGraph applications to Pydantic AI. Use for LangChain agents, chains, LCEL, or direct LangGraph graphs, persistence, interrupts, and streaming. Do not use for migrations centered on `create_deep_agent` or Deep Agents harness features.
---

# Migrate LangChain and LangGraph to Pydantic AI

Preserve behavior, not framework shape. Migrate the smallest behaviorally complete slice and leave application infrastructure outside that slice unchanged.

## Work from the running application

1. Read repository instructions, dependency files, tests, and the actual runtime entrypoints. Identify the installed LangChain, LangGraph, and Pydantic AI versions.
2. Trace one representative request through prompts, retrieval, model and tool calls, state, persistence, interrupts, emitted events, tracing/metrics callbacks, and the public result. Inspect every caller and sibling endpoint that consumes the migrated component; a narrow implementation slice can still have several public contracts. Include keyword parameter names and the sync, async, callback, and streaming forms callers actually use. Record only contracts those paths actually use.
3. Run the cheapest useful baseline. When the migration surface is broad or unclear, search dependency files and source for `langchain`, `langgraph`, `langsmith`, and `deepagents`, then confirm findings against imports, factories, and call sites.
4. Classify the slice before choosing a target:
   - **Chain or LCEL pipeline:** keep deterministic retrieval and transformation in plain Python; use a Pydantic AI agent only where a model/tool loop adds value.
   - **LangChain agent:** normally use one reusable `pydantic_ai.Agent` with typed dependencies, tools, and outputs.
   - **Direct LangGraph workflow:** use plain async Python for simple fixed control flow, or `pydantic_graph` when explicit typed nodes and branching remain useful. Treat persistence as a separate design decision.
   - **Product runtime:** retain queues, configured database backends, sandboxes, auth, schedulers, webhooks, tracing, and transport adapters unless the user placed them in scope. Extend an existing application seam before creating a parallel persistence or provider subsystem.
5. Add or preserve deterministic characterization tests, then migrate one vertical slice behind the existing public boundary.
6. Run the original tests and focused parity tests. Classify each observed contract by its evidence; never describe the migration as one-to-one merely because the happy path or trace shape looks similar.

Read [Concept Mapping](references/CONCEPT-MAPPING.md) for the detected source features. Read [Semantic Gaps](references/SEMANTIC-GAPS.md) only for state, middleware, retries, approval, concurrency, streaming, or other behavior where similar-looking APIs may differ. Use [Workaround Recipes](references/WORKAROUND-RECIPES.md) after a concrete gap is identified, not as a mandatory checklist. Read [Logfire Verification](references/LOGFIRE-VERIFICATION.md) when adding observability, comparing source and target runs, or debugging a semantic difference. Read [Verification and Cutover](references/VERIFICATION-AND-CUTOVER.md) before a production cutover.

## Explain semantic differences

When an observed source contract has no direct equivalent, explain it to the user before making a consequential design choice. State the source behavior, how the proposed Pydantic AI design differs, the user-visible or operational impact, and the available choices. Recommend one option and name its residual risk. Keep this proportional: do not turn ordinary import or naming changes into semantic warnings.

## Match rigor to risk

- For a stateless chain or ordinary agent port, focused characterization tests and a short residual-risk note are enough. Do not require a semantic-gap register or durability exercise for behavior the source does not have.
- For middleware, structured output transport, retrieval, tool retries, or streaming, probe the affected contract against the installed versions.
- For checkpointed graphs, interrupts, approvals, durable execution, concurrent fan-out, or external side effects, create a migration ledger. Separate dependencies, messages, workflow state, checkpoint state, and long-term memory. Fit those owners into the repository's existing backend-selection and service interfaces where possible. Test restart, replay, correlation, authorization, and idempotency only to the extent the source promises them.
- A `deepagents` dependency alone is not a reason to stop. If the active slice calls `create_deep_agent` or relies on its planning, skills, filesystem, subagent, sandbox, memory, or deployment contracts, report that it is a harness migration and ask whether those contracts are in scope. Do not assume another migration skill is installed.

## Pydantic AI defaults

- Put authenticated identity, service clients, and configuration in typed dependencies, never model-chosen tool arguments.
- Strengthen observed unstable seams, not the whole application: parameterize the agent's dependency and output types, and validate terminal choices, persisted workflow records, and framework adapters. Preserve stable public wire shapes and do not invent types for paths outside the migrated slice.
- Preserve public request, response, error, and event shapes with a small adapter while callers migrate.
- Keep retrieval, storage, provider, and transport integrations in place when they are outside the requested slice. Transitional LangChain integrations are acceptable when named and bounded.
- Use Pydantic models for terminal structured output when that preserves the contract; retain an existing parser when changing the wire contract would expand the migration.
- Do not force an `Agent` onto deterministic LCEL or `pydantic_graph` onto every `StateGraph`.
- Inspect the installed Pydantic AI API before choosing model classes, provider transports, hooks, streaming methods, or durable integrations.
- When adding Pydantic AI, prefer a currently supported stable release. Use the newest compatible release unless that would expand the migration through an unrelated major/runtime upgrade; explain and pin any exception. Resolve the whole project from a clean environment and run an import probe because an existing environment can hide incompatible transitive versions. Prefer `pydantic-ai-slim` with only the required provider and integration extras when the dependency surface is bounded, and use the full distribution when its broader integrations are actually needed. Do not pin an older release merely to match a remembered example.
- If the source already uses LangSmith, Langfuse, or another observability system, do not replace it silently. Explain that Logfire is the first-party Pydantic AI integration and normally provides the most direct agent, model, tool, retry, error, usage, and timing experience. Contrast that with the continuity of retaining the current system, including its dashboards, alerts, evaluations, retention, and export pipeline; recommend a choice and obtain agreement before switching. Offer Logfire at application startup when no tracing system exists or the user chooses it, make content capture an explicit privacy decision, and keep executable contract tests as the authority for parity.

## Completion

The slice is complete when every observed contract is either preserved by an executable check, intentionally changed by an accepted decision, or explicitly not applicable. An untested contract is `unverified`, not equivalent; an unresolved requested contract is unfinished work, not completion evidence. Constrain the slice or ask the user to accept the deferral. Remove LangChain or LangGraph dependencies only after no retained path needs them.
