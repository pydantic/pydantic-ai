---
name: migrating-langchain-to-pydantic-ai
description: Migrate Python agents and graphs from LangChain or LangGraph to native Pydantic AI while preserving tools, structured output, dependencies, streaming, memory, human approval, orchestration, durable execution, tests, and operational boundaries. Use when auditing, planning, implementing, or reviewing a port away from `langchain`, `langgraph`, `create_agent`, `StateGraph`, LCEL, or LangSmith, including large systems such as Open Deep Research. Do not use for `create_deep_agent` or Deep Agents Harness projects; route those to the dedicated `migrate-deep-agents-to-pydantic-ai` skill.
---

# Migrate LangChain to Pydantic AI

Treat migration as a behavior-preservation project, not an import rewrite. Preserve public contracts, tool semantics, authorization, state transitions, streaming events, recovery behavior, and eval baselines before removing LangChain.

## Declare non-equivalences first

Start every assessment by telling the user that there is no 1:1 framework translation. Create a **semantic-gap register** before proposing target code. For each row, record the source guarantee, observed target behavior, intended change, owner, and executable parity probe.

| Source construct | Do not equate it with | Decision to make explicitly |
|---|---|---|
| LangGraph state, reducers, checkpoints, and store | `deps`, message history, or one Pydantic model | Separate run dependencies, conversation, workflow state, and long-term memory; retain durability/fork/replay owners. |
| `interrupt()` or HITL middleware | deferred approval alone | Map positional vs tool-call-ID correlation, node re-execution, persistence, denial feedback, and idempotency. |
| middleware list | a list of hooks | Preserve before/after/wrap nesting, state updates, short-circuits, retries, and resume behavior. |
| `response_format=Schema` | `output_type=Schema` with identical wire behavior | Choose native/tool/prompted output deliberately and test retries plus co-emitted tools. |
| `recursion_limit`, node retry, or tool retry | `UsageLimits` or `ModelRetry` | Assign graph steps, model requests, tool calls, transport retries, and model-correctable retries separately. |
| LangGraph stream modes | Pydantic AI stream methods/events | Define an application event contract; test ordering, finalization, cancellation, reconnect, and backpressure. |
| `Send`, parallel nodes, or parallel tool calls | `asyncio.gather` or default tool concurrency | Preserve join order, reducers, partial failure, cancellation, limits, and side-effect ordering. |
| prompt construction and saved messages | Pydantic AI `instructions`, `system_prompt`, and `message_history` | Capture the model boundary; current `create_agent` prompts are not necessarily checkpointed messages, and Pydantic instructions replace historical instructions. |

Use [Semantic Gaps and Parity Spikes](references/SEMANTIC-GAPS.md) by feature: prompts/messages, tools, and output for agent ports; state/approval for graphs or HITL; middleware/limits for cross-cutting behavior; streaming/concurrency only when present. Read the full reference for broad audits. Spike uncertain behavior against the installed versions before calling a row `native`; do not rely on name similarity or documentation alone.

Do not stop at identifying a gap. For each non-equivalence, select and spike a concrete Pydantic AI capability, hook, output mode, durable integration, graph/application adapter, or service boundary. Read [Validated Workaround Recipes](references/WORKAROUND-RECIPES.md) for the affected feature. Classify the result as `validated-native`, `validated-adapter`, `integration-required`, or `blocker`; a row labeled only “redesign” is incomplete.

## Start with evidence

1. Read repository instructions and dependency files. Check `requires-python`, tool configuration, or active virtual environments across the whole scan scope; monorepo packages may require different syntax versions.
2. Run the inventory from the skill root with the highest Python syntax version required by any scanned workspace. The scanner uses the launching interpreter's grammar; an older interpreter can misclassify valid newer syntax as a parse error.

   ```bash
   /path/to/repository/.venv/bin/python scripts/inventory_langchain.py /path/to/repository
   # Or, replacing 3.12 with the highest syntax version required in the scan scope:
   uv run --python 3.12 python scripts/inventory_langchain.py /path/to/repository
   ```

   Use `--details` for every finding or `--format json` for machine-readable output. Confirm the reported scanner Python matches the target before treating syntax errors as source failures. After that check, treat parse errors as blockers; `--allow-parse-errors` is only for exploratory scans. The script covers Python, notebooks, and common dependency/config files, not JavaScript/TypeScript, generated code, or arbitrary plugin registries. Follow it with `rg -n 'langchain|langgraph|langsmith|deepagents'` and dependency/entrypoint inspection before declaring the repository clean.
3. Trace at least one representative request from entrypoint to final output. Include model calls, tool calls, state writes, interrupts, retries, persistence, and emitted events.
4. Record existing unit, integration, and eval commands. Run the cheapest useful baseline before editing.
5. Identify the installed LangChain, LangGraph, and Pydantic AI versions. Verify volatile APIs against the checked-out package or official docs before coding.

## Classify before choosing targets

- **Agent loop:** `create_agent`, tools, prompt, structured response. Prefer one `pydantic_ai.Agent`.
- **Agent plus middleware:** dynamic prompt/tools, guardrails, retries, logging, message trimming. Use dependencies, toolsets, hooks, capabilities, output validators, and model wrappers.
- **Explicit graph:** nodes, reducers, `Command`, fan-out, or deterministic routing. Keep orchestration in application code or `pydantic_graph`; do not hide deterministic control flow inside prompts.
- **Product runtime:** webhooks, queues, sandboxes, auth, schedulers, deployment, thread storage. Keep these as application infrastructure around Pydantic AI.

If inventory finds `create_deep_agent`, Deep Agents profiles/backends, repository skills, virtual files, or Harness-style sandbox/deployment behavior, stop this workflow and route the project to `migrate-deep-agents-to-pydantic-ai`. Do not let both migration skills produce competing plans for one source project.

Read only the relevant sections of [Concept Mapping](references/CONCEPT-MAPPING.md). Read [Project Patterns](references/PROJECT-PATTERNS.md) when the source uses LangGraph or is a substantial service. Read [Verification and Cutover](references/VERIFICATION-AND-CUTOVER.md) before implementing a production migration.

## Build a migration ledger

Create a concise ledger before changing code:

| Source behavior | Current owner | Target location | Pydantic workaround/capability | Throwaway parity spike | Outcome |
|---|---|---|---|---|---|
| tool access to DB/user | `ToolRuntime.context` | native plus application auth | typed `RunContext.deps` | unauthorized/authorized calls and schema diff | planned |
| structured result | `response_format` | native plus DTO adapter | explicit `NativeOutput` or `ToolOutput` | request transport, schema, retry, and co-emitted tools | planned |

Include every state field, middleware hook, tool, retriever, subagent, interrupt, checkpointer/store, stream consumer, observability dependency, and deployment contract. Record where the implementation lives (`native`, `bridge`, or `external`) separately from its proved outcome (`validated-native`, `validated-adapter`, `integration-required`, or `blocker`). Do not use `redesign` as an outcome without implementing and probing the proposed design.

## Migrate in vertical slices

1. Freeze externally visible input, output, error, and event contracts with characterization tests.
2. Define typed Pydantic AI dependencies and outputs. Keep secrets and service clients in dependencies, never model-visible tool arguments.
3. Port one end-to-end path: agent, smallest useful tool set, output, and caller adapter.
4. Port cross-cutting behavior deliberately. Replace middleware with the narrowest primitive: tool configuration, hook, capability, model wrapper, or application code. Run the selected workaround spike before integrating it.
5. Add orchestration only where the traced behavior requires it. Prefer plain async Python for fixed sequences and `asyncio.gather`; use `pydantic_graph` for inspectable typed state machines.
6. Add message storage, durable execution, approval, harness capabilities, and UI streaming as separate layers with separate tests.
7. Compare old and new implementations on the same fixtures/eval cases. Cut traffic over by slice, then remove bridges and LangChain dependencies.

## Apply these defaults

- Use `Agent(..., deps_type=..., output_type=...)`; use `instructions` for developer intent.
- Use `@agent.tool` only when a tool needs `RunContext`; otherwise use `@agent.tool_plain` or reusable `Tool`/toolsets.
- Use Pydantic models for terminal structured output. Do not parse final model text manually when an output type expresses the contract.
- Use `message_history` for conversation continuity, but keep durable storage and thread identity in the application layer unless a selected persistence capability owns them.
- Use `Hooks` for lightweight lifecycle interception and a custom `AbstractCapability` for reusable bundles of instructions, tools, hooks, and settings.
- Use deferred tool requests for approval. Keep the approve/deny UI and durable resume token outside the agent. Treat client-submitted deferred results as untrusted: authorize and correlate them server-side to the issued pending call, principal, and approved arguments or server-authorized overrides.
- Use delegation tools or explicit application orchestration for specialists. Preserve usage limits, dependency scope, and failure propagation.
- Use `tool_from_langchain` or `LangChainToolset` only as a named transitional bridge. These wrappers do not give Pydantic AI native argument validation.
- Use Logfire and `pydantic_evals` to replace behavior, not merely vendor names from LangSmith.

## Finish with proof

Run the original tests, new Pydantic AI unit tests with `TestModel` or `FunctionModel`, integration tests for real tool/provider boundaries, and representative evals. For dynamic instructions, prove at least two dependency values produce the intended different model-visible instructions without leaking authenticated identity. For approval, prove denial executes the protected tool zero times and an approved resume executes it exactly once with the original dependencies and correlation. Test `run()` and the chosen streaming API separately when tools and terminal output can be co-emitted. Report every semantic-gap row with its concrete workaround, outcome classification, residual limitation, and command evidence. Leave the migration slice blocked when no workaround has passed.
