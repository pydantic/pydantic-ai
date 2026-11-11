# Verification and cutover

Prove parity through the application's supported boundary. Do not require high-risk exercises for behavior the source does not use.

## Build a contract ledger

For each traced source behavior, maintain one row:

| Source lifecycle | Observable contract | Owner | Target difference | Evidence |
|---|---|---|---|---|
| What initiates and completes it | Inputs, outputs, events, errors, ordering, state or effects callers observe | Application, Core, Graph, Harness, Evals, or Gap | What will not remain identical | Baseline and focused target test |

Use the same evidence states as the LangChain migration skill:

| Status | Meaning |
|---|---|
| `verified-equivalent` | Source and target preserve the same observable contract in executable checks. |
| `verified-adapter` | Internal semantics differ, but an adapter preserves the public contract in executable checks. |
| `intentional-change` | The difference and impact were explained and explicitly accepted. |
| `external-owner` | A named application or infrastructure component preserves the contract, with evidence at that boundary. |
| `not-applicable` | The observed source path does not provide or consume this behavior. |
| `unverified` | Evidence is incomplete or only a candidate design exists; do not call it equivalent. |
| `blocked` | A required contract has no acceptable proved construction; do not cut over the slice. |

## Match testing to the slice

### Ordinary agent or tool

- Characterize the public request and result before rewriting TypeScript as Python.
- Use `TestModel` or `FunctionModel` to prove prompts, dependency-backed tool calls, validated output, retry/error mapping, messages, and usage without network access.
- Add a provider recording or live probe only when transport, native tools, structured-output mode, or provider deltas are part of the contract.

### Memory and state

- Choose and test how existing Mastra thread and working-memory records cross the cutover: one-time conversion, a read-through adapter, or starting fresh as an `intentional-change`. Serialize normalized target messages through the application's real store and load them in a fresh process for the next turn.
- Test thread ownership, retention, incomplete tool calls, store errors, and tenant authorization.
- Test semantic recall, working memory, observations, workflow state, and model-owned plans independently. Similar facts appearing in a prompt do not prove equivalent lifecycle or ownership.

### Workflows and durability

- Characterize branch selection, step inputs/outputs, joins, concurrency limits, ordering, partial failures, cancellation, retries, and terminal status that callers consume.
- For suspend/resume or durable execution, kill and restart at each promised boundary. Assert persisted state and step identity, validate resume input, and prove that external effects are idempotent.
- Keep deterministic workflow control in Python or `pydantic_graph`; use deterministic agents or fake services only within the steps that actually call them.

### Approvals and protected effects

- Assert the deferred call ID and validated arguments.
- Exercise deny and approve, including any source decline reason exposed to the model or caller.
- For inline resolution, assert the handler runs and the agent completes in one call. When the decision arrives later, assert the first run ends with `DeferredToolRequests`, persist that complete request or an equivalent pending-action record with category, validated arguments, and metadata, then assert a new run over the persisted messages with `DeferredToolResults` completes with the final output.
- Assert no protected effect before resolution, zero after denial, and exactly one after approval.
- Exercise authenticated identity, policy lookup, durable correlation, and audit trail. A local yes/no callback is not authorization proof.

### Streaming, processors, and subagents

- Record a source golden trace containing only event types and stable fields the caller consumes.
- Test text/object reconstruction, tool start/result order, processor order and mutations, tripwires, terminal detection, trailing events, cancellation, retry, and parallel calls.
- For subagents, test task-only input, separate history, result handback, dependencies, budget/cancellation/error propagation, event forwarding, approval propagation, and recursion limits.

### Coding and sandboxed agents

- Use a disposable workspace. Exercise ordinary reads, edits, searches, and commands plus `..`, absolute paths, escaping symlinks, denied or protected files, output limits, timeouts, cleanup, environment handling, and working-directory behavior.
- Demonstrate that an allowlisted interpreter can start another command. If untrusted code is in scope, run containment tests inside the selected container, VM, or cloud sandbox.

## Dependency and cutover checks

1. Resolve the Python project from a clean environment against supported Pydantic AI and Harness versions. Prefer `pydantic-ai-slim` with only required extras when the dependency surface is bounded.
2. Run the original Mastra characterization tests and focused target tests. Capture every existing HTTP/event, persisted-record, snapshot, or provider boundary with language-neutral fixtures and run them against both implementations. If the source is only in-process, migrate its caller in the slice or explicitly test the newly agreed application service boundary.
3. Search imports, factories, configuration, memory and snapshot readers, event/result adapters, server routes, and deployment scripts for retained `@mastra/*` dependencies.
4. Remove Mastra packages, Node build/runtime setup, and storage/event assumptions only when no retained path needs them.
5. Report which checks used deterministic fakes, recordings, live providers, fresh-process restart tests, or real sandbox tests, and state any remaining limitation.

## Completion criterion

Cut over only when every ledger row is verified, an accepted `intentional-change`, `external-owner`, or `not-applicable`; no required row is `unverified` or `blocked`; the supported boundary passes; dependencies resolve cleanly; and rollback remains possible.
