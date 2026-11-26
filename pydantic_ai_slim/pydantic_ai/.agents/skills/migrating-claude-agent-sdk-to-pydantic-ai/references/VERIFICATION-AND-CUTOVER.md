# Verification and cutover

Prove parity through the application's supported boundary. Do not require high-risk exercises for behavior the source does not use.

## Build a contract ledger

For each traced source behavior, maintain one row:

| Source lifecycle | Observable contract | Owner | Target difference | Evidence |
|---|---|---|---|---|
| What initiates and completes it | Inputs, outputs, events, errors, ordering, state or effects callers observe | Application, Core, Harness, or Gap | What will not remain identical | Baseline and focused target test |

Evidence states are `preserved`, `accepted change`, `application-owned`, `tested gap`, or `unverified`. Do not call `unverified` equivalent.

## Match testing to the slice

### Ordinary core agent

- Characterize the public request and result.
- Use `TestModel` or `FunctionModel` to prove prompt, dependency-backed tool calls, validated output, retry/error mapping, messages, and usage without network access.
- Add a provider recording or live probe only when transport, native tools, structured-output mode, or provider deltas are part of the contract.

### Stateful or branching agent

- Serialize the first result's normalized messages through the application's real store and load them in a fresh process for the next run.
- Test branch point, branch independence, `conversation_id`/`run_id` semantics, incomplete tool-call history, retention, tenant authorization, and store failures that callers can observe.
- If the source resumes in-flight work, kill and restart at each promised boundary. Select a durable runtime deliberately and assert that an external effect is not duplicated.

### Approvals and protected effects

- Assert the deferred call ID and validated arguments.
- Exercise deny, approve, and argument override where supported.
- For inline resolution, assert the handler runs and the agent completes in one call. When the decision arrives later, assert the first run ends with `DeferredToolRequests`, persist that complete request or an equivalent pending-action record with category, validated arguments, and metadata, then assert a new run over the persisted messages with `DeferredToolResults` completes with the final output.
- Assert no protected side effect before resolution, zero after denial, and exactly one after approval.
- Exercise the application's authenticated identity, policy lookup, and audit trail. A local yes/no callback is not authorization proof.

### Streaming, hooks, and subagents

- Record a source golden trace containing only event types and stable fields the caller consumes.
- Test text reconstruction, tool start/result order, hook order and mutations, terminal detection, trailing events, cancellation, retry, and parallel calls.
- For subagents, test task-only input, separate history, result handback, dependencies, budget/cancellation/error propagation, event forwarding, and recursion limits.

### Coding and shell agents

- Use a disposable workspace. Exercise normal reads/edits/search/commands plus `..`, absolute paths, escaping symlinks, denied/protected files, output limits, timeouts, background cleanup, environment stripping, and working-directory behavior.
- Demonstrate that an allowlisted interpreter can spawn another command. If untrusted code is in scope, run the containment test inside the selected container, VM, or cloud sandbox.
- If rollback is promised, verify filesystem or VCS state independently of conversation history.

## Dependency and cutover checks

1. Resolve the project from a clean environment against the chosen supported Pydantic AI and Harness versions. Prefer `pydantic-ai-slim` with only the needed extras when the dependency surface is bounded.
2. Run the original characterization tests and focused target tests. Test sync, async, callback, and streaming forms only where callers use them.
3. Search imports, factories, configuration, transcript readers, event/result adapters, and deployment scripts for retained Claude Agent SDK dependencies.
4. Remove `claude-agent-sdk`, Claude Code subprocess setup, and Claude transcript assumptions only when no retained path needs them.
5. Report which checks used deterministic fakes, recordings, live providers, restart tests, or sandbox tests, and state any remaining limitation.

## Completion criterion

Cut over only when each ledger row has a non-`unverified` evidence state, the original supported boundary passes, requested gaps are resolved or explicitly accepted, dependencies resolve cleanly, and rollback remains possible.
