# Verification and cutover

Prove parity through the application's supported boundary. Do not require high-risk exercises for behavior the traced source path does not use.

## Build a contract ledger

For each traced source behavior, maintain one row:

| Source lifecycle | Observable contract | Owner | Target difference | Evidence |
|---|---|---|---|---|
| What initiates and completes it | Inputs, outputs, events, errors, ordering, state or effects callers observe | Application, Core, Graph, Harness, Evals, or Gap | What will not remain identical | Baseline and focused target test |

Use these evidence states:

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
- Use `TestModel` or `FunctionModel` to prove prompts, dependency-backed tool calls, validated output, step limits, errors, messages, and usage without network access.
- Exercise tool failures that the source loop exposes to the model separately from failures that terminate the request.
- Add a provider recording or live probe only when provider transport, native tools, structured-output mode, retry behavior, or deltas are part of the contract.

### AI SDK UI and cross-language boundaries

- Keep the installed AI SDK version explicit. Capture the actual request body and a source golden stream containing only the headers, framing, event types, stable fields, and ordering the client consumes.
- Run the same language-neutral fixture against the Python endpoint. Test text reconstruction, tool input/output states, approvals, custom data, errors, abort, finish, and trailing events used by the application.
- Validate client-provided messages and files. Prove authenticated identity, server instructions, provider metadata, tool executors, and previous trusted history come from server-owned state.
- If reconnect is supported, disconnect mid-stream and continue in a fresh process or server instance. Assert cursor behavior, the no-active-stream response, terminal detection, and no duplicate message, tool call, or effect.
- If no existing transport boundary exists, migrate the nearest caller too. Treat a newly introduced Python service as an architectural change, not automatic parity.

### Structured output and streaming

- Test valid complete output, invalid complete output, and any retry. Do not validate a deep-partial stream as though it were the final object.
- Record only the event fields consumed by the caller. Assert reconstruction, ordering, terminal detection, cancellation propagation, provider errors, tool errors, retries, and whether a final structured-output step occurs.

### Approvals and protected effects

- Assert the pending call ID, validated arguments, policy category, and any signed or server-owned correlation data.
- Exercise deny and approve, including invalid, stale, and replayed decisions and any decline reason exposed to the model or caller.
- Assert no protected effect before resolution, zero after denial, and exactly one after approval, including across retry and process restart.
- Define the durable ordering explicitly: persist the authorized decision and an idempotency key before starting the effect, then persist the effect result before resuming the agent. Test crashes between each write and effect boundary.
- Test the actual lifecycle in scope: AI SDK's next request, Eve's parked durable session, inline Pydantic AI handling, or a later Pydantic AI run with persisted messages and `DeferredToolResults`.
- Map source approval, tool-call, session, and application request IDs explicitly. If Eve accepts unrelated turns while an approval is pending, store those turns on a separate branch because unresolved Pydantic AI tool calls cannot have a new prompt appended after them.
- Exercise authenticated identity, authorization lookup, audit, and tenant isolation. Approval is not authorization.

### Eve state, sessions, and recovery

- Test UI messages, normalized agent messages, `defineState` data, memory, plans, session records, and event cursors independently.
- Kill and restart at every promised model, tool, wait, background-result, and external-effect boundary. Assert step identity, replay unit, event IDs/cursors, terminal status, and exactly-once effects.
- If batching model calls changes Eve's replay unit, characterize that configuration explicitly.
- Exercise concurrent resumes and stale inputs. Validate resume payloads and prove only the selected session, user, and tenant can continue the run.
- Label Harness `StepPersistence` as equivalent only when these boundary tests pass; persisted snapshots alone are not proof of Eve protocol compatibility.

### Skills, subagents, connections, and sandboxes

- For a used Eve skill, test description-based selection, loaded `SKILL.md`, and every required reference, asset, or script. A successful instruction load does not prove supporting files are available.
- For subagents, test input projection, separate history and state, shared or isolated dependencies/tools/sandbox, result handback, background delivery, event forwarding, usage limits, cancellation, errors, recursion policy, and restart.
- For MCP/OpenAPI connections, test runtime discovery, qualified names, schema conversion, auth failure and resume, token ownership, revocation, remote errors, and lifecycle cleanup.
- For untrusted execution, use a disposable workspace in the actual container, VM, or cloud sandbox. Exercise `..`, absolute paths, escaping symlinks, secrets, network policy, resource limits, timeouts, termination, and cleanup. A mocked filesystem or command allow-list is not sandbox evidence.

### Channels, schedules, observability, and evals

- Exercise the real channel or schedule entrypoint with identity, authorization, deduplication, ordering, retry, timezone, cancellation, and delivery failure.
- Compare required spans, run/session IDs, usage, metadata, content/privacy policy, and dashboards before switching telemetry.
- Re-run observed evaluation cases and essential metrics through the supported public boundary. Keep sampling, scheduling, trace correlation, and result storage application-owned unless separately replaced.

## Dependency and cutover checks

1. Resolve the Python project from a clean environment against supported Pydantic AI and Harness versions. Prefer `pydantic-ai-slim` with only required extras when the dependency surface is bounded.
2. Run the original source characterization tests and focused target tests. Capture retained HTTP, stream, storage, provider, and durable-session boundaries with language-neutral fixtures and run them against both implementations.
3. Search imports, agents, tools, UI routes, transport adapters, stores, sessions, schedules, channels, connections, sandbox configuration, build scripts, and deployment for retained `ai`, `@ai-sdk/*`, or `eve` dependencies.
4. Remove TypeScript packages, Node runtime/build steps, Vercel services, and storage or event assumptions only when no retained path needs them.
5. Report which checks used deterministic fakes, recordings, live providers, fresh-process restart tests, or real sandboxes, and state every remaining limitation.

## Completion criterion

Cut over only when every ledger row is verified, an accepted `intentional-change`, `external-owner`, or `not-applicable`; no required row is `unverified` or `blocked`; the supported boundary passes; dependencies resolve cleanly; and rollback remains possible.
