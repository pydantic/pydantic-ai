# Verification and cutover

Prove parity through the application's supported boundary. Do not require high-risk exercises for behavior the source does not use.

## Build a contract ledger

For each traced source behavior, maintain one row:

| Source lifecycle | Observable contract | Owner | Target difference | Evidence |
|---|---|---|---|---|
| What initiates and completes it | Inputs, outputs, events, errors, ordering, state, or effects callers observe | Application, Core, Graph, Harness, Evals, or Gap | What will not remain identical | Baseline and focused target test |

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

- Characterize the public call and result before rewriting it.
- Use `TestModel` or `FunctionModel` to prove prompts, dependency-backed tool calls, validated output, retry/error mapping, messages, and usage without network access.
- Add a provider recording or live probe only when transport, native tools, structured-output mode, provider events, or multimodal behavior is part of the contract.

### Sessions, state, memory, and knowledge

- Choose and test how existing Agno records cross cutover: one-time conversion, a read-through adapter, or starting fresh as an `intentional-change`.
- Serialize normalized target messages through the application's real store and load them in a fresh process for the next turn.
- Test user/session ownership, retention, incomplete tool calls, store failures, and concurrent updates.
- Test mutable session state, summaries, user memory/learning, knowledge retrieval, workflow state, checkpoints, and model-owned plans independently. Similar facts appearing in a prompt do not prove equivalent lifecycle or ownership.

### Teams and workflows

- Characterize member selection, input/history sharing, aggregation, branch selection, step inputs/outputs, joins, concurrency limits, ordering, partial failures, cancellation, retries, and terminal status that callers consume.
- Keep deterministic collaboration and workflow control in Python or `pydantic_graph`; use deterministic agents only inside steps that actually call them.
- For pause/resume or durable execution, kill and restart at each promised boundary. Assert persisted state and step identity, validate resume input, and prove external effects are idempotent.

### Human-in-the-loop and protected effects

- Distinguish confirmation, free-form user input, and external execution in the ledger.
- Persist the pending request's category, validated arguments, call ID, and required metadata when it crosses a run or process boundary.
- Exercise deny, approve/input/result, timeout, duplicate response, and restart.
- Assert no protected effect before resolution, zero after denial, and exactly one after approval or authorized external execution.
- Exercise authenticated identity, policy lookup, correlation, and audit trail. A local yes/no callback is not authorization proof.

### Hooks, streaming, and skills

- Record a source golden trace containing only event types and stable fields the caller consumes.
- Test hook/guardrail order, mutation, short-circuit behavior, retries, text/object reconstruction, terminal detection, trailing events, cancellation, and parallel calls.
- For skills, test discovery, model selection, loaded instructions, every used bundled resource or script, trust policy, and behavior after restart.

### AgentOS and execution environments

- Run fixtures against retained or replacement API routes, auth, session selection, event serialization, errors, and transport cancellation.
- For file/shell/browser/code tools, use a disposable workspace and test ordinary effects plus traversal, symlinks, denied files, output limits, timeouts, cleanup, environment handling, and working directory.
- If untrusted code is in scope, run containment tests inside the selected container, VM, or cloud sandbox. Path restrictions and command allowlists are not containment evidence.

## Dependency and cutover checks

1. Resolve the Python project from a clean environment against supported Pydantic AI and Harness versions. Prefer `pydantic-ai-slim` with only required provider/integration extras when the surface is bounded.
2. Run source characterization tests and focused target tests through the same public boundary. Label each as deterministic fake, recording, live service, fresh-process restart, or real sandbox evidence.
3. Search imports, factories, configuration, session/state readers, workflow checkpoints, event/result adapters, AgentOS routes, and deployment scripts for retained `agno` dependencies.
4. Keep compatibility adapters and rollback until all callers consume the target shapes and every required ledger row is resolved.
5. Remove Agno, AgentOS, database adapters, and event assumptions only when no retained path needs them.

## Completion criterion

Cut over only when every ledger row is `verified-equivalent`, `verified-adapter`, an accepted `intentional-change`, `external-owner`, or `not-applicable`; no required row is `unverified` or `blocked`; the supported boundary passes; dependencies resolve cleanly; and rollback remains possible.
