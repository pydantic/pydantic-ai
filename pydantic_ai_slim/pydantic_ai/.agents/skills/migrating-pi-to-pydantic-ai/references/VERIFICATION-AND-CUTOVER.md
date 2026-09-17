# Verification and cutover

Prove parity through the application's supported boundary. Do not require terminal, provider, persistence, or sandbox exercises for behavior the source slice does not use.

## Build a contract ledger

Keep one row for each traced behavior and each responsibility in an active extension:

| Source lifecycle | Observable contract | Owner | Target difference | Evidence |
|---|---|---|---|---|
| What initiates and completes it | Inputs, outputs, events, errors, ordering, state, or effects callers observe | Application, Core, Harness, Graph, or Gap | What will not remain identical | Baseline and focused target test |

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

## Forward-test the extension split

For each active Pi extension or package, record:

| Source responsibility | Capability/Core target | Application/interface target | Gap | Proof |
|---|---|---|---|---|
| Tool, hook, command, UI, provider, resource, or state behavior | Exact reusable agent behavior | Host behavior retained/rebuilt | Unsupported difference | Executable observations on both sides |

Reject a one-row "extension becomes capability" claim when the source also owns commands, UI, providers, persistence, discovery, or host lifecycle. Test capability and application halves independently and then through the supported end-to-end boundary.

## Match testing to the slice

### Ordinary embedded agent or coding tool

- Characterize `prompt()` inputs, final messages, errors, usage, and effects before rewriting TypeScript as Python.
- Use `TestModel` or `FunctionModel` to prove prompts, typed dependency-backed tools, output, retries, and messages offline.
- Compare Pi coding tools with `Coder` or the selected smaller composition: schemas, exact edit matching, cwd, ignore/protected paths, truncation, command process lifetime, environment, and cancellation.
- Add provider recordings/live probes only when provider payloads, native tools, reasoning, or deltas are part of the contract.

### Extensions, hooks, and capabilities

- Record a source golden trace of only events and stable fields consumers use.
- Test handler/capability order, input/result mutation, validation timing, short-circuit behavior, exceptions, stream visibility, cancellation, and state lifetime.
- For custom capabilities, test composition with siblings, typed event attribution/order, per-run binding, concurrent runs, and cleanup.
- For mixed extensions, test commands/UI/provider/application behavior separately from the agent capability, then run one integrated boundary test.

### Sessions, compaction, and queued input

- Decide how existing Pi JSONL crosses cutover: conversion, compatibility reader, retained Pi session service, or fresh start as an `intentional-change`.
- Test messages separately from tree entries, labels, branches, compactions, extension entries, queued messages, and model/thinking changes.
- Continue in a fresh process; exercise incomplete tool calls, abandoned branches, overflow retry, repeated compaction, steering/follow-up order, concurrent input, cancellation, and storage failure.
- Do not claim `StepPersistence` or serialized model messages preserve Pi session semantics without those checks.

### Approvals, trust, and isolation

- Exercise allow, block, approve, deny, timeout, duplicate decision, and restart. Persist the complete pending action when the decision crosses a process boundary.
- Assert no protected effect before approval, zero after denial, and exactly one after approval.
- Test authenticated identity, policy lookup, audit, and correlation in application code.
- Test project-resource loading before/after trust if retained. State explicitly that trust, path restrictions, and allowlists are not containment.
- For untrusted commands/code, run escape and credential tests inside the selected container, VM, or cloud sandbox.

### Skills, dynamic tools, and subagents

- Test configured skill roots, catalog descriptions, model-directed loading, missing references/scripts, behavioral frontmatter, trust, and reload/restart.
- For dynamic tools, assert initial invisibility, selected schema availability on the next request, provider-native/fallback wire behavior, removals, and prompt-cache effects.
- For subagents, test task-only input, history isolation, shared tools/capabilities, result handback, event propagation, budgets, cancellation, errors, and recursion policy.

### RPC, TUI, and provider integrations

- Capture language-neutral fixtures for each consumed RPC/JSON command and event. Run them against Pi and the target adapter, including errors, abort, mid-run input, and session operations.
- Use terminal snapshot or interaction tests only for TUI behavior in scope; a passing agent test does not verify dialogs, keyboard handling, custom rendering, or mode fallbacks.
- For provider extensions, test auth precedence, catalog refresh, model selection, headers/payloads, streaming block assembly, tool calls/results, usage/cost, abort, overflow, and retry against recordings or a controlled server.

## Dependency and cutover checks

1. Resolve the Python project from a clean environment against supported Pydantic AI and Harness versions. Install only the provider and capability extras the slice requires.
2. Run Pi source characterization tests and target tests through the same supported boundary. Record whether each uses deterministic fakes, wire recordings, live providers, terminal automation, fresh-process restart, or a real sandbox.
3. Inspect the target wheel when Pi package behavior becomes a Python distribution. Confirm all capability modules and promised skill/assets are present and importable without development dependencies.
4. Search imports, extension settings, package manifests, resource roots, session readers, provider configuration, RPC/TUI launchers, and deployment scripts for retained `@earendil-works/pi-*` and Node requirements.
5. Keep adapters and rollback while callers or old records still need Pi. Remove Pi packages, settings, Node runtime, and JSONL assumptions only after no retained path uses them.

## Completion criterion

Cut over only when every ledger row is `verified-equivalent`, `verified-adapter`, an accepted `intentional-change`, `external-owner`, or `not-applicable`; no required row is `unverified` or `blocked`; every active extension has an evidenced responsibility split; the supported boundary passes; dependencies resolve cleanly; and rollback remains possible.
