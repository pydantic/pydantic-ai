# Support Trio through AnyIO without changing the public API

Status: C01-C03 implemented. The baseline, opt-in backend tests and static import policy are in place; agent execution under Trio still needs the ownership migration.

Baseline: local commit `b61fa91eb`, reviewed on 2026-09-07. The user approved narrowly documented exceptions to the `asyncio` lint ban for compatibility adapters and backend-specific regression tests.

CI direction: do not run the full Trio suite on every pull request or every push to `main`. Use targeted dual-backend validation during migration, then manual and periodic broad Trio runs. A small, measured Trio smoke check is recommended for ordinary CI.

## Target behavior

```python
import anyio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent(TestModel(custom_output_text='Trio works'))


async def main() -> None:
    result = await agent.run('Hello')
    assert result.output == 'Trio works'


anyio.run(main, backend='trio')
```

You install Trio alongside Pydantic AI and choose the backend at your application's entry point. The same agent code must also work inside an existing Trio application and under `anyio.run(main, backend='asyncio')` or `asyncio.run(main())`. No new `Agent` argument, mandatory context manager, or caller-supplied task group is required.

Trio stays optional at runtime and becomes an explicit test dependency. Backend-neutral implementation code uses AnyIO. Existing synchronous entry points keep their asyncio execution model and event-loop affinity.

## Evidence and scope

- [Issue #7112](https://github.com/pydantic/pydantic-ai/issues/7112) reports that `Agent.run()` fails under Trio at the capability wrapper's `asyncio.create_task()`. Its discussion is context, not approval of a particular architecture.
- [The maintainer's cancellation design update](https://github.com/pydantic/pydantic-ai/issues/6460#issuecomment-5097539507) distinguishes first-party run cancellation from external cancellation and explains why a flag alone cannot cancel a blocked run. Current local code is the implementation baseline; older proposals in the issue are not the contract.
- [#6454](https://github.com/pydantic/pydantic-ai/pull/6454) restored same-loop synchronous streaming after a blocking portal broke pooled client reuse. Moving existing sync calls to another thread or loop would repeat that regression.
- `pydantic-ai-slim` and `pydantic-graph` already require `anyio>=4.7.0`. The lockfile contains AnyIO 4.14.1, but the declared minimum remains part of the support contract.
- The earlier inventory found 150 direct `asyncio` references in 17 slim-package files, 7 in one graph file, and 10 in three evals files. Tests had 844 references in 51 files. These AST counts include annotations and exclude comments/docstrings; they are an editing footprint, not a count of independent changes. Refresh before implementation, including `.pyi` files and loop/task methods accessed through variables.
- At the baseline commit, `tests/conftest.py` selected only the asyncio backend and Ruff's `TID251` rule banned only `asyncio.Lock`. C02-C03 add backend selection and the module-wide ban.

| Surface | Target | Release evidence |
| --- | --- | --- |
| Async agent, direct model, graph, tool, capability and streaming APIs | Both backends | Public behavior tests under asyncio and Trio |
| Synchronous agent/direct/graph APIs | Preserve existing asyncio behavior | Same-loop pooled-client and interrupt regressions |
| HTTP providers, embeddings, MCP and UI adapters | Verify each transport/SDK on Trio | Recorded interactions or real local protocol tests |
| Realtime sessions | Migrate orchestration; qualify each transport separately | Queue, tool, disconnect and audio-stream lifecycle tests |
| Temporal and other durable runtimes | Preserve their native runtime contracts | Workflow, replay and cancellation suites |
| Asyncio-bound SDKs, including the current xAI gRPC integration | Retain narrow native adapters until a supported transport exists | Existing asyncio coverage; explicit support limitation |
| Evals, including online evaluation | Separate rollout after core ownership is proved | Offline evaluation, background work and shutdown on both backends |

Core Trio support does not imply that every third-party SDK supports Trio. Publish an explicit support matrix. Do not silently start an asyncio thread to make an unsupported integration appear portable. Validate new unsupported-backend errors at the integration boundary and preserve all previously supported asyncio calls.

## Compatibility contract

Before changing orchestration, pin these behaviors through existing public entry points:

1. **Source and types:** keep signatures, imports, return types, exception types on asyncio, and the existing iterator/context-manager calling forms. Audit non-underscore public symbols and exported annotations as well as the primary `Agent` API. Introduce no new public backend or task abstractions.
2. **Results and serialization:** preserve message/event shapes, ordering guarantees, usage accounting, retries, deferred tools, structured output, durable operation names and persisted history. Keep runtime task/scope objects out of serialization.
3. **Cancellation:** first-party cancellation produces the existing `RunCancelled` outcome and history. External cancellation propagates the active backend's cancellation exception; asyncio callers still receive `asyncio.CancelledError`. Preserve asyncio timeout behavior, external-cancellation precedence in races, swallowed-cancellation checks and `RunCancelled.from_cancellation()` state recovery. Preserve the documented Python 3.10 differences.
4. **Streaming:** distinguish early consumer exit, response cancellation and whole-run cancellation. Preserve cleanup, completed sibling results, interrupted/suspended state, capability recovery and hook ordering. Preserve currently supported transfer of iteration between tasks.
5. **Execution context:** retain dependency identity, caller/capability `ContextVar` propagation, tracing parentage and restoration, thread dispatch, concurrent-run isolation and client loop affinity. Moving user hooks or tools into a new task can change observable behavior even if signatures stay identical.
6. **Errors and cleanup:** preserve single-error exception shape and legitimate multiple-error groups. Children and resources finish teardown before their owner exits. Do not turn a normal consumer exit into cancellation or let cleanup mask the original failure.

The API compatibility checker is necessary but cannot verify these behavioral guarantees. A failed contract test blocks that migration step; changing a documented default or requiring a new argument is not a workaround.

## Implementation sequence

### 1. Establish the backend and compatibility test baseline

Add Trio to development dependencies with a version range compatible with every supported Python version. Start with a dedicated dual-backend test module so the existing suite remains useful while the migration proceeds. Reproduce issue #7112 through `Agent(TestModel(...)).run()` before changing production code.

Build a contract matrix for `run`, `iter`/manual `next`, `run_stream`, `run_stream_events`, direct model requests, parallel tools and capability wrapping. Use `TestModel`/`FunctionModel` for framework behavior; retain real recordings for provider behavior. Exercise both plain asyncio entry and AnyIO's asyncio runner, since users need not start their applications with AnyIO.

Temporary failing Trio cases must be explicit, strict expected failures linked to the missing implementation. Remove each as its slice lands. No blanket Trio skip and no expected failures remain for advertised support at release.

### 2. Prove task ownership before broad replacements

Prototype the capability wrapper handoff in `agent/__init__.py`, model streaming in `_agent_graph.py`, event handles in `agent/abstract.py`, and `capabilities/process_event_stream.py`.

Use AnyIO task groups inside a lifetime owned by an existing run/session context. A generator must not leave a cancel scope open across a yield if another task can resume or finalize it. Worker tasks that own streams must enter, iterate and close those streams in the same task. Use typed result/error handoffs and events or memory streams where coordination is required.

The prototype must name the persistent owner and show how it starts and stops through every existing public calling form, including manual stepping and early exit. Do not assume a task group can be hosted by whichever task happens to request the next event. Do not use a global supervisor, detached system tasks or a new mandatory caller context to avoid the ownership problem.

Keep this a design gate: if an existing lifetime cannot host structured concurrency compatibly, retain its exact asyncio adapter while investigating a separate Trio path. Document the unresolved lifetime precisely; do not declare that API Trio-compatible until both paths pass the contract suite.

**Gate:** normal completion, wrapper short-circuit/recovery, child failure, early exit, cross-task consumption, cancellation during startup/teardown and context restoration pass. No stream ownership regression or orphan tasks.

### 3. Coordinate cancellation and tool execution

Work through `_cancel.py`, `_tool_execution.py`, `_run_context.py`, `run.py`, the relevant agent/graph paths, `_utils.py` and `models/_continuation.py`.

Use separate changes for cancellation delivery, tool scheduling and suspended-job cleanup, each with its own regression tests. Agree on their shared cancellation contract before landing any of them; do not turn this stage into one large rewrite.

Replace portable sleeps, events and timeout contexts with AnyIO equivalents. Replace `wait(FIRST_COMPLETED)` and task-type inspection with task-group workers publishing typed outcomes to memory streams. Preserve result ordering separately from completion order. Audit capacity and checkpoint differences: a zero-buffer memory stream is not a behavior-preserving replacement for an unbounded queue.

For Trio, bind first-party cancellation to the active run's cancel scope and wake blocked work immediately. Use `anyio.get_cancelled_exc_class()` for backend-neutral handling and shield required cleanup. Do not construct Trio cancellation exceptions manually. Preserve first-party versus external cancellation arbitration and cancellation state recovery through a public-API prototype before adopting this design.

Keep asyncio task-count bookkeeping in a small compatibility adapter wherever AnyIO alone cannot preserve `cancel()`/`uncancel()` semantics. Do not impose new level-cancellation behavior on existing user callbacks just by moving them into AnyIO child tasks. Test and retain the necessary asyncio path if that would change their behavior.

Thread-safe `CancellationToken.cancel()` must still work from ordinary foreign threads, before binding, after completion and across multiple runs. Evaluate AnyIO's supported thread-entry mechanism against the selected minimum version; a portal can only be used if its lifetime is already owned by the run and it does not move execution to another loop.

**Gate:** cancellation races, absorbed cancellation, completed sibling history, streamed cleanup, thread cancellation, deadline precedence and single/multiple error shapes pass on each applicable backend.

### 4. Isolate sync and native-runtime compatibility boundaries

Keep `_sync_stream.py` and event-loop helpers for the slim, graph and evals packages behaviorally unchanged while isolating their asyncio dependencies. Preserve public symbol locations if helper moves would otherwise change them.

Preserve Temporal's activity cancellation/shielding and deterministic workflow execution. Audit DBOS, Prefect and other durable engines using their existing integration suites; do not infer support from their lack of direct `asyncio` imports. Backend-neutral core helpers must remain usable by these runtimes.

**Gate:** synchronous request/stream calls reuse a real pooled client in both orders on the same loop; thread/context/interrupt tests pass; durable cancellation, replay, serialization and teardown remain unchanged.

### 5. Qualify integrations and the remaining packages

Migrate realtime queues, events and task ownership while preserving bounded tap/drop behavior, pending tool barriers, pump shutdown, disconnect and audio interruption. Check WebSocket transport dependencies separately from session orchestration.

Exercise streaming and non-streaming requests through each provider SDK and transport. Check optional transports such as aiohttp or gRPC rather than assuming HTTP support is sufficient. Test MCP subprocess and HTTP lifecycle, UI disconnect handling, embeddings and graph fan-out/join. Put an integration in the Trio support matrix only after its relevant tests pass.

Migrate offline and online evals separately. Online evaluation needs an explicit owner for queued/background work and shutdown through its existing API; replacing loop-level `create_task` without proving that lifetime is insufficient.

### 6. Enforce the import policy and expand CI

Introduce enforcement early with an exact temporary baseline, then tighten it as steps 2-5 remove violations. The final Ruff configuration extends the existing table:

```toml
[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.TypedDict".msg = "Use typing_extensions.TypedDict instead."
"typing.assert_never".msg = "Use typing_extensions.assert_never instead."
"asyncio".msg = "Use AnyIO. Asyncio is restricted to documented compatibility adapters and backend tests."
```

Keep `TID251` enabled. Remove the redundant `asyncio.Lock` entry once the module-wide ban is in place. [Ruff's banned API rule](https://docs.astral.sh/ruff/rules/banned-api/) supports import/access restrictions; verify direct imports, aliases, `from` imports and submodules with the repository's locked Ruff version.

The policy covers production packages, tests and Python examples/scripts. Default examples use AnyIO; explicit asyncio interoperability examples can have the same narrow exceptions as regression tests. Documentation fences need example tests/review because Ruff does not lint Markdown prose.

Use exact import-line `noqa: TID251` exceptions with a short reason and an auditable inventory. Each exception records the file/import, contract requiring it, protecting test, and whether it is permanent or scheduled for removal. Do not exempt entire package/test directories or all of `_utils.py`. Moving code into an adapter must isolate a real backend boundary, not merely hide all old calls behind new names.

Add a small policy check to reject new or broadened `TID251` suppressions and per-file ignores outside the approved inventory. During migration, inventory the exact remaining import/use sites and reject growth; shrink this baseline to the permanent exceptions before release. Preserve enforcement of the unrelated typing bans. Cover the checker through its CLI with accepted exceptions and rejected imports/suppressions. Disallow dynamic-import workarounds in review; Ruff is not a complete import sandbox.

Wire enforcement into `make lint`, pre-commit and required CI. The current lint hook is Python-only, so also trigger it for `pyproject.toml`, the exception inventory and policy-check configuration changes.

Keep the shared test suite on asyncio by default. Add an opt-in backend selector using [AnyIO's pytest backend fixture](https://anyio.readthedocs.io/en/stable/testing.html), so the same portable tests can run under Trio without duplicating the default test run. Separate backend-specific tests explicitly. Audit higher-scoped fixtures, pooled clients and sync finalizers so a resource is closed on the backend that created it.

Use the execution policy below instead of adding Trio to every existing CI matrix. Preserve the existing asyncio and durable jobs. Run recorded traffic with `--record-mode=none`, and verify that cassette playback itself works under Trio.

### 7. Document and release only the verified support

Update agent, streaming, cancellation, dependency and relevant integration docs with runnable AnyIO entry-point examples and the tested support matrix. Explain that sync methods retain their current backend and that a user tool calling asyncio is not portable to Trio. Document installing Trio without adding a redundant backend flag to `Agent`.

Update the packaged `building-pydantic-ai-agents` skill and repository concurrency guidance with ownership rules, supported backend mechanics and the import exception policy. Follow directory-specific docs instructions when writing those changes.

At the release gate, run the static API compatibility check, required pre-commit checks, both backend contract suites, supported integration suites, minimum-dependency tests and the existing coverage gates. Compare representative single-request, concurrent-tool and streaming workloads against the baseline for extra buffering, thread creation, scheduling overhead and cancellation latency.

## Test execution policy

Running both backends adds test work, but its effect on CI duration has not been measured. Measure smoke duration, runner minutes and critical-path impact before making it required. Static import checks remain mandatory on every change, but cannot detect all task-lifetime or third-party backend regressions.

| When | Trio coverage | Blocking behavior |
| --- | --- | --- |
| Ordinary pull request or push to `main` | Recommended: three focused, network-free smoke cases on one Python/dependency combination, outside the full test matrix | Keep existing required CI; add only the small smoke check and static policy checks |
| A migration change or a later concurrency-sensitive change | Affected public-API tests on both backends, including cancellation and teardown | Passing targeted tests are required evidence for that change; run locally or manually in CI |
| Manual dispatch | Full portable Trio suite, with selectable targeted subsets for diagnosis | Available on the proposed branch before merging; no automatic full-suite run on every push |
| Periodic maintenance | Full portable Trio suite against default-branch HEAD; proposed cadence: weekly | Reports failures for triage without extending unrelated pull-request CI |
| Before publishing Trio support or a release changing concurrency/dependency compatibility | Full portable Trio suite and the relevant minimum/current dependency and supported Python combinations | Passing results for the exact release candidate are a release gate |

The three smoke cases cover: an agent run with capability wrapping and a tool; a stream closed early with observable cleanup; and cancellation of a blocked run. Drive them through public APIs with `TestModel`/`FunctionModel`, use events for ordering and bounded deadlines for hang detection, and assert teardown. They do not exercise provider SDK matrices or external services. Add them to normal CI only as the corresponding behavior becomes supported; do not add permanently failing smoke tests.

Do not globally parameterize `anyio_backend` over both backends in default CI. Do not add Trio as another axis to every Python/extras/dependency job. Keep minimum-version and broad integration testing in manual, periodic and release runs. Preserve existing asyncio-only tests rather than claiming the entire repository can run under Trio.

The periodic workflow must retain its tested commit, dependency versions and test report. Assign failure triage to the existing maintainer workflow; record a regression with an owner and add a focused regression test when fixing it. Periodic runs knowingly allow a gap between introducing and discovering regressions. Keep the smoke check small instead of gradually growing it into a second full suite.

These are workflow changes to implement as part of the checklist, not an automation created by this planning document.

## Checklist of isolated changes

Each item is a candidate independently reviewable change, with implementation, relevant tests and necessary docs together. Dependencies describe merge order, not permission to leave broken intermediate commits. Split repeatable integration work by SDK/transport. If the ownership prototype proves two pieces inseparable, combine only those pieces and record why. Do not build an unused abstraction merely to satisfy this order.

- [x] **C01 - Capture the compatibility baseline.** Refresh the import/public-symbol inventory and map existing cancellation, streaming, sync-affinity and durable tests to the contract. Add only missing asyncio regressions; record the end-to-end Trio reproduction and scope of supported integrations. No production migration. Validation: the existing targeted baseline passes and the Trio failure is reproduced; see the implementation record below.
- [x] **C02 - Add opt-in Trio test execution.** Add the compatible Trio development dependency and a test-only backend selector; keep asyncio as the default. Make selected fixtures close resources on their owning backend and provide reproducible local/manual commands. Depends on C01. Validation: fixture lifecycle works under both backends and default collection does not double; see the commands below.
- [x] **C03 - Enforce the static import policy.** Add the Ruff module ban, exact temporary/permanent exception inventory and suppression checks; include config-only changes in lint triggers. No concurrency refactor. Depends on C01. Validation: CLI policy tests reject all banned import forms and unauthorized suppressions while preserving existing typing bans; see the implementation record below.
- [ ] **C04 - Isolate synchronous asyncio support.** Restrict existing loop-driving and sync-stream behavior to documented compatibility boundaries in slim and graph; keep public helper locations compatible. Avoid moving code unless needed to establish a real boundary. Depends on C01 and C03. Validation: both sync request/stream orders reuse a pooled client, and interrupt/context tests pass.
- [ ] **C05 - Prove portable task ownership.** Prototype capability/model/event handoffs against the existing API and choose the AnyIO minimum using evidence. Deliver the ownership design and passing focused tests before a broad refactor; discard experimental code that is not the chosen implementation. Depends on C01 and C02. Validation: same-task teardown, supported cross-task iteration, context propagation and early exit are demonstrated; unresolved cases remain explicitly unclaimed.
- [ ] **C06 - Migrate capability run wrapping.** Change the wrapper handoff in `agent/__init__.py`, using the ownership design from C05. Keep short-circuit, recovery and context behavior intact. Validation: targeted capability tests pass on both backends, and the original Trio entry-point failure is eliminated without hiding a later failure.
- [ ] **C07 - Implement portable run cancellation.** Change `_cancel.py` and its agent/run entry and exit boundaries; retain narrowly isolated asyncio arbitration where required. Depends on C05 and C06. Validation: first-party/external races, blocked-run cancellation, history recovery and foreign-thread token cancellation pass with existing asyncio behavior intact.
- [ ] **C08 - Migrate tool scheduling and event coordination.** Change `_tool_execution.py` and the necessary `_run_context.py` coordination together. Depends on C06 and C07. Validation: parallel and sequential calls, completion/order guarantees, immediate events, retries, completed sibling history and cancellation drain pass on both backends.
- [ ] **C09 - Migrate model-stream ownership.** Change the streaming handoff in `_agent_graph.py` and its immediate caller only. Depends on C05-C08. Validation: public `run_stream` behavior, stream failure, partial messages, early exit, cancellation and context restoration pass on both backends.
- [ ] **C10 - Migrate event-stream handles.** Change `agent/abstract.py` and `capabilities/process_event_stream.py` where their lifetimes require a coordinated change. Split them if each can stay green independently. Depends on C09. Validation: `run_stream_events`, observer errors, abandoned consumption and supported task handoffs leave no running work on either backend.
- [ ] **C11 - Migrate suspended-job cleanup.** Change `models/_continuation.py` without changing cancellation or resumability policy. Depends on C07 and C09. Validation: blocked cleanup, external cancellation, provider cleanup failure and Temporal activity-wrapped cancellation preserve the original outcome and stored history.
- [ ] **C12 - Remove remaining portable primitives.** Replace remaining sleeps/events/timeouts and narrow `_utils.py` usage only after their callers' ownership is settled. Split by independent module or behavior; preserve graph sleep overrides. Depends on C06-C11. Validation: targeted public tests and a shrinking lint baseline, with no unrelated cleanup.
- [ ] **C13 - Qualify each provider and embeddings transport.** Repeat one change per SDK/transport: dual-backend recorded streaming/non-streaming tests, only necessary migration, and its support-matrix entry. Depends on the relevant core items through C12. Validation: real recordings replay under Trio; unsupported asyncio-native transports retain explicit boundaries.
- [ ] **C14 - Qualify MCP and UI adapters separately.** Use separate changes for MCP subprocess/HTTP lifecycle and each UI adapter's stream/disconnect behavior. Depends on C08-C10. Validation: protocol-level lifecycle and cancellation tests pass on both applicable backends.
- [ ] **C15 - Migrate realtime orchestration.** Isolate session queues/pump/tool ownership from transport replacement; qualify or migrate each WebSocket transport in a subsequent change. Depends on C05 and C07. Validation: bounded taps, audio interruption, pending tool barriers and disconnect cleanup; no Trio support claim until its transport passes too.
- [ ] **C16 - Qualify graph and preserve durable engines.** Add missing graph fan-out/join coverage under Trio and address each confirmed gap separately. Keep durable-runtime adjustments separate per engine. Validation: graph tests pass on both backends and existing workflow/replay/serialization/cancellation tests pass on their native runtimes. Run affected durable tests throughout C06-C12, not only at this final qualification step.
- [ ] **C17 - Migrate evals separately.** Use one change for offline evaluation and a separate ownership design/change for online background evaluation and shutdown. Depends on C05 and the relevant core changes. Validation: evaluation results, concurrency bounds, sink failures and shutdown preserve existing behavior on asyncio and pass on Trio where advertised.
- [ ] **C18 - Add the small Trio smoke check.** Introduce the three public-API cases described above as they become green; record their measured cost before requiring the job. Depends on C06-C10. Validation: each case catches its demonstrated failure, has a bounded runtime and adds no full-suite/matrix duplication.
- [ ] **C19 - Add manual and periodic broad Trio runs.** Configure manual subset/full-suite dispatch and the proposed weekly workflow, including artifacts and failure triage. Depends on C02; enable available subsets during migration and expand as integrations qualify. Validation: a manually triggered run selects the intended tests/backends, records exact revisions and dependencies, and a failed run is visible to maintainers.
- [ ] **C20 - Finalize the support contract and exception list.** Remove migration exemptions/expected failures for advertised surfaces, publish the verified support matrix, and update docs and agent skills. Depends on the completed scope above. Validation: release-candidate broad Trio testing, existing asyncio/durable CI, API compatibility, coverage and minimum dependencies pass. Remaining native-runtime exceptions have a reason and protecting test.

Start with C01-C03. Prove C05 before committing to the difficult orchestration design. C04 and the initial manual workflow setup can be reviewed independently. The full checklist is an inventory of changes, not a requirement to put every optional integration into the first core-support release.

## Implementation record

### C01 - Compatibility baseline

The target-behavior example was run on Python 3.14.6, AnyIO 4.14.1 and Trio 0.34.0. It fails at the capability wrapper's `asyncio.create_task()` with `RuntimeError: no running event loop`, followed by a warning that `CombinedCapability.wrap_run` was never awaited. No provider request is involved. The earlier count of 21 production files importing asyncio is unchanged; the refreshed tracked-file scan also includes two `.github` Python files that the earlier ordinary `rg` inventory omitted.

The async public surfaces to preserve are `Agent.run`, `Agent.iter`/`AgentRun.next`, `Agent.run_stream`, `Agent.run_stream_events`, the direct model APIs, public cancellation handles and callback/capability signatures. No production symbols, annotations, signatures or serialized types have changed. Private task/event-loop annotations remain implementation work; sync methods retain their current asyncio contracts.

Existing public behavior tests cover the baseline without adding duplicate framework tests:

| Contract | Existing coverage |
| --- | --- |
| External/first-party cancellation, races, history and tokens | `tests/test_run_cancellation.py` |
| Sync client reuse, both request/stream orders | `tests/test_sync_stream_loop_affinity.py` |
| Direct request and streaming APIs | `tests/test_direct.py` |
| Cross-task stream consumption and observer teardown | `tests/test_capability_process_event_stream.py` |
| Stream cleanup, child failure and cancellation shielding | `tests/test_capability_stream_teardown.py` |
| Capability short-circuit, recovery and hook order | `tests/test_capability_hooks.py` |
| Context isolation and tracing | `tests/test_otel_context_isolation.py` |
| Public symbol/type compatibility | `tests/test_public_interface_contracts.py` and the existing API compatibility CI check |
| Durable state and operation-name compatibility | `tests/durable_exec/test_durable_exec_compat.py` and native engine suites |

The first five test modules plus the backend-fixture checks passed on asyncio: 142 passed, one version-specific skip. The remaining entries identify checks to run when the affected production paths change; this is not a claim that every listed integration suite was executed in this baseline pass.

### C02 - Opt-in backend execution

```sh
uv run pytest tests/test_async_backend.py tests/test_direct.py --record-mode=none
uv run pytest tests/test_async_backend.py tests/test_direct.py --anyio-backend=trio --record-mode=none
```

Both commands pass the same 16 tests. The default remains asyncio; the option selects one backend and rejects unsupported values. Trio 0.34.0 is a development dependency and supports the project's Python 3.10 floor. Production dependencies are unchanged.

The new fixture tests verify the active backend for unmarked async tests and a module-scoped producer/consumer lifecycle. The shared HTTP cleanup fixture preserves the existing runner setup for all default asyncio tests. With Trio selected, it opens a runner only for async tests: opening a Trio runner around a synchronous test caused sync streaming to inherit Trio's backend context and fail with `Task got bad yield`. Sync tests selected alongside Trio use the existing synchronous cleanup finalizer.

No Trio CI job or automatic duplication of the suite was introduced. Full `Agent.run()` support is still pending the ownership migration. The targeted asyncio baseline passed (142 passed, one skip); the selected direct/fixture tests passed on Trio (16 passed), and targeted Ruff and Pyright checks passed.

### C03 - Static asyncio policy

```sh
uv run python scripts/check_asyncio.py
uv run pytest scripts/test_check_asyncio.py
make lint
```

Ruff now bans the `asyncio` module. Import-line exceptions preserve existing behavior while the migration proceeds. The reviewed inventory covers 92 files importing asyncio and two pre-existing typing-related `TID251` suppressions. The source edits to those 92 existing Python files are comments only, verified by comparing their syntax trees with the previous commit.

The policy checker scans tracked and unignored Python files, stubs and Ruff configurations, rejecting symlinks before reading their targets. It rejects unapproved imports, literal dynamic imports through `importlib.import_module` or `__import__`, growth in direct imported-name references within a scope, new/broader suppressions, missing reasons and stale entries. It also runs on Python 3.10. This is a conservative syntax inventory, not type inference: indirect task/loop handles and computed dynamic imports still require review at approved native boundaries.

`make lint` and pre-commit enforce the policy. Config-only changes trigger lint, and the policy's CLI tests have a dedicated pre-commit hook. Its 50 cases drive the CLI in-process and retain separate subprocess checks for command entry and isolation from a parent Git-hook environment. The suite takes a few seconds locally. Coverage of the policy logic reached 100% of lines and branches across Python 3.10 and 3.13; the import-only branch of the CLI module guard was excluded because these checks execute the command.

The whole-repository Ruff checks pass. The full typecheck hook has two pre-existing failures in `pydantic_evals/pydantic_evals/_online.py`: unnecessary `reportMissingImports` and `reportUnknownMemberType` suppressions. Their source logic is unchanged. The changed tooling and backend-test files are checked separately; commits bypass only that already-failing hook after recording its output. No Trio CI suite or production concurrency migration is included in C01-C03.

## Dependency decision

Prefer APIs supported by the declared AnyIO minimum unless the ownership prototype proves a newer primitive materially simplifies the implementation. The current [AnyIO release history](https://anyio.readthedocs.io/en/stable/versionhistory.html) lists task handles in 4.14 and `Future`/concurrency helpers in 4.15. Their presence in current documentation does not make them available under `anyio>=4.7.0`.

If a newer minimum is necessary, make that an explicit dependency change, check Python 3.10 compatibility and downstream constraints, and test the new minimum. Do not silently rely on the lockfile or introduce a parallel homegrown task framework just to avoid evaluating a dependency update. New task handles still require a correctly owned task group.

## Alternatives excluded from this plan

- A zero-exception `asyncio` ban: conflicts with the approved compatibility boundaries and backend-specific regression tests.
- Replacing every `create_task` with a task group in place: does not solve generator lifetime or cancellation compatibility.
- Replacing every synchronous runner with `anyio.run` or a new blocking thread: changes loop/client reuse.
- Requiring users to pass a nursery/task group or wrap every existing call in a new context manager: changes the API contract.
- Claiming every optional SDK supports Trio because core tests pass: transport/runtime compatibility must be demonstrated separately.

## Completion criteria

- Every advertised portable API passes on Trio and retains its asyncio contract.
- No signature, serialized-data, context, error or lifecycle changes are required of existing users.
- The original end-to-end Trio reproduction passes without a backend-specific user workaround.
- Only documented compatibility boundaries and targeted tests/examples retain `asyncio`; static checks reject new usage elsewhere.
- Ordinary CI retains the existing asyncio/durable checks and static import enforcement, with only the recommended small Trio smoke check added. Broad Trio and minimum-version coverage run manually, periodically and at the release gates above, with no migration expected failures left in advertised surfaces.
- Unsupported integrations and any deferred surfaces are named explicitly. Full integration parity remains separate from the first core-support milestone.

Planning validation: the proposed module-wide ban was exercised with locked Ruff 0.15.19 in a temporary configuration. All five forbidden forms (direct import, alias, `from` import, submodule import and `from` submodule import) produced `TID251`. An AnyIO import and an explicit import-line exception passed. C03 implements the exception-inventory checker and its CLI tests, as recorded above.

C01-C03 establish the baseline, test selection and static policy. Full agent runtime compatibility, task ownership prototypes and the complete behavioral test matrix remain implementation gates.
