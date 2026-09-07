# Support Trio through AnyIO without changing the public API

Status: proposed implementation plan, not an implemented compatibility guarantee.

Baseline: local commit `b61fa91eb`, reviewed on 2026-09-07. The user approved narrowly documented exceptions to the `asyncio` lint ban for compatibility adapters and backend-specific regression tests.

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
- `tests/conftest.py` selects only the asyncio backend. Ruff already enables `TID251`, but currently bans only `asyncio.Lock`.

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

### 3. Migrate cancellation and tool execution together

Work through `_cancel.py`, `_tool_execution.py`, `_run_context.py`, `run.py`, the relevant agent/graph paths, `_utils.py` and `models/_continuation.py`.

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

Once the core passes, parameterize shared async tests for asyncio and Trio using [AnyIO's pytest backend fixture](https://anyio.readthedocs.io/en/stable/testing.html). Separate backend-specific tests explicitly. Audit higher-scoped fixtures, pooled clients and sync finalizers so a resource is closed on the backend that created it.

Add a required Trio CI job initially; expand it across the supported Python and minimum/current dependency combinations without unnecessarily duplicating asyncio-only integrations. Preserve the existing asyncio and durable jobs. Make the aggregate required CI check depend on the new job. Run recorded traffic with `--record-mode=none`, and verify that cassette playback itself works under Trio.

### 7. Document and release only the verified support

Update agent, streaming, cancellation, dependency and relevant integration docs with runnable AnyIO entry-point examples and the tested support matrix. Explain that sync methods retain their current backend and that a user tool calling asyncio is not portable to Trio. Document installing Trio without adding a redundant backend flag to `Agent`.

Update the packaged `building-pydantic-ai-agents` skill and repository concurrency guidance with ownership rules, supported backend mechanics and the import exception policy. Follow directory-specific docs instructions when writing those changes.

Run the static API compatibility check, required pre-commit checks, both backend contract suites, supported integration suites, minimum-dependency tests and the existing coverage gates. Compare representative single-request, concurrent-tool and streaming workloads against the baseline for extra buffering, thread creation, scheduling overhead and cancellation latency.

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
- Required CI includes Trio, the existing asyncio/durable suites and minimum-version coverage, with no migration expected failures left in advertised surfaces.
- Unsupported integrations and any deferred surfaces are named explicitly. Full integration parity remains separate from the first core-support milestone.

Planning validation: the proposed module-wide ban was exercised with locked Ruff 0.15.19 in a temporary configuration. All five forbidden forms (direct import, alias, `from` import, submodule import and `from` submodule import) produced `TID251`. An AnyIO import and an explicit import-line exception passed. These seven probes confirm the basic rule behavior; the exception-inventory checker remains planned work.

This planning pass reviews source and existing issue evidence. Runtime compatibility, task ownership prototypes and the full behavioral test matrix are implementation gates, not results already demonstrated by this document.
