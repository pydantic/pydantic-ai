# Capability Conformance

> What every capability, in Pydantic AI core and Pydantic AI Harness, must satisfy before it ships

**When to check**: Whenever you add a capability, change what an existing one contributes (tools, tool hooks, instructions, parts it enqueues, a workspace), or review a PR that does

A capability is conformant when its entry in `CASES` (`tests/capability_cases.py` for core, `tests/harness/capability_cases.py` for Pydantic AI Harness) passes every contract of each area below, or carries an exemption with a reason a reviewer can check. `test_every_capability_has_a_case` fails until a new capability has an entry, so every area reaches it without anyone listing it. A box is ticked by a passing contract or a reasoned `Exempt`, never by prose: a PR description or a comment saying a capability "handles" an area doesn't count.

Each area says what it requires and how the suite checks it. Contracts are `Contract(applies, exempt)` entries in `CONTRACTS`, run by `tests/harness/test_capability_contracts.py` or by the module an area names; an exempt case must still break its contract, so fixing the capability fails the suite until the exemption is removed. Link an exemption that records a bug to the issue that tracks it.

The contracts are numbered C0, C1, … in the order they land, and an area with contracts of its own numbers them under its letter, as Durable execution's D1. C1–C5 are planned and work toward [#13118](https://github.com/pydantic/pydantic-ai/issues/13118): no capability content below `'operator'` reaches the model in system voice.

## Coverage

**Requirements**

- A capability that contributes a toolset, a wrapper toolset or a tool hook (`before_tool_validate` through `on_tool_execute_error`) declares `calls`: tool calls that run its tools offline. Every contract that runs a capability's tools applies only to a case with `calls`, so without them those contracts pass vacuously.
- A capability whose tools can't run offline carries an exemption saying why, such as `'needs a sandbox backend: …'`. A case that could have `calls` but doesn't yet is exempt with a reason that starts with `PENDING_CALLS`.

**How it's tested**

- `DurableEngineSuite` (`pydantic_ai.durable_exec.conformance`) is the public suite every durable engine runs, shaped like `WorkspaceBackendSuite`: a subclass provides an `engine` fixture implementing `DurableEngine` (`durability()`, `run(agent, prompt)`, `recover(run, after=None)`), and the suite runs one test per cell, a case's calls (`test_case`) or a scenario (`test_scenario`). It runs the agent under the engine, recovers the run, and checks the cell's declared level.
  - `TestDBOS` (`tests/harness/test_durable_tool_io.py`) runs it under DBOS: a run is a workflow, and recovery forks it from after its last step and from the middle of its record.
  - `TestTemporal` (`tests/harness/test_durable_temporal_replay.py`) runs it under Temporal, on one dev server per class and a worker per run. Recovery replays the history with a `Replayer`, which must succeed without a request. Temporal can't resume from part of a history, so `can_fork_mid_run` is `False`.
  - Both feed it, through `BuiltInEngineSuite` (`tests/harness/_durable_cases.py`), every public case with `calls` and every scenario `SCENARIO_RUNS` scripts, and count the requests the cases' stand-ins make. In S1 the suite's `cases` and `scenarios` come from the test side; making `Calls` and `SCENARIO_RUNS` public is the follow-up.
- Each engine declares its support as plain data in its own package, `pydantic_ai/durable_exec/<engine>/_conformance.py`: `SUPPORT`, the cells that aren't `supported`, keyed by case or scenario name, and `RECOVERY`. The engine's subclass points at them, `ENGINE_SUPPORT_MODULES` (`tests/capability_cases.py`) lists them, and `scripts/durable_exec_support.py --write` renders them into `docs/durable_execution/compatibility.md`, which `test_the_compatibility_page_is_current` keeps current. A cell the table doesn't list is `supported`, and asserted. A third-party subclass can set `unlisted = 'pending'` to report a failing unlisted cell as an expected failure.
  - `supported`: the run returns the answer and makes each counted request once. Recovery returns the same output, records the same units, observes the same events and enqueued messages, and makes no counted request again. Recovery from part of the record keeps the units before the cut. When `expected_units` names them, the case's units are its `Calls.durable_steps` for the engine (`DurableSteps(dbos=..., temporal=...)`, names without the agent prefix, the run's own model requests left out but counted, one per call and one to answer). A case whose tools do no I/O lists nothing for DBOS, which is checked too. When a name changes, the failure prints the new list: update the case, with a compatibility note if a recorded name changed.
  - `limited`: the same, and the cell's `check` asserts the limit on the run; `reason` is the docs caveat.
  - `refused`: the engine raises a `UserError` matching `match` before the refused operation records anything. A refusal at build or run start records no unit. A refusal inside a unit (`ctx.emit`, `ctx.enqueue` in an activity or step) comes after the run's model requests are recorded, so the units before it must be unaffected: recovering the record records the same units, refuses again, and makes no counted request again.
  - `broken`: the engine runs it and gets it wrong. `match` is a regex of the symptom, searched for in what the run raised and in each check it fails, and `issue` tracks the bug. Once the bug is fixed the symptom is gone and the cell fails, saying to update it.
  - `untested` is only for the docs table of engines that don't run the suite yet; the suite fails a cell declared `untested`.
  - `Recovery('restart', reason, issue)` declares an engine that restarts a run instead of replaying it (Render). The suite then checks each cell that would assert recovery equality as limited: a recovery makes each counted request exactly once more and returns the same output. The docs show those cells as limited, linking the declaration's issue.
- `DURABLE_DBOS` and `DURABLE_TEMPORAL` (D1) exempt exactly the cases their engine's table marks `refused` or `broken`; they read the table on first use, as it imports the engine's SDK.
- `tests/harness/test_engine_support.py` checks the tables: every cell names a public case with `calls` or a scenario, `test_every_durability_capability_is_an_engine` fails until a new engine's durability capability is in `ENGINE_CAPABILITIES`, and `test_every_durability_capability_has_a_suite` fails until it has a suite or is in `PENDING_ENGINES`, which may only shrink.
- Scenarios, keyed `<owner>.<scenario>` (`Scenario` in `tests/capability_cases.py`), are runs beyond one call per response, such as `BackgroundTools.survives_approval_wait`. The suite runs those `SCENARIO_RUNS` scripts: `ctx.emit` and `ctx.enqueue` from a function tool, from inside a durable unit, and from a hook. A recorder in workflow code notes the custom and capability events and enqueued messages the run observes; the run must observe each scenario's list once, and its recovery the same, with nothing lost or added. The recorder makes the run stream, which changes the model request's unit name, so scenarios' units aren't pinned. A capability's own tests run its other scenarios.
- A case that can't run under an engine is a `refused` or `broken` cell, with the reason: never a skip. A case whose optional dependency (`requires`) isn't installed is skipped: `BrowserUse`'s is installed only by its own CI job, which doesn't run these suites yet.
- Prefect and the Harness engines are next, one per PR, each lowering `PENDING_ENGINES`; until then, `test_prefect_wraps_every_toolset_temporal_does` checks Prefect accepts every case Temporal runs.

## Workspaces and sandboxes

To be written. Planned scope: the workspace authority table is generated from `CASES` (every case that reads workspace files) rather than listed by hand, and every `WorkspaceBackend` class runs `WorkspaceBackendSuite`.
