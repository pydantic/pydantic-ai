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

- `DECLARES_CALLS` (C0) builds every buildable public case and checks it: `test_a_capability_that_runs_with_tools_declares_calls`. A capability made of others is checked through its parts.
- `test_pending_calls_backfill` pins how many cases are still pending. A PR that writes a case's `calls` removes its exemption and lowers the count; a new capability doesn't add a pending one.
- For a capability whose hooks run around other tools' calls, `with_echo` builds it beside an `echo` tool, and `ECHO` calls it.

## Prompt cache

**Requirements**

- Every request a capability sends repeats the earlier one as a prefix: it appends to history and never rewrites what an earlier request sent, its instructions are addressable, and the same history projects to the same bytes. The rules are in [Message History & Prompt Caching](message-history-and-caching.md).
- Two of a capability on one agent mean what its `combine` policy says, and stating one configuration twice reaches the model the same as stating it once. The capability decides that policy explicitly: a default `id` when two are one configuration (fixed tool names, or one run-wide policy), so they merge, and none when they compose; see [Deciding What Two Of It Mean](../src/pydantic_ai_harness/agent_docs/capability-authoring.md#deciding-what-two-of-it-mean).

**How it's tested**

- `tests/test_history_append_only.py` and `tests/harness/test_history_append_only.py` run every buildable case through a scripted conversation on each provider adapter. A capability that keeps state the conversation should change gets a `SCENARIOS` entry; a prefix move that is a bug goes in `KNOWN`, linked to its issue, and one that is documented behavior (compaction, a request-only tail) goes in `ALLOWED`, which pins how far it may move. A case whose `calls` rewrite history by design sets `moves_cache_prefix`.
- `tests/test_capability_combine.py` and its Harness twin check each case's `combine` policy; `COMBINE_FIXED_POINT` checks that a configuration stated twice is stated once.

## Voice and authority

**Requirements**

- Every model-visible text a capability produces carries its source: a tool's output is `ToolResultSource`, another model's text `ModelSource`, a workspace file `WorkspaceSource`.
- Below `'operator'`, it never renders as a system or developer message, and never authors a `ModelResponse`.
- Authority only goes down: a part reaches the model with the lower of its source's and its kind's authority, never raised.
- The attribution is the same on every route the content takes: a native tool call, Code Mode, IPython Code Mode, `ToolBridge`, `BackgroundTools`, a sub-agent, a durable engine, and a round trip through `ModelMessagesTypeAdapter` or a UI protocol.

The rules behind these are in [Message History & Prompt Caching](message-history-and-caching.md), "Trust tiers map to channels, and nothing is ever promoted".

**How it's tested**

- C0 (above) makes these contracts reach every capability that runs tools.
- C1, route matrix (planned): each case's `calls` run at `with_authority('none')` through every route, and the typed history must match the native route's source, authority and frame.
- C2, sink check (planned): `Model.prepare_messages` asserts no part below `'operator'` renders as system or developer, and no capability below `'operator'` authors a `ModelResponse`.
- C3, role-site guard (planned): every site that builds a system or developer part is in an allowlist with its reason.
- C4, transformation registry (planned): every function that returns `list[ModelMessage]` is registered, and sources survive each one.
- C5, attribution on spans (planned): OpenTelemetry message parts keep each capability part's source.

## Durable execution

**Requirements**

- Every request a tool makes goes through a `@durable_operation` or a workspace call (`ctx.workspace`). DBOS runs a function tool in workflow code, so a request made any other way is made again on every recovery.
- Every toolset a capability builds has an `id`, and is built once, when the agent is: Temporal and Prefect run each function tool in a unit named after it, and refuse an agent with an id-less toolset or a run that adds one. A capability with `@durable_operation`s and no default `id` needs one given (`ToolGuardrail(id=...)`), and its case gives one.
- Code that runs in the workflow is deterministic: hooks, instructions, `prepare_tools`, a toolset factory, and a tool marked `ToolDefinition.durable=False`, which runs outside its unit. It reads clocks, randomness and I/O only through a `@durable_operation`, and a workflow-side tool must not block.
- A capability added per run only adds hooks, instructions or settings: one that contributes an executing toolset, declares a `@durable_operation`, or chooses or wraps the model must be on the agent when it's built ([#13124](https://github.com/pydantic/pydantic-ai/pull/13124)).
- The names of the units a capability's calls record (its id, its operation names, its toolset ids) don't change between releases, as an in-flight workflow replays by them. A change that renames one says how in-flight runs recover.
- A capability doesn't check which engine it runs under. When it needs an engine fact, such as whether the engine runs function tools in workflow code, it reads the engine's `DurabilityEngineSpec`, so a new engine gets the right answer, refusal included, without a branch in the capability. A fact the spec doesn't have yet is an engine gap to raise with the durable owner.

The engine-side rules (what crosses a durable boundary, workspace routing, workflow-side determinism) are in `pydantic_ai_slim/pydantic_ai/durable_exec/AGENTS.md`, and the core definitions the engines depend on are listed in `scripts/durable_exec_contract.json`. This section covers what a capability must do; those cover what the engines and core must keep.

**How it's tested**

- D1, durable per case: every case with `calls` runs them, one per response, under each engine, and the units the run records must be the ones its `Calls.durable_steps` lists (`DurableSteps(dbos=..., temporal=...)`, names without the agent prefix, the run's own model requests left out). A case whose tools do no I/O lists nothing for DBOS, which is checked too. When a name changes, the failure prints the new list: update the case, with a compatibility note if a recorded name changed.
  - `DURABLE_DBOS`: `tests/harness/test_durable_tool_io.py` runs the calls in a DBOS workflow, forks it from after its last step, as recovery does, and checks that no request the stand-ins count is made again.
  - `DURABLE_TEMPORAL`: `tests/harness/test_durable_temporal_replay.py` runs them in a Temporal workflow, on one dev server per module, and replays the recorded history with a `Replayer`, which must succeed without a request. An agent Temporal refuses breaks it, so an id-less toolset is a `refused` cell, with its issue.
- `ENGINE_SUPPORT` (in `tests/capability_cases.py`) is the engine x case table: what each engine in `ENGINES` does with each case's calls, as a `Support` cell. A cell it doesn't list is `supported` under the engines a suite runs every case under (`SUITE_ENGINES`: DBOS and Temporal) and `untested` under the others. `Support` is shared, so W1 can key the same type by workspace backend.
  - `supported`: the engine's suite runs the calls and checks what the engine records.
  - `limited`: the same, and `reason` names the limit, which the suite must also assert.
  - `refused`: `raises` is a regex of what the engine raises, and the suite fails once the engine stops raising it.
  - `broken`: the engine runs the calls and gets them wrong. It carries the `issue` that tracks the bug, and the suite fails once the bug no longer reproduces.
  - `untested`: no suite runs the calls under the engine yet; it carries the `issue` that tracks adding one. `test_measured_cells_only_go_up` pins how many cells each engine measures, the ones that aren't `untested`: a PR may raise that count by testing a cell, never lower it. It pins the number of cases with calls too, so a new case that no suite measures changes a snapshot.
  - A cell is the only place a case's support is written down: `DURABLE_DBOS` and `DURABLE_TEMPORAL` exempt exactly the cases their engine's column marks `refused` or `broken`, and `test_every_durability_capability_is_an_engine` fails until a new engine's durability capability is in `ENGINE_CAPABILITIES`.
- `SCENARIO_SUPPORT` (also in `tests/capability_cases.py`) holds scenarios a capability, or `RunContext`, must handle under every engine beyond D1's one-call-per-response run, keyed `<owner>.<scenario>`, such as `BackgroundTools.survives_approval_wait`. Every scenario has a `Support` cell for every engine. A `broken` cell's issue has a regex in `BROKEN_SIGNATURES` that the observed outcome must still match. Scenarios never feed D1's exemptions.
  - A capability's own tests run its scenarios, under every engine.
  - The D1 suites run the scenarios `SCENARIO_RUNS` scripts: `ctx.emit` and `ctx.enqueue` from a function tool, from inside a durable unit, and from a hook. A `ScenarioRecorder` notes the custom and capability events and enqueued messages the run observes, in workflow code. The run must observe each scenario's list once, and its recovery (DBOS) or replay (Temporal) must observe the same, with nothing lost or added. The recorder makes the run stream, which changes the model request's unit name, so cases don't get one.
- A case that can't run under an engine is exempt from that engine's contract, with the reason: an exemption, never a skip. As in every contract, a case whose optional dependency (`requires`) isn't installed is skipped: `BrowserUse`'s is installed only by its own CI job, which doesn't run these suites yet.
- Prefect and the Harness engines are next; until then, `test_prefect_wraps_every_toolset_temporal_does` checks Prefect accepts every case Temporal runs.

## Workspaces and sandboxes

To be written. Planned scope: the workspace authority table is generated from `CASES` (every case that reads workspace files) rather than listed by hand, and every `WorkspaceBackend` class runs `WorkspaceBackendSuite`.
