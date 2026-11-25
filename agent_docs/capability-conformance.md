# Capability Conformance

> What every capability, in Pydantic AI core and Pydantic AI Harness, must satisfy before it ships

**When to check**: Whenever you add a capability, change what an existing one contributes (tools, tool hooks, instructions, parts it enqueues, a workspace), or review a PR that does

A capability is conformant when its entry in `CASES` (`tests/capability_cases.py` for core, `tests/harness/capability_cases.py` for Pydantic AI Harness) passes every contract of each area below, or carries an exemption with a reason a reviewer can check. `test_every_capability_has_a_case` fails until a new capability has an entry, so every area reaches it without anyone listing it. A box is ticked by a passing contract or a reasoned `Exempt`, never by prose: a PR description or a comment saying a capability "handles" an area doesn't count.

Each area says what it requires and how the suite checks it. Contracts are `Contract(applies, exempt)` entries in `CONTRACTS`, run by `tests/harness/test_capability_contracts.py`; an exempt case must still break its contract, so fixing the capability fails the suite until the exemption is removed. Link an exemption that records a bug to the issue that tracks it.

The contracts are numbered C0, C1, … in the order they land. C1–C5 are planned and work toward [#13118](https://github.com/pydantic/pydantic-ai/issues/13118): no capability content below `'operator'` reaches the model in system voice.

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
- `DEFAULT_ID_MATCHES_COMBINE` checks every case's class, the unbuildable and private ones included: under `Combines` or `Narrows` it declares a default `id`, under `Anonymous`, `Collides` or `Rejected` none, and a new policy fails until it is classified: `test_a_default_id_matches_the_combine_policy`.
- `tests/test_capability_combine.py` and its Harness twin check that each case combines the way its `combine` policy says; `COMBINE_FIXED_POINT` checks that a configuration stated twice is stated once.

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

To be written. Planned scope: DBOS first, each case's `calls` re-executed from the last recorded step; then Temporal (the toolset id present, a `Replayer` run equal), then Prefect; the journaled steps recorded next to each case (`Calls.durable_steps` per engine).

## Workspaces and sandboxes

To be written. Planned scope: the workspace authority table is generated from `CASES` (every case that reads workspace files) rather than listed by hand, and every `WorkspaceBackend` class runs `WorkspaceBackendSuite`.
