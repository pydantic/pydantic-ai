# Capability Conformance

> What every capability, in Pydantic AI core and Pydantic AI Harness, must satisfy before it ships

**When to check**: Whenever you add a capability, change what an existing one contributes (tools, tool hooks, instructions, parts it enqueues, a workspace), or review a PR that does

A capability is conformant when its entry in `CASES` (`tests/capability_cases.py` for core, `tests/harness/capability_cases.py` for Pydantic AI Harness) passes every contract of each area below, or carries an exemption with a reason a reviewer can check. `test_every_capability_has_a_case` fails until a new capability has an entry, so every area reaches it without anyone listing it. A box is ticked by a passing contract or a reasoned `Exempt`, never by prose: a PR description or a comment saying a capability "handles" an area doesn't count.

Each area says what it requires and how the suite checks it. Contracts are `Contract(applies, exempt)` entries in `CONTRACTS`, run by `tests/harness/test_capability_contracts.py`; an exempt case must still break its contract, so fixing the capability fails the suite until the exemption is removed. Link an exemption that records a bug to the issue that tracks it.

The contracts are numbered C0, C1, … as this checklist assigns them, not in the order they land. C1–C5 work toward [#13118](https://github.com/pydantic/pydantic-ai/issues/13118): no capability content below `'operator'` reaches the model in system voice.

## Coverage

**Requirements**

- A capability that contributes a toolset, a wrapper toolset or a tool hook (`before_tool_validate` through `on_tool_execute_error`) declares `calls`: tool calls that run its tools offline. Every contract that runs a capability's tools applies only to a case with `calls`, so without them those contracts pass vacuously.
- A capability whose tools can't run offline carries an exemption saying why, such as `'needs a sandbox backend: …'`. A case that could have `calls` but doesn't yet is exempt with a reason that starts with `PENDING_CALLS`.

**How it's tested**

- `DECLARES_CALLS` (C0) builds every buildable public case and checks it: `test_a_capability_that_runs_with_tools_declares_calls`. A capability made of others is checked through its parts.
- `test_pending_calls_backfill` pins how many cases are still pending. A PR that writes a case's `calls`, or gives a case a reasoned exemption of its own (such as a route other tools run through, which C1 covers), removes its pending exemption and lowers the count; no PR raises it, so a new capability doesn't add a pending one.
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
- Text a capability reads from a source below `'operator'` (a workspace file, a skill, an MCP server's instructions or tool descriptions, a model-written sub-agent definition) never reaches the instructions, a system or user part, or a tool definition: only tool output, or a framed data part naming its source.
- Authority only goes down: a part reaches the model with the lower of its source's and its kind's authority, never raised.
- The attribution is the same on every route the content takes: a native tool call, Code Mode, IPython Code Mode, `ToolBridge`, `BackgroundTools`, a sub-agent, a durable engine, and a round trip through `ModelMessagesTypeAdapter` or a UI protocol.

The rules behind these are in [Message History & Prompt Caching](message-history-and-caching.md), "Trust tiers map to channels, and nothing is ever promoted".

**How it's tested**

- C0 (above) makes these contracts reach every capability that runs tools.
- C1, route matrix: `test_calls_reach_the_model_the_same_on_every_route` in `tests/harness/test_capability_routes.py` (with its IPython twin) runs each case's `calls` at `with_authority('none')` natively and through Code Mode, IPython Code Mode, `ToolBridge`, `BackgroundTools`, a sub-agent, a prepare hook that rebuilds each tool definition, a `ModelMessagesTypeAdapter` round trip, and a Vercel AI and an AG-UI run that keep the history on the server. The same tools must run, and every part the calls add besides each call's own result must match the native route's type, source, capability id, authority and frame. `PROBES` there add calls that return `ToolReturn.content` and call `ctx.enqueue`, so every route has a part to compare. A route that breaks this for a case is in `KNOWN`, a strict `xfail` linked to its fix; one that doesn't apply is skipped with its reason. The durable routes (Temporal, DBOS) are D1's. To add a route, add it to `Route` and give `_through` the way to run calls on it.
- C2, sink check (planned): `Model.prepare_messages` asserts no part below `'operator'` renders as system or developer, and no capability below `'operator'` authors a `ModelResponse`.
- C3, role-site guard (planned): every site that builds a system or developer part is in an allowlist with its reason.
- C4, transformation registry: `tests/harness/test_history_transformations.py` runs each case's `calls` (and C1's `PROBES`) natively, hands the history to each transformation in `TRANSFORMS` (a `ModelMessagesTypeAdapter` round trip, `sanitize_messages`, a Vercel AI and an AG-UI `dump_messages`/`load_messages` round trip, the history a resumed sub-agent starts from, and the run-level ones), and checks by each request part's type, source, authority and frame that every part it doesn't drop survives unchanged, that a part it creates has no more authority than what it replaces, and that nothing else appears. A finding is a strict `xfail` in the transform's `exempt`, linked to its issue. `test_every_history_function_is_registered` fails until a new function that returns `list[ModelMessage]` is registered on a `Transform` or exempt in `TRANSFORM_EXEMPT` with its reason.
- C5, attribution on spans (planned): OpenTelemetry message parts keep each capability part's source.
- C6, instruction sources: `test_instruction_source_authority.py`'s canary test puts a marker in each instruction source (`Skills`, `RepoContext`'s working-directory, ancestor and nested files, `Memory`, MCP server instructions and tool descriptions, `SubAgents` `create_agent` instructions) at `'none'` and at `'operator'`, and checks the marker appears only in tool output or framed data at `'none'` and where it does today at `'operator'`. A source that breaks this is a strict xfail linked to its fix (MCP: [#13117](https://github.com/pydantic/pydantic-ai/issues/13117); `SubAgents` `create_agent`: [#12433](https://github.com/pydantic/pydantic-ai/issues/12433)).

## Durable execution

To be written. Planned scope: DBOS first, each case's `calls` re-executed from the last recorded step; then Temporal (the toolset id present, a `Replayer` run equal), then Prefect; the journaled steps recorded next to each case (`Calls.durable_steps` per engine).

## Workspaces and sandboxes

To be written. Planned scope: the workspace authority table is generated from `CASES` (every case that reads workspace files) rather than listed by hand, and every `WorkspaceBackend` class runs `WorkspaceBackendSuite`.
