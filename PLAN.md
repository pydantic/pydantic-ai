# AG-UI Interrupts ↔ Pydantic AI DeferredTools

Tracks issue [#2889](https://github.com/pydantic/pydantic-ai/issues/2889). Branch already exists: `feat-2889-tool-approval-agui`.

## Context

AG-UI just shipped a new **interrupts** lifecycle (AG-UI [PR #1569](https://github.com/ag-ui-protocol/ag-ui/pull/1569), merged 2026-04-30; spec at [docs.ag-ui.com/concepts/interrupts](https://docs.ag-ui.com/concepts/interrupts)). It formalizes a pause/resume protocol on top of `RUN_FINISHED` — the agent can end a run with `outcome.type = "interrupt"` and a list of `Interrupt` objects, and the next `RunAgentInput` carries a `resume[]` array to address each one.

Pydantic AI already has the underlying primitive on the agent side: a tool can declare `requires_approval=True`, and when the model proposes such a call the agent run pauses and returns `DeferredToolRequests(approvals=[ToolCallPart, …])` as its output. The caller then resumes the same run via `agent.iter(deferred_tool_results=DeferredToolResults(approvals={tc_id: ToolApproved(...) | ToolDenied(...)}))`.

What is missing today is the **AG-UI ⇄ DeferredTools translation in the adapter**. Currently [`AGUIEventStream.after_stream`](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py#L120-L126) emits a bare `RunFinishedEvent` with no `outcome`, so a frontend has no signal that the run is awaiting human approval. And `RunAgentInput.resume` is never read, so frontends have no way to feed an approval/denial back through the protocol.

**Goal:** translate `DeferredToolRequests` ↔ AG-UI interrupts in both directions, behind a version gate, with no break for older `ag-ui-protocol` installs.

## Scope (v1)

Per user decision:

- **Approvals only.** Map only `DeferredToolRequests.approvals` to AG-UI interrupts (`reason: "tool_call"`). Leave the external `DeferredToolRequests.calls` path (frontend toolset execution) unchanged — that flow already works via AG-UI's `ToolMessage` round-trip in the message history.
- **`reason: "input_required"` and `reason: "confirmation"` are out of scope for v1.** They have no source in Pydantic AI's `DeferredTools` today and would expand public-API surface without a concrete use case.
- **Silent skip on old SDKs.** If the installed `ag-ui-protocol` is older than the version that ships `RunFinishedInterruptOutcome` / `Interrupt` / `ResumeEntry`, the adapter behaves exactly as today (bare `RunFinishedEvent`, no outcome). This matches the existing version-gate philosophy in [pydantic_ai_slim/pydantic_ai/ui/CLAUDE.md](pydantic_ai_slim/pydantic_ai/ui/CLAUDE.md).

## Spec ↔ Pydantic AI mapping (v1)

| AG-UI field (Python) | Pydantic AI source |
| --- | --- |
| `RunFinishedEvent.outcome.type == "interrupt"` | `result.output` is a `DeferredToolRequests` with non-empty `approvals` |
| `Interrupt.id` | Deterministic, e.g. `f"int-{tool_call_id}"` (stable across redelivery) |
| `Interrupt.reason` | Literal `"tool_call"` |
| `Interrupt.tool_call_id` | `ToolCallPart.tool_call_id` from the approval entry |
| `Interrupt.message` | `f"Approve {tool_name}({args})?"` (overridable later via metadata) |
| `Interrupt.response_schema` | `{"type": "object", "properties": {"approved": {"type": "boolean"}, "editedArgs": <tool args schema>}, "required": ["approved"]}` |
| `Interrupt.metadata` | `DeferredToolRequests.metadata.get(tool_call_id)` |
| `Interrupt.expires_at` | Not set in v1 (spec field, no source today) |
| `ResumeEntry.payload.approved == true` | `DeferredToolResults.approvals[tc_id] = ToolApproved()` |
| `ResumeEntry.payload.approved == true` + `editedArgs` | `ToolApproved(override_args=edited_args)` |
| `ResumeEntry.payload.approved == false` | `ToolDenied(message=payload.get("reason", "The tool call was denied."))` |
| `ResumeEntry.status == "cancelled"` | `ToolDenied(message="Cancelled by user.")` (regardless of payload) |

**Spec corrections to the original mapping notes:**

- `ResumeEntry.status` is exactly `Literal["resolved", "cancelled"]` — denials live in `payload.approved == false`, not in `status`.
- `Interrupt` also has an optional `expires_at` field — captured above (not set in v1, but the field exists).
- The spec is explicit (quoted from [docs.ag-ui.com/concepts/interrupts](https://docs.ag-ui.com/concepts/interrupts)): on resume, the agent **must not** re-emit `TOOL_CALL_START`/`ARGS`/`END` — only `TOOL_CALL_RESULT` against the original `tool_call_id`. This is satisfied automatically by Pydantic AI today: the agent loops back into tool execution with the same `tool_call_id`, and [`AGUIEventStream._handle_tool_result`](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py#L251-L263) emits a `ToolCallResultEvent` keyed by that id with no preceding start event. No change needed here.

## Implementation

### 1. New version gate

[pydantic_ai_slim/pydantic_ai/ui/ag_ui/_utils.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_utils.py)

Add a new constant alongside `REASONING_VERSION` and `MULTIMODAL_VERSION`:

```python
INTERRUPTS_VERSION = (0, 1, 19)
"""AG-UI version that introduced run-interrupt outcomes and resume entries (ag-ui-protocol 0.1.19, shipping PR #1569)."""
```

Sanity check on the version: AG-UI [PR #1569](https://github.com/ag-ui-protocol/ag-ui/pull/1569) merged 2026-04-30 (commit `df613e4`) and its own description names `0.1.19` as the follow-up release target. 0.1.18 was cut 2026-04-21 and predates the merge, so it does **not** contain interrupts. Re-verify against PyPI when opening the PR — if 0.1.19 hasn't been published yet, we either (a) wait for it, or (b) merge with the gate already in place and the feature dormant until the SDK release lands.

### 2. Try-imports for new types

In [pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py) and [_adapter.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py), add a guarded import block following the existing `ReasoningMessage` stub pattern (see `_adapter.py:99-120`):

```python
try:
    from ag_ui.core import (
        Interrupt,
        ResumeEntry,
        RunFinishedInterruptOutcome,
        RunFinishedSuccessOutcome,
    )
    _HAS_INTERRUPTS = True
except ImportError:
    _HAS_INTERRUPTS = False
    # Stubs so the production code below typechecks under pyright when the import fails.
    # Field shapes must match the merged main schema verbatim (see PR #1569).
```

Behavior: when `_HAS_INTERRUPTS is False`, the outbound encoder skips outcome emission and the inbound translator never inspects `run_input.resume` (older `RunAgentInput` may not even have the field).

### 2a. Developing against the unreleased ag-ui-protocol

Because `ag-ui-protocol 0.1.19` is not yet on PyPI, no contributor has the new types available from the standard install. Two practical steps:

- **Local development:** install the unreleased SDK from the merged-main commit (`df613e4`, PR #1569) into your venv so the new symbols actually resolve and the new tests can run:

    ```bash
    uv pip install --reinstall \
      'ag-ui-protocol @ git+https://github.com/ag-ui-protocol/ag-ui.git@df613e4#subdirectory=sdks/python'
    ```

  This affects only the local environment — `pydantic_ai_slim/pyproject.toml`'s `ag-ui = ["ag-ui-protocol>=0.1.10", ...]` pin stays put.

- **CI behavior until 0.1.19 publishes:** the default CI install will resolve to `ag-ui-protocol 0.1.18` (the latest released), which does **not** have the new types. The new test cases must be guarded with `pytest.mark.skipif(not _HAS_INTERRUPTS, ...)` so they skip cleanly on CI. The "silent skip on old SDK" tests still run and prove backward compatibility. Once 0.1.19 hits PyPI, CI picks it up automatically (no PR change needed) and the gated tests light up.

- **Optional follow-up (don't gate v1 on this):** if maintainers want CI coverage of the new path before 0.1.19 ships, add a CI matrix entry that installs `ag-ui-protocol` from the git commit above. Flag this in the PR description but don't implement unilaterally — it's the maintainers' call.

### 3. Outbound: emit `RunFinishedInterruptOutcome` when the run ends on approvals

[pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py)

Modify `after_stream` (current implementation at lines 120-126):

```python
async def after_stream(self) -> AsyncIterator[BaseEvent]:
    if self._error:
        return
    outcome = self._build_outcome() if _HAS_INTERRUPTS else None
    yield RunFinishedEvent(
        thread_id=self.run_input.thread_id,
        run_id=self.run_input.run_id,
        outcome=outcome,
        timestamp=self._get_timestamp(),
    )

def _build_outcome(self) -> RunFinishedInterruptOutcome | RunFinishedSuccessOutcome | None:
    if parse_ag_ui_version(self.ag_ui_version) < INTERRUPTS_VERSION:
        return None  # older client doesn't understand outcomes
    output = getattr(self._result, "output", None) if self._result else None
    if isinstance(output, DeferredToolRequests) and output.approvals:
        return RunFinishedInterruptOutcome(
            type="interrupt",
            interrupts=[_approval_to_interrupt(call, output.metadata) for call in output.approvals],
        )
    return RunFinishedSuccessOutcome(type="success")
```

`_approval_to_interrupt(call: ToolCallPart, metadata)` builds the `Interrupt` per the mapping table. To derive `response_schema`, the toolset already knows each tool's args schema — accessible via `self.run_input.tools` (AG-UI frontend tools) and the agent's own toolsets. For v1, the simplest correct shape is the table above with `editedArgs` as `Any`; deriving a precise schema from the originating tool is a follow-up.

`_result` is already set by the base `UIEventStream` at [_event_stream.py:201](pydantic_ai_slim/pydantic_ai/ui/_event_stream.py#L201) so no extra plumbing is needed.

### 4. Inbound: translate `run_input.resume[]` into `DeferredToolResults`

[pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py)

Override the existing `deferred_tool_results` property (base class hook at [_adapter.py:246](pydantic_ai_slim/pydantic_ai/ui/_adapter.py#L246)) on `AGUIAdapter`:

```python
@cached_property
def deferred_tool_results(self) -> DeferredToolResults | None:
    if not _HAS_INTERRUPTS:
        return None
    resume = getattr(self.run_input, "resume", None)
    if not resume:
        return None
    approvals: dict[str, DeferredToolApprovalResult] = {}
    for entry in resume:
        tool_call_id = self._interrupt_id_to_tool_call_id(entry.interrupt_id)
        approvals[tool_call_id] = self._resume_entry_to_approval(entry)
    return DeferredToolResults(approvals=approvals)
```

Helpers:

- `_interrupt_id_to_tool_call_id` reverses the `f"int-{tool_call_id}"` convention. If the convention is changed (e.g. metadata-driven), keep this round-trip private to the adapter.
- `_resume_entry_to_approval` returns `ToolDenied("Cancelled by user.")` when `status == "cancelled"`; otherwise inspects `payload.approved` (`bool`) and `payload.get("editedArgs")` to build `ToolApproved(override_args=…)` / `ToolDenied(payload.get("reason", …))`.

The `deferred_tool_results` property already flows through `run_stream_native` at [_adapter.py:506-507](pydantic_ai_slim/pydantic_ai/ui/_adapter.py#L506-L507), so no other plumbing is required.

### 5. Public re-exports

[pydantic_ai_slim/pydantic_ai/ui/ag_ui/__init__.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/__init__.py)

No new Pydantic AI types to export — interrupts live in `ag_ui.core`, and the existing public `DeferredToolRequests` / `DeferredToolResults` / `ToolApproved` / `ToolDenied` from `pydantic_ai` are unchanged.

## Tests

[tests/test_ag_ui.py](tests/test_ag_ui.py)

Mirror the existing pattern (TestModel + inline snapshots, no VCR). Reference: `test_client_submitted_tool_call_resolved_by_deferred_results_runs` already exercises the resume-with-`DeferredToolResults` path; the new tests extend the same fixture pattern.

New tests:

1. **Outbound: success outcome.** A run that ends normally emits `RunFinishedEvent` with `outcome.type == "success"` on modern SDK. Snapshot the encoded event.
2. **Outbound: interrupt outcome.** An agent with a `requires_approval=True` tool returns `DeferredToolRequests(approvals=[…])`. Assert the final `RunFinishedEvent.outcome` is `RunFinishedInterruptOutcome` with one `Interrupt` per approval, correct `tool_call_id`, `reason == "tool_call"`, and `response_schema` shape.
3. **Outbound: old SDK silent-skip.** Force `ag_ui_version="0.1.10"` on the adapter and assert no `outcome` field is set (matches today's behavior).
4. **Inbound: approve.** `RunAgentInput.resume = [{interrupt_id, status: "resolved", payload: {approved: true}}]` → adapter's `deferred_tool_results` yields `{tool_call_id: ToolApproved()}`.
5. **Inbound: approve with edits.** `payload: {approved: true, editedArgs: {...}}` → `ToolApproved(override_args={...})`.
6. **Inbound: deny.** `payload: {approved: false, reason: "no"}` → `ToolDenied("no")`.
7. **Inbound: cancel.** `status: "cancelled"` → `ToolDenied("Cancelled by user.")` regardless of payload.
8. **End-to-end roundtrip.** Two `run_stream` invocations against the same `AGUIAdapter` lineage: turn 1 returns `DeferredToolRequests`, turn 2 (with `resume[]`) executes the approved tool and emits `TOOL_CALL_RESULT` against the original `tool_call_id` (and **does not** emit a fresh `TOOL_CALL_START` — assert via snapshot).

All tests must pass on both old and new `ag-ui-protocol` versions where applicable; gate version-specific tests with `pytest.mark.skipif(parse_ag_ui_version(DEFAULT_AG_UI_VERSION) < INTERRUPTS_VERSION, ...)`.

## Example

[examples/pydantic_ai_examples/ag_ui/api/](examples/pydantic_ai_examples/ag_ui/api/)

Add a new file, e.g. `tool_approval.py`, mirroring the minimal pattern of [human_in_the_loop.py](examples/pydantic_ai_examples/ag_ui/api/human_in_the_loop.py):

```python
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent("openai:gpt-5-mini")

@agent.tool(requires_approval=True)
async def delete_file(path: str) -> str:
    """Delete a file from the project. Requires user approval."""
    ...  # actual deletion

async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(request, agent=agent)

app = Starlette(routes=[Route("/", run_agent, methods=["POST"])])
```

A short docstring at the top of the file should explain the AG-UI interrupt flow: first POST proposes the call and ends with an `interrupt` outcome; second POST includes `resume[]` to approve, deny, or edit the call.

## Docs

[docs/ui/ag-ui.md](docs/ui/ag-ui.md)

Add a new section "Human-in-the-loop with interrupts" after the existing usage patterns. Cover:

- When the run pauses (any tool declared with `requires_approval=True`)
- The shape of the outgoing `RUN_FINISHED` `outcome.interrupts[]`
- How the client should populate `resume[]` on the next `RunAgentInput`
- Version requirement (interrupts ship in `ag-ui-protocol >= 0.1.19`; older versions silently fall back to no outcome)
- Cross-link to [docs/deferred-tools.md](docs/deferred-tools.md) for the underlying agent-side primitive

Keep it under ~60 lines and reuse code from the new example file.

## Version gating & backward compatibility

Per [pydantic_ai_slim/pydantic_ai/ui/CLAUDE.md](pydantic_ai_slim/pydantic_ai/ui/CLAUDE.md):

- `ag-ui-protocol` minimum stays at `>=0.1.10` — no version bump.
- New code is gated on `_HAS_INTERRUPTS` (import-time check) **and** `parse_ag_ui_version(self.ag_ui_version) >= INTERRUPTS_VERSION` (negotiated-version check, so a server running a new SDK can still talk old protocol to an old client).
- On the old path, behavior is byte-for-byte identical to today: bare `RunFinishedEvent`, no `outcome`, `resume[]` ignored even if a client sends it.

## Verification

1. **Unit tests:** `make test` — all new tests in `tests/test_ag_ui.py` pass.
2. **Type check:** `make typecheck` — no new `cast` / `Any` introduced; `DeferredToolApprovalResult` is fully typed.
3. **Lint:** `make lint && make format`.
4. **Docs examples:** `tests/test_examples.py` continues to pass (any code blocks added to `docs/ui/ag-ui.md` are executed).
5. **Manual end-to-end:** run the new `tool_approval.py` example via `python -m pydantic_ai_examples.ag_ui` (per [examples/pydantic_ai_examples/ag_ui/__main__.py](examples/pydantic_ai_examples/ag_ui/__main__.py)) and exercise it with a `curl` POST that mimics an AG-UI client posting `resume[]`. Confirm:
   - Turn 1 response stream contains a `RUN_FINISHED` event with `outcome.type == "interrupt"`.
   - Turn 2 (with `resume: [{interrupt_id, status: "resolved", payload: {approved: true}}]`) executes the tool and emits a single `TOOL_CALL_RESULT` against the same `tool_call_id` — no second `TOOL_CALL_START`.
6. **Old-SDK regression:** install `ag-ui-protocol==0.1.10` in a fresh venv and rerun the AG-UI test suite; confirm everything still passes and no `outcome` is emitted.

## Out of scope (follow-ups)

- Mapping `DeferredToolRequests.calls` (external/frontend execution) to interrupts.
- `reason: "input_required"` and `reason: "confirmation"` — no Pydantic AI source today.
- Precise per-tool `response_schema` for `editedArgs` (will use `Any` in v1; can derive from `ToolDefinition.parameters_json_schema` in a follow-up).
- `Interrupt.expires_at` enforcement.
- Multi-interrupt parallel-tool-call resume ordering (AG-UI [PR #1493](https://github.com/ag-ui-protocol/ag-ui/pull/1493) WIP) — Pydantic AI already handles multiple approvals per run via the `approvals` list; just need to confirm ordering is preserved.

## Critical files to modify

- [pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py) — outbound `outcome` in `after_stream`.
- [pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py) — inbound `deferred_tool_results` property override.
- [pydantic_ai_slim/pydantic_ai/ui/ag_ui/_utils.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_utils.py) — `INTERRUPTS_VERSION` constant.
- [tests/test_ag_ui.py](tests/test_ag_ui.py) — new tests (8 cases above).
- [examples/pydantic_ai_examples/ag_ui/api/tool_approval.py](examples/pydantic_ai_examples/ag_ui/api/) — new example file.
- [docs/ui/ag-ui.md](docs/ui/ag-ui.md) — new "Human-in-the-loop with interrupts" section.
