# AG-UI Interrupts ↔ Pydantic AI DeferredTools

Tracks issue [#2889](https://github.com/pydantic/pydantic-ai/issues/2889). Branch: `feat-2889-tool-approval-agui`.

## Status

Implementation complete on this branch. Diff: **7 files modified, 1 added; +568 / -6**.

| Plan step                                                       | Status | Notes                                                                                                                                                                                                                                                                                                |
| --------------------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. `INTERRUPTS_VERSION = (0, 1, 19)` in `_utils.py`            | done   | Verified against the merged-main commit `df613e4`; PyPI release 0.1.19 not yet cut at time of writing.                                                                                                                                                                                              |
| 2. Try-imports + stubs for `Interrupt`/`ResumeEntry`/outcomes  | done   | Followed the existing `ReasoningMessage`/`AudioInputContent` `TYPE_CHECKING` + runtime-stub pattern. Single `_HAS_INTERRUPTS` flag in each module.                                                                                                                                                  |
| 2a. Dev install of unreleased ag-ui-protocol                    | done   | Solved by `[tool.uv.sources]` in root `pyproject.toml` pinning to commit `df613e4` + regenerated `uv.lock`. Public `>=0.1.10` floor in `pydantic_ai_slim/pyproject.toml` stays put so PyPI users still get the released SDK. **Remove the entry once `ag-ui-protocol 0.1.19` publishes.**                                                  |
| 3. Outbound `_build_outcome` + `after_stream`                  | done   | `RunFinishedEvent` construction branches on `_HAS_INTERRUPTS` so old SDKs never see an `outcome=None` kwarg (`ConfiguredBaseModel.extra="forbid"`).                                                                                                                                                  |
| 4. Inbound `deferred_tool_results` override + helpers           | done   | Added `_interrupt_id_to_tool_call_id`, `_resume_entry_to_approval`, `_payload_dict` — the last one was an extra needed to keep pyright happy when narrowing AG-UI's `Optional[Any]` payload to a typed `dict[str, Any]`.                                                                              |
| 5. Public re-exports                                            | n/a    | Confirmed unchanged — interrupts live in `ag_ui.core`; existing `pydantic_ai` exports already cover `DeferredToolRequests`/`DeferredToolResults`/`ToolApproved`/`ToolDenied`.                                                                                                                          |
| Tests (planned 8 cases)                                         | done   | Landed **9** cases: added `test_resume_unknown_interrupt_id_prefix_raises` to lock in the `UserError` on a malformed interrupt id. All guarded with `@pytestmark_interrupts` (`skipif` on `try_import` of `ResumeEntry`).                                                                              |
| Example                                                         | done   | [`examples/pydantic_ai_examples/ag_ui/api/tool_approval.py`](examples/pydantic_ai_examples/ag_ui/api/tool_approval.py), wired into the dojo app at `/tool_approval`.                                                                                                                                  |
| Docs                                                            | done   | New "Tool approval (interrupts)" section in [`docs/ui/ag-ui.md`](docs/ui/ag-ui.md) between `### Tools` and `### Events`, with the mapping table and `!!! note` admonition for the version requirement.                                                                                                |
| Lint / typecheck / tests                                        | done   | `ruff check` ✓, `ruff format` ✓, `pyright` ✓ (0 errors on changed files), `pytest tests/test_ag_ui.py` ✓ 115/115, broader sanity `pytest test_capabilities + test_ui + test_ag_ui + test_tools` ✓ 895/895, `pytest tests/test_examples.py -k ag_ui` ✓ 10/0 (new doc snippet picked up). |

### Deviations / findings during implementation

- **`output_type` widening is conditional on a frontend toolset.** [_adapter.py:516](pydantic_ai_slim/pydantic_ai/ui/_adapter.py#L516) only widens to include `DeferredToolRequests` when `self.toolset` is truthy. For an agent that uses only server-side `@agent.tool_plain(requires_approval=True)` (the natural shape for human-in-the-loop on the server), the caller must declare `output_type=[str, DeferredToolRequests]` explicitly. The example does this; the docs example does too; the docs section calls it out so users hit it once and not twice. Possible follow-up: always widen `output_type` in the adapter regardless of toolset presence, since deferred approvals are a first-class adapter concern — kept out of scope here.
- **`RunFinishedEvent.outcome` field absence on old SDKs.** Passing `outcome=None` to the old (`<0.1.19`) `RunFinishedEvent` triggers `extra="forbid"`, so the call sites split into two literal `RunFinishedEvent(...)` constructions inside `after_stream` rather than building kwargs conditionally. Worth keeping — readable and grep-able.
- **`Interrupt.id` ↔ `tool_call_id` round-trip.** Settled on a deterministic `f"int-{tool_call_id}"` convention exposed as `INTERRUPT_ID_PREFIX` in `_event_stream.py`. The inbound helper raises `UserError` on a malformed prefix rather than silently mapping to a wrong tool call (this surfaces protocol drift early). A new test pins the behavior.
- **Pyright narrowing on `entry.payload`.** AG-UI types `ResumeEntry.payload` as `Optional[Any]`. Pyright's `isinstance(x, Mapping)` narrow ignores explicit type annotations and casts to `Mapping[Unknown, Unknown]`, which then poisons downstream `.get(...)` calls. Refactored into a tiny `_payload_dict(raw: Any) -> dict[str, Any]` helper that uses `isinstance(raw, dict)` + `cast` and returns a cleanly-typed dict. Trades one micro-helper for zero pyright noise — net positive.
- **uv source pin closes the CI coverage gap.** Original plan flagged that the 9 gated tests would skip on CI until 0.1.19 published. The `[tool.uv.sources]` entry on `ag-ui-protocol` resolves this — the lockfile points at commit `df613e4`, so `make install` (which does `uv sync --frozen --all-extras --all-packages`) gets the merged-main snapshot, pyright resolves the new types, and all 9 gated tests run on CI. The pin is dev-only (does not affect PyPI consumers). Must be removed when 0.1.19 publishes — see "Cleanup required" in section 2a.

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

`ag-ui-protocol 0.1.19` (which ships PR #1569) is not yet on PyPI. The branch handles this with a temporary `[tool.uv.sources]` entry in the **root** `pyproject.toml`:

```toml
[tool.uv.sources]
# ... existing workspace entries ...
ag-ui-protocol = { git = "https://github.com/ag-ui-protocol/ag-ui.git", rev = "df613e40b857668be9a8986aa7ee388fe65aee31", subdirectory = "sdks/python" }
```

`uv sync` resolves dev/test/CI installs against this git revision (recorded in `uv.lock` as `source = { git = "..." }`), while the public **floor** in `pydantic_ai_slim/pyproject.toml` (`ag-ui = ["ag-ui-protocol>=0.1.10", ...]`) is untouched — so external PyPI users continue to get the latest released SDK on `pip install pydantic-ai-slim[ag-ui]`.

**Cleanup required when 0.1.19 publishes:**

1. Remove the `ag-ui-protocol = { git = ... }` line from root `pyproject.toml` `[tool.uv.sources]`.
2. Run `uv sync` to refresh `uv.lock` against PyPI.
3. (Optional but recommended) Bump the floor in `pydantic_ai_slim/pyproject.toml` to `>=0.1.19` and drop the `try_import` / `_HAS_INTERRUPTS` gate, simplifying the adapter — but only if the maintainers approve raising the minimum.

**CI behavior with this branch as-is:** CI runs `make install` which uses `uv sync --frozen --all-extras ...`. The lockfile pins to the git revision, so CI installs the merged-main snapshot; `pyright` resolves the new types; all 9 new gated tests run. No CI coverage gap.

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

Tests landed (9 cases, all decorated with `@pytestmark_interrupts` — a module-level `pytest.mark.skipif` keyed off `try_import` of `ResumeEntry` from `ag_ui.core`, so they skip cleanly on old SDKs):

1. **`test_run_finished_success_outcome_on_modern_version`** — a normal run emits `outcome.type == "success"` on a modern SDK.
2. **`test_run_finished_no_outcome_on_legacy_version`** — forcing `ag_ui_version="0.1.10"` yields a bare `RUN_FINISHED` with no `outcome` key (proves silent skip).
3. **`test_run_finished_interrupt_outcome_for_pending_approval`** — an agent with a `requires_approval=True` tool returns `DeferredToolRequests`; assert `outcome.type == "interrupt"` with one `Interrupt` carrying `reason == "tool_call"`, the original `tool_call_id`, and the documented `response_schema` shape. Also asserts the `id` round-trips back to `tool_call_id`.
4. **`test_resume_resolved_approves_tool`** — `{status: "resolved", payload: {approved: true}}` → `ToolApproved()`.
5. **`test_resume_resolved_with_edited_args_passes_override_args`** — `payload.editedArgs` → `ToolApproved(override_args=...)`.
6. **`test_resume_resolved_with_approved_false_denies_tool`** — `{approved: false, reason: "..."}` → `ToolDenied("...")`.
7. **`test_resume_cancelled_denies_tool_regardless_of_payload`** — `status: "cancelled"` overrides payload contents.
8. **`test_resume_unknown_interrupt_id_prefix_raises`** *(extra, not in original plan)* — an `interrupt_id` missing the `int-` prefix raises `UserError` rather than silently mapping to the wrong tool call.
9. **`test_interrupt_resume_roundtrip_executes_approved_tool`** — end-to-end: turn 1 produces the interrupt outcome; turn 2 supplies `resume[]` and the tool actually runs. Also asserts the spec rule that the resumed turn **does not** re-emit `TOOL_CALL_START` for the same `tool_call_id` — only `TOOL_CALL_RESULT`.

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

Local results on this branch (lockfile pinned to `ag-ui-protocol @ git@df613e4`):

1. **Type check:** `make typecheck` (`PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run pyright`) → **0 errors, 0 warnings**.
2. **Unit tests:** `uv run pytest tests/test_ag_ui.py` → **115/115 pass** (106 existing + 9 new).
3. **Broader sanity:** `pytest tests/test_capabilities.py tests/test_ui.py tests/test_ag_ui.py tests/test_tools.py` → **895/895 pass**.
4. **Lint / format:** `ruff check` and `ruff format` → clean.
5. **Docs examples:** `pytest tests/test_examples.py -k ag_ui` → **10 pass, 0 fail** — the new doc snippet is picked up automatically.
6. **Manual end-to-end (suggested for reviewer):** run the new example via `python -m pydantic_ai_examples.ag_ui` (per [examples/pydantic_ai_examples/ag_ui/__main__.py](examples/pydantic_ai_examples/ag_ui/__main__.py)) and POST to `/tool_approval`. Confirm:
   - Turn 1 stream ends with `RUN_FINISHED` carrying `outcome.type == "interrupt"`.
   - Turn 2 (with `resume: [{interrupt_id, status: "resolved", payload: {approved: true}}]`) emits a single `TOOL_CALL_RESULT` against the same `tool_call_id` — no second `TOOL_CALL_START`.
7. **Old-SDK regression:** `test_run_finished_no_outcome_on_legacy_version` proves the silent-skip path. A fresh-venv `ag-ui-protocol==0.1.10` install rerun is **not** needed because the version is negotiated at the adapter level (`ag_ui_version="0.1.10"`) rather than at import time. Two `cast` calls are present, both in `_payload_dict`, scoped to AG-UI's `Optional[Any]` payload boundary (explained inline).

## Out of scope (follow-ups)

- Mapping `DeferredToolRequests.calls` (external/frontend execution) to interrupts.
- `reason: "input_required"` and `reason: "confirmation"` — no Pydantic AI source today.
- Precise per-tool `response_schema` for `editedArgs` (will use `Any` in v1; can derive from `ToolDefinition.parameters_json_schema` in a follow-up).
- `Interrupt.expires_at` enforcement.
- Multi-interrupt parallel-tool-call resume ordering (AG-UI [PR #1493](https://github.com/ag-ui-protocol/ag-ui/pull/1493) WIP) — Pydantic AI already handles multiple approvals per run via the `approvals` list; just need to confirm ordering is preserved.

## Files changed

- [pyproject.toml](pyproject.toml) — added a temporary `[tool.uv.sources]` git pin on `ag-ui-protocol` (cleanup when 0.1.19 publishes).
- `uv.lock` — regenerated by `uv sync` to point `ag-ui-protocol` at git commit `df613e4`.
- [pydantic_ai_slim/pydantic_ai/ui/ag_ui/_utils.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_utils.py) — `INTERRUPTS_VERSION = (0, 1, 19)` constant.
- [pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py) — outbound `_build_outcome` + `_approval_to_interrupt`, version-gated `RunFinishedEvent.outcome` emission, `INTERRUPT_ID_PREFIX` constant.
- [pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py](pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py) — inbound `deferred_tool_results` `cached_property` override + `_interrupt_id_to_tool_call_id`, `_resume_entry_to_approval`, `_payload_dict` helpers.
- [tests/test_ag_ui.py](tests/test_ag_ui.py) — 9 new tests in a dedicated `# region: Interrupts` block, guarded by `@pytestmark_interrupts`.
- [examples/pydantic_ai_examples/ag_ui/api/tool_approval.py](examples/pydantic_ai_examples/ag_ui/api/tool_approval.py) — new example *(added)*.
- [examples/pydantic_ai_examples/ag_ui/api/__init__.py](examples/pydantic_ai_examples/ag_ui/api/__init__.py) and [examples/pydantic_ai_examples/ag_ui/__init__.py](examples/pydantic_ai_examples/ag_ui/__init__.py) — wire the new example into the dojo app at `/tool_approval`.
- [docs/ui/ag-ui.md](docs/ui/ag-ui.md) — new "Tool approval (interrupts)" section between `### Tools` and `### Events`.
