# Capabilities — Niche Caveats Worth Documenting

Brief notes on edge-case behaviors discovered during bug hunting (PR #4640).
These don't warrant dedicated doc sections but should appear as caveats/notes
near the relevant feature documentation once the bugs are fixed.

## 1. DynamicToolset factory errors leave ambiguous state

**Where to document:** `docs/toolsets.md`, near the dynamic toolset section.

**Caveat:** If a `per_run_step=True` factory raises mid-run, the DynamicToolset
doesn't roll back — the previous toolset stays referenced but the overall state
is ambiguous. Worse: if the factory succeeds but the new toolset's `__aenter__`
fails, the old toolset has already been exited and the new one never entered.
`self._toolset` then points to an un-initialized toolset.

**User guidance:** Factory functions used with `per_run_step=True` should be
defensive — catch and handle their own errors rather than letting exceptions
propagate into the toolset lifecycle. If the factory can fail, consider
returning the previous toolset as a fallback instead of raising.

## 2. Tool retry counts persist across DynamicToolset tool swaps

**Where to document:** `docs/toolsets.md`, near dynamic toolsets or tool retries.

**Caveat:** `ToolManager` tracks tool retries by **name** across run steps. If a
`DynamicToolset` with `per_run_step=True` replaces a tool's implementation
between steps but keeps the same name, the new implementation inherits the
accumulated retry count from the old one. A fresh tool can be born with N
retries already counted against it, hitting `max_retries` sooner than expected.

**User guidance:** If tool implementations change between steps, use distinct
tool names to avoid inheriting retry state. Alternatively, design tools with
retry budgets that account for the possibility of inherited counts.

## 3. History processor composition doesn't validate message consistency

**Where to document:** `docs/message-history.md`, near the history processors section.

**Caveat:** When multiple history processors are registered, they compose in
sequence — processor 2 sees processor 1's output. There's no validation that
the resulting messages are semantically consistent. A processor that removes
`ModelResponse` messages containing `ToolCallPart` but leaves the subsequent
`ModelRequest` with `ToolReturnPart` creates orphaned tool returns — the model
sees a tool result with no preceding tool call.

**User guidance:** When writing history processors that remove messages, ensure
tool calls and their corresponding tool returns are removed together. A safe
pattern is to track tool call IDs and remove both the call and return as a pair.
