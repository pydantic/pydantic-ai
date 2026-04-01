# TODOS

## Deferred from Agentic Evaluators (branch: techno-anthropology/agentic-metrics)

### AgentTrace public class
- **What:** Promote private `_extract_tool_calls()` / `ToolCallInfo` to a public `AgentTrace.from_span_tree()` API
- **Why:** Users need typed access to agent execution traces for custom evaluators and debugging
- **Effort:** S (human) → S (CC)
- **Priority:** P2
- **Blocked by:** Instrumentation versions (v1-v5) stabilizing. Shape is too narrow while span naming is still churning.
- **Context:** CEO review reverted from ACCEPTED to DEFERRED after Codex outside voice argued the abstraction was premature. Private helper ships in v1; promote when instrumentation stabilizes.

### Trace-to-contract generator
- **What:** Utility that auto-generates evaluator configs from known-good SpanTrees ("record a good run, get an eval suite")
- **Why:** 10x DX improvement — eliminates manual evaluator configuration
- **Effort:** M (human) → S (CC)
- **Priority:** P2
- **Blocked by:** Stable AgentTrace public API (above)
- **Context:** Codex called this the "coolest version" during office-hours. Deferred until v1 evaluators prove the pattern.

### Logfire dashboard attributes
- **What:** Structured OTEL attributes on eval result spans so Logfire can render evaluator results in its dashboard
- **Why:** Eval results should light up in Logfire without custom dashboard config
- **Effort:** S (human) → S (CC)
- **Priority:** P3
- **Blocked by:** Logfire team coordination on attribute schema
- **Context:** Makes the observability story complete. Deferred because it requires cross-team alignment.

### ArgumentCorrectness semantic mode
- **What:** Add match_mode='semantic' to ArgumentCorrectness using LLMJudge internally
- **Why:** Exact/subset matching is brittle for natural language arguments
- **Effort:** S (human) → S (CC)
- **Priority:** P3
- **Blocked by:** Nothing (can ship independently)
- **Context:** Consistent with "ship fast, iterate later." Exact and subset cover most use cases.
