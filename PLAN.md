# PLAN: Multi-Turn Conversation Quality Metrics for pydantic-evals

## Problem

Developers building multi-turn agents with Pydantic AI have no way to measure conversation quality across turns. Logfire/OTel captures token counts, latency, and tool execution, but not whether the agent achieves goals, retains knowledge, or stays coherent over a multi-turn conversation.

## Proposal

Add a `GoalAchievement` conversation evaluator with a **shared extraction module** that both this evaluator and future agentic evaluators can reuse. The extraction module is the real reusable primitive. GoalAchievement is the first consumer.

**Related:** This builds on the same span extraction patterns needed by the "Batteries-Included Agentic Evaluators" design (separate branch, `techno-anthropology/agentic-metrics`).

## Design Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | File location | `evaluators/conversation.py` + `evaluators/extraction.py` | Matches existing pattern (evaluators in one dir), avoids new subpackage |
| 2 | Architecture | Shared extraction module + GoalAchievement | Extraction is the reusable primitive; prevents duplication with agentic evaluators branch |
| 3 | Judge strategy | Single LLM call per conversation | 10x cheaper than per-turn, holistic judgment is more reliable |
| 4 | Return shape | 2 entries: `goal_achieved` (bool) + `goal_achievement_score` (float w/ reason) | Matches LLMJudge dict pattern, blame lives in reason text |
| 5 | Data model | Parse typed OTel message parts, flatten to turns | `all_messages` contains `SystemPromptPart`, `UserPromptPart`, `TextPart`, etc. |
| 6 | Blame fields | No structured `first_bad_turn`/`largest_drop_turn` | Single holistic call can't reliably pinpoint exact turns; blame in reason text is more honest |
| 7 | Error handling | Let `SpanTreeRecordingError` propagate | Matches `HasMatchingSpan` behavior; consistent error pattern |
| 8 | ConversationTurn enrichment | Optional `tool_name`/`tool_arguments` fields | Enables agentic evaluators to reuse extraction layer without modification |

## Implementation Steps

### Step 1: Create shared extraction module

**File:** `pydantic_evals/pydantic_evals/evaluators/extraction.py` (NEW)

```
SPAN DATA FLOW
═══════════════════════════════════════════════════════

  SpanTree
    │
    ├── find(agent_run_query)
    │   └── SpanNode
    │       └── attributes['pydantic_ai.all_messages']
    │           └── JSON string
    │               └── list[TypedMessagePart]
    │                   ├── SystemPromptPart
    │                   ├── UserPromptPart
    │                   ├── TextPart
    │                   ├── ToolCallPart
    │                   └── ToolReturnPart
    │
    ├── FALLBACK: find(model_request_query)
    │   └── attributes['gen_ai.input.messages']
    │       └── attributes['gen_ai.output.messages']
    │
    └── Output: list[ConversationTurn]
        (flattened from typed parts, with optional tool metadata)

═══════════════════════════════════════════════════════
```

```python
@dataclass
class ConversationTurn:
    """A flattened conversation turn extracted from OTel spans."""
    role: Literal['user', 'assistant', 'system', 'tool']
    content: str
    turn_index: int
    # Optional tool metadata (populated for ToolCallPart/ToolReturnPart)
    # Enables agentic evaluators to reuse this extraction layer
    tool_name: str | None = None
    tool_arguments: str | None = None  # JSON string

def extract_conversation_turns(span_tree: SpanTree) -> list[ConversationTurn]:
    """Extract ordered conversation turns from span tree. (PUBLIC)

    Parses typed message parts from pydantic_ai.all_messages attribute.
    Falls back to gen_ai.input/output.messages on model request spans.
    Handles pydantic_ai.new_message_index to avoid re-scoring prior history.

    Wraps all JSON/key parsing in try/except per-part: malformed parts are
    skipped with a logger.warning(), not raised.
    """

def _flatten_typed_parts(messages_json: str) -> list[ConversationTurn]:
    """Convert typed OTel message parts to flat turns. (PRIVATE)

    Handles: SystemPromptPart -> system, UserPromptPart -> user,
    TextPart -> assistant, ToolCallPart -> assistant (tool_name + tool_arguments populated),
    ToolReturnPart -> tool.

    Catches json.JSONDecodeError and KeyError per-part, logs warning, skips.
    """

def format_transcript(turns: list[ConversationTurn]) -> str:
    """Format turns as numbered transcript for judge prompts. (PUBLIC)"""
```

**Key implementation details:**
- Parse `pydantic_ai.all_messages` JSON per `_otel_messages.py` typed format
- Use `pydantic_ai.new_message_index` to identify new turns in continued runs
- Span selection: `SpanQuery(or_=[SpanQuery(name_equals='agent run'), SpanQuery(name_contains='invoke_agent')])`
- Multimodal content: placeholder `[image]` or `[file: name]`
- **Error resilience:** Wrap JSON parsing and field access in try/except per-part. Log `logger.warning()` for malformed data. Never crash on unexpected span data.
- **Public API:** `extract_conversation_turns()` and `format_transcript()` are public (no underscore). `_flatten_typed_parts()` is private.

### Step 2: Add GoalAchievement evaluator

**File:** `pydantic_evals/pydantic_evals/evaluators/conversation.py` (NEW)

```python
@dataclass(repr=False)
class GoalAchievement(Evaluator[object, object, object]):
    """Evaluates whether a multi-turn conversation achieves a stated goal."""
    goal: str
    threshold: float = 0.7
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    evaluation_name: str | None = field(default=None)

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
        # 1. Access span tree — let SpanTreeRecordingError propagate
        span_tree = ctx.span_tree

        # 2. Extract turns from typed OTel message parts
        turns = extract_conversation_turns(span_tree)
        if not turns:
            return EvaluationReason(value=False, reason='No conversation turns found in span tree.')

        # 3. Format transcript, include ctx.output, and judge
        transcript = format_transcript(turns)
        result = await self._judge_conversation(transcript, ctx.output)

        # 4. Return 2-entry dict
        # NOTE: We compute achieved from score >= threshold, NOT from result.pass_.
        # GradingOutput.pass_ is the judge's opinion; threshold is user-configurable.
        achieved = result.score >= self.threshold
        return {
            'goal_achieved': EvaluationReason(value=achieved, reason=result.reason),
            'goal_achievement_score': result.score,
        }
```

**Judge prompt template:**
```
You are evaluating a multi-turn conversation against a stated goal.

GOAL: {goal}

CONVERSATION TRANSCRIPT:
{transcript}

AGENT OUTPUT: {output}

Evaluate the conversation holistically:
1. Did the conversation achieve the stated goal? Score 0.0 to 1.0.
2. If the goal was not fully achieved, explain which part of the conversation
   went wrong and why. Reference specific turns by number.
3. If quality degraded at a specific point, identify the turn where it happened.

Respond with:
- reason: Your assessment including blame attribution (which turns helped/hurt)
- pass_: true if score >= {threshold}
- score: 0.0 to 1.0
```

**Judge implementation:** Reuse `GradingOutput` from `llm_as_a_judge.py:27-32`. Create a judge agent following the `_judge_output_agent` pattern. Single call per conversation. Pass `ctx.output` as the agent output in the prompt.

### Step 3: Wire up public API

**File:** `pydantic_evals/pydantic_evals/evaluators/__init__.py` — add exports:
- `GoalAchievement` from `conversation.py`
- `ConversationTurn`, `extract_conversation_turns`, `format_transcript` from `extraction.py`

Import path: `from pydantic_evals.evaluators import GoalAchievement`

### Step 4: Write tests

**File:** `tests/test_conversation_evaluators.py` — unit tests
**File:** `tests/test_conversation_evaluators_integration.py` — integration test with recording

#### Unit tests (no LLM calls):

1. **`extract_conversation_turns` with real OTel message format** — mock SpanTree with typed parts, verify correct flattening including tool_name/tool_arguments population
2. **`_flatten_typed_parts` edge cases:**
   - Empty messages list -> empty turns
   - System prompt only -> single system turn
   - Multi-part assistant response (TextPart + ToolCallPart) -> multiple turns, tool metadata populated
   - Multimodal content (image part) -> placeholder text
   - Malformed JSON in all_messages -> returns empty list, logs warning
   - Missing fields in message part -> skips that part, logs warning
   - Unknown message part type -> skips with warning
3. **Continued conversation handling** — SpanTree with `pydantic_ai.new_message_index`, verify only new turns extracted
   - Invalid new_message_index value -> falls back to scoring all turns
4. **Fallback to model request spans** — without `all_messages`, use `gen_ai.input.messages`/`gen_ai.output.messages`
5. **Missing content (`include_content=False`)** — no message attributes, verify clear error
6. **`format_transcript`** — verify numbered output format
7. **Nested agent runs** — parent calls child agent, extract from outermost only
8. **Threshold boundary** — score exactly equals threshold -> `goal_achieved` is True

#### Integration test (with recording):

1. **GoalAchievement end-to-end** — record real LLM judge call, assert on `goal_achieved` and `goal_achievement_score` using `inline-snapshot`

**Testing pattern:** Follow `tests/test_evals.py`.

### Step 5: Write documentation

**File:** `docs/evals.md` — add "Conversation Evaluators" section

Content:
- What multi-turn conversation evaluation is
- GoalAchievement usage example (3-5 lines)
- How the judge prompt works (single call, holistic, includes agent output)
- Reading the blame narrative in the reason text
- Cost: 1 LLM judge call per conversation evaluation
- **Limitation note:** Very long conversations may be truncated by the judge model's context window
- Integration with Dataset.evaluate

## Critical Files

| File | Action | Purpose |
|------|--------|---------|
| `pydantic_evals/pydantic_evals/evaluators/extraction.py` | CREATE | Shared extraction: ConversationTurn, extract_conversation_turns, format_transcript |
| `pydantic_evals/pydantic_evals/evaluators/conversation.py` | CREATE | GoalAchievement evaluator |
| `pydantic_evals/pydantic_evals/evaluators/__init__.py` | EDIT | Add exports |
| `tests/test_conversation_evaluators.py` | CREATE | Unit tests |
| `tests/test_conversation_evaluators_integration.py` | CREATE | Integration test with recording |
| `docs/evals.md` | EDIT | Add conversation evaluators section |

## What Already Exists (reuse, don't rebuild)

| What | Where | How to reuse |
|------|-------|--------------|
| `GradingOutput` BaseModel | `evaluators/llm_as_a_judge.py:27-32` | Use directly as judge output type |
| `_judge_output_agent` pattern | `evaluators/llm_as_a_judge.py:35-70` | Follow same Agent + system_prompt + output_type pattern |
| `LLMJudge` model param handling | `evaluators/common.py:198-240` | Same model/model_settings/default pattern |
| `HasMatchingSpan` span access | `evaluators/common.py:270-280` | Same `ctx.span_tree` access, same error propagation |
| `SpanTree.find()` + `SpanQuery` | `otel/span_tree.py` | Query for agent run spans |
| `EvaluatorContext` | `evaluators/context.py` | Access span_tree, output |
| OTel message format | `pydantic_ai/_otel_messages.py` | Reference for typed part parsing |
| `Evaluator` base class | `evaluators/_base.py` | Extend via `Evaluator[object, object, object]` |

## NOT in scope (v1)

| Item | Rationale |
|------|-----------|
| Structured blame fields (`first_bad_turn`, `largest_drop_turn`) | Single holistic judge call can't reliably pinpoint; blame in reason text is more honest |
| Per-turn scoring / `TurnGradingOutput` | Overbuilt for v1; single holistic score + blame narrative is sufficient |
| `ConversationScore` / `TurnScore` dataclasses | Not needed when blame is in reason text |
| `conversation/` subpackage | Single file in `evaluators/` is enough |
| `ConversationTranscript` wrapper | Premature abstraction; wait for real consumers |
| Token budget awareness | Medium effort, model-specific complexity |
| Regression comparison (`compare_runs`) | Needs baseline storage design; follow-up |
| Additional metrics (Coherence, Knowledge Retention, Role Adherence) | Ship GoalAchievement first, iterate based on usage |
| `ConversationDataset` specialized type | Wait for usage patterns |
| Batch mode (multiple conversations per judge call) | Optimization for later |

## Error & Rescue Registry

| Method/Codepath | Exception | Rescued? | Action | User Sees |
|-----------------|-----------|----------|--------|-----------|
| `ctx.span_tree` | `SpanTreeRecordingError` | N (propagate) | — | Clear error: "configure logfire" |
| `_flatten_typed_parts` | `json.JSONDecodeError` | Y (catch per-part) | Log warning, skip part | Partial transcript |
| `_flatten_typed_parts` | `KeyError` (missing fields) | Y (catch per-part) | Log warning, skip part | Partial transcript |
| `extract_conversation_turns` | `IndexError` (bad new_message_index) | Y (catch) | Fall back to all turns | Full transcript scored |
| `_judge_conversation` | `httpx.TimeoutException` | N (propagate) | — | Standard eval error |
| `_judge_conversation` | `ValidationError` (GradingOutput) | Y (agent retry) | Retry up to max | Transparent |

## Failure Modes

| Codepath | Failure | Test? | Error handling? | User experience |
|----------|---------|-------|-----------------|-----------------|
| `ctx.span_tree` access | OTel not configured | YES | `SpanTreeRecordingError` propagates | Clear error |
| `extract_conversation_turns` | No agent run spans | YES | Returns empty -> EvaluationReason(False) | Clear message |
| `extract_conversation_turns` | `include_content=False` | YES | Returns empty + reason | Clear message |
| `_flatten_typed_parts` | Malformed JSON | YES | Catch, log warning, return empty | Warning in logs |
| `_flatten_typed_parts` | Missing part fields | YES | Catch per-part, skip | Partial transcript |
| `_flatten_typed_parts` | Unknown part type | YES | Skip with warning | Silent skip (logged) |
| `_flatten_typed_parts` | Multimodal content | YES | Placeholder `[image]` | Judge sees placeholder |
| `extract_conversation_turns` | Invalid new_message_index | YES | Fall back to all turns | Correct behavior |
| Judge agent call | LLM API error | YES (integration) | Propagates (eval runner) | Standard eval error |
| Continued conversation | new_message_index present | YES | Only score new turns | Correct behavior |
| Nested agent runs | Parent calls child | YES | Outermost agent only | Correct behavior |
| GoalAchievement | score == threshold exactly | YES | achieved = True | Correct behavior |
| Very long conversation | Token limit on judge | NO (docs note) | Judge truncates | Potentially inaccurate |

**Critical gaps:** None. All failure modes have tests or are documented limitations.

## Usage Example

```python
import logfire
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import GoalAchievement

logfire.configure(send_to_logfire='if-token-present')

dataset = Dataset(
    name='test-conversations',
    cases=[
        Case(
            name='refund-request',
            inputs='Help me get a refund',
            evaluators=[
                GoalAchievement(goal='Help the user get a refund for their product'),
            ],
        ),
    ],
)
report = dataset.evaluate_sync(run_my_agent)
print(report)
```

## Verification

1. **Unit tests:** `uv run pytest tests/test_conversation_evaluators.py -v`
2. **Integration test:** `uv run pytest tests/test_conversation_evaluators_integration.py -v --record-mode=once`
3. **Type checking:** `make typecheck`
4. **Linting:** `make lint && make format`

## Future Work

- **Token budget awareness** — count transcript tokens before judge call, truncate smartly if needed
- **ConversationTranscript wrapper** — richer return type once multiple consumers exist
- **Structured blame fields** — if users need programmatic `first_bad_turn`, add structured output
- **Regression comparison** (`compare_runs`) — needs baseline storage design
- **Per-turn scoring** — O(turns) judge calls, expensive but more granular
- **Additional metrics** — Coherence, Knowledge Retention, Role Adherence
- **Batch mode** — cost optimization for scoring many conversations

## 12-Month Vision

```
12-MONTH IDEAL                          THIS PLAN GETS US
─────────────────────────────────────   ─────────────────────────────────────
Full quality suite (5+ metrics)         1 metric (GoalAchievement)
Shared extraction layer                 extraction.py as shared module
Automated regression detection          Deferred (needs baseline storage)
Logfire dashboard integration           Works with existing Logfire eval reports
Agentic evaluators on same extraction   ConversationTurn enriched for reuse
```

The extraction module is the load-bearing piece. Once it exists, adding metrics is straightforward.
