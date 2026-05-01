# PLAN: Continuation Support — ContinueRequestNode, ModelResponseState, Fallback Pinning

> **Issue refs:** #3365 (Anthropic/OpenAI Skills), #3963 (Shell/Bash builtin)
> **Stack:** `continuation-support` -> `skill-support-v2` -> `local-tools`

---

## Scope

General-purpose agent graph infrastructure for models that pause mid-turn and expect a continuation request. This enables Anthropic `pause_turn` and OpenAI background mode (both added in the follow-up `skill-support-v2` change).

---

## 1. ModelResponseState

New type on `ModelResponse` indicating whether the response is final or requires further action:

```python
ModelResponseState: TypeAlias = Literal['complete', 'suspended']
```

- `'complete'` — default, response is done
- `'suspended'` — model paused mid-turn, expects continuation

Added as `state` field on both `ModelResponse` (messages.py) and `StreamedResponse` (models/__init__.py). Also adds `metadata` field on `StreamedResponse` for fallback model stamping.

## 2. ContinueRequestNode

New node in the agent graph (`_agent_graph.py`) that handles automatic continuation when a model response has `state='suspended'`.

- Merges parts from the suspended response with the continuation response
- Tracks continuation count in `GraphAgentState.continuations`
- Enforces `_MAX_CONTINUATIONS = 50` safety limit
- Supports both streaming and non-streaming paths
- If continuation response is still suspended, chains to another `ContinueRequestNode`
- If complete, transitions to `CallToolsNode`

## 3. Fallback Model Continuation Pinning

When using `FallbackModel`, a model that starts a continuation must handle subsequent continuation requests — you can't switch models mid-continuation.

- `_stamp_continuation()` writes the model name into `response.metadata` under `__pydantic_ai__` key
- `_get_continuation_model()` reads the stamp from message history to find the pinned model
- `_rewind_messages()` strips the suspended response and trailing request when a pinned model fails, allowing fallback to proceed cleanly
- Both `request()` and `request_stream()` check for pinned continuation before entering the normal fallback chain

## 4. Test Coverage

All 18 new tests in `test_fallback.py` covering:
- Primary model continuation success (single and multiple pauses)
- Secondary model continuation after primary fails
- Continuation failure propagation
- Non-fallback error propagation during continuation
- Recovery with message rewinding
- Streaming variants of all above
- Stamp/metadata edge cases
