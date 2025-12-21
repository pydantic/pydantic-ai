# Design Document: Model Response Hooks

**Status**: Draft
**Authors**: @sarth6, Claude
**Issue References**: [#3640](https://github.com/pydantic/pydantic-ai/issues/3640), [#3408](https://github.com/pydantic/pydantic-ai/issues/3408)
**Created**: 2025-12-13

## Overview

This document proposes a generic hook system for processing model responses in pydantic-ai. Instead of adding `fallback_on_response` directly to `FallbackModel`, this approach implements response hooks at the agent level, providing a more powerful and composable abstraction.

### Motivation

**Issue #3640: Response-Based Fallback for FallbackModel**

Currently, `FallbackModel` only supports exception-based fallback via the `fallback_on` parameter. This works for API errors but cannot handle semantic failures where:
- The model returns HTTP 200 (no exception raised)
- But the response content indicates the operation failed

Example: Google's `WebFetchTool` may return a successful response with `url_retrieval_status: URL_RETRIEVAL_STATUS_FAILED`.

**Issue #3408: Pre/Post Hooks for Evaluation Cases**

The pydantic-evals package needs hooks for setup/teardown around task execution. A unified hook system can address both use cases.

**Douwe's Insight (from #3640)**:
> "I'm wondering if instead of a FallbackModel feature, this should be implemented as a generic hook for processing model responses, where it's up to the user whether they want to modify the response, validate it and raise an error, or anything else. If it raises an error, that can then be caught by FallbackModel."

---

## Proposed API

### 1. Response Processor Type Definitions

**Location**: `pydantic_ai_slim/pydantic_ai/_response_processor.py` (new file)

```python
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from pydantic_ai._run_context import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse

if TYPE_CHECKING:
    from pydantic_ai._agent_graph import GraphAgentDeps

# Type aliases following the history_processors pattern
_ResponseProcessorSync: TypeAlias = Callable[[ModelResponse, list[ModelMessage]], ModelResponse]
_ResponseProcessorAsync: TypeAlias = Callable[[ModelResponse, list[ModelMessage]], Awaitable[ModelResponse]]
_ResponseProcessorSyncWithCtx: TypeAlias = Callable[
    [RunContext['AgentDepsT'], ModelResponse, list[ModelMessage]], ModelResponse
]
_ResponseProcessorAsyncWithCtx: TypeAlias = Callable[
    [RunContext['AgentDepsT'], ModelResponse, list[ModelMessage]], Awaitable[ModelResponse]
]

ResponseProcessor: TypeAlias = (
    _ResponseProcessorSync
    | _ResponseProcessorAsync
    | _ResponseProcessorSyncWithCtx['AgentDepsT']
    | _ResponseProcessorAsyncWithCtx['AgentDepsT']
)
"""A function that processes a model response after it's received.

Can optionally accept a `RunContext` as the first parameter.

Returns:
    The (potentially modified) ModelResponse.

Raises:
    ResponseValidationError: To trigger fallback in FallbackModel.
    Any other exception: Re-raised to caller.
"""
```

### 2. Response Validation Exception

**Location**: `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class ResponseValidationError(PydanticAIError):
    """Raised by response processors to indicate semantic validation failure.

    When raised inside a response processor, this exception signals that the
    model response is semantically invalid (e.g., a tool returned an error status).

    FallbackModel will catch this exception and trigger fallback to the next model.

    Args:
        message: Human-readable description of the validation failure.
        response: The ModelResponse that failed validation.
    """

    def __init__(self, message: str, response: ModelResponse | None = None):
        super().__init__(message)
        self.response = response
```

### 3. Agent Constructor Extension

**Location**: `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

```python
from pydantic_ai._response_processor import ResponseProcessor

class Agent(Generic[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        model: ...,
        *,
        # ... existing parameters ...
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        response_processors: Sequence[ResponseProcessor[AgentDepsT]] | None = None,  # NEW
        # ... rest of parameters ...
    ):
        # ...
        self._response_processors = list(response_processors) if response_processors else []
```

### 4. GraphAgentDeps Extension

**Location**: `pydantic_ai_slim/pydantic_ai/_agent_graph.py` (lines ~130-157)

```python
@dataclasses.dataclass(kw_only=True)
class GraphAgentDeps(Generic[DepsT, OutputDataT]):
    # ... existing fields ...
    history_processors: Sequence[HistoryProcessor[DepsT]]
    response_processors: Sequence[ResponseProcessor[DepsT]]  # NEW
    # ...
```

### 5. ModelRequestNode Integration

**Location**: `pydantic_ai_slim/pydantic_ai/_agent_graph.py` (in `_make_request` method, after line 487)

```python
async def _make_request(
    self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
) -> CallToolsNode[DepsT, NodeRunEndT]:
    if self._result is not None:
        return self._result

    model_settings, model_request_parameters, message_history, run_context = await self._prepare_request(ctx)

    # Make the model request
    model_response = await ctx.deps.model.request(message_history, model_settings, model_request_parameters)
    ctx.state.usage.requests += 1

    # NEW: Process response through response processors
    model_response = await _process_model_response(
        model_response,
        message_history,
        ctx.deps.response_processors,
        run_context,
    )

    return self._finish_handling(ctx, model_response)
```

### 6. Response Processing Function

**Location**: `pydantic_ai_slim/pydantic_ai/_agent_graph.py` (new function)

```python
async def _process_model_response(
    response: _messages.ModelResponse,
    messages: list[_messages.ModelMessage],
    processors: Sequence[ResponseProcessor[DepsT]],
    run_context: RunContext[DepsT],
) -> _messages.ModelResponse:
    """Process a model response through all registered response processors.

    Args:
        response: The ModelResponse to process.
        messages: The message history that produced this response.
        processors: Sequence of response processor functions.
        run_context: The current run context.

    Returns:
        The (potentially modified) ModelResponse.

    Raises:
        ResponseValidationError: If a processor determines the response is invalid.
    """
    for processor in processors:
        takes_ctx = _takes_run_context(processor)
        is_async = _utils.is_async_callable(processor)

        if takes_ctx:
            args = (run_context, response, messages)
        else:
            args = (response, messages)

        if is_async:
            response = await processor(*args)
        else:
            response = await _utils.run_in_executor(processor, *args)

    return response
```

---

## Integration with FallbackModel

### How It Works

1. User registers a `response_processor` on the Agent
2. Processor inspects `ModelResponse` and raises `ResponseValidationError` if invalid
3. `FallbackModel` catches `ResponseValidationError` via its `fallback_on` parameter
4. Fallback triggers to the next model

### Updated FallbackModel Default

**Location**: `pydantic_ai_slim/pydantic_ai/models/fallback.py`

```python
from pydantic_ai.exceptions import ModelAPIError, ResponseValidationError

class FallbackModel(Model):
    def __init__(
        self,
        default_model: Model | KnownModelName | str,
        *fallback_models: Model | KnownModelName | str,
        fallback_on: Callable[[Exception], bool] | tuple[type[Exception], ...] = (
            ModelAPIError,
            ResponseValidationError,  # NEW: Include by default
        ),
    ):
        # ...
```

### Usage Example

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ResponseValidationError
from pydantic_ai.messages import (
    BuiltinToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.anthropic import AnthropicModel


def check_web_fetch_success(
    response: ModelResponse,
    messages: list[ModelMessage]
) -> ModelResponse:
    """Raise ResponseValidationError if web_fetch tool failed."""
    for call, result in response.builtin_tool_calls:
        if call.tool_name == 'web_fetch':
            content = result.content
            if isinstance(content, dict):
                status = content.get('url_retrieval_status', '')
                if status and status != 'URL_RETRIEVAL_STATUS_SUCCESS':
                    raise ResponseValidationError(
                        f"web_fetch failed: {status}",
                        response=response,
                    )
    return response


# With RunContext for advanced use cases
def check_with_context(
    ctx: RunContext[MyDeps],
    response: ModelResponse,
    messages: list[ModelMessage],
) -> ModelResponse:
    """Example showing RunContext access."""
    # Can access ctx.deps, ctx.model, ctx.usage, etc.
    if ctx.deps.strict_mode:
        # Perform stricter validation
        pass
    return response


google_model = GoogleModel('gemini-2.0-flash')
anthropic_model = AnthropicModel('claude-3-5-haiku-latest')

fallback_model = FallbackModel(google_model, anthropic_model)

agent = Agent(
    model=fallback_model,
    builtin_tools=[WebFetchTool()],
    response_processors=[check_web_fetch_success],
)

# If Google's web_fetch fails, ResponseValidationError triggers fallback to Anthropic
result = await agent.run("Summarize https://example.com")
```

---

## Streaming Considerations

For `request_stream()`, response processing happens **after** the stream completes:

**Location**: `pydantic_ai_slim/pydantic_ai/_agent_graph.py` (in `ModelRequestNode.stream` method)

```python
@asynccontextmanager
async def stream(
    self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
) -> AsyncIterator[StreamedResponse]:
    # ... existing code ...

    async with ctx.deps.model.request_stream(...) as streamed_response:
        yield streamed_response

        # After stream completes, get final response
        model_response = streamed_response.get()

        # NEW: Process response (may raise ResponseValidationError)
        model_response = await _process_model_response(
            model_response,
            message_history,
            ctx.deps.response_processors,
            run_context,
        )

        self._result = self._finish_handling(ctx, model_response)
```

**Important**: Response processors are called **after** streaming completes. This means:
- Users receive streamed content in real-time
- Validation happens at the end
- If validation fails, the error is raised after streaming completes

For use cases requiring mid-stream validation, users should use the existing `UIEventStream` hooks instead.

---

## Integration with Evaluation Hooks (Issue #3408)

The same pattern can extend to pydantic-evals for pre/post task hooks.

### Proposed Evaluation Hook Types

**Location**: `pydantic_evals/pydantic_evals/dataset.py`

```python
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from pydantic_evals.evaluators.context import EvaluatorContext

# Pre-task hook: Called before task execution
PreTaskHook: TypeAlias = (
    Callable[[Case[InputsT, OutputT, MetadataT]], None]
    | Callable[[Case[InputsT, OutputT, MetadataT]], Awaitable[None]]
)

# Post-task hook: Called after task execution with results
PostTaskHook: TypeAlias = (
    Callable[[EvaluatorContext[InputsT, OutputT, MetadataT]], None]
    | Callable[[EvaluatorContext[InputsT, OutputT, MetadataT]], Awaitable[None]]
)

# Metric extraction hook (replacing hardcoded logic at lines 945-962)
MetricExtractorHook: TypeAlias = (
    Callable[['_TaskRun', SpanTree], None]
    | Callable[['_TaskRun', SpanTree], Awaitable[None]]
)
```

### Dataset.evaluate() Extension

```python
class Dataset(BaseModel, Generic[InputsT, OutputT, MetadataT]):
    async def evaluate(
        self,
        task: Callable[[InputsT], Awaitable[OutputT] | OutputT],
        *,
        # ... existing parameters ...
        pre_task_hook: PreTaskHook[InputsT, OutputT, MetadataT] | None = None,  # NEW
        post_task_hook: PostTaskHook[InputsT, OutputT, MetadataT] | None = None,  # NEW
        metric_extractor: MetricExtractorHook | None = None,  # NEW
    ) -> EvaluationReport[InputsT, OutputT, MetadataT]:
        # ...
```

---

## Implementation Files Summary

| File | Changes |
|------|---------|
| `pydantic_ai_slim/pydantic_ai/_response_processor.py` | **NEW** - Type definitions |
| `pydantic_ai_slim/pydantic_ai/exceptions.py` | Add `ResponseValidationError` |
| `pydantic_ai_slim/pydantic_ai/agent/__init__.py` | Add `response_processors` parameter |
| `pydantic_ai_slim/pydantic_ai/_agent_graph.py` | Add to `GraphAgentDeps`, implement `_process_model_response()`, call in `_make_request()` and `stream()` |
| `pydantic_ai_slim/pydantic_ai/models/fallback.py` | Add `ResponseValidationError` to default `fallback_on` |
| `pydantic_evals/pydantic_evals/dataset.py` | (Optional) Add pre/post task hooks |

---

## Test Strategy

### 1. Unit Tests for Response Processors

**Location**: `tests/test_response_processors.py` (new file)

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ResponseValidationError
from pydantic_ai.messages import ModelResponse, ModelMessage, TextPart
from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.anyio


async def test_response_processor_called():
    """Test that response processors are invoked after model request."""
    call_count = 0

    def counting_processor(response: ModelResponse, messages: list[ModelMessage]) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return response

    model = TestModel()
    agent = Agent(model, response_processors=[counting_processor])

    await agent.run('test')
    assert call_count == 1


async def test_response_processor_can_modify_response():
    """Test that processors can modify the response."""
    def modify_processor(response: ModelResponse, messages: list[ModelMessage]) -> ModelResponse:
        # Modify the response text
        new_parts = [TextPart(content='MODIFIED')]
        return ModelResponse(parts=new_parts, usage=response.usage, model_name=response.model_name)

    model = TestModel(custom_output_text='original')
    agent = Agent(model, response_processors=[modify_processor])

    result = await agent.run('test')
    assert 'MODIFIED' in result.output


async def test_response_processor_with_run_context():
    """Test processor that takes RunContext."""
    from dataclasses import dataclass
    from pydantic_ai import RunContext

    @dataclass
    class MyDeps:
        strict: bool

    def context_processor(
        ctx: RunContext[MyDeps],
        response: ModelResponse,
        messages: list[ModelMessage],
    ) -> ModelResponse:
        if ctx.deps.strict:
            # Could perform stricter validation here
            pass
        return response

    model = TestModel()
    agent = Agent(model, deps_type=MyDeps, response_processors=[context_processor])

    await agent.run('test', deps=MyDeps(strict=True))


async def test_response_processor_raises_validation_error():
    """Test that ResponseValidationError propagates correctly."""
    def failing_processor(response: ModelResponse, messages: list[ModelMessage]) -> ModelResponse:
        raise ResponseValidationError("Test failure", response=response)

    model = TestModel()
    agent = Agent(model, response_processors=[failing_processor])

    with pytest.raises(ResponseValidationError, match="Test failure"):
        await agent.run('test')


async def test_multiple_processors_chained():
    """Test that multiple processors are called in order."""
    call_order = []

    def processor_a(response: ModelResponse, messages: list[ModelMessage]) -> ModelResponse:
        call_order.append('a')
        return response

    def processor_b(response: ModelResponse, messages: list[ModelMessage]) -> ModelResponse:
        call_order.append('b')
        return response

    model = TestModel()
    agent = Agent(model, response_processors=[processor_a, processor_b])

    await agent.run('test')
    assert call_order == ['a', 'b']


async def test_async_response_processor():
    """Test async response processor."""
    async def async_processor(response: ModelResponse, messages: list[ModelMessage]) -> ModelResponse:
        # Simulate async operation
        import anyio
        await anyio.sleep(0)
        return response

    model = TestModel()
    agent = Agent(model, response_processors=[async_processor])

    await agent.run('test')
```

### 2. FallbackModel Integration Tests

**Location**: `tests/models/test_fallback.py` (additions)

```python
async def test_fallback_on_response_validation_error():
    """Test that ResponseValidationError triggers fallback."""
    from pydantic_ai.exceptions import ResponseValidationError

    call_count = {'primary': 0, 'fallback': 0}

    def primary_func(messages, info):
        call_count['primary'] += 1
        return ModelResponse(parts=[TextPart(content='primary response')])

    def fallback_func(messages, info):
        call_count['fallback'] += 1
        return ModelResponse(parts=[TextPart(content='fallback response')])

    def failing_processor(response: ModelResponse, messages: list[ModelMessage]) -> ModelResponse:
        if 'primary' in response.text:
            raise ResponseValidationError("Primary model response invalid")
        return response

    primary_model = FunctionModel(primary_func)
    fallback_model_impl = FunctionModel(fallback_func)

    fallback = FallbackModel(primary_model, fallback_model_impl)
    agent = Agent(fallback, response_processors=[failing_processor])

    result = await agent.run('test')

    assert call_count['primary'] == 1
    assert call_count['fallback'] == 1
    assert 'fallback' in result.output


async def test_fallback_web_fetch_failure_scenario():
    """Test the original issue scenario: web_fetch failure triggers fallback."""
    from pydantic_ai.messages import BuiltinToolCallPart, BuiltinToolReturnPart

    def google_response(messages, info):
        # Simulate Google's failed web_fetch
        return ModelResponse(parts=[
            BuiltinToolCallPart(tool_name='web_fetch', args={}, tool_call_id='1'),
            BuiltinToolReturnPart(
                tool_name='web_fetch',
                tool_call_id='1',
                content={'url_retrieval_status': 'URL_RETRIEVAL_STATUS_FAILED'},
            ),
            TextPart(content='Could not fetch URL'),
        ])

    def anthropic_response(messages, info):
        return ModelResponse(parts=[TextPart(content='Successfully fetched content')])

    def check_web_fetch(response: ModelResponse, messages: list[ModelMessage]) -> ModelResponse:
        for call, result in response.builtin_tool_calls:
            if call.tool_name == 'web_fetch':
                content = result.content
                if isinstance(content, dict):
                    status = content.get('url_retrieval_status', '')
                    if 'FAILED' in status:
                        raise ResponseValidationError(f"web_fetch failed: {status}")
        return response

    google = FunctionModel(google_response)
    anthropic = FunctionModel(anthropic_response)

    fallback = FallbackModel(google, anthropic)
    agent = Agent(fallback, response_processors=[check_web_fetch])

    result = await agent.run('Summarize https://example.com')
    assert 'Successfully' in result.output
```

### 3. Streaming Tests

**Location**: `tests/test_response_processors.py` (additions)

```python
async def test_response_processor_with_streaming():
    """Test that response processors work with streaming."""
    processor_called = False

    def processor(response: ModelResponse, messages: list[ModelMessage]) -> ModelResponse:
        nonlocal processor_called
        processor_called = True
        return response

    model = TestModel()
    agent = Agent(model, response_processors=[processor])

    async with agent.run_stream('test') as result:
        output = await result.get_output()

    assert processor_called


async def test_streaming_validation_error_after_completion():
    """Test that validation errors are raised after stream completes."""
    def failing_processor(response: ModelResponse, messages: list[ModelMessage]) -> ModelResponse:
        raise ResponseValidationError("Post-stream validation failed")

    model = TestModel()
    agent = Agent(model, response_processors=[failing_processor])

    with pytest.raises(ResponseValidationError):
        async with agent.run_stream('test') as result:
            # Stream content received
            chunks = []
            async for chunk in result.stream_text():
                chunks.append(chunk)
            # Error raised when exiting context
```

### 4. Test Coverage Requirements

Per CLAUDE.md, all PRs must have 100% coverage. Key areas to cover:

1. **Sync vs Async processors**: Both paths in `_process_model_response()`
2. **With/without RunContext**: Both signature detection branches
3. **Error propagation**: ResponseValidationError raised and caught correctly
4. **Multiple processors**: Chain execution order
5. **Response modification**: Processors can modify and return new response
6. **FallbackModel integration**: ResponseValidationError triggers fallback
7. **Streaming integration**: Processors called after stream completes
8. **Edge cases**: Empty processor list, None response parts

---

## Alternatives Considered

### Alternative 1: Add `fallback_on_response` directly to FallbackModel

```python
fallback_on_response: Callable[[ModelResponse, list[ModelMessage]], bool] | None = None
```

**Rejected because**:
- Less composable - only works within FallbackModel
- Cannot be reused for other purposes (logging, metrics, modification)
- Response inspection tied to fallback behavior

### Alternative 2: Extend `fallback_on` to accept response

```python
fallback_on: Callable[[Exception | ModelResponse], bool] | ...
```

**Rejected because**:
- Mixing exception and response handling is confusing
- Breaks existing type signatures
- Response inspection is not a fallback condition, it's a validation step

### Alternative 3: Custom exception wrapping (original suggestion)

Wrap response inspection in a model subclass that raises exceptions.

**Rejected because**:
- Requires subclassing models
- Doesn't provide composability
- Harder to share validation logic across different model configurations

### Alternative 4: Output validator with ModelRetry

Use `@agent.output_validator` to raise `ModelRetry`.

**Rejected because**:
- `ModelRetry` triggers retry with the **same** model, not fallback
- Output validators run after tool execution, too late for response-level checks
- Response inspection needs to happen before tool execution for some use cases

---

## Migration Guide

### For Users Currently Using Workarounds

**Before** (manual fallback logic):
```python
class MyAgent:
    def __init__(self):
        self._google_agent = Agent(model=google_model, builtin_tools=[WebFetchTool()])
        self._anthropic_agent = Agent(model=anthropic_model, builtin_tools=[WebFetchTool()])

    async def run(self, prompt: str) -> str:
        try:
            result = await self._google_agent.run(prompt)
            if self._check_success(result.all_messages()):
                return result.output
        except Exception:
            pass
        return (await self._anthropic_agent.run(prompt)).output
```

**After** (using response processors):
```python
def check_web_fetch(response: ModelResponse, messages: list[ModelMessage]) -> ModelResponse:
    for call, result in response.builtin_tool_calls:
        if call.tool_name == 'web_fetch':
            content = result.content
            if isinstance(content, dict) and 'FAILED' in content.get('url_retrieval_status', ''):
                raise ResponseValidationError("web_fetch failed")
    return response

fallback = FallbackModel(google_model, anthropic_model)
agent = Agent(
    model=fallback,
    builtin_tools=[WebFetchTool()],
    response_processors=[check_web_fetch],
)

result = await agent.run(prompt)  # Automatic fallback on failure
```

---

## Open Questions

1. **Should `ResponseValidationError` be included in `fallback_on` by default?**
   - Pro: Intuitive behavior for the primary use case
   - Con: May surprise users who want validation errors to propagate

2. **Should response processors be able to access tool results?**
   - Currently: `builtin_tool_calls` property provides access to built-in tool results
   - User-defined tool results are in subsequent `ModelRequest` messages
   - May need additional API for accessing pending tool results

3. **Should there be a `pre_request_processor` as well?**
   - `history_processors` already serve this purpose
   - Could rename for clarity: `request_processors` and `response_processors`

4. **Streaming behavior: validate during or after?**
   - Current proposal: validate after stream completes
   - Alternative: provide streaming-aware hooks via `UIEventStream`

---

## Implementation Checklist

- [ ] Create `pydantic_ai_slim/pydantic_ai/_response_processor.py` with type definitions
- [ ] Add `ResponseValidationError` to `pydantic_ai_slim/pydantic_ai/exceptions.py`
- [ ] Add `response_processors` parameter to `Agent.__init__()`
- [ ] Add `response_processors` to `GraphAgentDeps` dataclass
- [ ] Implement `_process_model_response()` function in `_agent_graph.py`
- [ ] Call `_process_model_response()` in `ModelRequestNode._make_request()`
- [ ] Call `_process_model_response()` in `ModelRequestNode.stream()`
- [ ] Update `FallbackModel` default `fallback_on` to include `ResponseValidationError`
- [ ] Add comprehensive unit tests (100% coverage required)
- [ ] Add integration tests for FallbackModel + response processors
- [ ] Add streaming tests
- [ ] Update documentation
- [ ] (Optional) Implement evaluation hooks for pydantic-evals

---

## References

- [Issue #3640: Response-Based Fallback for FallbackModel](https://github.com/pydantic/pydantic-ai/issues/3640)
- [Issue #3408: Pre- and post-hooks for specific cases](https://github.com/pydantic/pydantic-ai/issues/3408)
- [Issue #2837: FallbackModel doesn't handle UnexpectedModelBehavior](https://github.com/pydantic/pydantic-ai/issues/2837)
- Current `history_processors` implementation: `pydantic_ai_slim/pydantic_ai/_agent_graph.py:71-86`
- Current `FallbackModel` implementation: `pydantic_ai_slim/pydantic_ai/models/fallback.py`
- Current `output_validator` pattern: `pydantic_ai_slim/pydantic_ai/_output.py:161-211`
