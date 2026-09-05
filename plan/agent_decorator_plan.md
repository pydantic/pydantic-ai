# Implementation Specification: Functional `@Agent` Decorator API

This specification outlines the integration of a functional decorator API into Pydantic AI. This feature enables `Agent` instances to act as decorators, allowing developers to orchestrate agent-driven logic using standard Python function signatures.

---

## 1. Technical Design Rationale

### Objective
To provide a declarative, signature-based alternative to the imperative `.run()` API. This reduces boilerplate when agents are used for simple task-oriented functions or integrated into graph-based workflows.

### Why `AbstractAgent.__call__`?
Integrating at the `AbstractAgent` level ensures that all agent variants (standard `Agent`, `WrapperAgent`) inherit this capability automatically. The decorator remains "inert" until called, at which point it performs signature-based dependency injection and prompt formatting.

### Decorator Pattern Support
The implementation supports three primary patterns:
1.  **Static Decoration**: `@my_agent`
2.  **Factory Decoration**: `@my_agent()`
3.  **Inline Instantiation**: `@Agent('model')`

---

## 2. Source Code Changes

> [!IMPORTANT]
> For the complete reference implementation of the helper functions and wrapper logic described below (including exact Python syntax, error handling, and lazy imports), please refer to the `decorator_appendix.md` file adjacent to this document.

### A. Target File: `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

#### 1. Required Imports
Add the following to support runtime introspection and type resolution:
```python
from collections.abc import AsyncGenerator, Generator, Iterable
from functools import wraps
from typing import get_args, get_origin
```

#### 2. `AbstractAgent.__call__` Integration
Implement the entry point within the `AbstractAgent` base class:
```python
def __call__(self, func: Callable[..., Any] | None = None) -> Callable[..., Any]:
    if callable(func):
        return _wrap_agent_function(self, func)
    return lambda f: _wrap_agent_function(self, f)
```

#### 3. Core Logic Helpers (Module Scope)
The logic is encapsulated in three private functions to keep `AbstractAgent` clean.

**`_is_fn_stream(annotation)`**
Uses `get_origin` to detect `Iterable`, `AsyncIterable`, `Generator`, or `AsyncGenerator`. This triggers the use of `run_stream` instead of `run`.

**`_get_fn_base_type(annotation)`**
Normalizes types by unwrapping generic streaming types and extracting the inner content (e.g., `AsyncIterable[str]` -> `str`).

**`_wrap_agent_function(target_agent, func)`**
This is the primary orchestration logic. It performs:
- **Async Detection**: Uses `inspect.iscoroutinefunction` to determine the wrapper type.
- **Output Type Mapping**: Maps the function's return annotation to the agent's `output_type`.
  - *Note:* If streaming, it requests `list[T]` from the engine to facilitate item-by-item yielding via `res.stream_output()`.
  - *Note:* Corrects for `PromptedOutput` and `NativeOutput` by properly wrapping the inner type (e.g., `PromptedOutput[list[T]]`).
- **Instructions Extraction**: Automatically extracts `inspect.getdoc(func)` to serve as per-run instructions.
- **Metadata Stewardship**: Uses `@wraps(func)` and sets an `.__agent__` attribute for tool introspection compatibility.

#### 4. Argument Processing (`_prep_run_context`)
The internal wrapper uses a loop to separate prompt data from context data:
- **Multimodal Bypass**: Media types like `ImageUrl` or `BinaryContent` bypass XML formatting and are sent as raw message parts.
- **Context Injection**:
  - `StepContext`: Extracts `.inputs` for the prompt and `.deps` for forwarding.
  - `RunContext`: Extracts `.deps` and omits the argument from the user prompt.
- **Auto-XML**: All remaining scalar/model arguments are sent to `format_as_xml`.

---

## 3. Test Scenarios & Validation

The implementation should be validated in `tests/test_agent.py` against the following matrix:

| Scenario | Objective | Expected Result |
|---|---|---|
| **Sync/Async Dispatch** | Call sync and async decorated functions. | Correct execution of `run_sync` vs `run`. |
| **Docstring Routing** | Use a detailed docstring on the decorated function. | Docstring appears as `instructions` in model request. |
| **XML Argument Tagging** | Pass multiple scalar arguments (e.g., `x: int, y: int`). | User prompt contains `<x>...</x><y>...</y>`. |
| **Multimodal Passthrough** | Pass an `ImageUrl` object as a parameter. | Media part is passed raw, bypassing XML text. |
| **StepContext Handling** | Decorate a function typed with Pydantic Graph's `StepContext`. | `.inputs` used for prompt; `.deps` forwarded. |
| **RunContext Handling** | Decorate a function typed with `RunContext[Deps]`. | `.deps` used for execution; `.usage` linked if possible. |
| **Output Wrapping** | Annotate with `PromptedOutput[MyModel]`. | Result is the proxy object, not raw data. |
| **Stream Exhaustion** | Use `AsyncIterable[T]` return type. | Items are yielded correctly from `res.stream_output()`. |
| **Empty Prompt Safety** | Call a no-arg, no-docstring decorated function. | Engine-level `UserError` is raised and caught. |

---

## 4. Dependencies & Implementation Nuances

- **Lazy Imports**: Media types (e.g., `ImageUrl`) should be imported lazily inside `_wrap_agent_function` to avoid circular dependency issues with `messages.py`.
- **Tool Compatibility**: The use of `@wraps` ensures that a decorated function can itself be registered as a tool with another agent (e.g., `@other_agent.tool(my_decorated_fn)`).