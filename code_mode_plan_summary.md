## Vendor-Agnostic Code Runtime — Design Decisions Summary

### 1. Two-Object Split: `CodeRuntime` (ABC) + `CodeExecution` (Concrete)

- **`CodeRuntime`** is an abstract base class — each sandbox (Monty, E2B, Modal) subclasses it. This is the *only* abstract class.
- **`CodeExecution`** is a **concrete class** with behavior injected via callables (closures). No subclassing per runtime. Every runtime returns the same `CodeExecution` type.
- **Rationale:** The runtime genuinely differs per sandbox (start a Rust VM vs. spin up a container). But the execution is just a state-machine protocol (pull event, push result, repeat) — the same everywhere. Making it concrete with injected closures eliminates per-runtime subclass boilerplate and makes testing trivial (pass lambdas instead of writing mock classes).

### 2. Pull-Based Interface (not callbacks)

- Consumer drives the loop: `next()` → `FunctionCall | ExecutionResult`, then `provide_result(value)`, repeat.
- **Rationale:** The host needs to inject cross-cutting concerns (tracing, approval/checkpointing, logging) between each tool call. Pull-based gives the host control of the loop. A push-based/callback design would tangle control flow — especially around the approval/checkpoint path where the host needs the execution handle to call `dump()`.

### 3. Handle Pattern: `execute()` returns `CodeExecution`

- `CodeRuntime` is a factory; `CodeExecution` is a stateful handle for one run.
- **Rationale:** Separating the factory from the per-execution state allows multiple concurrent executions and a clear lifecycle (create → drive → exhaust). Same pattern as `open()` → file handle, `cursor()` → DB cursor.

### 4. Union Return Type on `next()`

- `next()` returns `FunctionCall | ExecutionResult` — no separate boolean flags or getter methods.
- **Rationale:** Makes invalid states unrepresentable. The `isinstance` check is the state check — no way to call the wrong method for the wrong state.

### 5. Exception Hierarchy: Three Error Types

- `CodeSyntaxError`, `CodeTypeError`, `CodeRuntimeError` — all subclass `CodeExecutionError`.
- The runtime adapter (e.g., `MontyRuntime`) maps vendor exceptions to these, pre-formatting error messages.
- **Rationale:** Different error categories warrant different LLM retry messages and potentially different retry strategies in the future. The consumer never imports vendor-specific exception types.

### 6. Optional Capabilities via No-Op Defaults

- `type_check()` has a `pass` default — only Monty overrides it.
- `restore()` returns `None` by default — only runtimes with checkpoint support override it.
- `dump()` returns `bytes | None` — `None` means checkpointing not supported.
- **Rationale:** Consumer code stays uniform (`await runtime.type_check(...)` always works). No forced boilerplate stubs in runtimes that lack the capability. EAFP over check-then-act.

### 7. Graceful Checkpoint Degradation (Three Tiers)

- **Tier 1 (Monty):** Full VM serialization — unlimited approval wait time.
- **Tier 2 (Cloud):** Session persistence — bounded by session timeout.
- **Tier 3 (Simple):** No checkpoint — outer approval system re-executes from scratch.
- Same consumer code handles all tiers via `None` checks on `dump()` and `restore()`.

### 8. Monty Becomes an Optional Dependency

- `import monty` removed from `code_mode/__init__.py`. Monty-specific code isolated in `runtime/monty.py` with a `try/except ImportError` guard.
- `monty-python` moves from required deps to optional: `runtime-monty = ["monty-python"]`.
- `runtime=` is a **required** parameter on `CodeModeToolset` — explicit, no hidden default.

### 9. Coupling Surface Is Narrow

- Only `call_tool()` (~15 lines) references Monty. Everything else (signatures, prompts, tracing, sanitization) is already vendor-agnostic. The refactor is surgical.

### 10. Transport (Deferred, Design Documented)

- For cloud sandboxes, tool interception works via HTTP callback: stub functions in the sandbox make blocking HTTP calls to the host, simulating Monty's native pause/resume.
- A `CallbackTransport` abstraction is designed (HTTP, stdin/stdout, provider-native) but **deferred until the first cloud runtime implementation**. Transport is public API (operational concern, not implementation detail).

### File Layout

```
pydantic_ai_slim/pydantic_ai/runtime/
├── __init__.py      # Re-exports all public types
├── abstract.py      # CodeRuntime ABC, CodeExecution concrete, data types, exceptions
└── monty.py         # MontyRuntime adapter (guarded monty import)
```

### Implementation Order

1. `abstract.py` — types, exceptions, `CodeExecution`, `CodeRuntime` ABC
2. `__init__.py` — re-exports
3. `monty.py` — adapter wrapping existing Monty calls
4. Interface + MontyRuntime tests
5. Refactor `code_mode/__init__.py` to use `self.runtime`
6. Move `monty-python` to optional deps
7. Verify: `make test`, `pre-commit run --all-files`, 100% coverage

Steps 1–4 are pure additions (nothing breaks). Step 5 is the refactor. Step 6 is cleanup.
