# Vendor-Agnostic Code Sandbox: A Design Reasoning Walkthrough

This document walks through every design decision from first principles. The goal isn't just to produce a plan — it's to teach you the reasoning framework so you can make similar decisions independently.

---

## Step 0: Before You Abstract Anything — Understand What You Have

The first instinct when someone says "make X vendor-agnostic" is to start designing interfaces. **Resist that instinct.** If you design an abstraction without deeply understanding the concrete thing you're abstracting, you'll create a bad abstraction — one that's either too specific (mirrors the current implementation) or too generic (doesn't capture the essential behavior).

### What to do instead: Map the territory

Ask these questions:
1. What does the current code actually do, step by step?
2. Which parts are specific to the current vendor vs. inherent to the problem?
3. What are the other vendors' APIs? Where do they overlap, where do they diverge?

Let's do this for code mode.

### The current execution flow

Open `pydantic_ai_slim/pydantic_ai/toolsets/code_mode.py` and trace `call_tool()` (lines 226-347). Here's what happens in plain English:

```
1. LLM generates Python code containing tool calls
2. Code is validated (type-checked) before execution
3. Code starts running in a sandbox
4. When code calls get_weather("London"), execution PAUSES
5. We extract the function name and arguments
6. We call the REAL get_weather tool on the host side
7. We feed the result back into the sandbox
8. Code RESUMES from where it paused
9. Steps 4-8 repeat for every tool call in the code
10. When code finishes, we get the final output value
```

Now ask: **which of these steps are "Monty things" vs. "any sandbox things"?**

- Steps 1, 5, 6: Not sandbox-related at all — these are CodeModeToolset's job
- Steps 2: Monty-specific (it has a built-in type checker). Other sandboxes might not.
- Steps 3, 10: Every sandbox must do this (run code, get output). **Common.**
- Steps 4, 7, 8: This is the hard part. Monty can pause/resume natively. Others can't.
- Step 9: Loop structure. **Common** — we always loop over tool calls.

**Key finding: The "pause at function call, yield control, resume" pattern is the essential behavior.** Every sandbox must support it somehow. Monty does it natively. Others will simulate it.

### Exercise: Grep for the coupling

Before designing any abstraction, literally find every line that references the vendor:

```
grep -n "monty" code_mode.py
```

This gives you the **coupling surface** — the exact set of lines that need to change. In our case, it's about 15 lines out of 348 (lines 10, 253, 255, 257, 261, 263, 264, 268, 271, 279, 280, 283, 329, 342, 347).

**Takeaway: The coupling is concentrated in `call_tool()`.** The rest of the file (get_tools, prompt building, signatures) is already vendor-agnostic. This tells you the abstraction boundary is narrow — you're wrapping one method's worth of vendor-specific code.

**General principle: Small coupling surface = simple abstraction. If the coupling were spread across 20 files, you'd need a much bigger redesign.**

---

## Step 1: What Shape Should the Interface Be?

This is the hardest question in abstraction design. Let's reason through it.

### Start from the consumer, not the provider

A common mistake: look at Monty's API and design an interface that mirrors it. This creates a "Monty-shaped hole" that other providers struggle to fit.

**Instead: look at what the CONSUMER (CodeModeToolset) needs to do.** The consumer's algorithm is:

```python
# Pseudocode for what call_tool needs to do, stripped of Monty specifics:

def call_tool(code, tools):
    maybe_type_check(code)          # Optional pre-validation
    execution = start(code, tools)   # Begin running

    while execution.has_pending_call():
        call = execution.get_pending_call()     # What function? What args?
        result = actually_run_tool(call)         # Run the real tool
        execution.feed_result(result)            # Give result back

    return execution.get_output()    # Final value
```

This pseudocode IS the interface. It tells us we need:
- Something that starts execution: `execute(code, tools) -> handle`
- Something that yields events: `handle.next() -> call_or_result`
- Something that feeds results back: `handle.provide_result(value)`
- Optional pre-validation: `type_check(code)`

**General principle: The interface = the consumer's algorithm with vendor-specific types replaced by abstract types.**

### Why not a simpler interface?

You might think: why not just this?

```python
class CodeSandbox:
    async def run(self, code: str, call_tool_callback) -> Any:
        """Run code. When it calls a tool, invoke the callback."""
```

One method. Simple. The sandbox takes a callback, calls it when needed, returns the final output. Why is this worse?

**Reason 1: Who controls the loop?**

With the callback approach, the *sandbox* controls the execution loop. The host just responds to callbacks. This is called **push-based** (the sandbox pushes events to the host).

With our approach, the *host* controls the loop. It pulls events from the sandbox. This is **pull-based**.

Why does this matter? Because the host needs to do things *between* tool calls:

```python
# Things the host does between tool calls:
while isinstance(event, FunctionCall):
    # 1. OpenTelemetry tracing spans
    with tracer.start_span(f'code_mode_tool:{event.function_name}'):
        # 2. Check if this tool needs approval
        try:
            result = await call_tool(event.function_name, ...)
        except ApprovalRequired:
            # 3. Checkpoint the execution state
            checkpoint = execution.dump()
            # 4. Re-raise with checkpoint metadata
            raise ApprovalRequired(metadata={'checkpoint': checkpoint})
    # 5. Resume execution
    await execution.provide_result(result)
    event = await execution.next()
```

With push-based (callback), all this logic must go *inside the callback*. The sandbox calls your callback, and the callback must handle tracing, approval, checkpointing. But the callback doesn't have access to the execution handle — it can't call `dump()` to checkpoint, because it doesn't know about the sandbox's internals.

With pull-based, the host has the execution handle and full control. It can trace, checkpoint, rate-limit, log, or do anything else between steps.

**When to use pull-based:** When the consumer needs to inject cross-cutting concerns (tracing, auth, logging) between events.

**When to use push-based:** When the consumer just needs to react to events without inter-step logic. (e.g., a simple event listener)

**Reason 2: Checkpointing and approval**

This is the killer feature. When a tool needs human approval:

- **Pull-based:** Host receives `FunctionCall` event. Before calling the tool, it checks approval. If not approved, it calls `execution.dump()` to serialize state, then raises `ApprovalRequired` with the checkpoint. Later, it calls `sandbox.restore(checkpoint)` to resume. Clean, linear code.

- **Push-based:** Callback raises `ApprovalRequired`. The sandbox catches it... but wait, should it? The sandbox is in the middle of running code. It needs to serialize its state. But serialization is sandbox-specific (Monty dumps VM state, cloud sandboxes keep the session alive). So the sandbox catches the exception from the callback, serializes, and... re-raises? Returns a special value? The control flow becomes tangled.

**General principle: When your protocol has "pause and resume" semantics, pull-based is almost always cleaner.** Pull naturally models pausing (just stop calling `.next()`). Push requires interrupting the flow mid-callback.

### Why two objects (CodeSandbox + CodeExecution)?

Could we put everything on one class?

```python
class CodeSandbox:
    async def execute(self, code, functions) -> None  # starts it
    async def next(self) -> FunctionCall | ExecutionResult
    async def provide_result(self, value) -> None
```

This works, but has a problem: **the sandbox can only run one execution at a time.** If you call `execute()` again, what happens to the first execution?

By separating `CodeSandbox` (the factory/runtime) from `CodeExecution` (the handle for one run), you enable:
- Multiple concurrent executions (if the sandbox supports it)
- Clear lifecycle: `execute()` → loop → done. The execution handle is garbage collected.
- The sandbox can hold configuration (API keys, timeouts), while the execution holds state (current position, variables)

This is the **Handle pattern**: a factory method creates a stateful handle. You see it everywhere:
- `open(file)` → file handle
- `connect(url)` → connection handle
- `cursor()` → database cursor
- `sandbox.execute(code)` → execution handle

**General principle: When an operation creates ongoing state, return a handle. When the operation is stateless, use the object directly.**

---

## Step 2: Designing Each Interface Method

Now we know the shape: `CodeSandbox` + `CodeExecution`. Let's design each method carefully.

### `CodeExecution.next() -> FunctionCall | ExecutionResult`

**Why a union return type?** Because there are exactly two things that can happen next: either the code calls a function (and we need to process it), or the code finishes (and we have the output). This is a **discriminated union** — the caller checks `isinstance()` to handle each case.

Alternative: separate methods like `has_next_call()`, `get_next_call()`, `get_result()`. But this creates invalid states — what if someone calls `get_result()` before execution is done? The union makes it impossible to misuse.

**General principle: Make invalid states unrepresentable.** A union type that's either A or B is better than two methods where one might not be valid.

### `CodeExecution.dump() -> bytes | None`

**Why `bytes | None` and not always `bytes`?**

Because not all sandboxes can checkpoint. Monty can (it serializes VM state). Cloud sandboxes with session persistence can (they return a session ID as bytes). But some sandboxes might have no checkpoint support at all.

Returning `None` means "I can't checkpoint." The consumer checks for `None` and handles it:

```python
checkpoint = execution.dump()
if checkpoint is None:
    # Can't checkpoint — approval must re-execute from scratch
    raise ApprovalRequired(metadata={'no_checkpoint': True})
else:
    raise ApprovalRequired(metadata={'checkpoint': checkpoint})
```

**Why not a separate `supports_checkpointing() -> bool` method?** Because it's redundant — `dump()` returning `None` already tells you. And it avoids the "check-then-act" race condition (what if support changes between check and dump?).

**General principle: Prefer "try and handle failure" over "check then act" for capability queries.** The Python community calls this EAFP (Easier to Ask Forgiveness than Permission).

### `CodeSandbox.type_check(code, prefix_code) -> None`

**Why is this optional (default no-op)?**

Monty has a built-in type checker. E2B doesn't. We could:
1. Make it abstract (force every sandbox to implement it) — but cloud sandboxes would just `pass`
2. Make it optional with a no-op default — sandboxes that have type checking override it
3. Remove it from the interface — type checking is a "Monty thing"

Option 2 is best because:
- It acknowledges that type checking is valuable but not universal
- The consumer code is the same regardless (`await sandbox.type_check(code, prefix)` — if it's a no-op, nothing happens)
- A future sandbox might integrate mypy or pyright — the interface supports it without changes

**General principle: For optional capabilities, use a no-op default method rather than forcing implementation or removing the concept.** This lets the consumer code be uniform while implementations opt in.

### `CodeSandbox.restore(checkpoint_data) -> CodeExecution | None`

**Same pattern as `dump()`** — `None` means "I can't restore from this." This handles:
- Sandbox doesn't support checkpointing at all → always returns `None`
- Checkpoint data is corrupted or expired → returns `None`
- Sandbox type mismatch (trying to restore a Monty checkpoint in E2B) → returns `None`

### Error types: Why three exception classes?

```python
class CodeExecutionError(Exception): ...  # Base
class CodeSyntaxError(CodeExecutionError): ...
class CodeTypeError(CodeExecutionError): ...
class CodeRuntimeError(CodeExecutionError): ...
```

The consumer (CodeModeToolset) handles all three the same way — `raise ModelRetry(message)`. So why distinguish them?

1. **Different error messages** for the LLM: "Syntax error" tells the LLM to fix syntax. "Type error" tells it to fix types. "Runtime error" might mean logic is wrong.
2. **Different retry strategies** in the future: Syntax/type errors are fixable by the LLM. Runtime errors might indicate a deeper problem.
3. **Debugging/logging**: Knowing the error category helps developers understand what went wrong.

**General principle: Use an exception hierarchy when different error types need different handling — even if current handling is the same, future handling might differ.**

---

## Step 3: The Hardest Problem — Function Call Interception

This deserves its own section because it's the core technical challenge that makes vendor-agnostic sandboxing non-trivial.

### What Monty does (and why it's special)

Monty is a **custom Python interpreter written in Rust**. It doesn't run "real" CPython — it interprets a restricted Python subset. Because it *is* the interpreter, it can:

1. Hit `get_weather(city="London")` in the code
2. Recognize it's an "external function" (not defined in the code)
3. **Freeze the entire VM state** — call stack, local variables, instruction pointer
4. Return a `MontySnapshot` object to the host with the function name and args
5. Later, accept a return value and **unfreeze**, resuming from exactly where it stopped

This is like Python's `yield` statement, but at the VM level. Monty's execution is essentially a generator:

```
Host: snapshot = monty.start()
      # snapshot says: "code wants to call get_weather(city='London')"
      result = call_real_tool("get_weather", city="London")
      snapshot = snapshot.resume(return_value=result)
      # snapshot says: "code wants to call get_forecast(...)"
      ...
```

**Why this is unique:** No other sandbox can do this. E2B, Modal, Daytona — they all run real CPython. You can't pause CPython mid-execution and serialize its state (without OS-level process freezing, which is fragile and non-portable).

### How to simulate interception in cloud sandboxes

If the sandbox runs real Python and can't pause it, we need the Python code *itself* to pause. How?

**The trick: Make the tool functions block on I/O.**

Instead of the sandbox having "real" `get_weather()`, we give it a *fake* `get_weather()` that makes a network call:

```python
# Code the sandbox actually runs (auto-generated wrapper + user code):

import urllib.request, json

def get_weather(*, city: str):
    # This blocks until the host responds
    resp = urllib.request.urlopen(
        urllib.request.Request("http://host:9876/call",
            data=json.dumps({"fn": "get_weather", "kwargs": {"city": city}}).encode(),
            headers={"Content-Type": "application/json"})
    )
    return json.loads(resp.read())["result"]

# --- LLM's code (runs in sandbox) ---
weather = get_weather(city="London")  # Blocks on HTTP, host runs real tool, returns result
```

On the host side:
```
1. Start HTTP server on port 9876
2. Send wrapper_code + user_code to sandbox
3. Sandbox runs code, hits get_weather(), makes HTTP request
4. Host HTTP server receives request → this is CodeExecution.next() returning FunctionCall
5. Host calls real tool → get_weather returns {"temp": 20}
6. Host sends HTTP response → this is CodeExecution.provide_result()
7. Sandbox receives response, continues executing
8. If code calls another tool, goto 3
9. When code finishes, sandbox sends a "completion" HTTP request with the output
```

**The execution timeline is identical to Monty's — just the pause/resume mechanism differs:**

```
Monty:  run → call get_weather → VM PAUSES → host runs tool → VM RESUMES → continue
HTTP:   run → call get_weather → HTTP BLOCKS → host runs tool → HTTP UNBLOCKS → continue
```

From `CodeExecution`'s perspective, both look the same: `next()` returns a `FunctionCall`, `provide_result()` sends the result back.

### The Callback Transport abstraction

The HTTP approach works when the sandbox can reach the host. But what about:
- **NAT/firewall**: Host is behind a router, sandbox is in the cloud — can't connect
- **Subprocess sandboxes**: Sandbox is a local process — HTTP is overkill
- **Provider-specific channels**: E2B has built-in file/stream APIs

So we abstract the *communication channel* between sandbox and host:

```python
class CallbackTransport(ABC):
    """How sandbox communicates function calls to the host."""

    async def start(self, available_functions: list[str]) -> TransportConfig:
        """Start the channel. Returns config for the sandbox side."""
        # TransportConfig includes:
        # - stub_code: Python code to inject into sandbox (defines fake functions)
        # - env_vars: environment variables the sandbox needs

    async def wait_for_call(self) -> FunctionCall:
        """Wait for the sandbox to call a function."""

    async def send_result(self, value: Any) -> None:
        """Send the function result back to the sandbox."""

    async def wait_for_completion(self) -> ExecutionResult:
        """Wait for sandbox code to finish."""

    async def stop(self) -> None:
        """Shut down the channel."""
```

**Implementations:**

| Transport | Mechanism | Use when |
|---|---|---|
| `HttpCallbackTransport` | HTTP server on host, HTTP requests from sandbox | Sandbox has outbound HTTP to host |
| `StdinStdoutTransport` | JSON over process stdin/stdout | Local subprocess sandbox |
| Provider-native | E2B filesystem, Modal streams, etc. | Provider offers better channels |

Each cloud sandbox implementation (`E2BSandbox`, `ModalSandbox`, etc.) picks a default transport but lets the user override:

```python
# Default: uses HTTP callback
sandbox = E2BSandbox(api_key="...")

# Override: user behind NAT, use provider's filesystem API instead
sandbox = E2BSandbox(api_key="...", transport=E2BFileTransport())
```

### Why transport is public API, not internal

You decided this should be public. Here's the reasoning formalized:

**Internal** would mean: users can't see or change the transport. The sandbox picks one, end of story. Simpler API surface, fewer things to document.

**Public** means: users can construct their own transport and pass it in. They can debug network issues by swapping transports. They can write custom transports for unusual environments.

**The deciding factor: Is transport choice an "implementation detail" or an "operational concern"?**

It's operational. When your sandbox can't reach your host because of a firewall, that's YOUR problem to solve, not a bug in the library. The library should give you the tools to solve it — not hide the transport behind a wall.

**General principle: Make configuration public when the user might legitimately need to change it for operational reasons (networking, auth, performance). Keep it internal when it's truly an implementation detail that users would never need to touch.**

---

## Step 4: Making Capabilities Discoverable — CodeConstraints

### The problem

Monty only supports a restricted Python subset: no imports, no while loops, no comprehensions, etc. Cloud sandboxes support full Python. The LLM needs to know what it can write.

Currently, the restrictions are hardcoded in the prompt:
```
CRITICAL Syntax restrictions:
- No imports
- No while loops - use for loops instead
...
```

If we switch to E2B, these restrictions become wrong — E2B supports everything.

### The design question: How do sandboxes declare their capabilities?

**Option A: Free-form string**
```python
class CodeSandbox:
    def get_prompt_instructions(self) -> str:
        return "No imports. No while loops. ..."
```

Problem: The prompt builder can't make *decisions* based on this. It's just text to paste in. What if you want different example code for restricted vs. full Python? You'd have to parse the string.

**Option B: Boolean flags**
```python
@dataclass
class CodeConstraints:
    supports_imports: bool = True
    supports_while_loops: bool = True
    supports_comprehensions: bool = True
    # ... etc
```

Now the prompt builder can branch:
```python
if constraints.supports_comprehensions:
    example = "[x for x in items if x > 0]"
else:
    example = "results = []\nfor x in items:\n    if x > 0:\n        results.append(x)"
```

**Option C: Enum of capability levels**
```python
class SandboxLevel(Enum):
    RESTRICTED = "restricted"   # Monty-like
    STANDARD = "standard"       # Full Python, no system access
    FULL = "full"               # Full Python with system access
```

Simpler, but too coarse. What if a sandbox supports imports but not async? You'd need levels like "RESTRICTED_PLUS_IMPORTS" — combinatorial explosion.

**Decision: Option B (boolean flags).** Most granular, most useful for prompt building. Each flag maps to one prompt restriction. The `additional_instructions` field handles anything not covered by the flags.

**General principle: Use structured capability flags when the consumer needs to make branching decisions based on capabilities. Use free-form text only for supplementary information.**

### Why defaults are `True` (full Python), not `False` (no support)

```python
@dataclass
class CodeConstraints:
    supports_imports: bool = True       # Default: supported
    supports_while_loops: bool = True   # Default: supported
    ...
```

This means: **new sandboxes support everything by default.** Only restrictive sandboxes (Monty) override to `False`.

Why not default `False`? Because most sandboxes ARE full Python. E2B, Modal, Daytona, Cloudflare — they all support everything. Only Monty is restricted. Defaulting to `True` means most implementations don't need to touch `CodeConstraints` at all.

**General principle: Default to the common case.** If 6 out of 7 implementations have the same value, that should be the default.

---

## Step 5: Checkpoint/Approval — Designing for Graceful Degradation

### The spectrum of checkpoint support

Not all sandboxes checkpoint equally. Rather than demanding the "best" from everyone, we design for three tiers:

**Tier 1: Full serialization (Monty)**
- `dump()` → bytes containing entire VM state
- `restore(bytes)` → exact resumption
- Sandbox can be destroyed — state survives independently
- Approval wait: unlimited (state is just bytes)

**Tier 2: Session persistence (E2B, Daytona)**
- `dump()` → bytes containing session ID
- `restore(session_id)` → reconnects to the still-running sandbox
- Sandbox must stay alive — the HTTP response is held open
- Approval wait: bounded by session timeout (1-24 hours)

**Tier 3: No checkpoint (simple sandboxes)**
- `dump()` → `None`
- Can't resume — must re-execute from scratch if needed
- Approval: the approval toolset around the *outer* `run_code` tool handles it at a coarser level

### How CodeModeToolset handles each tier

```python
# In call_tool(), when approval is needed:
except ApprovalRequired as e:
    checkpoint = execution.dump()

    if checkpoint is not None:
        # Tier 1 or 2: checkpoint available
        raise ApprovalRequired(metadata={
            'code_mode': {
                'checkpoint_data': checkpoint,
                'tool_name': event.function_name,
                'tool_args': tool_kwargs,
            },
            'original_metadata': e.metadata,
            '_approval_call': {
                'tool_name': event.function_name,
                'args': tool_kwargs,
            },
        })
    else:
        # Tier 3: no checkpoint — propagate approval without code_mode metadata
        # The approval will be handled at the outer level (re-executing run_code entirely)
        raise
```

**The key insight: The SAME code handles all three tiers.** The `if checkpoint is not None` branch is the only conditional. This is graceful degradation — the interface is the same, the behavior adapts.

**General principle: Design the main interface for the best case. Handle lesser cases with `None`/sentinel checks, not with separate code paths for each tier.**

---

## Step 6: Making Monty Optional — The Dependency Design

### The problem with `import monty` at module level

Line 10 of `code_mode.py`: `import monty`

This means: **every Python process that imports CodeModeToolset must have monty-python installed.** Even if you're using E2B and never touch Monty.

This is a transitive dependency problem. If module A imports module B which imports monty, then A needs monty too.

### The lazy import pattern

Move Monty-specific code into its own module with a guarded import:

```python
# sandboxes/monty.py
try:
    import monty as _monty
except ImportError:
    raise ImportError(
        "MontySandbox requires 'monty-python'. "
        "Install with: pip install 'pydantic-ai[sandbox-monty]'"
    )
```

Now:
- `from pydantic_ai.toolsets.code_mode import CodeModeToolset` — works without monty
- `from pydantic_ai.sandboxes.monty import MontySandbox` — fails with clear message if monty not installed
- `CodeModeToolset(wrapped=toolset, sandbox=MontySandbox())` — only this path needs monty

This is the same pattern PydanticAI uses for model providers (`from pydantic_ai.models.openai import OpenAIModel` requires the `openai` package).

### Making `sandbox=` required

Since Monty is no longer a default dependency, we can't default to it. We make `sandbox=` a required parameter:

```python
@dataclass(kw_only=True)
class CodeModeToolset(WrapperToolset[AgentDepsT]):
    sandbox: CodeSandbox  # Required — no default
```

Since the API hasn't shipped yet, this isn't a breaking change — it's just the API design:
- **Explicit > implicit**: The user knows which sandbox they're using
- **No surprise dependencies**: No hidden attempt to import monty
- **Clear error messages**: Python itself says "missing required argument 'sandbox'"

---

## Step 7: Putting It All Together — File-by-File Plan

### New files to create

**`pydantic_ai_slim/pydantic_ai/sandboxes/__init__.py`**
- Abstract types: `CodeSandbox`, `CodeExecution`, `CodeConstraints`
- Data types: `FunctionCall`, `ExecutionResult`
- Error types: `CodeExecutionError`, `CodeSyntaxError`, `CodeTypeError`, `CodeRuntimeError`
- No provider-specific code, no imports of monty/e2b/etc.

**`pydantic_ai_slim/pydantic_ai/sandboxes/monty.py`**
- `MontySandbox(CodeSandbox)` — the Adapter for Monty
- `MontyExecution(CodeExecution)` — wraps MontySnapshot/MontyComplete iteration
- Maps: `MontyTypingError → CodeTypeError`, `MontySyntaxError → CodeSyntaxError`, `MontyRuntimeError → CodeRuntimeError`
- Lazy import of `monty` with helpful error message

**`pydantic_ai_slim/pydantic_ai/sandboxes/transports/__init__.py`** (public)
- `CallbackTransport` abstract base class
- `TransportConfig` dataclass
- Re-exports from submodules

**`pydantic_ai_slim/pydantic_ai/sandboxes/transports/http.py`**
- `HttpCallbackTransport(CallbackTransport)` — asyncio HTTP server
- Auth token generation and validation

**`pydantic_ai_slim/pydantic_ai/sandboxes/transports/stdio.py`**
- `StdinStdoutTransport(CallbackTransport)` — JSON protocol over stdin/stdout

**`pydantic_ai_slim/pydantic_ai/sandboxes/_stub_generator.py`**
- Generates Python wrapper code (the fake function stubs for cloud sandboxes)
- Used by transports to create sandbox-side code

### Files to modify

**`pydantic_ai_slim/pydantic_ai/toolsets/code_mode.py`**
- Remove `import monty`
- Change `CodeModeToolset` to accept `sandbox: CodeSandbox` (required)
- Rewrite `call_tool()` to use `CodeSandbox` / `CodeExecution` interface
- Update `build_code_mode_prompt()` signature to accept `CodeConstraints`
- Make `_build_type_check_prefix()` conditional on sandbox supporting type checking
- Keep `_find_await_expressions()` (it's a general Python AST check, not Monty-specific)

**`pydantic_ai_slim/pyproject.toml`**
- Move `monty-python` from required deps to optional: `sandbox-monty = ["monty-python"]`

### Test files

**`tests/code_mode/test_sandbox_interface.py`**
- Mock `TestSandbox` and `TestExecution` implementations
- Test the execution loop, checkpoint/restore, error propagation
- Test prompt generation with different `CodeConstraints` values

**`tests/code_mode/test_monty_sandbox.py`**
- Test `MontySandbox` + `MontyExecution` specifically
- Migrate relevant tests from existing code mode test files

---

## Step 8: How The Refactored `call_tool()` Looks

### Before (coupled to Monty):

```python
async def call_tool(self, name, tool_args, ctx, tool):
    code = tool_args['code']
    checkpoint = ctx.tool_call_metadata.get('code_mode') if ctx.tool_call_approved else None

    m = monty.Monty(code, external_functions=list(tool.original_tools.keys()))
    result = None

    if checkpoint:
        result = monty.MontySnapshot.load(checkpoint['checkpoint_dump'])
    else:
        m.type_check(prefix_code=prefix)
        result = m.start()

    while isinstance(result, monty.MontySnapshot):
        tool_name = result.function_name
        tool_kwargs = dict(result.kwargs)
        # ... (args handling, approval, tracing)
        tool_return = await super().call_tool(tool_name, tool_kwargs, inner_ctx, original_tool)
        result = result.resume(return_value=tool_return)

    return result.output
```

### After (provider-agnostic):

```python
async def call_tool(self, name, tool_args, ctx, tool):
    code = tool_args['code']
    sandbox = self.sandbox
    functions = list(tool.original_tools.keys())
    checkpoint = ctx.tool_call_metadata.get('code_mode') if ctx.tool_call_approved else None

    # Type check — sandbox decides if this does anything
    try:
        prefix = _build_type_check_prefix(self._cached_signatures)
        await sandbox.type_check(code, prefix)
    except CodeTypeError as e:
        raise ModelRetry(f'Type error in generated code:\n{e.message}')
    except CodeSyntaxError as e:
        raise ModelRetry(f'Syntax error in generated code:\n{e.message}')

    # Start or restore execution
    try:
        if checkpoint:
            execution = await sandbox.restore(checkpoint['checkpoint_data'])
            if execution is None:
                execution = await sandbox.execute(code, functions)
        else:
            execution = await sandbox.execute(code, functions)
    except CodeSyntaxError as e:
        raise ModelRetry(f'Syntax error in generated code:\n{e.message}')
    except CodeRuntimeError as e:
        raise ModelRetry(f'Runtime error in generated code:\n{e.message}')

    next_call_approved = checkpoint is not None

    # Drive the execution loop — identical regardless of sandbox provider
    try:
        event = await execution.next()
        while isinstance(event, FunctionCall):
            original_tool = tool.original_tools[event.function_name]
            tool_kwargs = dict(event.kwargs)
            if event.args:
                param_names = list(original_tool.tool_def.parameters_json_schema.get('properties', {}).keys())
                for i, arg in enumerate(event.args):
                    if i < len(param_names):
                        tool_kwargs[param_names[i]] = arg

            inner_ctx = replace(ctx, tool_name=event.function_name, tool_call_approved=next_call_approved, ...)
            next_call_approved = False

            try:
                with ctx.tracer.start_as_current_span(f'code_mode_tool:{event.function_name}', ...):
                    tool_return = await super().call_tool(event.function_name, tool_kwargs, inner_ctx, original_tool)
            except ApprovalRequired as e:
                checkpoint_data = execution.dump()
                raise ApprovalRequired(metadata={
                    'code_mode': {
                        'checkpoint_data': checkpoint_data,
                        'tool_name': event.function_name,
                        'tool_args': tool_kwargs,
                    },
                    'original_metadata': e.metadata,
                    '_approval_call': {'tool_name': event.function_name, 'args': tool_kwargs},
                })

            await execution.provide_result(tool_return)
            event = await execution.next()

    except CodeRuntimeError as e:
        raise ModelRetry(f'Runtime error in generated code:\n{e.message}')

    assert isinstance(event, ExecutionResult)
    return event.output
```

**What changed:** Every `monty.X` reference is replaced with `CodeSandbox`/`CodeExecution` method calls. The algorithm is the same. The tracing, approval, positional-args handling — all preserved exactly.

---

## Step 9: Dynamic Prompt Generation

### Current (hardcoded for Monty):

```python
def build_code_mode_prompt(*, signatures: list[str]) -> str:
    return f"""
    CRITICAL Syntax restrictions:
    - No imports
    - No while loops - use for loops instead
    - No comprehensions
    ...
    """
```

### New (driven by CodeConstraints):

```python
def build_code_mode_prompt(*, signatures: list[str], constraints: CodeConstraints) -> str:
    # Build restriction list dynamically
    restrictions = []
    if not constraints.supports_imports:
        restrictions.append("- No imports - use only the provided functions and builtins")
    if not constraints.supports_while_loops:
        restrictions.append("- No while loops - use for loops instead")
    if not constraints.supports_comprehensions:
        restrictions.append("- No comprehensions (list/dict/set) or generator expressions - use explicit for loops")
    if not constraints.supports_lambdas:
        restrictions.append("- No lambdas - define logic inline")
    if not constraints.supports_tuple_unpacking:
        restrictions.append("- No tuple unpacking (e.g., `a, b = 1, 2`) - assign variables separately")
    if not constraints.supports_indexing_slicing:
        restrictions.append("- No list indexing or slicing (e.g., `lst[0]`, `lst[:10]`) - use for loops to iterate")
    if not constraints.supports_break_continue:
        restrictions.append("- No break or continue statements - use conditional logic instead")
    if not constraints.supports_string_methods:
        restrictions.append("- No string methods (.join, .split, .upper, etc.) - return data structures, not formatted strings")

    if restrictions:
        syntax_section = "CRITICAL Syntax restrictions (the runtime uses a restricted Python subset):\n" + "\n".join(restrictions)
    else:
        syntax_section = "Full Python syntax is supported including imports, comprehensions, string methods, and all standard library features."

    if constraints.additional_instructions:
        syntax_section += "\n\n" + constraints.additional_instructions

    # Rest of prompt (function signatures, examples, etc.) stays the same
    ...
```

**Result:** Monty sees all restrictions (no change). E2B sees "Full Python syntax is supported." Custom sandboxes get exactly the restrictions they declare.

---

## Step 10: Verification — How to Know It Works

1. **All existing tests pass** — the MontySandbox adapter must produce identical behavior to the current direct Monty usage. Run `make test`.
2. **New interface tests** — a `TestSandbox` mock verifies the execution loop, checkpoint/restore, and error handling without any real sandbox.
3. **Prompt tests** — verify that `build_code_mode_prompt(constraints=...)` generates correct text for both restricted and full-Python constraints.
4. **Import tests** — verify that `from pydantic_ai.toolsets.code_mode import CodeModeToolset` works without monty installed. Verify that `from pydantic_ai.sandboxes.monty import MontySandbox` fails with a clear message without monty.
5. **Coverage** — maintain 100% coverage per project requirements.
6. **Linting** — `pre-commit run --all-files` passes.

---

## Step 11: What We're NOT Doing (Scope Boundary)

- **No cloud provider implementations yet** — just the abstraction + Monty adapter
- **No actual HTTP callback server yet** — just the transport interface (cloud providers will need it later)
- **No changes to tool registration** — FunctionToolset, MCP servers, etc. stay the same
- **No changes to signature generation** — `_signature_from_schema.py` is untouched
- **No changes to the approval UI** — DeferredToolRequests/ToolApproved stay the same
- **No package management interface** — installing packages in cloud sandboxes is provider-specific config

---

## Summary of Design Principles Used

| Principle | Where Applied |
|---|---|
| Find all coupling points before abstracting | Step 0: grep for `monty` |
| Don't abstract what's already decoupled | Step 0: signatures, prompts, tracing are fine |
| Derive interface from consumer's algorithm | Step 1: call_tool's pseudocode → interface |
| Prefer pull-based for consumer control | Step 1: next()/provide_result() vs callbacks |
| Handle pattern for stateful operations | Step 1: execute() → CodeExecution |
| Make invalid states unrepresentable | Step 2: union return types |
| EAFP over check-then-act | Step 2: dump() returns None vs supports_checkpointing() |
| No-op default for optional capabilities | Step 2: type_check() default pass |
| Exception hierarchy for different error handling | Step 2: three error types |
| Structured data > unstructured text | Step 4: CodeConstraints booleans vs string |
| Default to the common case | Step 4: True defaults (most sandboxes support everything) |
| Design for best case, degrade gracefully | Step 5: checkpoint tiers |
| Public API for operational concerns | Step 3: transport is public |
| Lazy imports for optional deps | Step 6: try/except ImportError |
| Explicit > implicit | Step 6: sandbox= required parameter |

---
---

# Appendix A: HTTP Callback Transport — A Mechanical Deep Dive

The original document describes the HTTP callback at a high level. This appendix makes it fully concrete: what bytes go over the wire, what code runs where, what happens when things go wrong, and how the pieces map back to the `CodeExecution` interface.

## The Core Problem, Restated

Monty is a Rust interpreter. When code calls `get_weather(city="London")`, Monty literally stops its instruction pointer, packages up the call as a Python object, and hands it back to the host. The host is in the same process — it's a function call away.

Cloud sandboxes (E2B, Modal, Daytona) run **real CPython in a remote VM**. The host and sandbox are in **different processes on different machines**. The sandbox's CPython cannot be paused from outside. We need a way to make the sandbox "pause itself" when it hits a tool call, tell the host what it wants, wait for the answer, and continue.

The only mechanism CPython has for "pause and wait" is **blocking I/O**. If you call `socket.recv()` or `urllib.request.urlopen()`, the thread blocks until data arrives. This is our pause mechanism.

## The Two Sides

There are two programs running simultaneously:

**Host side** (your machine, the pydantic-ai process):
- Runs `CodeModeToolset.call_tool()`
- Has access to real tools (`get_weather`, `send_email`, etc.)
- Drives the `CodeExecution` interface (calls `next()`, `provide_result()`)

**Sandbox side** (remote VM in E2B/Modal/etc.):
- Runs the LLM-generated Python code
- Has **fake** tool functions that phone home to the host instead of doing real work
- Has no idea it's being intercepted — the code looks normal

## What the Sandbox Actually Runs

The LLM writes this code:
```python
weather = get_weather(city="London")
forecast = get_forecast(city="London", days=3)
{"weather": weather, "forecast": forecast}
```

But the sandbox doesn't run this directly. The transport **prepends auto-generated stub code** that defines fake versions of each tool. The sandbox actually runs:

```python
# ===== AUTO-GENERATED TRANSPORT PREAMBLE =====
import urllib.request
import json as _json
import sys as _sys

_CALLBACK_URL = "http://203.0.113.50:9876"  # Host's callback server
_AUTH_TOKEN = "a1b2c3d4e5f6..."              # One-time token for this execution

def _call_host(fn_name, args, kwargs):
    """Send a function call to the host and block until we get the result."""
    payload = _json.dumps({
        "type": "function_call",
        "function_name": fn_name,
        "args": args,
        "kwargs": kwargs,
    }).encode("utf-8")

    req = urllib.request.Request(
        _CALLBACK_URL + "/call",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_AUTH_TOKEN}",
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=300)  # 5min timeout per call
    body = _json.loads(resp.read())

    if body.get("error"):
        raise RuntimeError(body["error"])
    return body["result"]

def get_weather(*, city: str):
    return _call_host("get_weather", [], {"city": city})

def get_forecast(*, city: str, days: int = 7):
    return _call_host("get_forecast", [], {"city": city, "days": days})

# ===== END PREAMBLE =====

# ===== LLM-GENERATED CODE (untouched) =====
weather = get_weather(city="London")
forecast = get_forecast(city="London", days=3)

# ===== AUTO-GENERATED EPILOGUE =====
# Send the final result back to host
_result_payload = _json.dumps({
    "type": "execution_complete",
    "output": {"weather": weather, "forecast": forecast},
}).encode("utf-8")
_result_req = urllib.request.Request(
    _CALLBACK_URL + "/complete",
    data=_result_payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_AUTH_TOKEN}",
    },
    method="POST",
)
urllib.request.urlopen(_result_req)
```

**Key observations:**

1. Each stub function (`get_weather`, `get_forecast`) has the **same signature** as the real tool. The LLM's code doesn't know the difference.
2. Every stub calls `_call_host()`, which makes an HTTP POST and **blocks** until the host responds.
3. The epilogue captures the code's final expression and sends it to a `/complete` endpoint.
4. The preamble only uses Python standard library (`urllib.request`, `json`). No pip installs needed in the sandbox.

## What the Host Runs

The host side is an `asyncio` HTTP server embedded inside the `CodeExecution` implementation. Here's the mechanical flow:

```python
class HttpCallbackExecution(CodeExecution):
    """One execution session over HTTP callback."""

    def __init__(self):
        # asyncio primitives for the host↔sandbox rendezvous
        self._pending_call: asyncio.Future[FunctionCall] = asyncio.get_event_loop().create_future()
        self._pending_result: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._completion: asyncio.Future[ExecutionResult] = asyncio.get_event_loop().create_future()

    async def next(self) -> FunctionCall | ExecutionResult:
        # Race: either the sandbox calls a function, or the sandbox completes.
        # Whichever happens first is what we return.
        done, _ = await asyncio.wait(
            [self._pending_call, self._completion],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self._pending_call in done:
            return self._pending_call.result()
        else:
            return self._completion.result()

    async def provide_result(self, value: Any) -> None:
        # Unblock the sandbox's HTTP request by resolving the future.
        # The HTTP handler is awaiting this future — when it resolves,
        # the handler sends the HTTP response, which unblocks the sandbox.
        self._pending_result.set_result(value)

        # Prepare a fresh future for the next call
        self._pending_call = asyncio.get_event_loop().create_future()
        self._pending_result = asyncio.get_event_loop().create_future()
```

And the HTTP handler that bridges the two:

```python
async def _handle_request(self, request):
    body = await request.json()

    if body["type"] == "function_call":
        # Sandbox is calling a tool. Unblock next() by resolving _pending_call.
        call = FunctionCall(
            function_name=body["function_name"],
            args=tuple(body["args"]),
            kwargs=body["kwargs"],
        )
        self._execution._pending_call.set_result(call)

        # Now BLOCK this HTTP response until the host provides a result.
        # The host will call provide_result(), which sets _pending_result.
        result = await self._execution._pending_result
        return JSONResponse({"result": result})

    elif body["type"] == "execution_complete":
        self._execution._completion.set_result(
            ExecutionResult(output=body["output"])
        )
        return JSONResponse({"ok": True})
```

## Step-by-Step Timeline

Here's what happens with the weather example, second by second. The left column is the host process. The right column is the sandbox process. Time flows downward. Indentation shows blocking.

```
HOST (your machine)                          SANDBOX (remote VM)
═══════════════════                          ══════════════════
                                             [sandbox starts running preamble + LLM code]
                                             weather = get_weather(city="London")
                                               → _call_host("get_weather", [], {"city": "London"})
                                               → POST http://host:9876/call
                                                 {"type":"function_call",
                                                  "function_name":"get_weather",
                                                  "kwargs":{"city":"London"}}
                                               → [BLOCKS waiting for HTTP response]

[HTTP server receives POST /call]
[_pending_call.set_result(FunctionCall(...))]
                                              │
execution.next() returns FunctionCall         │ (still blocked)
  function_name="get_weather"                 │
  kwargs={"city": "London"}                   │
                                              │
[Host calls real get_weather tool]            │
  → returns {"temp": 20, "wind": "5mph"}     │
                                              │
execution.provide_result({"temp":20,...})      │
  → _pending_result.set_result(...)           │
  → HTTP handler resumes, sends response:     │
    {"result": {"temp": 20, "wind": "5mph"}}  │
                                              │
                                             [HTTP response arrives]
                                             [_call_host returns {"temp":20,"wind":"5mph"}]
                                             weather = {"temp": 20, "wind": "5mph"}

                                             forecast = get_forecast(city="London", days=3)
                                               → _call_host("get_forecast", [], {"city":"London","days":3})
                                               → POST http://host:9876/call
                                               → [BLOCKS again]

[HTTP server receives POST /call]
execution.next() returns FunctionCall
  function_name="get_forecast"
  kwargs={"city": "London", "days": 3}

[Host calls real get_forecast tool]
  → returns [{"day":"Mon","temp":18}, ...]

execution.provide_result([...])
  → HTTP response sent

                                             [HTTP response arrives]
                                             forecast = [{"day":"Mon","temp":18}, ...]

                                             [Code reaches final expression]
                                             → POST http://host:9876/complete
                                               {"type":"execution_complete",
                                                "output":{"weather":{...},"forecast":[...]}}

[HTTP server receives POST /complete]
[_completion.set_result(ExecutionResult(...))]

execution.next() returns ExecutionResult
  output={"weather": {...}, "forecast": [...]}

[Done. Host returns output to CodeModeToolset.]
```

## The Rendezvous Pattern

The HTTP callback is a **rendezvous** — two concurrent processes meet at a synchronization point. The mechanism is:

```
Sandbox thread:          HTTP POST ──────────────┐
                         (blocks on recv)         │
                                                  ▼
Host asyncio loop:       [receives request] ──→ resolves _pending_call
                         [calls next()] ──→ returns FunctionCall to host code
                         [host does work] ──→ calls provide_result()
                         [resolves _pending_result]
                                                  │
                                                  ▼
                         [HTTP handler resumes] ──→ sends HTTP response
                                                  │
Sandbox thread:          [recv returns] ──────────┘
                         (continues executing)
```

The `asyncio.Future` is the synchronization primitive. It's the in-process equivalent of a condition variable. The HTTP request/response is the wire protocol that carries the rendezvous across process boundaries.

**Why `asyncio.Future` specifically?** Because the host is an async Python program. `next()` and `provide_result()` are async methods. The HTTP server (`aiohttp` or similar) is async. Everything runs on the same event loop. The `Future` lets the HTTP handler "park" while waiting for the host to provide a result — without blocking any OS threads.

## Why This Is Equivalent to Monty's Native Pause

The `CodeExecution` interface abstracts over both mechanisms. From `CodeModeToolset.call_tool()`'s perspective, it looks identical:

```python
# This code is IDENTICAL whether sandbox is Monty or HTTP-based:
event = await execution.next()               # "What happened?"
while isinstance(event, FunctionCall):        # "It called a function"
    result = await run_real_tool(event)        # "Run the real tool"
    await execution.provide_result(result)     # "Here's the answer"
    event = await execution.next()             # "What happened next?"
return event.output                           # "Code finished"
```

Under the hood:
- **Monty**: `next()` calls `snapshot.resume()` synchronously. No I/O, no network. The Rust VM does the work in-process.
- **HTTP**: `next()` awaits an `asyncio.Future` that resolves when the sandbox's HTTP POST arrives. Network I/O between machines.

Same interface. Different physics.

## Edge Cases and Error Handling

### Sandbox code raises an exception

```python
# LLM writes buggy code:
x = 1 / 0  # ZeroDivisionError
```

The sandbox's CPython catches this. The epilogue code doesn't run (the exception kills the script). Instead, the cloud sandbox provider reports the execution failed. How this gets communicated depends on the provider:

- **E2B**: The `execute()` API returns an error object
- **Modal**: The function raises an exception that propagates to the caller

The `CloudSandboxExecution` (provider-specific implementation) catches this and raises `CodeRuntimeError`, which `CodeModeToolset` converts to `ModelRetry`.

But wait — the host's HTTP server is still waiting on `_pending_call` or `_completion`. If the sandbox dies without sending anything, those futures never resolve. Solution: the transport sets a **timeout** on the futures and the cloud sandbox provider has an execution status API to detect the failure:

```python
async def next(self) -> FunctionCall | ExecutionResult:
    try:
        done, _ = await asyncio.wait(
            [self._pending_call, self._completion, self._sandbox_error],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=self._timeout,
        )
    except asyncio.TimeoutError:
        # Check sandbox status via provider API
        status = await self._provider.get_execution_status(self._session_id)
        if status.failed:
            raise CodeRuntimeError(status.error_message)
        raise CodeRuntimeError("Sandbox execution timed out")

    if self._sandbox_error in done:
        raise CodeRuntimeError(self._sandbox_error.result())
    # ... normal handling
```

### Tool call raises an exception on the host

The host calls `super().call_tool()` which might fail. This is already handled — the exception propagates normally in the host process. The sandbox is blocked on HTTP. If the host decides not to call `provide_result()` (because it's raising an exception), the sandbox will eventually time out. That's fine — the execution is being abandoned anyway.

### Network failure mid-execution

The sandbox's HTTP POST to the host fails (network glitch). The sandbox sees a connection error. The stub code could retry:

```python
def _call_host(fn_name, args, kwargs):
    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            return _json.loads(resp.read())["result"]
        except (urllib.error.URLError, ConnectionError):
            if attempt == 2:
                raise
            import time
            time.sleep(1 * (attempt + 1))  # simple backoff
```

Or it could fail fast and let the host detect the sandbox died. The retry policy is transport-specific configuration.

### Authentication and security

The callback URL is reachable from the internet (the sandbox must be able to POST to it). Without auth, anyone could send fake function calls. The transport generates a one-time auth token:

```python
class HttpCallbackTransport:
    async def start(self, available_functions):
        self._token = secrets.token_urlsafe(32)
        # ... start HTTP server ...
        return TransportConfig(
            stub_code=self._generate_stubs(available_functions),
            env_vars={"_CALLBACK_TOKEN": self._token},
        )

    async def _handle_request(self, request):
        if request.headers.get("Authorization") != f"Bearer {self._token}":
            return Response(status=403)
        # ... normal handling
```

The token is injected into the sandbox via environment variable and embedded in the generated stub code. Each execution gets a fresh token.

## How `TransportConfig` Connects the Pieces

The transport's `start()` method returns a `TransportConfig` — the "instructions" that the cloud sandbox provider uses to set up the sandbox side:

```python
@dataclass
class TransportConfig:
    stub_code: str           # Python code to prepend (the fake function defs)
    env_vars: dict[str, str] # Environment variables the sandbox needs
```

The cloud sandbox implementation (`E2BSandbox`, etc.) uses this when starting execution:

```python
class E2BSandbox(CodeSandbox):
    async def execute(self, code: str, functions: list[str]) -> CodeExecution:
        # 1. Start the transport (creates HTTP server, generates stubs)
        transport_config = await self.transport.start(functions)

        # 2. Build the full code: preamble + LLM code + epilogue
        full_code = transport_config.stub_code + "\n" + code + "\n" + EPILOGUE_TEMPLATE

        # 3. Send to E2B with the required env vars
        self._e2b_sandbox.run_code(
            full_code,
            envs=transport_config.env_vars,
        )

        # 4. Return the execution handle
        return HttpCallbackExecution(transport=self.transport)
```

## How This Maps to `CodeExecution` Methods

| `CodeExecution` method | Monty implementation | HTTP callback implementation |
|---|---|---|
| `next()` | Call `snapshot.resume()` or `monty.start()`. Check if result is `MontySnapshot` or `MontyComplete`. | `await` the `_pending_call` or `_completion` future. Whichever resolves first. |
| `provide_result(value)` | Store value. On next `next()` call, pass it to `snapshot.resume(return_value=value)`. | Resolve the `_pending_result` future, which unblocks the HTTP handler, which sends the response to the sandbox. |
| `dump()` | `snapshot.dump()` → serialized Rust VM state as bytes. | Could serialize session ID + transport state. Or `None` if not supported. |

---
---

# Appendix B: The Yield/Pause-Resume Mechanism — Full Mechanics

## The Mental Model: Code Execution as a Generator

Think of sandbox code execution like a Python generator. The code "yields" every time it calls a tool, and the host "sends" the result back:

```python
# MENTAL MODEL ONLY — not real code
def code_as_generator():
    weather = yield FunctionCall("get_weather", kwargs={"city": "London"})
    forecast = yield FunctionCall("get_forecast", kwargs={"city": "London", "days": 3})
    return {"weather": weather, "forecast": forecast}

# Host drives it:
gen = code_as_generator()
event = next(gen)                              # FunctionCall("get_weather", ...)
event = gen.send(real_get_weather("London"))    # FunctionCall("get_forecast", ...)
event = gen.send(real_get_forecast("London", 3))  # StopIteration with return value
```

This is the conceptual model. Every sandbox implements this "generator protocol" using whatever mechanism it has available.

## Mechanism 1: Monty's Native Pause (Synchronous, In-Process)

Monty is the simplest case because it's literally built for this.

### How Monty represents state

Monty is a Rust bytecode interpreter. When it encounters an external function call:

1. It has a **call stack** (list of frames, each with local variables and an instruction pointer)
2. It has a **value stack** (operands for the current instruction)
3. It freezes all of this into a `MontySnapshot` object exposed to Python via PyO3

The `MontySnapshot` is a **frozen point in time**. The entire VM state is inside it. The `Monty` object itself is no longer needed — the snapshot is self-contained.

### The iteration in concrete code

Here's exactly what happens in the current `call_tool()` (lines 252-347), annotated:

```python
# Create interpreter. This PARSES the code but doesn't run it.
m = monty.Monty(code, external_functions=["get_weather", "get_forecast"])

# Type-check (optional). This validates without executing.
m.type_check(prefix_code=prefix)

# START execution. This runs code until:
#   (a) it hits an external function call → returns MontySnapshot
#   (b) it completes → returns MontyComplete
#   (c) it errors → raises MontyRuntimeError
result = m.start()
# At this point, if code calls get_weather on line 1, result is a MontySnapshot.
# The Monty VM is frozen mid-execution. `m` is now inert.

while isinstance(result, monty.MontySnapshot):
    # result.function_name = "get_weather"
    # result.kwargs = {"city": "London"}
    # The VM is paused at exactly the get_weather() call site.

    tool_return_value = await run_real_tool(result.function_name, result.kwargs)

    # RESUME: feed the return value back. The VM unfreezes, assigns the return
    # value to `weather`, and continues running until:
    #   (a) another external call → new MontySnapshot
    #   (b) code finishes → MontyComplete
    #   (c) runtime error → MontyRuntimeError
    result = result.resume(return_value=tool_return_value)
    # Old `result` (the snapshot we just resumed from) is now consumed/invalid.
    # New `result` is either the next snapshot or completion.

# Loop exited → result is MontyComplete
return result.output
```

### What `.resume()` actually does

`MontySnapshot.resume(return_value=X)` does this internally (in Rust):

1. Take the frozen VM state out of the snapshot
2. Push `X` onto the value stack (this is what the `get_weather()` call "returns")
3. Advance the instruction pointer past the `CALL` instruction
4. **Continue interpreting bytecode** until the next external call or completion
5. If another external call: freeze again, return new `MontySnapshot`
6. If complete: return `MontyComplete` with the output value

The old snapshot is consumed. You can't resume it twice. This is enforced by Rust's ownership system (the VM state is moved, not copied).

### What `.dump()` / `.load()` does

`MontySnapshot.dump()` serializes the frozen VM state to bytes. This includes:
- Call stack frames (local variables, instruction pointers)
- Value stack
- Global variables
- Any allocated objects (strings, lists, dicts in the code's heap)

`MontySnapshot.load(data)` deserializes back. The resulting snapshot can be `.resume()`d as normal.

This is what makes approval checkpointing work: we serialize the VM state, persist it across the approval round-trip, then deserialize and resume.

## Mechanism 2: HTTP Callback (Asynchronous, Cross-Process)

Cloud sandboxes can't freeze CPython. Instead, we use blocking I/O as the pause mechanism.

### How the "yield" is simulated

When the sandbox code calls `get_weather(city="London")`, it's actually calling a stub function that does:

```python
def get_weather(*, city: str):
    return _call_host("get_weather", [], {"city": city})
```

`_call_host` makes an HTTP POST and blocks. From the perspective of the sandbox's Python thread, execution is "paused" — the thread is sleeping on socket I/O. The OS scheduler doesn't give it CPU time until data arrives on the socket.

This is functionally identical to Monty's `MontySnapshot`: execution is frozen at the call site, waiting for a return value.

### How "send the result back" is simulated

When the host calls `provide_result(value)`, it resolves the `asyncio.Future` that the HTTP handler is awaiting. The handler then sends the HTTP response body `{"result": value}`. The sandbox's `urllib.request.urlopen()` returns, `_call_host` returns the value, and the stub function returns to the LLM's code.

This is functionally identical to `snapshot.resume(return_value=value)`.

### The async plumbing in detail

Here's how the `asyncio` futures connect the HTTP server to the `CodeExecution` interface:

```
                           HOST PROCESS (single asyncio event loop)
                    ┌──────────────────────────────────────────────┐
                    │                                              │
                    │  CodeModeToolset.call_tool()                 │
                    │    │                                         │
                    │    ├── event = await execution.next()        │
                    │    │      └── await _pending_call  ◄─────┐   │
                    │    │                                     │   │
                    │    │   (blocked until sandbox POSTs)     │   │
                    │    │                                     │   │
                    │    │   HTTP handler receives POST ───────┘   │
                    │    │      ├── _pending_call.set_result(...)  │
                    │    │      └── await _pending_result  ◄───┐   │
                    │    │                                     │   │
                    │    ├── tool_result = await run_real_tool()│   │
                    │    │                                     │   │
                    │    ├── await execution.provide_result()   │   │
                    │    │      └── _pending_result.set_result()───┘
                    │    │                                         │
                    │    │   HTTP handler resumes, sends response  │
                    │    │   Sandbox unblocks, continues           │
                    │    │                                         │
                    │    └── event = await execution.next()        │
                    │           └── (cycle repeats)                │
                    │                                              │
                    └──────────────────────────────────────────────┘
```

All of this happens on **one** asyncio event loop. No threads are blocked on the host side. The HTTP server, the `next()` call, and `provide_result()` are all cooperating coroutines.

On the sandbox side, a real OS thread IS blocked (on the HTTP socket). This is fine — the sandbox is a separate process on a separate machine.

### The `next()` implementation must handle two races

When `next()` is called, two things could happen:
1. The sandbox calls another tool → POST to `/call`
2. The sandbox finishes → POST to `/complete`

These are mutually exclusive for any given "step," but `next()` doesn't know which one will happen. So it awaits both:

```python
async def next(self) -> FunctionCall | ExecutionResult:
    # Create tasks for both possible events
    call_task = asyncio.ensure_future(self._wait_for_call())
    complete_task = asyncio.ensure_future(self._wait_for_completion())

    done, pending = await asyncio.wait(
        [call_task, complete_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel the one that didn't happen
    for task in pending:
        task.cancel()

    winner = done.pop()
    return winner.result()
```

After the first tool call, `_wait_for_completion` is still "possible" — maybe the next `next()` will be the last. So fresh futures are created each cycle.

## Mechanism 3: Stdin/Stdout (Synchronous, Local Process)

For subprocess sandboxes (e.g., running code in a restricted local Python process), HTTP is overkill. A simpler transport uses JSON messages over stdin/stdout:

### Sandbox side

```python
import sys, json

def _call_host(fn_name, args, kwargs):
    # Write the call to stdout (host reads this)
    sys.stdout.write(json.dumps({
        "type": "function_call",
        "function_name": fn_name,
        "args": args,
        "kwargs": kwargs,
    }) + "\n")
    sys.stdout.flush()

    # Read the result from stdin (host writes this)
    line = sys.stdin.readline()
    response = json.loads(line)
    return response["result"]
```

### Host side

```python
class StdioExecution(CodeExecution):
    def __init__(self, process: asyncio.subprocess.Process):
        self._proc = process

    async def next(self) -> FunctionCall | ExecutionResult:
        line = await self._proc.stdout.readline()
        if not line:
            # Process exited — read return code
            await self._proc.wait()
            # ... get output from a final message or exit code
        msg = json.loads(line)
        if msg["type"] == "function_call":
            return FunctionCall(...)
        elif msg["type"] == "execution_complete":
            return ExecutionResult(...)

    async def provide_result(self, value: Any) -> None:
        msg = json.dumps({"result": value}) + "\n"
        self._proc.stdin.write(msg.encode())
        await self._proc.stdin.drain()
```

Same pattern: sandbox blocks on `stdin.readline()`, host unblocks it by writing to `stdin`. The `CodeExecution` interface is identical.

## How The Three Mechanisms Compare

| Aspect | Monty (native) | HTTP callback | Stdin/Stdout |
|---|---|---|---|
| **Pause mechanism** | Rust VM freezes instruction pointer | HTTP `urlopen()` blocks on socket | `stdin.readline()` blocks on pipe |
| **Resume mechanism** | `snapshot.resume(return_value=X)` | HTTP response body `{"result": X}` | Write `{"result": X}\n` to stdin |
| **Latency per call** | ~microseconds (in-process) | ~1-50ms (network round trip) | ~0.1ms (pipe I/O) |
| **Checkpoint support** | Full (serialize Rust VM state) | Possible (serialize session/transport state) | Possible (keep process alive, serialize PID) |
| **Python support** | Restricted subset | Full CPython | Full CPython (can restrict with seccomp) |
| **Where sandbox runs** | Same process | Remote VM (E2B, Modal) | Local subprocess |
| **Max concurrent** | Many (lightweight Rust VMs) | Limited by sandbox provider | Limited by local resources |

## Why This All Looks the Same to CodeModeToolset

The `CodeExecution` interface flattens all three mechanisms into the same API:

```python
event = await execution.next()           # "What happened?"
while isinstance(event, FunctionCall):   # "It called a function"
    result = await real_tool(event)       # "Run the real thing"
    await execution.provide_result(result)# "Here's the answer"
    event = await execution.next()        # "What next?"
return event.output                      # "It finished"
```

This is the generator protocol reified as an object. `next()` is `__next__()`. `provide_result()` is `send()`. `ExecutionResult` is `StopIteration`. The mechanism behind the protocol — Rust VM freezing, HTTP blocking, pipe I/O — is invisible to the consumer.

That's the whole point of the abstraction: **CodeModeToolset doesn't care how the sandbox pauses. It only cares that it does.**

---
---

# Appendix C: asyncio Primitives Reference

This appendix documents every asyncio concept and API used in the HTTP callback and stdio transport designs. Each section explains *what* the primitive is, *why* we chose it, and links to the CPython docs.

All references are to the Python 3.12 documentation at `https://docs.python.org/3.12/library/`.

---

## C.1: The Event Loop

> "The event loop is the core of every asyncio application. Event loops run asynchronous tasks and callbacks, perform network IO operations, and run subprocesses."
>
> — [asyncio-eventloop.html](https://docs.python.org/3.12/library/asyncio-eventloop.html)

The event loop is a single-threaded scheduler. It maintains a queue of ready coroutines and polls for I/O events (socket data arriving, timers expiring). On each "tick," it:

1. Runs all ready coroutines until they `await` something
2. Checks for I/O events (via `epoll`/`kqueue`/`IOCP`)
3. Wakes up coroutines whose I/O is ready
4. Repeat

**Why this matters for our design:** The host-side HTTP server, the `next()` method, and `provide_result()` all run as coroutines on the **same** event loop. When `next()` awaits a `Future`, it yields control back to the event loop, which can then process the incoming HTTP request from the sandbox. There's no threading needed on the host side — concurrency comes from coroutine interleaving, not OS threads.

**Key rule:** Never call blocking (synchronous) I/O in a coroutine. If you call `time.sleep(5)` or `urllib.request.urlopen()` on the host side, you freeze the entire event loop for that duration — no HTTP requests can be processed, no other coroutines can run. The sandbox side is fine because it's a separate process with its own thread.

---

## C.2: `asyncio.Future` — The Synchronization Primitive

> "A Future represents an eventual result of an asynchronous operation. Not thread-safe."
>
> "Future objects are used to bridge **low-level callback-based code** with high-level async/await code."
>
> — [asyncio-future.html](https://docs.python.org/3.12/library/asyncio-future.html)

A `Future` is the simplest asyncio primitive: a box that will eventually contain a value. It has three states:

```
PENDING  ──set_result(value)──→  DONE (has a result)
    │
    └──set_exception(exc)────→  DONE (has an exception)
```

### Creating a Future

> "This is the preferred way to create Futures in asyncio. This lets third-party event loops provide alternative implementations of the Future object (with better performance or instrumentation)."
>
> — [asyncio-eventloop.html#asyncio.loop.create_future](https://docs.python.org/3.12/library/asyncio-eventloop.html#asyncio.loop.create_future)

```python
loop = asyncio.get_running_loop()
fut = loop.create_future()
```

Use `loop.create_future()`, not `asyncio.Future()` directly. The loop factory lets alternative event loop implementations (uvloop, etc.) provide optimized Future types.

Note: the design doc examples use `asyncio.get_event_loop().create_future()` for brevity, but in real code you should use `asyncio.get_running_loop()` (available since Python 3.7) which is safer — it raises `RuntimeError` if no loop is running rather than silently creating a new one.

### Awaiting a Future

```python
value = await fut  # Suspends the coroutine until fut has a result
```

> "A Future is an awaitable object. Coroutines can await on Future objects until they either have a result or an exception set, or until they are cancelled. A Future can be awaited multiple times and the result is same."
>
> — [asyncio-future.html](https://docs.python.org/3.12/library/asyncio-future.html)

When you `await fut`:
- If the future is already done: returns the result immediately (no suspension)
- If the future is pending: the coroutine **suspends**. The event loop records that this coroutine wants to be woken when `fut` completes. Control returns to the event loop, which can run other coroutines.
- When someone calls `fut.set_result(value)`: the event loop schedules the awaiting coroutine to resume. On the next tick, the `await` expression evaluates to `value`.

### Setting the result

> "Mark the Future as _done_ and set its result. Raises an `InvalidStateError` error if the Future is already done."
>
> — [asyncio-future.html#asyncio.Future.set_result](https://docs.python.org/3.12/library/asyncio-future.html#asyncio.Future.set_result)

```python
fut.set_result({"temp": 20})  # Wakes up anyone awaiting this future
```

You can only call `set_result()` **once**. A second call raises `InvalidStateError`. This is a feature — it prevents double-resolution bugs.

### Reading the result

> "Return the result of the Future. If the Future is done and has a result set by the `set_result()` method, the result value is returned."
>
> — [asyncio-future.html#asyncio.Future.result](https://docs.python.org/3.12/library/asyncio-future.html#asyncio.Future.result)

```python
value = fut.result()  # Synchronous — only call this on a done future
```

In our design, we mostly use `await fut` rather than `.result()` directly. The `.result()` method is used after `asyncio.wait()` tells us a future is in the `done` set.

### Thread safety warning

> "Not thread-safe."
>
> — [asyncio-future.html](https://docs.python.org/3.12/library/asyncio-future.html)

All Future operations (`set_result`, `await`, etc.) must happen on the same event loop thread. In our design this is naturally the case: both the HTTP handler and the `next()`/`provide_result()` calls run on the same asyncio event loop.

If you ever needed to resolve a future from a different thread (e.g., a thread-pool executor), you'd use `loop.call_soon_threadsafe(fut.set_result, value)`.

### Why Future and not `asyncio.Event` or `asyncio.Queue`?

- **`asyncio.Event`**: A boolean flag (set/clear). No value. We need to pass data (the `FunctionCall` or result), not just signal.
- **`asyncio.Queue`**: FIFO queue. Overkill — we have exactly one producer and one consumer per rendezvous, and exactly one item. A queue adds unnecessary buffering semantics.
- **`Future`**: Exactly one value, exactly one resolution, awaitable. Perfect fit for a single-shot rendezvous.

### How we use Futures in the HTTP callback

The three futures in `HttpCallbackExecution`:

```python
self._pending_call: asyncio.Future[FunctionCall]   # Set by HTTP handler when sandbox POSTs to /call
self._pending_result: asyncio.Future[Any]           # Set by provide_result() when host has the tool result
self._completion: asyncio.Future[ExecutionResult]   # Set by HTTP handler when sandbox POSTs to /complete
```

Data flow:

```
Sandbox POSTs /call  →  HTTP handler sets _pending_call  →  next() returns FunctionCall
Host runs tool       →  provide_result() sets _pending_result  →  HTTP handler sends response
Sandbox POSTs /complete  →  HTTP handler sets _completion  →  next() returns ExecutionResult
```

After each cycle, `provide_result()` creates **fresh** futures for the next round. The old ones are consumed (done state, garbage collected).

---

## C.3: `asyncio.wait()` — Racing Multiple Futures

> "Run Future and Task instances in the _aws_ iterable concurrently and block until the condition specified by _return_when_."
>
> "Returns two sets of Tasks/Futures: `(done, pending)`."
>
> — [asyncio-task.html#asyncio.wait](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.wait)

```python
done, pending = await asyncio.wait(
    [future_a, future_b],
    return_when=asyncio.FIRST_COMPLETED,
)
```

### What `return_when=FIRST_COMPLETED` does

> "The function will return when any future finishes or is cancelled."
>
> — [asyncio-task.html#asyncio.FIRST_COMPLETED](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.FIRST_COMPLETED)

When called with `FIRST_COMPLETED`:
- The coroutine suspends until **at least one** of the given awaitables is done
- It returns immediately with `done` containing the finished ones, `pending` containing the rest
- The pending futures are **not cancelled** — they keep running

### Timeout behavior

> "Unlike `wait_for()`, `wait()` does not cancel the futures when a timeout occurs."
>
> — [asyncio-task.html#asyncio.wait](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.wait)

```python
done, pending = await asyncio.wait(
    [future_a, future_b],
    return_when=asyncio.FIRST_COMPLETED,
    timeout=30.0,
)
```

If the timeout expires and neither future is done, `done` is empty and `pending` contains both. Unlike `asyncio.wait_for()`, the futures are NOT cancelled — you must handle this yourself.

**Note:** `asyncio.wait()` does NOT raise `asyncio.TimeoutError`. It just returns empty `done`. The timeout error handling shown in earlier code examples checks `if not done:` rather than catching an exception. (The earlier Appendix A example was simplified for clarity.)

### Why we use `wait()` in `next()`

```python
async def next(self) -> FunctionCall | ExecutionResult:
    done, pending = await asyncio.wait(
        [self._pending_call, self._completion],
        return_when=asyncio.FIRST_COMPLETED,
    )
    # ...
```

We're racing two futures: "sandbox calls a function" vs. "sandbox completes." We don't know which will happen first. `asyncio.wait(FIRST_COMPLETED)` lets us await both and react to whichever fires.

Alternative considered: `asyncio.gather()`. But `gather()` waits for ALL to complete. We want the first one.

Alternative considered: `asyncio.wait_for()`. But that takes a single awaitable, not multiple. We need to race two.

---

## C.4: `asyncio.Task` — Running Coroutines Concurrently

> "A Future-like object that runs a Python coroutine. Not thread-safe."
>
> "Tasks are used to run coroutines in event loops. If a coroutine awaits on a Future, the Task suspends the execution of the coroutine and waits for the completion of the Future. When the Future is done, the execution of the wrapped coroutine resumes."
>
> — [asyncio-task.html#asyncio.Task](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.Task)

A `Task` wraps a coroutine and schedules it on the event loop. It inherits from `Future`, so it can be awaited and passed to `asyncio.wait()`.

### Task vs Future

> "`asyncio.Task` inherits from `Future` all of its APIs except `Future.set_result()` and `Future.set_exception()`."
>
> — [asyncio-task.html#asyncio.Task](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.Task)

- **Future**: A box for a value. Someone **externally** calls `set_result()`. No code runs "inside" it.
- **Task**: A running coroutine. The coroutine's return value becomes the task's result. You don't call `set_result()` — the coroutine does that by returning.

In our design:
- `_pending_call`, `_pending_result`, `_completion` are **Futures** — they're resolved externally (by the HTTP handler or by `provide_result()`)
- The HTTP server coroutine, if wrapped, would be a **Task** — it's a running coroutine

### `asyncio.ensure_future()` and `asyncio.create_task()`

`asyncio.ensure_future(coro_or_future)` converts a coroutine into a Task if needed:
- If given a coroutine: wraps it in a Task and schedules it
- If given a Future/Task: returns it unchanged

In modern Python (3.7+), prefer `asyncio.create_task(coro)` for coroutines. Use `ensure_future()` only when you might receive either a coroutine or a Future.

### Cancellation

> "Request the Task to be cancelled. This arranges for a `CancelledError` exception to be thrown into the wrapped coroutine on the next cycle of the event loop."
>
> "The coroutine then has a chance to clean up or even deny the request by suppressing the exception with a `try ... except CancelledError ... finally` block. Therefore, unlike `Future.cancel()`, `Task.cancel()` does not guarantee that the Task will be cancelled."
>
> — [asyncio-task.html#asyncio.Task.cancel](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.Task.cancel)

In the `next()` implementation, when we race two tasks and one wins, we cancel the loser:

```python
done, pending = await asyncio.wait([call_task, complete_task], return_when=asyncio.FIRST_COMPLETED)
for task in pending:
    task.cancel()
```

This is important for cleanup: without cancellation, the losing task's coroutine would remain suspended indefinitely, leaking resources.

Note: For bare `Future` objects (not Tasks), `cancel()` is simpler — it just marks the future as cancelled. But since `asyncio.wait()` accepts both, the cancellation pattern works uniformly.

---

## C.5: `asyncio.StreamReader` and `asyncio.StreamWriter` — Subprocess I/O

These are used in the stdin/stdout transport for local subprocess sandboxes.

### StreamReader

> "Represents a reader object that provides APIs to read data from the IO stream. As an asynchronous iterable, the object supports the `async for` statement."
>
> — [asyncio-stream.html#asyncio.StreamReader](https://docs.python.org/3.12/library/asyncio-stream.html#asyncio.StreamReader)

#### `readline()`

> "Read one line, where 'line' is a sequence of bytes ending with `\n`. If EOF is received and `\n` was not found, the method returns partially read data. If EOF is received and the internal buffer is empty, return an empty `bytes` object."
>
> — [asyncio-stream.html#asyncio.StreamReader.readline](https://docs.python.org/3.12/library/asyncio-stream.html#asyncio.StreamReader.readline)

```python
line = await self._proc.stdout.readline()
if not line:
    # EOF — process exited
```

This is the host-side read in the stdio transport. It's async — the coroutine suspends until a full line (`\n`-terminated) arrives from the subprocess's stdout. The event loop monitors the pipe file descriptor via its I/O polling mechanism.

### StreamWriter

> "Represents a writer object that provides APIs to write data to the IO stream."
>
> — [asyncio-stream.html#asyncio.StreamWriter](https://docs.python.org/3.12/library/asyncio-stream.html#asyncio.StreamWriter)

#### `write(data)`

> "The method attempts to write the data to the underlying socket immediately. If that fails, the data is queued in an internal write buffer until it can be sent."
>
> "The method should be used along with the `drain()` method:
> ```python
> stream.write(data)
> await stream.drain()
> ```"
>
> — [asyncio-stream.html#asyncio.StreamWriter.write](https://docs.python.org/3.12/library/asyncio-stream.html#asyncio.StreamWriter.write)

`write()` is **synchronous** (not a coroutine). It doesn't `await`. It just buffers the data. The actual I/O happens later.

#### `drain()`

> "Wait until it is appropriate to resume writing to the stream. This is a flow control method that interacts with the underlying IO write buffer. When the size of the buffer reaches the high watermark, `drain()` blocks until the size of the buffer is drained down to the low watermark and writing can be resumed. When there is nothing to wait for, the `drain()` returns immediately."
>
> — [asyncio-stream.html#asyncio.StreamWriter.drain](https://docs.python.org/3.12/library/asyncio-stream.html#asyncio.StreamWriter.drain)

**Why `drain()` is mandatory after `write()`:** Without it, you could write faster than the pipe can consume, filling the buffer without bound. `drain()` applies **backpressure** — it pauses the writing coroutine if the buffer is too full, preventing unbounded memory growth.

In practice, for our use case (writing a single JSON line per tool result), the buffer will never be full. `drain()` will return immediately. But calling it is the correct pattern and costs nothing.

```python
async def provide_result(self, value: Any) -> None:
    msg = json.dumps({"result": value}) + "\n"
    self._proc.stdin.write(msg.encode())  # Synchronous — buffers data
    await self._proc.stdin.drain()         # Async — ensures data is actually sent
```

### Subprocess pipes

> "`stdin`: Standard input stream (StreamWriter) or None if the process was created with `stdin=None`."
>
> "`stdout`: Standard output stream (StreamReader) or None if the process was created with `stdout=None`."
>
> — [asyncio-subprocess.html#asyncio.subprocess.Process](https://docs.python.org/3.12/library/asyncio-subprocess.html#asyncio.subprocess.Process)

When you create a subprocess with `stdin=PIPE, stdout=PIPE`, asyncio wraps the OS pipes in `StreamWriter` (stdin) and `StreamReader` (stdout). This lets you use the same async read/write API for subprocess I/O as for network sockets.

**Deadlock warning from the docs:**

> "Use the `communicate()` method rather than `process.stdin.write()`, `await process.stdout.read()` or `await process.stderr.read()`. This avoids deadlocks due to streams pausing reading or writing and blocking the child process."
>
> — [asyncio-subprocess.html#asyncio.subprocess.Process](https://docs.python.org/3.12/library/asyncio-subprocess.html#asyncio.subprocess.Process)

This warning applies to the pattern of "write all input, then read all output." Our use case is different — we're doing **interleaved** reads and writes (read a line, write a line, read a line, ...), which is safe from deadlock because neither side accumulates unbounded data. The subprocess writes one JSON line and blocks on `stdin.readline()`. The host reads that line and writes one JSON line back. The "read all then write all" deadlock doesn't apply.

---

## C.6: `asyncio.create_subprocess_exec()` — Spawning the Sandbox

> "Create a subprocess."
>
> — [asyncio-subprocess.html#asyncio.create_subprocess_exec](https://docs.python.org/3.12/library/asyncio-subprocess.html#asyncio.create_subprocess_exec)

```python
proc = await asyncio.create_subprocess_exec(
    sys.executable, "-c", full_code,
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
```

This spawns a new Python process running the generated code. The `PIPE` constants tell asyncio to create pipe file descriptors and wrap them in `StreamReader`/`StreamWriter`.

#### `Process.wait()`

> "Wait for the child process to terminate. Set and return the `returncode` attribute."
>
> "This method can deadlock when using `stdout=PIPE` or `stderr=PIPE` and the child process generates so much output that it blocks waiting for the OS pipe buffer to accept more data. Use the `communicate()` method when using pipes to avoid this condition."
>
> — [asyncio-subprocess.html#asyncio.subprocess.Process.wait](https://docs.python.org/3.12/library/asyncio-subprocess.html#asyncio.subprocess.Process.wait)

In the stdio transport, we call `wait()` only after the subprocess has signaled completion (by sending the `execution_complete` message). At that point, the process is about to exit and won't produce more output, so the deadlock described in the docs doesn't apply.

---

## C.7: How It All Fits Together — The Event Loop Timeline

Here's a single event loop tick-by-tick view of one tool call via HTTP callback, showing which coroutines are active vs. suspended:

```
Event Loop Tick 1:
  ACTIVE:   call_tool() coroutine
            → calls await execution.next()
            → next() calls await asyncio.wait([_pending_call, _completion])
            → both futures are PENDING, so next() SUSPENDS
  SUSPENDED: call_tool() (waiting on next())

Event Loop Tick 2:
  I/O EVENT: Data arrives on HTTP server socket (sandbox's POST /call)
  ACTIVE:   HTTP server coroutine
            → reads request body
            → parses FunctionCall
            → calls _pending_call.set_result(FunctionCall(...))
            → _pending_call is now DONE
            → asyncio.wait() wakes up (FIRST_COMPLETED condition met)
            → calls await _pending_result (the HTTP handler now waits for the tool result)
  ACTIVE:   call_tool() coroutine RESUMES
            → next() returns FunctionCall
            → call_tool() processes the function call
            → calls await super().call_tool() to run the real tool
  SUSPENDED: HTTP handler (waiting on _pending_result)

Event Loop Tick 3:
  ACTIVE:   call_tool() coroutine (continues after real tool returns)
            → calls await execution.provide_result(tool_return_value)
            → provide_result() calls _pending_result.set_result(value)
            → _pending_result is now DONE
  ACTIVE:   HTTP handler coroutine RESUMES
            → sends HTTP response with {"result": value}
            → sandbox's urlopen() unblocks
  ACTIVE:   call_tool() coroutine
            → calls await execution.next() again
            → cycle repeats from Tick 1

[All of this is ONE thread. Coroutines take turns.]
```

**Key insight:** The event loop is doing cooperative multitasking. `call_tool()` and the HTTP handler take turns running. Neither blocks the other. The `Future` is the coordination point where one coroutine says "I'm done, wake up whoever is waiting."

---

## C.8: Summary of Primitives and Their Roles

| Primitive | CPython Docs | Role in Our Design |
|---|---|---|
| `asyncio.Future` | [asyncio-future.html](https://docs.python.org/3.12/library/asyncio-future.html) | Single-shot rendezvous between HTTP handler and `next()`/`provide_result()` |
| `Future.set_result()` | [asyncio-future.html#set_result](https://docs.python.org/3.12/library/asyncio-future.html#asyncio.Future.set_result) | HTTP handler resolves the future when sandbox sends a request |
| `await future` | [asyncio-future.html](https://docs.python.org/3.12/library/asyncio-future.html) | `next()` suspends until the sandbox communicates |
| `loop.create_future()` | [asyncio-eventloop.html#create_future](https://docs.python.org/3.12/library/asyncio-eventloop.html#asyncio.loop.create_future) | Factory for creating futures on the running event loop |
| `asyncio.wait()` | [asyncio-task.html#asyncio.wait](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.wait) | Race `_pending_call` vs `_completion` in `next()` |
| `FIRST_COMPLETED` | [asyncio-task.html#FIRST_COMPLETED](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.FIRST_COMPLETED) | Return as soon as either future resolves |
| `asyncio.Task` | [asyncio-task.html#asyncio.Task](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.Task) | Wraps coroutines; inherits from Future so can be passed to `wait()` |
| `Task.cancel()` | [asyncio-task.html#cancel](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.Task.cancel) | Clean up the losing future after `wait(FIRST_COMPLETED)` |
| `StreamReader.readline()` | [asyncio-stream.html#readline](https://docs.python.org/3.12/library/asyncio-stream.html#asyncio.StreamReader.readline) | Read one JSON message from subprocess stdout |
| `StreamWriter.write()` | [asyncio-stream.html#write](https://docs.python.org/3.12/library/asyncio-stream.html#asyncio.StreamWriter.write) | Buffer a JSON message to subprocess stdin |
| `StreamWriter.drain()` | [asyncio-stream.html#drain](https://docs.python.org/3.12/library/asyncio-stream.html#asyncio.StreamWriter.drain) | Flush the write buffer with backpressure control |
| `create_subprocess_exec()` | [asyncio-subprocess.html](https://docs.python.org/3.12/library/asyncio-subprocess.html#asyncio.create_subprocess_exec) | Spawn the sandbox subprocess with pipe-wrapped stdin/stdout |
| `Process.wait()` | [asyncio-subprocess.html#wait](https://docs.python.org/3.12/library/asyncio-subprocess.html#asyncio.subprocess.Process.wait) | Wait for subprocess exit after completion signal |
