# Tools Advanced

Read this file when the user wants advanced tool behavior: approval, retries, failed tool results, validation, timeouts, rich tool returns, or tool search/deferred loading.

## Require Tool Approval (Human in the Loop)

Use deferred tools when the run should pause for approval.

```python
from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ToolDenied,
)

agent = Agent('openai:gpt-5.2', name='approval_agent', output_type=[str, DeferredToolRequests])


@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    return f'File {path!r} deleted'


result = agent.run_sync('Delete __init__.py')
messages = result.all_messages()

assert isinstance(result.output, DeferredToolRequests)
results = DeferredToolResults()
for call in result.output.approvals:
    results.approvals[call.tool_call_id] = ToolDenied('Deleting files is not allowed')

result = agent.run_sync('Continue', message_history=messages, deferred_tool_results=results)
print(result.output)
```

Two key rules:

- `DeferredToolRequests` must be in the output type
- for conditional approval, raise `ApprovalRequired(...)` instead of marking the whole tool `requires_approval=True` — from the tool function, or from its `args_validator=` so invalid arguments are rejected before a human is asked (see below)

Deferred batches also surface in the event stream: `DeferredToolRequestsEvent` carries the `DeferredToolRequests` once per batch, before any `HandleDeferredToolCalls` handler runs; `DeferredToolResultsEvent` carries the `DeferredToolResults` when a handler resolves requests inline (not when results are provided to a new run via `deferred_tool_results`).

## Make an Agent Resilient with Retries

Raise `ModelRetry` from inside the tool when the model should correct and try again.

```python
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('openai:gpt-5.2', name='retry_agent', deps_type=dict[str, int])


@agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[dict[str, int]], name: str) -> int:
    user_id = ctx.deps.get(name)
    if user_id is None:
        raise ModelRetry(f'No user found with name {name!r}')
    return user_id
```

Use retries for recoverable model mistakes, not application crashes.

Set the agent-wide tool-retry default with `Agent(retries={'tools': N})`, and override it for a single run (or `iter`) with `agent.run(retries={'tools': N})` — explicit per-tool `retries=` and per-toolset `FunctionToolset(max_retries=N)` still win. A bare `int` at run time overrides both budgets (matching construction), so pass a dict like `{'tools': N}` or `{'output': N}` to change just one.

## Report a Failed Tool Result

Not every failure is a retry. Choose the exception by what you want the model to do next:

- `ModelRetry` — the model should **try again** with corrected arguments or a different approach. Consumes the tool's retry budget.
- `ToolFailed` — the call is **done and failed** (resource missing, operation unsupported, definitive upstream error). The model should **see the result and adapt**. Does **not** consume the retry budget — bound repeated failures with `UsageLimits` at the run level.
- Any other exception propagates and aborts the run.

```python
from pathlib import Path

from pydantic_ai import Agent, ToolFailed

agent = Agent('openai:gpt-5.2')


@agent.tool_plain
def read_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise ToolFailed(f'File not found: {path}')
    return file_path.read_text()
```

The failure is recorded in message history as a `ToolReturnPart` with `outcome='failed'` and traced as an error in telemetry. Pydantic AI uses the provider's native error field where one exists; otherwise model-visible content is JSON-framed as `{"error": ...}` so the failure remains explicit.

`ToolFailed` can also be raised from an `args_validator` (see below) and from tool validation/execution hooks with the same model-visible and retry-budget behavior. This is useful for converting a third-party exception into a failed result in one place instead of per tool. MCP servers expose the same retry-vs-failed choice via `tool_error_behavior`. For deferred tools, a `ToolFailed` instance can be supplied as a `DeferredToolResults.calls` value to report an external execution failure, just like `ModelRetry` requests a retry from there.

`ToolFailed` is handled only for function tools, their `args_validator`, and tool validation/execution hooks. Output functions and output validators use `ModelRetry` to request another attempt; there, `ToolFailed` is an ordinary exception that aborts the run unless an output-process error hook recovers from it.

## Validate or Require Approval Before Tool Execution

Use `args_validator=` when arguments are structurally valid but still need business-rule validation before execution or approval. A validator returns `None` on success, raises `ModelRetry` to ask the model to correct the arguments and try again, or raises `ToolFailed` to report a terminal failure the model should adapt to instead of retrying.

It can also raise `ApprovalRequired` or `CallDeferred` to defer the call, exactly as the tool function can — and this is the better place for a conditional-approval decision, since bad arguments are rejected before a human is asked to approve them. The tool isn't executed, the deferral doesn't consume the retry budget, and once the call is approved the validator runs again with `ctx.tool_call_approved` set to `True`.

```python
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    DeferredToolRequests,
    ModelRetry,
    RunContext,
)

agent = Agent('openai:gpt-5.2', name='validation_agent', deps_type=int, output_type=[str, DeferredToolRequests])


def validate_transfer(ctx: RunContext[int], amount: int) -> None:
    if amount > ctx.deps:
        raise ModelRetry(f'Amount must not exceed {ctx.deps}')
    if amount > 100 and not ctx.tool_call_approved:
        raise ApprovalRequired()


@agent.tool(args_validator=validate_transfer)
def transfer_funds(ctx: RunContext[int], amount: int) -> str:
    return f'Transferred {amount}'
```

Validation runs before the approval/deferral gate, so invalid arguments are retried instead of sent to an approver. Under durable execution, a tool with an `args_validator` gets its own validation activity, step, or task; tools without one schedule no extra durable unit.

## Use Advanced Tool Features

Reach for these features when the user needs more than a simple function tool:

- `ToolReturn` for rich return values, separate content/metadata, and `tools` names that reveal deferred tools
- `prepare=` for dynamic tool definitions
- `timeout=` for tool execution limits
- `sequential=True` to make a tool a barrier — it runs alone (tools emitted before it finish first, tools after it start once it finishes) while other tools parallelize around it; works on function tools and on output tools via `ToolOutput(sequential=True)`

Example with `ToolReturn`:

```python
from pydantic_ai import Agent, BinaryContent, ToolReturn

agent = Agent('openai:gpt-5.2', name='tool_return_agent')


@agent.tool_plain
def click_and_capture(x: int, y: int) -> ToolReturn:
    return ToolReturn(
        return_value=f'Successfully clicked at ({x}, {y})',
        content=['After:', BinaryContent(data=b'png-data', media_type='image/png')],
        metadata={'coordinates': {'x': x, 'y': y}},
    )
```

Whether a file returned from a tool can travel inside the tool result depends on the model and the file's media type. It goes inside the tool result on Anthropic and OpenAI Responses (images and documents), Gemini 3 (the types in `google_supported_mime_types_in_tool_returns`), and Bedrock (the media kinds a family supports). Everything else — OpenAI Chat Completions, Groq, Mistral, xAI, Hugging Face, Gemini 2.5 and earlier, plus a media type Gemini 3 or Bedrock can't carry, such as audio or video on Gemini 3 — travels on the user channel instead, framed as `<tool_result tool_name="..." tool_call_id="..." file_id="...">` ... `</tool_result>` — one set of tags per file — so the model doesn't read tool output as something the user uploaded; the tool result carries `See file <id>.` in its place, except on a failed Gemini return, whose native `error` string takes no file references. Anthropic and OpenAI Responses have no such fallback: audio or video from a tool raises `NotImplementedError`. Cohere drops such a file entirely (#7646). The framing is applied while the request is built — the file itself stays on the `ToolReturnPart` in history — and it is prompt text, so it says where content came from without attesting it. `ToolReturn.content` is sent unframed: that is user content you chose to add.

Set `tools=['tool_name']` when the call makes a tool declared with `defer_loading=True` available. The executor deduplicates names in first-occurrence order, omits names already revealed, and stores a `ToolAvailabilityDeltaPart` immediately after that call's `ToolReturnPart`. The recorded name remains revealed when history is resumed; an unknown or already-visible name is a no-op when rendered.

Every searchable deferred tool stays in the search corpus after discovery. A `CompactionPart` resets prospective discovery at its exact position, so future requests reveal pre-boundary tools again. For a call in the response currently being dispatched, earlier evidence still counts when the serving provider did not honor that boundary on the request wire; otherwise a call without visible evidence is refused with a "not available yet" retry.

## Control Tool Execution When an Output Tool Is Called

When a model calls an output tool (structured output) in the *same* response as other tools, the agent's `end_strategy` controls how those calls run and which one becomes the final result. Most agents never need to touch this, since most responses don't mix an output tool with other tools.

Three strategies (set on the agent, e.g. `Agent(..., end_strategy='exhaustive')`):

- `'graceful'` (default): tools run in emission order; function tools always run (in parallel where possible); the first successful output tool wins, later output tools are skipped. Use when function tool side effects (logging, notifications) should still happen.
- `'early'`: output tools run in emission order, stopping at the first success; function tools in the same response are skipped if an output succeeds, but run if *every* output fails. Fastest when you don't need those function tools once you have a result.
- `'exhaustive'`: every tool runs in parallel; the first valid output by emission order wins; other output tools still execute. Gives the model full visibility that each tool ran, at the cost of discarded output-tool side effects.

Retry-wins (under `'graceful'` / `'exhaustive'`): if a function tool raises `ModelRetry` (or its args fail validation) in the same response as a successful output, the output result is suppressed so the model addresses the retry next round. Does not apply under `'early'`, nor when streaming (`run_stream` commits the first matching output immediately, behaving like `'early'`).

Native/prompted/image output (`output_type` uses `NativeOutput`, `PromptedOutput`, or image output): the final result comes from the text/image the model returns, not an output tool. Because the model is asked to produce that output directly it usually returns it alone, but some models occasionally return it *and* a function tool call in one response. Under `'early'` a valid output ends the run and the co-emitted function tools are skipped (output that fails validation falls through to the tools); under `'graceful'` / `'exhaustive'` the function tools run and (outside streaming) the run continues. Only applies to *function* tools — a co-emitted output-tool or deferred call takes precedence and the output/image does not preempt it.

Plain text output (`output_type=str` / `TextOutput`, incl. a `str` fallback) is treated differently: the model isn't told its text is the final result, so text alongside a tool call is usually preamble, not an answer. Plain text never preempts a co-emitted function tool — the tool runs under `'early'` exactly as under `'graceful'`. (Streaming still commits the first text as it streams, regardless of `end_strategy`.)

To run a whole run's tools serially, use `with agent.parallel_tool_call_execution_mode('sequential'):` or set `parallel_tool_calls=False` on model settings.

See [Parallel Output Tool Calls](https://pydantic.dev/docs/ai/core-concepts/output/#parallel-output-tool-calls) and [tools-advanced docs](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#parallel-tool-calls-concurrency).

## Handle Network Errors and Rate Limiting Automatically

For tool-call retries, use `ModelRetry` and tool `retries=...`.

For HTTP request retries at the transport layer, use the library's retry configuration separately. Do not assume `ModelRetry` alone solves provider transport failures.

## Tool Search and Tool-Level Deferred Loading

Use tool-level deferred loading when the agent has many tools and the model should discover individual tools on demand via `search_tools`.

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', name='tool_search_agent')


@agent.tool_plain(defer_loading=True)
def lookup_internal_policy(policy_name: str) -> str:
    return f'policy details for {policy_name}'
```

Good fit:

- large MCP servers
- big tool catalogs
- situations where loading all tool schemas would bloat context

For bundle-level progressive disclosure of instructions plus tools, read [Capabilities on Demand](./ON-DEMAND-CAPABILITIES.md) instead.
