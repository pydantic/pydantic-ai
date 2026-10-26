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
    # Your decision logic (ask a human, check a policy); here an exact-match allowlist
    allowed = call.args_as_dict()['path'] in {'tmp/cache.txt'}
    # True approves; ToolDenied(message) denies and tells the model why
    results.approvals[call.tool_call_id] = allowed or ToolDenied('That file may not be deleted')

# Resume from the history plus the decisions; no new prompt is needed
result = agent.run_sync(message_history=messages, deferred_tool_results=results)
print(result.output)
```

Two key rules:

- `DeferredToolRequests` must be in the output type
- for conditional approval, raise `ApprovalRequired(...)` instead of marking the whole tool `requires_approval=True` — from the tool function, or from its `args_validator=` so invalid arguments are rejected before a human is asked (see below)

Deferred batches also surface in the event stream: `DeferredToolRequestsEvent` carries the `DeferredToolRequests` once per batch, before any `HandleDeferredToolCalls` handler runs; `DeferredToolResultsEvent` carries the `DeferredToolResults` when a handler resolves requests inline (not when results are provided to a new run via `deferred_tool_results`). A realtime session emits both the same way, with each deferred call as its own batch.

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

The retry is recorded as that call's own `ToolReturnPart` with `outcome='retried'`, carrying the `ModelRetry` message or the validation errors, and reaches the model on the provider's error channel. Its `retry` (a `RetryInfo`) says why in `cause` (a `ToolRetryCause`): `'validation_error'`, `'model_retry'`, or `'unknown_tool'` / `'unavailable_tool'` for a call rejected before any tool ran, which no tool hook sees. `attempt` is which retry of that tool this is (from 1) and `max_retries` the limit, both `None` when the retry isn't charged to the budget. None of it is sent to the model. Match these when you read history: `RetryPromptPart` is deprecated and no longer emitted, and a retry that answers no tool call is a `RetryFeedbackPart` instead.

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

The failure is recorded in message history as a `ToolReturnPart` with `outcome='failed'` — the other error outcome beside `'retried'` — and traced as an error in telemetry. Pydantic AI uses the provider's native error field where one exists; otherwise model-visible content is JSON-framed as `{"error": ...}` so the failure remains explicit.

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

Whether a file returned from a tool can travel inside the tool result depends on the model and the file's media type. It goes inside the tool result on Anthropic and OpenAI Responses (images and documents), Gemini 3 (the types in `google_supported_mime_types_in_tool_returns`), and Bedrock (the media kinds a family supports). Everything else — OpenAI Chat Completions, Groq, Mistral, xAI, Hugging Face, Gemini 2.5 and earlier, plus a media type Gemini 3 or Bedrock can't carry, such as audio or video on Gemini 3 — travels on the user channel instead, framed as `<tool_result tool_name="..." tool_call_id="..." file_id="...">` ... `</tool_result>` — one set of tags per file — so the model doesn't read tool output as something the user uploaded; the tool result carries `See file <id>.` in its place, except on a failed Gemini return, whose native `error` string takes no file references. A file the model can't read at all, as its profile's file-input flags (`supports_image_input`, `supports_document_input`, `supported_document_media_types`, `supports_audio_file_input`, `supports_video_input`) say, is replaced in the tool result by a note such as `File d9a13f (audio/mpeg, 48,213 bytes) was not sent: this model cannot read audio files.` instead of failing the request: audio and video on Anthropic and OpenAI Responses, any file on Cohere, and so on. `FileUnderstanding` describes such a file in text first when configured. A file in a user prompt or in `ToolReturn.content` is sent as it is and still fails a model that can't take it. The framing is applied while the request is built — the file itself stays on the `ToolReturnPart` in history — and it is prompt text, so it says where content came from without attesting it. `ToolReturn.content` is sent unframed: that is user content you chose to add.

Set `tools=['tool_name']` when the call makes a tool declared with `defer_loading=True` available. The executor deduplicates names in first-occurrence order, omits names already revealed, and stores a `ToolAvailabilityDeltaPart` immediately after that call's `ToolReturnPart`. The recorded name remains revealed when history is resumed; an unknown or already-visible name is a no-op when rendered.

Every searchable deferred tool stays in the search corpus after discovery. A `CompactionPart` resets prospective discovery at its exact position, so future requests reveal pre-boundary tools again. For a call in the response currently being dispatched, earlier evidence still counts when the serving provider did not honor that boundary on the request wire; otherwise a call without visible evidence is refused with a "not available yet" retry.

## Add Tools Mid-Run Without Busting the Prompt Cache

Tools may join a run after its first request: `FunctionToolset.add_function()`/`add_tool()` called from inside a tool, a dynamic toolset (`@agent.toolset`) returning more, an MCP server's `tools/list_changed`, or `prepare_tools`/a per-tool `prepare` letting one through. Don't hand-write a `ToolAvailabilityDeltaPart` for these. Each step, the framework compares the visible function tools with those the run started with, records every newcomer as a `ToolAvailabilityDeltaPart` beside that step's tool results, and sets `ModelRequestParameters.introduced_tool_names`. Only immediately available function tools count: a newcomer with `defer_loading=True` joins the tool search corpus and is revealed when a search finds it.

- First-party OpenAI Responses (`OpenAIProvider`) delivers the newcomer as an `additional_tools` item, so `tools[]` stays byte-identical.
- Anthropic models with tool search (Sonnet 4.5+, Opus 4.5+, Haiku 4.5+) deliver it as a `defer_loading` declaration plus a `tool_addition` block (or a synthesized `tool_reference` result). This is stable once a deferred tool is already declared; some models (e.g. Opus 4.8) add a one-time preamble with the first deferred declaration.
- Everything else, including older Anthropic models and OpenAI-compatible Responses endpoints, appends it to `tools[]` and adds a one-line announcement.

Exception: when no plainly visible tool is left to keep stable (the run started with no visible function, output or native tools, or every tool it started with has since left), the tools present go straight into `tools[]` with no delta, which also keeps Anthropic from receiving a request whose tools are all deferred.

Replays, retries and later runs over the same history don't re-announce it. A compaction during the run re-announces it once without changing `tools[]`; a later run that starts after the compaction declares it in `tools[]` from its first request. The comparison is per run, so a tool that is new to a later run (e.g. unlocked by an approval) goes into `tools[]` from that run's first request. Callability is unchanged: the tool is callable as soon as it exists. Under Temporal, tools run in activities, which aren't re-executed on replay, so drive mid-run changes from workflow-side code (a `prepare_tools` keyed on run state, a dynamic toolset) rather than from inside a tool.

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

For HTTP request retries at the transport layer, use the library's retry configuration separately: pass a retrying transport such as `AsyncHTTPX2TenacityTransport` to `create_async_httpx2_client(transport=...)` (from `pydantic_ai.models`), which keeps Pydantic AI's client timeouts and user agent, and give the client to the provider as `http_client`. Do not assume `ModelRetry` alone solves provider transport failures. The provider SDK client retries too, on top of the transport; set `model_settings={'max_retries': 0}` to leave retrying to the transport alone, or another count to change the SDK's.

## Recognize Tools by Kind, Not Name

Tool names can be renamed or prefixed, so never recognize a tool by `tool_name`. Match framework tools with `isinstance` on their typed parts (`ToolSearchCallPart`, `LoadCapabilityReturnPart`, ...).

To make your own tool recognizable, define a `ToolKind` with a namespace, a name and optionally the shapes of the tool's arguments and result. Its id, `'{namespace}.{name}'`, is stored on parts, so never rename it. Register typed parts for it with the `kind` class argument (no `@dataclass` needed), parameterizing the base with the kind's shape, and pass the kind as the tool's `tool_kind`:

```python
from typing_extensions import TypedDict

from pydantic_ai import Agent, ToolKind
from pydantic_ai.messages import ToolCallPart, ToolReturnPart


class LookupArgs(TypedDict):
    sku: str


class LookupResult(TypedDict):
    in_stock: bool


lookup = ToolKind('inventory', 'lookup', args=LookupArgs, content=LookupResult)


class LookupCallPart(ToolCallPart[LookupArgs], kind=lookup):
    pass


class LookupReturnPart(ToolReturnPart[LookupResult], kind=lookup):
    pass


agent = Agent('openai:gpt-5.2')


@agent.tool_plain(tool_kind=lookup)
def check_stock(args: LookupArgs) -> LookupResult:
    return {'in_stock': args['sku'].startswith('A')}
```

- The function can take the kind's `args` shape as its one parameter (the model still sees its fields as parameters) or the fields as separate parameters (`def check_stock(sku: str) -> LookupResult`).
- `part.typed_args` is the validated `LookupArgs` and `part.typed_content` the validated `LookupResult`, or `None` while the arguments stream in or if they don't fit. Don't redeclare `args` or `content`.
- The base's type parameter must match the kind's shape (`ToolCallPart[LookupArgs]` for `args=LookupArgs`); a mismatch is a type error and raises `UserError` when the class is defined. A kind without shapes takes a bare `ToolCallPart` / `ToolReturnPart`.
- A typed part may only add properties. Any field raises `UserError` when the class is defined.
- `tool_kind` is accepted by `Tool`, the `@agent.tool` / `@agent.tool_plain` / `FunctionToolset.tool` decorators and `ToolDefinition` (e.g. `replace(tool_def, tool_kind=lookup)` in `prepare`), as the `ToolKind` or its id.
- A run raises `UserError` for a kind that isn't defined, or when the tool's parameters don't fit `args` or its return type doesn't fit `content` (a required field missing or optional, or a mismatched JSON type). Type checkers check the return type already.
- Call and return parts are promoted whenever a `ModelResponse` / `ModelRequest` is built, so hooks and history processors can use `isinstance(part, LookupCallPart)`.
- A history whose kind isn't defined still loads, as base parts that keep their `tool_kind`.

See [Typed tool parts](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#typed-tool-parts).

To make a tool step aside when a more general tool is on the agent, set `ToolDefinition.unless_tool_kind` to that tool's kind (e.g. `replace(tool_def, unless_tool_kind=file_read)` in `prepare`); to name a tool in instructions or tool results, use `ctx.tool_name_for_kind(file_read)` instead of a hard-coded name:

- The tool is left out of the step when another tool in it has that kind: not sent to the model, not callable, and not returned by `tool_name_for_kind`. It's the local-tool counterpart of `unless_native`.
- A deferred (`defer_loading`) tool counts as present, so the answer doesn't change when it is revealed.
- A tool is only left out in favor of a tool that stays; when tools would each be left out in favor of the other, the one listed first stays. The kind doesn't need to be defined: only ids are compared, so an unknown kind means there's nothing to step aside for.
- `tool_name_for_kind` returns the prefixed name, `None` when no tool has the kind, and the first in `ctx.tools` order when several do. Instruction functions see the current step's tools; `get_tools` / `prepare_tools` see the previous step's.

See [Superseding a tool by kind](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#unless-tool-kind).

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
