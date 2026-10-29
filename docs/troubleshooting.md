---
description: "Fix common Pydantic AI errors: event loop errors in Jupyter, missing API keys and extras, V1 to V2 upgrade errors, Bedrock model routes, provider HTTP errors, usage limits and retries, unprocessed tool calls, Temporal toolset IDs, and errors fixed in past versions."
---

# Troubleshooting

Below are suggestions on how to fix some common errors you might encounter while using Pydantic AI. If the issue you're experiencing is not listed below or addressed in the documentation, please feel free to ask in the [Pydantic Slack](help.md) or create an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues).

## Jupyter Notebook Errors

### `RuntimeError: This event loop is already running`

**Modern Jupyter/IPython (7.0+)**: This environment supports top-level `await` natively. You can use [`Agent.run()`][pydantic_ai.agent.Agent.run] directly in notebook cells without additional setup:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent

agent = Agent('{{ openai:medium }}')
result = await agent.run('Who let the dogs out?')
```

**Legacy environments or specific integrations**: If you encounter event loop conflicts, use [`nest-asyncio`](https://pypi.org/project/nest-asyncio/):

```python {test="skip"}
import nest_asyncio

from pydantic_ai import Agent

nest_asyncio.apply()

agent = Agent('{{ openai:medium }}')
result = agent.run_sync('Who let the dogs out?')
```

**Note**: This also applies to Google Colab and [Marimo](https://github.com/marimo-team/marimo) environments.

## `RuntimeError: Event loop is closed`

Synchronous methods like [`Agent.run_sync()`][pydantic_ai.agent.AbstractAgent.run_sync] reuse the thread's current event loop, and install a fresh one if other code closed it. If this error is raised from inside `httpx2` (or legacy `httpx`) during a model request, the agent was already used before its event loop was closed: the provider's HTTP connection pool still holds connections bound to the dead loop. Recreate the agent together with its model and provider (or pass a fresh `http_client` to the provider); reusing an existing `Model` instance keeps the dead connection pool. Avoid closing an event loop that other code is still using.

## [`UserError`][pydantic_ai.exceptions.UserError]: `Agent.run_sync()` and `Agent.run_stream_sync()` cannot be used inside a synchronous tool, output function, or other function called during an agent run

This error means a synchronous [tool](tools.md), [output function](output.md#output-functions), or other function called during an agent run tried to start a nested run with [`Agent.run_sync()`][pydantic_ai.agent.AbstractAgent.run_sync] or [`Agent.run_stream_sync()`][pydantic_ai.agent.AbstractAgent.run_stream_sync]. The sync run methods can only be used from regular application code, outside of a run: inside one, the parent run is still waiting on your function while the nested run blocks it, which can deadlock, so Pydantic AI raises this error instead.

Make the delegating function `async def` and `await` the inner run, as shown in [Agent delegation](multi-agent-applications.md#agent-delegation). The parent agent can still be started with `run_sync()` from normal synchronous application code. If the delegating function also needs to do blocking work, push just that part into [`asyncio.to_thread()`][asyncio.to_thread].

## API Key Configuration

### [`UserError`][pydantic_ai.exceptions.UserError]: Set the `[PROVIDER]_API_KEY` environment variable or pass it via the provider's `api_key=...` argument

If you're running into issues with setting the API key for your model, visit the [Models](models/overview.md) page to learn more about how to set an environment variable and/or pass in an `api_key` argument.

To try Pydantic AI without an API key, use the built-in [`'test'` model](testing.md#unit-testing-with-testmodel): [`Agent('test')`][pydantic_ai.agent.Agent].

## Upgrading from V1

The [V1 → V2 Migration Map](migration.md) lists every renamed or removed name. If a library you depend on breaks on V2, check whether it declares `pydantic-ai>=1` without an upper bound: until it supports V2, pin `pydantic-ai<2` in your own project.

### `ImportError: cannot import name 'OpenAIModel' from 'pydantic_ai.models.openai'`

V2 removed the names V1 had deprecated. The ones code most often still imports are `OpenAIModel` (now [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]), `MCPServerStdio`, `MCPServerSSE` and `MCPServerStreamableHTTP` (now [`MCPToolset`][pydantic_ai.mcp.MCPToolset]), `Usage` from `pydantic_ai.usage` (now [`RunUsage`][pydantic_ai.usage.RunUsage]) and `GeminiModel` (now [`GoogleModel`][pydantic_ai.models.google.GoogleModel]). Look up any other name in the [Migration Map](migration.md).

### `TypeError: Agent.__init__() got an unexpected keyword argument 'result_type'`

`result_type` is the pre-1.0 name of `output_type`, and `result.data` the pre-1.0 name of [`result.output`][pydantic_ai.agent.AgentRunResult.output]. If the error names `output_retries` or `tool_retries` instead, use the V2 [`retries`][pydantic_ai.agent.AgentRetries] argument: `Agent(retries={'output': N, 'tools': N})`. See [Agent configuration](migration.md#agent-configuration) in the Migration Map.

### `TypeError: 'RunUsage' object is not callable`

In V2, `usage` on a run result is a property, not a method: write `result.usage` instead of `result.usage()`. If your code wraps the call in `try`/`except Exception`, the error is swallowed and the usage you record stays empty, so search for `.usage()` after upgrading.

### `AttributeError: 'FunctionToolResultEvent' object has no attribute 'result'`

V2 renamed the field to `part`: read [`event.part`][pydantic_ai.messages.ToolResultEvent.part], a [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart]. See [Messages, events and usage](migration.md#messages-events-and-usage) for the other event renames.

### `AttributeError: 'dict' object has no attribute 'supports_thinking'`

[`ModelProfile`][pydantic_ai.profiles.ModelProfile] is a `TypedDict` in V2, so read a field with `profile.get('supports_thinking')` instead of `profile.supports_thinking`. [Model profiles](migration.md#model-profiles) in the Migration Map has the replacements for setting and merging fields.

## Installation and Dependencies

### ``` ImportError: Please install `boto3` to use the Bedrock model ```

Each provider's SDK is an optional dependency, and the error names the extra that installs it. As of V2, a bare `pip install pydantic-ai` no longer includes `bedrock`, `groq`, `mistral`, `cohere`, `xai`, `huggingface`, `temporal`, `ag-ui`, `ui` or `spec`, so code that worked on V1 can hit this after upgrading. Add the extras you use, for example `uv add 'pydantic-ai[bedrock,groq]'`. [Slim Install](install.md#slim-install) has the full list.

If you already installed the extra the error names, install the missing package directly as a workaround, and please [report it](https://github.com/pydantic/pydantic-ai/issues): the extra should have pulled it in.

## Model Providers

### [`UserError`][pydantic_ai.exceptions.UserError]: `Model '...' is not served by the Bedrock Converse API`

Pydantic AI keeps a list of the OpenAI models that Bedrock serves on its Converse API (the `bedrock:` prefix) and refuses the others, which AWS serves only through Bedrock Mantle (the `bedrock-mantle:` prefix). When AWS starts serving a new model on Converse, the list catches up in a release, so upgrade first: Converse support arrived for GPT-5.6 Sol, Luna and Terra in v2.45.0, for GPT-6 Sol, Luna and Astra in v2.49.0, and for GPT-6.1 Sol in v2.55.0. If the model is Mantle-only, use the `bedrock-mantle:` prefix and install the `bedrock-mantle` extra; see [OpenAI model routes](models/bedrock.md#bedrock-openai-model-routes) and [Bedrock Mantle](models/bedrock.md#bedrock-mantle).

If AWS already serves the model on Converse and no release knows it yet, override the check with a partial profile: `BedrockConverseModel(model_name, profile={'bedrock_supported_on_converse': True})`. A partial `profile=` dict is merged over the provider's defaults for that model, so the same approach works on any provider when a new model gets the wrong capabilities. See [Models, providers, and profiles](models/overview.md#models-and-providers).

### [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError]: `status_code: 400, model_name: ..., body: ...`

The provider rejected the request, and `body` is the provider's own error response, also available as the exception's [`body`][pydantic_ai.exceptions.ModelAPIError.body]. Read it first: it usually names the field or setting the provider didn't accept.

To handle specific failures, catch the error categories rather than matching on the message text, which differs between providers: [`ModelContextWindowExceededError`][pydantic_ai.exceptions.ModelContextWindowExceededError] for "maximum context length" or "prompt is too long", [`ModelRateLimitError`][pydantic_ai.exceptions.ModelRateLimitError] for rate limits, [`ModelQuotaExceededError`][pydantic_ai.exceptions.ModelQuotaExceededError] for exhausted credits or billing, and [`ModelAuthenticationError`][pydantic_ai.exceptions.ModelAuthenticationError] for a 401 or 403. An HTTP error in one of these categories is still a `ModelHTTPError` too, so existing `except ModelHTTPError` blocks keep working. The first three arrived in v2.59.0 and `ModelAuthenticationError` in v2.62.0. See [Handling Model API Errors](models/overview.md#handling-model-api-errors).

## Usage Limits and Retries

### [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded]: `The next request would exceed the request_limit of 50`

Every run has a default [`request_limit`][pydantic_ai.usage.UsageLimits.request_limit] of 50 model requests, which stops a model that loops on tool calls from running forever. Long agentic runs can reach it legitimately. Raise it with `usage_limits=UsageLimits(request_limit=200)` on `run()`, `run_sync()` or `run_stream()`, or pass `request_limit=None` to remove it.

An agent run from inside a tool or capability with `usage=ctx.usage` counts its requests against the parent run's usage, but without `usage_limits` it applies the default limit of 50 to that shared count. Pass `usage_limits=ctx.usage_limits` along with `usage=ctx.usage`, or limits of its own. See [Usage Limits](agent.md#usage-limits) and [Agent delegation](multi-agent-applications.md#agent-delegation).

### [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior]: `Tool 'my_tool' exceeded max retries count of 1`

The model called `my_tool` with arguments that failed validation, or the tool raised [`ModelRetry`][pydantic_ai.exceptions.ModelRetry], more times than the tool's retry budget allows. The default budget is `1` retry per tool. Raise it for one tool with `@agent.tool(retries=N)`, or for every tool with `Agent(retries={'tools': N})`. If the model keeps sending the same wrong arguments, a clearer docstring or argument descriptions usually help more than extra retries. To see what the model sent, wrap the run in [`capture_run_messages()`][pydantic_ai.capture_run_messages] as shown in [Model errors](agent.md#model-errors). [Tool retries](tools-advanced.md#tool-retries) explains which retry setting wins.

### [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior]: `Exceeded maximum output retries (1)`

The model's final answer failed validation against your `output_type`, or an [output validator](output.md#output-validator-functions) raised `ModelRetry`, more often than the output retry budget allows. A response with no usable output, such as empty text or plain text when the output type needs structured data, also uses up this budget. The default is `1`: raise it with `Agent(retries={'output': N})`, or per run with `agent.run(..., retries={'output': N})`.

Smaller and local models often answer in text instead of calling the output tool. For those, try [`NativeOutput`](output.md#native-output) where the model supports it, or [`PromptedOutput`](output.md#prompted-output). If an empty answer is acceptable, make the output type optional (`output_type=str | None`) so that an empty response ends the run with `None`; see [Optional output](output.md#optional-output). Use [`capture_run_messages()`][pydantic_ai.capture_run_messages] to inspect the response that failed, and see [Output retries](retries.md#output-retries).

## Message History

### [`UserError`][pydantic_ai.exceptions.UserError]: `Cannot provide a new user prompt when the message history contains unprocessed tool calls.`

The `message_history` you passed ends with a model response whose tool calls have no results yet, and you also passed a new prompt. This usually happens when your application saved the history of a run that stopped after the model asked for tools but before they ran. Either run without a new prompt, which executes the pending tool calls and continues the run; pass their results with `deferred_tool_results=...` if the calls were [deferred](deferred-tools.md); or, to abandon them, pass the history through [`repair_messages()`][pydantic_ai.messages.repair_messages] first. See [Making histories provider-valid](message-history.md#making-histories-provider-valid).

Before v2.38.0, a run cancelled during its first tool call also left a history that raised this error. Upgrade if that is how you got here.

## Durable Execution

### [`UserError`][pydantic_ai.exceptions.UserError]: ```Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) need to have a unique `id` in order to be used with Temporal```

Durable execution names each toolset's activities after the toolset's `id`, so that a restarted workflow runs the same code. Give every toolset that runs its own tools a stable `id`: `FunctionToolset(id=...)`, `MCPToolset(..., id=...)` or `DynamicToolset(id=...)`. When a capability contributes the toolset, set the capability's `id` instead, for example `Capability(id='search', tools=[...])` or `MCP(url='...', id='docs')`. A capability function passed directly to `capabilities=` can't carry an `id`, so wrap it: `DynamicCapability(my_func, id='...')`. DBOS and Prefect raise the same error under their own names. See [Agent Names and Toolset IDs](durable_execution/temporal.md#agent-names-and-toolset-ids).

## Observability

### A banner saying `observability: off` appears in logs or in your application's output

Until instrumentation is configured, the first agent run in a process prints a short banner to `stderr`. It's only shown to a terminal or a coding agent, never under `pytest` or when `CI` is set. If your application or library captures `stderr`, set the `PYDANTIC_AI_NO_BANNER` environment variable to any value, or set `pydantic_ai.BANNER_ENABLED = False` before the first agent run. Configuring [instrumentation](logfire.md) also turns it off. See [The first-run banner](logfire.md#the-first-run-banner).

## Errors Fixed in Past Versions

These errors came from bugs that have since been fixed. If you still see one, you may be on an older version: check `pydantic_ai.__version__` and upgrade.

### `TypeError: Cannot instantiate typing.Union`

Depending on the Python version, the same error reads `TypeError: 'typing.Union' object is not callable`. Every `AnthropicModel` request failed this way with `anthropic` 2.x installed, because Pydantic AI v2.59.0 and older didn't support it while allowing it to be installed. Fixed in v2.60.0. If you can't upgrade yet, pin `anthropic<2`.

### [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError]: `status_code: 400, ... 'message': 'Schema is too complex.'`

Anthropic compiles strict schemas, including the output schema of Native Output, into a grammar and rejects them past its size limits. Since v2.63.0, Pydantic AI checks the documented limits before sending, and a structured `output_type` without an explicit output mode falls back to a non-strict output tool when its schema would be too large. If you still see the error on v2.63.0 or later, its message carries a hint: simplify the schema, or use `ToolOutput` without `strict=True` or `PromptedOutput`. See [Schema complexity limits](models/anthropic.md#schema-complexity-limits).

### [`UserError`][pydantic_ai.exceptions.UserError]: `CachePoint cannot be the first content in a user message`

The [Planning](harness/planning.md) and [SystemReminders](harness/system-reminders.md) capabilities of Pydantic AI Harness could put a [`CachePoint`][pydantic_ai.messages.CachePoint] at the start of a user message. On Bedrock this was fixed in Pydantic AI v2.28.0, and on OpenRouter and in the capabilities themselves in v2.57.0 (Pydantic AI Harness 0.57.0). If you place a `CachePoint` at the start of a prompt yourself, it has nothing before it to cache: to cache the instructions or tool definitions, use the `anthropic_cache_instructions` and `anthropic_cache_tool_definitions` settings, or their `bedrock_` equivalents. See [prompt caching](models/anthropic.md#prompt-caching).

### [`UserError`][pydantic_ai.exceptions.UserError]: `Anthropic does not support thinking and output tools at the same time.`

Older versions raised this for [Tool Output](output.md#tool-output) whenever thinking was on, including adaptive thinking on Claude models, which supports it. Fixed for `AnthropicModel` in v2.27.1, and for Claude on Bedrock, where the message starts with `Bedrock does not support`, in v2.46.0.

## Monitoring HTTPX Requests

You can use custom `httpx2` (or legacy `httpx`) clients in your models in order to access specific requests, responses, and headers at runtime.

It's particularly helpful to use `logfire`'s [HTTPX integration](logfire.md#monitoring-http-requests) to monitor the above.
