# Troubleshooting

Below are suggestions on how to fix some common errors you might encounter while using Pydantic AI. If the issue you're experiencing is not listed below or addressed in the documentation, please feel free to ask in the [Pydantic Slack](help.md) or create an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues).

## Jupyter Notebook Errors

### `RuntimeError: This event loop is already running`

**Modern Jupyter/IPython (7.0+)**: This environment supports top-level `await` natively. You can use [`Agent.run()`][pydantic_ai.agent.Agent.run] directly in notebook cells without additional setup:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')
result = await agent.run('Who let the dogs out?')
```

**Legacy environments or specific integrations**: If you encounter event loop conflicts, use [`nest-asyncio`](https://pypi.org/project/nest-asyncio/):

```python {test="skip"}
import nest_asyncio

from pydantic_ai import Agent

nest_asyncio.apply()

agent = Agent('openai:gpt-5.2')
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

## Upgrading from Older Versions

The [V1 → V2 Migration Map](migration.md) lists every renamed or removed name. The entries below cover the errors people search for most.

### `TypeError: Agent.__init__() got an unexpected keyword argument 'result_type'`

`result_type` was renamed to `output_type` before V1, and the other `result_*` names were renamed with it. Use the current names:

| Old | Current |
| --- | --- |
| `Agent(result_type=...)` | `Agent(output_type=...)`, see [Output](output.md) |
| `result.data` | [`result.output`][pydantic_ai.agent.AgentRunResult.output] |
| `@agent.result_validator` | [`@agent.output_validator`][pydantic_ai.agent.Agent.output_validator] |
| `Agent(result_retries=N)`, `Agent(output_retries=N)` | `Agent(retries={'output': N})`, see [Output retries](retries.md#output-retries) |

`output_retries=` is also gone as of V2, where both retry budgets moved onto the single [`retries`][pydantic_ai.agent.AgentRetries] argument, so a `TypeError` for `output_retries` has the same fix.

### [`UserError`][pydantic_ai.exceptions.UserError]: `Unknown model: gpt-5.2. Did you mean 'openai-chat:gpt-5.2'?`

As of V2, a model name needs a provider prefix: the V1 fallback that guessed the provider from a bare name now raises this error. Write `'openai:gpt-5.2'` (the Responses API), `'openai-chat:gpt-5.2'` (Chat Completions) or `'anthropic:claude-opus-5'`. Some prefixes were renamed too, such as `google-gla:` to `google:` and `google-vertex:` to `google-cloud:`. See [Model name prefixes](migration.md#model-name-prefixes) and [Model Providers](models/overview.md).

### `ImportError: cannot import name 'OpenAIModel' from 'pydantic_ai.models.openai'`

V2 removed the model and MCP classes that V1 had deprecated. `OpenAIModel` is now [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel], `GeminiModel` (and the `pydantic_ai.models.gemini` module) is now [`GoogleModel`][pydantic_ai.models.google.GoogleModel], and `MCPServerStdio`, `MCPServerSSE` and `MCPServerStreamableHTTP` are all [`MCPToolset`][pydantic_ai.mcp.MCPToolset]. Look up any other name in the [V1 → V2 Migration Map](migration.md).

### `ImportError: Please install boto3 to use the Bedrock model`

Each provider's SDK is an optional dependency, and the error names the extra that installs it. As of V2, a bare `pip install pydantic-ai` no longer includes `bedrock`, `groq`, `mistral`, `cohere`, `xai`, `huggingface`, `temporal`, `ag-ui`, `ui` or `spec`, so code that worked on V1 can hit this after upgrading. Add the extras you use, for example `uv add 'pydantic-ai[bedrock,groq]'`. [Slim Install](install.md#slim-install) has the full list.

An `ImportError` naming a class inside a third-party package, such as `cannot import name 'ThinkingEndEvent' from 'ag_ui.core'`, usually means that package is older than the version Pydantic AI requires. Upgrade it along with Pydantic AI.

## Model Providers

### [`UserError`][pydantic_ai.exceptions.UserError]: `Model '...' is not served by the Bedrock Converse API`

Bedrock serves models through two APIs: Converse on the `bedrock-runtime` endpoint (the `bedrock:` prefix), and Mantle, an OpenAI-compatible API (the `bedrock-mantle:` prefix). Some OpenAI models, such as GPT-5.4 and GPT-5.5, are available only on Mantle, so `bedrock:` refuses them. Use `Agent('bedrock-mantle:openai.gpt-5.5')` and install the `bedrock-mantle` extra. Both routes use the same AWS credentials. [OpenAI model routes](models/bedrock.md#bedrock-openai-model-routes) lists which models each route serves, and [Bedrock Mantle](models/bedrock.md#bedrock-mantle) covers its setup.

If AWS has since started serving the model on Converse, upgrade Pydantic AI: the list of Mantle-only models is updated as AWS adds models to Converse.

### [`UserError`][pydantic_ai.exceptions.UserError]: `Anthropic does not support extended thinking and output tools at the same time`

Anthropic's manual extended thinking can't be combined with a forced tool call, and [Tool Output](output.md#tool-output) forces one. Bedrock raises the same error for Claude models. You only get the error with an explicit `ToolOutput(...)`: a plain structured `output_type` switches away from tool output by itself. To fix it:

- Use `output_type=NativeOutput(...)` ([Native Output](output.md#native-output)), or [`PromptedOutput`](output.md#prompted-output) on models without JSON schema support.
- On models that support it, use adaptive thinking (`anthropic_thinking={'type': 'adaptive'}`), which works with output tools.

See [Forced tool choice](models/anthropic.md#forced-tool-choice), and for Bedrock, [Thinking and structured output](models/bedrock.md#thinking-and-structured-output).

### [`UserError`][pydantic_ai.exceptions.UserError]: `CachePoint cannot be the first content in a user message`

A [`CachePoint`][pydantic_ai.messages.CachePoint] marks the content before it for caching, so it needs content ahead of it in the same user message. To cache the instructions or tool definitions, don't put a `CachePoint` at the start of the prompt: use the `anthropic_cache_instructions` and `anthropic_cache_tool_definitions` settings, or `bedrock_cache_instructions` and `bedrock_cache_tool_definitions` on Bedrock. See prompt caching for [Anthropic](models/anthropic.md#prompt-caching) and [Bedrock](models/bedrock.md#prompt-caching).

## Usage Limits and Retries

### [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded]: `The next request would exceed the request_limit of 50`

Every run has a default [`request_limit`][pydantic_ai.usage.UsageLimits.request_limit] of 50 model requests, which stops a model that loops on tool calls from running forever. Long agentic runs with many tool calls can reach it legitimately. Raise it with `usage_limits=UsageLimits(request_limit=200)` on `run()`, `run_sync()` or `run_stream()`, or pass `request_limit=None` to remove it. [Usage Limits](agent.md#usage-limits) covers the other limits: tokens, tool calls and cost.

### [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior]: `Tool 'my_tool' exceeded max retries count of 1`

The model called `my_tool` with arguments that failed validation, or the tool raised [`ModelRetry`][pydantic_ai.exceptions.ModelRetry], more times than the tool's retry budget allows. The default budget is `1` retry per tool. Raise it for one tool with `@agent.tool(retries=N)`, or for every tool with `Agent(retries={'tools': N})`. If the model keeps sending the same wrong arguments, a clearer docstring or argument descriptions usually help more than extra retries. To see what the model sent, wrap the run in [`capture_run_messages()`][pydantic_ai.capture_run_messages] as shown in [Model errors](agent.md#model-errors). [Tool retries](tools-advanced.md#tool-retries) explains which retry setting wins.

### [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior]: `Exceeded maximum output retries (1)`

The model's final answer failed validation against your `output_type`, or an [output validator](output.md#output-validator-functions) raised `ModelRetry`, more often than the output retry budget allows. A response with no usable output, such as empty text, also uses up this budget. The default is `1`: raise it with `Agent(retries={'output': N})`, or per run with `agent.run(..., retries={'output': N})`. If an empty answer is acceptable, make the output type optional (`output_type=str | None`) so that an empty response ends the run with `None` instead of a retry. See [Output retries](retries.md#output-retries), and use [`capture_run_messages()`][pydantic_ai.capture_run_messages] to inspect the response that failed.

## Message History and Deferred Tools

### [`UserError`][pydantic_ai.exceptions.UserError]: `Cannot provide a new user prompt when the message history contains unprocessed tool calls.`

The `message_history` you passed ends with a model response whose tool calls have no results yet, and you also passed a new prompt. Either run without a new prompt, which executes the pending tool calls and continues the run, or, if the calls were [deferred](deferred-tools.md), pass their results with `deferred_tool_results=...`. A history left behind by a cancelled or interrupted run is repaired automatically; see [Making histories provider-valid](message-history.md#making-histories-provider-valid).

### [`UserError`][pydantic_ai.exceptions.UserError]: `Tool call results need to be provided for all deferred tool calls.`

When you resume a run that ended with [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests], the [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] you pass need one entry for each pending call, keyed by its `tool_call_id`: approvals go in `results.approvals` and externally executed results in `results.calls`. The error prints the IDs it expected and the ones it got, so compare the two. See [Human-in-the-Loop Tool Approval](deferred-tools.md#human-in-the-loop-tool-approval) and [External Tool Execution](deferred-tools.md#external-tool-execution).

## Durable Execution

### [`UserError`][pydantic_ai.exceptions.UserError]: `Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) need to have a unique id in order to be used with Temporal`

Durable execution names each toolset's activities after the toolset's `id`, so that a restarted workflow runs the same code. Give every `FunctionToolset`, `MCPToolset` and `DynamicToolset` a stable `id`. When a capability contributes the toolset, set the capability's `id` instead, for example `Capability(id='search', tools=[...])` or `MCP(url='...', id='docs')`. Prefect and DBOS have the same requirement. See [Agent Names and Toolset IDs](durable_execution/temporal.md#agent-names-and-toolset-ids).

### [`UserError`][pydantic_ai.exceptions.UserError]: `FunctionToolset cannot be added at runtime with Temporal`

A toolset that runs its own tools needs its activities registered with the worker before the workflow starts, so it can't be added later through `run(toolsets=...)`, `override(toolsets=...)` or `@agent.toolset`. Pass it to the agent constructor instead. Toolsets whose tools run outside the agent, like [`ExternalToolset`][pydantic_ai.toolsets.ExternalToolset], can still be passed per run. See [Toolsets at Runtime](durable_execution/temporal.md#toolsets-at-runtime).

## Monitoring HTTPX Requests

You can use custom `httpx2` (or legacy `httpx`) clients in your models in order to access specific requests, responses, and headers at runtime.

It's particularly helpful to use `logfire`'s [HTTPX integration](logfire.md#monitoring-http-requests) to monitor the above.
