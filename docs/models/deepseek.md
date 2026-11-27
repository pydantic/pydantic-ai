---
description: "Use DeepSeek V4 models with Pydantic AI through the Chat Completions or Responses API, with thinking, structured output and the deepseek: prefix."
---

# DeepSeek

Use DeepSeek through its Chat Completions or Responses API. The `deepseek:` prefix selects Chat Completions; use [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] explicitly for Responses.

Use `deepseek-flash` for the current Flash model. DeepSeek also accepts the legacy `deepseek-v4-flash` name and [serves it with the current Flash model](https://api-docs.deepseek.com/quick_start/pricing/).

## Install

Install Pydantic AI with the OpenAI SDK used by this integration:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

To use the [DeepSeek](https://deepseek.com) provider, first create an API key by following the [Quick Start guide](https://api-docs.deepseek.com/).

You can then set the `DEEPSEEK_API_KEY` environment variable and use [`DeepSeekProvider`][pydantic_ai.providers.deepseek.DeepSeekProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('deepseek:deepseek-flash')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

model = OpenAIChatModel(
    'deepseek-flash',
    provider=DeepSeekProvider(api_key='your-deepseek-api-key'),
)
agent = Agent(model)
...
```

You can customize the HTTP client:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

custom_http_client = AsyncClient(timeout=30)
model = OpenAIChatModel(
    'deepseek-flash',
    provider=DeepSeekProvider(
        api_key='your-deepseek-api-key', http_client=custom_http_client
    ),
)
agent = Agent(model)
...
```

OpenAI-compatible providers also accept a legacy `httpx.AsyncClient` during Pydantic AI v2, but emit a deprecation warning. Use `httpx2.AsyncClient` for new code; legacy HTTPX client support will be removed in Pydantic AI v3.

DeepSeek only enforces [strict tool definitions](../tools-advanced.md#strict-mode) on its beta base URL, and accepts but ignores `strict` on the default one. To use strict mode, pass a client for that URL: `DeepSeekProvider(openai_client=AsyncOpenAI(base_url='https://api.deepseek.com/beta', api_key=...))`.

## Structured output and thinking

`deepseek-flash` and DeepSeek's V4 models think by default, and DeepSeek rejects a forced tool choice while thinking is on, answering `Thinking mode does not support this tool_choice`. Pydantic AI therefore sends `tool_choice='auto'` on those requests, which leaves the model free to answer in prose instead of calling the output tool — on `deepseek-v4-pro` that costs a retry often enough to exhaust the retry budget. Turn thinking off when you need [structured output](../output.md) to be reliable, and forcing is used again:

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings


class Answer(BaseModel):
    text: str


agent = Agent(
    OpenAIChatModel('deepseek-v4-pro', provider='deepseek'),
    output_type=Answer,
    model_settings=OpenAIChatModelSettings(thinking=False),
)
...
```

Passing `tool_choice='required'` explicitly while thinking is on raises a [`UserError`][pydantic_ai.exceptions.UserError] rather than failing at the API.

DeepSeek also rejects a thinking request when a tool-calling assistant turn after the last user message lacks `reasoning_content`, such as one served by another model or with thinking off. Whenever the request thinks, [`DeepSeekProvider`][pydantic_ai.providers.deepseek.DeepSeekProvider] sends the field empty on every tool-calling turn that lacks it. Set [`OpenAIModelProfile.openai_chat_thinking_requires_field_on_tool_calls`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_chat_thinking_requires_field_on_tool_calls] on your own profile, alongside `openai_chat_send_back_thinking_parts='field'` and `openai_chat_thinking_field='reasoning_content'`, if you serve DeepSeek's Chat Completions shape from another endpoint.

## Responses API

As an alternative to the Chat Completions API shown above, DeepSeek also serves an OpenAI-compatible [Responses API](openai.md#responses-api-features) for [both V4 models](https://api-docs.deepseek.com/guides/responses_api). Use it by pairing [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] with [`DeepSeekProvider`][pydantic_ai.providers.deepseek.DeepSeekProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

model = OpenAIResponsesModel(
    'deepseek-flash',
    provider=DeepSeekProvider(api_key='your-deepseek-api-key'),
)
agent = Agent(model)
...
```

DeepSeek [documents](https://api-docs.deepseek.com/guides/responses_api) which parts of the Responses API it implements, and unsupported fields are silently ignored rather than rejected, so it's worth knowing what does nothing:

- The API is stateless, so [`openai_conversation_id`](openai.md#using-durable-conversations), [background mode](openai.md#background-mode) and [message compaction](openai.md#message-compaction) are unavailable. Pass [message history](../message-history.md) back on each run instead.
- Leave [`openai_previous_response_id`](openai.md#referencing-earlier-responses) unset. Setting it makes Pydantic AI drop the earlier turns it assumes the server already holds, and DeepSeek stores nothing, so the model silently loses the conversation instead of erroring.
- [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] uses the Responses `web_search` declaration with `deepseek-v4-pro`. See [Native web tools](#native-web-tools) for Flash.
- Image and document inputs are replaced with placeholder text rather than rejected.
- Reasoning is configured with `openai_reasoning_effort` (or the unified [`thinking`](../capabilities/thinking.md) setting); `openai_reasoning_summary` is accepted but produces no summary.
- [`NativeOutput`][pydantic_ai.output.NativeOutput] with a strict JSON Schema is supported on Responses for requests without native web tools. Chat Completions rejects a strict JSON Schema with `This response_format type is unavailable now`.

The one difference Pydantic AI handles for you: DeepSeek merges each function call into its adjacent assistant message. Replaying a turn that interleaves calls with thinking or text would therefore create separate messages with unanswered calls, which DeepSeek rejects with `No tool output found for tool call ...`. Pydantic AI moves the calls after the other items when building the request. This reorders only the request; your [message history](../message-history.md) is unchanged.

Reordering applies only when every function call has a result and the turn contains no provider-owned native tool or compaction items. DeepSeek rejects unresolved calls in any order, while provider-owned items are left unchanged. Set [`OpenAIModelProfile.openai_responses_supports_interleaved_function_calls`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_responses_supports_interleaved_function_calls] on your own profile if you serve DeepSeek's Responses shape from another endpoint, or to turn the reordering off.

## Native web tools

With `deepseek-flash`, [`WebSearch()`][pydantic_ai.capabilities.WebSearch] enables `deepseek_web_search`, and [`WebFetch()`][pydantic_ai.capabilities.WebFetch] enables `deepseek_read_url` on the Responses API. The alias `deepseek-v4-flash` also supports both tools. The model chooses when to search or read a link. URL reading considers links in the last user message. With thinking enabled, web search runs in deep search mode and decides which pages to open. This mode takes precedence over URL reading when both tools are declared. See the [DeepSeek web tools guide](https://api-docs.deepseek.com/guides/web_tools/).

The tools have no domain filters or caller-specified use limits. Keep the complete message history to let DeepSeek restore signed page context in later turns, including turns with web tools disabled. Use the Responses API for those turns.

DeepSeek Responses requires a text response format when these tools are enabled. Use plain text or [`ToolOutput`][pydantic_ai.output.ToolOutput] for these requests. Leave `openai_top_logprobs` unset.

```python {title="deepseek_web_fetch.py" test="skip" gateway_toggle="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.openai import OpenAIResponsesModel

model = OpenAIResponsesModel('deepseek-flash', provider='deepseek')
agent = Agent(model, capabilities=[WebFetch()])
result = agent.run_sync('Read https://example.com and summarize the page.')
print(result.output)
```
