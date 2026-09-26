# DeepSeek

Use DeepSeek through its Chat Completions or Responses API. The `deepseek:` prefix selects Chat Completions; use [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] explicitly for Responses.

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

agent = Agent('deepseek:deepseek-v4-flash')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

model = OpenAIChatModel(
    'deepseek-v4-flash',
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
    'deepseek-v4-flash',
    provider=DeepSeekProvider(
        api_key='your-deepseek-api-key', http_client=custom_http_client
    ),
)
agent = Agent(model)
...
```

OpenAI-compatible providers also accept a legacy `httpx.AsyncClient` during Pydantic AI v2, but emit a deprecation warning. Use `httpx2.AsyncClient` for new code; legacy HTTPX client support will be removed in Pydantic AI v3.

## Structured output and thinking

DeepSeek's V4 models think by default, and DeepSeek rejects a forced tool choice while thinking is on, answering `Thinking mode does not support this tool_choice`. Pydantic AI therefore sends `tool_choice='auto'` on those requests, which leaves the model free to answer in prose instead of calling the output tool — on `deepseek-v4-pro` that costs a retry often enough to exhaust the retry budget. Turn thinking off when you need [structured output](../output.md) to be reliable, and forcing is used again:

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

## Responses API

As an alternative to the Chat Completions API shown above, DeepSeek also serves an OpenAI-compatible [Responses API](openai.md#responses-api-features) for [both V4 models](https://api-docs.deepseek.com/guides/responses_api). Use it by pairing [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] with [`DeepSeekProvider`][pydantic_ai.providers.deepseek.DeepSeekProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

model = OpenAIResponsesModel(
    'deepseek-v4-flash',
    provider=DeepSeekProvider(api_key='your-deepseek-api-key'),
)
agent = Agent(model)
...
```

DeepSeek [documents](https://api-docs.deepseek.com/guides/responses_api) which parts of the Responses API it implements, and unsupported fields are silently ignored rather than rejected, so it's worth knowing what does nothing:

- The API is stateless, so [`openai_conversation_id`](openai.md#using-durable-conversations), [background mode](openai.md#background-mode) and [message compaction](openai.md#message-compaction) are unavailable. Pass [message history](../message-history.md) back on each run instead.
- Leave [`openai_previous_response_id`](openai.md#referencing-earlier-responses) unset. Setting it makes Pydantic AI drop the earlier turns it assumes the server already holds, and DeepSeek stores nothing, so the model silently loses the conversation instead of erroring.
- Of the [native tools](../native-tools.md), DeepSeek runs only [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool]; it ignores the others instead of reporting an error.
- Image and document inputs are replaced with placeholder text rather than rejected.
- Reasoning is configured with `openai_reasoning_effort` (or the unified [`thinking`](../capabilities/thinking.md) setting); `openai_reasoning_summary` is accepted but produces no summary.
- [`NativeOutput`][pydantic_ai.output.NativeOutput] is available here but not on Chat Completions: DeepSeek honors a strict JSON Schema on the Responses API, while its Chat Completions endpoint rejects one with `This response_format type is unavailable now`.

The one difference Pydantic AI handles for you: DeepSeek merges each function call into its adjacent assistant message. Replaying a turn that interleaves calls with thinking or text would therefore create separate messages with unanswered calls, which DeepSeek rejects with `No tool output found for tool call ...`. Pydantic AI moves the calls after the other items when building the request. This reorders only the request; your [message history](../message-history.md) is unchanged.

Reordering applies only when every function call has a result and the turn contains no provider-owned native tool or compaction items. DeepSeek rejects unresolved calls in any order, while provider-owned items are left unchanged. Set [`OpenAIModelProfile.openai_responses_supports_interleaved_function_calls`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_responses_supports_interleaved_function_calls] on your own profile if you serve DeepSeek's Responses shape from another endpoint, or to turn the reordering off.
