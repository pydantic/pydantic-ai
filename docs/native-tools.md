# Native Tools

Native tools are native tools provided by LLM providers that can be used to enhance your agent's capabilities. Unlike [common tools](common-tools.md), which are custom implementations that Pydantic AI executes, native tools are executed directly by the model provider.

## Overview

Pydantic AI supports the following native tools:

- **[`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool]**: Allows agents to search the web
- **[`XSearchTool`][pydantic_ai.native_tools.XSearchTool]**: Allows agents to search X/Twitter (xAI only)
- **[`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool]**: Enables agents to execute code in a secure environment
- **[`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool]**: Enables agents to generate images
- **[`WebFetchTool`][pydantic_ai.native_tools.WebFetchTool]**: Enables agents to fetch web pages
- **[`MemoryTool`][pydantic_ai.native_tools.MemoryTool]**: Enables agents to use memory
- **[`MCPServerTool`][pydantic_ai.native_tools.MCPServerTool]**: Enables agents to use remote MCP servers with communication handled by the model provider
- **[`FileSearchTool`][pydantic_ai.native_tools.FileSearchTool]**: Enables agents to search through uploaded files using vector search (RAG)
- **[`AdvisorTool`][pydantic_ai.native_tools.AdvisorTool]**: Lets a faster executor model consult a stronger advisor model mid-generation (Anthropic, OpenRouter)

These tools are passed to the agent's `capabilities` list, wrapped in [`NativeTool`][pydantic_ai.capabilities.NativeTool], and are executed by the model provider's infrastructure.

!!! warning "Provider Support"
    Not all model providers support native tools. If you use a native tool with an unsupported provider, Pydantic AI will raise a [`UserError`][pydantic_ai.exceptions.UserError] when you try to run the agent.

    If a provider supports a native tool that is not currently supported by Pydantic AI, please file an issue.

!!! tip "Provider-adaptive capabilities"
    For a higher-level, model-agnostic approach, consider the [provider-adaptive tool capabilities](capabilities/overview.md#provider-adaptive-tools): [`WebSearch`][pydantic_ai.capabilities.WebSearch], [`WebFetch`][pydantic_ai.capabilities.WebFetch], [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration], and [`MCP`][pydantic_ai.capabilities.MCP]. These automatically use the model's native tool when supported and fall back to a local implementation, so your agent works across providers without code changes.

### Google tool combinations

[Gemini 3 models](https://ai.google.dev/gemini-api/docs/structured-output#structured_outputs_with_tools) support combining native tools with function tools, including [output tools](output.md#tool-output), and [`NativeOutput`][pydantic_ai.output.NativeOutput]. Earlier Gemini models cannot use these combinations; use [`PromptedOutput`][pydantic_ai.output.PromptedOutput] for structured output alongside native tools.

## Dynamic Configuration

Sometimes you need to configure a native tool dynamically based on the [run context][pydantic_ai.tools.RunContext] (e.g., user dependencies), or conditionally omit it. You can achieve this by wrapping a function with [`NativeTool`][pydantic_ai.capabilities.NativeTool] in `capabilities`. The function takes [`RunContext`][pydantic_ai.tools.RunContext] as an argument and returns an [`AbstractNativeTool`][pydantic_ai.native_tools.AbstractNativeTool] or `None`.

This is particularly useful for tools like [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] where you might want to set the user's location based on the current request, or disable the tool if the user provides no location.

```python {title="dynamic_native_tool.py"}
from pydantic_ai import Agent, RunContext, WebSearchTool
from pydantic_ai.capabilities import NativeTool


async def prepared_web_search(ctx: RunContext[dict]) -> WebSearchTool | None:
    if not ctx.deps.get('location'):
        return None

    return WebSearchTool(
        user_location={'city': ctx.deps['location']},
    )

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[NativeTool(prepared_web_search)],
    deps_type=dict,
)

# Run with location
result = agent.run_sync(
    'What is the weather like?',
    deps={'location': 'London'},
)
print(result.output)
#> It's currently raining in London.

# Run without location (tool will be omitted)
result = agent.run_sync(
    'What is the capital of France?',
    deps={'location': None},
)
print(result.output)
#> The capital of France is Paris.
```

## Web Search Tool

!!! tip
    For a model-agnostic approach with automatic local fallback, see the [`WebSearch`][pydantic_ai.capabilities.WebSearch] [capability](capabilities/overview.md#provider-adaptive-tools).

The [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] allows your agent to search the web,
making it ideal for queries that require up-to-date data.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| OpenAI Responses | ✅ | Full feature support. To include search results on the [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] that's available via [`ModelResponse.native_tool_calls`][pydantic_ai.messages.ModelResponse.native_tool_calls], enable the [`OpenAIResponsesModelSettings.openai_include_web_search_sources`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_include_web_search_sources] [model setting](agent.md#model-run-settings). |
| Anthropic | ✅ | Full feature support |
| Google | ✅ | No parameter support. No [`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart] or [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] is generated when streaming. See [Google tool combinations](#google-tool-combinations). |
| xAI | ✅ | Supports `blocked_domains`, `allowed_domains`, and `user_location` parameters. |
| Groq | ✅ | Limited parameter support. To use web search capabilities with Groq, you need to use the [compound models](https://console.groq.com/docs/compound). |
| OpenRouter | ✅ | Web search via [plugins](https://openrouter.ai/docs/features/web-search). Supports `search_context_size`. Uses native search for supported providers (OpenAI, Anthropic, Perplexity, xAI), Exa for others. |
| OpenAI Chat Completions | ❌ | Not supported |
| Bedrock | ❌ | Not supported |
| Mistral | ❌ | Not supported |
| Cohere | ❌ | Not supported |
| HuggingFace | ❌ | Not supported |

### Usage

```py {title="web_search_anthropic.py"}
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[NativeTool(WebSearchTool())])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.
```

_(This example is complete, it can be run "as is")_

With OpenAI, you must use their Responses API to access the web search tool.

```py {title="web_search_openai.py"}
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent('openai-responses:gpt-5.2', capabilities=[NativeTool(WebSearchTool())])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.
```

_(This example is complete, it can be run "as is")_

### Configuration Options

The `WebSearchTool` supports several configuration parameters:

```py {title="web_search_configured.py"}
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        NativeTool(
            WebSearchTool(
                search_context_size='high',
                user_location=WebSearchUserLocation(
                    city='San Francisco',
                    country='US',
                    region='CA',
                    timezone='America/Los_Angeles',
                ),
                blocked_domains=['example.com', 'spam-site.net'],
                allowed_domains=None,  # Cannot use both blocked_domains and allowed_domains with Anthropic
                max_uses=5,  # Anthropic only: limit tool usage
            )
        )
    ],
)

result = agent.run_sync('Use the web to get the current time.')
print(result.output)
#> In San Francisco, it's 8:21:41 pm PDT on Wednesday, August 6, 2025.
```

_(This example is complete, it can be run "as is")_

#### Provider Support

| Parameter | OpenAI | Anthropic | xAI | Groq | OpenRouter |
|-----------|--------|-----------|-----|------|------------|
| `search_context_size` | ✅ | ❌ | ❌ | ❌ | ✅ |
| `user_location` | ✅ | ✅ | ✅ | ❌ | ❌ |
| `blocked_domains` | ❌ | ✅ | ✅ | ✅ | ❌ |
| `allowed_domains` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `max_uses` | ❌ | ✅ | ❌ | ❌ | ❌ |

!!! note "Anthropic Domain Filtering"
    With Anthropic, you can only use either `blocked_domains` or `allowed_domains`, not both.

!!! note "Anthropic Web Search Tool Versions"
    Pydantic AI does not expose a `dynamic_filtering` option. For Anthropic, Pydantic AI selects
    the web search tool version from the model profile and Anthropic client: `web_search_20260209`
    for models and platforms that support Anthropic's dynamic-filtering web tools, and
    `web_search_20250305` otherwise.
    The legacy Amazon Bedrock client does not support Anthropic web search, so Pydantic AI raises
    a `UserError` if you use `WebSearchTool` with `AsyncAnthropicBedrock`.
    On Vertex AI, `WebSearchTool` always uses `web_search_20250305`, as Anthropic does not offer the
    dynamic-filtering version there, so dynamic filtering is unavailable even on otherwise-supported models.
    See the [Anthropic web search docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)
    and [tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)
    for current model support and platform availability.

    Add [`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool] only when you want
    Anthropic's standalone code execution tool; it is not needed to use `web_search_20260209`.
    For Zero Data Retention behavior with `_20260209` web tools, see Anthropic's
    [server tools docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools#zdr-and-allowed-callers).

## X Search Tool

!!! tip
    For a model-agnostic approach with a subagent fallback, see the [`XSearch`][pydantic_ai.capabilities.XSearch] [capability](capabilities/overview.md#provider-adaptive-tools).

The [`XSearchTool`][pydantic_ai.native_tools.XSearchTool] allows your agent to search X/Twitter for real-time posts and content. Natively supported by xAI models; usable on other models via the [`XSearch`][pydantic_ai.capabilities.XSearch] capability with `fallback_model` set. See the [xAI X Search documentation](https://docs.x.ai/developers/tools/x-search) for more details.

### Usage

```py {title="x_search_xai.py"}
from pydantic_ai import Agent, XSearchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent('xai:grok-4.3', capabilities=[NativeTool(XSearchTool())])

result = agent.run_sync('What are people saying about AI on X today?')
print(result.output)
#> There's a lot of excitement about new AI models being released...
```

_(This example is complete, it can be run "as is")_

### Configuration Options

The `XSearchTool` supports several configuration parameters:

```py {title="x_search_configured.py"}
from datetime import datetime

from pydantic_ai import Agent, XSearchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'xai:grok-4.3',
    capabilities=[
        NativeTool(
            XSearchTool(
                allowed_x_handles=['OpenAI', 'AnthropicAI', 'dasfacc'],
                from_date=datetime(2024, 1, 1),
                to_date=datetime(2024, 12, 31),
                enable_image_understanding=True,
                enable_video_understanding=True,
            )
        )
    ],
)

result = agent.run_sync('What have AI companies been posting about?')
print(result.output)
"""
OpenAI announced their latest model updates, while Anthropic shared research on AI safety...
"""
```

_(This example is complete, it can be run "as is")_

!!! note "Handle Filtering"
    You can only use one of `allowed_x_handles` or `excluded_x_handles`, not both. Each list is limited to 20 handles maximum.

!!! note "Including raw search results"
    By default, xAI only returns the model's text summary of the search. To get programmatic access to the underlying posts, sources, and metadata, set `include_output=True` on [`XSearchTool`][pydantic_ai.native_tools.XSearchTool] (analogous to [`OpenAIResponsesModelSettings.openai_include_web_search_sources`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_include_web_search_sources] for OpenAI web search). The raw results are then available on the [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] exposed via [`ModelResponse.native_tool_calls`][pydantic_ai.messages.ModelResponse.native_tool_calls]. As an alternative, you can enable it globally via the [`XaiModelSettings.xai_include_x_search_output`][pydantic_ai.models.xai.XaiModelSettings.xai_include_x_search_output] [model setting](agent.md#model-run-settings). See the [xAI docs](models/xai.md#x-search) for the recommended `XSearch` capability-based approach.

## Code Execution Tool

The [`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool] enables your agent to execute code
in a secure environment, making it perfect for computational tasks, data analysis, and mathematical operations.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| OpenAI | ✅ | To include code execution output on the [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] that's available via [`ModelResponse.native_tool_calls`][pydantic_ai.messages.ModelResponse.native_tool_calls], enable the [`OpenAIResponsesModelSettings.openai_include_code_execution_outputs`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_include_code_execution_outputs] [model setting](agent.md#model-run-settings). If the code execution generated images, like charts, they will be available on [`ModelResponse.images`][pydantic_ai.messages.ModelResponse.images] as [`BinaryImage`][pydantic_ai.messages.BinaryImage] objects. The generated image can also be used as [image output](output.md#image-output) for the agent run. |
| Google | ✅ | See [Google tool combinations](#google-tool-combinations). |
| Anthropic | ✅ | Available on compatible Anthropic models. Pydantic AI selects a compatible code execution tool version automatically; see [Anthropic code execution tool version](models/anthropic.md#code-execution-tool-version) to override it. |
| xAI | ✅ | Full feature support. |
| Groq | ❌ | |
| Bedrock | ✅ | Only available for Nova 2.0 models. |
| Mistral | ❌ | |
| Cohere | ❌ | |
| HuggingFace | ❌ | |

### Usage

```py {title="code_execution_basic.py"}
from pydantic_ai import Agent, CodeExecutionTool
from pydantic_ai.capabilities import NativeTool

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[NativeTool(CodeExecutionTool())])

result = agent.run_sync('Calculate the factorial of 15.')
print(result.output)
#> The factorial of 15 is **1,307,674,368,000**.
print(result.response.native_tool_calls)
"""
[
    (
        NativeToolCallPart(
            tool_name='code_execution',
            args={'command': 'python3 -c "import math; print(math.factorial(15))"'},
            tool_call_id='srvtoolu_017qRH1J3XrhnpjP2XtzPCmJ',
            provider_name='anthropic',
            provider_details={'anthropic_tool_name': 'bash_code_execution'},
        ),
        NativeToolReturnPart(
            tool_name='code_execution',
            content={
                'content': [],
                'return_code': 0,
                'stderr': '',
                'stdout': '1307674368000\n',
                'type': 'bash_code_execution_result',
            },
            tool_call_id='srvtoolu_017qRH1J3XrhnpjP2XtzPCmJ',
            timestamp=datetime.datetime(...),
            provider_name='anthropic',
            provider_details={'anthropic_tool_name': 'bash_code_execution'},
        ),
    )
]
"""
```

_(This example is complete, it can be run "as is")_

In addition to text output, code execution with OpenAI can generate images as part of their response. Accessing this image via [`ModelResponse.images`][pydantic_ai.messages.ModelResponse.images] or [image output](output.md#image-output) requires the [`OpenAIResponsesModelSettings.openai_include_code_execution_outputs`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_include_code_execution_outputs] [model setting](agent.md#model-run-settings) to be enabled.

```py {title="code_execution_openai.py"}
from pydantic_ai import Agent, BinaryImage, CodeExecutionTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[NativeTool(CodeExecutionTool())],
    output_type=BinaryImage,
    model_settings=OpenAIResponsesModelSettings(openai_include_code_execution_outputs=True),
)

result = agent.run_sync('Generate a chart of y=x^2 for x=-5 to 5.')
assert isinstance(result.output, BinaryImage)
```

_(This example is complete, it can be run "as is")_

### File Uploads

You can upload files via the provider's Files API and make them available to the code execution container. This allows the agent to process data files, analyze CSVs, work with images, and more.
Files whose [`UploadedFile.provider_name`][pydantic_ai.messages.UploadedFile.provider_name] does not match the model provider are ignored.

#### Anthropic

```py {title="code_execution_with_files_anthropic.py" test="skip" line_length="120"}
import asyncio

import anthropic

from pydantic_ai import Agent, CodeExecutionTool, UploadedFile
from pydantic_ai.capabilities import NativeTool


async def main():
    # Upload a file via the Anthropic Files API
    client = anthropic.AsyncAnthropic()
    with open('data.csv', 'rb') as f:
        file = await client.beta.files.upload(file=('data.csv', f.read(), 'text/csv'), betas=['files-api-2025-04-14'])

    # Create an agent with CodeExecutionTool that has access to the uploaded file
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[NativeTool(CodeExecutionTool(files=[UploadedFile(file_id=file.id, provider_name='anthropic')]))],
    )

    result = await agent.run('Analyze the data.csv file and summarize the key statistics.')
    print(result.output)
    #> The CSV file contains 1000 rows with columns: name, age, salary...


asyncio.run(main())
```

For details on file management, persistence, and container behavior, see the [Anthropic Files API documentation](https://platform.claude.com/docs/en/build-with-claude/files).

#### OpenAI

```py {title="code_execution_with_files_openai.py" test="skip" line_length="120"}
import asyncio

from openai import AsyncOpenAI

from pydantic_ai import Agent, CodeExecutionTool, UploadedFile
from pydantic_ai.capabilities import NativeTool


async def main():
    # Upload a file via the OpenAI Files API
    client = AsyncOpenAI()
    with open('data.csv', 'rb') as f:
        file = await client.files.create(file=f, purpose='assistants')

    # Create an agent with CodeExecutionTool that has access to the uploaded file
    agent = Agent(
        'openai-responses:gpt-5.2',
        capabilities=[NativeTool(CodeExecutionTool(files=[UploadedFile(file_id=file.id, provider_name='openai')]))],
    )

    result = await agent.run('Analyze the data.csv file and summarize the key statistics.')
    print(result.output)
    #> The CSV file contains 1000 rows with columns: name, age, salary...


asyncio.run(main())
```

For details on file management, container lifecycle, and persistence behavior, see the [OpenAI Responses API documentation](https://platform.openai.com/docs/api-reference/responses).

#### Provider Support

| Parameter | Anthropic | OpenAI | Google | xAI |
|-----------|-----------|--------|--------|-----|
| `files` | ✅ | ✅ | ❌ | ❌ |

## Image Generation Tool

!!! tip
    For a model-agnostic approach with automatic local fallback, see the [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] [capability](capabilities/overview.md#provider-adaptive-tools).

The [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] enables your agent to generate images.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| OpenAI Responses | ✅ | Full feature support. Only supported by models newer than `gpt-5.2`. Metadata about the generated image, like the [`revised_prompt`](https://platform.openai.com/docs/guides/tools-image-generation#revised-prompt) sent to the underlying image model, is available on the [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] that's available via [`ModelResponse.native_tool_calls`][pydantic_ai.messages.ModelResponse.native_tool_calls]. |
| Google | ✅ | Limited parameter support. Only supported by [image generation models](https://ai.google.dev/gemini-api/docs/image-generation) like `gemini-3-pro-image-preview` and `gemini-3-pro-image-preview`. These models do not support [function tools](tools.md) and will always have the option of generating images, even if this native tool is not explicitly specified. |
| Anthropic | ❌ | |
| xAI | ❌ | |
| Groq | ❌ | |
| Bedrock | ❌ | |
| Mistral | ❌ | |
| Cohere | ❌ | |
| HuggingFace | ❌ | |

### Usage

Generated images are available on [`ModelResponse.images`][pydantic_ai.messages.ModelResponse.images] as [`BinaryImage`][pydantic_ai.messages.BinaryImage] objects:

```py {title="image_generation_openai.py"}
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool
from pydantic_ai.capabilities import NativeTool

agent = Agent('openai-responses:gpt-5.2', capabilities=[NativeTool(ImageGenerationTool())])

result = agent.run_sync('Tell me a two-sentence story about an axolotl with an illustration.')
print(result.output)
"""
Once upon a time, in a hidden underwater cave, lived a curious axolotl named Pip who loved to explore. One day, while venturing further than usual, Pip discovered a shimmering, ancient coin that granted wishes!
"""

assert isinstance(result.response.images[0], BinaryImage)
```

_(This example is complete, it can be run "as is")_

Image generation with Google [image generation models](https://ai.google.dev/gemini-api/docs/image-generation) does not require the `ImageGenerationTool` native tool to be explicitly specified:

```py {title="image_generation_google.py"}
from pydantic_ai import Agent, BinaryImage

agent = Agent('google:gemini-3-pro-image-preview')

result = agent.run_sync('Tell me a two-sentence story about an axolotl with an illustration.')
print(result.output)
"""
Once upon a time, in a hidden underwater cave, lived a curious axolotl named Pip who loved to explore. One day, while venturing further than usual, Pip discovered a shimmering, ancient coin that granted wishes!
"""

assert isinstance(result.response.images[0], BinaryImage)
```

_(This example is complete, it can be run "as is")_

The `ImageGenerationTool` can be used together with `output_type=BinaryImage` to get [image output](output.md#image-output). If the `ImageGenerationTool` native tool is not explicitly specified, it will be enabled automatically:

```py {title="image_generation_output.py"}
from pydantic_ai import Agent, BinaryImage

agent = Agent('openai-responses:gpt-5.2', output_type=BinaryImage)

result = agent.run_sync('Generate an image of an axolotl.')
assert isinstance(result.output, BinaryImage)
```

_(This example is complete, it can be run "as is")_

### Configuration Options

The `ImageGenerationTool` supports several configuration parameters:

```py {title="image_generation_configured.py"}
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[
        NativeTool(
            ImageGenerationTool(
                action='generate',
                background='transparent',
                input_fidelity='high',
                model='gpt-image-2',
                moderation='low',
                output_compression=100,
                output_format='png',
                partial_images=3,
                quality='high',
                size='1024x1024',
            )
        )
    ],
    output_type=BinaryImage,
)

result = agent.run_sync('Generate an image of an axolotl.')
assert isinstance(result.output, BinaryImage)
```

_(This example is complete, it can be run "as is")_

OpenAI Responses models also respect the `aspect_ratio` parameter. Because the OpenAI API only exposes discrete image sizes,
Pydantic AI maps `'1:1'` -> `1024x1024`, `'2:3'` -> `1024x1536`, and `'3:2'` -> `1536x1024`. Providing any other aspect ratio
results in an error, and if you also set `size` it must match the computed value.

The OpenAI Responses image generation tool defaults to `action='auto'`, where the model decides whether to generate a new
image or edit one already in context. Use `action='generate'` or `action='edit'` to force either behavior. You can also set
`model` to select the underlying image generation model used by the tool, for example `model='gpt-image-2'`; this does not
change the agent's conversational model.

To control the aspect ratio when using Gemini image models, include the `ImageGenerationTool` explicitly:

```py {title="image_generation_google_aspect_ratio.py"}
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'google:gemini-3-pro-image-preview',
    capabilities=[NativeTool(ImageGenerationTool(aspect_ratio='16:9'))],
    output_type=BinaryImage,
)

result = agent.run_sync('Generate a wide illustration of an axolotl city skyline.')
assert isinstance(result.output, BinaryImage)
```

_(This example is complete, it can be run "as is")_

To control the image resolution with Google image generation models (starting with Gemini 3 Pro Image), use the `size` parameter:

```py {title="image_generation_google_resolution.py"}
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'google:gemini-3-pro-image-preview',
    capabilities=[NativeTool(ImageGenerationTool(aspect_ratio='16:9', size='4K'))],
    output_type=BinaryImage,
)

result = agent.run_sync('Generate a high-resolution wide landscape illustration of an axolotl.')
assert isinstance(result.output, BinaryImage)
```

_(This example is complete, it can be run "as is")_

For more details, check the [API documentation][pydantic_ai.native_tools.ImageGenerationTool].

#### Provider Support

| Parameter | OpenAI | Google |
|-----------|--------|--------|
| `action` | ✅ (auto (default), generate, edit) | ❌ |
| `background` | ✅ | ❌ |
| `input_fidelity` | ✅ | ❌ |
| `moderation` | ✅ | ❌ |
| `model` | ✅ (gpt-image-2, gpt-image-1.5, gpt-image-1, gpt-image-1-mini, or another OpenAI image model ID) | ❌ |
| `output_compression` | ✅ (100 (default), jpeg or webp only) | ✅ (75 (default), jpeg only, Google Cloud only) |
| `output_format` | ✅ | ✅ (Google Cloud only) |
| `partial_images` | ✅ | ❌ |
| `quality` | ✅ | ❌ |
| `size` | ✅ (auto (default), 1024x1024, 1024x1536, 1536x1024) | ✅ (512, 1K (default), 2K, 4K) |
| `aspect_ratio` | ✅ (1:1, 2:3, 3:2) | ✅ (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9) |

!!! note "Notes"
    - **OpenAI**: `auto` lets the model select the value.
    - **Google Cloud**: Setting `output_compression` will default `output_format` to `jpeg` if not specified.

## Web Fetch Tool

!!! tip
    For a model-agnostic approach with automatic local fallback, see the [`WebFetch`][pydantic_ai.capabilities.WebFetch] [capability](capabilities/overview.md#provider-adaptive-tools).

The [`WebFetchTool`][pydantic_ai.native_tools.WebFetchTool] enables your agent to pull URL contents into its context,
allowing it to pull up-to-date information from the web.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| Anthropic | ✅ | Full feature support. Uses Anthropic's [Web Fetch Tool](https://docs.claude.com/en/docs/agents-and-tools/tool-use/web-fetch-tool) internally to retrieve URL contents. |
| Google | ✅ | No parameter support. The limits are fixed at 20 URLs per request with a maximum of 34MB per URL. See [Google tool combinations](#google-tool-combinations). |
| xAI | ❌ | Web browsing is implemented as part of [`WebSearchTool`](#web-search-tool) with xAI. |
| OpenAI | ❌ | |
| Groq | ❌ | |
| Bedrock | ❌ | |
| Mistral | ❌ | |
| Cohere | ❌ | |
| HuggingFace | ❌ | |

### Usage

```py {title="web_fetch_basic.py"}
from pydantic_ai import Agent, WebFetchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent('google:gemini-3-flash-preview', capabilities=[NativeTool(WebFetchTool())])

result = agent.run_sync('What is this? https://ai.pydantic.dev')
print(result.output)
#> A Python agent framework for building Generative AI applications.
```

_(This example is complete, it can be run "as is")_

### Configuration Options

The `WebFetchTool` supports several configuration parameters:

```py {title="web_fetch_configured.py"}
from pydantic_ai import Agent, WebFetchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        NativeTool(
            WebFetchTool(
                allowed_domains=['ai.pydantic.dev', 'docs.pydantic.dev'],
                max_uses=10,
                enable_citations=True,
                max_content_tokens=50000,
            )
        )
    ],
)

result = agent.run_sync(
    'Compare the documentation at https://ai.pydantic.dev and https://docs.pydantic.dev'
)
print(result.output)
"""
Both sites provide comprehensive documentation for Pydantic projects. ai.pydantic.dev focuses on PydanticAI, a framework for building AI agents, while docs.pydantic.dev covers Pydantic, the data validation library. They share similar documentation styles and both emphasize type safety and developer experience.
"""
```

_(This example is complete, it can be run "as is")_

#### Provider Support

| Parameter | Anthropic | Google |
|-----------|-----------|--------|
| `max_uses` | ✅ | ❌ |
| `allowed_domains` | ✅ | ❌ |
| `blocked_domains` | ✅ | ❌ |
| `enable_citations` | ✅ | ❌ |
| `max_content_tokens` | ✅ | ❌ |

!!! note "Anthropic Domain Filtering"
    With Anthropic, you can only use either `blocked_domains` or `allowed_domains`, not both.

!!! note "Anthropic Web Fetch Tool Versions"
    Pydantic AI does not expose a `dynamic_filtering` option. For Anthropic, Pydantic AI selects
    the web fetch tool version from the model profile and Anthropic client: `web_fetch_20260209`
    for models and platforms that support Anthropic's dynamic-filtering web tools, and
    `web_fetch_20250910` otherwise.
    `WebFetchTool` is unavailable on the legacy Amazon Bedrock and Vertex AI Anthropic clients, so
    Pydantic AI raises a `UserError` if you use it with `AsyncAnthropicBedrock` or
    `AsyncAnthropicVertex`.
    See the
    [Anthropic web fetch docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool)
    and [tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)
    for current model support and platform availability.

    Add [`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool] only when you want
    Anthropic's standalone code execution tool; it is not needed to use `web_fetch_20260209`.
    For Zero Data Retention behavior with `_20260209` web tools, see Anthropic's
    [server tools docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools#zdr-and-allowed-callers).

## Memory Tool

The [`MemoryTool`][pydantic_ai.native_tools.MemoryTool] enables your agent to use memory.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| Anthropic | ✅ | Requires a tool named `memory` to be defined that implements [specific sub-commands](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool#tool-commands). You can use a subclass of [`anthropic.lib.tools.BetaAbstractMemoryTool`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/tools/_beta_builtin_memory_tool.py) as documented below. |
| Google | ❌ | |
| OpenAI | ❌ | |
| Groq | ❌ | |
| Bedrock | ❌ | |
| Mistral | ❌ | |
| Cohere | ❌ | |
| HuggingFace | ❌ | |

### Usage

The Anthropic SDK provides an abstract [`BetaAbstractMemoryTool`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/tools/_beta_builtin_memory_tool.py) class that you can subclass to create your own memory storage solution (e.g., database, cloud storage, encrypted files, etc.). Their [`LocalFilesystemMemoryTool`](https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/memory/basic.py) example can serve as a starting point.

The following example uses a subclass that hard-codes a specific memory. The bits specific to Pydantic AI are the `MemoryTool` native tool and the `memory` tool definition that forwards commands to the `call` method of the `BetaAbstractMemoryTool` subclass.

```py {title="anthropic_memory.py"}
from typing import Any

from anthropic.lib.tools import BetaAbstractMemoryTool
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)

from pydantic_ai import Agent, MemoryTool
from pydantic_ai.capabilities import NativeTool


class FakeMemoryTool(BetaAbstractMemoryTool):
    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        return 'The user lives in Mexico City.'

    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        return f'File created successfully at {command.path}'

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        return f'File {command.path} has been edited'

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        return f'Text inserted at line {command.insert_line} in {command.path}'

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        return f'File deleted: {command.path}'

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        return f'Renamed {command.old_path} to {command.new_path}'

    def clear_all_memory(self) -> str:
        return 'All memory cleared'

fake_memory = FakeMemoryTool()

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[NativeTool(MemoryTool())])


@agent.tool_plain
def memory(**command: Any) -> Any:
    return fake_memory.call(command)


result = agent.run_sync('Remember that I live in Mexico City')
print(result.output)
"""
Got it! I've recorded that you live in Mexico City. I'll remember this for future reference.
"""

result = agent.run_sync('Where do I live?')
print(result.output)
#> You live in Mexico City.
```

_(This example is complete, it can be run "as is")_

## Advisor Tool

The [`AdvisorTool`][pydantic_ai.native_tools.AdvisorTool] lets an executor model consult another model mid-generation. See the [Anthropic](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool) and [OpenRouter](https://openrouter.ai/docs/guides/features/server-tools/advisor) documentation for current model compatibility.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| Anthropic | ✅ | Available on the Claude API and Claude Platform on AWS. |
| OpenRouter | ✅ | Works with any executor model. |
| OpenAI | ❌ | |
| Google | ❌ | |
| xAI | ❌ | |
| Groq | ❌ | |
| Bedrock | ❌ | |
| Mistral | ❌ | |
| Cohere | ❌ | |
| HuggingFace | ❌ | |

### Usage

```py {title="advisor_anthropic.py" test="skip"}
from pydantic_ai import AdvisorTool, Agent
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[NativeTool(AdvisorTool(model='claude-opus-4-8'))],
)

result = agent.run_sync('Design a caching strategy for our API. Consult your advisor first.')
print(result.output)
```

For OpenRouter, use any `openrouter:` executor and pass an OpenRouter model slug to `model`, for example `anthropic/claude-opus-4.8`. Pydantic AI sends `forward_transcript=false`; `max_uses` and `caching` are ignored. Pydantic AI surfaces aggregate consultation counts under [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details] `['server_tool_use']`.

With Anthropic, Pydantic AI preserves plaintext and encrypted advisor results in message history, and strips advisor blocks when the tool is no longer enabled. Streaming pauses while the advisor runs. Advisor usage is reported under `advisor_*` keys in [`RequestUsage.details`][pydantic_ai.usage.RequestUsage.details] and excluded from the executor's top-level token totals.

### Configuration Options

| Parameter | Anthropic | OpenRouter |
|-----------|-----------|------------|
| `model` | ✅ (required — the advisor model to consult) | ✅ (required — an OpenRouter catalog slug) |
| `max_uses` | ✅ (cap on advisor consultations per request) | ❌ (fixed gateway limit; ignored) |
| `max_tokens` | ✅ (cap on advisor output tokens, minimum 1024; makes the result carry a `stop_reason`) | ✅ (maps to `max_completion_tokens`) |
| `caching` | ✅ (`'5m'` or `'1h'` — ephemeral caching of the advisor context) | ❌ (no equivalent; ignored) |

## MCP Server Tool

!!! tip
    For a model-agnostic approach with automatic local fallback, see the [`MCP`][pydantic_ai.capabilities.MCP] [capability](capabilities/overview.md#provider-adaptive-tools).

The [`MCPServerTool`][pydantic_ai.native_tools.MCPServerTool] allows your agent to use remote MCP servers with communication handled by the model provider.

This requires the MCP server to live at a public URL the provider can reach and does not support many of the advanced features of Pydantic AI's agent-side [MCP support](mcp/client.md),
but can result in optimized context use and caching, and faster performance due to the lack of a round-trip back to Pydantic AI.

### Provider Support

| Provider | Supported | Notes                 |
|----------|-----------|-----------------------|
| OpenAI Responses | ✅ | Full feature support. [Connectors](https://platform.openai.com/docs/guides/tools-connectors-mcp#connectors) can be used by specifying a special `x-openai-connector:<connector_id>` URL.  |
| Anthropic | ✅ | Full feature support |
| xAI | ✅ | Full feature support |
| Google  | ❌ | Not supported |
| Groq  | ❌ | Not supported |
| OpenAI Chat Completions | ❌ | Not supported |
| Bedrock | ❌ | Not supported |
| Mistral | ❌ | Not supported |
| Cohere | ❌ | Not supported |
| HuggingFace | ❌ | Not supported |

### Usage

```py {title="mcp_server_anthropic.py"}
from pydantic_ai import Agent, MCPServerTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        NativeTool(
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',  # (1)
            )
        )
    ]
)

result = agent.run_sync('Tell me about the pydantic/pydantic-ai repo.')
print(result.output)
"""
The pydantic/pydantic-ai repo is a Python agent framework for building Generative AI applications.
"""
```

1. The [DeepWiki MCP server](https://docs.devin.ai/work-with-devin/deepwiki-mcp) does not require authorization.

_(This example is complete, it can be run "as is")_

With OpenAI, you must use their Responses API to access the MCP server tool:

```py {title="mcp_server_openai.py"}
from pydantic_ai import Agent, MCPServerTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[
        NativeTool(
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',  # (1)
            )
        )
    ]
)

result = agent.run_sync('Tell me about the pydantic/pydantic-ai repo.')
print(result.output)
"""
The pydantic/pydantic-ai repo is a Python agent framework for building Generative AI applications.
"""
```

1. The [DeepWiki MCP server](https://docs.devin.ai/work-with-devin/deepwiki-mcp) does not require authorization.

_(This example is complete, it can be run "as is")_

### Configuration Options

The `MCPServerTool` supports several configuration parameters for custom MCP servers:

```py {title="mcp_server_configured_url.py"}
import os

from pydantic_ai import Agent, MCPServerTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[
        NativeTool(
            MCPServerTool(
                id='github',
                url='https://api.githubcopilot.com/mcp/',
                authorization_token=os.getenv('GITHUB_ACCESS_TOKEN', 'mock-access-token'),  # (1)
                allowed_tools=['search_repositories', 'list_commits'],
                description='GitHub MCP server',
                headers={'X-Custom-Header': 'custom-value'},
            )
        )
    ]
)

result = agent.run_sync('Tell me about the pydantic/pydantic-ai repo.')
print(result.output)
"""
The pydantic/pydantic-ai repo is a Python agent framework for building Generative AI applications.
"""
```

1. The [GitHub MCP server](https://github.com/github/github-mcp-server) requires an authorization token.

For OpenAI Responses, you can use a [connector](https://platform.openai.com/docs/guides/tools-connectors-mcp#connectors) by specifying a special `x-openai-connector:` URL:

_(This example is complete, it can be run "as is")_

```py {title="mcp_server_configured_connector_id.py"}
import os

from pydantic_ai import Agent, MCPServerTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[
        NativeTool(
            MCPServerTool(
                id='google-calendar',
                url='x-openai-connector:connector_googlecalendar',
                authorization_token=os.getenv('GOOGLE_API_KEY', 'mock-api-key'), # (1)
            )
        )
    ]
)

result = agent.run_sync('What do I have on my calendar today?')
print(result.output)
#> You're going to spend all day playing with Pydantic AI.
```

1. OpenAI's Google Calendar connector requires an [authorization token](https://platform.openai.com/docs/guides/tools-connectors-mcp#authorizing-a-connector).

_(This example is complete, it can be run "as is")_

#### Provider Support

| Parameter             | OpenAI | Anthropic | xAI |
|-----------------------|--------|-----------|-----|
| `authorization_token` | ✅ | ✅ | ✅ |
| `allowed_tools`       | ✅ | ✅ | ✅ |
| `description`         | ✅ | ❌ | ✅ |
| `headers`             | ✅ | ❌ | ✅ |

## File Search Tool

The [`FileSearchTool`][pydantic_ai.native_tools.FileSearchTool] enables your agent to search through uploaded files using vector search, providing a fully managed Retrieval-Augmented Generation (RAG) system. This tool handles file storage, chunking, embedding generation, and context injection into prompts.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| OpenAI Responses | ✅ | Full feature support. Requires files to be uploaded to vector stores via the [OpenAI Files API](https://platform.openai.com/docs/api-reference/files). To include search results on the [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] available via [`ModelResponse.native_tool_calls`][pydantic_ai.messages.ModelResponse.native_tool_calls], enable the [`OpenAIResponsesModelSettings.openai_include_file_search_results`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_include_file_search_results] [model setting](agent.md#model-run-settings). |
| Google (Gemini) | ✅ | Requires files to be uploaded via the [Gemini Files API](https://ai.google.dev/gemini-api/docs/files). Files are automatically deleted after 48 hours. Supports up to 2 GB per file and 20 GB per project. See [Google tool combinations](#google-tool-combinations). |
| xAI | ✅ | Mapped to xAI collections search. Requires collection IDs. To include search results on the [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart], enable the [`XaiModelSettings.xai_include_collections_search_output`][pydantic_ai.models.xai.XaiModelSettings.xai_include_collections_search_output] [model setting](agent.md#model-run-settings). |
|| Google Cloud | ❌ | Not supported |
| Anthropic | ❌ | Not supported |
| Groq | ❌ | Not supported |
| OpenAI Chat Completions | ❌ | Not supported |
| Bedrock | ❌ | Not supported |
| Mistral | ❌ | Not supported |
| Cohere | ❌ | Not supported |
| HuggingFace | ❌ | Not supported |

### Usage

#### OpenAI Responses

With OpenAI, you need to first [upload files to a vector store](https://platform.openai.com/docs/assistants/tools/file-search), then reference the vector store IDs when using the `FileSearchTool`.

```py {title="file_search_openai_upload.py" test="skip"}
import asyncio

from pydantic_ai import Agent, FileSearchTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.models.openai import OpenAIResponsesModel


async def main():
    model = OpenAIResponsesModel('gpt-5.2')

    with open('my_document.txt', 'rb') as f:
        file = await model.client.files.create(file=f, purpose='assistants')

    vector_store = await model.client.vector_stores.create(name='my-docs')
    await model.client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file.id
    )

    agent = Agent(
        model,
        capabilities=[NativeTool(FileSearchTool(file_store_ids=[vector_store.id]))]
    )

    result = await agent.run('What information is in my documents about pydantic?')
    print(result.output)
    #> Based on your documents, Pydantic is a data validation library for Python...

asyncio.run(main())
```

#### Google (Gemini)

With Gemini, you need to first [create a file search store via the Files API](https://ai.google.dev/gemini-api/docs/files), then reference the file search store names.

```py {title="file_search_google_upload.py" test="skip"}
import asyncio

from pydantic_ai import Agent, FileSearchTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.models.google import GoogleModel


async def main():
    model = GoogleModel('gemini-3-flash-preview')

    store = await model.client.aio.file_search_stores.create(
        config={'display_name': 'my-docs'}
    )

    with open('my_document.txt', 'rb') as f:
        await model.client.aio.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store.name,
            file=f,
            config={'mime_type': 'text/plain'}
        )

    agent = Agent(
        model,
        capabilities=[NativeTool(FileSearchTool(file_store_ids=[store.name]))]
    )

    result = await agent.run('Summarize the key points from my uploaded documents.')
    print(result.output)
    #> The documents discuss the following key points: ...

asyncio.run(main())
```

#### xAI

With xAI, `FileSearchTool` maps to the [collections search](https://docs.x.ai/developers/tools/collections-search) tool. Pass collection IDs as `file_store_ids`.

```py {title="file_search_xai.py" test="skip"}
import asyncio

from pydantic_ai import Agent, FileSearchTool
from pydantic_ai.capabilities import NativeTool


async def main():
    agent = Agent(
        'xai:grok-4.3',
        capabilities=[NativeTool(FileSearchTool(file_store_ids=['collection_abc123']))]
    )

    result = await agent.run('What does the collection say about pydantic?')
    print(result.output)
    #> Based on the collection, Pydantic is ...

asyncio.run(main())
```

## API Reference

For complete API documentation, see the [API Reference](api/native_tools.md).
