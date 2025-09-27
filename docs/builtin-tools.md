# Builtin Tools

Builtin tools are native tools provided by LLM providers that can be used to enhance your agent's capabilities. Unlike [common tools](common-tools.md), which are custom implementations that Pydantic AI executes, builtin tools are executed directly by the model provider.

## Overview

Pydantic AI supports the following builtin tools:

- **[`WebSearchTool`][pydantic_ai.builtin_tools.WebSearchTool]**: Allows agents to search the web
- **[`CodeExecutionTool`][pydantic_ai.builtin_tools.CodeExecutionTool]**: Enables agents to execute code in a secure environment
- **[`UrlContextTool`][pydantic_ai.builtin_tools.UrlContextTool]**: Enables agents to pull URL contents into their context

These tools are passed to the agent via the `builtin_tools` parameter and are executed by the model provider's infrastructure.

!!! warning "Provider Support"
    Not all model providers support builtin tools. If you use a builtin tool with an unsupported provider, Pydantic AI will raise a [`UserError`][pydantic_ai.exceptions.UserError] when you try to run the agent.

    If a provider supports a built-in tool that is not currently supported by Pydantic AI, please file an issue.

## Web Search Tool

The [`WebSearchTool`][pydantic_ai.builtin_tools.WebSearchTool] allows your agent to search the web,
making it ideal for queries that require up-to-date data.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| OpenAI Responses | ✅ | Full feature support. To include search results on the [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart], set the `openai_include_web_search_sources` setting to `True` on [`OpenAIResponsesModelSettings`][pydantic_ai.models.openai.OpenAIResponsesModelSettings]. |
| Anthropic | ✅ | Full feature support |
| Google | ✅ | No parameter support. No [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] or [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] is generated when streaming. Using built-in tools and user tools (including [output tools](output.md#tool-output)) at the same time is not supported; to use structured output, use [`PromptedOutput`](output.md#prompted-output) instead. |
| Groq | ✅ | Limited parameter support. To use web search capabilities with Groq, you need to use the [compound models](https://console.groq.com/docs/compound). |
| OpenAI Chat Completions | ❌ | Not supported |
| Bedrock | ❌ | Not supported |
| Mistral | ❌ | Not supported |
| Cohere | ❌ | Not supported |
| HuggingFace | ❌ | Not supported |

### Usage

```py title="web_search_anthropic.py"
from pydantic_ai import Agent, WebSearchTool

agent = Agent('anthropic:claude-sonnet-4-0', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.
```

With OpenAI, you must use their responses API to access the web search tool.

```py title="web_search_openai.py"
from pydantic_ai import Agent, WebSearchTool

agent = Agent('openai-responses:gpt-4.1', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.
```

### Configuration Options

The `WebSearchTool` supports several configuration parameters:

```py title="web_search_configured.py"
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation

agent = Agent(
    'anthropic:claude-sonnet-4-0',
    builtin_tools=[
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
    ],
)

result = agent.run_sync('Use the web to get the current time.')
# > In San Francisco, it's 8:21:41 pm PDT on Wednesday, August 6, 2025.
```

### Parameter Support by Provider

| Parameter | OpenAI | Anthropic | Groq |
|-----------|--------|-----------|------|
| `search_context_size` | ✅ | ❌ | ❌ |
| `user_location` | ✅ | ✅ | ❌ |
| `blocked_domains` | ❌ | ✅ | ✅ |
| `allowed_domains` | ❌ | ✅ | ✅ |
| `max_uses` | ❌ | ✅ | ❌ |

!!! note "Anthropic Domain Filtering"
    With Anthropic, you can only use either `blocked_domains` or `allowed_domains`, not both.

## Code Execution Tool

The [`CodeExecutionTool`][pydantic_ai.builtin_tools.CodeExecutionTool] enables your agent to execute code
in a secure environment, making it perfect for computational tasks, data analysis, and mathematical operations.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| OpenAI | ✅ | To include outputs on the [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart], set the `openai_include_code_execution_outputs` setting to `True` on [`OpenAIResponsesModelSettings`][pydantic_ai.models.openai.OpenAIResponsesModelSettings]. |
| Anthropic | ✅ | |
| Google | ✅ | Using built-in tools and user tools (including [output tools](output.md#tool-output)) at the same time is not supported; to use structured output, use [`PromptedOutput`](output.md#prompted-output) instead. |
| Groq | ❌ | |
| Bedrock | ❌ | |
| Mistral | ❌ | |
| Cohere | ❌ | |
| HuggingFace | ❌ | |

### Usage

```py title="code_execution_basic.py"
from pydantic_ai import Agent, CodeExecutionTool

agent = Agent('anthropic:claude-sonnet-4-0', builtin_tools=[CodeExecutionTool()])

result = agent.run_sync('Calculate the factorial of 15 and show your work')
# > The factorial of 15 is **1,307,674,368,000**.
```

## URL Context Tool

The [`UrlContextTool`][pydantic_ai.builtin_tools.UrlContextTool] enables your agent to pull URL contents into its context,
allowing it to pull up-to-date information from the web.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| Google | ✅ | No [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] or [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] is currently generated; please submit an issue if you need this. Using built-in tools and user tools (including [output tools](output.md#tool-output)) at the same time is not supported; to use structured output, use [`PromptedOutput`](output.md#prompted-output) instead. |
| OpenAI | ❌ | |
| Anthropic | ❌ | |
| Groq | ❌ | |
| Bedrock | ❌ | |
| Mistral | ❌ | |
| Cohere | ❌ | |
| HuggingFace | ❌ | |

### Usage

```py title="url_context_basic.py"
from pydantic_ai import Agent, UrlContextTool

agent = Agent('google-gla:gemini-2.5-flash', builtin_tools=[UrlContextTool()])

result = agent.run_sync('What is this? https://ai.pydantic.dev')
# > A Python agent framework for building Generative AI applications.
```

## API Reference

For complete API documentation, see the [API Reference](api/builtin_tools.md).
