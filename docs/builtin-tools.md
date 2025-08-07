# Builtin Tools

Builtin tools are native tools provided by LLM providers that can be used to enhance your agent's capabilities. Unlike [common tools](common-tools.md), which are custom implementations that PydanticAI executes, builtin tools are executed directly by the model provider.

## Overview

PydanticAI supports two types of builtin tools:

- **[`WebSearchTool`][pydantic_ai.builtin_tools.WebSearchTool]**: Allows agents to search the web
- **[`CodeExecutionTool`][pydantic_ai.builtin_tools.CodeExecutionTool]**: Enables agents to execute code in a secure environment

These tools are passed to the agent via the `builtin_tools` parameter and are executed by the model provider's infrastructure.

!!! warning "Provider Support"
    Not all model providers support builtin tools. If you use a builtin tool with an unsupported provider, PydanticAI will raise a [`UserError`][pydantic_ai.exceptions.UserError] when you try to run the agent.

## Web Search Tool

The [`WebSearchTool`][pydantic_ai.builtin_tools.WebSearchTool] allows your agent to search the web,
making it ideal for queries that require up-to-date data.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| OpenAI | ✅ | Full feature support |
| Anthropic | ✅ | Full feature support |
| Groq | ✅ | Limited parameter support |
| Google | ❌ | Not supported |
| Bedrock | ❌ | Not supported |
| Mistral | ❌ | Not supported |
| Cohere | ❌ | Not supported |
| HuggingFace | ❌ | Not supported |

!!! note "Groq Support"
    To use web search capabilities with Groq, you need to use the [compound models](https://console.groq.com/docs/compound).

### Basic Usage

```py title="web_search_basic.py"
from pydantic_ai import Agent, WebSearchTool

agent = Agent(
    'anthropic:claude-4-0-sonnet',
    builtin_tools=[WebSearchTool()],
    system_prompt='You are a helpful assistant that can search the web.',
)

result = agent.run_sync('What are the latest developments in AI this week?')
...
```

### Configuration Options

The `WebSearchTool` supports several configuration parameters:

```py title="web_search_configured.py"
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation

# Configure web search with location and domain filtering
web_search = WebSearchTool(
    search_context_size='high',  # OpenAI only: 'low', 'medium', 'high'
    user_location=WebSearchUserLocation(
        city='San Francisco',
        country='US',  # 2-letter code for OpenAI
        region='CA',
        timezone='America/Los_Angeles'
    ),
    blocked_domains=['example.com', 'spam-site.net'],
    allowed_domains=None,  # Cannot use both blocked_domains and allowed_domains with Anthropic
    max_uses=5  # Anthropic only: limit tool usage
)

agent = Agent(
    'anthropic:claude-4-0-sonnet',
    builtin_tools=[web_search],
    system_prompt='Search for local information and provide current results.',
)

result = agent.run_sync('What is the weather like today?')
...
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

| Provider | Supported |
|----------|-----------|
| OpenAI | ✅ |
| Anthropic | ✅ |
| Google | ✅ |
| Groq | ❌ |
| Bedrock | ❌ |
| Mistral | ❌ |
| Cohere | ❌ |
| HuggingFace | ❌ |

### Basic Usage

```py title="code_execution_basic.py"
from pydantic_ai import Agent, CodeExecutionTool

agent = Agent(
    'openai:gpt-4o',
    builtin_tools=[CodeExecutionTool()],
    system_prompt='You can execute Python code to solve problems and perform calculations.',
)

result = agent.run_sync('Calculate the factorial of 15 and show your work')
...
```

### Advanced Examples

```py title="code_execution_advanced.py"
from pydantic_ai import Agent, CodeExecutionTool

# Agent that can perform data analysis
agent = Agent(
    'anthropic:claude-4-0-sonnet',
    builtin_tools=[CodeExecutionTool()],
    system_prompt='''You are a data analyst that can execute Python code.
    When given data or mathematical problems, write and execute code to provide accurate solutions.
    Always show your work and explain your approach.''',
)

# Complex mathematical calculation
result = agent.run_sync('''
    Calculate the compound interest on $10,000 invested at 5% annual interest
    for 10 years, compounded monthly. Show the formula and intermediate steps.
    '''
)
...

# Data analysis task
result = agent.run_sync('''
    Create a dataset of 100 random numbers from a normal distribution (mean=50, std=10).
    Calculate the mean, median, standard deviation, and create a histogram.
    '''
)
...
```

## API Reference

For complete API documentation, see the [API Reference](api/builtin_tools.md).
