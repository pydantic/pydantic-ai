# Web Chat UI

Pydantic AI includes a built-in web chat interface that you can use to interact with your agents through a browser.

![Web Chat UI](https://github.com/user-attachments/assets/8a1c90dc-f62b-4e35-9d66-59459b45790d)

## Installation

Install the `web` extra (installs Starlette):

```bash
pip/uv-add 'pydantic-ai-slim[web]'
```

For CLI usage with `clai web`, see the [CLI documentation](cli.md#web-chat-ui).

## Basic Usage

Create a web app from an agent instance using [`Agent.to_web()`][pydantic_ai.Agent.to_web]:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', instructions='You are a helpful assistant.')

@agent.tool_plain
def get_weather(city: str) -> str:
    return f'The weather in {city} is sunny'

app = agent.to_web()
```

Run the app with any ASGI server:

```bash
uvicorn my_module:app --host 127.0.0.1 --port 7932
```

## Configuring Models

You can specify additional models to make available in the UI. Models can be provided as a list of model names or a dictionary mapping display labels to model names/instances.

### Using Model Names

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')

# List of model names (display names are auto-generated)
app = agent.to_web(
    models=['openai:gpt-5', 'anthropic:claude-sonnet-4-5'],
)

# Or with custom display labels
app = agent.to_web(
    models={'GPT 5': 'openai:gpt-5', 'Claude': 'anthropic:claude-sonnet-4-5'},
)
```

### Using Model Instances

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

# Create models with custom configuration
anthropic_model = AnthropicModel('claude-sonnet-4-5')
openai_model = OpenAIChatModel('gpt-5', provider='openai')

agent = Agent(openai_model)

# Use instances directly
app = agent.to_web(
    models=[openai_model, anthropic_model],
)

# Or mix instances and strings with custom labels
app = agent.to_web(
    models={'Custom GPT': openai_model, 'Claude': 'anthropic:claude-sonnet-4-5'},
)
```

## Builtin Tool Support

You can specify a list of [builtin tools](builtin-tools.md) that will be shown as options to the user, if the selected model supports them:

```python
from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool, CodeExecutionTool

agent = Agent('openai:gpt-5')

app = agent.to_web(
    models=['openai:gpt-5', 'anthropic:claude-sonnet-4-5'],
    builtin_tools=[WebSearchTool(), CodeExecutionTool()],
)
```

The UI will only show tools that the selected model supports.

!!! note "Memory Tool"
    The `memory` builtin tool is not supported via `to_web()` or `clai web`. If your agent needs memory, configure the [`MemoryTool`][pydantic_ai.builtin_tools.MemoryTool] directly on the agent at construction time.

## Extra Instructions

You can pass extra instructions that will be included in each agent run:

```python
app = agent.to_web(
    models=['openai:gpt-5'],
    instructions='Always respond in a friendly tone.',
)
```

## Reserved Routes

The web UI app uses the following routes which should not be overwritten:

- `/` and `/{id}` - Serves the chat UI
- `/api/chat` - Chat endpoint (POST, OPTIONS)
- `/api/configure` - Frontend configuration (GET)
- `/api/health` - Health check (GET)

The app cannot currently be mounted at a subpath (e.g., `/chat`) because the UI expects these routes at the root. You can add additional routes to the app, but avoid conflicts with these reserved paths.
