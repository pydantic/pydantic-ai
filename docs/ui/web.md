# Web Chat UI

Pydantic AI includes a built-in web chat interface that you can use to interact with your agents through a browser.

<video src="https://github.com/user-attachments/assets/8a1c90dc-f62b-4e35-9d66-59459b45790d" autoplay loop muted playsinline></video>

## Installation

Install the `web` extra (installs Starlette):

```bash
pip/uv-add 'pydantic-ai-slim[web]'
```

For CLI usage with `clai web`, see the [CLI documentation](../cli.md#web-chat-ui).

## Usage

Create a web app from an agent instance using [`Agent.to_web()`][pydantic_ai.agent.Agent.to_web]:

=== "Using Model Names"

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.builtin_tools import WebSearchTool

    model = 'openai:gpt-5'
    agent = Agent(model)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'The weather in {city} is sunny'

    # Create app with model names (their display names are auto-generated)
    app = agent.to_web(
        models=['openai:gpt-5', 'anthropic:claude-sonnet-4-5'],
        builtin_tools=[WebSearchTool()],
    )

    # Or with custom display labels
    app = agent.to_web(
        models={'GPT 5': 'openai:gpt-5', 'Claude': 'anthropic:claude-sonnet-4-5'},
        builtin_tools=[WebSearchTool()],
    )
    ```

=== "Using Model Instances"

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.builtin_tools import WebSearchTool
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIModel

    # Create separate models with their own custom configuration
    anthropic_model = AnthropicModel('claude-sonnet-4-5')
    openai_model = OpenAIModel('gpt-5', api_key='custom-key')

    agent = Agent(openai_model)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'The weather in {city} is sunny'

    # Use the instances directly
    app = agent.to_web(
        models=[openai_model, anthropic_model],
        builtin_tools=[WebSearchTool()],
    )

    # Or mix instances and strings with custom labels
    app = agent.to_web(
        models={'Custom GPT': openai_model, 'Claude': 'anthropic:claude-sonnet-4-5'},
        builtin_tools=[WebSearchTool()],
    )

    # With extra instructions passed to each run
    app = agent.to_web(
        models=[openai_model],
        instructions='Always respond in a friendly tone.',
    )
    ```

The returned Starlette app can be run with any ASGI server:

```bash
uvicorn my_module:app --host 0.0.0.0 --port 8080
```

## Builtin Tool Support

Builtin tool support is automatically determined from each model's profile. The UI will only show tools that the selected model supports.

Available [builtin tools](../builtin-tools.md):

- `web_search` - Web search capability ([`WebSearchTool`][pydantic_ai.builtin_tools.WebSearchTool])
- `code_execution` - Code execution in a sandbox ([`CodeExecutionTool`][pydantic_ai.builtin_tools.CodeExecutionTool])
- `image_generation` - Image generation ([`ImageGenerationTool`][pydantic_ai.builtin_tools.ImageGenerationTool])
- `web_fetch` - Fetch content from URLs ([`WebFetchTool`][pydantic_ai.builtin_tools.WebFetchTool])
- `memory` - Persistent memory across conversations ([`MemoryTool`][pydantic_ai.builtin_tools.MemoryTool])

!!! note "Memory Tool Requirements"
    The `memory` tool requires the agent to have memory configured via the
    `memory` parameter when creating the agent.


## Reserved Routes

The web UI app uses the following routes which should not be overwritten:

- `/` and `/{id}` - Serves the chat UI
- `/api/chat` - Chat endpoint (POST, OPTIONS)
- `/api/configure` - Frontend configuration (GET)
- `/api/health` - Health check (GET)

The app cannot currently be mounted at a subpath (e.g., `/chat`) because the UI expects these routes at the root. You can add additional routes to the app, but avoid conflicts with these reserved paths.
