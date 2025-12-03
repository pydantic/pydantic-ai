# Web Chat UI

Pydantic AI includes a built-in web chat interface that you can use to interact with your agents through a browser.

https://github.com/user-attachments/assets/8a1c90dc-f62b-4e35-9d66-59459b45790d

## Installation

Install the `web` extra to get the required dependencies:

```bash
pip/uv-add 'pydantic-ai-slim[web]'
```

## Usage

There are two ways to launch the web chat UI:

### 1. Using the CLI (`clai web`)

The simplest way to start the web UI is using the `clai web` command:

```bash
# With a custom agent
clai web --agent my_module:my_agent

# With specific models
clai web -m openai:gpt-5 -m anthropic:claude-sonnet-4-5

# With builtin tools
clai web -m openai:gpt-5 -t web_search -t code_execution

# Generic agent with system instructions
clai web -m openai:gpt-5 -i 'You are a helpful coding assistant'
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--agent`, `-a` | Agent to serve in `module:variable` format |
| `--model`, `-m` | Model to make available (repeatable) |
| `--tool`, `-t` | Builtin tool to enable (repeatable) |
| `--instructions`, `-i` | System instructions (when `--agent` not specified) |
| `--host` | Host to bind server (default: 127.0.0.1) |
| `--port` | Port to bind server (default: 7932) |
| `--mcp` | Path to MCP server config JSON file |

Model names without a provider prefix are automatically inferred:

- `gpt-*`, `o1`, `o3` → OpenAI
- `claude-*`, `sonnet`, `opus`, `haiku` → Anthropic
- `gemini-*` → Google

### 2. Using `Agent.to_web()` Programmatically

For more control, you can create a web app from an agent instance:

```python
from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool

agent = Agent('openai:gpt-5')

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

The returned Starlette app can be run with any ASGI server:

```bash
uvicorn my_module:app --host 0.0.0.0 --port 8080
```

!!! note "Reserved Routes"
    The web UI app uses the following routes which should not be overwritten:

    - `/` and `/{id}` - Serves the chat UI
    - `/api/chat` - Chat endpoint (POST, OPTIONS)
    - `/api/configure` - Frontend configuration (GET)
    - `/api/health` - Health check (GET)

    The app cannot currently be mounted at a subpath (e.g., `/chat`) because the UI expects these routes at the root. You can add additional routes to the app, but avoid conflicts with these reserved paths.

## MCP Server Configuration

You can enable [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) servers using a JSON configuration file:

```bash
clai web --agent my_agent:my_agent --mcp mcp-servers.json
```

Example JSON configuration:

```json
{
  "mcpServers": {
    "deepwiki": {
      "url": "https://mcp.deepwiki.com/mcp"
    },
    "github": {
      "url": "https://api.githubcopilot.com/mcp",
      "authorizationToken": "${GITHUB_TOKEN}"
    }
  }
}
```

Environment variables can be referenced using `${VAR_NAME}` syntax, with optional defaults: `${VAR_NAME:-default_value}`.

Each server entry supports:

| Field | Description |
|-------|-------------|
| `url` (required) | The MCP server URL |
| `authorizationToken` | Authorization token for the server |
| `description` | Description shown in the UI |
| `allowedTools` | List of allowed tool names |
| `headers` | Additional HTTP headers |

## Builtin Tool Support

When using the new models API, builtin tool support is automatically determined from each model's profile. The UI will only show tools that the selected model supports.

Available builtin tools:

- `web_search` - Web search capability
- `code_execution` - Code execution in a sandbox
- `image_generation` - Image generation
- `web_fetch` - Fetch content from URLs
- `memory` - Persistent memory across conversations

!!! note "Memory Tool Requirements"
    The `memory` tool requires the agent to have memory configured via the
    `memory` parameter when creating the agent. It cannot be enabled via
    the CLI `-t memory` flag alone - an agent with memory must be provided
    via `--agent`.
