# Command Line Interface (CLI)

**Pydantic AI** comes with a CLI, `clai` (pronounced "clay") which you can use to interact with various LLMs from the command line.
It provides a convenient way to chat with language models and quickly get answers right in the terminal.

We originally developed this CLI for our own use, but found ourselves using it so frequently that we decided to share it as part of the Pydantic AI package.

We plan to continue adding new features, such as interaction with MCP servers, access to tools, and more.

## Usage

<!-- Keep this in sync with clai/README.md -->

You'll need to set an environment variable depending on the provider you intend to use.

E.g. if you're using OpenAI, set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Then with [`uvx`](https://docs.astral.sh/uv/guides/tools/), run:

```bash
uvx clai
```

Or to install `clai` globally [with `uv`](https://docs.astral.sh/uv/guides/tools/#installing-tools), run:

```bash
uv tool install clai
...
clai
```

Or with `pip`, run:

```bash
pip install clai
...
clai
```

Either way, running `clai` will start an interactive session where you can chat with the AI model. Special commands available in interactive mode:

- `/exit`: Exit the session
- `/markdown`: Show the last response in markdown format
- `/multiline`: Toggle multiline input mode (use Ctrl+D to submit)
- `/cp`: Copy the last response to clipboard

### Web Chat UI

Launch a web-based chat interface for your agent:

```bash
clai web --agent module:agent_variable
```

For example, if you have an agent defined in `my_agent.py`:

```python
from pydantic_ai import Agent

my_agent = Agent('openai:gpt-5', system_prompt='You are a helpful assistant.')
```

Launch the web UI with:

```bash
clai web --agent my_agent:my_agent
```

This will start a web server (default: http://127.0.0.1:7932) with a chat interface for your agent.

#### Web Command Options

- `--agent`, `-a`: Agent to serve in `module:variable` format
- `--models`, `-m`: Comma-separated models to make available (e.g., `gpt-5,sonnet-4-5`)
- `--tools`, `-t`: Comma-separated builtin tool IDs to enable (e.g., `web_search,code_execution`)
- `--mcp-json`: Path to JSON file with MCP server configurations
- `--instructions`, `-i`: System instructions for generic agent (when `--agent` not specified)
- `--host`: Host to bind the server to (default: 127.0.0.1)
- `--port`: Port to bind the server to (default: 7932)

#### Using with Models and Tools

You can specify which models and builtin tools are available in the UI via CLI flags:

```bash
# Generic agent with specific models and tools
clai web -m gpt-5,sonnet-4-5 -t web_search,code_execution

# Custom agent with additional models
clai web --agent my_agent:my_agent -m gpt-5,gemini-2.5-pro

# Generic agent with system instructions
clai web -m gpt-5 -i 'You are a helpful coding assistant'
```

Model names without a provider prefix are automatically inferred:
- `gpt-*`, `o1`, `o3` → OpenAI
- `claude-*`, `sonnet`, `opus`, `haiku` → Anthropic
- `gemini-*` → Google

#### MCP Server Configuration

You can enable MCP (Model Context Protocol) servers using a JSON configuration file:

```bash
clai web --agent my_agent:my_agent --mcp-json mcp-servers.json
```

The JSON file format:

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
- `url` (required): The MCP server URL
- `authorizationToken` (optional): Authorization token for the server
- `description` (optional): Description for the server
- `allowedTools` (optional): List of allowed tool names
- `headers` (optional): Additional HTTP headers

#### Programmatic Web UI

You can also launch the web UI directly from an `Agent` instance using [`Agent.to_web()`][pydantic_ai.Agent.to_web]:

```python
from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.ui.web import AIModel

agent = Agent('openai:gpt-5')

# Use defaults
app = agent.to_web()

# Or customize models and tools
app = agent.to_web(
    models=[
        AIModel(id='openai:gpt-5', name='GPT 5', builtin_tools=['web_search']),
    ],
    builtin_tools=[WebSearchTool()],
)
```

The returned Starlette app can be run with your preferred ASGI server (uvicorn, hypercorn, etc.):

```bash
# If you saved the code above in my_agent.py and created an app variable:
# app = agent.to_web()
uvicorn my_agent:app --host 0.0.0.0 --port 8080
```

### Help

To get help on the CLI, use the `--help` flag:

```bash
uvx clai --help
```

### Choose a model

You can specify which model to use with the `--model` flag:

```bash
uvx clai --model anthropic:claude-sonnet-4-0
```

(a full list of models available can be printed with `uvx clai --list-models`)

### Custom Agents

You can specify a custom agent using the `--agent` flag with a module path and variable name:

```python {title="custom_agent.py" test="skip"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', system_prompt='You always respond in Italian.')
```

Then run:

```bash
uvx clai --agent custom_agent:agent "What's the weather today?"
```

The format must be `module:variable` where:

- `module` is the importable Python module path
- `variable` is the name of the Agent instance in that module

Additionally, you can directly launch CLI mode from an `Agent` instance using `Agent.to_cli_sync()`:

```python {title="agent_to_cli_sync.py" test="skip" hl_lines=4}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', system_prompt='You always respond in Italian.')
agent.to_cli_sync()
```

You can also use the async interface with `Agent.to_cli()`:

```python {title="agent_to_cli.py" test="skip" hl_lines=6}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', system_prompt='You always respond in Italian.')

async def main():
    await agent.to_cli()
```

_(You'll need to add `asyncio.run(main())` to run `main`)_

### Message History

Both `Agent.to_cli()` and `Agent.to_cli_sync()` support a `message_history` parameter, allowing you to continue an existing conversation or provide conversation context:

```python {title="agent_with_history.py" test="skip"}
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

agent = Agent('openai:gpt-5')

# Create some conversation history
message_history: list[ModelMessage] = [
    ModelRequest([UserPromptPart(content='What is 2+2?')]),
    ModelResponse([TextPart(content='2+2 equals 4.')])
]

# Start CLI with existing conversation context
agent.to_cli_sync(message_history=message_history)
```

The CLI will start with the provided conversation history, allowing the agent to refer back to previous exchanges and maintain context throughout the session.
