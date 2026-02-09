# Command Line Interface (CLI)

**Pydantic AI** comes with a CLI, `clai` (pronounced "clay"). You can use it to chat with various LLMs and quickly get answers, right from the command line, or spin up a uvicorn server to chat with your Pydantic AI agents from your browser.

## Installation

You can run the `clai` using [`uvx`](https://docs.astral.sh/uv/guides/tools/):

```bash
uvx clai
```

Or install `clai` globally [with `uv`](https://docs.astral.sh/uv/guides/tools/#installing-tools):

```bash
uv tool install clai
...
clai
```

Or with `pip`:

```bash
pip install clai
...
clai
```

## CLI Usage

<!-- clai/README.md links here for full docs -->

You'll need to set an environment variable depending on the provider you intend to use.

E.g. if you're using OpenAI, set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Then running `clai` will start an interactive session where you can chat with the AI model. Special commands available in interactive mode:

- `/exit`: Exit the session
- `/markdown`: Show the last response in markdown format
- `/multiline`: Toggle multiline input mode (use Ctrl+D to submit)
- `/cp`: Copy the last response to clipboard

### CLI Options

| Option | Description |
|--------|-------------|
| `prompt` | AI prompt for one-shot mode (positional). If omitted, starts interactive mode. |
| `-m`, `--model` | Model to use in `provider:model` format (e.g., `openai:gpt-5.2`) |
| `-a`, `--agent` | Custom agent in `module:variable` format |
| `-t`, `--code-theme` | Syntax highlighting theme (`dark`, `light`, or [pygments theme](https://pygments.org/styles/)) |
| `--no-stream` | Disable streaming from the model |
| `-l`, `--list-models` | List all available models and exit |
| `--version` | Show version and exit |

### Choose a model

You can specify which model to use with the `--model` flag:

```bash
clai --model anthropic:claude-sonnet-4-5
```

(a full list of models available can be printed with `clai --list-models`)

### Custom Agents

You can specify a custom agent using the `--agent` flag with a module path and variable name:

```python {title="custom_agent.py" test="skip"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', instructions='You always respond in Italian.')
```

Then run:

```bash
clai --agent custom_agent:agent "What's the weather today?"
```

The format must be `module:variable` where:

- `module` is the importable Python module path
- `variable` is the name of the Agent instance in that module

Additionally, you can directly launch CLI mode from an `Agent` instance using `Agent.to_cli_sync()`:

```python {title="agent_to_cli_sync.py" test="skip" hl_lines=4}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', instructions='You always respond in Italian.')
agent.to_cli_sync()
```

You can also use the async interface with `Agent.to_cli()`:

```python {title="agent_to_cli.py" test="skip" hl_lines=6}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', instructions='You always respond in Italian.')

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

agent = Agent('openai:gpt-5.2')

# Create some conversation history
message_history: list[ModelMessage] = [
    ModelRequest([UserPromptPart(content='What is 2+2?')]),
    ModelResponse([TextPart(content='2+2 equals 4.')])
]

# Start CLI with existing conversation context
agent.to_cli_sync(message_history=message_history)
```

The CLI will start with the provided conversation history, allowing the agent to refer back to previous exchanges and maintain context throughout the session.

## Web Chat UI

Launch a web-based chat interface by running:

```bash
clai web -m openai:gpt-5.2
```

This will start a web server (default: http://127.0.0.1:7932) with a chat interface.

You can also serve an existing agent. For example, if you have an agent defined in `my_agent.py`:

```python
from pydantic_ai import Agent

my_agent = Agent('openai:gpt-5.2', instructions='You are a helpful assistant.')
```

Launch the web UI:

```bash
# With a custom agent
clai web --agent my_module:my_agent

# With specific models (first is default when no --agent)
clai web -m openai:gpt-5.2 -m anthropic:claude-sonnet-4-5

# With builtin tools
clai web -m openai:gpt-5.2 -t web_search -t code_execution

# Generic agent with system instructions
clai web -m openai:gpt-5.2 -i 'You are a helpful coding assistant'

# Custom agent with extra instructions for each run
clai web --agent my_module:my_agent -i 'Always respond in Spanish'
```

!!! note "Memory Tool"
    The [`memory`](builtin-tools.md#memory-tool) builtin tool cannot be enabled via `-t memory`. If your agent needs memory, configure the [`MemoryTool`][pydantic_ai.builtin_tools.MemoryTool] directly on the agent and provide it via `--agent`.

### Web UI Options

| Option | Description |
|--------|-------------|
| `--agent`, `-a` | Agent to serve in [`module:variable` format](#custom-agents) |
| `--model`, `-m` | Models to list as options in the UI (repeatable) |
| `--tool`, `-t` | [Builtin tool](builtin-tools.md)s to list as options in the UI (repeatable). See [available tools](web.md#builtin-tool-support). |
| `--instructions`, `-i` | System instructions. When `--agent` is specified, these are additional to the agent's existing instructions. |
| `--host` | Host to bind server (default: 127.0.0.1) |
| `--port` | Port to bind server (default: 7932) |
| `--html-source` | URL or file path for the chat UI HTML. |

When using `--agent`, the agent's configured model becomes the default. CLI models (`-m`) are additional options. Without `--agent`, the first `-m` model is the default.

The web chat UI can also be launched programmatically using [`Agent.to_web()`][pydantic_ai.agent.Agent.to_web], see the [Web UI documentation](web.md).

Run the `web` command with `--help` to see all available options:

```bash
clai web --help
```
