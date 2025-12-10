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

Launch a web-based chat interface:

```bash
clai web -m openai:gpt-5
```

This will start a web server (default: http://127.0.0.1:7932) with a chat interface.

You can also serve an existing agent. For example, if you have an agent defined in `my_agent.py`:

```python
from pydantic_ai import Agent

my_agent = Agent('openai:gpt-5', instructions='You are a helpful assistant.')
```

Launch the web UI with:

```bash
clai web --agent my_agent:my_agent
```

#### CLI Options

```bash
# With a custom agent
clai web --agent my_module:my_agent

# With specific models (first is default when no --agent)
clai web -m openai:gpt-5 -m anthropic:claude-sonnet-4-5

# With builtin tools
clai web -m openai:gpt-5 -t web_search -t code_execution

# Generic agent with system instructions
clai web -m openai:gpt-5 -i 'You are a helpful coding assistant'

# Custom agent with extra instructions for each run
clai web --agent my_module:my_agent -i 'Always respond in Spanish'
```

| Option | Description |
|--------|-------------|
| `--agent`, `-a` | Agent to serve in [`module:variable` format](#custom-agents) |
| `--model`, `-m` | Models to list as options in the UI (repeatable, agent's model is default if present) |
| `--tool`, `-t` | [Builtin tool](builtin-tools.md)s to list as options in the UI (repeatable). See [available tools](web.md#builtin-tool-support). |
| `--instructions`, `-i` | System instructions. In generic mode (no `--agent`), these are the agent instructions. With `--agent`, these are passed as extra instructions to each run. |
| `--host` | Host to bind server (default: 127.0.0.1) |
| `--port` | Port to bind server (default: 7932) |

!!! note "Memory Tool"
    The `memory` tool requires the agent to have memory configured and cannot be enabled via `-t memory` alone. An agent with memory must be provided via `--agent`.

The web chat UI can also be launched programmatically using [`Agent.to_web()`][pydantic_ai.Agent.to_web], see the [Web UI documentation](web.md).

### Help

To get help on the CLI, use the `--help` flag:

```bash
uvx clai --help
uvx clai chat --help
uvx clai web --help
```

### Choose a model

You can specify which model to use with the `--model` flag:

```bash
uvx clai chat --model anthropic:claude-sonnet-4-0
```

(a full list of models available can be printed with `uvx clai --list-models`)

### Custom Agents

You can specify a custom agent using the `--agent` flag with a module path and variable name:

```python {title="custom_agent.py" test="skip"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', instructions='You always respond in Italian.')
```

Then run:

```bash
uvx clai chat --agent custom_agent:agent "What's the weather today?"
```

The format must be `module:variable` where:

- `module` is the importable Python module path
- `variable` is the name of the Agent instance in that module

Additionally, you can directly launch CLI mode from an `Agent` instance using `Agent.to_cli_sync()`:

```python {title="agent_to_cli_sync.py" test="skip" hl_lines=4}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', instructions='You always respond in Italian.')
agent.to_cli_sync()
```

You can also use the async interface with `Agent.to_cli()`:

```python {title="agent_to_cli.py" test="skip" hl_lines=6}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', instructions='You always respond in Italian.')

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
