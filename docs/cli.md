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
clai --web --agent module:agent_variable
```

For example, if you have an agent defined in `my_agent.py`:

```python
from pydantic_ai import Agent

my_agent = Agent('openai:gpt-5', system_prompt='You are a helpful assistant.')
```

Launch the web UI with:

```bash
clai --web --agent my_agent:my_agent
```

This will start a web server (default: http://127.0.0.1:8000) with a chat interface for your agent.

#### Web Command Options

- `--host`: Host to bind the server to (default: 127.0.0.1)
- `--port`: Port to bind the server to (default: 8000)
- `--config`: Path to custom `agent_options.py` config file
- `--no-auto-config`: Disable auto-discovery of `agent_options.py` in current directory

#### Configuring Models and Tools

You can customize which AI models and builtin tools are available in the web UI by creating an `agent_options.py` file:

```python title="agent_options.py"
from pydantic_ai.builtin_tools import WebSearchTool, CodeExecutionTool
from pydantic_ai.ui.web import AIModel, BuiltinToolDef

models = [
    AIModel(
        id='openai:gpt-5',
        name='GPT 5',
        builtin_tools=['web_search', 'code_execution'],
    ),
    AIModel(
        id='anthropic:claude-sonnet-4-5',
        name='Claude Sonnet 4.5',
        builtin_tools=['web_search'],
    ),
]

builtin_tool_definitions = [
    BuiltinToolDef(
        id='web_search',
        name='Web Search',
        tool=WebSearchTool(),
    ),
    BuiltinToolDef(
        id='code_execution',
        name='Code Execution',
        tool=CodeExecutionTool(),
    ),
]
```

If an `agent_options.py` file exists in your current directory, it will be automatically loaded when you run `clai --web`. You can also specify a custom config path with `--config`.

You can also launch the web UI directly from an `Agent` instance using [`Agent.to_web()`][pydantic_ai.Agent.to_web]:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')
app = agent.to_web()  # Returns a FastAPI application
```

The returned FastAPI app can be run with your preferred ASGI server (uvicorn, hypercorn, etc.):

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
