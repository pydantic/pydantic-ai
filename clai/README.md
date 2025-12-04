# clai

[![CI](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai)
[![PyPI](https://img.shields.io/pypi/v/clai.svg)](https://pypi.python.org/pypi/clai)
[![versions](https://img.shields.io/pypi/pyversions/clai.svg)](https://github.com/pydantic/pydantic-ai)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai.svg?v)](https://github.com/pydantic/pydantic-ai/blob/main/LICENSE)

(pronounced "clay")

Command line interface to chat to LLMs, part of the [Pydantic AI project](https://github.com/pydantic/pydantic-ai).

## Usage

<!-- Keep this in sync with docs/cli.md -->

You'll need to set an environment variable depending on the provider you intend to use.

E.g. if you're using OpenAI, set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Then with [`uvx`](https://docs.astral.sh/uv/guides/tools/), run:

```bash
uvx clai chat
```

Or to install `clai` globally [with `uv`](https://docs.astral.sh/uv/guides/tools/#installing-tools), run:

```bash
uv tool install clai
...
clai chat
```

Or with `pip`, run:

```bash
pip install clai
...
clai chat
```

Either way, running `clai chat` will start an interactive session where you can chat with the AI model. Special commands available in interactive mode:

- `/exit`: Exit the session
- `/markdown`: Show the last response in markdown format
- `/multiline`: Toggle multiline input mode (use Ctrl+D to submit)
- `/cp`: Copy the last response to clipboard

## Web Chat UI

Launch a web-based chat interface for your agent:

```bash
clai web --agent module:agent_variable
```

![Web Chat UI](https://github.com/user-attachments/assets/8a1c90dc-f62b-4e35-9d66-59459b45790d)

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

### Web Command Options

- `--agent`, `-a`: Agent to serve in `module:variable` format
- `--model`, `-m`: Model to make available (repeatable, agent's model is default if present)
- `--tool`, `-t`: [Builtin tool](https://ai.pydantic.dev/builtin-tools/) to enable (repeatable). See [available tools](https://ai.pydantic.dev/ui/web/#builtin-tool-support).
- `--mcp`: Path to MCP server config JSON file
- `--instructions`, `-i`: System instructions. In generic mode (no `--agent`), these are the agent instructions. With `--agent`, these are passed as extra instructions to each run.
- `--host`: Host to bind the server to (default: 127.0.0.1)
- `--port`: Port to bind the server to (default: 7932)

### Using with Models and Tools

You can specify which models and builtin tools are available in the UI via CLI flags:

```bash
# Generic agent with specific models and tools
clai web -m openai:gpt-5 -m anthropic:claude-sonnet-4-5 -t web_search -t code_execution

# Custom agent with additional models
clai web --agent my_agent:my_agent -m openai:gpt-5 -m google:gemini-2.5-pro

# Generic agent with system instructions
clai web -m openai:gpt-5 -i 'You are a helpful coding assistant'

# Custom agent with extra instructions for each run
clai web --agent my_agent:my_agent -i 'Always respond in Spanish'
```

When using `--agent`, the agent's configured model becomes the default. CLI models (`-m`) are additional options. Without `--agent`, the first `-m` model is the default.

For full documentation, see [Web Chat UI](https://ai.pydantic.dev/ui/web/).

## Help

```
usage: clai [-h] [-l] [--version] {chat,web} ...

Pydantic AI CLI v...

positional arguments:
  {chat,web}         Available commands
    chat             Interactive chat with an AI model
    web              Launch web chat UI for an agent

options:
  -h, --help         show this help message and exit
  -l, --list-models  List all available models and exit
  --version          Show version and exit
```
