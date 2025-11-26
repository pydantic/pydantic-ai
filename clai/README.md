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

## Web Chat UI

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

### Web Command Options

- `--agent`, `-a`: Agent to serve in `module:variable` format
- `--models`, `-m`: Comma-separated models to make available (e.g., `gpt-5,sonnet-4-5`)
- `--tools`, `-t`: Comma-separated builtin tool IDs to enable (e.g., `web_search,code_execution`)
- `--instructions`, `-i`: System instructions for generic agent (when `--agent` not specified)
- `--host`: Host to bind the server to (default: 127.0.0.1)
- `--port`: Port to bind the server to (default: 7932)

### Using with Models and Tools

You can specify which models and builtin tools are available in the UI via CLI flags:

```bash
# Generic agent with specific models and tools
clai web -m gpt-5,sonnet-4-5 -t web_search,code_execution

# Custom agent with additional models
clai web --agent my_agent:my_agent -m gpt-5,gemini-2.5-pro

# Generic agent with system instructions
clai web -m gpt-5 -i 'You are a helpful coding assistant'
```

You can also launch the web UI directly from an `Agent` instance using `Agent.to_web()`:

```python
from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool

agent = Agent('openai:gpt-5')

# Use defaults
app = agent.to_web()

# Or customize models and tools
app = agent.to_web(builtin_tools=[WebSearchTool()])
```

## Help

```
usage: clai [-h] [-m [MODEL]] [-a AGENT] [-l] [-t [CODE_THEME]] [--no-stream] [--version] [prompt] {web} ...

Pydantic AI CLI v...

Special prompts:
* `/exit` - exit the interactive mode (ctrl-c and ctrl-d also work)
* `/markdown` - show the last markdown output of the last question
* `/multiline` - toggle multiline mode
* `/cp` - copy the last response to clipboard

positional arguments:
  prompt                AI Prompt, if omitted fall into interactive mode
  {web}                 Available commands
    web                 Launch web chat UI for an agent

options:
  -h, --help            show this help message and exit
  -m [MODEL], --model [MODEL]
                        Model to use, in format "<provider>:<model>" e.g. "openai:gpt-5" or "anthropic:claude-sonnet-4-5". Defaults to "openai:gpt-5".
  -a AGENT, --agent AGENT
                        Custom Agent to use, in format "module:variable", e.g. "mymodule.submodule:my_agent"
  -l, --list-models     List all available models and exit
  -t [CODE_THEME], --code-theme [CODE_THEME]
                        Which colors to use for code, can be "dark", "light" or any theme from pygments.org/styles/. Defaults to "dark" which works well on dark terminals.
  --no-stream           Disable streaming from the model
  --version             Show version and exit
```
