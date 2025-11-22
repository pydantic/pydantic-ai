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

### Web Command Options

- `--host`: Host to bind the server to (default: 127.0.0.1)
- `--port`: Port to bind the server to (default: 8000)
- `--config`: Path to custom `agent_options.py` config file
- `--no-auto-config`: Disable auto-discovery of `agent_options.py` in current directory

### Configuring Models and Tools

You can customize which AI models and builtin tools are available in the web UI by creating an `agent_options.py` file. For example:

```python
from pydantic_ai.ui.web import AIModel, BuiltinToolDef
from pydantic_ai.builtin_tools import WebSearchTool

models = [
    AIModel(id='openai:gpt-5', name='GPT 5', builtin_tools=['web_search']),
]

builtin_tool_definitions = [
    BuiltinToolDef(id='web_search', name='Web Search', tool=WebSearchTool()),
]
```

See the [default configuration](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/ui/web/agent_options.py) for more examples.

If an `agent_options.py` file exists in your current directory, it will be automatically loaded when you run `clai --web`. You can also specify a custom config path with `--config`.

You can also launch the web UI directly from an `Agent` instance using `Agent.to_web()`:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')
app = agent.to_web()  # Returns a FastAPI application
```

## Help

```
usage: clai [-h] [-m [MODEL]] [-a AGENT] [-l] [-t [CODE_THEME]] [--no-stream] [--version] [--web] [--host HOST] [--port PORT] [--config CONFIG]
            [--no-auto-config]
            [prompt]

Pydantic AI CLI v...

Special prompts:
* `/exit` - exit the interactive mode (ctrl-c and ctrl-d also work)
* `/markdown` - show the last markdown output of the last question
* `/multiline` - toggle multiline mode
* `/cp` - copy the last response to clipboard

positional arguments:
  prompt                AI Prompt, if omitted fall into interactive mode

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
  --web                 Launch web chat UI for the agent (requires --agent)
  --host HOST           Host to bind the server to (default: 127.0.0.1)
  --port PORT           Port to bind the server to (default: 8000)
  --config CONFIG       Path to agent_options.py config file (overrides auto-discovery)
  --no-auto-config      Disable auto-discovery of agent_options.py in current directory
```
