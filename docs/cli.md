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

agent = Agent('openai:gpt-4.1', instructions='You always respond in Italian.')
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

agent = Agent('openai:gpt-4.1', instructions='You always respond in Italian.')
agent.to_cli_sync()
```

You can also use the async interface with `Agent.to_cli()`:

```python {title="agent_to_cli.py" test="skip" hl_lines=6}
from pydantic_ai import Agent

agent = Agent('openai:gpt-4.1', instructions='You always respond in Italian.')

async def main():
    await agent.to_cli()
```

_(You'll need to add `asyncio.run(main())` to run `main`)_
