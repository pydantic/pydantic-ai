# Command Line Interface (CLI)

**PydanticAI** comes with a CLI which you can use to interact with various LLMs from the command line.
It provides a convenient way to chat with language models and quickly get answers right in the terminal.

We originally developed this CLI for our own use, but found ourselves using it so frequently that we decided to share it as part of the PydanticAI package.

We plan to continue adding new features, such as interaction with MCP servers, access to tools, and more.

## Usage

You'll need to set an environment variable depending on the provider you intend to use.

E.g. if using OpenAI, set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Then with `uvx` (part of [uv](https://docs.astral.sh/uv/)) installed, simply run:

```bash
uvx clai
```

You can also install `clai` globally with:

```bash
uv tool install clai
```

This will start an interactive session where you can chat with the AI model. Special commands available in interactive mode:

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
uvx clai --model anthropic:claude-3-7-sonnet-latest
```

(a full list of models available can be printed with `uvx clai --list-models`)
