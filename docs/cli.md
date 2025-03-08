# Command Line Interface (CLI)

**PydanticAI** comes with a simple reference CLI application which you can use to interact with various LLMs directly from the command line.
It provides a convenient way to chat with language models and quickly get answers right in the terminal.

We originally developed this CLI for our own use, but found ourselves using it so frequently that we decided to share it as part of the PydanticAI package.

We plan to continue adding new features, such as interaction with MCP servers, access to tools, and more.

## Installation

To use the CLI, you need to either install [`pydantic-ai`](install.md), or install
[`pydantic-ai-slim`](install.md#slim-install) with the `cli` optional group:

```bash
pip/uv-add 'pydantic-ai[cli]'
```

To enable command-line argument autocompletion, run:

```bash
register-python-argcomplete pai >> ~/.bashrc  # for bash
register-python-argcomplete pai >> ~/.zshrc   # for zsh
```

## Usage

You'll need to set an environment variable depending on the provider you intend to use.

If using OpenAI, set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Then simply run:

```bash
$ pai
```

This will start an interactive session where you can chat with the AI model. Special commands available in interactive mode:

- `/exit`: Exit the session
- `/markdown`: Show the last response in markdown format
- `/multiline`: Toggle multiline input mode (use Ctrl+D to submit)

### Choose a model

You can specify which model to use with the `--model` flag:

```bash
$ pai --model=openai:gpt-4 "What's the capital of France?"
```
