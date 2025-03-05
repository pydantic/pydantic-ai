# Command Line Interface (CLI)

**PydanticAI** offers a CLI that lets you interact with various LLMs straight from your terminal.

The goal is to provide a user-friendly interface for engaging with the models without needing to write any code.

This CLI is adapted from [`aicli`](https://github.com/samuelcolvin/aicli/) by Samuel Colvin, with modifications to
work seamlessly with PydanticAI.

At Pydantic, we frequently use the CLI during development to query the models. We're sharing this tool with
PydanticAI users, opening up a new way to interact with the models. Moreover, we plan to enhance the CLI
with additional features, such as interaction with MCP servers, access to tools, and more.

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
