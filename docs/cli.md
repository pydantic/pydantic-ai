# Command Line Interface (CLI)

**PydanticAI** comes with a simple reference CLI application which you can use to interact with various LLMs directly from the command line.
It provides a convenient way to chat with language models and quickly get answers right in the terminal.

We originally developed this CLI for our own use, but found ourselves using it so frequently that we decided to share it as part of the PydanticAI package.

We plan to continue adding new features, such as interaction with MCP servers, access to tools, and more.

## Installation

To use the CLI, you need to either install [`pydantic-ai`](install.md), or install
[`pydantic-ai-slim`](install.md#slim-install) with the `cli` optional group:

```bash
pip/uv-add "pydantic-ai[cli]"
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
pai
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

### Configure MCP Servers

You can configure MCP (Model Control Protocol) servers using the `--edit-mcp-servers` flag:

```bash
$ pai --edit-mcp-servers
```

This will open your default text editor (or the one specified in your `EDITOR` environment variable)
to edit the MCP servers configuration file located at `~/.pydantic-ai/mcp_servers.json`.
The configuration file uses the following format:

```json
{
  "mcpServers": {
    "my-stdio-server": {
      "command": "uvx",
      "args": ["mcp_server"]
    },
    "my-http-server": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

For more information about MCP servers and their configuration, visit the [MCP documentation](mcp/index.md).

### Usage with `uvx`

If you have [uv](https://docs.astral.sh/uv/) installed, the quickest way to run the CLI is with `uvx`:

```bash
uvx --from pydantic-ai pai
```
