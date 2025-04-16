# clai

PydanticAI CLI: command line interface to chat to LLMs.

## Usage

```bash
uvx clai
```

Or,

```
pip install clai
clai
```

## Help

```
usage: clai [-h] [-m [MODEL]] [-l] [-t [CODE_THEME]] [--no-stream] [--version] [prompt]

PydanticAI CLI v0.1.1.dev5+4d261d50

Special prompt:
* `/exit` - exit the interactive mode
* `/markdown` - show the last markdown output of the last question
* `/multiline` - toggle multiline mode

positional arguments:
  prompt                AI Prompt, if omitted fall into interactive mode

options:
  -h, --help            show this help message and exit
  -m [MODEL], --model [MODEL]
                        Model to use, in format "<provider>:<model>" e.g. "openai:gpt-4o" or "anthropic:claude-3-7-sonnet-latest". Defaults to "openai:gpt-4o".
  -l, --list-models     List all available models and exit
  -t [CODE_THEME], --code-theme [CODE_THEME]
                        Which colors to use for code, can be "dark", "light" or any theme from pygments.org/styles/. Defaults to "dark" which works well on dark terminals.
  --no-stream           Whether to stream responses from the model
  --version             Show version and exit
```
