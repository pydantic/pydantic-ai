# clai

[![CI](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai)
[![PyPI](https://img.shields.io/pypi/v/clai.svg)](https://pypi.python.org/pypi/clai)
[![versions](https://img.shields.io/pypi/pyversions/clai.svg)](https://github.com/pydantic/pydantic-ai)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai.svg?v)](https://github.com/pydantic/pydantic-ai/blob/main/LICENSE)

(pronounced "clay")

Command line interface to chat to LLMs, part of the [PydanticAI project](https://github.com/pydantic/pydantic-ai).

## Usage

```bash
uvx clai
```

Or,

```bash
pip install clai
clai
```

## Help

```
usage: clai [-h] [-m [MODEL]] [-l] [-t [CODE_THEME]] [--no-stream] [--version] [prompt]

PydanticAI CLI v...

Special prompts:
* `/exit` - exit the interactive mode (ctrl-c and ctrl-d also work)
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
  --no-stream           Disable streaming from the model
  --version             Show version and exit
```
