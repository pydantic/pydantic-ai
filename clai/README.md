# clai

[![CI](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai)
[![PyPI](https://img.shields.io/pypi/v/clai.svg)](https://pypi.python.org/pypi/clai)
[![versions](https://img.shields.io/pypi/pyversions/clai.svg)](https://github.com/pydantic/pydantic-ai)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai.svg?v)](https://github.com/pydantic/pydantic-ai/blob/main/LICENSE)

(pronounced "clay")

Command line interface to chat to LLMs, part of the [PydanticAI project](https://github.com/pydantic/pydantic-ai).

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

## Help

```
Usage: clai [OPTIONS] [PROMPT]...

  PydanticAI CLI v...

  Special prompts: * `/exit` - exit the interactive mode (ctrl-c and ctrl-d
  also work) * `/markdown` - show the last markdown output of the last
  question * `/multiline` - toggle multiline mode

Options:
  -m, --model [anthropic:claude-2.0|anthropic:claude-2.1|anthropic:claude-3-5-haiku-20241022|anthropic:claude-3-5-haiku-latest|anthropic:claude-3-5-sonnet-20240620|anthropic:claude-3-5-sonnet-20241022|anthropic:claude-3-5-sonnet-latest|anthropic:claude-3-7-sonnet-20250219|anthropic:claude-3-7-sonnet-latest|anthropic:claude-3-haiku-20240307|anthropic:claude-3-opus-20240229|anthropic:claude-3-opus-latest|anthropic:claude-3-sonnet-20240229|anthropic:claude-4-opus-20250514|anthropic:claude-4-sonnet-20250514|anthropic:claude-opus-4-0|anthropic:claude-opus-4-20250514|anthropic:claude-sonnet-4-0|anthropic:claude-sonnet-4-20250514|bedrock:amazon.titan-tg1-large|bedrock:amazon.titan-text-lite-v1|bedrock:amazon.titan-text-express-v1|bedrock:us.amazon.nova-pro-v1:0|bedrock:us.amazon.nova-lite-v1:0|bedrock:us.amazon.nova-micro-v1:0|bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0|bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0|bedrock:anthropic.claude-3-5-haiku-20241022-v1:0|bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0|bedrock:anthropic.claude-instant-v1|bedrock:anthropic.claude-v2:1|bedrock:anthropic.claude-v2|bedrock:anthropic.claude-3-sonnet-20240229-v1:0|bedrock:us.anthropic.claude-3-sonnet-20240229-v1:0|bedrock:anthropic.claude-3-haiku-20240307-v1:0|bedrock:us.anthropic.claude-3-haiku-20240307-v1:0|bedrock:anthropic.claude-3-opus-20240229-v1:0|bedrock:us.anthropic.claude-3-opus-20240229-v1:0|bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0|bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0|bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0|bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0|bedrock:anthropic.claude-opus-4-20250514-v1:0|bedrock:us.anthropic.claude-opus-4-20250514-v1:0|bedrock:anthropic.claude-sonnet-4-20250514-v1:0|bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0|bedrock:cohere.command-text-v14|bedrock:cohere.command-r-v1:0|bedrock:cohere.command-r-plus-v1:0|bedrock:cohere.command-light-text-v14|bedrock:meta.llama3-8b-instruct-v1:0|bedrock:meta.llama3-70b-instruct-v1:0|bedrock:meta.llama3-1-8b-instruct-v1:0|bedrock:us.meta.llama3-1-8b-instruct-v1:0|bedrock:meta.llama3-1-70b-instruct-v1:0|bedrock:us.meta.llama3-1-70b-instruct-v1:0|bedrock:meta.llama3-1-405b-instruct-v1:0|bedrock:us.meta.llama3-2-11b-instruct-v1:0|bedrock:us.meta.llama3-2-90b-instruct-v1:0|bedrock:us.meta.llama3-2-1b-instruct-v1:0|bedrock:us.meta.llama3-2-3b-instruct-v1:0|bedrock:us.meta.llama3-3-70b-instruct-v1:0|bedrock:mistral.mistral-7b-instruct-v0:2|bedrock:mistral.mixtral-8x7b-instruct-v0:1|bedrock:mistral.mistral-large-2402-v1:0|bedrock:mistral.mistral-large-2407-v1:0|cohere:c4ai-aya-expanse-32b|cohere:c4ai-aya-expanse-8b|cohere:command|cohere:command-light|cohere:command-light-nightly|cohere:command-nightly|cohere:command-r|cohere:command-r-03-2024|cohere:command-r-08-2024|cohere:command-r-plus|cohere:command-r-plus-04-2024|cohere:command-r-plus-08-2024|cohere:command-r7b-12-2024|deepseek:deepseek-chat|deepseek:deepseek-reasoner|google-gla:gemini-1.5-flash|google-gla:gemini-1.5-flash-8b|google-gla:gemini-1.5-pro|google-gla:gemini-1.0-pro|google-gla:gemini-2.0-flash|google-gla:gemini-2.0-flash-lite-preview-02-05|google-gla:gemini-2.0-pro-exp-02-05|google-gla:gemini-2.5-flash-preview-05-20|google-gla:gemini-2.5-flash|google-gla:gemini-2.5-flash-lite-preview-06-17|google-gla:gemini-2.5-pro-exp-03-25|google-gla:gemini-2.5-pro-preview-05-06|google-gla:gemini-2.5-pro|google-vertex:gemini-1.5-flash|google-vertex:gemini-1.5-flash-8b|google-vertex:gemini-1.5-pro|google-vertex:gemini-1.0-pro|google-vertex:gemini-2.0-flash|google-vertex:gemini-2.0-flash-lite-preview-02-05|google-vertex:gemini-2.0-pro-exp-02-05|google-vertex:gemini-2.5-flash-preview-05-20|google-vertex:gemini-2.5-flash|google-vertex:gemini-2.5-flash-lite-preview-06-17|google-vertex:gemini-2.5-pro-exp-03-25|google-vertex:gemini-2.5-pro-preview-05-06|google-vertex:gemini-2.5-pro|groq:distil-whisper-large-v3-en|groq:gemma2-9b-it|groq:llama-3.3-70b-versatile|groq:llama-3.1-8b-instant|groq:llama-guard-3-8b|groq:llama3-70b-8192|groq:llama3-8b-8192|groq:whisper-large-v3|groq:whisper-large-v3-turbo|groq:playai-tts|groq:playai-tts-arabic|groq:qwen-qwq-32b|groq:mistral-saba-24b|groq:qwen-2.5-coder-32b|groq:qwen-2.5-32b|groq:deepseek-r1-distill-qwen-32b|groq:deepseek-r1-distill-llama-70b|groq:llama-3.3-70b-specdec|groq:llama-3.2-1b-preview|groq:llama-3.2-3b-preview|groq:llama-3.2-11b-vision-preview|groq:llama-3.2-90b-vision-preview|heroku:claude-3-5-haiku|heroku:claude-3-5-sonnet-latest|heroku:claude-3-7-sonnet|heroku:claude-4-sonnet|heroku:claude-3-haiku|mistral:codestral-latest|mistral:mistral-large-latest|mistral:mistral-moderation-latest|mistral:mistral-small-latest|openai:chatgpt-4o-latest|openai:gpt-3.5-turbo|openai:gpt-3.5-turbo-0125|openai:gpt-3.5-turbo-0301|openai:gpt-3.5-turbo-0613|openai:gpt-3.5-turbo-1106|openai:gpt-3.5-turbo-16k|openai:gpt-3.5-turbo-16k-0613|openai:gpt-4|openai:gpt-4-0125-preview|openai:gpt-4-0314|openai:gpt-4-0613|openai:gpt-4-1106-preview|openai:gpt-4-32k|openai:gpt-4-32k-0314|openai:gpt-4-32k-0613|openai:gpt-4-turbo|openai:gpt-4-turbo-2024-04-09|openai:gpt-4-turbo-preview|openai:gpt-4-vision-preview|openai:gpt-4.1|openai:gpt-4.1-2025-04-14|openai:gpt-4.1-mini|openai:gpt-4.1-mini-2025-04-14|openai:gpt-4.1-nano|openai:gpt-4.1-nano-2025-04-14|openai:gpt-4o|openai:gpt-4o-2024-05-13|openai:gpt-4o-2024-08-06|openai:gpt-4o-2024-11-20|openai:gpt-4o-audio-preview|openai:gpt-4o-audio-preview-2024-10-01|openai:gpt-4o-audio-preview-2024-12-17|openai:gpt-4o-mini|openai:gpt-4o-mini-2024-07-18|openai:gpt-4o-mini-audio-preview|openai:gpt-4o-mini-audio-preview-2024-12-17|openai:gpt-4o-mini-search-preview|openai:gpt-4o-mini-search-preview-2025-03-11|openai:gpt-4o-search-preview|openai:gpt-4o-search-preview-2025-03-11|openai:o1|openai:o1-2024-12-17|openai:o1-mini|openai:o1-mini-2024-09-12|openai:o1-preview|openai:o1-preview-2024-09-12|openai:o3|openai:o3-2025-04-16|openai:o3-mini|openai:o3-mini-2025-01-31|openai:o4-mini|openai:o4-mini-2025-04-16]
                                  Model to use, in format "<provider>:<model>"
                                  e.g. "openai:gpt-4o" or
                                  "anthropic:claude-3-7-sonnet-latest".
                                  Defaults to "openai:gpt-4o".
  -a, --agent TEXT                Custom Agent to use, in format
                                  "module:variable", e.g.
                                  "mymodule.submodule:my_agent"
  -l, --list-models               List all available models and exit
  -t, --code-theme [dark|light]   Which colors to use for code, can be "dark",
                                  "light" or any theme from
                                  pygments.org/styles/. Defaults to "dark"
                                  which works well on dark terminals.
  --no-stream                     Disable streaming from the model
  --version                       Show version and exit
  -h, --help                      Show this message and exit.
```
