---
description: "Use your Claude subscription with Pydantic AI through Claude Code, signed in with your own Claude login, instead of a pay-per-token API key."
---

# Claude Code

Use your [Claude subscription](https://claude.com/pricing) with Pydantic AI instead of a pay-per-token API key. The `claude-code` provider runs [Claude Code](https://code.claude.com/docs/en/overview), unmodified, through the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), and Claude Code signs each request with the login you set up for it. Pydantic AI never reads, stores or forwards your Claude credentials. For API keys, use the [`anthropic` provider](anthropic.md) instead.

!!! warning "For your own use only"
    Your use of Claude Code is governed by your agreement with Anthropic. Anthropic's [legal and compliance page](https://code.claude.com/docs/en/legal-and-compliance) lets you sign in to the unmodified Claude Code with your own subscription, but doesn't let developers offer Claude subscription login in their products, including agents built on the Agent SDK, or route requests through a subscription on behalf of their users. Use the `claude-code` provider for your own work, and the [`anthropic` provider](anthropic.md) with an API key for anything you offer to others. Usage draws from your plan's limits.

## Install

To use the Claude Code provider, install `pydantic-ai-slim` with the `claude-code` optional group:

```bash
pip/uv-add "pydantic-ai-slim[claude-code]"
```

This installs the [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk) package, which includes a Claude Code executable for your platform (about 170 MB).

## Usage

Sign in to Claude Code once with `claude auth login`, from any Claude Code installation: every `claude` on the machine shares the login stored in its configuration directory. Claude Code always uses that login: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL` and the `CLAUDE_CODE_USE_*` variables that switch it to a cloud provider don't reach it. Then use the `claude-code:` prefix:

```python {title="claude_code.py" test="skip - requires a Claude Code sign-in"}
from pydantic_ai import Agent

agent = Agent('claude-code:{{ anthropic:medium.name }}', instructions='Be concise.')
result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
```

This resolves to [`ClaudeCodeModel`][pydantic_ai.models.claude_code.ClaudeCodeModel] backed by [`ClaudeCodeProvider`][pydantic_ai.providers.claude_code.ClaudeCodeProvider]. Model names are Claude model IDs, or one of Claude Code's aliases, such as `sonnet`.

Use [`ClaudeCodeProvider`][pydantic_ai.providers.claude_code.ClaudeCodeProvider] to pick the `claude` executable (`cli_path`, the bundled one by default) or the working directory Claude Code runs in (`cwd`, the current one by default):

```python {title="claude_code_provider.py" test="skip - requires a Claude Code sign-in"}
from pydantic_ai import Agent
from pydantic_ai.models.claude_code import ClaudeCodeModel
from pydantic_ai.providers.claude_code import ClaudeCodeProvider

provider = ClaudeCodeProvider(cli_path='/usr/local/bin/claude', cwd='/srv/my-app')
agent = Agent(ClaudeCodeModel('{{ anthropic:medium.name }}', provider=provider))
```

## How it works

Each request goes to a Claude Code process. Your agent's [instructions](../agent.md#instructions) replace Claude Code's system prompt, and its [function tools](../tools.md) reach Claude Code as tools of an in-process MCP server: when the model calls one, Pydantic AI runs it, as with any other model. Claude Code's own tools, settings files, `CLAUDE.md` files, skills, hooks, MCP servers and claude.ai connectors are turned off, and so are its side requests, memory and automatic compaction.

A conversation keeps its Claude Code process for the length of an agent run, so the requests of one run continue in the same process and read from its prompt cache. To keep each conversation's process across runs too, for example one run per chat message, enter the agent (or the model) for as long as you serve the conversation:

```python {title="claude_code_chat.py" test="skip - requires a Claude Code sign-in"}
from pydantic_ai import Agent

agent = Agent('claude-code:{{ anthropic:medium.name }}')


async def main():
    async with agent:
        result = await agent.run('Pick a number between 1 and 10.')
        result = await agent.run('Add 5.', message_history=result.all_messages())
        print(result.output)
```

A request whose history the process hasn't seen, such as one with different instructions or tools, edited history, or a conversation picked up from storage, starts a new process and sends it the history first. That's always correct, but it doesn't read from the earlier process's prompt cache. At most eight conversations keep a process at a time: the one used least recently makes room, and when all eight are busy, a new conversation's process stops after its request. When the agent, model, or run that holds them exits, the processes are stopped.

## What Claude Code adds to the request

With your instructions as its system prompt, Claude Code still adds:

- before your instructions, a billing block and the line "You are a Claude agent, built on Anthropic's Claude Agent SDK.";
- in the first user message, a `<system-reminder>` with the email address of the Claude account;
- after the latest user message, a system message describing the environment: the working directory, platform, shell, model name and date.

## Limitations

- Claude Code sets up the request itself, so [`ClaudeCodeModel`][pydantic_ai.models.claude_code.ClaudeCodeModel] ignores the sampling settings (`temperature`, `top_p`, `top_k`, `seed`, `presence_penalty`, `frequency_penalty` and `logit_bias`), as well as `stop_sequences`, `parallel_tool_calls=False`, `extra_headers`, `extra_body`, `service_tier`, `timeout` and `max_retries`. Claude Code retries failed requests itself. `max_tokens`, [`thinking`][pydantic_ai.settings.ModelSettings.thinking], `tool_choice` (except values that force a tool call, which raise a `UserError`), `anthropic_thinking` and `anthropic_effort` apply. Other `anthropic_*` settings are ignored with an `UnsupportedSettingWarning`.
- Claude Code places its own cache breakpoints, so the unified [`cache`][pydantic_ai.settings.ModelSettings.cache] setting and [`CachePoint`][pydantic_ai.messages.CachePoint]s have no effect.
- The model can't be forced to call a tool, so structured output uses [tool output](../output.md#tool-output), with a retry when the model answers in text. Native (JSON schema) output isn't available.
- Anthropic's server-side tools, such as web search and code execution, aren't available, and neither are uploaded files, batches, on-demand compaction, fast mode or [token counting](../agent.md#usage-limits) ahead of a request.
- Documents and images in a user prompt reach the model, and so does [`citations`][pydantic_ai.settings.ModelSettings.citations] for documents. A tool's result reaches Claude Code over MCP, which carries text and images only, so a document a tool returns is replaced by a note.
- Claude Code runs a conversation's tool calls one at a time, after the model's response ends.
- Errors Claude Code reports before it responds come as the [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] the Anthropic API would have raised: a [`ModelAuthenticationError`][pydantic_ai.exceptions.ModelAuthenticationError] when Claude Code isn't signed in, and a status of 429 with a `Retry-After` header when you've reached your plan's limit. When Claude Code fails to start, or exits before it responds, the error has a status of 500. A failure after the response has started raises a [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], as a broken API stream would.
