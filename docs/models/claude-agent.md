---
description: "Use your logged-in Claude account, including Claude Code subscriptions (Pro and Max), with Pydantic AI through the Claude Agent SDK, instead of a pay-per-token API key."
---

# Claude Agent

The `claude-agent` provider runs your agent on your logged-in Claude account, including [Claude Code subscriptions](https://claude.com/pricing) (Pro and Max), instead of a pay-per-token API key. It runs [Claude Code](https://code.claude.com/docs/en/overview), unmodified, through the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), and Claude Code signs each request with the login you set up for it. Pydantic AI never reads, stores or forwards your Claude credentials. For API keys, use the [`anthropic` provider](anthropic.md) instead.

!!! warning "For your own use only"
    Your use of Claude Code is governed by your agreement with Anthropic. Anthropic's [legal and compliance page](https://code.claude.com/docs/en/legal-and-compliance) lets you sign in to the unmodified Claude Code with your own subscription, but doesn't let developers offer Claude subscription login in their products, including agents built on the Agent SDK, or route requests through a subscription on behalf of their users. Use the `claude-agent` provider for your own work, and the [`anthropic` provider](anthropic.md) with an API key for anything you offer to others. Usage draws from your plan's limits.

## Install

To use the Claude Agent provider, install `pydantic-ai-slim` with the `claude-agent` optional group:

```bash
pip/uv-add "pydantic-ai-slim[claude-agent]"
```

This installs the [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk) package, which includes a Claude Code executable for your platform (about 170 MB).

## Usage

Sign in to Claude Code once with `claude auth login`, from any Claude Code installation: every `claude` on the machine shares the login stored in its configuration directory. Claude Code always uses that login: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL` and the `CLAUDE_CODE_USE_*` variables that switch it to a cloud provider don't reach it. Then use the `claude-agent:` prefix:

```python {title="claude_agent.py" test="skip - requires a Claude Code sign-in"}
from pydantic_ai import Agent

agent = Agent('claude-agent:{{ anthropic:medium.name }}', instructions='Be concise.')
result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
```

This resolves to [`ClaudeAgentModel`][pydantic_ai.models.claude_agent.ClaudeAgentModel] backed by [`ClaudeAgentProvider`][pydantic_ai.providers.claude_agent.ClaudeAgentProvider]. Model names are Claude model IDs, or one of Claude Code's aliases, such as `sonnet`.

Use [`ClaudeAgentProvider`][pydantic_ai.providers.claude_agent.ClaudeAgentProvider] to pick the `claude` executable (`cli_path`, the bundled one by default) or the working directory Claude Code runs in (`cwd`, the current one by default):

```python {title="claude_agent_provider.py" test="skip - requires a Claude Code sign-in"}
from pydantic_ai import Agent
from pydantic_ai.models.claude_agent import ClaudeAgentModel
from pydantic_ai.providers.claude_agent import ClaudeAgentProvider

provider = ClaudeAgentProvider(cli_path='/usr/local/bin/claude', cwd='/srv/my-app')
agent = Agent(ClaudeAgentModel('{{ anthropic:medium.name }}', provider=provider))
```

## How it works

Each request goes to a Claude Code process. Your agent's [instructions](../agent.md#instructions) replace Claude Code's system prompt, and its [function tools](../tools.md) reach Claude Code as tools of an in-process MCP server: when the model calls one, Pydantic AI runs it, as with any other model. Claude Code's own tools, settings files, `CLAUDE.md` files, skills, hooks, side requests, memory and automatic compaction are turned off, and so are your MCP servers and claude.ai connectors unless you [bring them](#your-own-mcp-servers-and-connectors).

A conversation keeps its Claude Code process for the length of an agent run, so the requests of one run continue in the same process and read from its prompt cache. To keep each conversation's process across runs too, for example one run per chat message, enter the agent (or the model) for as long as you serve the conversation:

```python {title="claude_agent_chat.py" test="skip - requires a Claude Code sign-in"}
from pydantic_ai import Agent

agent = Agent('claude-agent:{{ anthropic:medium.name }}')


async def main():
    async with agent:
        result = await agent.run('Pick a number between 1 and 10.')
        result = await agent.run('Add 5.', message_history=result.all_messages())
        print(result.output)
```

A request that no live process can continue, such as the next run without `async with`, a retry, one with different instructions or tools, or one that goes on from an earlier answer, starts a new process. When one of the provider's earlier processes ended an answer in the request's history, the new process forks that process's transcript at the latest such answer and is sent only the rest. With the same instructions, tools and model, it reads the history up to there from the prompt cache. Otherwise, as for a conversation picked up from storage after a restart, it's sent the whole history first: always correct, but without reading from the prompt cache. The provider remembers the last 256 answers its processes ended. At most eight conversations keep a process at a time: the one used least recently makes room, and when all eight are busy, a new conversation's process stops after its request. When the agent, model, or run that holds them exits, the processes are stopped.

Like any Claude Code session, each one leaves its transcript in Claude Code's configuration directory, under `~/.claude/projects/` by default, so it also shows up in `claude --resume` in the directory it ran in. Claude Code writes its prompt cache entries for 5 minutes, as the [`anthropic` provider](anthropic.md) asks for, rather than the hour it uses by default on a plan.

Under [durable execution](../durable_execution/overview.md), each model request starts its own Claude Code process. On the worker that ran the request before it, that process forks the earlier one's transcript and reads from the prompt cache; on another worker it's sent the whole history. Tools from [your own MCP servers](#your-own-mcp-servers-and-connectors) run inside the model request, so a retried request runs them again.

## Your own MCP servers and connectors

By default, Claude Code offers the model only your agent's tools. To let it also offer the MCP servers you set up for Claude Code and your Claude account's claude.ai connectors, set [`claude_agent_inherit_mcp_servers`][pydantic_ai.models.claude_agent.ClaudeAgentModelSettings.claude_agent_inherit_mcp_servers] to `True` for all of them, or to the names of the ones to use, as `claude mcp list` shows them:

```python {title="claude_agent_mcp.py" test="skip - requires a Claude Code sign-in"}
from pydantic_ai import Agent
from pydantic_ai.models.claude_agent import ClaudeAgentModelSettings

agent = Agent(
    'claude-agent:{{ anthropic:medium.name }}',
    model_settings=ClaudeAgentModelSettings(claude_agent_inherit_mcp_servers=['claude.ai Google Calendar']),
)
result = agent.run_sync('What do I have on tomorrow?')
print(result.output)
```

Claude Code runs those tools itself, as it would in a Claude Code session, and each call and its result come back in the response as a [`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart] and a [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart], like those of Anthropic's own [MCP connector](../native-tools.md#mcp-server-tool), so they're in the message history and your UI and traces show them. A tool result is data from that server, given to the model as a tool result. A request with a server name Claude Code doesn't have fails with a 400 that lists the ones it has.

Every one of a server's tools goes into each request, so bringing all of a large account's connectors can add tens of thousands of tokens; name the servers you need. The servers in a project's `.mcp.json` aren't brought, since Claude Code loads them only with the project's own settings, which could also run its hooks. When the model calls one of your agent's tools and one of Claude Code's in the same response, the result of Claude Code's comes at the start of the next response.

## What Claude Code adds to the request

With your instructions as its system prompt, Claude Code still adds:

- before your instructions, a billing block and the line "You are a Claude agent, built on Anthropic's Claude Agent SDK.";
- in the first user message, a `<system-reminder>` with the email address of the Claude account;
- after the latest user message, a system message describing the environment: the working directory, platform, shell, model name and date.

## Limitations

- Claude Code sets up the request itself, so [`ClaudeAgentModel`][pydantic_ai.models.claude_agent.ClaudeAgentModel] ignores the sampling settings (`temperature`, `top_p`, `top_k`, `seed`, `presence_penalty`, `frequency_penalty` and `logit_bias`), as well as `stop_sequences`, `parallel_tool_calls=False`, `extra_headers`, `extra_body`, `service_tier` and `timeout`. `max_tokens`, `max_retries` (Pydantic AI retries a failed request, as Claude Code is set not to, but doesn't wait for a plan's usage limit to reset; a [durable execution](../durable_execution/overview.md) engine that retries the model request may still wait for it, until its `Retry-After`), [`thinking`][pydantic_ai.settings.ModelSettings.thinking], `tool_choice` (except values that force a tool call, which raise a `UserError`), `anthropic_thinking` and `anthropic_effort` apply. Other `anthropic_*` settings are ignored with an `UnsupportedSettingWarning`.
- Claude Code places its own cache breakpoints, so the unified [`cache`][pydantic_ai.settings.ModelSettings.cache] setting is ignored with an `UnsupportedSettingWarning`, and [`CachePoint`][pydantic_ai.messages.CachePoint]s have no effect.
- The model can't be forced to call a tool, so structured output uses [tool output](../output.md#tool-output), with a retry when the model answers in text. Native (JSON schema) output isn't available.
- Anthropic's server-side tools, such as web search and code execution, aren't available, and neither are uploaded files, batches, on-demand compaction, fast mode or [token counting](../agent.md#usage-limits) ahead of a request.
- Documents and images in a user prompt reach the model, and so does [`citations`][pydantic_ai.settings.ModelSettings.citations] for documents. A tool's result reaches Claude Code over MCP, which carries text and images only, so a document a tool returns is replaced by a note.
- Claude Code runs a conversation's tool calls one at a time, after the model's response ends.
- Errors Claude Code reports before it responds come as the [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] the Anthropic API would have raised: a [`ModelAuthenticationError`][pydantic_ai.exceptions.ModelAuthenticationError] when Claude Code isn't signed in, and a status of 429 with a `Retry-After` header when you've reached your plan's limit. When Claude Code fails to start, or exits before it responds, the error has a status of 500. A failure after the response has started raises a [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], as a broken API stream would.
