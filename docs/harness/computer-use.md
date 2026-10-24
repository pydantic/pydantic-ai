---
title: Computer Use
description: "Let a Pydantic AI agent see and control a desktop with the ComputerUse capability: screenshots, mouse and keyboard on the display of its workspace."
---

# Computer Use

`ComputerUse` lets your agent see and control a graphical desktop. It adds one tool, `computer`: the model sends a
list of mouse and keyboard actions, they run on the display of the run's [workspace](../workspace.md), and the tool
returns a screenshot of the screen afterwards.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/computer_use/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

```bash
pip/uv-add "pydantic-ai-harness[anthropic]"
```

Swap `anthropic` for your model provider's extra.

## Usage

```python {title="computer_use.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse

agent = Agent(
    '{{ anthropic:medium }}',
    capabilities=[my_desktop_sandbox, ComputerUse()],  # (1)!
)


async def main():
    result = await agent.run('Open the text editor and write a haiku about sandboxes.')
    print(result.output)
```

1. Any capability whose workspace has a display: its backend implements
   [`SupportsDisplay`][pydantic_ai.workspaces.SupportsDisplay]. See
   [Controlling a display](../workspace.md#controlling-a-display) to give a sandbox one. On a
   [read-only](../workspace.md#read-only-access) workspace the agent can take screenshots but not act.

The model must accept images. Gemini 3 models point on a 0-999 grid, which `ComputerUse` scales to pixels for them.

## Computer use, Playwright, or Browser Use

| You want the agent to | Use |
|---|---|
| Work in desktop applications, or any website exactly as a person sees it | `ComputerUse` |
| Drive a browser with typed actions such as `navigate` and `click(selector)` | [Playwright Browser](playwright.md) |
| Hand a whole web task to a browsing sub-agent | [Browser Use](browser-use.md) |

## Configuration

| Setting | Default | What it does |
|---|---|---|
| `requires_approval` | `False` | `True` asks before any call that clicks or types; a set of action types, such as `{'type', 'keypress', 'hold_key'}`, asks only before those. Needs a [deferred tool handler](../deferred-tools.md#human-in-the-loop-tool-approval). Turn it on for anything you'd regret an agent clicking. |
| `keep_screenshots` | `None` | Send only the most recent N screenshots (between N and N+24, so prompt caching keeps working); the run's history keeps all of them. Use it to bound long runs, but not on Claude Fable 5.1 or the Claude 5.5 models: dropping a screenshot makes Claude lose its earlier reasoning. |
| `settle_seconds` | `0.5` | Pause after acting before the screenshot, so animations and page loads can finish. |

A call its provider flags with safety checks, as OpenAI does, needs approval even with `requires_approval=False`.
Give such an agent a deferred tool handler, such as [`HandleDeferredToolCalls`](../deferred-tools.md); without one,
a flagged call ends the run with an error. The checks are on the call's `provider_details`.

## Limitations

- A 1280x800 screenshot costs about 1,200 input tokens on GPT-5.5 and 1,340 on Claude Sonnet 5.5, and every later
  request carries it again. With [prompt caching](../capabilities/caching.md) (Claude needs `Caching()`; OpenAI
  caches on its own), screenshots already sent cost about a tenth of that.
- Under [durable execution](../durable_execution/overview.md), a retried action may repeat a click that already
  happened, and long runs can exceed Temporal's payload size limit, since every screenshot rides in each request.
- Inside a [`FallbackModel`](../models/overview.md#fallback-model), Gemini 3 gets pixel coordinates and points less
  accurately, so don't put it in a fallback with other models.

## API reference

::: pydantic_ai_harness.computer_use.ComputerUse
