---
description: "Let a Pydantic AI agent see and control a desktop with the ComputerUse capability: screenshots, mouse and keyboard on the display of its workspace."
---

# Computer Use

The [`ComputerUse`][pydantic_ai.capabilities.ComputerUse] [capability](overview.md) lets your agent see and control a
graphical desktop. It adds one tool, `computer`: the model sends a list of mouse and keyboard actions, they run on
the display of the run's [workspace](../workspace.md), and the tool returns a screenshot of the screen afterwards.

```python {title="computer_use.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ComputerUse

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
   [Controlling a display](../workspace.md#controlling-a-display) to give a sandbox one.

The agent keeps the rest of the workspace, so it can mix desktop steps with commands and file tools on the same
machine.

On a model trained with a computer tool of its own, the `computer` tool is sent as that native tool, so the model uses
the vocabulary it was trained on: OpenAI's `computer` tool on GPT-5.4 and later. Every other model with image input
gets the same tool as a function.
OpenAI can attach safety checks to a call; they are acknowledged automatically when the call runs, so turn on
`require_approval` to review calls first. With `keep_screenshots` set, each request changes earlier history, so OpenAI's
`openai_previous_response_id` sends the whole conversation again instead of continuing from the last response.

## Requirements

- **A workspace with a display.** A run without a workspace, or a workspace without a display, fails at its start
  with an error saying so. A [read-only](../workspace.md#read-only-access) workspace allows screenshots and reports
  every mouse and keyboard action to the model as refused.
- **A model that accepts images.** Every call returns a screenshot, so a request to a text-only model fails before
  it is sent, including a model that [`SelectModel`](select-model.md) picks later in the run.

## Computer use, Playwright, or Browser Use

| You want the agent to | Use |
|---|---|
| Work in desktop applications, or any website exactly as a person sees it | `ComputerUse` |
| Drive a browser with typed actions such as `navigate` and `click(selector)` | [Playwright Browser](../harness/playwright.md) |
| Hand a whole web task to a browsing sub-agent | [Browser Use](../harness/browser-use.md) |

## Settings

| Setting | Default | What it does |
|---|---|---|
| `require_approval` | `False` | Ask before any call that does more than look or wait; the run needs a [deferred tool handler](../deferred-tools.md#human-in-the-loop-tool-approval). Turn it on for anything you'd regret an agent clicking. |
| `keep_screenshots` | `None` | The fewest recent screenshots each model request carries; older ones become a short note in the request, 25 at a time so the prompt cache keeps working in between: `keep_screenshots=5` sends between 5 and 29. The run's history keeps them all. Use it to bound a long run; a short run costs the same either way. |
| `settle_seconds` | `0.5` | Pause after acting before the screenshot, so animations and page loads can finish. |

## Limits

- A 1280x800 screenshot costs about 1,200 input tokens on GPT-5.5 and 1,340 on Claude Sonnet 5.5, and every later
  request carries it again. With [prompt caching](caching.md) (Claude needs `Caching()`; OpenAI caches on its own),
  screenshots already sent cost about a tenth of that.
- Under [durable execution](../durable_execution/overview.md), each action and screenshot is its own durable
  workspace operation, so a retry repeats only that one, which may already have taken effect, such as a click.
  Every screenshot in the history rides in each model request's payload, so long runs can reach Temporal's payload
  size limit.
- Inside a [`FallbackModel`](../models/overview.md#fallback-model), every model gets pixel coordinates, since the
  fallback has no profile of its own. Gemini 3 models point less accurately in pixels than on their own grid.
