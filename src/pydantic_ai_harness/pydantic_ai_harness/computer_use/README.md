# Computer Use

`ComputerUse` lets your agent see and control a graphical desktop. It adds one tool, `computer`: the model sends a
list of mouse and keyboard actions, they run on the display of the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), and the tool
returns a screenshot of the screen afterwards.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/computer_use/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Installation

uv:

```bash
uv add "pydantic-ai-harness[anthropic]"
```

pip:

```bash
pip install "pydantic-ai-harness[anthropic]"
```

Swap `anthropic` for your model provider's extra.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse

agent = Agent(
    'anthropic:claude-sonnet-5-5',
    capabilities=[my_desktop_sandbox, ComputerUse()],  # (1)!
)


async def main():
    result = await agent.run('Open the text editor and write a haiku about sandboxes.')
    print(result.output)
```

1. Any capability whose workspace has a display: its backend implements
   [`SupportsDisplay`](https://pydantic.dev/docs/ai/api/pydantic-ai/workspaces/#pydantic_ai.workspaces.SupportsDisplay). See
   [Controlling a display](https://pydantic.dev/docs/ai/core-concepts/workspace/#controlling-a-display) to give a sandbox one.

Gemini 3 models point on a 0-999 grid, which `ComputerUse` scales to pixels for them.

## Requirements

- **A workspace with a display.** On a [read-only](https://pydantic.dev/docs/ai/core-concepts/workspace/#read-only-access) workspace the agent can take
  screenshots but not act.
- **A model that accepts images.**

## Computer use, Playwright, or Browser Use

| You want the agent to | Use |
|---|---|
| Work in desktop applications, or any website exactly as a person sees it | `ComputerUse` |
| Drive a browser with typed actions such as `navigate` and `click(selector)` | [Playwright Browser](../playwright/) |
| Hand a whole web task to a browsing sub-agent | [Browser Use](../browser_use/) |

## Configuration

| Setting | Default | What it does |
|---|---|---|
| `require_approval` | `False` | Ask before any call that clicks or types; needs a [deferred tool handler](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/#human-in-the-loop-tool-approval). Turn it on for anything you'd regret an agent clicking. |
| `keep_screenshots` | `None` | Send only the most recent N screenshots (between N and N+24, so prompt caching keeps working); the run's history keeps all of them. Use it to bound long runs. |
| `settle_seconds` | `0.5` | Pause after acting before the screenshot, so animations and page loads can finish. |

A call its provider flags with safety checks, as OpenAI does, needs approval even with `require_approval=False`.
Give such an agent a deferred tool handler, such as [`HandleDeferredToolCalls`](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/); without one,
a flagged call ends the run with an error. The checks are on the call's `provider_details`.

## Limitations

- A 1280x800 screenshot costs about 1,200 input tokens on GPT-5.5 and 1,340 on Claude Sonnet 5.5, and every later
  request carries it again. With [prompt caching](https://pydantic.dev/docs/ai/capabilities/caching/) (Claude needs `Caching()`; OpenAI
  caches on its own), screenshots already sent cost about a tenth of that.
- Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), a retried action may repeat a click that already
  happened, and long runs can exceed Temporal's payload size limit, since every screenshot rides in each request.
- Inside a [`FallbackModel`](https://pydantic.dev/docs/ai/models/overview/#fallback-model), Gemini 3 gets pixel coordinates and points less
  accurately, so don't put it in a fallback with other models.
