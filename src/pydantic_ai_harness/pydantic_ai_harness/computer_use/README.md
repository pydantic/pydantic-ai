# Computer Use

`ComputerUse` lets your agent see and control a graphical desktop. It adds one tool, `computer`: the model sends a
list of mouse and keyboard actions, they run on the display of the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), and the tool
returns a screenshot of the screen afterwards.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/computer_use/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Usage

uv:

```bash
uv add pydantic-ai-harness
```

pip:

```bash
pip install pydantic-ai-harness
```

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

The agent keeps the rest of the workspace, so it can mix desktop steps with commands and file tools on the same
machine.

## Requirements

- **A workspace with a display.** A run without a workspace, or a workspace without a display, fails at its start
  with an error saying so. A [read-only](https://pydantic.dev/docs/ai/core-concepts/workspace/#read-only-access) workspace allows screenshots and reports
  every mouse and keyboard action to the model as refused.
- **A model that accepts images.** Every call returns a screenshot, so a request to a text-only model fails before
  it is sent, including a model that [`SelectModel`](https://pydantic.dev/docs/ai/capabilities/select-model/) picks later in the run.

## Computer use, Playwright, or Browser Use

| You want the agent to | Use |
|---|---|
| Work in desktop applications, or any website exactly as a person sees it | `ComputerUse` |
| Drive a browser with typed actions such as `navigate` and `click(selector)` | [Playwright Browser](../playwright/) |
| Hand a whole web task to a browsing sub-agent | [Browser Use](../browser_use/) |

## Settings

| Setting | Default | What it does |
|---|---|---|
| `require_approval` | `False` | Ask before any call that does more than look or wait; the run needs a [deferred tool handler](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/#human-in-the-loop-tool-approval). Turn it on for anything you'd regret an agent clicking. |
| `keep_screenshots` | `None` | The fewest recent screenshots each model request carries; older ones become a short note in the request, 25 at a time so the prompt cache keeps working in between: `keep_screenshots=5` sends between 5 and 29. The run's history keeps them all. Use it to bound a long run; a short run costs the same either way. |
| `settle_seconds` | `0.5` | Pause after acting before the screenshot, so animations and page loads can finish. |

A call whose provider attached safety checks to it waits for approval even without `require_approval`: the reviewer
finds the checks on the call's `provider_details`. Without a way to approve calls, such as
[`HandleDeferredToolCalls`](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/), the run stops at that call instead of running it.

## Limits

- A 1280x800 screenshot costs about 1,200 input tokens on GPT-5.5 and 1,340 on Claude Sonnet 5.5, and every later
  request carries it again. With [prompt caching](https://pydantic.dev/docs/ai/capabilities/caching/) (Claude needs `Caching()`; OpenAI
  caches on its own), screenshots already sent cost about a tenth of that.
- Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), each action and screenshot is its own durable
  workspace operation, so a retry repeats only that one, which may already have taken effect, such as a click.
  Every screenshot in the history rides in each model request's payload, so long runs can reach Temporal's payload
  size limit.
- Inside a [`FallbackModel`](https://pydantic.dev/docs/ai/models/overview/#fallback-model), every model gets pixel coordinates, since the
  fallback has no profile of its own. Gemini 3 models point less accurately in pixels than on their own grid.
