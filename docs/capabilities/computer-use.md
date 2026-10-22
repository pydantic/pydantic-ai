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
   [`SupportsComputerUse`][pydantic_ai.workspaces.SupportsComputerUse]. See
   [Controlling a display](../workspace.md#controlling-a-display) to give a sandbox one.

The agent keeps the rest of the workspace, so it can mix desktop steps with commands and file tools on the same
machine.

## Requirements

- **A workspace with a display.** A run without a workspace, or a workspace without a display, fails at its start
  with an error saying so. A [read-only](../workspace.md#read-only-access) workspace allows screenshots and reports
  every mouse and keyboard action to the model as refused.
- **A model that accepts images.** Every call returns a screenshot, so a text-only model fails at the start of the
  run.

## Computer use, Playwright, or Browser Use

| You want the agent to | Use |
|---|---|
| Work in desktop applications, or any website exactly as a person sees it | `ComputerUse` |
| Drive a browser with typed actions such as `navigate` and `click(selector)` | [Playwright Browser](../harness/playwright.md) |
| Hand a whole web task to a browsing sub-agent | [Browser Use](../harness/browser-use.md) |

## Approving actions

To approve each `computer` call before it runs, require approval for the tool as for any other; see
[deferred tools](../deferred-tools.md#human-in-the-loop-tool-approval).

## Limits

- Each call adds a screenshot to the message history, about 1,000 to 3,000 input tokens on most models. Long runs
  get expensive; trim old messages with [`ProcessHistory`](process-history.md).
- Under [durable execution](../durable_execution/overview.md), a retried tool call can repeat actions it already
  performed, such as a click.
