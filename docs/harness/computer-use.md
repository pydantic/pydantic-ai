---
title: Computer Use
description: "Let a Pydantic AI agent see and control a desktop with the ComputerUse capability: screenshots, mouse and keyboard on the display of its workspace."
---

# Computer Use

`ComputerUse` lets your agent use a desktop. It adds one tool, `computer`, that the model calls with mouse and keyboard
actions. The actions run on the screen of the run's [workspace](../workspace.md), and the tool returns a screenshot of
the result.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/computer_use/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Get started

```bash
pip/uv-add "pydantic-ai-harness[anthropic]"
```

Use your model provider's extra instead of `anthropic`. Set `ANTHROPIC_API_KEY`, and have Docker running.

```python {title="computer_use.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse, DockerContainer

agent = Agent('{{ anthropic:medium }}', capabilities=[DockerContainer(desktop=True), ComputerUse()])

result = agent.run_sync('Open Firefox and find the current Python release on python.org.')
print(result.output)
```

The first run builds a desktop image with Firefox, which takes a few minutes. The agent then looks at the screen,
clicks and types until it has the answer, and prints it. The container keeps running afterwards; to remove it, see
[Clean up](docker-container.md#clean-up).

## Models

The model must accept images. On the models listed under [OpenAI](../models/openai.md#computer-use), [Anthropic](../models/anthropic.md#computer-use) and [Google](../models/google.md#computer-use), it gets its provider's own computer use
tool. Every other model, and any model with `ComputerUse(native=False)`, gets `computer` as a regular function
tool.

## Sandboxes

| Sandbox | Turn on its screen |
|---|---|
| Docker | [`DockerContainer(desktop=True)`](docker-container.md#give-the-agent-a-desktop) |
| Daytona | [`DaytonaSandbox(desktop=True)`](daytona-sandbox.md#give-the-agent-a-desktop) |

For any other workspace, see [Your own display](#your-own-display).

### What the display needs

[Daytona](daytona-sandbox.md#give-the-agent-a-desktop) starts its own desktop. Every other sandbox above runs one
that needs these programs:

- Xvfb, unless an X server is already running
- `xdotool`
- `xwininfo` (Debian's `x11-utils`)
- `scrot`
- `setsid`
- `fluxbox`, `tint2` and `xterm`, for the windows, the panel and its terminal

The first use of the display installs any that are missing with `apt-get`, as root or with passwordless `sudo`. Add
them to your image so the first use doesn't wait for the install. If one of the first five can't be installed, the run
ends with a `WorkspaceUnavailableError` naming what's missing. Without the last three, the screen works without the
panel. To give the agent a browser, install one in your image, such as `firefox-esr`.

Drags move the pointer in steps with short pauses, so a web page's drag-and-drop sees them. Now and then a
page still misses one.

## Safety

The model controls a real screen, and anything it clicks or types happens. Run it in a sandbox that holds only what
the task needs. To approve each call that does more than look at the screen, raise `ApprovalRequired` from a
[`before_tool_execute` hook](../hooks.md):

```python {title="computer_use_ask_before_acting.py" test="skip" lint="skip"}
from pydantic_ai import ApprovalRequired
from pydantic_ai.capabilities import Hooks
from pydantic_ai.computer_use import COMPUTER_KIND

hooks = Hooks()


@hooks.on.before_tool_execute
async def ask_before_acting(ctx, *, call, tool_def, args):
    looking = {'screenshot', 'zoom', 'wait', 'cursor_position'}
    if tool_def.tool_kind == COMPUTER_KIND.id and not ctx.tool_call_approved:
        if any(action.type not in looking for action in args['actions']):
            raise ApprovalRequired
    return args
```

Gemini can flag an action for a person to approve. `ComputerUse` then asks for approval itself, after your hooks run,
so no hook can skip it. If your own hook already asked and the call was approved, that approval counts. See
[Google](../models/google.md#actions-that-need-approval).

A read-only workspace gets no `computer` tool, since actions on the screen can change anything the workspace holds.

## Durable execution

Use `TemporalDurability` or `PrefectDurability`; the deprecated `TemporalAgent`, `PrefectAgent` and `DBOSAgent` can't
run `ComputerUse`. Each `computer` call is one [durable](../durable_execution/overview.md) step, and a step cut off
before it was recorded can run again. On Temporal, set these on `TemporalDurability`:

- `activity_config=ActivityConfig(start_to_close_timeout=..., retry_policy=RetryPolicy(maximum_attempts=1))`: a
  timeout longer than your longest call (the default is 60 seconds), and one attempt so a call never runs twice (the
  default retries without limit).
- `model_activity_config=ActivityConfig(retry_policy=RetryPolicy())`, and `toolset_activity_config` for other tools:
  this gives them back the retries the line above takes away.
- [External storage](../durable_execution/temporal.md#large-payloads): after a few steps, the screenshots pass
  Temporal's 2 MB payload limit.

## Your own display

To give your own workspace backend a screen, implement
[`SupportsDisplay`][pydantic_ai_harness.computer_use.SupportsDisplay]: a `display` property that returns the same
[`Display`][pydantic_ai_harness.computer_use.Display] every time, or `None` for no screen, without doing I/O. A
`Display` has `display_size()`, `screenshot()`, which returns PNG bytes, and `act()`, which runs one
[`DisplayAction`][pydantic_ai_harness.computer_use.DisplayAction]. Positions are in screenshot pixels. Raise
[`UnsupportedActionError`][pydantic_ai_harness.computer_use.UnsupportedActionError] for an action it can't do exactly.
For the `zoom` and `cursor_position` actions, also implement
[`SupportsZoom`][pydantic_ai_harness.computer_use.SupportsZoom] and
[`SupportsCursorPosition`][pydantic_ai_harness.computer_use.SupportsCursorPosition]. See the
[API reference](#api-reference).

## Errors and limits

- If the workspace is gone, for example because its sandbox was deleted, the run ends with
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError].
- Each `computer` call takes one model request, so a long task can reach the default `request_limit` of 50. To allow
  more, pass higher [`UsageLimits`][pydantic_ai.usage.UsageLimits].
- Every request sends all the earlier screenshots again. They are the main cost, which `ComputerUse` lowers by turning
  on prompt caching, and a long run can reach a provider's image limit: Claude refuses a request with more than 100
  images on its models with a 200k-token context.
- A 1280x800 screen suits every model. A larger one costs more tokens per screenshot.
- On Claude, a screenshot over the model's image size limit fails the request or makes clicks miss; see
  [Anthropic](../models/anthropic.md#computer-use).
- If the screen changes size, a call that gives positions is refused until the model takes a new screenshot.
- When one `computer` call in a model response fails, the response's later `computer` calls don't run, and the model
  is told so.
- A drag must finish within one run, approvals included: when a run ends, held mouse buttons are released.

## API reference

::: pydantic_ai_harness.computer_use.ComputerUse

::: pydantic_ai_harness.computer_use.ActionsPerformedEvent

::: pydantic_ai_harness.computer_use.Display

::: pydantic_ai_harness.computer_use.SupportsDisplay

::: pydantic_ai_harness.computer_use.SupportsZoom

::: pydantic_ai_harness.computer_use.SupportsCursorPosition

::: pydantic_ai_harness.computer_use.DisplayAction

::: pydantic_ai_harness.computer_use.UnsupportedActionError
