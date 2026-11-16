---
title: Computer Use
description: "Let a Pydantic AI agent see and control a desktop with the ComputerUse capability: screenshots, mouse and keyboard on the display of its workspace."
---

# Computer Use

`ComputerUse` lets your agent use a desktop. It adds one tool, `computer`: the model sends mouse and keyboard actions,
they run on the display of the run's [workspace](../workspace.md), and the tool returns a screenshot.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/computer_use/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

```bash
pip/uv-add "pydantic-ai-harness[anthropic]"
```

Use your model provider's extra instead of `anthropic`.

## Usage

```python {title="computer_use.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse

agent = Agent('{{ anthropic:medium }}', capabilities=[my_desktop_sandbox, ComputerUse()])  # (1)!


async def main():
    result = await agent.run('Open the text editor and write a haiku about sandboxes.')
    print(result.output)
```

1. Any capability whose workspace has a display, such as `DockerSandbox(desktop=True)`. See
   [Giving a workspace a screen](#giving-a-workspace-a-screen) to write your own.

On a [read-only](../workspace.md#read-only-access) workspace the agent can take screenshots but not act.

A `screenshot` action before a call's last action returns the screen at that point, besides the screenshot the
call ends with.

## What the model is told

The instructions tell the model that the screen belongs to the computer its shell and file tools run on, so a program
it starts from the shell opens on the screen. They tell it to prefer those tools unless the work needs the screen:
using a program's window, checking what the screen shows, or doing what the user asked to be done on the screen.

With `defer_loading=True`, the model reads only the capability's `description` until it loads it. The default names
the same three cases. Pass `description=` to change when the model loads it.

## Models

The model must accept images.

- GPT-5.4 (not nano), GPT-5.5, GPT-5.6 and GPT-6 get OpenAI's own computer tool through `OpenAIResponsesModel` on OpenAI and Bedrock Mantle; see
  [OpenAI](../models/openai.md#native-tools).
- Every other model gets `computer` as a regular function tool.

Gemini 3 points on a 0-999 grid, which `ComputerUse` converts to pixels.

## Computer use, Playwright, or Browser Use

| You want the agent to | Use |
|---|---|
| Work in desktop applications, or any website exactly as a person sees it | `ComputerUse` |
| Drive a browser with typed actions such as `navigate` and `click(selector)` | [Playwright Browser](playwright.md) |
| Hand a whole web task to a browsing sub-agent | [Browser Use](browser-use.md) |

## Configuration

| Setting | Default | What it does |
|---|---|---|
| `settle_seconds` | `0` | Wait this long after acting before the screenshot. Set it when screenshots show pages still loading or animating. |
| `native` | `True` | Use the model's own computer use tool where it has one, by adding a [`ComputerTool`][pydantic_ai.native_tools.ComputerTool] that this tool runs. `False` gives every model this tool and all its actions; on GPT that adds zoom, triple-click, holding keys, pressing or releasing a mouse button, and reading the pointer position. |

`ComputerUse` turns on [prompt caching](../capabilities/caching.md), since every request resends the screenshots. Set
`cache=False` in the model settings to turn it off.

## Approving calls

`ComputerUse` runs every call without asking. To ask before every call that does more than look, raise
`ApprovalRequired` from a [`before_tool_execute` hook](../hooks.md):

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

The same hook can refuse a call instead, such as one that waits or types more than you allow: [raise `ToolFailed`](../hooks.md#triggering-retries-with-modelretry) with a message for the model, and no action runs.

## Screen size

Some models only take screenshots up to a size, set by
[`computer_use_screenshot_max_side`][pydantic_ai.profiles.ModelProfile.computer_use_screenshot_max_side] and
`computer_use_screenshot_max_pixels` in their profile. `ComputerUse` sends screenshots at the screen's size, so a
larger screen stops the run with an error naming the largest size that fits. A 1280x800 screen fits every model.

GPT-5.5 clicks accurately up to 2560x1440 and GPT-5.4 up to 2048x1280; larger screens make them miss. Keep the screen at
or below those sizes.

## When something goes wrong

- Actions run in order. If one fails, the rest don't run, and the result says which failed and why.
- A position outside the screen fails the whole call before anything runs.
- If the environment is gone, the run ends with [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError].
- Agents on one screen and one event loop, such as parallel sub-agents, take turns a whole call at a time, so their clicks never interleave. Agents on separate event loops or in separate processes don't wait for each other.
- A button pressed with `mouse_down` stays pressed across calls until a `mouse_up`, so a drag can span calls, and each result says which buttons are still pressed. A call that fails or is cancelled releases them, and so does the end of the run however it ends, so a drag must finish within one run, which matters when calls wait for approval. A `ComputerUse` loaded mid-run with `defer_loading=True` releases them at the end of each call instead. A failed call says which buttons it released, or that a button may still be held if its release failed too.

## Watching a run

Each screenshot is in a `FunctionToolResultEvent` for the `computer` tool, so an
[event stream handler](../agent.md#streaming-all-events) can save or show them as the run goes.

Each call that finishes and returns a screenshot also emits
[capability events](../capabilities/overview.md#capability-events) in the `computer_use` namespace, read with
`@agent.on_event` or `run_stream_events()`:

| Event | Payload |
|---|---|
| `ActionsPerformedEvent` | `result`, the call's `ComputerResult`, and `tool_call_id` |

They are emitted once per such call, also under durable execution, and add no telemetry spans, since core already traces
the tool call. The history keeps the same facts: each result is a `ComputerReturnPart` whose `result` is the call's
`ComputerResult`.

## Limitations

- Screenshots are the main cost, resent with every request. Prompt caching makes the resends cheaper.
- Every request carries every earlier screenshot, so a long run reaches the provider's limits: Claude refuses a
  request with more than 100 images on its 200k-context models. `ComputerUse` doesn't drop
  old screenshots, and [`ClearToolResults`](compaction.md) leaves `computer` results alone.
- Each call is one model request, so long tasks can hit the default [`request_limit`](../agent.md#usage-limits) of 50.
  Pass higher [`UsageLimits`][pydantic_ai.usage.UsageLimits].
- Under [durable execution](../durable_execution/overview.md), each call's screen work is one durable step, so replay or recovery doesn't
  repeat a finished call. A call cut off before its step was recorded can run again under the engine's retry settings
  for capability steps. On Temporal those are `TemporalDurability`'s `activity_config`, which defaults to a 60-second
  timeout and unlimited retries: raise `start_to_close_timeout` above your longest call, and set
  `retry_policy=RetryPolicy(maximum_attempts=1)` so a call isn't repeated. When the engine gives up, the model is told some actions may have run. Use `TemporalDurability`
  or `PrefectDurability`; the deprecated `TemporalAgent`, `PrefectAgent` and `DBOSAgent` can't run it. A run under
  `AbsurdDurability` or `PGTaskDurability` fails with a `UserError` before its first call: they checkpoint the whole
  tool call, so on replay a button the call pressed would never be released.
- On Temporal, screenshots pass the 2 MB payload limit after a few steps. Configure
  [external storage](../durable_execution/temporal.md#large-payloads).
- If the display restarts, the model isn't told; its next screenshot shows the new screen.
- [Instrumentation](../logfire.md#excluding-binary-content) records every screenshot. Pass
  `InstrumentationSettings(include_binary_content=False)` to leave them out.
- [`Advisor(forward_history=True)`](advisor.md) forwards every screenshot; Anthropic refuses one past 600 images.
- With [`ModelFallback`](../capabilities/model-fallback.md), a model that takes over is told its own coordinates
  (pixels or a grid). One that can't take images, or would see the latest screenshot scaled down, ends the run with an
  error before its request. In a [`FallbackModel`](../models/overview.md#fallback-model), Gemini 3 gets its grid only if
  every model in it does.

## Giving a workspace a screen

`ComputerUse` drives the display of the run's workspace backend. A backend with a screen implements
[`SupportsDisplay`][pydantic_ai_harness.computer_use.SupportsDisplay]: a `display` property that returns a
[`Display`][pydantic_ai_harness.computer_use.Display], or `None` without a screen. It must not do I/O, and returns the
same object every time.

A `Display` has three methods: `display_size()`, `screenshot()` (PNG bytes), and `act()`, which runs one
[`DisplayAction`][pydantic_ai_harness.computer_use.DisplayAction] such as a click or a key press, in screenshot
pixels. A display can also implement:

- [`SupportsZoom`][pydantic_ai_harness.computer_use.SupportsZoom]: capture part of the screen at full resolution.
- [`SupportsCursorPosition`][pydantic_ai_harness.computer_use.SupportsCursorPosition]: report where the pointer is.

Without one of these, the model is told the action failed. Raise
[`UnsupportedActionError`][pydantic_ai_harness.computer_use.UnsupportedActionError] for any action the display can't
do exactly, including action types added later. The protocols are structural, so a display needs no imports from
here:

```python {title="my_display.py"}
from pydantic_ai.computer_use import Click
from pydantic_ai_harness.computer_use import DisplayAction, Display, UnsupportedActionError


class MyDisplay:
    async def display_size(self) -> tuple[int, int]:
        return (1280, 800)

    async def screenshot(self) -> bytes:
        raise NotImplementedError  # capture the screen as PNG bytes

    async def act(self, action: DisplayAction) -> None:
        if not isinstance(action, Click):
            raise UnsupportedActionError(f'this display cannot {action.type}')
        # click at (action.x, action.y), or at the pointer when they're `None`


assert isinstance(MyDisplay(), Display)
```

## API reference

::: pydantic_ai_harness.computer_use.ComputerUse

::: pydantic_ai_harness.computer_use.ActionsPerformedEvent

::: pydantic_ai_harness.computer_use.Display

::: pydantic_ai_harness.computer_use.SupportsDisplay

::: pydantic_ai_harness.computer_use.SupportsZoom

::: pydantic_ai_harness.computer_use.SupportsCursorPosition

::: pydantic_ai_harness.computer_use.DisplayAction

::: pydantic_ai_harness.computer_use.UnsupportedActionError
