---
title: Computer Use
description: "Let a Pydantic AI agent see and control a desktop with the ComputerUse capability: screenshots, mouse and keyboard on the display of its workspace."
---

# Computer Use

`ComputerUse` lets your agent use a desktop. It adds one tool, `computer`. The model calls it with a list of mouse and
keyboard actions. The actions run on the screen of the run's [workspace](../workspace.md), and the tool returns a
screenshot of the result.

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

1. Any capability whose workspace has a screen, such as `DockerContainer(desktop=True)`. To write your own, see
   [Giving a workspace a screen](#giving-a-workspace-a-screen).

On a [read-only](../workspace.md#read-only-access) workspace, the `computer` tool offers only the actions that change
nothing on the screen: `screenshot`, `zoom`, `cursor_position` and `wait`.

## What the model is told

`ComputerUse` adds instructions that tell the model how to group actions into calls. They also tell it never to
follow instructions shown on the screen. If the agent also has a tool that runs shell commands, such as `Shell`'s, the
model is told two more things: a program it starts from the shell opens on the screen, and it should use the screen
only for work that needs it.

With `ComputerUse(defer_loading=True)`, the model sees only the capability's `description` until it chooses to load
the tool. The description decides when the model loads it, so pass your own `description=` to change that.

## Models

The model must accept images. Every model gets `computer` as a regular function tool. Gemini 3 gives positions on a
grid from 0 to 999 instead of in pixels, and `ComputerUse` converts them to pixels.

## Computer use, Playwright, or Browser Use

| You want the agent to | Use |
|---|---|
| Work in desktop applications, or any website exactly as a person sees it | `ComputerUse` |
| Drive a browser with typed actions such as `navigate` and `click(selector)` | [Playwright Browser](playwright.md) |
| Hand a whole web task to a browsing sub-agent | [Browser Use](browser-use.md) |

## Configuration

| Setting | Default | What it does |
|---|---|---|
| `settle_seconds` | `2` | How long, at most, to wait for the screen to stop changing after the actions run. `ComputerUse` takes screenshots until two in a row match. `0` takes the screenshot at once. |
| `native` | `True` | If the model's provider has its own computer use tool, the model gets that tool instead: `ComputerUse` adds a [`ComputerTool`][pydantic_ai.native_tools.ComputerTool] and runs its calls on the screen. `False` gives every model the `computer` function tool with all its actions. On GPT, that adds zoom, triple-click, holding keys, pressing or releasing a mouse button, and reading the pointer position. |

Every request sends all the earlier screenshots again, so `ComputerUse` turns on
[prompt caching](../capabilities/caching.md) to make that cheaper. To turn caching off, set `cache=False` in the model
settings.

## Approving calls

`ComputerUse` runs every call without asking you first. To approve each call that clicks, types or does anything
else beyond looking at the screen, raise `ApprovalRequired` from a [`before_tool_execute` hook](../hooks.md):

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

The same hook can refuse a call instead, such as one that waits longer or types more than you allow. [Raise `ToolFailed`](../hooks.md#triggering-retries-with-modelretry) with a message for the model. None of the call's actions run.

## Screen size

Some models accept screenshots only up to a set size. Their profile sets it with
[`computer_use_screenshot_max_side`][pydantic_ai.profiles.ModelProfile.computer_use_screenshot_max_side] and
`computer_use_screenshot_max_pixels`. `ComputerUse` sends every screenshot at the screen's full size. A model is
refused the request that would send it a screenshot larger than it accepts, with a `UserError` that names the largest
size that fits. Under a fallback, the next model answers instead. A 1280x800 screen fits every model.

## When something goes wrong

- A call's actions run in order. If one fails, the actions after it don't run, and the result tells the model which
  action failed and why.
- If any action in a call has a position outside the screen, none of the call's actions run, and the model is asked to
  try again. Each such retry counts against the tool's retry limit.
- If the workspace is gone, for example because its sandbox was deleted, the run ends with [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError].
- Several agents can share one screen, such as sub-agents running in parallel. If they run in the same `asyncio` event loop, they take turns one whole call at a time, so their clicks never mix. Agents in separate event loops or separate processes don't wait for each other.
- A mouse button pressed with `mouse_down` stays pressed across calls until a `mouse_up`, and each result tells the model which buttons are still pressed. `ComputerUse` releases them when a call fails or is cancelled while it acts on the screen, and when the run ends, so a drag must finish within one run, approvals included. If the model loads `ComputerUse` during the run, because of `defer_loading=True`, the buttons are released at the end of every call, so a drag must happen within one call. If a release after a failed call fails too, the result says the button may still be pressed.

## Watching a run

Each screenshot arrives in a `FunctionToolResultEvent` for the `computer` tool. An
[event stream handler](../agent.md#streaming-all-events) can save or show the screenshots while the run goes on.

Each call that finishes with a screenshot also sends one
[capability event](../capabilities/overview.md#capability-events) in the `computer_use` namespace. Read it with
`@agent.on_event` or `run_stream_events()`:

| Event | Payload |
|---|---|
| `ActionsPerformedEvent` | `result`, the call's `ComputerResult`, and `tool_call_id` |

On Temporal, event listeners run in workflow code, so they [run again when Temporal replays the workflow](../durable_execution/temporal.md#choosing-what-to-publish).
The message history holds the same record. Each result is a `ComputerReturnPart`, and its `result` is the call's
`ComputerResult`. `ComputerUse` adds no tracing spans of its own, because the tool call already has one.

## Limitations

- Every request sends every earlier screenshot again. They are the main cost, which prompt caching lowers, and a
  long run reaches the provider's limits: Claude, for example, refuses a request with more than 100 images on its
  models with a 200k-token context. `ComputerUse` doesn't drop old screenshots, and
  [`ClearToolResults`](compaction.md) doesn't clear `computer` results.
- Each `computer` call takes one model request, so a long task can reach the default [`request_limit`](../agent.md#usage-limits) of 50.
  To allow more, pass higher [`UsageLimits`][pydantic_ai.usage.UsageLimits].
- If the screen restarts, the model isn't told. Its next screenshot shows the new screen.
- [Instrumentation](../logfire.md#excluding-binary-content) records every screenshot. To leave them
  out, pass `InstrumentationSettings(include_binary_content=False)`.
- [`Advisor(forward_history=True)`](advisor.md) sends every screenshot to the advisor model. Anthropic refuses an
  advisor request with more than 600 images.
- With [`ModelFallback`](../capabilities/model-fallback.md) or a [`FallbackModel`](../models/overview.md#fallback-model), each model
  that answers is told which coordinates it uses: pixels, or Gemini 3's grid. A model that can't take images, or
  can't see the latest screenshot, is always skipped, whatever `fallback_on` says. If every model is skipped, the run
  ends with the last model's `UserError`; if another model failed in a way `fallback_on` moves on from, with a
  `FallbackExceptionGroup`. Any other failure is raised by itself. The previous `computer` call has already acted on
  the screen by then.

## Durable execution

Under [durable execution](../durable_execution/overview.md), everything one call does on the screen is one durable
step. The engine records each finished step, so replaying or recovering the run doesn't repeat a finished call. A call
cut off before its step was recorded can run again, as often as the engine's retry settings for capability steps
allow. When the engine stops retrying, the model is told that some actions may have run.

On Temporal, those settings are `TemporalDurability`'s `activity_config`. By default it has a 60-second timeout and
unlimited retries. Set `start_to_close_timeout` longer than your longest call. Set
`retry_policy=RetryPolicy(maximum_attempts=1)` so a call is never run twice. After a few steps, the screenshots also
pass Temporal's 2 MB payload limit, so configure [external storage](../durable_execution/temporal.md#large-payloads).

Use `TemporalDurability` or `PrefectDurability`. The deprecated `TemporalAgent`, `PrefectAgent` and `DBOSAgent` can't
run `ComputerUse`.

## Giving a workspace a screen

`ComputerUse` drives the screen of the run's workspace backend. A backend with a screen implements
[`SupportsDisplay`][pydantic_ai_harness.computer_use.SupportsDisplay]. That means it has a `display` property that
returns a [`Display`][pydantic_ai_harness.computer_use.Display], or `None` when there is no screen. The property must
return the same object every time, and must not do I/O such as network calls or starting programs.

A `Display` has three methods: `display_size()`, `screenshot()`, which returns PNG bytes, and `act()`, which runs one
[`DisplayAction`][pydantic_ai_harness.computer_use.DisplayAction], such as a click or a key press. Positions are in
screenshot pixels. A display can also implement:

- [`SupportsZoom`][pydantic_ai_harness.computer_use.SupportsZoom]: capture part of the screen at full resolution.
- [`SupportsCursorPosition`][pydantic_ai_harness.computer_use.SupportsCursorPosition]: report where the pointer is.

If a display doesn't implement one of these, a `zoom` or `cursor_position` action fails, and the model is told so.
Raise [`UnsupportedActionError`][pydantic_ai_harness.computer_use.UnsupportedActionError] for any action the display
can't do exactly, including action types added in later releases. A display doesn't have to inherit from these
classes. Any class with the right methods counts, as the `isinstance` check at the end shows:

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
