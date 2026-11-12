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

Use your model provider's extra instead of `anthropic`. The example needs Docker; the first run builds a desktop
image, which takes a few minutes.

## Usage

```python {title="computer_use.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse, DockerSandbox

agent = Agent('{{ anthropic:medium }}', capabilities=[DockerSandbox(desktop=True), ComputerUse()])


async def main():
    result = await agent.run('Open Firefox and find the current Python release on python.org.')
    print(result.output)
```

The container keeps running after the run; see [Clean up](docker-sandbox.md#clean-up). If the run fails, remove it
in an `on_run_error` hook, as shown for [E2B](e2b-sandbox.md#clean-up).

### Where the screen comes from

| Capability | Screen |
|---|---|
| `DockerSandbox(desktop=True)` | A container on your machine |
| [`DaytonaSandbox(desktop=True)`](daytona-sandbox.md#give-the-agent-a-desktop) | A Daytona cloud sandbox |
| [`E2BSandbox(desktop=True)`](e2b-sandbox.md#give-the-agent-a-desktop) | An E2B cloud sandbox |
| [`ModalSandbox(desktop=True)`](modal-sandbox.md#give-the-agent-a-desktop) | A Modal cloud sandbox |
| [`SpritesSandbox(desktop=True)`](sprites-sandbox.md#give-the-agent-a-desktop) | A Fly.io Sprite |
| [`BoxdSandbox(desktop=True)`](boxd-sandbox.md#give-the-agent-a-desktop) | A boxd microVM |
| [`BoxLiteSandbox(desktop=True)`](boxlite-sandbox.md#give-the-agent-a-desktop) | A BoxLite micro-VM on your machine |
| [`SmolSandbox(desktop=True)`](smol-sandbox.md#give-the-agent-a-desktop) | A smolvm microVM on your machine |

To give your own sandbox a screen, see [Giving a workspace a screen](#giving-a-workspace-a-screen).

On a [read-only](../workspace.md#read-only-access) workspace the agent can take screenshots but not act.

## Models

The model must accept images.

- Claude 4 and later get Anthropic's own computer use tool; see [Anthropic](../models/anthropic.md#computer-use).
- GPT-5.4 and later, except GPT-5.4 nano, get OpenAI's on the Responses API; see
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
| `action_limits` | no limits | The most one action may ask for, as a [`ComputerUseActionLimits`][pydantic_ai_harness.computer_use.ComputerUseActionLimits]: `max_wait_seconds`, `max_hold_seconds`, `max_typed_chars`, `max_key_repeat`, `max_scroll_amount` and `max_drag_points`. A call over a limit is refused before any of its actions run, and the model is told which limit it went over. |
| `native` | `True` | Use the model's own computer use tool where it has one, by adding a [`ComputerTool`][pydantic_ai.native_tools.ComputerTool] that this tool runs. `False` gives every model this tool and all its actions; on GPT that adds zoom, triple-click, holding keys, pressing or releasing a mouse button, and reading the pointer position. |

`ComputerUse` turns on [prompt caching](../capabilities/caching.md), since every request resends the screenshots. Set
`cache=False` in the model settings to turn it off.

## Approving calls

`ComputerUse` runs every call without asking, except one OpenAI flagged.

OpenAI can attach safety checks to a call, such as `malicious_instructions` when the screen seems to carry instructions
that could hijack the model. OpenAI's guidance is that a person confirms every flagged call before it runs, so a flagged
call waits for approval like any [deferred tool](../deferred-tools.md) call. Its `ComputerCallPart` carries the checks in
`safety_checks`, each a [`SafetyCheck`][pydantic_ai.computer_use.SafetyCheck] with OpenAI's `code` and `message`. Once the call is approved and has
run, the OpenAI model tells OpenAI the checks were acknowledged. Give the agent a deferred tool handler, or add
`DeferredToolRequests` to its output types to ask a person; without either, a flagged call ends the run with an error.

To decide in code instead, resolve flagged calls with [`HandleDeferredToolCalls`](../deferred-tools.md#resolving-deferred-calls-with-a-handler):

```python {title="computer_use_safety_checks.py" test="skip" lint="skip"}
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext, ToolDenied
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.computer_use import ComputerCallPart
from pydantic_ai_harness import ComputerUse


async def review(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
    approvals: dict[str, bool | ToolDenied] = {}
    for call in requests.approvals:
        if isinstance(call, ComputerCallPart) and call.safety_checks:
            if all(check.code == 'irrelevant_domain' for check in call.safety_checks):
                approvals[call.tool_call_id] = True
            else:
                approvals[call.tool_call_id] = ToolDenied('This page needs a person to check it first.')
    return requests.build_results(approvals=approvals)


agent = Agent(
    '{{ openai:medium }}',
    capabilities=[my_desktop_sandbox, ComputerUse(), HandleDeferredToolCalls(handler=review)],
)
```

To ask before every call that does more than look, raise `ApprovalRequired` from a
[`before_tool_execute` hook](../hooks.md):

```python {title="computer_use_ask_before_acting.py" test="skip" lint="skip"}
from pydantic_ai import ApprovalRequired
from pydantic_ai.capabilities import Hooks
from pydantic_ai.computer_use import ComputerCallPart

hooks = Hooks()


@hooks.on.before_tool_execute
async def ask_before_acting(ctx, *, call, tool_def, args):
    looking = {'screenshot', 'zoom', 'wait', 'cursor_position'}
    if isinstance(call, ComputerCallPart) and not ctx.tool_call_approved:
        if any(action.type not in looking for action in call.actions or []):
            raise ApprovalRequired
    return args
```

## Screen size

Some models only take screenshots up to a size, set by
[`computer_use_screenshot_max_side`][pydantic_ai.profiles.ModelProfile.computer_use_screenshot_max_side] and
`computer_use_screenshot_max_pixels` in their profile:

- Claude Opus 4.7, Opus 4.8 and the Claude 5 models: 2576 pixels on the longer side and 3.6 megapixels.
- Earlier Claude models: 1568 pixels and about 1.15 megapixels.
- GPT: 2560x1440, or 1920x1200 for GPT-5.5. It clicks less accurately on a larger screen.

A 1280x800 screen fits every model. `ComputerUse` sends screenshots at the screen's size, so on a screen larger
than the model takes, the run stops with an error naming the largest size that fits.

## When something goes wrong

- Actions run in order. If one fails, the rest don't run, and the result says which failed and why.
- A position outside the screen fails the whole call before anything runs.
- If the environment is gone, the run ends with [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError].
- Agents on one screen and one event loop, such as parallel sub-agents, take turns a whole call at a time, so their clicks never interleave. Agents on separate event loops or in separate processes don't wait for each other.
- A call releases every mouse button it presses with `mouse_down`, so the next call, perhaps another agent's, starts with none pressed. A call that doesn't is refused before any action runs, and a button pressed by a call that fails is released.

## Watching a run

A cloud desktop's `live_view_url()` gives a link to watch the run live, or take over; see
[Daytona](daytona-sandbox.md#give-the-agent-a-desktop) and [E2B](e2b-sandbox.md#give-the-agent-a-desktop).

Each screenshot is in a `FunctionToolResultEvent` for the `computer` tool, so an
[event stream handler](../agent.md#streaming-all-events) can save or show them as the run goes.

Each finished call also emits [capability events](../capabilities/overview.md#capability-events) in
the `computer_use` namespace, read with `@agent.on_event` or `run_stream_events()`:

| Event | Payload |
|---|---|
| `ActionsPerformedEvent` | `result`, the call's `ComputerResult`, and `tool_call_id` |

They are emitted once per call, also under durable execution. The same facts stay in the history: the `computer`
tool's results are `ComputerReturnPart`s, whose `result` is the call's `ComputerResult`, and a call OpenAI flagged
is a `ComputerCallPart` whose `safety_checks` say what the person approving it should check. The events add
no telemetry spans; core already traces the tool call.

## Limitations

- Screenshots are the main cost: about 1,300 input tokens each at 1280x800, resent with every request. Prompt caching
  cuts the resends to about a tenth.
- Every request carries every earlier screenshot, so a long run reaches the provider's limits: Claude refuses a
  request with more than 100 images on its 200k-context models, and Gemini one over 20 MB. `ComputerUse` doesn't drop
  old screenshots, and [`ClearToolResults`](compaction.md) leaves `computer` results alone.
- Each call is one model request, so long tasks can hit the default [`request_limit`](../agent.md#usage-limits) of 50.
  Pass higher [`UsageLimits`][pydantic_ai.usage.UsageLimits].
- Under [durable execution](../durable_execution/overview.md), each call's work on the screen is one durable step, so a finished call isn't
  run again on replay or recovery. Like any tool with side effects, a call cut off before its step was recorded can
  run again under the engine's retry settings for capability steps. On Temporal, those are `TemporalDurability`'s
  `activity_config`, whose default 60-second timeout and unlimited retries apply to every call: raise
  `start_to_close_timeout` above your longest call, and set `retry_policy=RetryPolicy(maximum_attempts=1)` so a call
  isn't repeated. When the engine gives up, the model is told some actions may have run. Use `TemporalDurability`,
  `PrefectDurability` or `DBOSDurability`; the deprecated `TemporalAgent`, `PrefectAgent` and `DBOSAgent` can't run it.
- On Temporal, screenshots pass the 2 MB payload limit after a few steps. Configure
  [external storage](../durable_execution/temporal.md#large-payloads).
- If the display restarts, the model isn't told; its next screenshot shows the new screen.
- [Instrumentation](../logfire.md#excluding-binary-content) records every screenshot. Pass
  `InstrumentationSettings(include_binary_content=False)` to leave them out.
- [`Advisor(forward_history=True)`](advisor.md) forwards every screenshot; Anthropic refuses one past 600 images.
- With [`ModelFallback`](../capabilities/model-fallback.md), a model that answers instead is told its own coordinates,
  pixels or a grid, and screenshots are sized to fit it too. One that can't take images, or would see the latest
  screenshot scaled down, ends the run with an error before its request. In a
  [`FallbackModel`](../models/overview.md#fallback-model), Gemini 3 gets its grid only if every model in it does.

## Giving a workspace a screen

`ComputerUse` drives the display of the run's workspace backend: a backend with a screen implements
[`SupportsDisplay`][pydantic_ai_harness.computer_use.SupportsDisplay], a `display` property that returns a
[`Display`][pydantic_ai_harness.computer_use.Display], or `None` when there's no screen. It must not do I/O, and must
return the same object every time.

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

::: pydantic_ai_harness.computer_use.ComputerUseActionLimits

::: pydantic_ai_harness.computer_use.ActionsPerformedEvent

::: pydantic_ai_harness.computer_use.Display

::: pydantic_ai_harness.computer_use.SupportsDisplay

::: pydantic_ai_harness.computer_use.SupportsZoom

::: pydantic_ai_harness.computer_use.SupportsCursorPosition

::: pydantic_ai_harness.computer_use.DisplayAction

::: pydantic_ai_harness.computer_use.UnsupportedActionError
