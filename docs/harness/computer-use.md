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

Use your model provider's extra instead of `anthropic`. The example needs Docker. The first run builds a desktop
image, which takes a few minutes.

## Usage

```python {title="computer_use.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse, DockerContainer

agent = Agent('{{ anthropic:medium }}', capabilities=[DockerContainer(desktop=True), ComputerUse()])


async def main():
    result = await agent.run('Open Firefox and find the current Python release on python.org.')
    print(result.output)
```

The container keeps running after the run. To remove it, see [Clean up](docker-container.md#clean-up). If the run fails, remove it
in an `on_run_error` hook, as shown for [E2B](e2b-sandbox.md#clean-up).

### Where the screen comes from

| Capability | Screen |
|---|---|
| [`DockerContainer(desktop=True)`](docker-container.md#give-the-agent-a-desktop) | A container on your machine |
| [`DaytonaSandbox(desktop=True)`](daytona-sandbox.md#give-the-agent-a-desktop) | A Daytona cloud sandbox |

To give your own sandbox a screen, see [Giving a workspace a screen](#giving-a-workspace-a-screen).

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

The model has to accept images to see the screen. A model that can't gets a note in place of each screenshot, saying it wasn't sent.

- Claude models on `AnthropicModel` get Anthropic's own computer use tool on the clients and models listed under
  [Anthropic](../models/anthropic.md#computer-use). Other Claude models and clients get the function tool.
- GPT-5.4 (except GPT-5.4 nano), GPT-5.5, GPT-5.6 and GPT-6 get OpenAI's own computer tool when you use them
  through `OpenAIResponsesModel` with `OpenAIProvider` or `BedrockMantleProvider`. See [OpenAI](../models/openai.md#native-tools).
- Every other model gets `computer` as a regular function tool.

An Azure client gets the function tool too. The Pydantic AI Gateway's OpenAI route uses `OpenAIProvider`, so it gets
OpenAI's tool. Pydantic AI checks the provider class, not the server it calls. If you point
`OpenAIProvider(base_url=...)` at a server that doesn't support the computer tool, such as Ollama or vLLM, remove it in
the model's `profile=` argument:
`profile=lambda p: {**p, 'supported_native_tools': p['supported_native_tools'] - {ComputerTool}}`. To send it through
another provider whose server supports it, use `| {ComputerTool}` instead. Start from the model's profile like this,
since a new `supported_native_tools` set replaces all of the model's native tools. Setting it in the provider's profile
has no effect.

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

Some providers flag an action for a person to confirm, and give their reason in the call's
[`confirmation`][pydantic_ai.computer_use.ComputerCallPart.confirmation]. `ComputerUse` asks for approval of
such a call without a hook: the tool raises `ApprovalRequired` before it does anything, so no hook can skip it,
with the reason in the request's `metadata` as `{'reason': ...}`. A hook like the one above runs first and asks
without the reason. Once you approve the call, its result has `confirmed` set, so the provider is told a person
approved it. A flagged call fails the run unless the agent has `DeferredToolRequests` in its `output_type` or a
`HandleDeferredToolCalls` handler.

## Screen size

A 1280x800 screen suits every model, and `ComputerUse` sends each screenshot at the screen's full size. A larger
screen shows more at once, but each screenshot costs more tokens, and every request sends every earlier screenshot
again: a 1920x1080 screenshot costs about twice as much as a 1280x800 one. Below about 1024x768, models start to miss
detail.

The providers' own advice:

- Anthropic: "For general desktop tasks, use 1024x768 or 1280x720; for web applications, use 1280x800 or 1366x768"
  ([computer use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)). For each
  model's image size limits, see [Vision](https://platform.claude.com/docs/en/build-with-claude/vision#evaluate-image-size).
- OpenAI: 1440x900 and 1600x900 work well
  ([computer use](https://developers.openai.com/api/docs/guides/tools-computer-use-integration#capture-screenshots)).
  For each model's image size limits, see
  [Images and vision](https://developers.openai.com/api/docs/guides/images-vision#model-sizing-behavior).

On `AnthropicModel`, what happens to a screenshot too large for Claude depends on the tool it goes through
([computer use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)):

- Claude's `computer_toolset_20260801` tool refuses a screenshot over 2576 pixels on its longer side or over 4784 image
  tokens, where one token covers 28x28 pixels. A 2549x1457 screen is refused. The request fails with Anthropic's 400
  as a `ModelHTTPError`, which `FallbackModel` and the `ModelFallback` capability move on from.
- The older computer use tools, such as Claude Haiku 4.5's `computer_20250124`, and `ComputerUse(native=False)` on any
  Claude model, shrink a larger screenshot without saying so. Claude then gives positions on the smaller image, so its
  clicks miss: Haiku 4.5 at 1920x1080 clicked at three quarters of the right position, and Claude Sonnet 5.5 through
  the function tool at 2560x1600 missed a button by about 100 pixels. Keep the screen at a size that fits: Haiku 4.5
  clicked on target at 1280x800, and Sonnet 5.5 through the function tool at 1920x1200.

GPT-5.5 clicks accurately on screens up to 1920x1200, and GPT-5.4 on screens up to 2048x1280. On a larger screen,
they can miss what they click.

## When something goes wrong

- A call's actions run in order. If one fails, the actions after it don't run, and the result tells the model which
  action failed and why.
- If any action in a call has a position outside the screen, none of the call's actions run, and the model is asked to
  try again. Each such retry counts against the tool's retry limit.
- If the workspace is gone, for example because its sandbox was deleted, the run ends with [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError].
- Several agents can share one screen, such as sub-agents running in parallel. If they run in the same `asyncio` event loop, they take turns one whole call at a time, so their clicks never mix. Agents in separate event loops or separate processes don't wait for each other.
- A mouse button pressed with `mouse_down` stays pressed across calls until a `mouse_up`, and a key pressed with `key_down` until a `key_up`. Each result tells the model which buttons and keys are still pressed. `ComputerUse` releases them when a call fails or is cancelled while it acts on the screen, and when the run ends, so a drag must finish within one run, approvals included. If the model loads `ComputerUse` during the run, because of `defer_loading=True`, they are released at the end of every call, so a drag must happen within one call. If a release after a failed call fails too, the result says the button or key may still be pressed.

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

## Durable execution

Under [durable execution](../durable_execution/overview.md), everything one call does on the screen is one durable
step. The engine records each finished step, so replaying or recovering the run doesn't repeat a finished call. A call
cut off before its step was recorded can run again, as often as the engine's retry settings for capability steps
allow. When the engine stops retrying, the model is told that some actions may have run.

On Temporal, those settings are `TemporalDurability`'s `activity_config`. By default it has a 60-second timeout and
unlimited retries. Set `start_to_close_timeout` longer than your longest call. Set
`retry_policy=RetryPolicy(maximum_attempts=1)` so a call is never run twice. That also stops retries of model requests
and other tools, so give model requests theirs back with
`model_activity_config=ActivityConfig(retry_policy=RetryPolicy())`, and a toolset its own with
`toolset_activity_config`. After a few steps, the screenshots also pass Temporal's 2 MB payload limit, so configure
[external storage](../durable_execution/temporal.md#large-payloads).

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
