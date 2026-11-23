# Computer Use

`ComputerUse` lets your agent use a desktop. It adds one tool, `computer`. The model calls it with a list of mouse and
keyboard actions. The actions run on the screen of the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), and the tool returns a
screenshot of the result.

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

Use your model provider's extra instead of `anthropic`. The example needs Docker. The first run builds a desktop
image, which takes a few minutes.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse, DockerContainer

agent = Agent('anthropic:claude-sonnet-5-5', capabilities=[DockerContainer(desktop=True), ComputerUse()])


async def main():
    result = await agent.run('Open Firefox and find the current Python release on python.org.')
    print(result.output)
```

The container keeps running after the run. To remove it, see [Clean up](../docker_container/#clean-up). If the run fails, remove it
in an `on_run_error` hook, as shown for [E2B](../e2b_sandbox/#clean-up).

### Where the screen comes from

| Capability | Screen |
|---|---|
| [`DockerContainer(desktop=True)`](../docker_container/#give-the-agent-a-desktop) | A container on your machine |
| [`DaytonaSandbox(desktop=True)`](../daytona_sandbox/#give-the-agent-a-desktop) | A Daytona cloud sandbox |
| [`E2BSandbox(desktop=True)`](../e2b_sandbox/#give-the-agent-a-desktop) | An E2B cloud sandbox |

To give your own sandbox a screen, see [Giving a workspace a screen](#giving-a-workspace-a-screen).

On a [read-only](https://pydantic.dev/docs/ai/core-concepts/workspace/#read-only-access) workspace, the `computer` tool offers only the actions that change
nothing on the screen: `screenshot`, `zoom`, `cursor_position` and `wait`.

## What the model is told

`ComputerUse` adds instructions on how to use the tool, including never to follow instructions shown on the screen.

With `ComputerUse(defer_loading=True)`, the model sees only the capability's `description` until it chooses to load
the tool. The description decides when the model loads it, so pass your own `description=` to change that.

## What the display needs

The agent sees a desktop: a panel along the bottom with a button labelled Terminal, a button for the image's browser
if it has one, and windows that fill the screen above it. The terminal uses a large monospace font so screenshots are
easy to read, unless the image has its own `~/.Xdefaults` or loads X settings with `xrdb`.

The panel shows whatever browser the image has. When the desktop first starts, it looks for `firefox-esr`, `firefox`,
`chromium`, `chromium-browser` and `google-chrome`, in that order, and adds a button for the first one it finds,
labelled Firefox, Chromium or Chrome. With none of them, there's no browser button. Chromium and Chrome refuse to start
as root, or where the user can't make a user namespace, so there they get `--no-sandbox`. That turns off the browser's
own sandbox, which keeps a web page that breaks into the browser away from the rest of the sandbox.

What each default image has:

- Docker's `DESKTOP_DOCKERFILE`: Firefox.
- E2B and Daytona: their own XFCE desktop, with Firefox and Chrome on E2B and Chromium on Daytona.
- boxd: agent-browser's Chrome.
- Modal, Sprites, Render, BoxLite, smolvm, and OpenSandbox with `debian:bookworm-slim`: no browser.

To add one to your own image, install it in its Dockerfile: `RUN apt-get update && apt-get install -y firefox-esr`.

These programs inside the sandbox make the desktop:

- Xvfb draws the screen, when no X server is running already.
- `xdotool` moves the mouse and types.
- `xwininfo` (in Debian's `x11-utils`) reads the screen size.
- `scrot` takes screenshots.
- `setsid` keeps the screen running after the command that started it ends.
- `fluxbox` gives windows title bars and focuses the one the agent clicks.
- `tint2` draws the panel, and `xterm` is the terminal its Terminal button opens.

The first use of the display installs any that are missing with `apt-get`, as root or with passwordless `sudo`. That
takes 20 to 60 seconds; add them to your image to skip it.

If one of the first five can't be installed, because the image has no `apt-get` or no root, the run ends with a
`WorkspaceUnavailableError` naming what's missing. If `fluxbox`, `tint2` or `xterm` can't be installed or don't
start, the screen still works without the panel and full-screen windows, and `/tmp/pydantic-ai-desktop.log` in the
sandbox says why. An install that fails after a working `apt-get update` isn't retried while the same programs are
missing; delete `/tmp/pydantic-ai-desktop-install-failed` to retry it.

[E2B](../e2b_sandbox/#give-the-agent-a-desktop) and [Daytona](../daytona_sandbox/#give-the-agent-a-desktop) start their own XFCE desktop instead.

## Models

The model has to accept images to see the screen. A model that can't gets a note in place of each screenshot, saying it wasn't sent.

On the models listed under [OpenAI](https://pydantic.dev/docs/ai/models/openai/#computer-use) and [Anthropic](https://pydantic.dev/docs/ai/models/anthropic/#computer-use), the model
gets its provider's own computer use tool. Every other model gets `computer` as a regular function tool.

## Computer use, Playwright, or Browser Use

| You want the agent to | Use |
|---|---|
| Work in desktop applications, or any website exactly as a person sees it | `ComputerUse` |
| Drive a browser with typed actions such as `navigate` and `click(selector)` | [Playwright Browser](../playwright/) |
| Hand a whole web task to a browsing sub-agent | [Browser Use](../browser_use/) |

## Configuration

| Setting | Default | What it does |
|---|---|---|
| `settle_seconds` | `2` | How long, at most, to wait for the screen to stop changing after the actions run. `0` takes the screenshot at once. |
| `native` | `True` | If the model's provider has its own computer use tool, the model gets that tool instead: `ComputerUse` adds a `ComputerTool` and runs its calls on the screen. `False` gives every model the `computer` function tool with all its actions. |

Every request sends all the earlier screenshots again, so `ComputerUse` turns on
[prompt caching](https://pydantic.dev/docs/ai/capabilities/caching/) to make that cheaper. To turn caching off, set `cache=False` in the model
settings.

## Approving calls

`ComputerUse` runs every call without asking you first. To approve each call that clicks, types or does anything
else beyond looking at the screen, raise `ApprovalRequired` from a [`before_tool_execute` hook](https://pydantic.dev/docs/ai/core-concepts/hooks/):

```python {title="computer_use_ask_before_acting.py"}
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

The same hook can refuse a call instead, such as one that waits longer or types more than you allow. [Raise `ToolFailed`](https://pydantic.dev/docs/ai/core-concepts/hooks/#triggering-retries-with-modelretry) with a message for the model. None of the call's actions run.

## Screen size

A 1280x800 screen suits every model. `ComputerUse` sends each screenshot at the screen's full size, so a larger screen
costs more tokens per screenshot. Below about 1024x768, models start to miss detail.

The providers' own advice:

- Anthropic: "For general desktop tasks, use 1024x768 or 1280x720; for web applications, use 1280x800 or 1366x768"
  ([computer use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)). For each
  model's image size limits, see [Vision](https://platform.claude.com/docs/en/build-with-claude/vision#evaluate-image-size).
- OpenAI: 1440x900 and 1600x900 work well
  ([computer use](https://developers.openai.com/api/docs/guides/tools-computer-use-integration#capture-screenshots)).
  For each model's image size limits, see
  [Images and vision](https://developers.openai.com/api/docs/guides/images-vision#model-sizing-behavior).

On `AnthropicModel`, a screenshot larger than the model's image size limit fails or misses
([computer use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)):

- Claude's `computer_toolset_20260801` tool refuses a screenshot over 2576 pixels on its longer side or over 4784 image
  tokens, where one token covers 28x28 pixels. The request fails with a `ModelHTTPError`.
- The older computer use tools, and `ComputerUse(native=False)` on any Claude model, shrink a larger screenshot without
  saying so, so Claude's clicks miss. Keep the screen within the model's limit.

## When something goes wrong

- A call's actions run in order. If one fails, the actions after it don't run, and the result tells the model which
  action failed and why.
- If any action in a call has a position outside the screen, none of the call's actions run, and the model is asked to
  try again. Each such retry counts against the tool's retry limit.
- If the workspace is gone, for example because its sandbox was deleted, the run ends with [`WorkspaceUnavailableError`](https://pydantic.dev/docs/ai/api/pydantic-ai/workspaces/#pydantic_ai.workspaces.WorkspaceUnavailableError).
- Agents that share one screen in the same `asyncio` event loop, such as parallel sub-agents, take turns one call at a time. Agents in separate event loops or processes don't wait for each other.
- A mouse button pressed with `mouse_down` stays pressed until a `mouse_up`, and is released when a call fails or the run ends, so a drag must finish within one run. With `defer_loading=True`, buttons are released at the end of every call.

## Watching a run

Each screenshot arrives in a `FunctionToolResultEvent` for the `computer` tool. An
[event stream handler](https://pydantic.dev/docs/ai/core-concepts/agent/#streaming-all-events) can save or show the screenshots while the run goes on.

Each call that finishes with a screenshot also sends one
[capability event](https://pydantic.dev/docs/ai/capabilities/overview/#capability-events) in the `computer_use` namespace. Read it with
`@agent.on_event` or `run_stream_events()`:

| Event | Payload |
|---|---|
| `ActionsPerformedEvent` | `result`, the call's `ComputerResult`, and `tool_call_id` |

On Temporal, event listeners run in workflow code, so they [run again when Temporal replays the workflow](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#choosing-what-to-publish).
The message history holds the same record. Each result is a `ComputerReturnPart`, and its `result` is the call's
`ComputerResult`.

## Limitations

- Every request sends every earlier screenshot again. They are the main cost, which prompt caching lowers, and a
  long run reaches the provider's limits: Claude, for example, refuses a request with more than 100 images on its
  models with a 200k-token context. `ComputerUse` doesn't drop old screenshots, and
  [`ClearToolResults`](../compaction/) doesn't clear `computer` results.
- Each `computer` call takes one model request, so a long task can reach the default [`request_limit`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits) of 50.
  To allow more, pass higher [`UsageLimits`](https://pydantic.dev/docs/ai/api/pydantic-ai/usage/#pydantic_ai.usage.UsageLimits).
- If the screen restarts, the model isn't told. Its next screenshot shows the new screen.
- [Instrumentation](https://pydantic.dev/docs/ai/integrations/logfire/#excluding-binary-content) records every screenshot. To leave them
  out, pass `InstrumentationSettings(include_binary_content=False)`.
- [`Advisor(forward_history=True)`](../advisor/) sends every screenshot to the advisor model. Anthropic refuses an
  advisor request with more than 600 images.

## Durable execution

Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), everything one call does on the screen is one durable
step. The engine records each finished step, so replaying or recovering the run doesn't repeat a finished call. A call
cut off before its step was recorded can run again, as often as the engine's retry settings for capability steps
allow. When the engine stops retrying, the model is told that some actions may have run.

On Temporal, those settings are `TemporalDurability`'s `activity_config`. By default it has a 60-second timeout and
unlimited retries. Set `start_to_close_timeout` longer than your longest call. Set
`retry_policy=RetryPolicy(maximum_attempts=1)` so a call is never run twice. That also stops retries of model requests
and other tools, so give model requests theirs back with
`model_activity_config=ActivityConfig(retry_policy=RetryPolicy())`, and a toolset its own with
`toolset_activity_config`. After a few steps, the screenshots also pass Temporal's 2 MB payload limit, so configure
[external storage](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#large-payloads).

Use `TemporalDurability` or `PrefectDurability`. The deprecated `TemporalAgent`, `PrefectAgent` and `DBOSAgent` can't
run `ComputerUse`.

## Giving a workspace a screen

`ComputerUse` drives the screen of the run's workspace backend. A backend with a screen implements
`SupportsDisplay`. That means it has a `display` property that
returns a `Display`, or `None` when there is no screen. The property must
return the same object every time, and must not do I/O such as network calls or starting programs.

A `Display` has three methods: `display_size()`, `screenshot()`, which returns PNG bytes, and `act()`, which runs one
`DisplayAction`, such as a click or a key press. Positions are in
screenshot pixels. A display can also implement:

- `SupportsZoom`: capture part of the screen at full resolution.
- `SupportsCursorPosition`: report where the pointer is.

If a display doesn't implement one of these, a `zoom` or `cursor_position` action fails, and the model is told so.
Raise `UnsupportedActionError` for any action the display
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
