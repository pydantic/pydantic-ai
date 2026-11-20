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
| [`ModalSandbox(desktop=True)`](../modal_sandbox/#give-the-agent-a-desktop) | A Modal cloud sandbox |
| [`SpritesSandbox(desktop=True)`](../sprites_sandbox/#give-the-agent-a-desktop) | A Fly.io Sprite |
| [`BoxLiteSandbox(desktop=True)`](../boxlite_sandbox/#give-the-agent-a-desktop) | A BoxLite micro-VM on your machine |

To give your own sandbox a screen, see [Giving a workspace a screen](#giving-a-workspace-a-screen).

On a [read-only](https://pydantic.dev/docs/ai/core-concepts/workspace/#read-only-access) workspace, the `computer` tool offers only the actions that change
nothing on the screen: `screenshot`, `zoom`, `cursor_position` and `wait`. Some models use their provider's own
computer tool instead, and its list of actions can't be narrowed. On a read-only workspace, a call from that tool that
clicks or types is refused, and none of its actions run.

## What the model is told

`ComputerUse` adds instructions that tell the model how to group actions into calls. They also tell it never to
follow instructions shown on the screen. If the agent also has a tool that runs shell commands, such as `Shell`'s, the
model is told two more things: a program it starts from the shell opens on the screen, and it should use the screen
only for work that needs it.

With `ComputerUse(defer_loading=True)`, the model sees only the capability's `description` until it chooses to load
the tool. The description decides when the model loads it, so pass your own `description=` to change that.

## What the display needs

The agent sees a desktop: a panel along the bottom with a button that opens a terminal. By default, the terminal
uses a large monospace font, so its text is easier to read in screenshots. If the image already has a
`~/.Xdefaults` file, or loads X settings with `xrdb`, the terminal uses those instead. Each new window fills the
screen above the panel. There's no web browser unless the image has one, or the agent installs one with
[`Coder`](../coder/).

Eight programs inside the sandbox make this desktop:

- Xvfb draws the screen. It's needed only when no X server is running already.
- `xdotool` moves the mouse and types.
- `xwininfo` (in Debian's `x11-utils`) reads the screen size.
- `scrot` takes screenshots.
- `setsid` keeps the screen running after the command that started it ends.
- `fluxbox` gives each window a title bar and focuses the window the agent clicks.
- `tint2` draws the panel, and `xterm` is the terminal its button opens.

The first use of the display installs any of them that's missing with `apt-get`, as root or with passwordless `sudo`.
That takes 20 to 60 seconds; add them to your image to skip it.

If the first five can't be installed, because the image has no `apt-get` or no root, the run ends with a
`WorkspaceUnavailableError` that names what's missing. If `fluxbox`, `tint2` or `xterm` can't be installed or
don't start, the screen still works, but without what they add: the panel, its terminal button, and windows that fill
the screen. A later run, or a restart of the screen after a failure, tries again. When an install or start failed
and the sandbox could write `/tmp/pydantic-ai-desktop.log`, that file says why. To get the desktop for sure, add them
to your image.

[E2B](../e2b_sandbox/#give-the-agent-a-desktop) and [Daytona](../daytona_sandbox/#give-the-agent-a-desktop) start their own desktop,
XFCE, and don't get this one.

## Models

The model must accept images.

- Claude models on `AnthropicModel` get Anthropic's own computer use tool on the clients and models listed under
  [Anthropic](https://pydantic.dev/docs/ai/models/anthropic/#computer-use). Other Claude models and clients get the
  function tool.
- GPT-5.4 (except GPT-5.4 nano), GPT-5.5, GPT-5.6 and GPT-6 get OpenAI's own computer tool when you use them
  through `OpenAIResponsesModel` with `OpenAIProvider` or `BedrockMantleProvider`. See [OpenAI](https://pydantic.dev/docs/ai/models/openai/#native-tools).
- Every other model gets `computer` as a regular function tool.

By default, OpenAI's computer tool is sent only through `OpenAIProvider` and `BedrockMantleProvider`, except with an Azure client.
The Pydantic AI Gateway's OpenAI route uses `OpenAIProvider`, so it gets the tool too. Pydantic AI checks the provider class, not the server it calls. If you point
`OpenAIProvider(base_url=...)` at another server, such as Ollama or vLLM, that doesn't support the computer tool,
remove `ComputerTool` in the model's `profile=` argument, starting from the profile the model already has:
`profile=lambda p: {**p, 'supported_native_tools': p['supported_native_tools'] - {ComputerTool}}`. Every other provider
gets the function tool. If its server supports OpenAI's computer tool, add it the same way, with `| {ComputerTool}`.
Passing a new `supported_native_tools` set instead replaces all of the model's native tools, so web search and the
others would be dropped. Setting it in the provider's profile has no effect.

Gemini 3 gives positions on a grid from 0 to 999 instead of in pixels. `ComputerUse` converts them to pixels.

## Computer use, Playwright, or Browser Use

| You want the agent to | Use |
|---|---|
| Work in desktop applications, or any website exactly as a person sees it | `ComputerUse` |
| Drive a browser with typed actions such as `navigate` and `click(selector)` | [Playwright Browser](../playwright/) |
| Hand a whole web task to a browsing sub-agent | [Browser Use](../browser_use/) |

## Configuration

| Setting | Default | What it does |
|---|---|---|
| `settle_seconds` | `2` | How long, at most, to wait for the screen to stop changing after the actions run. `ComputerUse` takes screenshots until two in a row match. `0` takes the screenshot at once. |
| `native` | `True` | If the model's provider has its own computer use tool, the model gets that tool instead: `ComputerUse` adds a `ComputerTool` and runs its calls on the screen. `False` gives every model the `computer` function tool with all its actions. On GPT, that adds zoom, triple-click, holding keys, pressing or releasing a mouse button, and reading the pointer position. |

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

Some models accept screenshots only up to a set size. Their profile sets it with
`computer_use_screenshot_max_side` and `computer_use_screenshot_max_pixels`:

- Claude Opus 4.8 and the Claude 5 models: 2576 pixels on the longer side and about 3.75 megapixels (2576x1456).
- Claude Opus 4.7: 2576 pixels on the longer side and 3.6 megapixels.
- Claude Opus 4.5 and 4.6, Sonnet 4.6 and Haiku 4.5: 1568 pixels on the longer side and about 1.15 megapixels.

`ComputerUse` sends every screenshot at the screen's full size and doesn't shrink it. If the screen is larger than the
model accepts, the run stops with a `UserError` that names the largest size that fits. A 1280x800 screen fits every
model.

Claude's `computer_toolset_20260801` tool also caps a screenshot at 4784 image tokens, where one token covers 28x28
pixels. It refuses a larger screenshot, such as 2549x1457, even when it fits the limits above, and the run stops with
Anthropic's error.

GPT-5.5 clicks accurately on screens up to 1920x1200, and GPT-5.4 on screens up to 2048x1280. On a larger screen,
they can miss what they click.

## When something goes wrong

- A call's actions run in order. If one fails, the actions after it don't run, and the result tells the model which
  action failed and why.
- If any action in a call has a position outside the screen, none of the call's actions run, and the model is asked to
  try again. Each such retry counts against the tool's retry limit.
- If the workspace is gone, for example because its sandbox was deleted, the run ends with [`WorkspaceUnavailableError`](https://pydantic.dev/docs/ai/api/pydantic-ai/workspaces/#pydantic_ai.workspaces.WorkspaceUnavailableError).
- Several agents can share one screen, such as sub-agents running in parallel. If they run in the same `asyncio` event loop, they take turns one whole call at a time, so their clicks never mix. Agents in separate event loops or separate processes don't wait for each other.
- A mouse button pressed with `mouse_down` stays pressed across calls until a `mouse_up`. Each result tells the model which buttons are still pressed. `ComputerUse` releases them when a call fails or is cancelled while it acts on the screen, and when the run ends. So a drag must finish within one run, including any calls that wait for approval. If the model loads `ComputerUse` during the run, because of `defer_loading=True`, the buttons are released at the end of every call instead, so a drag must happen within one call. If the release after a failed call fails too, that call's result says the button may still be pressed.

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
`ComputerResult`. `ComputerUse` adds no tracing spans of its own, because the tool call already has one.

## Limitations

- Screenshots are the main cost, because every request sends them all again. Prompt caching makes the repeats
  cheaper.
- Every request carries every earlier screenshot, so a long run reaches the provider's limits. For example, Claude
  refuses a request with more than 100 images on its models with a 200k-token context. `ComputerUse` doesn't drop old
  screenshots, and [`ClearToolResults`](../compaction/) doesn't clear `computer` results.
- Each `computer` call takes one model request, so a long task can reach the default [`request_limit`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits) of 50.
  To allow more, pass higher [`UsageLimits`](https://pydantic.dev/docs/ai/api/pydantic-ai/usage/#pydantic_ai.usage.UsageLimits).
- If the screen restarts, the model isn't told. Its next screenshot shows the new screen.
- [Instrumentation](https://pydantic.dev/docs/ai/integrations/logfire/#excluding-binary-content) records every screenshot. To leave them
  out, pass `InstrumentationSettings(include_binary_content=False)`.
- [`Advisor(forward_history=True)`](../advisor/) sends every screenshot to the advisor model. Anthropic refuses an
  advisor request with more than 600 images.
- With [`ModelFallback`](https://pydantic.dev/docs/ai/capabilities/model-fallback/), a model that takes over is told which coordinates it
  uses: pixels, or Gemini 3's grid. If that model can't take images, or would get the latest screenshot scaled down,
  the run ends with a `UserError` before the model is called. In a
  [`FallbackModel`](https://pydantic.dev/docs/ai/models/overview/#fallback-model), Gemini 3 gets its grid only if every model in the
  `FallbackModel` uses the grid.

## Durable execution

Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), everything one call does on the screen is one durable
step. The engine records each finished step, so replaying or recovering the run doesn't repeat a finished call. A call
cut off before its step was recorded can run again, as often as the engine's retry settings for capability steps
allow. When the engine stops retrying, the model is told that some actions may have run.

On Temporal, those settings are `TemporalDurability`'s `activity_config`. By default it has a 60-second timeout and
unlimited retries. Set `start_to_close_timeout` longer than your longest call. Set
`retry_policy=RetryPolicy(maximum_attempts=1)` so a call is never run twice. After a few steps, the screenshots also
pass Temporal's 2 MB payload limit, so configure [external storage](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#large-payloads).

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
