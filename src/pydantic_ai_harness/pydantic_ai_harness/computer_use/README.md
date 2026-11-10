# Computer Use

`ComputerUse` lets your agent use a desktop. It adds one tool, `computer`: the model sends mouse and keyboard actions,
they run on the display of the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), and the tool returns a screenshot.

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

Use your model provider's extra instead of `anthropic`. The example needs Docker; the first run builds a desktop
image, which takes a few minutes.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse, DockerSandbox

agent = Agent('anthropic:claude-sonnet-5-5', capabilities=[DockerSandbox(desktop=True), ComputerUse()])


async def main():
    result = await agent.run('Open Firefox and find the current Python release on python.org.')
    print(result.output)
```

The container keeps running after the run; see [Clean up](../docker_sandbox/#clean-up). If the run fails, remove it
in an `on_run_error` hook, as shown for [E2B](../e2b_sandbox/#clean-up).

### Where the screen comes from

| Capability | Screen |
|---|---|
| `DockerSandbox(desktop=True)` | A container on your machine |
| [`DaytonaSandbox(desktop=True)`](../daytona_sandbox/#give-the-agent-a-desktop) | A Daytona cloud sandbox |
| [`E2BSandbox(desktop=True)`](../e2b_sandbox/#give-the-agent-a-desktop) | An E2B cloud sandbox |
| [`ModalSandbox(desktop=True)`](../modal_sandbox/#give-the-agent-a-desktop) | A Modal cloud sandbox |

To give your own sandbox a screen, see [Giving a workspace a screen](#giving-a-workspace-a-screen).

On a [read-only](https://pydantic.dev/docs/ai/core-concepts/workspace/#read-only-access) workspace the agent can take screenshots but not act.

## Models

The model must accept images.

- Claude 4 and later get Anthropic's own computer use tool; see [Anthropic](https://pydantic.dev/docs/ai/models/anthropic/#computer-use).
- GPT-5.4 and later, except GPT-5.4 nano, get OpenAI's on the Responses API; see
  [OpenAI](https://pydantic.dev/docs/ai/models/openai/#native-tools).
- Every other model gets `computer` as a regular function tool.

Gemini 3 points on a 0-999 grid, which `ComputerUse` converts to pixels.

## Computer use, Playwright, or Browser Use

| You want the agent to | Use |
|---|---|
| Work in desktop applications, or any website exactly as a person sees it | `ComputerUse` |
| Drive a browser with typed actions such as `navigate` and `click(selector)` | [Playwright Browser](../playwright/) |
| Hand a whole web task to a browsing sub-agent | [Browser Use](../browser_use/) |

## Configuration

| Setting | Default | What it does |
|---|---|---|
| `requires_approval` | `False` | Ask before any call that does more than look, zoom or wait. Needs a [deferred tool handler](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/#human-in-the-loop-tool-approval). |
| `keep_screenshots` | `None` | Send at least the newest N images (screenshots and zooms), to cap the cost of long runs. Older ones are dropped 25 at a time so prompt caching keeps working. Don't use it on Claude Fable 5.1 or the Claude 5.5 models: they lose their earlier reasoning. |
| `settle_seconds` | `0.5` | Wait this long after acting before the screenshot, so pages can load. |
| `native` | `True` | Use the model's own computer use tool where it has one, by adding a `ComputerTool` that this tool runs. `False` gives every model this tool and all its actions; on GPT that adds zoom, triple-click, holding keys, pressing or releasing a mouse button, and reading the pointer position. |

OpenAI can flag a call with safety checks. A flagged call always needs approval, so give the agent a deferred tool
handler such as [`HandleDeferredToolCalls`](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/); without one, the run ends with an error.

`ComputerUse` turns on [prompt caching](https://pydantic.dev/docs/ai/capabilities/caching/), since every request resends the screenshots. Set
`cache=False` in the model settings to turn it off.

## Screen size

Some models only take screenshots up to a size, set by `computer_use_screenshot_max_side` and
`computer_use_screenshot_max_pixels` in their profile:

- Claude Opus 4.7, Opus 4.8 and the Claude 5 models: 2576 pixels on the longer side.
- Earlier Claude models: 1568 pixels and about 1.15 megapixels.
- GPT: 2560x1440. It clicks less accurately on a larger screen.

A 1280x800 screen fits every model. On a larger screen, `ComputerUse` sends a scaled-down screenshot and scales the
model's positions back up, if the display can scale screenshots (`SupportsScaledScreenshot`). If it can't, the run
stops with an error naming the largest size that fits.

## When something goes wrong

- Actions run in order. If one fails, the rest don't run, and the result says which failed and why.
- A position outside the screen fails the whole call before anything runs.
- If the environment is gone, the run ends with [`WorkspaceUnavailableError`](https://pydantic.dev/docs/ai/api/pydantic-ai/workspaces/#pydantic_ai.workspaces.WorkspaceUnavailableError).
- Agents on one screen and one event loop, such as parallel sub-agents, take turns a whole call at a time, so their clicks never interleave. Agents on separate event loops or in separate processes don't wait for each other.
- A call releases every mouse button it presses with `mouse_down`, so the next call, perhaps another agent's, starts with none pressed. A call that doesn't is refused before any action runs, and a button pressed by a call that fails is released.

## Watching a run

A cloud desktop's `live_view_url()` gives a link to watch the run live, or take over; see
[Daytona](../daytona_sandbox/#give-the-agent-a-desktop) and [E2B](../e2b_sandbox/#give-the-agent-a-desktop).

Each screenshot is in a `FunctionToolResultEvent` for the `computer` tool, so an
[event stream handler](https://pydantic.dev/docs/ai/core-concepts/agent/#streaming-all-events) can save or show them as the run goes.

Each finished call also emits [capability events](https://pydantic.dev/docs/ai/capabilities/overview/#capability-events) in
the `computer_use` namespace, read with `@agent.on_event` or `run_stream_events()`:

| Event | Payload |
|---|---|
| `ActionsPerformedEvent` | `result`, the call's `ComputerResult`, and `tool_call_id` |

They are emitted once per call, also under durable execution. The same facts stay in the history: the `computer`
tool's results are `ComputerReturnPart`s, whose `result` is the call's `ComputerResult`, and a call a provider
flagged is a `ComputerCallPart` whose `safety_checks` say what the person approving it should check. The events add
no telemetry spans; core already traces the tool call.

## Limitations

- Screenshots are the main cost: about 1,300 input tokens each at 1280x800, resent with every request. Prompt caching
  cuts the resends to about a tenth.
- Each call is one model request, so long tasks can hit the default [`request_limit`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits) of 50.
  Pass higher [`UsageLimits`](https://pydantic.dev/docs/ai/api/pydantic-ai/usage/#pydantic_ai.usage.UsageLimits).
- Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), each call is one durable step that runs at most once,
  for up to 15 minutes. If it's cut off, the model is told some actions may have run. Temporal and Render never
  repeat a call; DBOS, and Prefect when a flow is rerun, can repeat one cut off by a crash; Absurd, pgtask and AWS
  Lambda use their own retry settings. Use `TemporalDurability`, `PrefectDurability` or `DBOSDurability`; the
  deprecated `TemporalAgent`, `PrefectAgent` and `DBOSAgent` can't run it.
- On Temporal, screenshots pass the 2 MB payload limit after a few steps. Configure
  [external storage](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#large-payloads).
- If the display restarts, the model isn't told; its next screenshot shows the new screen.
- [Instrumentation](https://pydantic.dev/docs/ai/integrations/logfire/#excluding-binary-content) records every screenshot. Pass
  `InstrumentationSettings(include_binary_content=False)` to leave them out.
- [`Advisor(forward_history=True)`](../advisor/) forwards every screenshot; Anthropic refuses one past 600 images.
- In a [`FallbackModel`](https://pydantic.dev/docs/ai/models/overview/#fallback-model), Gemini 3 gets its grid only if every model in it does.

## API reference

::: pydantic_ai_harness.computer_use.ComputerUse

## Giving a workspace a screen

`ComputerUse` drives the display of the run's workspace backend: a backend with a screen implements
`SupportsDisplay`, a `display` property that returns a `Display`, or `None` when there's no screen. It must not do
I/O, and must return the same object every time.

A `Display` has three methods: `display_size()`, `screenshot()` (PNG bytes), and `act()`, which runs one
`ComputerAction` such as a click or a key press, in screenshot pixels. A display can also implement:

- `SupportsZoom`: capture part of the screen at full resolution.
- `SupportsScaledScreenshot`: capture the whole screen smaller, for a model with a screenshot size limit.
- `SupportsCursorPosition`: report where the pointer is.

Without one of these, the model is told the action failed. Raise `UnsupportedActionError` for any action the display
can't do exactly, including action types added later. The protocols are structural, so a display needs no imports
from here:

```python {title="my_display.py"}
from pydantic_ai.computer_use import Click
from pydantic_ai_harness.computer_use import ComputerAction, Display, UnsupportedActionError


class MyDisplay:
    async def display_size(self) -> tuple[int, int]:
        return (1280, 800)

    async def screenshot(self) -> bytes:
        raise NotImplementedError  # capture the screen as PNG bytes

    async def act(self, action: ComputerAction) -> None:
        if not isinstance(action, Click):
            raise UnsupportedActionError(f'this display cannot {action.type}')
        # click at (action.x, action.y), or at the pointer when they're `None`


assert isinstance(MyDisplay(), Display)
```
