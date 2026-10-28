# Computer Use

`ComputerUse` lets your agent see and control a graphical desktop. It adds one tool, `computer`: the model sends a
list of mouse and keyboard actions, they run on the display of the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), and the tool
returns a screenshot of the screen afterwards.

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

Swap `anthropic` for your model provider's extra. The example needs Docker running locally; the first run builds a
desktop image, which takes a few minutes.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse, DockerSandbox

agent = Agent('anthropic:claude-sonnet-5-5', capabilities=[DockerSandbox(desktop=True), ComputerUse()])


async def main():
    result = await agent.run('Open Firefox and find the current Python release on python.org.')
    print(result.output)
```

Each run leaves its container running; see [Clean up](../docker_sandbox/#clean-up). A run that fails, such as on a
usage limit, returns no ref to clean up with: remove its desktop in an `on_run_error` hook, as shown for
[E2B](../e2b_sandbox/#clean-up).

For a desktop in a cloud sandbox instead, use [`DaytonaSandbox(desktop=True)`](../daytona_sandbox/#give-the-agent-a-desktop) or [`E2BSandbox(desktop=True)`](../e2b_sandbox/#give-the-agent-a-desktop).

To give your own sandbox a display, implement [`SupportsDisplay`](https://pydantic.dev/docs/ai/api/pydantic-ai/workspaces/#pydantic_ai.workspaces.SupportsDisplay); see
[Controlling a display](https://pydantic.dev/docs/ai/core-concepts/workspace/#controlling-a-display). On a [read-only](https://pydantic.dev/docs/ai/core-concepts/workspace/#read-only-access)
workspace the agent can take screenshots but not act.

The model must accept images. Gemini 3 models point on a 0-999 grid, which `ComputerUse` scales to pixels for them. Gemini gets `computer` as a function tool, not its own computer use tool, until `GoogleModel` moves to the Interactions API ([#6192](https://github.com/pydantic/pydantic-ai/issues/6192)).

Claude 4 and later models and GPT-5.4 and later (except GPT-5.4 nano) get `computer` as their provider's native
computer tool, the format they were trained on, on the APIs listed for [Anthropic](https://pydantic.dev/docs/ai/models/anthropic/#computer-use) and
[OpenAI](https://pydantic.dev/docs/ai/models/openai/); GPT gets it only through the Responses API. Every other model that accepts images gets it as a
function tool.

## Computer use, Playwright, or Browser Use

| You want the agent to | Use |
|---|---|
| Work in desktop applications, or any website exactly as a person sees it | `ComputerUse` |
| Drive a browser with typed actions such as `navigate` and `click(selector)` | [Playwright Browser](../playwright/) |
| Hand a whole web task to a browsing sub-agent | [Browser Use](../browser_use/) |

## Configuration

| Setting | Default | What it does |
|---|---|---|
| `requires_approval` | `False` | `True` asks before any call that does more than look, zoom or wait. Needs a [deferred tool handler](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/#human-in-the-loop-tool-approval). Turn it on for anything you'd regret an agent clicking. |
| `keep_screenshots` | `None` | Send at least the most recent N images, screenshots and zooms, dropping older ones 25 at a time so prompt caching keeps working; the run's history keeps all of them. Use it to bound the cost of long runs, but not on Claude Fable 5.1 or the Claude 5.5 models: dropping a screenshot makes Claude lose its earlier reasoning. With `None`, the oldest are still dropped once a request would carry more than 100 images or about 15 MB of them, which providers refuse. |
| `settle_seconds` | `0.5` | Pause after acting before the next zoom or the screenshot, so animations and page loads can finish. A `wait` the model asks for adds to it. |

A call its provider flags with safety checks, as OpenAI does, needs approval even with `requires_approval=False`.
Give such an agent a deferred tool handler, such as [`HandleDeferredToolCalls`](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/); without one,
a flagged call ends the run with an error. The checks are in the call's `provider_details['safety_checks']`.

The actions to approve are in the model's coordinates: pixels in the latest screenshot, or the 0-999 grid for Gemini 3.
If an approval changes them, the result tells the model the actions that ran.

`ComputerUse` turns on [prompt caching](https://pydantic.dev/docs/ai/capabilities/caching/), since every request carries the screenshots so
far: on Claude, a run costs several times more without it. Set `cache=False` in the model settings to turn it off.

## Screen size

A model's profile can limit the size of the screenshots it takes, as `computer_use_screenshot_limit`: Claude Opus 4.7,
Opus 4.8 and the Claude 5 models take up to 2576 pixels on the longer side, earlier Claude models 1568 pixels and
about 1.15 megapixels, and GPT 2560x1440, since it clicks less accurately on a larger screen. A 1280x800 screen fits
every model. On a larger screen, `ComputerUse` sends the model the screenshot scaled down and scales its positions
back up, when the display can scale its screenshots (`SupportsScaledScreenshot`); otherwise the run stops with an error naming the largest
screen that fits. A zoom into a region larger than the limit fails the call before anything runs, and the result
names the largest zoom of that shape that fits.

## When an action fails

The actions in a call run in order, and one that fails stops the call: the actions after it don't run, and the
result says which action failed and why, ends with a screenshot as usual, and has the error in its `metadata`, a
`ComputerResult`. A position outside the screen fails the whole call before anything runs.
When the environment is gone, the run ends with `WorkspaceUnavailableError`.
When the display restarted since the model's last screenshot, such as after its sandbox was stopped, the next call
runs none of its actions, which would land on whatever took the place of the windows the model saw, and its result
tells the model those windows are gone. Under Temporal, Prefect and Render the tool runs in workflow code, which can't
tell that the display restarted, so the model gets no such notice there.

## Watching a run

To watch the desktop live, or take over, open the link a cloud desktop's `live_view_url()` returns; see
[Daytona](../daytona_sandbox/#give-the-agent-a-desktop) and [E2B](../e2b_sandbox/#give-the-agent-a-desktop).

Every screenshot arrives as a tool result event, so an [event stream handler](https://pydantic.dev/docs/ai/core-concepts/agent/#streaming-all-events) can save
or show each one as the run goes:

```python
from collections.abc import AsyncIterable
from pathlib import Path

from pydantic_ai import AgentStreamEvent, BinaryContent, FunctionToolResultEvent, RunContext


async def save_screenshots(ctx: RunContext, events: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in events:
        if isinstance(event, FunctionToolResultEvent) and event.part.tool_name == 'computer':
            for item in event.part.content_items():
                if isinstance(item, BinaryContent):
                    Path(f'step-{ctx.run_step}.png').write_bytes(item.data)


async def main():
    result = await agent.run('Open the text editor and write a haiku.', event_stream_handler=save_screenshots)
```

## Limitations

- A 1280x800 screenshot costs about 1,200 input tokens on GPT-5.5 and 1,340 on Claude Sonnet 5.5, and every later
  request carries it again; with prompt caching, screenshots already sent cost about a tenth of that.
- Each `computer` call is one model request, so a long task can reach the default
  [`request_limit`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits) of 50. Pass higher [`UsageLimits`](https://pydantic.dev/docs/ai/api/pydantic-ai/usage/#pydantic_ai.usage.UsageLimits) for
  long tasks.
- Under durable execution, use `TemporalDurability`, `PrefectDurability` or `DBOSDurability`: the deprecated
  `TemporalAgent`, `PrefectAgent` and `DBOSAgent` refuse a workspace, so they can't run `ComputerUse`.
- Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), a retried action may repeat a click that already
  happened. Temporal, Prefect and Render retry each action on its own; DBOS, Absurd and pgtask retry the whole call.
- Every screenshot rides in each model request, so under Temporal a run passes the 2 MB payload limit after a few
  steps and then stops making progress. Configure Temporal's [external storage](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#large-payloads),
  which keeps large payloads outside the history. `keep_screenshots` doesn't avoid this: it drops screenshots 25 at a
  time.
- With [instrumentation](https://pydantic.dev/docs/ai/integrations/logfire/#excluding-binary-content), every request's screenshots are recorded on its span; pass
  `InstrumentationSettings(include_binary_content=False)` to leave them out.
- [`Advisor(forward_history=True)`](../advisor/) forwards the run's history with every screenshot, not only the newest
  ones, so each local consultation costs more as the conversation grows, and Anthropic refuses one past 600 images.
- Inside a [`FallbackModel`](https://pydantic.dev/docs/ai/models/overview/#fallback-model) with models that don't all point on Gemini 3's
  grid, Gemini 3 gets pixel coordinates and points less accurately, so put it only in a fallback with other Gemini 3
  models.
