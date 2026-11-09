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

Use your model provider's extra instead of `anthropic`.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse

agent = Agent('anthropic:claude-sonnet-5-5', capabilities=[my_desktop_sandbox, ComputerUse()])  # (1)!


async def main():
    result = await agent.run('Open the text editor and write a haiku about sandboxes.')
    print(result.output)
```

1. Any capability whose workspace has a display, meaning its backend implements
   [`SupportsDisplay`](https://pydantic.dev/docs/ai/api/pydantic-ai/workspaces/#pydantic_ai.workspaces.SupportsDisplay). See
   [Controlling a display](https://pydantic.dev/docs/ai/core-concepts/workspace/#controlling-a-display).

On a [read-only](https://pydantic.dev/docs/ai/core-concepts/workspace/#read-only-access) workspace the agent can take screenshots but not act.

## Models

The model must accept images.

Every model gets `computer` as a regular function tool.

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
| `keep_screenshots` | `None` | Send only the newest N screenshots, to cap the cost of long runs. Older ones are dropped 25 at a time so prompt caching keeps working. Don't use it on Claude Fable 5.1 or the Claude 5.5 models: they lose their earlier reasoning. |
| `settle_seconds` | `0.5` | Wait this long after acting before the screenshot, so pages can load. |
| `native` | `True` | Use the model's own computer use tool where it has one, by adding a `ComputerTool` that this tool runs. `False` gives every model this tool and all its actions; on GPT that adds zoom, triple-click, holding keys, pressing or releasing a mouse button, and reading the pointer position. |

OpenAI can flag a call with safety checks. A flagged call always needs approval, so give the agent a deferred tool
handler such as [`HandleDeferredToolCalls`](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/); without one, the run ends with an error.

`ComputerUse` turns on [prompt caching](https://pydantic.dev/docs/ai/capabilities/caching/), since every request resends the screenshots. Set
`cache=False` in the model settings to turn it off.

## Screen size

Some models only take screenshots up to a size, set by `computer_use_screenshot_limit` in their profile:

A 1280x800 screen fits every model. On a larger screen, `ComputerUse` sends a scaled-down screenshot and scales the
model's positions back up, if the display can scale screenshots
([`SupportsScaledScreenshot`](https://pydantic.dev/docs/ai/api/pydantic-ai/workspaces/#pydantic_ai.workspaces.SupportsScaledScreenshot)). If it can't, the run stops with an
error naming the largest size that fits.

## When something goes wrong

- Actions run in order. If one fails, the rest don't run, and the result says which failed and why.
- A position outside the screen fails the whole call before anything runs.
- If the environment is gone, the run ends with [`WorkspaceUnavailableError`](https://pydantic.dev/docs/ai/api/pydantic-ai/workspaces/#pydantic_ai.workspaces.WorkspaceUnavailableError).
- Agents in one process on one screen take turns, a whole call at a time, so their clicks never interleave.

## Watching a run

Each screenshot is in a `FunctionToolResultEvent` for the `computer` tool, so an
[event stream handler](https://pydantic.dev/docs/ai/core-concepts/agent/#streaming-all-events) can save or show them as the run goes.

Each finished call also emits [capability events](https://pydantic.dev/docs/ai/capabilities/overview/#capability-events) in
the `computer_use` namespace, read with `@agent.on_event` or `run_stream_events()`:

| Event | Payload |
|---|---|
| `ActionsPerformedEvent` | `performed`, `error`, `screenshot_size` |
| `DisplayRestartedEvent` | none: the display restarted during the call, so what was on screen may be gone |

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
- If the display restarts, the model isn't told; its next screenshot shows the new screen. The app gets a
  `DisplayRestartedEvent` from displays that report restarts.
- [Instrumentation](https://pydantic.dev/docs/ai/integrations/logfire/#excluding-binary-content) records every screenshot. Pass
  `InstrumentationSettings(include_binary_content=False)` to leave them out.
- [`Advisor(forward_history=True)`](../advisor/) forwards every screenshot; Anthropic refuses one past 600 images.
- In a [`FallbackModel`](https://pydantic.dev/docs/ai/models/overview/#fallback-model), Gemini 3 gets its grid only if every model in it does.

## API reference

::: pydantic_ai_harness.computer_use.ComputerUse
