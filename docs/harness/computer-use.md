---
title: Computer Use
description: "Let a Pydantic AI agent see and control a computer's screen, mouse, and keyboard through one batched tool, on this machine or any custom backend."
---

# Computer Use

Let an agent see and control a computer's screen, mouse, and keyboard: click,
type, scroll, and drag in any application, and check a screenshot of the result
after every step.

Reach for it when the work only exists in a GUI: a desktop app with no API or
CLI, a settings pane, or a flow that spans several applications. When a shell,
file, or browser tool can do the job, prefer that tool. It is faster, cheaper,
and less error-prone than working through pixels.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/computer_use/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

The `computer-use` extra installs [mss](https://github.com/BoboTiG/python-mss)
for screenshots, [pynput](https://github.com/moses-palmer/pynput) for the mouse
and keyboard, and Pillow for scaling. You only need it for `LocalComputer`, the
default backend. A custom `Computer` needs nothing beyond the base package.

```bash
pip/uv-add "pydantic-ai-harness[computer-use]"
```

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse

agent = Agent('anthropic:claude-fable-5', capabilities=[ComputerUse()])
result = await agent.run('Open System Settings and turn on dark mode.')
```

The agent gets one tool, `computer`. Each call carries a list of actions that
run in order, and the tool returns a screenshot of the screen afterwards:

| Action | Fields | Does |
|---|---|---|
| `click` | `x`, `y`, `button`, `count`, `modifiers` | Click; `count=2` double-clicks, `modifiers` such as `['shift']` are held during the click |
| `move` | `x`, `y` | Move the pointer, for hover menus and tooltips |
| `drag` | `path` (two or more points) | Hold the left button along the path |
| `scroll` | `x`, `y`, `direction`, `amount` | Scroll `amount` wheel clicks with the pointer over a position |
| `type` | `text` | Type at the keyboard focus |
| `keypress` | `keys` | Press keys together as a chord: `['enter']`, `['ctrl', 'c']` |
| `wait` | `seconds` | Pause up to 10 seconds |
| `screenshot` | | Only look; every call ends with a screenshot anyway |

Batching is how the model saves round trips: it can click a field, type, and
press Enter in one call, then look once. If an action fails, for example on an
unknown key name or a coordinate off the screen, the remaining actions are
skipped and the model gets the error with a screenshot of where things stand.
Calls to `computer` never overlap: the tool is sequential, so parallel tool
calls in one response run one after another.

The screenshot is part of the tool result. Pydantic AI sends it inside the
result where the provider accepts images there, and alongside it where the
provider does not, so any model with image input can use the tool.

## Options

| Field | Default | Meaning |
|---|---|---|
| `computer` | `None` | What the tool drives. `None` uses `LocalComputer()` |
| `require_approval` | `False` | Raise `ApprovalRequired` for any call that does more than `screenshot` and `wait` |
| `environment` | `None` | How the instructions describe the computer, such as `'an Ubuntu 24.04 desktop'`; defaults to the host OS for `LocalComputer` |
| `settle_seconds` | `0.5` | Pause after the last action before the closing screenshot |
| `keep_screenshots` | `3` | How many recent screenshots stay in the conversation; `None` keeps all |

### Screenshots in the context window

Each screenshot costs roughly a thousand or more input tokens, and a long
session takes hundreds of them. Each model request carries only the most recent
`keep_screenshots` screenshots: older ones are replaced with a one-line note in
that request, so the model still sees its earlier actions and their text results
without the images it no longer needs. Screenshots in the request being sent are
always included, even when one response made more parallel `computer` calls than
`keep_screenshots`, because the model has not seen them yet.

Only the request changes. The run's history, and so `all_messages()` and
anything that stores it, keeps every screenshot. Replacing images in older
messages changes the prompt prefix, so the provider's prompt cache stops
matching from the first replaced screenshot. Because only a few of the latest
turns change, that cost stays small.

## The local computer

`LocalComputer` drives this machine's display:

```python
from pydantic_ai_harness import ComputerUse
from pydantic_ai_harness.computer_use import LocalComputer

ComputerUse(computer=LocalComputer(monitor=2, max_width=1024, max_height=768))
```

Screenshots are scaled to fit `max_width` x `max_height` (default 1280x800) and
the model's coordinates are mapped back to the display, so the model works in
one coordinate space whatever the resolution or pixel density. Smaller images
cost fewer tokens; models place clicks most reliably at about 1280x800 or below.

Platform notes:

- **macOS**: the app running Python (your terminal or IDE) needs the **Screen
  Recording** and **Accessibility** permissions, under System Settings >
  Privacy & Security. Without them macOS returns only the wallpaper and drops
  synthetic input without reporting an error, so `LocalComputer` checks both
  on first use and raises `UserError` with these steps.
- **Linux**: needs an X11 session (`DISPLAY` set). Wayland is not supported.
- **Windows**: works without extra setup.

It drives your real pointer and keyboard. Actions land on whichever window has
focus, and moving the mouse during a run moves it out from under the model. For
unattended work, run the agent inside a virtual machine or a container with a
virtual display, either with `LocalComputer` inside it or with your own
`Computer` that talks to it.

## Your own computer

Anything that implements the `Computer` protocol can be driven: a VM over VNC,
a container running Xvfb, a cloud desktop, or a test double.

```python
from collections.abc import Sequence

from pydantic_ai_harness import ComputerUse
from pydantic_ai_harness.computer_use import MouseButton, ScrollDirection


class RemoteDesktop:
    async def screenshot(self) -> bytes: ...  # PNG bytes
    async def click(
        self, x: int, y: int, *, button: MouseButton = 'left', count: int = 1, modifiers: Sequence[str] = ()
    ) -> None: ...
    async def move(self, x: int, y: int) -> None: ...
    async def drag(self, path: Sequence[tuple[int, int]]) -> None: ...
    async def scroll(self, x: int, y: int, *, direction: ScrollDirection, amount: int) -> None: ...
    async def type_text(self, text: str) -> None: ...
    async def press_keys(self, keys: Sequence[str]) -> None: ...


ComputerUse(computer=RemoteDesktop(), environment='an Ubuntu 24.04 desktop in a VM')
```

Coordinates are pixels in your latest screenshot. If you scale screenshots,
map coordinates back yourself. Key names follow the vocabulary in the tool
instructions: single characters, `enter`, `tab`, `escape`, `backspace`,
`delete`, `space`, arrows, `home`, `end`, `pageup`, `pagedown`, `f1`-`f12`, and
the modifiers `ctrl`, `shift`, `alt`, `cmd`. Raise `ComputerError` for a failure
the model can correct; it is reported with a fresh screenshot. Any other
exception fails the tool call. Retries and reconnects belong in your
implementation.

## Safety

A model controlling a real computer can do anything the user can do. Three
layers limit that:

- **Instructions**: the tool instructions tell the model to treat everything on
  screen as untrusted data, and to stop and ask before purchases, sending
  messages, deleting data, accepting agreements, entering credentials, granting
  permissions, or weakening security settings. This guidance is adapted from
  the confirmation policy OpenAI ships with Codex. It lowers the risk but
  cannot rule out a mistake or a prompt injection.
- **Approval**: `require_approval=True` raises `ApprovalRequired` for any call
  that clicks, types, moves, drags, or scrolls. Answer it with core's
  [`HandleDeferredToolCalls`](../deferred-tools.md) or
  return `DeferredToolRequests` from the run:

  ```python
  from pydantic_ai import Agent, RunContext
  from pydantic_ai.capabilities import HandleDeferredToolCalls
  from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
  from pydantic_ai_harness import ComputerUse


  async def confirm(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
      approvals = {call.tool_call_id: input(f'Allow {call.args}? [y/N] ') == 'y' for call in requests.approvals}
      return requests.build_results(approvals=approvals)


  agent = Agent(
      'anthropic:claude-fable-5',
      capabilities=[ComputerUse(require_approval=True), HandleDeferredToolCalls(handler=confirm)],
  )
  ```

- **Isolation**: a VM or container limits what a mistake can reach.

## Events and telemetry

Each `computer` call emits a `ComputerActionsEvent` after its closing
screenshot is attempted. The event carries the actions, how many ran, the error
if one failed, and the screenshot size (`None` when the screenshot failed). A UI can render it from `on_event` or
`run_stream_events`. The actions are model-chosen and typed text can be
sensitive, so treat them as untrusted when you display them.

The capability adds no spans. Core's tool-execution span records each
`computer` call with its arguments, behind `include_content`, and its duration,
and core's model-request span shows which screenshots a request carried.
Leaving out old screenshots changes only what a request sends, not the run's
history, so there is no decision to record beyond what those spans show.

## Composition

- **Two `ComputerUse` on one agent** collide on the `computer` tool name. Use
  one agent per computer. Renaming the tool (for example with `PrefixTools`) is
  not supported: the instructions and screenshot pruning look for `computer`.
- **Agent specs** take `require_approval`, `environment`, `settle_seconds`, and
  `keep_screenshots`, and always drive this machine's `LocalComputer`.
- **Durable execution** works: the capability holds no run state, so each
  `computer` call can run as an activity, provided the worker can reach the
  computer.
- **How it compares to provider computer-use tools**: OpenAI's `computer` tool
  and Anthropic's `computer_20250124` are native tools whose actions the client
  still has to carry out. Core can [send OpenAI's](../models/openai.md#native-tools) through
  `openai_native_tools`, but does not execute the actions it asks for or record
  them in message history. This capability uses an ordinary function tool that
  every vision model can call, with a similar set of actions.

## API reference

::: pydantic_ai_harness.computer_use.ComputerUse

::: pydantic_ai_harness.computer_use.LocalComputer

::: pydantic_ai_harness.computer_use.Computer
