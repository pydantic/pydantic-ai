---
title: Local Desktop
description: "Give a Pydantic AI agent a screen on this Linux machine for computer use: a private virtual screen, or the one you're logged in to."
---

# Local Desktop

Give [computer use](computer-use.md) a screen on this Linux machine: a private, invisible one started for the agent,
or the screen you're logged in to.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/local_desktop/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

```bash
pip/uv-add "pydantic-ai-harness[anthropic]"
```

```bash
sudo apt install xvfb fluxbox x11-utils xdotool x11-xserver-utils scrot util-linux
```

Swap `anthropic` for your model provider's extra. The `apt` packages are the screen (Xvfb), its window manager
(fluxbox), and the tools that take screenshots and press keys; other distributions name them similarly. Your own
screen needs only `x11-utils xdotool x11-xserver-utils scrot`.

## Usage

```python {title="local_desktop.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse, LocalDesktop
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:medium }}', capabilities=[LocalDesktop('.'), Coder(), ComputerUse()])


async def main():
    result = await agent.run('Start Firefox, find the current Python release on python.org, and tell me.')
    print(result.output)
```

`LocalDesktop` is a [`LocalWorkspace`](../workspace.md) with a screen: commands and files are on this machine, in
this directory, and apps the agent starts with a command, through [`Coder`](coder.md) here, open on the screen.

The screen is private to the agent: the `computer` tool never sees your windows or moves your mouse, and other users
of the machine can't connect to it. It starts on the first run, at
1280x800 (`LocalDesktop('.', size=(1920, 1080))` for another size), and every later run of the agent uses the same
one, with the windows earlier runs left open. Runs at the same time share it too, so give each concurrent user a
`LocalDesktop` of their own. It stops when your program exits; if only its window manager exits, such as from its
menu, that starts again and the windows stay. A `LocalDesktopBackend` you build yourself for `workspace=` shares one
private 1280x800 screen with every other backend in the process given none; pass `screen=VirtualScreen(size)` for
one of its own, and `await screen.stop()` to stop it.

## Your own screen

`LocalDesktop('.', real_screen=True)` drives the screen you're logged in to: the agent sees your windows, moves your
mouse and types into whatever has focus.

- Pair it with `ComputerUse(requires_approval=True)` to approve each call first.
- Typing a character outside ASCII, such as `é`, borrows the code of a rarely used key, such as Wi-Fi, Bluetooth or a
  brightness key (with many characters, also Suspend or Play/Pause). That key then types the character until the
  keyboard layout is next set, such as by `setxkbmap`.
- What you type during a run mixes with what the agent types, and a modifier either of you holds applies to both: X11
  has one keyboard for everyone. Agents in one process that use your screen, in any working directory, take turns at
  it a whole `computer` call at a time.

It works in an X11 (Xorg) session. In a Wayland session, the default on recent Ubuntu and Fedora, a desktop can
neither see nor control the screen, so the run fails at its first request (commands still run); use the private
screen there.

## Limitations

- Linux only. For a desktop elsewhere, use [`DockerSandbox(desktop=True)`](docker-sandbox.md),
  [`DaytonaSandbox(desktop=True)`](daytona-sandbox.md#give-the-agent-a-desktop) or
  [`E2BSandbox(desktop=True)`](e2b-sandbox.md#give-the-agent-a-desktop).
- Commands run as you, with access to everything you can reach, as with `LocalWorkspace`.

## API reference

::: pydantic_ai_harness.local_desktop.LocalDesktop

::: pydantic_ai_harness.local_desktop.LocalDesktopBackend

::: pydantic_ai_harness.local_desktop.VirtualScreen
