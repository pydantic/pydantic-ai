# Local Desktop

Give [computer use](https://pydantic.dev/docs/ai/harness/computer-use/) a screen on this Linux machine: a private, invisible one started for the agent,
or the screen you're logged in to.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/local_desktop/)

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

```bash
sudo apt install xvfb fluxbox x11-utils xdotool x11-xserver-utils scrot util-linux
```

Use your model provider's extra instead of `anthropic`. The `apt` packages are the screen (Xvfb), a window manager
(fluxbox), and the tools for screenshots and keys; other distributions use similar names. Using your own screen
needs only `x11-utils xdotool x11-xserver-utils scrot`.

## Usage

```python {title="local_desktop.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import ComputerUse, LocalDesktop
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-sonnet-5-5', capabilities=[LocalDesktop('.'), Coder(), ComputerUse()])


async def main():
    result = await agent.run('Start Firefox, find the current Python release on python.org, and tell me.')
    print(result.output)
```

`LocalDesktop` is a [`LocalWorkspace`](https://pydantic.dev/docs/ai/core-concepts/workspace/) with a screen. Commands and files are on this machine, in this
directory, and apps the agent starts, here through [`Coder`](https://pydantic.dev/docs/ai/harness/coder/), open on the screen.

The screen is private: the agent never sees your windows or moves your mouse, and other users can't connect to it.

- It starts on the first run at 1280x800; pass `size=(1920, 1080)` for another size.
- Later runs, and runs at the same time, use the same screen, with the windows earlier runs left open. Give each
  concurrent user their own `LocalDesktop`.
- It stops when your program exits. If only the window manager exits, it starts again and the windows stay.
- A `LocalDesktopBackend` you build for `workspace=` shares one private screen with the other backends in the
  process. Pass `screen=VirtualScreen(size)` for its own, and `await screen.stop()` to stop it.

## Your own screen

`LocalDesktop('.', real_screen=True)` uses the screen you're logged in to: the agent sees your windows, moves your
mouse and types into whatever has focus.

- Use `ComputerUse(requires_approval=True)` to approve each call first.
- To type a character outside ASCII, such as `é`, it borrows a rarely used key, such as Wi-Fi or a brightness key.
  That key types the character until the keyboard layout is reset, for example with `setxkbmap`.
- Your typing mixes with the agent's, and a held modifier applies to both: X11 has one keyboard. Agents in one
  process take turns on your screen, a whole call at a time.

It needs an X11 (Xorg) session. On Wayland, the default on recent Ubuntu and Fedora, the run fails at its first
request; use the private screen there.

## Limitations

- Linux only. For a desktop elsewhere, use [`DockerSandbox(desktop=True)`](https://pydantic.dev/docs/ai/harness/docker-sandbox/),
  [`DaytonaSandbox(desktop=True)`](https://pydantic.dev/docs/ai/harness/daytona-sandbox/#give-the-agent-a-desktop) or
  [`E2BSandbox(desktop=True)`](https://pydantic.dev/docs/ai/harness/e2b-sandbox/#give-the-agent-a-desktop).
- Commands run as you, with access to everything you can reach, as with `LocalWorkspace`.
