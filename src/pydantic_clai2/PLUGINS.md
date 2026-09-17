# CLAI plugins

A plugin is a Python file that teaches CLAI new tricks: react when something
happens, add a `/command`, give the agent a tool, or change how a tool's output
is shown on screen.

Everything a plugin can do goes through one object, the `PluginHost`. There is no
global registry to import and no magic file to name. You get a `host`, you tell it
what you want, you're done.

## On-demand authoring help

The default CLAI agent exposes `read_clai_customization_guide`. When you ask for
customization, its instructions tell it to read this guide first. Only the short
hint and tool description are present initially; the bundled text is read on tool
invocation. Reading it needs neither a checkout nor a network connection.

The [bundled guide](src/pydantic_clai2/customization.md) includes examples for
commands, hooks, settings, renderers, custom Termflow menus, and custom model
launchers. It also names current limits: PluginHost does not register providers,
replace the prompt editor, or alter the built-in model catalog. Those need a
custom agent launcher or a source change, as explained in the guide.

Custom agents can opt in with `customization_guide()` from
`pydantic_clai2.customization`. The tool only returns documentation; it does not
write files, activate plugins, or grant permission to execute generated code.
Keep the bundled guide aligned with this contract when changing plugin APIs.

## Credentials

CLAI's `/login openai-codex` stores tokens in the configured keyring backend,
not plugin settings. Large token bundles use multiple entries to fit Windows
Credential Manager's size limit. This does not change plugin APIs or add a
plaintext fallback. See [Codex authentication](README.md#codex-authentication)
for storage and security details.

## Where plugins live

Plugins are trusted Python code. Drop-in files execute automatically at startup;
this directory is an executable startup configuration, not a sandbox. The default
Coder runs as your OS user and can modify it, just as it can modify your shell
startup files. Use a separate OS identity or sandbox for untrusted agent work.

Two ways to install one:

1. Drop a `.py` file (or a package folder) into
   `$XDG_CONFIG_HOME/pydantic-clai2/plugins/` (default `~/.config/pydantic-clai2/plugins/`).
   Its name is the file name without `.py`.
2. Point CLAI at anything importable, from the shell or from inside CLAI:

   ```sh
   clai2 plugins add notify my_package.notify
   /plugins add coder pydantic_ai_harness.coder:Coder '{"unrestricted_filesystem": true}'
   ```

No restart needed when you do it from inside CLAI. A plugin you add or enable is
active for the next prompt; one you disable or remove is gone for the next
prompt. From the shell, `clai2 plugins ...` only saves; it loads on the next
start. Plugins in the folder and the ones you added by name are managed the
same way. Explicit module declarations take precedence over a same-named drop-in
file. Reloading a disabled plugin is rejected; enable it first. Drop-in entry
modules are compiled from current source. Installed modules use `importlib.reload`,
which can retain globals removed from source; initialize plugin state explicitly.

Plugins are trusted code running as you. Only install what you trust.

## The built-in plugin

The coding tools are a plugin too. `/plugins list` shows `coder`, backed by
`pydantic_ai_harness.coder:Coder`, marked `(built-in)` and enabled unless you
say otherwise. `/plugins disable coder` gives you a chat-only CLAI (a
writing or research setup with `ExaSearch` instead, say); `/plugins enable
coder` brings the tools back; `/plugins remove coder` cannot forget a built-in,
so it resets it to its defaults. To run `Coder` with different options, add your
own declaration under the same name and it takes the built-in's place:

```text
/plugins add coder pydantic_ai_harness.coder:Coder '{"unrestricted_filesystem": false}'
```

## Managing plugins

`/plugins` on its own opens a full-screen menu, the same kind Code Puppy uses
for `/agent` and `/mcp`:

```text
 Plugins

 [x] coder         pydantic_ai_harness.coder:Coder   | coder
 [x] notify        ~/.config/pydantic-clai2/plugins  | source  ~/.config/.../notify.py
 [ ] audit         my_package.audit                  | state   enabled, loaded
                                                     | adds    2 commands, 1 hook, 0 tools
                                                     | error   none

 Up/Down move - Space enable/disable - R reload - D remove - Enter/Q close
```

The left side lists every plugin with `[x]` for on and `[ ]` for off. The right
side shows details for the highlighted one: where it came from, whether it
loaded, what it registered, and the last error if loading failed. Every key
acts immediately; there is no save step, so Enter, Q, and Esc all just close.
Adding a plugin needs a name and a module, so that stays a typed command.

With arguments `/plugins` is a plain command, and `clai2 plugins ...` outside
CLAI does the same thing:

| Command | Does |
|---|---|
| `/plugins list` | show every plugin and whether it is on |
| `/plugins add NAME module[:attr] [JSON]` | save it and load it now |
| `/plugins remove NAME` | forget an installed declaration; persistently disable a drop-in (delete its file yourself to remove it); reset a built-in to its defaults |
| `/plugins enable NAME` / `disable NAME` | load or unload, remembered across restarts |
| `/plugins reload NAME` | re-import the file and load it again (for editing a plugin while CLAI runs) |

What "load" and "unload" mean for your plugin:

- Load runs `activate(host)` and then fires `session_start` for that plugin, so a
  plugin loaded mid-session sees the same first event as one loaded at start.
- Unload fires `session_end` for that plugin, then drops everything it
  registered: handlers, commands, tools, renderers. Nothing else is touched.
- Both only happen between prompts, never while the agent is running.
- Python cannot truly forget a module. Reload re-imports it; if a plugin keeps
  state at module level, that state comes back fresh.

## The first plugin: give the agent web search

Pydantic AI Harness ships capabilities that are plugins as they are. `ExaSearch`
adds `web_search` and `get_page` tools backed by [Exa](https://exa.ai). Install
the extra and set the key, then add the class by name:

```sh
pip install 'pydantic-ai-harness[exa]'
export EXA_API_KEY=...
```

```text
/plugins add exa pydantic_ai_harness.exa:ExaSearch '{"num_results": 8}'
```

The JSON is passed to the constructor, so any keyword `ExaSearch` accepts that
JSON can express works here; an option that takes a Python object, like
`client`, needs the file form below. Ask the agent something that needs the web on the next prompt and it has
the tools. `/plugins disable exa` takes them away again.

The same thing as a plugin file, `~/.config/pydantic-clai2/plugins/search.py`,
which is the shape to start from when you want more than one capability, or a
`/command`, or a hook alongside it:

```python
from pydantic_ai_harness.exa import ExaSearch

from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost) -> None:
    host.add(ExaSearch(num_results=8, text_summary=True))
```

A plugin module exposes one function, `activate(host)`. CLAI calls it once when
the plugin loads. Everything you register inside it stays until the plugin is
unloaded or CLAI quits. Any Pydantic AI capability goes through `host.add`, so
`YouSearch` from `pydantic_ai_harness.youdotcom`, core's `WebSearch`, or one you
wrote yourself all work the same way.

`module:attr` may also name a function that takes a host, if you prefer a name
other than `activate`.

## Reacting to a moment

Plugins are not only for tools. This one rings the terminal bell when a turn
finishes, so you can tab away during a long run:

```python
from pydantic_clai2.plugins import PluginHost, TurnEnd


def activate(host: PluginHost) -> None:
    @host.on('turn_end')
    async def ping(event: TurnEnd) -> None:
        host.console.bell()
```

## What you can register

### React to a moment: `@host.on(name)`

`name` is a string. Your editor autocompletes it, and the type checker knows what
`event` your handler receives for each name. A typo is an error, not silence.
`host.on` is always used as a decorator.

Four names belong to CLAI itself. They fire outside the agent run, in the shell:

| Name | When | Event fields | Can change things? |
|---|---|---|---|
| `session_start` | CLAI has started, before the first prompt | `agent`, `settings` | no |
| `session_end` | CLAI is quitting | `reason`: `exit`, `eof`, or `error` | no |
| `turn_start` | you pressed Enter on a prompt | `text` | yes: edit `event.text`, or `event.cancel()` |
| `turn_end` | the turn finished, failed, or was interrupted | `text`, `outcome`, `result`, `error` | no |

Ctrl-C during an agent run keeps the prompt and captured partial messages in
conversation history for the next turn. Cancellation still reaches the running
tools for cleanup; it does not undo completed side effects or retry the run.
A prompt cancelled by `turn_start` never starts an agent run and is not retained.

Every other name is a Pydantic AI lifecycle hook, spelled exactly as on core's
`Hooks().on`, with the same handler signature. The ones people reach for:

| Name | When |
|---|---|
| `before_run` / `after_run` | an agent run starts / finishes |
| `before_model_request` | just before the model is called; you can edit the request |
| `before_tool_execute` | a tool is about to run; raise to stop it |
| `after_tool_execute` | a tool has returned |
| `tool_execute_error` | a tool raised |
| `event` | every stream event; prefer `@host.on(EventClass)` below |

The full list and every signature are in the
[hooks reference](https://pydantic.dev/docs/ai/core-concepts/hooks/). Core's
`wrap_*` and `on_*_error` methods drop their prefix here (`run`, `tool_execute`,
`run_error`), matching `Hooks().on`.

### React to a typed event: `@host.on(EventClass)`

Tools and capabilities emit typed events (a shell started, a file was written).
Pass the class instead of a string:

```python
from pydantic_ai import RunContext
from pydantic_ai_harness.filesystem import FileWrittenEvent
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    @host.on(FileWrittenEvent)
    async def log_write(ctx: RunContext[None], event: FileWrittenEvent) -> None:
        host.console.print(f'wrote {event.path}')
```

Some events let you say no. If the event has a `cancel()` method, calling it stops
the action before it happens (for example `FileChangeRequestEvent`).

### Add a `/command`: `host.commands.register(...)`

```python
from pydantic_clai2.commands import Command

host.commands.register(
    Command(
        name='greet',
        description='Say hello',
        handler=lambda args: 'Hello ' + (' '.join(args) or 'there'),
    )
)
```

The handler gets the arguments as a list of strings and returns the text to show.
It may be `async`. Add `complete=` to offer Tab suggestions. Names must be unique;
clashing with a built-in is an error at startup, not a silent override.

### Give the agent tools or instructions: `host.add(capability)`

```python
from pydantic_ai.capabilities import Capability

tools = Capability(instructions='Prefer British spelling.')


@tools.tool_plain
def word_count(text: str) -> int:
    return len(text.split())


host.add(tools)
```

`host.add` also accepts a function that takes a `RunContext` and returns a
capability (or `None`), for tools that should only exist in some runs.

### Draw an event yourself: `@host.render(EventClass)`

By default CLAI shows unknown tool calls as a dim `● tool_name`. To show something
better, return a Rich renderable (a `str` is fine). Return `None` to say "not mine,
use the default".

```python
from pydantic_ai_harness.filesystem import FileWrittenEvent
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    @host.render(FileWrittenEvent)
    def show_write(event: FileWrittenEvent) -> str:
        return f'wrote {event.path}'
```

CLAI flushes any streaming text before it prints what you return, so your output
never lands in the middle of a paragraph.

### Read your settings: `host.settings(Model)`

The JSON passed to `plugins add` is validated against a model you define:

```python
from pydantic import BaseModel


class NotifySettings(BaseModel):
    sound: bool = True


settings = host.settings(NotifySettings)
```

Bad or missing values fail at startup with a message naming your plugin.

## Rules that keep plugins predictable

- Handlers are `async`. There is no sync variant of anything.
- CLAI host observers return `None`. Core hooks keep their exact core return
  contracts: for example `before_model_request` must return its `ModelRequestContext`.
  For cancelable host events, edit the event or call `event.cancel()`.
- Raising in `turn_start` prevents the turn. Raising in core `before_tool_execute`
  fails the agent run, not only that tool. Use core's documented tool-denial
  mechanisms when the model should recover instead of ending the run, such as
  `pydantic_ai.exceptions.SkipToolExecution` for skipping an individual tool.
- Startup plugins load in alphabetical ID order. CLAI host handlers run in
  activation order, while `/plugins list` remains alphabetical. Core hook ordering
  follows core composition, including reverse order for `after_*` hooks. The first renderer that returns something wins. A
  plugin loaded later goes to the end of the line.
- Anything you print, print through `host.console`, so it stays in step with
  streaming output.

## Testing a plugin

`PluginHost` is an ordinary object. Build one with a `Console` writing to a
`StringIO`, call your `activate`, then call the handlers you registered with
hand-made events. No terminal, no model, no network.

```python
from io import StringIO

from rich.console import Console

from pydantic_clai2.plugins import PluginHost, TurnEnd

host = PluginHost(name='notify', console=Console(file=StringIO()), settings={})
activate(host)
for handler in host.handlers:
    await handler(TurnEnd(text='hi', outcome='completed'))
```

## vllm connection

Open `/model`, choose `vllm`, then enter a trusted HTTP(S) server root or `/v1` URL, and optionally a token. CLAI queries `/v1/models` and opens a searchable model picker. HTTP sends tokens unencrypted; use HTTPS outside trusted local networks.

## openrouter connection

Open `/model`, choose `openrouter`, then paste an API key from https://openrouter.ai/keys in the masked prompt, then select a model from the live catalog. CLAI validates the key with `/api/v1/key` before fetching `/api/v1/models`. This flow uses API-key authentication, not browser OAuth.

The connection is saved in the configured Python keyring backend after selection; backend security depends on your keyring configuration. Tokens are not stored in SQLite or command history. The selected model persists across restarts. Select the provider again to browse its live models or reconfigure the saved connection. Discovery is explicit and has a 20-second network timeout; redirects are not followed. Agent inference uses Pydantic AI core.
