# Customizing CLAI 2

This guide describes the installed pydantic_clai2 shell. Read relevant installed
source before editing: versions of Pydantic AI and Termflow can differ. Do not
invent registration APIs. PLUGINS.md in the CLAI repository is the full plugin
contract; this guide is shipped with the package for use without a checkout.

## Choose the extension point

- Add tools, instructions, or agent hooks: a Pydantic AI capability, registered
  with host.add, or installed directly as a capability class.
- Add slash commands or a custom menu: host.commands.register(Command(...)).
- Change tool output: host.render(EventClass), returning a Rich renderable.
- Add a fragment to the status row: host.status_segment(fn), where fn returns a short string.
- React to prompts or session lifecycle: host.on with a typed handler.
- Configure a plugin: host.settings with a Pydantic settings model.
- Use a custom model/provider: supply a Pydantic AI Agent to chat from a Python
  launcher. There is no host.register_provider or host.register_model API.
- Select colours: /theme opens the Termflow palette picker; /theme tokyo_night
  selects directly and persists display.theme. /theme default restores CLAI's
  existing appearance. Browsing previews a sample conversation without applying
  the candidate. /set exposes the same preference.
- Replace the prompt editor, splash, streaming Markdown, built-in model catalog,
  or add custom palettes: currently a CLAI source change, not a supported
  PluginHost extension. A command can own its own UI instead. You can add a
  fragment to the status row with host.status_segment; you cannot redesign or
  replace the row itself.

Pydantic AI core owns the agent loop, model/provider protocols, hooks and tools.
Harness owns reusable, non-terminal capabilities. CLAI owns the prompt loop,
commands, plugin loading and rendering. Do not implement a second agent loop in
CLAI or print terminal output from a reusable Harness capability.

## Create and install a plugin

The quickest plugin is an existing Pydantic AI capability. Pydantic AI Harness
ships several; ExaSearch adds web_search and get_page tools backed by Exa. It
needs the exa extra installed in CLAI's Python environment and EXA_API_KEY set,
then one command, no file:

```text
/plugins add exa pydantic_ai_harness.exa:ExaSearch '{"num_results": 8}'
```

The JSON supplies constructor keyword arguments, so only values JSON can
express work there; read the capability's signature in the installed source to
know which exist. Options that take Python objects, such as ExaSearch's client,
need a drop-in file that constructs the capability in code. Other capabilities
register the same way: YouSearch from pydantic_ai_harness.youdotcom, core's
WebSearch, or a user-written AbstractCapability subclass.

When a plugin needs anything beyond one capability, write a drop-in file. One
plugin may do as many things as it likes: several capabilities, commands, hooks,
renderers, in any combination. The example below deliberately does two
unrelated things, adding ExaSearch and registering a /greet command, to show
both shapes side by side; a real plugin would usually pick one purpose. Create
~/.config/pydantic-clai2/plugins/search.py, or use
$XDG_CONFIG_HOME/pydantic-clai2/plugins/search.py when XDG_CONFIG_HOME is set:

```python
from pydantic_ai_harness.exa import ExaSearch

from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    host.add(ExaSearch(num_results=8, text_summary=True))
    host.commands.register(
        Command(
            name='greet',
            description='Say hello',
            handler=lambda args: 'Hello ' + (' '.join(args) or 'there'),
        )
    )
```

Restart CLAI to discover the drop-in, or use /plugins enable search while running.
The agent has the search tools on the next prompt and /greet Ada prints Hello Ada.
Use /plugins reload search after editing. Alternatively install an importable
Python package in CLAI's Python environment and run /plugins add search
my_package.search. A module exports activate(host); module:attr may identify a
host activation function or a bare capability class. For a bare class, the
optional JSON supplies constructor keyword arguments:

```text
/plugins add coder pydantic_ai_harness.coder:Coder '{"unrestricted_filesystem": true}'
/plugins list
/plugins disable search
/plugins enable search
/plugins reload search
/plugins remove search
```

The coding tools are themselves the built-in plugin named coder, shown by
/plugins list as pydantic_ai_harness.coder:Coder (built-in). Do not add a
second Coder under another name. To change its options, declare coder again
with the same name and different JSON; that replaces the built-in. Keep
"repo_context": false in that JSON: the second built-in, repo_context
(pydantic_clai2.repo_context), already reads AGENTS.md or CLAUDE.md from the
launch directory, and Coder's bundled RepoContext would load it again. To run
without coding tools, /plugins disable coder; to stop reading the instruction
file, /plugins disable repo_context. /plugins remove coder resets the
built-in to its defaults rather than removing it. A repository's
.clai/settings.json can declare plugins too; they show as (project), rank
just above the built-ins, and start off until the user runs /plugins enable
NAME, so repository code never runs without that approval. Outside a session,
clai2 plugins add NAME module[:attr] [JSON] saves for the next startup.
/plugins opens the management menu. Removing a drop-in disables it persistently;
delete its source file yourself to remove it from disk.

The second built-in is ask_user (pydantic_clai2.ask_user_menu:activate): the
harness AskUser capability with an inline numbered picker as its answerer, so
the model can ask the user multiple-choice questions mid-run through
ask_user_question. The conversation remains visible. Enter or a number selects;
for multiple selections it toggles, then Done submits. /plugins disable ask_user removes the tool. To answer the
questions somewhere other than the terminal, declare ask_user again with a
module whose activate(host) calls host.add(AskUser(answerer=...)) with your own
async answerer; see PLUGINS.md.

The inline `ask_user_question` picker also offers `Other (type answer)`.
Choose it to type your own answer instead of the suggested options, including for
multi-select questions. Enter submits nonblank text. Esc returns to the choices
and keeps your draft; Ctrl-C declines the whole request. Backspace and arrow keys
edit the text. Multiline paste is inserted as text and waits for Enter; it does
not submit an answer or select choices. The conversation stays visible while you type. Custom answers
appear in the transcript and reach the model as a one-item list under the question's header.

The built-in logfire plugin (pydantic_clai2.logfire) is enabled by default in the
stock CLI. It contributes core's Instrumentation capability using an isolated
Logfire instance. It exports to Logfire only when credentials are present, with
no interactive setup or console logging. Text and binary images are included by
default, so review the telemetry destination before setting LOGFIRE_TOKEN. Use
/plugins disable logfire to remove it, or replace its settings with:

```text
/plugins add logfire pydantic_clai2.logfire '{"include_content": false, "include_binary_content": false}'
```

Other options are service_name (default pydantic-clai2) and send_to_logfire
(default "if-token-present", or false). This explicit option overrides
LOGFIRE_SEND_TO_LOGFIRE. Use LOGFIRE_TOKEN or the SDK credential file in
$XDG_CONFIG_HOME/pydantic-clai2/logfire (default ~/.config/pydantic-clai2/logfire).
Both SDK configuration and credentials are read from that user directory, not
from the checkout. LOGFIRE_CONFIG_DIR and LOGFIRE_CREDENTIALS_DIR are ignored;
relative XDG_CONFIG_HOME falls back to ~/.config. Keep tokens out of plugin JSON. Disabling or reloading shuts down only the plugin's own
providers, without mutating the agent or global tracer/meter providers. The
existing global propagator is preserved, but SDK-installed executor propagation
helpers are not removed on unload. While enabled, its per-run instrumentation takes precedence over the supplied
agent's instrumentation; disabling restores that agent's own behavior. Standard
SDK configuration, including explicit OTLP exporters, still applies.

Plugins are trusted Python executed as the user. Drop-ins execute at startup,
not in a sandbox. Do not install code or change executable startup configuration
without the user's intent. Keep secrets out of plugin JSON: it is plaintext in
SQLite. Use environment variables or plugin-owned credential storage instead.

Each plugin gets its own PluginHost. Keep mutable state inside activate, not in
module globals. Loading calls activate then session_start; unloading calls
session_end and discards that host's registrations. Changes happen between turns.
Failed or cancelled loading calls registered session_end handlers with reason=error
under cancellation shielding, with a five-second cooperative timeout per handler,
then discards partial registrations. Handler errors/timeouts are reported and later
handlers still run. Blocking code and nested shields can exceed the deadline.
Cleanup can run before session_start finishes; register it once the plugin owns a
resource. Drop-in entry modules
reload from fresh source; installed modules use importlib.reload, which can retain
globals absent from the new source. Initialize state explicitly on activation.
Do not mutate another plugin's host or the agent to register a plugin's tools.

## Reload the shell during development

```text
/reload
```

Use `/reload` after editing `pydantic_clai2` itself, not just a plugin. It uses
`importlib.reload` and rebuilds the prompt loop, commands, and session without
restarting Python. Conversation history, the agent and its dependencies, selected
model, and active settings are kept. Enabled plugins unload and activate again
against the refreshed shell types; disabled and unapproved project plugins stay
off. Plugin hosts and their registrations are recreated, but installed module
globals not overwritten by the new source can survive. Initialize mutable state
in `activate`. Failed imports or shell rebuilds restore previous module bindings
and report the error; correct the source and retry. Import-time side effects
cannot be undone.

Reload ordering follows module-scope imports in the current source, including
newly added dependencies between CLAI modules and new local modules. Function-local
imports and `TYPE_CHECKING` guards do not create eager dependencies. Literal
guards and direct platform/version comparisons select their active branch without
executing source expressions. Other conditions are analyzed conservatively and
may require a restart if their alternatives form a cycle. New modules
are imported only if reached by the updated code; invalid source or a detected
import cycle fails before reloads begin. Restart for changes to startup code,
agent construction, or dynamically loaded dependencies. Third-party dependencies
are not recursively reloaded. Use `/plugins reload NAME` to reload only one plugin.

## Hooks, tools and settings

The four host hooks use async observers returning None:

| Hook | Event | Use |
| --- | --- | --- |
| session_start | SessionStart(agent, settings) | Initialize session resources |
| session_end | SessionEnd(reason) | Clean up; reason is exit, eof, or error |
| turn_start | TurnStart(text) | Rewrite event.text or event.cancel() |
| turn_end | TurnEnd(text, outcome, result, error) | Observe completion or failure |

Import these event classes from pydantic_clai2.plugins. Event payloads are typed,
keyword-only dataclasses. A raising turn_start handler cancels the turn. Do not
return a replacement string or a boolean to decide a host action.

```python
from pydantic import BaseModel
from pydantic_ai.capabilities import Capability
from pydantic_clai2.plugins import PluginHost, TurnStart


class Options(BaseModel):
    prefix: str = 'Please answer concisely. '


def activate(host: PluginHost[None]) -> None:
    options = host.settings(Options)

    @host.on('turn_start')
    async def prefix(event: TurnStart) -> None:
        event.text = options.prefix + event.text

    tools: Capability[None] = Capability(instructions='Use word_count for exact counts.')

    @tools.tool_plain
    def word_count(*, text: str) -> int:
        return len(text.split())

    host.add(tools)
```

Pass options as JSON with /plugins add NAME module '{"prefix": "..."}'. Settings
are validated on activation. host.add also accepts a RunContext-to-capability
factory returning a capability or None.

All other named hooks match Pydantic AI Hooks().on names and signatures. Check
https://pydantic.dev/docs/ai/core-concepts/hooks/ and installed source. For example,
before_model_request returns its ModelRequestContext; it is not a None-returning
host observer. Core wrappers and error hooks use Hooks().on's spelling, such as
run, tool_execute, and run_error. Do not rename them or add host hooks for core
behaviour. Raising before_tool_execute fails the run; consult core's documented
SkipToolExecution when only one tool execution should be skipped.

For typed capability events use @host.on(EventClass) with an async handler taking
RunContext and the event. Match on event classes rather than tool-name strings.
If an event supports cancel(), use its documented cancellation semantics.

## CLI UX and rendering

Command handlers receive list[str] arguments and return a string or an awaitable
string. Register complete= on Command for Tab suggestions. Command names must be
unique, including built-ins. Unknown command-shaped input such as `/missing`
does not reach the model. Path-like input (a slash, dot, or backslash in the first
token after `/`) is passed through as a prompt instead. Routing itself does not
read files. Separately, the editor converts bracketed pastes of existing image
paths into attachments. Ctrl-V or Alt-V attaches clipboard images. Attachment
markers are removed before host turn hooks run; those hooks receive the text
caption, while core hooks receive the multimodal request. Quoted paths are also
prompts.
Commands run between turns, which makes them suitable for configuration menus.

Use a renderer for output during a stream:

```python
from pydantic_ai_harness.filesystem import FileWrittenEvent
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    @host.render(FileWrittenEvent)
    def show_write(event: FileWrittenEvent) -> str:
        return f'wrote {event.path}'
```

Return a Rich renderable, or None to let the next renderer/default handle it.
CLAI flushes streaming text before printing it. First matching non-None renderer
wins. Do not print from an event observer when a renderer can do the job.
Use host.console for plugin-owned console output outside streaming handlers.
Resolve pydantic_clai2.theme roles ACCENT, INFO, WARNING, ERROR, MUTED, THINKING
with theme.color(role) at render time, not hard-coded colours. Raw ANSI uses
theme.sgr(role), which resolves the selected colours itself. Choices are default
(the unchanged CLAI appearance) and termflow.themes.PALETTES. theme.current()
returns the selected TerminalPalette or None for default. Starting and exiting
in default leaves terminal colours untouched. For bundled palettes, Termflow
applies foreground, background, and ANSI slots via OSC; returning to default or
exiting resets terminal colours. Redirected output receives no palette changes.
The early splash keeps brand colours. Code uses the terminal foreground and ANSI
syntax colours. Default diffs stay unchanged; bundled palettes use Termflow's diff defaults. StreamRenderer owns text and
thinking, not tool-specific rendering.

Add a fragment to the status row with host.status_segment:

```python
import os

from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    @host.status_segment
    def where() -> str:
        return os.getcwd()
```

The fragment is appended after the built-in figures and painted muted. The row
repaints about ten times a second, so it runs that often: no blocking IO, no
awaits, no printing. Return an empty string to contribute nothing that frame.
Fragments are truncated from the right on narrow terminals and cannot set their
own colours. host.status_segment adds to the row; replacing the row, prompt
editor, splash, or adding custom palettes is still a CLAI source change.

## Custom TUI menus

Register an async slash command that builds and runs a Termflow MenuBuilder.
Keep the builder pure so tests can drive it without a terminal. The existing
plugin_menu.py, model_menu.py and field_menu.py are working source examples.
The following uses CLAI's internal UI helpers; check them when upgrading:

```python
from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu
from pydantic_clai2._rendering import markdown_style
from pydantic_clai2.commands import Command
from pydantic_clai2.menu_worker import menu_key, run_worker
from pydantic_clai2.plugins import PluginHost


def build_menu() -> Menu:
    return (
        MenuBuilder('My plugin')
        .style(markdown_style())
        .items([MenuItem('About', value='about')])
        .preview(lambda item: 'My plugin details')
        .footer_hint('Enter selects - Esc closes')
        .key_source(menu_key)
        .build()
    )


def activate(host: PluginHost[None]) -> None:
    async def show_menu(args: list[str]) -> str:
        await run_worker(lambda: build_menu().run())
        return ''

    host.commands.register(Command(name='my_menu', description='Open my menu', handler=show_menu))
```

Termflow owns the alternate screen. Do not print while it is open. Put errors and
empty states in disabled rows or the preview. Use .on_key for single-key actions,
apply changes immediately, then replace_items to redraw. Esc and Ctrl-C should
close normally. menu_key and run_worker cooperate on cancellation, keeping the
terminal owned until the worker restores its screen. Do not fire-and-forget a
thread that is still reading input. If a menu key needs an async action, follow
plugin_menu.py's bridge back to the main event loop.

For named validated fields, reuse FieldSource, FieldMenu and run_flow in
field_menu.py rather than write another editor. SettingsSource in set_menu.py
shows the adapter; model_menu.py uses the same editor for model settings. Tests
inject Runners with scripted widget results (tests/menu_script.py). These are
internal shell helpers, not a promise of a stable third-party UI API. Plugins do
not currently receive CommandContext from PluginHost; do not invent host.context.

## Custom models and providers

A model identifier accepted by an existing core provider can be selected with
/add_model PROVIDER:NAME or /set model PROVIDER:NAME even if it is absent from the
catalog. `/model` and its Tab suggestions select only previously added models.
Adding a model also selects it and saves it for later sessions. Install optional provider dependencies in the same environment as CLAI
and supply credentials via the provider's supported environment variables.

For an OpenAI-compatible endpoint, write a Python launcher:

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.coder import Coder
from pydantic_clai2 import chat
from pydantic_clai2.customization import customization_guide

model = OpenAIChatModel(
    'my-model',
    provider=OpenAIProvider(
        base_url='https://your-service.example/v1',
        api_key=os.environ['MY_MODEL_API_KEY'],
    ),
)
agent = Agent(model, capabilities=[Coder(), customization_guide()])
asyncio.run(chat(agent, deps=None))
```

Coder() here restricts file tools to the workspace, unlike the stock CLI's
unrestricted Coder. A launcher gets no built-in plugins unless it passes them.
Choose one source of coding tools, never both: either keep Coder() in
capabilities as above, or drop it from capabilities and call chat(agent,
deps=None, builtin_plugins=DEFAULT_PLUGINS) with DEFAULT_PLUGINS from
pydantic_clai2, which supplies the stock coder and ask_user plugins and lets
/plugins manage them. Passing both loads two sets of coding tools. A fully custom protocol belongs in a Pydantic AI Model and
Provider implementation, not a terminal plugin. See
https://pydantic.dev/docs/ai/models/overview/ and inspect installed core abstract
classes for required methods. Supply that Model instance to Agent as above.
There is no --agent option in the current CLI; run the Python launcher instead.

chat preserves a supplied agent's model when no settings override selects another
one. /model or /set model changes subsequent turns to the selected core model
identifier, not an alias for your custom instance. Noninteractive Session exposes
resolve_model for translating overrides; chat currently configures its own
resolver for Codex authentication, not a plugin provider registry.

For `openai-codex` models, open `/model_settings openai-codex:gpt-6-astra`
(or your saved Codex model), then **Service Tier / Fast Mode**. Choose
**Fast (priority)** to request fast processing, or **Standard (default)** to
turn it off. [Codex fast mode](https://developers.openai.com/codex/speed)
uses more ChatGPT credits and depends on model and account availability. It does
not lower reasoning effort. Reset restores the existing model default; it does
not enable fast mode. The stored values remain `service_tier=priority` and
`service_tier=default`, so older CLAI versions can read them. A custom
`service_tier` body parameter still takes precedence.

To extend the built-in picker in a CLAI source change, add a source returning
CatalogModel values in model_catalog.py and merge it in catalog(). Adding a
catalog row does not implement provider support. Editable per-model settings
are declared in ModelSettingsForm in model_settings.py; extend that form, not a
second editor. Credentials belong in provider-supported storage, not model
settings. /login currently covers Codex, not arbitrary provider authentication.

## Test and verify

Construct PluginHost(name='example', console=Console(file=StringIO()), settings={})
and call activate directly. Check commands and typed event effects, not just
registration counts. Use synthetic events for renderers and Pydantic AI TestModel
for agent/tool integration; no real model or credentials are required. Use real
cancel scopes and Events, not sleeps, to order cancellation tests. Drive menu
builders headlessly, with injected runners instead of a real terminal.

In a CLAI checkout run Ruff format/check, strict Pyright on changed source/tests,
and focused pytest tests. Check /plugins list for loading errors, reload after
edits, and verify disable removes your commands and tools. Keep README.md and
PLUGINS.md aligned with user-facing API changes. For UI capabilities not exposed
by PluginHost, state that limitation and propose a focused source change rather
than monkeypatching a private global registry.

## Headless invocation

`clai2 -p "PROMPT" -m PROVIDER:NAME` runs one saved turn and prints only the final
answer. Prompt text is required as an argument; stdin is not read. Use
`--resume SESSION-ID` to continue saved history without opening the browser.
The `ask_user` plugin is skipped without changing saved preferences. Full-screen
requests fail, stream renderers are not called, and host console output is
suppressed. Plugins must not read input or print directly to stdout. Errors go
to stderr with a nonzero exit status. `-m` also works in the interactive CLI.
