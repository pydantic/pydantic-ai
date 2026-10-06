# Customizing CLAI 2

This guide describes the installed pydantic_clai2 shell. Read relevant installed
source before editing: versions of Pydantic AI and Termflow can differ. Do not
invent registration APIs. PLUGINS.md in the CLAI repository is the full plugin
contract; this guide is shipped with the package for use without a checkout.

## Choose the extension point

A plugin is a `Plugin` subclass from pydantic_clai2.plugins. Like a Pydantic AI
AbstractCapability, it declares what it contributes by overriding `get_*`
methods, each defaulting to nothing; CLAI calls them once when the plugin loads.

- Add tools, instructions, or agent hooks: return Pydantic AI capabilities
  (including `Hooks`) from get_capabilities, or install a capability class
  directly.
- Add slash commands or a custom menu: return Command(...) values from get_commands.
- Change tool output: override render(event), returning a Rich renderable or None.
- Add a fragment to the status row: return functions from get_status_segments,
  each returning a short string.
- Add a working animation: return make_spinner(name, frames, interval=...,
  description=...) values from get_spinners, or add an entry in spinners.json next
  to CLAI's settings (/spinner init writes a starter). /spinner picks one; the
  choice persists as display.spinner.
- React to prompts or session lifecycle: override on_session_start,
  on_session_end, on_turn_start, or on_turn_end.
- Configure a plugin: declare a Pydantic settings model as the class's type
  parameter, `Plugin[MySettings]`, and read self.settings.
- Offer a settings menu: override async configure().
- Use a custom model/provider: return ModelProvider(prefix=..., resolve=...,
  models=...) from get_model_providers, where resolve returns a Pydantic AI
  Model, or supply a Pydantic AI Agent to chat from a Python launcher.
  If its models need a sign-in, return PluginLogin(name=..., handler=...,
  models=...) values from get_logins to add /login NAME and save those models
  once it succeeds. ModelProvider(settings_from='anthropic') gives its models
  Anthropic's /model settings controls.
- Select colours: /theme opens the Termflow palette picker; /theme tokyo_night
  selects directly and persists display.theme. /theme default restores CLAI's
  existing appearance. Browsing previews a sample conversation without applying
  the candidate. /set exposes the same preference.
- Replace the prompt editor, splash, streaming Markdown, built-in model catalog,
  or add custom palettes: currently a CLAI source change, not a supported
  plugin extension. A command can own its own UI instead. You can add a
  fragment to the status row with get_status_segments; you cannot redesign or
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
from collections.abc import Sequence

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.exa import ExaSearch

from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import Plugin


class Search(Plugin):
    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (ExaSearch(num_results=8, text_summary=True),)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='greet',
                description='Say hello',
                handler=lambda args: 'Hello ' + (' '.join(args) or 'there'),
            ),
        )
```

Restart CLAI to discover the drop-in, or use /plugins enable search while running.
The agent has the search tools on the next prompt and /greet Ada prints Hello Ada.
Use /plugins reload search after editing. Alternatively install an importable
Python package in CLAI's Python environment and run /plugins add search
my_package.search. A module defining exactly one public Plugin subclass can be
named on its own; module:Class names one Plugin subclass or a bare capability
class. For a bare capability class, the optional JSON supplies constructor
keyword arguments; for a Plugin, it is validated against its settings model:

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
(pydantic_clai2.builtin_plugins.repo_context), already reads AGENTS.md or CLAUDE.md from the
launch directory, and Coder's bundled RepoContext would load it again. CLAI
enables task delegation in the stock Coder plugin. When the active plugin
capabilities change, CLAI rebuilds its stock agent before the next prompt with
those capabilities bound to it. A delegated task gets a fresh conversation with
the same plugin tools, instructions, and guardrails. Supplied agents are unchanged
and still receive plugins per run; self-delegation on them requires capabilities
bound at agent construction. Saved Coder declarations that omit "sub_agents"
still default to false; set "sub_agents": true in /plugins configure coder to
opt in. An explicit false remains an opt-out. To run without coding tools,
/plugins disable coder; to stop reading the instruction file,
/plugins disable repo_context. /plugins remove coder resets the
built-in to its defaults rather than removing it. A repository's
.clai/settings.json can declare plugins too; they show as (project), rank
just above the built-ins, and start off until the user runs /plugins enable
NAME, so repository code never runs without that approval. Outside a session,
clai2 plugins add NAME module[:Class] [JSON] saves for the next startup.
/plugins opens the management menu. Removing a drop-in disables it persistently;
delete its source file yourself to remove it from disk.

Inside a CLAI session only, /plugins add GIT_URL clones and loads a trusted
repository over HTTPS or SSH (or a local file:// URL). Its root __init__.py or
plugin.py must define one public Plugin subclass; dependencies must already be
installed in CLAI's environment. Checkouts live in plugins/_git/ID under the
configuration directory. Reload reads local code, not Git updates: pull into the
checkout with Git first, then /plugins reload ID. Removing a Git plugin forgets
its declaration but keeps its checkout; clear the printed checkout directory
before reinstalling. Plugin code runs as the user; install only trusted repositories.

The second built-in is ask_user (pydantic_clai2.builtin_plugins.ask_user_menu): the
harness AskUser capability with an inline numbered picker as its answerer, so
the model can ask the user multiple-choice questions mid-run through
ask_user_question. The conversation remains visible. Enter or a number selects;
for multiple selections it toggles, then Done submits. /plugins disable ask_user removes the tool. To answer the
questions somewhere other than the terminal, declare ask_user again with a
module whose Plugin returns AskUser(answerer=...) from get_capabilities with your
own async answerer; see PLUGINS.md.

The inline `ask_user_question` picker also offers `Other (type answer)`.
Choose it to type your own answer instead of the suggested options, including for
multi-select questions. Enter submits nonblank text. Esc returns to the choices
and keeps your draft; Ctrl-C declines the whole request. Backspace and arrow keys
edit the text. Multiline paste is inserted as text and waits for Enter; it does
not submit an answer or select choices. The conversation stays visible while you type. Custom answers
appear in the transcript and reach the model as a one-item list under the question's header.

The built-in observability plugin (pydantic_clai2.builtin_plugins.logfire) is enabled by default in the
stock CLI. It contributes core's Instrumentation capability using an isolated
Logfire instance. It exports to Logfire only when credentials are present, with
no interactive setup or console logging. Text and binary images are included by
default, so review the telemetry destination before setting LOGFIRE_TOKEN. Use
/plugins disable observability to remove it, or replace its settings with:

```text
/plugins add observability pydantic_clai2.builtin_plugins.logfire '{"include_content": false, "include_binary_content": false}'
```

Other options are service_name (default pydantic-clai2), send_to_logfire
(default "if-token-present", or false), token (the name of a /keys entry
holding a Logfire write token, as {"name": "CLAI2_LOGFIRE_TOKEN"}, whose project
then receives the telemetry), httpx (default false; set true to trace httpx and httpx2 requests process-wide, with headers and bodies captured when include_content is true), and ui_events (default true; set false to opt out: also record UI
interactions such as menus, commands, settings, plugin actions, keys, and prompt
submissions, by name; while include_content is on, a submitted prompt also
carries its text, but ! lines and slash-command arguments never do).
/plugins configure observability opens a settings menu that edits these options.
Its Logfire project row sets token and base_url for you, and turns sending on:
pick Logfire US, EU, or a self-hosted URL, sign in in the browser, and pick a
project; its new write token is saved in /keys. This explicit option overrides
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

Each plugin instance gets its own PluginHost: the console, conversation, status,
full screen, and its saved settings. Keep mutable state on the instance, set up
in __init__, not in module globals. Loading builds the instance, calls each get_*
method once, then on_session_start; unloading calls on_session_end and discards
everything the plugin declared. /plugins can change plugins during a turn: the
running agent keeps the capabilities it started with until the next prompt, and
a plugin unloaded mid-turn gets on_session_end when the turn ends. Failed or cancelled
loading calls on_session_end with reason=error under cancellation shielding, with
a five-second cooperative timeout, then discards the plugin. Errors and timeouts
are reported without replacing the load error. Blocking code and nested shields
can exceed the deadline. Cleanup can run before on_session_start finishes, so
guard it on what was acquired. Drop-in entry modules reload from fresh source;
installed modules use importlib.reload, which can retain globals absent from the
new source. Do not mutate another plugin or the agent to add a plugin's tools.

## Finding CLAI source

The package root keeps the shell entry point (`_app.py`) and the documented
plugin-author imports `pydantic_clai2.plugins` and `pydantic_clai2.commands`.
Implementations live in `cli/` (launch and command context), `config/` (settings
and storage), `runtime/` (sessions and reload), `models/` (catalog and provider
adapters), `ui/prompt/` (editor and terminal painting), `ui/menus/` (pickers),
`ui/rendering/` (themes and streamed output), `plugins/` (the `Plugin` base, host, and loader),
and `builtin_plugins/`. The MCP server implementation has its own `mcp/` package. UI helpers are internal: check
their current import paths before writing an extension. The package initializer
for `config/` still provides the `pydantic_clai2.config` settings types.

## Reload the shell during development

```text
/reload
```

Use `/reload` after editing `pydantic_clai2` itself, not just a plugin. It uses
`importlib.reload` and rebuilds the prompt loop, commands, and session without
restarting Python. Conversation history, the agent and its dependencies, selected
model, and active settings are kept. Enabled plugins unload and load again
against the refreshed shell types; disabled and unapproved project plugins stay
off. Plugin instances and their contributions are recreated, but installed module
globals not overwritten by the new source can survive. Initialize mutable state
in `__init__`. Failed imports or shell rebuilds restore previous module bindings
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

The four CLAI moments are async methods returning None:

| Method | Event | Use |
| --- | --- | --- |
| on_session_start | SessionStart(agent, settings) | Initialize session resources |
| on_session_end | SessionEnd(reason) | Clean up; reason is exit, eof, or error |
| on_turn_start | TurnStart(text) | Rewrite event.text or event.cancel() |
| on_turn_end | TurnEnd(text, outcome, result, error) | Observe completion or failure |

Import these event classes from pydantic_clai2.plugins. Event payloads are typed,
keyword-only dataclasses. A raising on_turn_start cancels the turn. Do not
return a replacement string or a boolean to decide a host action.

```python
from collections.abc import Sequence

from pydantic import BaseModel
from pydantic_ai.capabilities import AgentCapability, Capability
from pydantic_clai2.plugins import Plugin, TurnStart


class Options(BaseModel):
    prefix: str = 'Please answer concisely. '


class Concise(Plugin[Options]):
    async def on_turn_start(self, event: TurnStart) -> None:
        event.text = self.settings.prefix + event.text

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        tools = Capability[None](instructions='Use word_count for exact counts.')

        @tools.tool_plain
        def word_count(*, text: str) -> int:
            return len(text.split())

        return (tools,)
```

Pass options as JSON with /plugins add NAME module '{"prefix": "..."}'. Settings
are validated when the plugin loads; a plugin without a type parameter takes none.
get_capabilities may also return a RunContext-to-capability function returning a
capability or None, for example to read settings saved since the plugin loaded.

Hooks into the agent run are Pydantic AI capabilities: return a Hooks capability
(or your own AbstractCapability subclass) from get_capabilities, using Hooks().on
names and signatures. Check https://pydantic.dev/docs/ai/core-concepts/hooks/ and
installed source. For example, before_model_request returns its
ModelRequestContext; it is not a None-returning host observer. Do not add CLAI
moments for core behaviour. Raising before_tool_execute fails the run; consult
core's documented SkipToolExecution when only one tool execution should be skipped.

For typed capability events use hooks.on.event(EventClass) with a handler taking
RunContext and the event, or core's @on_event(EventClass) on a capability method.
Match on event classes rather than tool-name strings. If an event supports
cancel(), use its documented cancellation semantics.

## Settings that need a feature: requirement tags

Every CLAI on a machine shares one settings database (~/.config/pydantic-clai2/config.db),
whatever code it runs: other worktrees, branches, and installs. A plugin setting that one
build supports can break another. Version numbers cannot tell them apart, because worktrees
are diverging branches that report the same dev version. Feature names can.

### When a setting needs a tag

Tag a setting whenever its valid values or its meaning depend on code that other builds may
lack. Typical cases: a new value of an existing setting, a setting that only works with a new
code path, or an old setting whose meaning changed. A setting every build understands the same
way needs no tag.

Tag only settings whose default is the safe choice. A build without the feature drops the
saved value and uses the default, so dropping must never loosen a restriction.

### Name the feature and declare it

1. Pick a stable, descriptive name: lowercase words joined by hyphens, such as
   stock-bound-delegation. Name the capability, not the branch or ticket. Never rename or reuse
   a name; other builds compare names as plain strings.
2. Add it to SUPPORTED_FEATURES in pydantic_clai2/config/features.py, in the same change that
   adds the code behind it:

```python
SUPPORTED_FEATURES: frozenset[str] = frozenset({'stock-bound-delegation'})
```

### Tag the setting

A Plugin subclass declares tags where it reads its settings. Override from_host so
requirements are recorded before the plugin is built. Keys are the saved names,
aliases included:

```python
class Fancy(Plugin[Options]):
    @classmethod
    def from_host(cls, host):
        settings = host.settings(Options, requires={'mode': ['fancy-mode']})
        return cls(host, settings)
```

A capability class declared as module:Class has no Plugin subclass, so list its tags in
CAPABILITY_REQUIREMENTS in config/features.py, keyed by the factory string.

That is all. Writers attach the tags for you: host.save_settings, /plugins add, /plugins
enable and disable, and the settings menus all store them beside the declaration, in a separate
plugin_requirements table. Never put tags in the settings JSON or in PluginSettings: builds
before this feature pass settings to the plugin and validate declarations strictly, so an
unknown key would break their plugin loading.

### What each build does with a tag

- A build that lists every feature a setting needs applies it as saved.
- A build that lacks one, or does not recognize a name, ignores that one setting. It uses the
  shipped declaration's value (for a built-in) or the plugin's own default, keeps every other
  setting, and prints one line per plugin, such as
  coder: ignored saved sub_agents (needs stock-bound-delegation); using defaults.
- Reading never rewrites anything. When that build saves the plugin's settings, it writes the
  ignored values back unchanged, still tagged. A tag only goes away when its value changes.
- Builds older than requirement tags never read the table, so they apply every setting as
  before. Tags cannot protect them. The fail-soft layer below limits the damage in builds that
  have it.

### Worked example: Coder's sub_agents

A branch binds Coder delegation to its stock agent, so its users can save "sub_agents": true.
A build without that support passes Coder at run level, where SubAgents(include_self=True)
raises UserError on every turn. The branch with support:

```python
# config/features.py on the branch with delegation support
SUPPORTED_FEATURES: frozenset[str] = frozenset({'stock-bound-delegation'})
CAPABILITY_REQUIREMENTS = {
    'pydantic_ai_harness.coder:Coder': {'sub_agents': frozenset({'stock-bound-delegation'})},
}
```

Saving coder there stores {"sub_agents": ["stock-bound-delegation"]} for it. Another build
with tags but without the feature loads coder with the built-in "sub_agents": false and shows
the notice. A build older than tags still applies true; with fail-soft, only its first turn fails.

### Fail-soft at run setup

A setting that slips through untagged costs one turn and then one capability, not every
turn. CLAI guards each capability it builds itself from a module:Class declaration's saved
settings (such as coder), unless any part of it is a Hooks. Capabilities a plugin's get_capabilities
returns, and every policy hook, are never guarded. When a guarded one raises UserError while
the run is set up (in for_run, or in wrap_run before it hands over to the run), that turn fails
closed with the plugin named, and CLAI leaves the capability out of later turns until
/plugins reload. The turn is not retried, so no other capability's setup runs twice, and a
capability that refuses to run never gets skipped within the turn it refused.
Errors from the model, tools, or hooks once the run is under way propagate as before, and
Plugin handlers are never guarded, so a raising handler still fails closed.

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
from pydantic_ai import AgentStreamEvent
from pydantic_ai_harness.filesystem import FileWrittenEvent
from rich.console import RenderableType
from pydantic_clai2.plugins import Plugin


class Writes(Plugin):
    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        if isinstance(event, FileWrittenEvent):
            return f'wrote {event.path}'
        return None
```

Return a Rich renderable, or None to let the next renderer/default handle it.
CLAI flushes streaming text before printing it. First matching non-None renderer
wins. Do not print from an event observer when a renderer can do the job.
Use self.host.console for plugin-owned console output outside streaming handlers.
Resolve pydantic_clai2.ui.rendering.theme roles ACCENT, INFO, WARNING, ERROR, MUTED, THINKING
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

Add a fragment to the status row with get_status_segments:

```python
import os
from collections.abc import Sequence

from pydantic_clai2.plugins import Plugin
from pydantic_clai2.ui.rendering.status import StatusSegment


class Where(Plugin):
    def get_status_segments(self) -> Sequence[StatusSegment]:
        return (os.getcwd,)
```

The fragment is appended after the built-in figures and painted muted. The row
repaints about ten times a second, so it runs that often: no blocking IO, no
awaits, no printing. Return an empty string to contribute nothing that frame.
Fragments are truncated from the right on narrow terminals and cannot set their
own colours. get_status_segments adds to the row; replacing the row, prompt
editor, splash, or adding custom palettes is still a CLAI source change.

Offer a working animation with get_spinners; the user selects it with /spinner:

```python
from collections.abc import Sequence

from pydantic_clai2.plugins import Plugin
from pydantic_clai2.ui.rendering.spinners import Spinner, make_spinner


class Wave(Plugin):
    def get_spinners(self) -> Sequence[Spinner]:
        return (make_spinner('wave', ['~   ', ' ~  ', '  ~ ', '   ~'], interval=0.1, description='a small wave'),)
```

Frames are padded to one width and interval is clamped to 0.02-1 seconds. The
user's spinners.json replaces a plugin spinner of the same name, and an entry
without frames only retunes an existing spinner's interval or description.

## Custom TUI menus

Register an async slash command that builds and runs a Termflow MenuBuilder.
Keep the builder pure so tests can drive it without a terminal. The existing
plugin_menu.py, model_menu.py and field_menu.py are working source examples.
The following uses CLAI's internal UI helpers; check them when upgrading:

```python
from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu
from pydantic_clai2.ui.rendering._rendering import markdown_style
from pydantic_clai2.commands import Command
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.plugins import Plugin


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


async def show_menu(args: list[str]) -> str:
    await run_worker(lambda: build_menu().run())
    return ''


class MyMenu(Plugin):
    def get_commands(self) -> list[Command]:
        return [Command(name='my_menu', description='Open my menu', handler=show_menu)]
```

Termflow owns the alternate screen. Do not print while it is open. Put errors and
empty states in disabled rows or the preview. Use .on_key for single-key actions,
apply changes immediately, then replace_items to redraw. Esc and Ctrl-C should
close normally. menu_key and run_worker cooperate on cancellation, keeping the
terminal owned until the worker restores its screen. Do not fire-and-forget a
thread that is still reading input. If a menu key needs an async action, follow
plugin_menu.py's bridge back to the main event loop.

Pass during_turn=True to Command when the menu is safe to open mid-turn, so the
bare command opens at once instead of queueing behind the running turn. While
run_worker runs, CLAI leaves the live panel temporarily. The turn's output stays
in the transcript and paints when the menu closes.
Only opt in when the running turn cannot observe what the menu changes.
during_turn_subcommands=('add',) does the same for a bare subcommand such as
/model add. Pass args_during_turn=True when every form of the command is safe
mid-turn; it then runs at once with arguments too, ahead of queued follow-ups,
as /plugins does.

For named validated fields, reuse FieldSource, FieldMenu and run_flow in
field_menu.py rather than write another editor. SettingsSource in set_menu.py
shows the adapter; model_menu.py uses the same editor for model settings. Tests
inject Runners with scripted widget results (tests/menu_script.py). These are
internal shell helpers, not a promise of a stable third-party UI API. Plugins do
not currently receive CommandContext through PluginHost; do not invent host.context.

## Custom models and providers

A model identifier accepted by an existing core provider can be selected with
/model PROVIDER:NAME (or /model add PROVIDER:NAME, /set model PROVIDER:NAME) even
if it is absent from the catalog. A name not yet saved is added and selected
without checking that it exists; a wrong name fails on the next prompt with the
provider's error. Bare `/model` picks from previously added models, and its Tab
suggestions are the `add` and `settings` subcommands plus saved models.
In `/model`, Ctrl+D or Delete removes a saved model and its per-model settings
after confirmation. The current model and saved default are protected; select
another model or change the default with `/set model NAME` first. Provider
credentials and plugins are left alone.
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

CLAI attaches the launch directory as the workspace, so Coder() needs none of
its own. Here it restricts file tools to that directory, unlike the stock CLI's
unrestricted Coder. A launcher gets no built-in plugins unless it passes them.
Choose one source of coding tools, never both: either keep Coder() in
capabilities as above, or drop it from capabilities and call chat(agent,
deps=None, builtin_plugins=DEFAULT_PLUGINS) with DEFAULT_PLUGINS from
pydantic_clai2, which supplies the stock coder and ask_user plugins and lets
/plugins manage them. Passing both loads two sets of coding tools. A fully custom protocol belongs in a Pydantic AI Model and
Provider implementation, not a terminal plugin. See
https://pydantic.dev/docs/ai/models/overview/ and inspect installed core abstract
classes for required methods. Supply that Model instance to Agent as above.
To chat with an Agent instance you already have, skip the launcher and run
clai2 --agent MODULE:ATTR (or -a), for example
clai2 --agent pydantic_ai.main:my_cool_agent. CLAI appends the launch directory to
sys.path, after installed packages, so a module next to where you start CLAI
resolves without installing it. ATTR must
name an Agent instance, not a class or factory; the agent runs with deps=None.
For that session only, CLAI loads no plugins at all: no built-ins (so no stock
Coder or ask_user), no saved or drop-in user plugins, and no project plugins, and
/plugins reports that plugins are off. Nothing saved changes, so the next plain
clai2 loads plugins as before. The agent keeps its own model unless -m or
CLAI_MODEL selects another; a saved or project model does not replace it. -p runs
the same agent headlessly, --resume and --worktree work as usual, and --agent
cannot be combined with the config or plugins subcommands. A launcher is still the
way to pass deps, plugins, or builtin_plugins.

chat preserves a supplied agent's model when no settings override selects another
one. /model or /set model changes subsequent turns to the selected core model
identifier, not an alias for your custom instance. Noninteractive Session exposes
resolve_model for translating overrides; chat configures its own resolver for
Codex and CLAI's other connections, then for prefixes plugins register.

To offer a Model you build from a plugin, keeping the stock agent, Coder, and
every other plugin, register a prefix of your own:

```python
from collections.abc import Sequence

from pydantic_ai.models import Model
from pydantic_clai2.plugins import ModelProvider, Plugin


def resolve(name: str) -> Model:
    return MyModel(name, provider=MyProvider())  # your Model and Provider


class MyService(Plugin):
    def get_model_providers(self) -> Sequence[ModelProvider]:
        return (ModelProvider(prefix='my-service', resolve=resolve, models=('fast', 'smart')),)
```

`/model add` then lists my-service:fast and my-service:smart, and any
my-service:NAME works with /model add or /set model. resolve receives NAME
without the prefix and runs in a worker thread before every run with that
model, so it may read the keyring; raise UserError with setup instructions
when it cannot build the model. The prefix starts with a lowercase letter,
followed by lowercase letters, digits, and hyphens. A prefix Pydantic AI or
CLAI already runs, aliases like openai-chat included, is rejected with
ValueError.

`/effort` shows the active model's configured reasoning effort and supported values.
`/effort high` (or another listed value) saves it for that model; `/effort reset`
removes the native effort override. Custom parameters stay unchanged: remove any
that override effort with `/model settings` first. Models without an effort
control say so. Like `/fast`, `/effort` waits for the current turn to finish.

For `openai-codex` models, open `/model settings openai-codex:gpt-6-astra`
(or your saved Codex model), then **Service Tier / Fast Mode**. Choose
**Fast (priority)** to request fast processing, or **Standard (default)** to
turn it off. [Codex fast mode](https://developers.openai.com/codex/speed)
uses more ChatGPT credits and depends on model and account availability. It does
not lower reasoning effort. Reset restores the existing model default; it does
not enable fast mode. The stored values remain `service_tier=priority` and
`service_tier=default`, so older CLAI versions can read them. A custom
`service_tier` body parameter still takes precedence.

While using an `openai-codex:` model, `/fast` toggles priority processing;
`/fast on` and `/fast off` select explicitly. The service tier is saved for that
model's next prompts and sessions. Reasoning effort is unchanged. Other models
neither expose nor accept `/fast`. Remove a custom `service_tier` parameter with
`/model settings` before using `/fast`.

To extend the built-in picker in a CLAI source change, add a source returning
CatalogModel values in model_catalog.py and merge it in catalog(). Adding a
catalog row does not implement provider support. Editable per-model settings
are declared in ModelSettingsForm in model_settings.py; extend that form, not a
second editor. Credentials belong in provider-supported storage, not model
settings. /login signs in to subscriptions: /login openai-codex,
/login github-copilot, and any sign-in a plugin returns from get_logins.

## Test and verify

Construct PluginHost(name='example', console=Console(file=StringIO()), settings={})
and call load_plugin(MyPlugin, host) from pydantic_clai2.plugins: it builds the
plugin and collects its contributions as the loader does. Send events with
`await loaded.dispatch(event)`, run commands through `loaded.commands`, and pass
`loaded.capabilities` to an Agent. Check effects, not just contribution counts. Use synthetic events for renderers and Pydantic AI TestModel
for agent/tool integration; no real model or credentials are required. Use real
cancel scopes and Events, not sleeps, to order cancellation tests. Drive menu
builders headlessly, with injected runners instead of a real terminal.

In a CLAI checkout run Ruff format/check, strict Pyright on changed source/tests,
and focused pytest tests. Check /plugins list for loading errors, reload after
edits, and verify disable removes your commands and tools. Keep README.md and
PLUGINS.md aligned with user-facing API changes. For UI capabilities not exposed
by Plugin or PluginHost, state that limitation and propose a focused source change rather
than monkeypatching a private global registry.

## Headless invocation

`clai2 -p "PROMPT" -m PROVIDER:NAME` runs one saved turn and prints only the final
answer. Prompt text is required as an argument; stdin is not read. Use
`--resume SESSION-ID` to continue saved history without opening the browser.
The `ask_user` plugin is skipped without changing saved preferences. Full-screen
requests fail, stream renderers are not called, and host console output is
suppressed. Plugins must not read input or print directly to stdout. Errors go
to stderr with a nonzero exit status. `-m` also works in the interactive CLI.

## The stock agent from code

`open_stock_agent(workspace=..., model=..., capabilities=..., plugin_settings=...)` from
pydantic_clai2 is an async context manager that yields the CLI's stock agent without the
terminal, for agent.run, run_stream_events, iter, or harness run_acp_stdio. It binds the
coder, repo_context, and compaction built-ins, then `capabilities`, so delegated tasks carry
them. The tools work in `workspace` unless a passed capability supplies a workspace, such as
a sandbox. Model names resolve as in the CLI, with CLAI's per-model defaults.
`plugin_settings` merges settings over a built-in's stock ones, for example
{'coder': {'sub_agents': False, 'agent_folders': []}}. Saved, drop-in, and project plugins,
.clai/settings.json, saved model settings, and chain: fallback chains do not apply; nor do
ask_user, observability, mcp, or this guide. When approvals end the run, as over ACP, set
coder sub_agents to false: a delegated task's approval fails the run
(https://github.com/pydantic/pydantic-ai/issues/4302).

## Managed delegation UI

Interactive stock agents use harness `DelegationTasks`: `/tasks` inspects children,
Enter opens a full-width live transcript, `b` backgrounds, and `x` stops the selected
tree. Ctrl+B backgrounds foreground children. `/tasks resume ID` is explicit user
authorization to resume a general-purpose/custom child with its independent history.
Explore and Plan are read-only, inherit the selected model, and cannot resume.
Task reports are automated untrusted evidence, never user instructions or permission
grants. Supplied agents and headless runs retain their existing delegation behavior.
Background execution requires a local workspace; plugin changes wait for children
to settle. These are shell services, not additional `PluginHost` hooks.
