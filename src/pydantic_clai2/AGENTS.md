# CLAI 2 guide for AI code assistants

Read this before touching `pydantic-clai2/`. The repository-level `AGENTS.md`
still applies (no em-dashes, no `Any`, pyright strict, keyword-only arguments,
100% branch coverage). This file adds what is specific to the terminal shell.

## What CLAI is

A thin terminal around any Pydantic AI agent. It reads a prompt, runs the agent,
streams the answer, and repeats. Everything beyond that is a plugin, including
the default coding tools.

The three layers, and who owns what:

| Layer | Owns | Never does |
|---|---|---|
| Pydantic AI core | the agent loop, hooks, events, toolsets | know CLAI exists |
| `pydantic_ai_harness` | reusable capabilities (`Coder`, `Shell`, ...) | print to a terminal |
| `pydantic_clai2` | the prompt loop, rendering, `/commands`, plugin loading | reimplement a core hook |

If a change needs the agent loop to behave differently, propose it in core. If it
is a reusable behavior with no terminal in it, it belongs in harness. Only the
shell itself lives here.

## The plugin model

A plugin is a `Plugin` subclass, or a bare capability class. It is declarative
in the same way core's `AbstractCapability` is: every contribution is a method
with a default that contributes nothing, and a plugin overrides only what it
needs. `get_capabilities` for tools, instructions, and agent-run hooks (a `Hooks`
capability, or `@on_event` on your own capability); `get_commands` for
`/commands`; `render` for custom output; `get_status_segments`, `get_spinners`,
`get_model_providers`, and `get_side_panels` (pages the web UI shows beside the
chat); `configure` for a settings menu; `on_session_start`,
`on_session_end`, `on_turn_start`, `on_turn_end`, `on_conversation_changed`, `on_command_end`, `on_plugin_load_failed`, `on_task_progress`, `on_fork_end`, and `on_install_progress` for CLAI's own moments;
`prepare` for async reads the `get_*` methods depend on. The
settings model is the class's type parameter (`Plugin[Settings]`), validated into
`self.settings`. `self.host` is a `PluginHost`, the plugin's runtime context
(console, conversation, `session_id`, status, full screen, saved settings, the
prompt through `send_input`/`interrupt`/`complete`, the saved `sessions`, the
delegated `tasks`, and read-only `setting(KEY)`); it
registers nothing. `docs/declarative-plugins.md` records the design.
`PLUGINS.md` is the user contract. If code and `PLUGINS.md` disagree, fix one so
they agree in the same PR.

## Rules for the plugin API

- **Declare, do not register.** A plugin returns what it offers from `get_*`
  methods, which the loader calls once per load (`collect`), after awaiting
  `prepare()`. No registration calls, no module-level dicts, no import-time side
  effects. Tests call `load_plugin(PluginClass, host)`; it raises `TypeError` for
  a plugin that overrides `prepare`, whose tests call `from_host`, then
  `await plugin.prepare()`, then `collect`.
- **One method per moment, one typed event per method.** A handler never receives
  `*args`, `**kwargs`, `dict`, or `context: object = None`.
- **No string sub-dispatch.** A handler does not receive `event_type: str` and
  switch on it. Use the event class as the key.
- **One async spelling.** No `_sync` or `_async` pairs.
- **Observers return `None`.** Deciders mutate the event (`event.text = ...`,
  `event.cancel()`). Nothing collects a list of return values.
- **Fail closed, one way.** A raising handler on a decidable moment cancels the
  action and reports the error. No per-registration "fail open" flag.
- **Sign in only when the user asks.** Loading waits for every plugin and a run
  waits for its tools, so neither may open a browser or wait on a sign-in. Use
  `plugins/sign_in.py`: a `SignInMethod` (`mcp.OAuthSignIn` for MCP OAuth),
  `sign_in_now` from a menu or `/NAME login`, and no tools while signed out.
- **A built-in's per-run capability is a `WorkspacelessCapability`.** A bare
  capability function might return a sandbox, so `local_workspace()` attaches the
  launch directory only after every `for_run`, and harness `Skills` (which reads its
  libraries in `for_run`) fails the turn. Wrap a function that never returns a
  workspace in `runtime/session.py`'s `WorkspacelessCapability`.
- **CLAI moments are `on_<subject>_<moment>`** (`on_turn_start`). Core hooks stay
  core's, reached through a `Hooks` capability. Do not wrap a core hook in a
  plugin method.
- **Payloads are kw-only dataclasses.** Not `BaseModel`. Pydantic is for the
  settings JSON boundary only.
- **No `getattr`/`hasattr` on a plugin** to discover what it supports. It
  returned the thing from a `get_*` method or it did not; `has_configure` and
  `has_render` compare against the base-class default.
- **Only shell-owned moments.** `on_session_start`, `on_session_end`,
  `on_turn_start`, `on_turn_end`, `on_conversation_changed`, `on_command_end`, `on_plugin_load_failed`, `on_task_progress`,
  `on_fork_end`, `on_install_progress`.
  Startup load failures happen before an agent run, so core's `before_run` cannot
  observe them. The loader reports them after trying every enabled plugin so
  observability can receive failures that preceded its own load. Which conversation
  the shell holds, and its title, change between runs (`/new`, `/resume`, background
  naming); core only sees a `conversation_id` per run. `Session._publish` is the one
  place `ConversationChanged` is fired from. `/commands` never reach the agent, so no core
  hook sees them; the shell fires `CommandEnd` around `_execute_command`, outside the scope
  Esc cancels: a refused `/plugins` fires its refusal instead, and a cancelled command
  fires nothing. A delegated task's own events never enter the parent run's stream, and a
  background task outlives that run, so no hook of the parent sees them; `Tasks.observe`,
  the harness owner's one observer, fires `TaskProgress`. A `/fork` runs a separate session in the
  background, so its run's hooks belong to that session and nothing of the parent sees it end;
  `Forks._run` fires `ForkEnd`. `/update now` is a command, so no core
  hook sees its install either, and `CommandEnd` comes only once uv has finished; `Updates.command`
  fires `InstallProgress` as it goes. Additional moments need the same
  justification: say which core hook you checked and why it does not fit.

## Loading and unloading

Plugins load and unload while CLAI runs. The rules that make that safe:

- **One `LoadedPlugin` per plugin.** It holds the instance and what its `get_*`
  methods returned, so unloading is "discard this `LoadedPlugin`". No
  `callback -> owner` map, no scanning registries for a plugin's name.
- **A running agent run never changes its plugins.** Core binds capabilities once
  per run, so `/plugins` (an `args_during_turn` command) may load and unload
  mid-turn, but the run keeps the snapshot `run_turn` bound. `SessionPlugins.turn()`
  defers `on_session_end` of plugins unloaded meanwhile until the run is over,
  shielded from cancellation. Do not swap capabilities inside a run; that needs
  core support.
- **Plugins are per live session; declarations are per process.** Each live
  session's `SessionPlugins` (`plugins/loader.py`) holds its loaded instances,
  guards, and deferred ends, and is its `host.plugins`. They all join the
  process's `ProcessPlugins` (`plugins/process_plugins.py`, held by `ClaiProcess`):
  declarations, staged `/plugins add` settings, open settings menus, the change
  lock, and imports. `enable`, `disable`, `remove`, and `reload` run in the session
  they came from under that lock (`ProcessPlugins.change`), then every other live
  session `follow`s them, also when the origin failed; `/plugins add` and settings
  edits are followed once saved. Each session follows under the `turn()` rule above.
  `PluginLoader` is the former name of `SessionPlugins`, kept for tests; built with
  `store=` and no `process=`, it gets a process of its own, as one-session hosts do.
- **`Plugin.scope = 'process'` builds a plugin once per process.** `observability`,
  `workspaces`, and `web` are. The first session to load one builds it and fires
  `on_session_start`; it then joins every live session (capabilities, commands, and
  each moment, dispatched per session with `host.live.current()` naming it). It ends
  once, when unloaded (after every running turn that bound it) or when the last
  session closes; sessions load a name one at a time, so it is built once.
  Its host's `session_id`, `workspace`, `tasks`, `plugins`, `setting`, `send_input`,
  `submit_prompt`, and `interrupt` follow the session being handled (moments,
  commands, and runs set it); its other attributes are the loading session's.
  `restart` keeps it. A process plugin that includes a session plugin (`web` and
  `ask_user`) keeps it off in every session. `Plugin.scope`, `host.live`, and the
  `LiveSession`/`LiveSessions` protocols are public: `PLUGINS.md`'s "Several live
  sessions" is their contract, with the six rules every plugin follows. Change them
  as public API, and update that section in the same PR.
- **A process holds many live sessions; a conversation is live in one.**
  `ClaiProcess.live` (`runtime/live_runtimes.py`) is `host.live`. `create_shell`
  adds each `SessionRuntime` and `SessionRuntime.close` removes it; close a
  runtime with `close`, never with `loader.close` alone. `open` returns the session
  already holding a conversation, and `Session.resume` refuses one another session
  holds or is resuming (`LiveRuntimes.claim`). Sessions `open` and `new` start run as tasks of
  `serving`, in their own directory, with `Interrupts(signals=False)`, refuse
  `/reload` and `/update now` unless the terminal shows them (only that session, or the one
  CLAI started with, restarts it, never from another folder or while another session runs a
  turn, and a failed `/reload` continues the shown session's conversation), and close
  after `IDLE_AFTER` (an hour) with no turn, command, queued input, fork, task,
  watcher, or terminal focus. `/set` of anything but `model` and `run.*` reaches
  every live session (`CommandContext.adopt`) unless its project file sets it.
  There is no turn cap.
- **Per session on `SessionRuntime`, per process on `ClaiProcess`; never the other
  way.** `ClaiProcess` (`runtime/process.py`) holds what exists once: the stores,
  `ProcessPlugins`, the task owner and router, the namer, the Codex providers, and
  `live`. `SessionRuntime` (`runtime/session_runtime.py`) holds what each live session
  has: its `Session`, commands, `SessionPlugins`, status, interrupts, input queue,
  forks, and folder (`Session.workspace`, which plugins read as `host.workspace`).
  `Terminal` holds what only the terminal has, whichever session it shows.
  A per-session value on `ClaiProcess` leaks between sessions; a process value on a
  runtime is built twice or closes with one session. Module-level mutable state is
  process state too, so keep it off modules. Never call `os.chdir` or read
  `Path.cwd()` for a session's folder after startup: pass `Session.workspace` (or
  `host.workspace`). `Path.cwd` stays only as the default of hosts built outside the
  shell, and for `--worktree` setup and cleanup around the session.
- **The terminal shows one live session at a time.** `Terminal` (`ui/prompt/terminal.py`)
  owns the terminal console, the `LivePrompt`, the line prompt, the transcript, input
  history, and pasted images. Only the session shown (`SessionRuntime.terminal`,
  `LiveRuntimes.focused`) reads the editor, prints to the console, and takes Ctrl-C
  (`Interrupts.focus`); the others print nowhere (`Terminal.session_console`) and read
  their inbox. `chat` sets `live.build`, so the web UI opens sessions beside the
  terminal's. `/resume` of a conversation another live session holds shows that one
  (`Sessions.show_live`), mid-turn too, and never forks it; the one shown before stays
  live, and the session CLAI started with is closed for idleness like any other while
  hidden. Prompts queued for a session stay its own when the terminal shows another.
  `clai2 --web` keeps its terminal on the session it started with. With one live
  session nothing changes, and the toolbar adds `N live` only for more.
- **Load runs `on_session_start` for that plugin; unload runs `on_session_end`.** A
  plugin cannot tell whether it was loaded at startup or later, and must not
  need to.
- **`reload` is unload, re-import, load.** Drop-in entry modules use fresh source.
  Installed modules use `importlib.reload`, which retains globals absent from the
  new source. Plugins keep their state on the instance, set up in `__init__`.
- **Instruction order is capability order.** Placement is a core
  `CapabilityOrdering` (`position`, `wraps`, `wrapped_by`), not a CLAI list.
- **Registration is idempotent per name.** "Active for the next prompt" is the
  natural unit. Stock agents are rebuilt when the capability snapshot or CLAI's
  unchosen default model changes. Plugins are bound at construction so
  self-delegation carries their tools, instructions, and guardrails, and the
  default model is bound as the agent's own so a capability can replace it.
  Supplied agents still receive plugins per run (`agent.run(capabilities=...)`)
  and are never rebuilt.
- **Shipped plugins register first, in declared order.** The menu's alphabetical
  order is for scanning only. Registration order is the order instructions,
  renderers, and status segments are consulted in, so `coder`'s guidance leads
  the prompt.
- **Integrations are deferred.** Only the coding loop and run plumbing are eager.
  The customization guide (`clai_customization`) and every hosted integration
  plugin pass `defer_loading=True` on the capability they build (on the one a
  `for_run` function returns, never on the `DynamicCapability`), so each costs one
  catalog line until the model calls `load_capability`. A new integration does the same.
- **Built-ins are declarations, not code paths.** `DEFAULT_PLUGINS` in
  `config/builtins.py` lists the library's built-ins (re-exported by `app.py`,
  kept apart so `clai2 plugins` can toggle one without importing the shell); `STOCK_PLUGINS` opts the CLI-owned
  agent into delegation without changing supplied-agent defaults. CLAI ships enabled (`coder`, `ask_user`, `repo_context`,
  `compaction`, `persistence`, `observability`, `hooks`). The loader treats them like drop-ins with the lowest
  precedence: a store declaration with the same id replaces one, `disable`
  persists an override, `remove` resets it. Do not special-case `Coder`
  anywhere else; the agent from `create_agent()` has no coding tools of its
  own. `coder` is declared with `repo_context: false` because `repo_context`
  binds harness `RepoContext` itself; keep it that way or `AGENTS.md` reaches
  the model twice. When a built-in takes the id of a row the former harness
  catalog offered, add that old row to `RETIRED_BUILTINS` in `plugins/process_plugins.py`, so a
  user's saved toggle of it maps to the built-in instead of outranking it.
- **`open_stock_agent` loads `coder`, `repo_context`, and `compaction` only.** Add a
  built-in to that list only when it needs no terminal and reads nothing from the
  user's CLAI configuration directory. Hosts rely on a configuration that the user's
  saved settings cannot change.
- **Project declarations rank just above built-ins and start off.**
  `.clai/settings.json` (`project_settings.py`) may declare plugins; the loader
  takes them as `project=`, every one `enabled=False`, because a repository
  must not run code as the user on launch. `/plugins enable` is the approval
  and persists the approved declaration in the store. Precedence is store,
  drop-in folder, project, built-in. CLAI never writes the project file.
  The exception is a factory in `APPROVED_AS_WRITTEN` (`plugins/process_plugins.py`),
  such as the `hooks` table's `project_hooks`: its declaration is what runs, so
  a stored approval with the same id and factory that differs from the
  project's is stale and the project's, off, replaces it. A drop-in or a user's
  own plugin under that id is never replaced.
  Default-on CLI `.env` loading is a deliberate exception to this repository-content
  trust rule: it assumes trusted launch and parent directories, without plugin
  approval. [Environment variables](docs/reference.md#environment-variables) documents the
  scope and opt-out.
- **A load failure leaves the session as it was.** Import, construction, or
  `get_*` errors are reported and the plugin stays unloaded; nothing it declared
  is kept. When construction succeeded, `on_session_end` runs with
  `reason='error'` under a shield with a five-second cooperative timeout before a
  failed/cancelled load drops the plugin. Cleanup must tolerate an incomplete
  `on_session_start`.

**Overlapping plugins go in the compatibility matrix.** `plugins/compatibility.py`
maps a plugin's factory to the factories it already includes, each with the
capability classes it must bind for that to hold (`coder` includes harness
`SubAgents` while it binds one, which it does with `sub_agents` on, and
`compaction` only while it binds a history compaction, which it does not, so
both run by default). While the including plugin is loaded, the
loader keeps the included ones unloaded and refuses to enable or configure them, and `/plugins`
greys their rows out. Their saved `enabled` flag is untouched, so turning the
including plugin off loads them again. Add a row there; do not special-case ids.

Compaction registers harness `FallbackCompaction` directly with `max_fraction`
and `context_window` for both strategies. Harness owns the trigger; do not add
threshold math or an orchestrator in CLAI. Register the usage gauge after the
chain so yellow means the compacted request still exceeds the threshold.
`/compact` drives the same chain regardless of threshold, keeps the result only
when harness `estimate_message_tokens` (provider-reported usage) says it shrank,
and passes `runtime/agent_instrumentation.tracer()` so it is traced. Only `ModelAPIError`,
`FallbackExceptionGroup`, and `UsageLimitExceeded` select truncation after a
summary failure; other exceptions propagate. Its `configure` is a `FieldMenu`
over `CompactionSettings`; the loader's reload builds the new chain. Background
compaction is harness `FallbackCompaction(background_start=...)`, not CLAI code: the plugin
only passes the setting, closes the chain in `on_session_end`, discards the summaries of
conversations other than the shown one and its running tasks at each turn end and conversation
change (not per task event, which would build one for every streamed event), and has `/compact` try the
chain's `take_background_compaction` (without a focus) or `discard_background_compaction`
(with one) before the capped chain. The status
row's window does not depend on it: `Session.on_context_window` reports each
streamed request model's window, and the shell fills it in when no gauge has.
Until a turn on the selected model runs in the conversation, `show_model` sets
`Status.model_window` from its profile (`infer_model_profile`; a plugin model's from
its `settings_from` provider's), which the row and web UI show while `context_window`
is `None`; after one, an unresolved window stays `?`. A conversation without a figure
yet, such as a resumed one, reads its last response's `context_tokens`.

## Adding or changing a CLAI moment or contribution

1. Add the event dataclass and an `on_*` method (or a `get_*` method with a
   no-op default) to `Plugin`, and route it in `_HANDLERS` or `collect`.
2. Fire it, or read the collected value, from exactly one place in the shell.
3. Document it in `PLUGINS.md` in the table it belongs to, and in
   `docs/declarative-plugins.md` if it changes the design.

## The `/plugins`, `/set`, `/theme`, `/model`, and `/model add` menus

Built on termflow's `MenuBuilder` (and `TextInputBuilder` for typed values),
exactly like Code Puppy's `/agent`, `/mcp`, `/set`, and `/model` menus:
alternate screen, a `.preview` panel on the right, `markdown_style()` for
colours. Single-key actions and the key legend go through `slash_search`'s
`hotkeys` and `footer` in a searchable menu, `.on_key` and `.footer_hint` in
one that is not.

- Split it in two: a pure `build_plugins_menu(...)` that returns the menu (so
  tests drive it headless, no terminal), and a thin async runner that owns the
  screen and calls `menu.run` in a thread.
- Every key mutates immediately and `replace_items` redraws. No pending-changes
  state, no save/cancel pair. Settings menus still end with a **Save & close**
  row (`save_and_close_item()`, recognized by `picked(result)`) that only
  leaves; `FieldMenu`, `/keys`, and `/plugins` add it for you.
- Turning a plugin on opens its `configure` menu. A widget cannot open
  from inside another menu's key handler, so `PluginMenu` closes with a
  `Configure` result and `open_plugins_menu` runs the settings menu, then
  reopens the list.
- Enter performs the highlighted row's primary action (`/plugins` toggles, like
  Space) and never silently closes a list menu. Termflow's default Enter selects
  the row and closes, so where that is not the row's action, bind `enter` as a
  hotkey that does the action, or nothing; Esc closes.
- Footer hints separate actions with ` - ` and capitalize named keys (`Enter`,
  `Esc`, `Space`); letter hotkeys stay as typed (`c configure`, `R reset`).
- Nothing prints to the console while the menu is open; the alternate screen
  would hide it. Show empty states and errors inside the menu as disabled rows.
- Esc and Ctrl-C close cleanly. They are not errors.
- Make a new or changed menu searchable with `slash_search`
  (`ui/menus/slash_search.py`), not a bare `.searchable()`: plain letters stay
  hotkeys, `/` starts a search, Esc leaves it. Pass single-key actions as its
  `hotkeys` so they type into the search instead of firing while the user
  searches.
- A widget opened mid-run (including the inline `ask_user` picker) goes inside
  `async with host.full_screen()`, which flushes streamed text and suspends the
  editor's input reader first, preserving its draft. Slash-command handlers
  already run with the editor suspended. Do not start a second input reader
  alongside the live editor.
- Adding a plugin is not in the menu. It needs free text, so it stays
  `/plugins add ID MODULE[:ATTR] [JSON]` or `/plugins add GIT_URL`. Git installs
  use the existing file loader, with HTTPS/SSH for network sources; see
  `PLUGINS.md` for the repository layout, trust boundary, and checkout management.
- Anything that is "edit named, validated fields" uses `field_menu.py`: a
  `FieldSource` supplies rows, current values, validation, apply, and reset;
  `FieldMenu` builds the widgets; `run_flow` is the loop. `/set` and per-model
  settings are two sources, not two editors. Do not write a third editor.
- `/set` edits go through `CommandContext.set_setting` / `reset_setting`, the
  same path as the typed command, so validation lives in one place.
- Multi-line text goes through `text_editor.edit_text`: `$VISUAL` or `$EDITOR`
  on a temporary file, else (on Windows, unset, or failing to start) a built-in
  prompt_toolkit editor, since termflow has no multi-line input. Call it from a
  menu worker between widgets, or ask with `Dialogs.text(multiline=True)` from a
  settings menu action, which the web UI shows as a text area. Do not write another editor.
- Widget runners are a `Runners` value passed into the loops; tests pass
  scripted ones (`tests/menu_script.py`). Only the real `widget.run()`
  one-liners are `no cover`. Code that asks through `Dialogs` (settings menu
  actions, `plugins/keys.py`, `plugins/sign_in.py`) is tested with
  `tests/clai2/dialog_script.py`'s `ScriptedDialogs` instead.
- Model sources live in `model_catalog.py`. To add one (models.dev, a provider
  API), write a function returning `CatalogModel`s and merge it in `catalog()`.
  The menu never talks to a source directly.
- Per-model settings are the editable subset of core's `ModelSettings`,
  declared once as `ModelSettingsForm` with descriptions and bounds. Extend the
  form, not the menu, to expose another setting.

## Rendering

`StreamRenderer` owns text and thinking. It knows nothing about any specific
tool. Tool-specific output (shell previews, diffs, grep) comes from `render` on
the plugin that owns the event. Match on event classes, never on `tool_name`
strings. Always flush the stream before printing anything else; the shell does
this for renderers, so do not call `console.print` from an event hook when a
renderer would do.

## Colours

`/theme` offers the unchanged `default` appearance and `termflow.themes.PALETTES`.
Do not define new palettes. Resolve brand roles with `theme.color(...)` for Rich;
`theme.sgr(...)` resolves raw ANSI itself. A theme change repaints retained
transcript lines by translating these roles, so colour output through them,
and print logo branding inside `theme.branded()` so it keeps its colours.
`theme.current()` returns a Termflow palette or `None` for the original appearance. Termflow owns palette application
and reset; `theme.use(...)` leaves the terminal untouched in the default session.
Markdown keeps its original style by default and uses `to_render_style()` for a
selected palette. The preview renders a sample without OSC changes or persistence.
Heavy imports in `theme.py` stay lazy for the splash. Code uses the terminal
foreground and ANSI syntax colours through `theme.syntax_theme()`, shared by
streamed fences, shell command headers, and theme previews. Shell command headers
add `dim` and drop underlines over the lexed code (`command_syntax._SECONDARY`), so
a tool call reads as secondary; keep fences unchanged. The default theme's diff lines use Claude Code's
green and red; bundled palettes get diff lines from `theme.diff_renderer()`, tinted
from the palette.

## Source layout

The package root composes the shell (`app.py`, `__main__.py`) and keeps the
plugin-author imports `pydantic_clai2.plugins` and `pydantic_clai2.commands`
stable. Other code is grouped by responsibility:

- `cli/` owns argument handling, command context, headless output, and shell passthrough.
- `config/` owns settings, credential storage, and project declarations. `config/__init__.py` provides the existing `pydantic_clai2.config` settings API.
- `runtime/` owns sessions, forks, worktrees, reload, and speculative turns.
- `models/` owns model discovery, settings validation, and optional provider adapters. Pure model parameter expansion belongs here, not in a menu.
- `ui/prompt/` owns editing, terminal input, surface painting, and resize.
- `ui/menus/` owns widgets and command-specific pickers; `ui/rendering/` owns themes, streaming, status, and output formatting.
- `plugins/` owns the host API, loading, and credential picker helpers. `builtin_plugins/` and `mcp/` retain their plugin and server boundaries.

Move a capability to core or harness if it does not need a terminal. Keep UI
imports out of reusable validation code. The CLI and app may compose these
packages; new package initializers should not eagerly import the app or heavy UI.
Do not add broad backwards-compatibility shim modules for internal file moves.
Keep documented plugin-author paths (`pydantic_clai2.plugins` and
`pydantic_clai2.commands`) intact; update example imports when moving UI helpers.

## File map

`FILE_MAP.md` lists what each module holds. Read it to find where a change belongs,
and update its row when you add, move, or change the responsibility of a module. It
is kept out of this file because this file is sent with every request.

Keep files concise - we don't need any 10,000 line files. Single responsibility.

## Windows-only CLI operations

Check whether an existing interactive command already completes the operation on Windows.
If a new non-interactive path cannot observe completion because CLAI must exit first,
fail early with a non-zero status and name the working command. Do not report success
for a detached operation or add a process trampoline just to make the new path work.
Add Windows-specific orchestration only when a concrete requirement cannot use the
existing path. For updates, `/update now` is the working interactive path.

## Testing

- When adding or modifying a CLAI2 CLI UX feature, test every affected UX feature
  with your changes in a fresh tmux window before opening a PR. Run CLAI2 with
  `uv run clai2`. Verify the actual terminal behavior against the intention of
  the user's request, not just the implementation or automated tests.
- `pytest-anyio`; real model calls are blocked globally.
- Drive the shell with `TestModel` and a `Console(file=StringIO())`.
- Test a moment by loading the plugin with `load_plugin` and firing the event
  from the shell path that owns it (or `LoadedPlugin.dispatch`). Assert the handler's effect (the
  cancelled turn, the rewritten text), not a mock call count.
- Renderers get synthetic events. They must not need `Coder` installed.
- Cancellation: use a real `anyio` cancel scope, order with `Event`s, no sleeps.

### Settings and database compatibility

Before adding, removing, or renaming a setting, changing its type, meaning, or
default, or changing the SQL schema, consider upgrades, downgrades, and branch
switches. Different versions can share the same settings database.

- Adding a setting needs no API compatibility waiver: `SETTING_FIELDS` and
  `STRING_SETTINGS` are `growing_collections` in `.github/api-compatibility-allowlist.json`.
  Removing or renaming one still needs a deprecation and a waiver.
- Add regression cases to `tests/test_settings_compatibility.py` and affected CLI
  tests using the previous stored format. Keep historical fixtures unchanged;
  do not regenerate them with current models or rewrite them to make a change pass.
- Verify older databases load with documented defaults for missing settings and
  preserve existing preferences, plugin declarations, and model settings. Test
  migrations and repeated initialization for data preservation and idempotence.
- Preserve unknown saved setting names and their values when reading, editing
  other settings, resetting, and reopening. Unknown does not mean obsolete.
  Keep validation strict for known values, new writes, and plugin declarations.
- For renamed fields or changed types, meanings, or defaults, define the migration
  or intentional behavior change and test it explicitly. Comparing only against
  the current `Settings()` defaults will not catch an unintended default change.
- Verify rejected values and unsupported schema versions leave stored data intact.
  If a change introduces a migration, test failure rollback as well as success.
- When a plugin setting's valid values or meaning depend on code other builds may
  lack, add a feature name to `SUPPORTED_FEATURES` in `config/features.py` and tag
  the setting (`host.settings(Model, requires=...)`, or `CAPABILITY_REQUIREMENTS`
  for a `module:Class` capability). Tags live in the `plugin_requirements` table;
  never add a field to `PluginSettings`, a key to its `settings`, or bump
  `user_version`, since older builds reject all three. A build lacking a feature
  drops only that setting and uses the default. Run setup `UserError`s are caught
  only by `PluginGuard`, and only around a capability CLAI built from a
  `module:Class` declaration's settings with no `Hooks` inside, so policy hooks
  and plugin-contributed capabilities always fail closed. See "Settings that need a feature" in `customization.md`.

## Local verification

Run from the repository root. CLAI shares the root `uv.lock`, `.venv`, and
Pyright configuration with Harness.

```bash
uv sync --locked --all-packages --group lint
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run --no-sync pyright pydantic-clai2/src pydantic-clai2/tests
uv run --no-sync pytest -p no:cacheprovider -c pydantic-clai2/pyproject.toml pydantic-clai2/tests
```

## Docs parity

`README.md` is the product tour, `docs/reference.md` is the user reference,
`PLUGINS.md` is the plugin contract, and this file is for agents. A user-facing
change updates the reference (and the README when it changes what the tour shows)
and `PLUGINS.md` in the same PR. The README is the PyPI page, so its links and
images are absolute URLs on `main`; `scripts/screenshots/capture.sh` regenerates
the images in `docs/images/`. Plain
language, short sentences, no jargon a first-time plugin author would not know.

The resume browser is a focused exception to the single `MenuBuilder` convention:
it needs independently focused projects/sessions and two-line cards. Keep its
frame pure, inject keys/size/IO, use Termflow terminal ownership and layout, and
run through `menu_worker`. Metadata refreshes must preserve selection by ID.
History-changing plugins use `await conversation.commit_messages`, not the legacy
in-memory `replace_messages`, so exiting immediately after `/compact` is durable.

The interactive editor owns its layout explicitly. Do not reintroduce a
PromptSession renderer. `PromptSurface` composes a termflow.live `ScreenBuffer`,
`TranscriptView` draws its transcript region, and `render_diff` paints changed
cells. Do not invent a parallel canvas. The hardware cursor stays hidden;
the input cursor is a painted reverse-video cell. Keep terminal mutations in
`PromptSurface`, and detach the key reader before a menu owns the screen.
The prompt-toolkit decoder preserves paste, modified keys, and SGR mouse reports.
PageUp/PageDown and wheel input scroll the transcript, not the draft.
Wheel and drag reports arrive in bursts, so they paint once after the input
callback (`PromptSurface._paint_soon`), never once per report.
An idle session must cost next to no CPU, since users keep dozens open. The
editor's timed repaint (`LivePrompt.tick`) runs ten times a second only while a
turn runs or the panel animates, and once a second (`IDLE_REPAINT`) otherwise,
sleeping in between; a read wakes it, since the work it starts animates.
`PromptSurface.paint` skips a frame identical to the last one. Input, output,
and resizes paint at once; output inside a frame interval paints when it ends
(`PromptSurface.changed`), without waiting for a tick. Do not add a timer that
repaints unconditionally, or one that wakes an idle editor more than once a second.
Mouse reporting stops most terminals from selecting text, so a left-button drag
selects painted cells in `PromptSurface` and its release copies them through
`text_clipboard.copy_text`. Do not drop the drag modes (`?1002h`) or copy-out breaks.

Resize rebuilds from `TranscriptBuffer`, not guessed row coordinates or cursor
reports. Never send erase-scrollback (CSI 3 J). Preserve the draft and scroll
anchor. `SIGWINCH` invalidates the next frame; it must not perform terminal IO.
Full-screen menus leave the live panel temporarily. Inline questions borrow it
with `run_worker(inline=True)` and pin the question text in their rows, not the
transcript, so output streamed meanwhile cannot push it away. Streamed text and thinking keep Markdown source,
a tool-call group its call names, and a shell call's header its command, for width/theme repaint
(syntax colours share hex values with brand roles under a palette, so translating them would guess);
tool output keeps styled lines, each tagged with the
theme that painted it, and `recolor.py` translates them role by role (`theme.roles`)
when the theme changes. On exit, `restore`
prints retained output into native scrollback once, skipping startup output
already printed there. Transcript memory is bounded, including Markdown parts.
Reload rebinds the retained transcript and its nested classes in place after a
successful shell rebuild, including migration from the pre-live Text-line buffer.
The suspended `chat` coroutine still holds that transcript; do not only replace
it on the new shell or mutate it before a rebuild that can fail.

The `ask_user` picker borrows the released `PromptSurface` while the editor is
suspended. Keep its numbered choices and Enter toggles; do not switch screens
inside the picker or reintroduce Space-to-toggle.
