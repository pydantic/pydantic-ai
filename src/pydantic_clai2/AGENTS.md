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
and `get_model_providers`; `configure` for a settings menu; `on_session_start`,
`on_session_end`, `on_turn_start`, `on_turn_end`, and `on_plugin_load_failed` for CLAI's own moments. The
settings model is the class's type parameter (`Plugin[Settings]`), validated into
`self.settings`. `self.host` is a `PluginHost`, the plugin's runtime context
(console, conversation, `session_id`, status, full screen, saved settings); it
registers nothing. `docs/declarative-plugins.md` records the design.
`PLUGINS.md` is the user contract. If code and `PLUGINS.md` disagree, fix one so
they agree in the same PR.

## Rules for the plugin API

- **Declare, do not register.** A plugin returns what it offers from `get_*`
  methods, which the loader calls once per load (`collect`). No registration
  calls, no module-level dicts, no import-time side effects. Tests call
  `load_plugin(PluginClass, host)`.
- **One method per moment, one typed event per method.** A handler never receives
  `*args`, `**kwargs`, `dict`, or `context: object = None`.
- **No string sub-dispatch.** A handler does not receive `event_type: str` and
  switch on it. Use the event class as the key.
- **One async spelling.** No `_sync` or `_async` pairs.
- **Observers return `None`.** Deciders mutate the event (`event.text = ...`,
  `event.cancel()`). Nothing collects a list of return values.
- **Fail closed, one way.** A raising handler on a decidable moment cancels the
  action and reports the error. No per-registration "fail open" flag.
- **CLAI moments are `on_<subject>_<moment>`** (`on_turn_start`). Core hooks stay
  core's, reached through a `Hooks` capability. Do not wrap a core hook in a
  plugin method.
- **Payloads are kw-only dataclasses.** Not `BaseModel`. Pydantic is for the
  settings JSON boundary only.
- **No `getattr`/`hasattr` on a plugin** to discover what it supports. It
  returned the thing from a `get_*` method or it did not; `has_configure` and
  `has_render` compare against the base-class default.
- **Only shell-owned moments.** `on_session_start`, `on_session_end`,
  `on_turn_start`, `on_turn_end`, `on_plugin_load_failed`. Startup load failures
  happen before an agent run, so core's `before_run` cannot observe them. The
  loader reports them after trying every enabled plugin so observability can
  receive failures that preceded its own load. Additional moments need the same
  justification: say which core hook you checked and why it does not fit.

## Loading and unloading

Plugins load and unload while CLAI runs. The rules that make that safe:

- **One `LoadedPlugin` per plugin.** It holds the instance and what its `get_*`
  methods returned, so unloading is "discard this `LoadedPlugin`". No
  `callback -> owner` map, no scanning registries for a plugin's name.
- **A running agent run never changes its plugins.** Core binds capabilities once
  per run, so `/plugins` (the one `args_during_turn` command) may load and unload
  mid-turn, but the run keeps the snapshot `run_turn` bound. `PluginLoader.turn()`
  defers `on_session_end` of plugins unloaded meanwhile until the run is over,
  shielded from cancellation. Do not swap capabilities inside a run; that needs
  core support.
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
  the prompt. `customization_guide()` orders itself after the guidance plugins
  contribute and before harness `RepoContext`, so the CLAI hint never leads.
- **Built-ins are declarations, not code paths.** `DEFAULT_PLUGINS` in
  `_app.py` lists the library's built-ins; `STOCK_PLUGINS` opts the CLI-owned
  agent into delegation without changing supplied-agent defaults. CLAI ships enabled (`coder`, `ask_user`, `repo_context`,
  `compaction`, `persistence`, `observability`). The loader treats them like drop-ins with the lowest
  precedence: a store declaration with the same id replaces one, `disable`
  persists an override, `remove` resets it. Do not special-case `Coder`
  anywhere else; the agent from `create_agent()` has no coding tools of its
  own. `coder` is declared with `repo_context: false` because `repo_context`
  binds harness `RepoContext` itself; keep it that way or `AGENTS.md` reaches
  the model twice. When a built-in takes the id of a row the former harness
  catalog offered, add that old row to `_RETIRED_BUILTINS` in `plugins/loader.py`, so a
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
  Default-on CLI `.env` loading is a deliberate exception to this repository-content
  trust rule: it assumes trusted launch and parent directories, without plugin
  approval. [Environment variables](README.md#environment-variables) documents the
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
`/compact` drives the same chain regardless of threshold. Only `ModelAPIError`,
`FallbackExceptionGroup`, and `UsageLimitExceeded` select truncation after a
summary failure; other exceptions propagate. Its `configure` is a `FieldMenu`
over `CompactionSettings`; the loader's reload builds the new chain. The status
row's window does not depend on it: `Session.on_context_window` reports each
streamed request model's window, and the shell fills it in when no gauge has.

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
  menu worker between widgets. Do not write another editor.
- Widget runners are a `Runners` value passed into the loops; tests pass
  scripted ones (`tests/menu_script.py`). Only the real `widget.run()`
  one-liners are `no cover`.
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
streamed fences and theme previews. The default theme's diff lines use Claude Code's
green and red; bundled palettes get diff lines from `theme.diff_renderer()`, tinted
from the palette.

## Source layout

The package root composes the shell (`_app.py`, `__main__.py`) and keeps the
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

| File | Holds |
|---|---|
| `cli/_cli.py` | argument parsing, startup, `--agent` |
| `cli/agent_import.py` | resolves `--agent MODULE:ATTR` to an agent instance |
| `cli/self_update.py` | `/update` and the status-row notice: PyPI (`stable`) or the `clai2-bleeding` GitHub release (`main`: sdists built from the `main` branch by `.github/workflows/clai2-bleeding.yml` with `scripts/build_bleeding.sh`, installed with `--overrides`, no git or GitHub API; `CLAI_BLEEDING_URL` points it elsewhere), reinstalled with `uv tool install --force` |
| `_app.py` | the prompt loop, built-in `/commands`, and `open_stock_agent`, the stock agent for code outside the terminal |
| `runtime/_session.py` | conversation state, revision-checked saves, restore-only resume (a session busy in another process resumes as a saved fork), plugin snapshots and stock-agent rebuilding |
| `runtime/sessions.py` | resume command and background namer ownership; built-in step capture |
| `runtime/imported_sessions.py` | Claude Code and Codex sessions for `/resume`, `--resume-claude`, and `--resume-codex`: the on-disk catalog, stable CLAI IDs, and copying into the store (re-reading an uncontinued copy on every resume) |
| `runtime/claude_code_sessions.py`, `runtime/codex_sessions.py` | read each agent's JSON Lines transcripts: headers for the browser, and the history to continue (the last Claude Code branch, Codex's compacted history) |
| `runtime/imported_history.py` | shared transcript reading and `HistoryBuilder`, which groups parts into requests and responses; core's `repair_messages` pairs the tool calls |
| `runtime/session_naming.py` | resume-browser naming prompt, `SessionName` card schema, and the bounded `SessionNamer` worker |
| `runtime/forks.py` | `/fork` and `/forks`: history snapshot, background child sessions, deferred fork output |
| `ui/menus/session_browser.py` | project/session browser using Termflow layout and terminal primitives |
| `ui/menus/rewind.py` | double-Esc rewind picker: run boundaries, compaction guard, and durable history replacement before draft restoration |
| `ui/rendering/_rendering.py` | streaming Markdown and thinking |
| `ui/rendering/tool_group.py` | `display.tool_calls = grouped`: `ToolCallGroup`, one live line counting consecutive calls by tool, kept as its own transcript item in the live prompt; `finish()` closes it, `_drain()` does not |
| `ui/menus/tool_calls_preview.py` | the sample of each `display.tool_calls` style shown in the `/set` choice picker (`FieldRow.preview`) |
| `plugins/__init__.py` | `Plugin`, `PluginHost`, `LoadedPlugin`/`collect`, event dataclasses |
| `plugins/_factories.py` | resolving a declaration's `factory` to a `Plugin` (module, `module:Class`, capability class) |
| `plugins/loader.py` | discovery, load, unload, reload; the `/plugins` subcommands |
| `plugins/compatibility.py` | the compatibility matrix: which plugins another plugin already includes |
| `ui/menus/plugin_menu.py` | the `/plugins` full-screen menu (`PluginMenu` plus its runner) |
| `ui/menus/slash_search.py` | `slash_search`: plain-letter hotkeys plus `/` to search, for any termflow menu |
| `plugins/describe.py` | a plugin's description from its docstring, parsed with `ast`, never imported |
| `plugins/_git.py` | Git installation for `/plugins add GIT_URL`, with checkout rollback until its declaration is saved |
| `runtime/_processes.py` | process-tree cleanup shared by shell passthrough and Git cloning |
| `builtin_plugins/ask_user_menu.py` | the built-in `ask_user` plugin: `QuestionMenu`, `TerminalAnswerer`, the transcript renderer |
| `ui/prompt/screen.py` | `Screen`, what `host.full_screen()` binds to during a prompt |
| `ui/menus/field_menu.py` | the shared field editor (`FieldSource`, `FieldMenu`, `Runners`, `run_flow`) |
| `ui/menus/set_menu.py` | `/set`: `SettingsSource` over `CommandContext` |
| `ui/menus/system_prompt_menu.py` | `/system_prompt`: the user's `run.instructions`, which `Session` sends after the agent's and plugins' instructions, and the latest request's full instructions, read-only |
| `ui/menus/text_editor.py` | multi-line editing for menus: `$VISUAL`/`$EDITOR` (shared with the `/mcp` form), else a built-in prompt_toolkit editor |
| `ui/menus/model_menu.py` | `/model add`: provider discovery, `ModelSettingsSource`, `run_model_flow` |
| `ui/menus/model_picker.py` | `/model`: selection, completion, and confirmed deletion of saved models and chains, and `/model chains`; protects the current model and saved default |
| `ui/menus/chain_menu.py` | The `/model` picker's chain editors: new chain, models and their order (`edit_chain`), and `rename_chain` |
| `models/model_catalog.py` | model sources (genai-prices today) merged by `catalog()` |
| `models/profiles.py` | auth profiles: parse `PROVIDER@PROFILE:NAME`, the profile's credential account; read providers through `provider_of`/`base_model`, never `partition(':')` |
| `models/key_profiles.py` | `/login PROVIDER@PROFILE` for connection and API-key providers, and building a core model from a profile's key |
| `models/accounts.py` | accounts per provider for `/accounts` and `PROVIDER@*`: found from credential files or recorded at `/login`, ordered in the `accounts` table (no secrets) |
| `ui/menus/accounts_menu.py` | `/accounts` (`AccountsMenu`, `open_accounts_menu`), the add-account flow, and `choose_account`, the account step in `/model add` |
| `models/usage.py` | account usage fetchers: Codex through the core provider's own client (so tokens refresh as for model requests), Copilot with the saved GitHub login, plugins through `PluginLogin.usage` |
| `ui/menus/account_usage.py` | `UsageBoard`: `/accounts` usage loaded in the background while the menu is open; the menu thread redraws itself from `read_key`, never another thread |
| `models/chains.py` | `chain:NAME` fallback chains over core `FallbackModel`; a chain takes its first model's settings. Made in the `/model` picker (`/chain` is an alias of `/model chains`) |
| `models/model_settings.py` | `ModelSettingsForm`, the editable subset of `ModelSettings` |
| `models/custom_params.py` | dotted custom-parameter validation and expansion, independent of menus |
| `ui/menus/custom_params.py` | the editor for custom model parameters |
| `builtin_plugins/logfire.py` | the default-enabled `observability` plugin, configuring Logfire locally over core `Instrumentation`; `token` picks a `/keys` write token, `ui_events` subscribes it to UI telemetry; `configure` is a `FieldMenu` whose project row runs `logfire_setup` |
| `builtin_plugins/logfire_session.py` | the `observability` plugin's `CLAI session` roots: `SessionTracing` and the `git_email` lookup for `user_tag: git-email` |
| `builtin_plugins/logfire_setup.py` | the `observability` plugin's setup menu: region or self-hosted URL, Logfire's device sign-in (not the MCP OAuth in `logfire_oauth.py`), account email from `/v1/account/me`, project pick, write token saved in `/keys` |
| `ui/telemetry.py` | UI telemetry sinks, `record`/`span`, and the menu naming; instrument shared chokepoints (`run_worker`, `Commands.execute_async`, `FieldMenu`, the loader, `/keys`, the prompt), never one menu at a time, and record names, not content; typed text goes only through `prompt_text`, which the subscriber's `include_content` gates |
| `builtin_plugins/compaction.py` | the built-in `compaction` plugin: harness `FallbackCompaction([SummarizingCompaction, SlidingWindowCompaction])`, `/compact`, the context alert, its settings menu |
| `commands.py` | `Command`, the registry, completion |
| `ui/rendering/history.py` | `/resume` and `--resume` replay: the last turns of restored history as synthetic stream events through a non-smoothing `StreamRenderer` |
| `ui/rendering/usage_report.py` | `/usage`, `/cost`, and the footer cost, derived from `Session.messages` |
| `ui/rendering/status.py` | the footer `Status` fields, `StatusSegment`, and the `StatusLine` row painter |
| `ui/prompt/live_prompt.py` | pinned editor lifecycle, completion worker, submission queue, timed double-Esc gesture, and menu handoff |
| `ui/prompt/prompt_surface.py` | scroll-region ownership, serialized transcript writes and changed-row painting |
| `ui/prompt/prompt_transcript.py` | bounded styled transcript tail for viewport replay |
| `ui/prompt/prompt_resize.py` | scoped resize notifications, without terminal IO in signal handlers |
| `ui/prompt/prompt_buffer.py` | pure draft editing, history navigation, search and cell-width wrapping |
| `ui/prompt/prompt_completion.py` | bounded daemon completion worker; no terminal ownership |
| `ui/prompt/prompt_keys.py` | prompt-toolkit input attachment and paste, CSI-u, Kitty alternate-key and xterm report normalization; `PromptSurface` enables and releases xterm `CSI >4;1m` and Kitty `CSI >5u`; no prompt-toolkit renderer |
| `ui/prompt/prompt_selection.py` | SGR mouse report decoding and the left-button drag selection over the transcript's painted cells: highlight and text |
| `ui/prompt/text_clipboard.py` | copying text out: the local clipboard command on one background worker where the newest copy wins, or OSC 52 over SSH or without one |
| `config/__init__.py` | `Settings`, `PluginSettings` |
| `config/theme_names.py` | theme choices shared by settings validation and the picker |
| `config/settings_store.py` | the SQLite store under `$XDG_CONFIG_HOME/pydantic-clai2/`, including saved models and removal of their overrides |
| `config/features.py` | `SUPPORTED_FEATURES`, the feature names this build implements, and `CAPABILITY_REQUIREMENTS` for capability classes |
| `config/plugin_requirements.py` | pure rules for requirement tags: parse stored rows, drop unsupported settings, merge tags on save, the notice |
| `runtime/capability_guard.py` | `PluginGuard`: a plugin capability's run setup `UserError` becomes `CapabilitySetupError`; that turn fails, later turns leave the capability out |
| `config/project_settings.py` | `.clai/settings.json`: the walk-up to the git root, validation, `ProjectSettings` |
| `builtin_plugins/repo_context.py` | the built-in `repo_context` plugin over harness `RepoContext` |
| `builtin_plugins/coder.py` | the built-in `coder` plugin over harness `Coder`: validated settings, named agent folders (`.agents`/`.claude`/`.codex`, project then home), and its settings menu (file access, sub-agents, agent folders) |
| `builtin_plugins/slack.py` | the opt-in built-in `slack` plugin over harness `Slack`; its settings menu picks a `/keys` user token or a browser sign-in, resolved each turn |
| `slack_app.py` | Slack browser sign-in: the CLAI Slack app manifest (PKCE, MCP access, token rotation), scopes, and `PKCESignIn` for a Client ID |
| `plugins/keys.py` | `choose_key`, `browser_sign_in`, and `on_loop`: a plugin settings menu's credential rows, Esc-cancellable |
| `pkce.py` | `PKCESignIn`: browser sign-in for a registered public OAuth client (PKCE, no secret), with tokens in the credential store and locked refresh; built on core's `OAuthFlow` |
| `runtime/speculation.py` | the `run.speculative_code_mode` switch, `Ctrl+X Ctrl+S` toggle, session counters and pinned row |
| `runtime/speculative_mode.py` | harness `CodeMode` wiring (native writes, read-only speculation allowlist, guidance), imported only while on |
| `runtime/eager_timing.py` | eager `run_code` latency measurement and the nested-call id pattern |
| `runtime/sandbox_calls.py` | events and ordering that render calls from inside `run_code` like direct calls; no harness imports |
| `ui/rendering/theme.py` | Existing brand roles, opt-in Termflow palette scope, `color()`, `sgr()`, `roles()` |
| `ui/rendering/recolor.py` | repaints retained styled transcript lines in a newly selected theme |
| `ui/menus/theme_picker.py` | `/theme` picker over Termflow's bundled palettes |
| `ui/rendering/spinners.py` | the working-animation catalogue: builtins, plugin `get_spinners`, the user's `spinners.json`, `Spinners` |
| `ui/rendering/spinner_frames.py` | frame data for the Code Puppy cli-spinners pack |
| `ui/menus/spinner_picker.py` | `/spinner`: animated picker, by-name selection with speed, `init` |

The other built-in plugin implementations (`notifications`, `github`, `pylon`, `google_workspace`, `day_ai`, `ordinal`, `notion`, `logfire_mcp`, `posthog`, `grain`, and `linear`) also live in `builtin_plugins/`. The loader redirects old factory paths in saved declarations to this package.

Keep files concise - we don't need any 10,000 line files. Single responsibility.

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

`README.md` is the tour, `PLUGINS.md` is the plugin contract, this file is for
agents. A user-facing change updates the first two in the same PR. Plain
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
Mouse reporting stops most terminals from selecting text, so a left-button drag
selects painted cells in `PromptSurface` and its release copies them through
`text_clipboard.copy_text`. Do not drop the drag modes (`?1002h`) or copy-out breaks.

Resize rebuilds from `TranscriptBuffer`, not guessed row coordinates or cursor
reports. Never send erase-scrollback (CSI 3 J). Preserve the draft and scroll
anchor. `SIGWINCH` invalidates the next frame; it must not perform terminal IO.
Full-screen menus leave the live panel temporarily. Inline questions borrow it
with `run_worker(inline=True)` and pin the question text in their rows, not the
transcript, so output streamed meanwhile cannot push it away. Streamed text and thinking keep Markdown source,
and a tool-call group its call names, for width/theme repaint; tool output keeps styled lines, each tagged with the
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
