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

A plugin is a module with `activate(host: PluginHost)`, or a bare capability
class. `@host.on(name)` registers a handler for a lifecycle moment;
`@host.on(EventClass)` for a typed event; `host.commands.register` for a
`/command`; `host.add` for tools and instructions; `host.render` for custom
output; `host.settings(Model)` for validated config. `PLUGINS.md` is the user
contract. If code and `PLUGINS.md` disagree, fix one so they agree in the same PR.

## Rules for the plugin API

- **No global state.** Registration goes on a `PluginHost` instance someone
  constructed. No module-level dicts, no import-time side effects. Tests build a
  host and call `activate` directly.
- **Strings are keys, never payloads.** Hook names are a `Literal`; each name has
  an `@overload` binding one typed event dataclass. A handler never receives
  `*args`, `**kwargs`, `dict`, or `context: object = None`.
- **No string sub-dispatch.** A handler does not receive `event_type: str` and
  switch on it. Use the event class as the key.
- **One async spelling.** No `_sync` or `_async` pairs.
- **Observers return `None`.** Deciders mutate the event (`event.text = ...`,
  `event.cancel()`). Nothing collects a list of return values.
- **Fail closed, one way.** A raising handler on a decidable moment cancels the
  action and reports the error. No per-registration "fail open" flag.
- **Host hooks are `<subject>_<moment>`** (`turn_start`). Core hooks keep core's
  names verbatim (`before_tool_execute`). Do not rename a core hook for taste.
- **Payloads are kw-only dataclasses.** Not `BaseModel`. Pydantic is for the
  settings JSON boundary only.
- **No `getattr`/`hasattr` on a plugin** to discover what it supports. It
  registered the thing or it did not.
- **Only four host hooks.** `session_start`, `session_end`, `turn_start`,
  `turn_end`. Adding a fifth needs a use case that core cannot serve; say which
  core hook you checked and why it does not fit.

## Loading and unloading

Plugins load and unload while CLAI runs. The rules that make that safe:

- **One `PluginHost` per plugin.** The host is the ownership scope. Everything a
  plugin registers is recorded on its own host, so unloading is "discard this
  host". No `callback -> owner` map, no scanning registries for a plugin's name.
- **Load and unload only between turns.** `/commands` already run between turns,
  so this falls out for free; do not add a mid-run path.
- **Load fires `session_start` for that plugin; unload fires `session_end`.** A
  plugin cannot tell whether it was loaded at startup or later, and must not
  need to.
- **`reload` is unload, re-import, load.** Drop-in entry modules use fresh source.
  Installed modules use `importlib.reload`, which retains globals absent from the
  new source. Plugins must explicitly initialize their state on activation.
- **Registration is idempotent per name.** A capability is bound per run
  (`agent.run(capabilities=...)`), so "active for the next prompt" is the
  natural unit; nothing rebuilds the agent.
- **Built-ins are declarations, not code paths.** `DEFAULT_PLUGINS` in
  `_app.py` lists what CLAI ships enabled (`coder`, `ask_user`, `repo_context`,
  `compaction`, `persistence`, `logfire`). The loader treats them like drop-ins with the lowest
  precedence: a store declaration with the same id replaces one, `disable`
  persists an override, `remove` resets it. Do not special-case `Coder`
  anywhere else; the agent from `create_agent()` has no coding tools of its
  own. `coder` is declared with `repo_context: false` because `repo_context`
  binds harness `RepoContext` itself; keep it that way or `AGENTS.md` reaches
  the model twice.
- **Project declarations rank just above built-ins and start off.**
  `.clai/settings.json` (`project_settings.py`) may declare plugins; the loader
  takes them as `project=`, every one `enabled=False`, because a repository
  must not run code as the user on launch. `/plugins enable` is the approval
  and persists the approved declaration in the store. Precedence is store,
  drop-in folder, project, built-in. CLAI never writes the project file.
- **A load failure leaves the session as it was.** Import or `activate` errors
  are reported and the plugin stays unloaded; partial registrations from a
  failed `activate` are discarded with the host. Registered `session_end` handlers
  run with `reason='error'` under a shield with a five-second cooperative timeout
  per handler before a failed/cancelled load drops the host. Cleanup must tolerate
  incomplete `session_start`; a handler failure must not skip later cleanup.

Compaction registers harness `FallbackCompaction` directly with `max_fraction`
and `context_window` for both strategies. Harness owns the trigger; do not add
threshold math or an orchestrator in CLAI. Register the usage gauge after the
chain so yellow means the compacted request still exceeds the threshold.
`/compact` drives the same chain regardless of threshold. Only `ModelAPIError`,
`FallbackExceptionGroup`, and `UsageLimitExceeded` select truncation after a
summary failure; other exceptions propagate.

## Adding or changing a hook

1. Add the name to the `Literal`, the event dataclass, and the `@overload`.
2. Add the parity test entry: the `Literal` must equal `Hooks.on`'s attribute
   names plus the host names. Drift fails CI, not code review.
3. Fire it from exactly one place in the shell.
4. Document it in `PLUGINS.md` in the table it belongs to.

## The `/plugins`, `/set`, `/model`, and `/add_model` menus

Built on termflow's `MenuBuilder` (and `TextInputBuilder` for typed values),
exactly like Code Puppy's `/agent`, `/mcp`, `/set`, and `/model` menus:
alternate screen, a `.preview` panel on the right, `.on_key` for single-key
actions, `.footer_hint` for the key legend, `markdown_style()` for colours.

- Split it in two: a pure `build_plugins_menu(...)` that returns the menu (so
  tests drive it headless, no terminal), and a thin async runner that owns the
  screen and calls `menu.run` in a thread.
- Every key mutates immediately and `replace_items` redraws. No pending-changes
  state, no save/cancel pair.
- Nothing prints to the console while the menu is open; the alternate screen
  would hide it. Show empty states and errors inside the menu as disabled rows.
- Esc and Ctrl-C close cleanly. They are not errors.
- A menu opened mid-run (the `ask_user` question menu) goes inside
  `async with host.full_screen()`, which flushes streamed text and suspends the
  editor's input reader first, preserving its draft. Slash-command handlers
  already run with the editor suspended. Do not start a second input reader
  alongside the live editor.
- Adding a plugin is not in the menu. It needs free text, so it stays
  `/plugins add`.
- Anything that is "edit named, validated fields" uses `field_menu.py`: a
  `FieldSource` supplies rows, current values, validation, apply, and reset;
  `FieldMenu` builds the widgets; `run_flow` is the loop. `/set` and per-model
  settings are two sources, not two editors. Do not write a third editor.
- `/set` edits go through `CommandContext.set_setting` / `reset_setting`, the
  same path as the typed command, so validation lives in one place.
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
tool. Tool-specific output (shell previews, diffs, grep) is registered through
`host.render` by the plugin that owns the event. Match on event classes, never
on `tool_name` strings. Always flush the stream before printing anything else;
the host does this for renderers, so do not call `console.print` from inside an
`on` handler when a renderer would do.

## Colours

Every colour comes from `theme.py`, which holds the Pydantic brand palette and
the roles CLAI paints with (`ACCENT`, `INFO`, `WARNING`, `ERROR`, `MUTED`,
`THINKING`). Use a role, not a hex, and never a bare Rich colour name like
`'cyan'` or `'dim'`. Raw ANSI surfaces (status line, splash) go through
`theme.sgr(...)`, which handles the 16-colour fallback. `theme.py` is stdlib
only because the splash imports it before anything heavy. Source of truth is
the pydantic.dev `pydantic-visual-identity` skill's `brand-identity.md`.

## File map

| File | Holds |
|---|---|
| `_cli.py` | argument parsing, startup, `--agent` |
| `_app.py` | the prompt loop and built-in `/commands` |
| `_session.py` | conversation state, revision-checked saves, restore-only resume, per-run plugins |
| `sessions.py` | resume command and background namer ownership; built-in step capture |
| `session_browser.py` | project/session browser using Termflow layout and terminal primitives |
| `_rendering.py` | streaming Markdown and thinking |
| `plugins.py` | `PluginHost`, hook names, event dataclasses |
| `plugin_loader.py` | discovery, load, unload, reload; the `/plugins` subcommands |
| `plugin_menu.py` | the `/plugins` full-screen menu (`PluginMenu` plus its runner) |
| `ask_user_menu.py` | the built-in `ask_user` plugin: `QuestionMenu`, `TerminalAnswerer`, the transcript renderer |
| `screen.py` | `Screen`, what `host.full_screen()` binds to during a prompt |
| `field_menu.py` | the shared field editor (`FieldSource`, `FieldMenu`, `Runners`, `run_flow`) |
| `set_menu.py` | `/set`: `SettingsSource` over `CommandContext` |
| `model_menu.py` | `/add_model`: provider discovery, `ModelSettingsSource`, `run_model_flow` |
| `model_picker.py` | `/model`: selection and completion of saved models |
| `model_catalog.py` | model sources (genai-prices today) merged by `catalog()` |
| `model_settings.py` | `ModelSettingsForm`, the editable subset of `ModelSettings` |
| `logfire.py` | the default-enabled, locally configured Logfire plugin over core `Instrumentation` |
| `compaction.py` | the built-in `compaction` plugin: harness `FallbackCompaction([SummarizingCompaction, SlidingWindowCompaction])`, `/compact`, the context alert |
| `commands.py` | `Command`, the registry, completion |
| `usage_report.py` | `/usage`, `/cost`, and the footer cost, derived from `Session.messages` |
| `status.py` | the footer `Status` fields, `StatusSegment`, and the `StatusLine` row painter |
| `live_prompt.py` | pinned editor lifecycle, completion worker, submission queue and menu handoff |
| `prompt_surface.py` | scroll-region ownership, serialized transcript writes and changed-row painting |
| `prompt_transcript.py` | bounded styled transcript tail for viewport replay |
| `prompt_resize.py` | scoped resize notifications, without terminal IO in signal handlers |
| `prompt_buffer.py` | pure draft editing, history navigation, search and cell-width wrapping |
| `prompt_completion.py` | bounded daemon completion worker; no terminal ownership |
| `prompt_keys.py` | keyboard decoder attachment only; no prompt-toolkit Application or renderer |
| `config.py` | `Settings`, `PluginSettings` |
| `settings_store.py` | the SQLite store under `$XDG_CONFIG_HOME/pydantic-clai2/` |
| `project_settings.py` | `.clai/settings.json`: the walk-up to the git root, validation, `ProjectSettings` |
| `repo_context.py` | the built-in `repo_context` plugin over harness `RepoContext` |
| `theme.py` | brand palette, colour roles, `sgr()` |

Keep files concise - we don't need any 10,000 line files. Single responsibility.

## Testing

- When adding or modifying a CLAI2 CLI UX feature, test every affected UX feature
  with your changes in a fresh tmux window before opening a PR. Run CLAI2 with
  `uv run clai2`. Verify the actual terminal behavior against the intention of
  the user's request, not just the implementation or automated tests.
- `pytest-anyio`; real model calls are blocked globally.
- Drive the shell with `TestModel` and a `Console(file=StringIO())`.
- Test a hook by building a `PluginHost`, registering a handler, and firing the
  event from the shell path that owns it. Assert the handler's effect (the
  cancelled turn, the rewritten text), not a mock call count.
- Renderers get synthetic events. They must not need `Coder` installed.
- Cancellation: use a real `anyio` cancel scope, order with `Event`s, no sleeps.

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
PromptSession renderer or mutate generated layout children. Transcript writes
go directly to the scroll region, never through an erase/redraw of the editor.
The hardware cursor stays hidden until release; the input cursor is a painted
reverse-video cell. Keep terminal mutations in `PromptSurface`, and detach the
key reader before a menu owns the screen. The remaining prompt-toolkit decoder
preserves paste and modified keys not yet exposed by Termflow's `read_key`.

Physical resize blanks the viewport and defers output until size notifications
have been quiet for 250 ms. Rebuild from `TranscriptBuffer`, not guessed old row
coordinates or cursor reports. Never send erase-scrollback (CSI 3 J). Keep editor
height changes separate from physical resize, preserve the draft, and close the
resize output spool on both normal handoff and failure. `SIGWINCH` only marks the
resize and schedules a paint; the signal handler must not perform terminal IO.
