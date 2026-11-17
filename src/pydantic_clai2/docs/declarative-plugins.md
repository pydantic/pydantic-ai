# Declarative plugins

Design note for the `Plugin` base class in `pydantic_clai2.plugins`. `PLUGINS.md`
is the user contract; this note records why the API has its shape.

## Before: registration inside `activate(host)`

A plugin used to be a function that received a `PluginHost` and called
registration methods on it:

```python
def activate(host: PluginHost) -> None:
    settings = host.settings(Settings)
    host.add(MyCapability(settings.url))
    host.commands.register(Command(name='mine', ...))

    @host.on('session_start')
    async def start(event: SessionStart) -> None: ...

    @host.configure
    async def configure() -> str: ...
```

That had four costs:

- **Nothing is visible without running it.** What a plugin offers (commands,
  tools, a settings menu) was only known after calling `activate`, so the loader,
  the `/plugins` menu, and tests all had to run the plugin and then inspect a
  mutable host.
- **The host was two things.** `PluginHost` was both the plugin's runtime
  context (console, conversation, status, settings) and a bag of registries
  (`handlers`, `capabilities`, `renderers`, `commands`, ...). Every new kind of
  contribution added a registration method and a list.
- **State hid in closures.** Settings, clients, and caches lived in local
  variables captured by nested functions, which made them hard to reach from
  tests and easy to recreate by accident.
- **Two hook systems.** `host.on` accepted both CLAI's four moments and every
  core `Hooks().on` name, re-exporting core's hook surface with a
  `Literal`-and-`@overload` table that had to be kept in parity by a test.

## After: a class that declares what it contributes

Pydantic AI already solved "an extension that contributes several kinds of
things" with `AbstractCapability`: a class whose `get_toolset`,
`get_instructions`, `get_model_settings`, ... methods each default to
contributing nothing, and whose hook methods default to no-ops. A capability
overrides only what it needs, and the agent asks for each contribution.

`Plugin` is the same shape one layer up:

| `AbstractCapability` | `Plugin` |
|---|---|
| constructor arguments | `Plugin[Settings]`, validated from the saved JSON into `self.settings` |
| `get_toolset()`, `get_instructions()`, ... | `get_capabilities()`, `get_commands()`, `get_status_segments()`, `get_spinners()`, `get_model_providers()`, `get_logins()` |
| `before_run`, `after_tool_execute`, ... | `on_session_start`, `on_session_end`, `on_turn_start`, `on_turn_end`, `on_conversation_changed`, `on_command_end`, `on_plugin_load_failed`, `on_task_progress`, `on_fork_end`, `on_install_progress` |
| `@on_event(EventClass)` | `render(event)` for display; `@on_event` / `Hooks` inside `get_capabilities` for observing |
| `RunContext` | `self.host`, a `PluginHost` holding the console, conversation, status, full screen, and saved settings, the prompt (`send_input`/`interrupt`/`complete`), the saved `sessions`, the session's `workspace`, and, for a process-scoped plugin, the `live` sessions |

```python
class MyPlugin(Plugin[Settings]):
    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (MyCapability(self.settings.url),)

    def get_commands(self) -> Sequence[Command]:
        return (Command(name='mine', ...),)

    async def on_session_start(self, event: SessionStart) -> None: ...

    async def configure(self) -> str: ...
```

The consequences:

- **The loader collects instead of the plugin registering.** `collect(plugin)`
  calls each `get_*` method once and freezes the results in a `LoadedPlugin`.
  Unloading discards it; there is nothing to unregister. The loader first awaits
  `prepare()`, so contributions that depend on files, such as one `/command` per
  Agent Skill, are read asynchronously instead of blocking in a `get_*` method.
- **`PluginHost` is only context.** It lost every registration method and list.
- **Agent-run hooks are core's.** A plugin that wants `before_tool_execute` or a
  typed stream event returns a `Hooks` capability (or its own capability using
  `@on_event`) from `get_capabilities`, exactly as any Pydantic AI user would. CLAI
  no longer mirrors core's hook names.
- **Capabilities stay the unit of reuse.** A declaration can still name a bare
  capability class (`module:Class`), which is wrapped in a plugin whose only
  contribution is that capability, built from the settings JSON.
- **Tests construct, not activate.** `load_plugin(PluginClass, host)` returns the
  same `LoadedPlugin` the loader builds. It is synchronous, so for a plugin with
  `prepare` it raises `TypeError` naming the `from_host`, `await prepare()`,
  `collect` sequence to use instead.

Startup load failures happen before an agent run, so core's `before_run` and other
run hooks cannot report them. The loader delivers `PluginLoadFailed` through
`on_plugin_load_failed` after trying every enabled plugin. This lets observability
report earlier failures through its own configured instance without changing load order.

A delegated task runs detached from the run that started it, and a background one
outlives it, so the parent run's hooks and stream events never see the child's own
events: the harness owner hands them to one observer instead, CLAI's `Tasks`, which
also drives `/tasks`. `on_task_progress` passes that same feed on to plugins.

A background `/fork` runs a copy of the conversation as a separate session, so its run's
hooks are that session's, and its `on_turn_end` looks like any turn's. The terminal prints
its answer itself; `on_fork_end` says which fork ended and how, so a plugin with no
terminal, such as the web UI, can show the same.

`/update` installs a newer CLAI with uv, which takes a while, then restarts. Commands never
reach the agent, and `on_command_end` fires once uv has finished, so `on_install_progress`
tells plugins as it goes, such as the web UI, whose page has no terminal to watch.

## Scope: one instance per live session, or one per process

The web UI keeps several sessions live in one CLAI process, each with its own
conversation, folder, and running turn. A plugin instance holds state for the
session it was loaded in (its console, its conversation, a connection, a task), so
the default, `scope = 'session'`, builds one instance per live session. Each one
loads with its session and ends with it, including when an idle session closes, so a
plugin written for the terminal needs no change as long as it keeps its state on
`self` and reads its folder from `host.workspace` instead of the process's working
directory, which CLAI never changes after startup.

Some plugins own something the process can only have once: `web` listens on one port,
`observability` configures Logfire and `Agent.instrument_all` for the whole process,
and `workspaces` keeps one registry. Building those per session would open a second
server or reconfigure tracing under running sessions. `scope = 'process'` builds such
a plugin once, by the first session that loads it, and joins it to every live session:

- **`on_session_start` and `on_session_end` stay "load" and "unload".** They fire once
  per instance, so for a process plugin once per process. Closing one session only
  detaches it there. It ends when unloaded, after every running turn that bound it, or
  when the last session closes.
- **Every other moment is per session.** A turn, command, or conversation change
  belongs to one session, so the session's own loader dispatches it, and the plugin
  learns which session through `host.live.current()`, a context variable set around
  each dispatch, command, and run. A task started there inherits it.
- **The host follows the handled session where that is the only sensible answer.**
  `session_id`, `workspace`, `tasks`, `plugins`, `setting`, `send_input`,
  `submit_prompt`, and `interrupt` resolve to `host.live.current()`, else to the
  loading session, or the oldest live session once that one has closed. `conversation`, `status`, and `console` stay the loading session's;
  `host.live` reaches the others. A second, per-session host object would have been an
  alternative, but the existing `PluginHost` attributes already read lazily, so
  following the handled session kept one host and one plugin contract.
- **`host.live` is only for process plugins.** A session plugin has its own session in
  `host`, and listing or driving the others is a process-wide concern, so it gets
  `None`. That keeps the per-session contract free of cross-session reach.

The scope is a class attribute, not a setting, because it is a property of the code:
whether its state can be duplicated. It is public: a third-party plugin that runs one
server or shares one registry declares it the same way the built-ins do.

## Discovery

A declaration's `factory` resolves to a plugin class:

- `module`: the one public `Plugin` subclass defined in that module; several is
  an error asking for `module:Class`.
- `module:Class`: that `Plugin` subclass, or a capability class wrapped as above.

A factory that still resolves to an `activate(host)` function fails to load with
an error pointing at "What a plugin declares" in `PLUGINS.md`, rather than being run
through a compatibility shim. `pydantic-clai2` is 0.x, whose minor releases may
change APIs; it has shipped one release with `activate(host)`, every known plugin
is in this repository and was ported, and a shim would have to keep the whole
registration API this change removes. Saved declarations of the built-in `ask_user` plugin, the
only built-in declared as `module:activate`, are redirected to its new module.

## Why not ...

- **Class attributes instead of methods** (`commands = [...]`): contributions
  often depend on settings or instance state, and methods match
  `AbstractCapability`.
- **Calling `get_*` on every use:** the shell reads commands, segments, and
  renderers constantly. Calling once per load keeps the collected set stable
  between loads, which is what `/plugins` shows.
- **Keeping `host.on` for core hooks:** it duplicated core's API and drifted;
  core's `Hooks` is already declarative and typed.
