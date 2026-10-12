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
| `before_run`, `after_tool_execute`, ... | `on_session_start`, `on_session_end`, `on_turn_start`, `on_turn_end`, `on_plugin_load_failed` |
| `@on_event(EventClass)` | `render(event)` for display; `@on_event` / `Hooks` inside `get_capabilities` for observing |
| `RunContext` | `self.host`, a `PluginHost` holding the console, conversation, status, full screen, and saved settings |

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
  Unloading discards it; there is nothing to unregister.
- **`PluginHost` is only context.** It lost every registration method and list.
- **Agent-run hooks are core's.** A plugin that wants `before_tool_execute` or a
  typed stream event returns a `Hooks` capability (or its own capability using
  `@on_event`) from `get_capabilities`, exactly as any Pydantic AI user would. CLAI
  no longer mirrors core's hook names.
- **Capabilities stay the unit of reuse.** A declaration can still name a bare
  capability class (`module:Class`), which is wrapped in a plugin whose only
  contribution is that capability, built from the settings JSON.
- **Tests construct, not activate.** `load_plugin(PluginClass, host)` returns the
  same `LoadedPlugin` the loader builds.

Startup load failures happen before an agent run, so core's `before_run` and other
run hooks cannot report them. The loader delivers `PluginLoadFailed` through
`on_plugin_load_failed` after trying every enabled plugin. This lets observability
report earlier failures through its own configured instance without changing load order.

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
