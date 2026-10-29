"""The plugin declarations CLAI ships, kept apart from the shell so `clai2 plugins` reads them without importing it."""

from pydantic_clai2.config import PluginSettings

DEFAULT_PLUGINS: tuple[PluginSettings, ...] = (
    PluginSettings(
        id='coder',
        factory='pydantic_clai2.builtin_plugins.coder',
        settings={
            'unrestricted_filesystem': True,
            'repo_context': False,
            'sub_agents': False,
            'agent_folders': ['agents'],
        },
    ),
    PluginSettings(id='ask_user', factory='pydantic_clai2.builtin_plugins.ask_user_menu'),
    PluginSettings(id='repo_context', factory='pydantic_clai2.builtin_plugins.repo_context'),
    PluginSettings(id='compaction', factory='pydantic_clai2.builtin_plugins.compaction', settings={}),
    PluginSettings(id='persistence', factory='pydantic_clai2.runtime.sessions'),
    PluginSettings(id='observability', factory='pydantic_clai2.builtin_plugins.logfire'),
    PluginSettings(id='notifications', factory='pydantic_clai2.builtin_plugins.notifications'),
    PluginSettings(id='hooks', factory='pydantic_clai2.builtin_plugins.hooks'),
    PluginSettings(id='mcp', factory='pydantic_clai2.mcp'),
    PluginSettings(id='github', factory='pydantic_clai2.builtin_plugins.github', enabled=False),
    PluginSettings(id='pylon', factory='pydantic_clai2.builtin_plugins.pylon', enabled=False),
    PluginSettings(id='google_workspace', factory='pydantic_clai2.builtin_plugins.google_workspace', enabled=False),
    PluginSettings(id='day_ai', factory='pydantic_clai2.builtin_plugins.day_ai', enabled=False),
    PluginSettings(id='ordinal', factory='pydantic_clai2.builtin_plugins.ordinal', enabled=False),
    PluginSettings(id='notion', factory='pydantic_clai2.builtin_plugins.notion', enabled=False),
    PluginSettings(id='slack', factory='pydantic_clai2.builtin_plugins.slack', enabled=False),
    PluginSettings(id='logfire_mcp', factory='pydantic_clai2.builtin_plugins.logfire_mcp', enabled=False),
    PluginSettings(id='posthog', factory='pydantic_clai2.builtin_plugins.posthog', enabled=False),
    PluginSettings(id='grain', factory='pydantic_clai2.builtin_plugins.grain', enabled=False),
    PluginSettings(id='linear', factory='pydantic_clai2.builtin_plugins.linear', enabled=False),
    PluginSettings(id='herdr', factory='pydantic_clai2.builtin_plugins.herdr'),
    PluginSettings(id='aoe', factory='pydantic_clai2.builtin_plugins.aoe', enabled=False),
    PluginSettings(id='web', factory='pydantic_clai2.builtin_plugins.web', enabled=False),
    PluginSettings(id='workspaces', factory='pydantic_clai2.builtin_plugins.workspaces', enabled=False),
)
"""Built-in declarations, each integrated with the shell. `remove` restores their defaults.

Other harness capabilities are not listed here: a user adds one on purpose with `/plugins add` or a plugin module.

`coder` leaves out its own `RepoContext` because `repo_context` binds one, so instruction files load once.
`compaction` runs alongside `coder`, whose `ClearToolResults` only empties old tool results; see `plugins.compatibility`.
"""

STOCK_PLUGINS: tuple[PluginSettings, ...] = tuple(
    plugin.model_copy(update={'settings': {**plugin.settings, 'sub_agents': True}}) if plugin.id == 'coder' else plugin
    for plugin in DEFAULT_PLUGINS
)
"""CLI-owned agents bind plugins at construction, so Coder can delegate safely.

`DEFAULT_PLUGINS` keeps its run-level delegation opt-out for caller-supplied agents.
"""
