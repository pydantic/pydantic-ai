"""The features this CLAI build supports, by name, so settings saved by another build can say what they need.

Every CLAI on a machine shares one settings database, whatever code it runs: other worktrees,
branches, and installs. A plugin setting whose valid values or meaning depend on code another
build may lack is tagged with a feature name when it is saved (see `PLUGINS.md`). A build that
does not list that name here ignores the setting and uses the default instead.

Names are descriptive, stable, lowercase words joined by hyphens, such as
`stock-bound-delegation`. Never reuse or rename one: other builds compare them as plain strings.
"""

import re

SUPPORTED_FEATURES: frozenset[str] = frozenset({'logfire-user-tag', 'logfire-httpx'})
"""Feature names this build implements. Add a name in the change that adds the code behind it."""

CAPABILITY_REQUIREMENTS: dict[str, dict[str, frozenset[str]]] = {}
"""Requirement tags for capability classes declared as `module:Class`, keyed by that factory.

A capability class has no `Plugin` subclass to call `host.settings(Model, requires=...)` from, so the
build that ships a declaration for one lists its tagged settings here, for example
`{'pydantic_ai_harness.coder:Coder': {'sub_agents': frozenset({'stock-bound-delegation'})}}`.
"""

_FEATURE_NAME = re.compile(r'[a-z][a-z0-9]*(?:-[a-z0-9]+)*')


def check_feature_name(name: str) -> str:
    """Return `name`, or raise `ValueError` when it is not lowercase words joined by hyphens."""
    if not _FEATURE_NAME.fullmatch(name):
        raise ValueError(
            f'Feature name {name!r} must be lowercase words joined by hyphens, like stock-bound-delegation.'
        )
    return name
