"""Explicit trusted plugin loading; dispatch belongs to native capabilities."""

import importlib
from collections.abc import Sequence

from pydantic_ai.capabilities import AbstractCapability

from .config import PluginSettings


def load_plugins(declarations: Sequence[PluginSettings]) -> list[AbstractCapability[None]]:
    """Instantiate capability classes with plugin-owned, validated settings.

    Factories are classes accepting settings as keyword arguments. A plugin
    validates those arguments in its constructor, using its own Pydantic model.
    """
    plugins: list[AbstractCapability[None]] = []
    for declaration in declarations:
        if not declaration.enabled:
            continue
        module, name = declaration.factory.split(':')
        factory: object = getattr(importlib.import_module(module), name)
        if not isinstance(factory, type) or not issubclass(factory, AbstractCapability):
            raise TypeError(f'Plugin {declaration.id!r} must be an AbstractCapability class')
        plugin: AbstractCapability[None] = factory(**declaration.settings)  # pyright: ignore[reportUnknownVariableType]
        plugins.append(plugin)
    return plugins
