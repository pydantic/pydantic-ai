"""Deprecated alias of `pydantic_ai_pax`: `pydantic_clai2.X` imports `pydantic_ai_pax.X`, the same module object."""

import importlib
import sys
import warnings
from collections.abc import Sequence
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai_pax import DEFAULT_PLUGINS, Session, StreamRenderer, chat, create_agent, open_stock_agent

__all__ = ['DEFAULT_PLUGINS', 'Session', 'StreamRenderer', 'chat', 'create_agent', 'open_stock_agent']
"""`pydantic_ai_pax.__all__`, copied so that importing this package stays light."""

_ALIAS = f'{__name__}.'
_ENTRY_POINT = f'{__name__}.__main__'


def _deprecated(old: str, new: str) -> None:
    warnings.warn(f'`{old}` is deprecated, import `{new}` instead.', DeprecationWarning, stacklevel=3)


def __getattr__(name: str) -> object:
    # Only its exports: a submodule `pydantic_ai_pax` happens to hold, such as `__main__`, is not this package's.
    if name not in __all__:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    _deprecated(f'{__name__}.{name}', f'pydantic_ai_pax.{name}')
    return getattr(importlib.import_module('pydantic_ai_pax'), name)


class _Alias(MetaPathFinder, Loader):
    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if not fullname.startswith(_ALIAS) or fullname == _ENTRY_POINT:
            return None
        return spec_from_loader(fullname, self)

    def exec_module(self, module: ModuleType) -> None:
        target = f'pydantic_ai_pax.{module.__name__.removeprefix(_ALIAS)}'
        _deprecated(module.__name__, target)
        # The import system returns whatever `sys.modules` holds once this returns, so both names share one module.
        sys.modules[module.__name__] = importlib.import_module(target)


sys.meta_path.insert(0, _Alias())
