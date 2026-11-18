"""Deprecated alias of `pydantic_ai_pax`: `pydantic_clai2.X` imports `pydantic_ai_pax.X`, the same module object."""

import importlib
import sys
import warnings
from collections.abc import Sequence
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from types import ModuleType

_ALIAS = f'{__name__}.'
_ENTRY_POINT = f'{__name__}.__main__'


class _Alias(MetaPathFinder, Loader):
    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if not fullname.startswith(_ALIAS) or fullname == _ENTRY_POINT:
            return None
        return spec_from_loader(fullname, self)

    def exec_module(self, module: ModuleType) -> None:
        target = f'pydantic_ai_pax.{module.__name__.removeprefix(_ALIAS)}'
        warnings.warn(
            f'`{module.__name__}` is deprecated, import `{target}` instead.', DeprecationWarning, stacklevel=2
        )
        # The import system returns whatever `sys.modules` holds once this returns, so both names share one module.
        sys.modules[module.__name__] = importlib.import_module(target)


sys.meta_path.insert(0, _Alias())
