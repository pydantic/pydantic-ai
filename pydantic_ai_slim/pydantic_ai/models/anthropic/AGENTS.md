# pydantic_ai_slim/pydantic_ai/models/anthropic/ Guidelines

These rules add to [`models/AGENTS.md`](../AGENTS.md) and apply to the Anthropic package only.

- Keep `AnthropicModel` one class in `__init__.py`. Put a module-level helper in the `_*.py` submodule whose docstring names its concern. Do not split the class's methods into mixins: a mixin spreads one object's state across files.
- Never import from `__init__.py` at runtime in a submodule. `__init__.py` imports every submodule, so that import is circular. Import the name under `TYPE_CHECKING` instead, as `_streaming.py` does for `AnthropicModelName`.
- Import every submodule inside the `try` block of `__init__.py` that imports the `anthropic` SDK. A missing SDK then raises the install hint, not a bare `ImportError`.
- In a test, patch a name in the module that calls it: `pydantic_ai.models.anthropic.<name>` for a call in `__init__.py`, `pydantic_ai.models.anthropic._streaming.<name>` for a call in `_streaming.py`. Each `from ._x import name` creates its own binding, so a patch on the defining submodule reaches only calls inside that submodule.
