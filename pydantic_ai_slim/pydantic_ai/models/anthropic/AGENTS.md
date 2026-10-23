# pydantic_ai_slim/pydantic_ai/models/anthropic/ Guidelines

These rules add to [`models/AGENTS.md`](../AGENTS.md) and apply to the Anthropic package only.

- Keep `AnthropicModel` one class in `__init__.py`. Put a module-level helper in the `_*.py` submodule whose docstring names its concern. Do not split the class's methods into mixins: a mixin spreads one object's state across files.
- Never import from `__init__.py` at runtime in a submodule. `__init__.py` imports every submodule, so that import is circular. Import the name under `TYPE_CHECKING` instead, as `_streaming.py` does for `AnthropicModelName`.
- Import a new submodule inside the `try` block of `__init__.py` that imports the `anthropic` SDK. A missing SDK then raises the install hint, not a bare `ImportError`.
- In a test, patch a helper at the submodule that defines it, e.g. `pydantic_ai.models.anthropic._errors.map_api_errors`. A patch on `pydantic_ai.models.anthropic.<name>` replaces only the package's own binding, so a submodule keeps calling the original.
