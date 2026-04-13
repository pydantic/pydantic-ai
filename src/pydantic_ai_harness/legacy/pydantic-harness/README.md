# pydantic-harness (deprecated)

This package has been renamed to **[pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness)** to align with the Pydantic AI family naming convention (`pydantic-ai`, `pydantic-ai-slim`, `pydantic-ai-harness`).

## Migration

```bash
uv remove pydantic-harness
uv add pydantic-ai-harness
```

Update your imports:

```diff
- from pydantic_harness import CodeMode
+ from pydantic_ai_harness import CodeMode
```

## About this shim

This `pydantic-harness==0.1.1` release is a compatibility shim that depends on `pydantic-ai-harness>=0.1.1,<0.2` and re-exports its public API. Existing code keeps working, but importing `pydantic_harness` emits a `DeprecationWarning`.

When `pydantic-ai-harness==0.2.0` ships, this shim will stop resolving — `pip install pydantic-harness` will fail and users will be forced to migrate.

This is the final release of the `pydantic-harness` package name.
