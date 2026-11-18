# pydantic-clai2

`pydantic-clai2` is now [`pydantic-ai-pax`](https://pypi.org/project/pydantic-ai-pax/), and its `clai2`
command is now `pax`. Install the new package:

```bash
uv tool install pydantic-ai-pax
```

This package installs `pydantic-ai-pax` and keeps the `clai2` and `pydantic-clai2` commands, which print a
deprecation note and run `pax`, and `pydantic_clai2` imports, which warn and import from `pydantic_ai_pax`. Your settings, sessions, and plugins carry over. It will stop being released
in a future version.
