We'd love you to contribute to PydanticAI!

## Installation and Setup

1. Clone your fork and cd into the repo directory

```bash
git clone git@github.com:<your username>/pydantic.git
cd pydantic-ai
```

2. Install `uv` and `pre-commit`

We use pipx here, for other options see:

* [`uv` getting install docs](https://docs.astral.sh/uv/getting-started/installation/)
* [`pre-commit` install docs](https://pre-commit.com/#install)

To get `pipx` itself, see [these docs](https://pypa.github.io/pipx/)

```bash
pipx install uv pre-commit
```

3. Install `pydantic-ai`, deps, test deps, and docs deps

```bash
make install
```
