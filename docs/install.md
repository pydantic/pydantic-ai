# Installation

PydanticAI is available [on PyPI](https://pypi.org/project/pydantic-ai/) so installation is as simple as:


```bash
pip/uv-add pydantic-ai
```

It requires Python 3.9+.

## Use with Pydantic Logfire

PydanticAI has an excellent (but completely optional) integration with [Pydantic Logfire](https://pydantic.dev/logfire) to help you view and understand agent runs.

To use Logfire with PydanticAI, install PydanticAI with the `logfire` optional group:

```bash
pip/uv-add 'pydantic-ai[logfire]'
```

From there, follow the [Logfire setup cods](logfire.md#integrating-logfire) to configure Logfire.

## Running Examples

PydanticAI bundles its examples so you can run them very easily.

To install extra dependencies required to run examples, install the `examples` optional group:

```bash
pip/uv-add 'pydantic-ai[examples]'
```

For next steps, follow the instructions [in the examples](examples/index.md).
