# Installation

PydanticAI is available [on PyPI](https://pypi.org/project/pydantic-ai/) so installation is as simple as:

=== "pip"

    ```bash
    pip install pydantic-ai
    ```

=== "uv"

    ```bash
    uv add pydantic-ai
    ```

It requires Python 3.9+.

## Use with Pydantic Logfire

PydanticAI has an excellent (but completely optional) integration with [Pydantic Logfire](https://pydantic.dev/logfire) to help you view and understand agent runs.

To use Logfire with PydanticAI, install PydanticAI with the `logfire` optional group:

=== "pip"

    ```bash
    pip install 'pydantic-ai[logfire]'
    ```

=== "uv"

    ```bash
    uv add 'pydantic-ai[logfire]'
    ```

From there, follow the [Logfire documentation](https://logfire.pydantic.dev/docs/) to configure Logfire.

## Next Steps

To run PydanticAI, follow instructions [in examples](examples/index.md).
