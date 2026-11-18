"""Warning categories shared across pydantic-ai-harness."""


class HarnessDeprecationWarning(UserWarning):
    """Warning emitted when a deprecated pydantic-ai-harness API is used.

    Inherits from `UserWarning` instead of `DeprecationWarning` so that deprecations are
    visible by default at runtime, matching Pydantic AI's `PydanticAIDeprecationWarning`.
    Silence every harness deprecation at once with:

    ```python
    import warnings
    from pydantic_ai_harness import HarnessDeprecationWarning

    warnings.filterwarnings('ignore', category=HarnessDeprecationWarning)
    ```
    """


class MCPReadOnlyNoToolsWarning(UserWarning):
    """`read_only=True` removed every tool from a hosted MCP server, so the agent gets none of its tools.

    The filter keeps only tools the server annotates with `readOnlyHint: true`, and some servers
    publish no annotations at all. Disable `read_only`, or filter this category when an empty
    toolset is intended.
    """
