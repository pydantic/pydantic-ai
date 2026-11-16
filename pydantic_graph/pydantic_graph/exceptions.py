class GraphSetupError(TypeError):
    """Error caused by an incorrectly configured graph."""

    message: str
    """Description of the mistake."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class GraphBuildingError(ValueError):
    """An error raised during graph-building."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class GraphValidationError(ValueError):
    """An error raised during graph validation."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class GraphRuntimeError(RuntimeError):
    """Error caused by an issue during graph execution."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class UnsupportedEventLoopError(RuntimeError):
    """Error caused by calling a synchronous method on an event loop that cannot be driven by the caller.

    Synchronous methods run their asynchronous implementation using `loop.run_until_complete()`, which not every
    event loop implements. Temporal's workflow event loop is one that doesn't: it can only be driven by Temporal.

    Pydantic AI's synchronous methods report this as a `pydantic_ai.exceptions.UserError` instead.
    """

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
