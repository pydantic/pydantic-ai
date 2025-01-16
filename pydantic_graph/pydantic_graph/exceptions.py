class GraphSetupError(TypeError):
    """Error caused by an incorrectly configured graph."""

    message: str
    """Description of the mistake."""

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
