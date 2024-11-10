from __future__ import annotations as _annotations

import json

__all__ = 'ModelRetry', 'UserError', 'UnexpectedModelBehaviour'


class ModelRetry(Exception):
    """Exception raised when a retriever function should be retried.

    The agent will return the message to the model and ask it to try calling the function/tool again.
    """

    message: str

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class UserError(RuntimeError):
    """Error caused by a usage mistake by the application developer — You!"""

    message: str

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class UnexpectedModelBehaviour(RuntimeError):
    """Error caused by unexpected Model behavior, e.g. an unexpected response code."""

    message: str
    body: str | None

    def __init__(self, message: str, body: str | None = None):
        self.message = message
        if body is None:
            self.body: str | None = None
        else:
            try:
                self.body = json.dumps(json.loads(body), indent=2)
            except ValueError:
                self.body = body
        super().__init__(message)

    def __str__(self) -> str:
        if self.body:
            return f'{self.message}, body:\n{self.body}'
        else:
            return self.message
