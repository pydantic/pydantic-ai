"""A `FunctionModel` that classifies a provider's rejection of a too-complex schema, for the Tool Output switch tests.

It lives outside the test modules so the offline, DBOS and Temporal tests share one definition of the private hook it
overrides.
"""

from __future__ import annotations

from pydantic_ai.models.function import FunctionModel


class CompilingModel(FunctionModel):
    """A `FunctionModel` that, like `AnthropicModel`, raises `ModelSchemaTooComplexError` for a schema that's too complex."""

    def _classifies_schema_too_complex(self) -> bool:
        return True
