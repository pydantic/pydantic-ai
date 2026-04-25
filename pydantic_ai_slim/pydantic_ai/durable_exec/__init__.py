"""Building blocks for writing durable-execution capabilities.

These helpers are re-exported here so third-party packages can implement
`AbstractCapability` subclasses that route model requests through external
durable execution systems (Temporal, DBOS, Prefect, ...) without reaching
into Pydantic AI's private modules.

The built-in capabilities live in submodules: `pydantic_ai.durable_exec.temporal`,
`pydantic_ai.durable_exec.dbos`, and `pydantic_ai.durable_exec.prefect`.
"""

from pydantic_ai._agent_graph import call_model, open_model_stream
from pydantic_ai._utils import disable_threads

__all__ = ['call_model', 'disable_threads', 'open_model_stream']
