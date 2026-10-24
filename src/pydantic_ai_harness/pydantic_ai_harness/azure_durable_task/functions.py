"""Optional Azure Functions activity registration, without owning orchestrators or clients.

This design preview targets `azure-functions-durable` 2.0.0rc3 on Python 3.13+.
The 1.x Functions SDK uses a different orchestration runtime and is not supported.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable

try:
    import azure.durable_functions as df
except ImportError as import_error:
    raise ImportError(
        'Install `pydantic-ai-harness[azure-functions]` on Python 3.13+ to use Azure Functions.'
    ) from import_error

from pydantic_ai.tools import AgentDepsT

from ._capability import AzureDurableTaskDurability
from ._operation_backend import _Activity  # pyright: ignore[reportPrivateUsage]

__all__ = ('register_activities',)


def register_activities(app: df.Blueprint, durability: AzureDurableTaskDurability[AgentDepsT]) -> None:
    """Register a bound capability's activities on a Durable Functions app or blueprint.

    Call at module import time, after defining the agent and all its tools, using
    `AzureDurableTaskDurability.from_agent(agent)`. Call once per agent on an app.
    Unbound capabilities raise `UserError`; duplicate function names are rejected
    by the Functions SDK when it indexes the app.

    Activity bodies use the Functions worker's async loop and the existing parameter
    transports, dependency scopes and result codecs. No SDK worker, portal,
    orchestrator, HTTP route or client is created. Do not enter `register(worker)`
    for these functions.

    Scheduler activity names are preserved in the binding. Separate hashed host
    function names satisfy Functions' more restrictive naming rules.
    """
    for activity in durability.get_durable_operation_backend().registrations():
        assert isinstance(activity, _Activity)
        _register_activity(app, activity.__name__, activity.execute)


def _register_activity(app: df.Blueprint, name: str, execute: Callable[[object], Awaitable[object]]) -> None:
    async def invoke(payload: object) -> object:
        return await execute(payload)

    function_name = f'pydantic_ai_{hashlib.sha256(name.encode()).hexdigest()}'
    app.function_name(name=function_name)(app.activity_trigger(input_name='payload', activity=name)(invoke))
