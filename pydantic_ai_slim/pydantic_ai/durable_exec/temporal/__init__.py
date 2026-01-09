from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import is_dataclass, replace
from typing import Any

import temporalio.api.common.v1
from pydantic import BaseModel, TypeAdapter
from pydantic.errors import PydanticUserError
from pydantic_core import to_json
from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import (
    CompositePayloadConverter,
    DataConverter,
    DefaultPayloadConverter,
    EncodingPayloadConverter,
    JSONPlainPayloadConverter,
)
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkerConfig, WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from ...exceptions import UserError
from ._agent import TemporalAgent
from ._logfire import LogfirePlugin
from ._run_context import TemporalRunContext
from ._toolset import TemporalWrapperToolset
from ._workflow import PydanticAIWorkflow

__all__ = [
    'TemporalAgent',
    'PydanticAIPlugin',
    'LogfirePlugin',
    'AgentPlugin',
    'TemporalRunContext',
    'TemporalWrapperToolset',
    'PydanticAIWorkflow',
    'PydanticAIPayloadConverter',
    'pydantic_ai_data_converter',
]

# We need eagerly import the anyio backends or it will happens inside workflow code and temporal has issues
# Note: It's difficult to add a test that covers this because pytest presumably does these imports itself
# when you have a @pytest.mark.anyio somewhere.
# I suppose we could add a test that runs a python script in a separate process, but I have not done that...
import anyio._backends._asyncio  # pyright: ignore[reportUnusedImport]  # noqa: F401

try:
    import anyio._backends._trio  # pyright: ignore[reportUnusedImport]  # noqa: F401
except ImportError:
    pass


class PydanticAIJSONPayloadConverter(EncodingPayloadConverter):
    """Pydantic AI JSON payload converter that properly serializes Pydantic models.

    Unlike the default Temporal PydanticJSONPlainPayloadConverter which uses
    `pydantic_core.to_json()` without type information, this converter uses
    `TypeAdapter(type(value)).dump_json()` for Pydantic models and dataclasses.
    This ensures computed fields and field aliases are properly serialized,
    fixing issues like DocumentUrl losing its media_type during round-trips.
    """

    @property
    def encoding(self) -> str:
        return 'json/plain'

    def to_payload(self, value: Any) -> temporalio.api.common.v1.Payload | None:
        if isinstance(value, BaseModel) or is_dataclass(value):
            data = TypeAdapter(type(value)).dump_json(value)
        else:
            data = to_json(value)
        return temporalio.api.common.v1.Payload(metadata={'encoding': self.encoding.encode()}, data=data)

    def from_payload(
        self,
        payload: temporalio.api.common.v1.Payload,
        type_hint: type[Any] | None = None,
    ) -> Any:
        return TypeAdapter(type_hint if type_hint is not None else Any).validate_json(payload.data)


class PydanticAIPayloadConverter(CompositePayloadConverter):
    """Pydantic AI payload converter that properly handles Pydantic models.

    Replaces the default JSON converter with PydanticAIJSONPayloadConverter
    which uses TypeAdapter for serialization, ensuring computed fields and
    aliases work correctly during Temporal activity round-trips.
    """

    def __init__(self) -> None:
        json_payload_converter = PydanticAIJSONPayloadConverter()
        super().__init__(
            *(
                c if not isinstance(c, JSONPlainPayloadConverter) else json_payload_converter
                for c in DefaultPayloadConverter.default_encoding_payload_converters
            )
        )


pydantic_ai_data_converter = DataConverter(payload_converter_class=PydanticAIPayloadConverter)


def _data_converter(converter: DataConverter | None) -> DataConverter:
    if converter is None:
        return pydantic_ai_data_converter

    # If the payload converter class is already PydanticAIPayloadConverter or a subclass,
    # the converter is already compatible with Pydantic AI - return it as-is.
    if issubclass(converter.payload_converter_class, PydanticAIPayloadConverter):
        return converter

    # If the payload converter class is a custom subclass of PydanticPayloadConverter (not the
    # exact class), preserve it as the user may have custom serialization logic.
    if (
        issubclass(converter.payload_converter_class, PydanticPayloadConverter)
        and converter.payload_converter_class is not PydanticPayloadConverter
    ):
        return converter

    # If using the exact PydanticPayloadConverter, upgrade to PydanticAIPayloadConverter
    # to fix serialization of computed fields like DocumentUrl.media_type.
    if converter.payload_converter_class is PydanticPayloadConverter:
        return replace(converter, payload_converter_class=PydanticAIPayloadConverter)

    # If using a non-Pydantic payload converter, warn and replace just the payload converter class,
    # preserving any custom payload_codec or failure_converter_class.
    if converter.payload_converter_class is not DefaultPayloadConverter:
        warnings.warn(
            'A non-Pydantic Temporal payload converter was used which has been replaced with PydanticAIPayloadConverter. '
            'To suppress this warning, ensure your payload_converter_class inherits from PydanticAIPayloadConverter.'
        )

    return replace(converter, payload_converter_class=PydanticAIPayloadConverter)


def _workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
    if not runner:
        raise ValueError('No WorkflowRunner provided to the Pydantic AI plugin.')  # pragma: no cover

    if not isinstance(runner, SandboxedWorkflowRunner):
        return runner  # pragma: no cover

    return replace(
        runner,
        restrictions=runner.restrictions.with_passthrough_modules(
            'pydantic_ai',
            'pydantic',
            'pydantic_core',
            'logfire',
            'rich',
            'httpx',
            'anyio',
            'sniffio',
            'httpcore',
            # Used by fastmcp via py-key-value-aio
            'beartype',
            # Imported inside `logfire._internal.json_encoder` when running `logfire.info` inside an activity with attributes to serialize
            'attrs',
            # Imported inside `logfire._internal.json_schema` when running `logfire.info` inside an activity with attributes to serialize
            'numpy',
            'pandas',
        ),
    )


class PydanticAIPlugin(SimplePlugin):
    """Temporal client and worker plugin for Pydantic AI."""

    def __init__(self):
        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='PydanticAIPlugin',
            data_converter=_data_converter,
            workflow_runner=_workflow_runner,
            workflow_failure_exception_types=[UserError, PydanticUserError],
        )

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        config = super().configure_worker(config)

        workflows = list(config.get('workflows', []))  # type: ignore[reportUnknownMemberType]
        activities = list(config.get('activities', []))  # type: ignore[reportUnknownMemberType]

        for workflow_class in workflows:  # type: ignore[reportUnknownMemberType]
            agents = getattr(workflow_class, '__pydantic_ai_agents__', None)  # type: ignore[reportUnknownMemberType]
            if agents is None:
                continue
            if not isinstance(agents, Sequence):
                raise TypeError(  # pragma: no cover
                    f'__pydantic_ai_agents__ must be a Sequence of TemporalAgent instances, got {type(agents)}'
                )
            for agent in agents:  # type: ignore[reportUnknownVariableType]
                if not isinstance(agent, TemporalAgent):
                    raise TypeError(  # pragma: no cover
                        f'__pydantic_ai_agents__ must be a Sequence of TemporalAgent, got {type(agent)}'  # type: ignore[reportUnknownVariableType]
                    )
                activities.extend(agent.temporal_activities)  # type: ignore[reportUnknownMemberType]

        config['activities'] = activities

        return config


class AgentPlugin(SimplePlugin):
    """Temporal worker plugin for a specific Pydantic AI agent."""

    def __init__(self, agent: TemporalAgent[Any, Any]):
        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='AgentPlugin',
            activities=agent.temporal_activities,
        )
