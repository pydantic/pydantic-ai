from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import fields, replace
from typing import Any, Literal

from aws_durable_execution_sdk_python.config import Duration, StepConfig, StepSemantics
from aws_durable_execution_sdk_python.retries import RetryDecision, RetryPresets
from aws_durable_execution_sdk_python.serdes import SerDes
from pydantic import ConfigDict, TypeAdapter, ValidationError

from pydantic_ai.durable_exec import (
    DurableOperationId,
    JournalCallableOperationBackend,
    ModelCancelSuspendedResponseId,
    ModelCompactMessagesId,
    ModelRequestId,
    RoleBasedOperationConfig,
    model_error_retry_after,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets import ToolsetTool

from ._bridge import ENGINE_NAME, current_bridge

TOOL_CONFIG_KEY = 'aws_lambda'
# Derived rather than listed so a new SDK `StepConfig` field is accepted, not rejected as unknown.
_STEP_CONFIG_FIELDS = frozenset(field.name for field in fields(StepConfig))
_STEP_CONFIG_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
_RETRY_STRATEGY_ADAPTER: TypeAdapter[Callable[[Exception, int], RetryDecision] | None] = TypeAdapter(
    Callable[[Exception, int], RetryDecision] | None,
    config=ConfigDict(arbitrary_types_allowed=True),
)
_STEP_SEMANTICS_ADAPTER = TypeAdapter(StepSemantics)
_SERDES_ADAPTER: TypeAdapter[SerDes[object] | None] = TypeAdapter(
    SerDes[object] | None, config=ConfigDict(arbitrary_types_allowed=True)
)
_MINIMUM_RETRY_AFTER_CAP = 60
"""The shortest cap on a model error's `retry_after`, in seconds, when the retry strategy's own delay is shorter."""

_STEP_CONFIG_VALUE_ADAPTERS = {
    'retry_strategy': (_RETRY_STRATEGY_ADAPTER, 'a callable or None'),
    'step_semantics': (_STEP_SEMANTICS_ADAPTER, 'StepSemantics'),
    'serdes': (_SERDES_ADAPTER, 'SerDes or None'),
}


class AWSLambdaOperationConfig(RoleBasedOperationConfig[StepConfig | None]):
    def __init__(self, base: Mapping[str, Any] | None) -> None:
        base_config = _parse_step_config(base)
        self.base_step_config = base_config
        """The `StepConfig` every step gets unless a tool's metadata overrides it."""

        def resolve_tool(
            operation_id: DurableOperationId, tool: object | None, tool_name: str
        ) -> StepConfig | Literal[False] | None:
            del operation_id
            toolset_tool = _toolset_tool(tool)
            metadata = toolset_tool.tool_def.metadata if toolset_tool is not None else None
            metadata_config: object = metadata.get(TOOL_CONFIG_KEY) if metadata is not None else None
            if metadata_config is False:
                return False
            if metadata_config is not None and not isinstance(metadata_config, dict):
                raise UserError(
                    f'Tool {tool_name!r} has invalid {TOOL_CONFIG_KEY!r} metadata: expected a dict '
                    f'(`{ENGINE_NAME} durable config`) or `False`, got {type(metadata_config).__name__}.'
                )
            config = dict(base or {})
            if metadata_config:
                try:
                    config.update(_STEP_CONFIG_MAPPING_ADAPTER.validate_python(metadata_config, strict=True))
                except ValidationError:
                    # Raised as a `ValidationError`, this would reach the model as invalid tool arguments.
                    raise UserError(
                        f'Tool {tool_name!r} has invalid {TOOL_CONFIG_KEY!r} metadata: '
                        'step config keys must be strings.'
                    ) from None
            return _parse_step_config(config)

        super().__init__(
            model=base_config, event=base_config, capability=base_config, tool=base_config, resolve_tool=resolve_tool
        )


class AWSLambdaOperationBackend(JournalCallableOperationBackend[StepConfig | None]):
    def __init__(
        self,
        *,
        agent_name: str,
        default_model_id: str | None,
        config: AWSLambdaOperationConfig,
    ) -> None:
        super().__init__(agent_name=agent_name, default_model_id=default_model_id, config=config)

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: StepConfig | None,
    ) -> object:
        del cache_key

        bridge = current_bridge()
        # Execution in a durable context guarantees a bridge.
        assert bridge is not None
        if isinstance(operation_id, ModelRequestId | ModelCompactMessagesId | ModelCancelSuspendedResponseId):
            config = _waiting_retry_after(config)
        return await bridge.run_step(name, body, config)


def _waiting_retry_after(config: StepConfig | None) -> StepConfig:
    """A model step's config whose retry strategy waits the `retry_after` a model error asked for.

    The step's retry strategy, the SDK's default when none is set, still decides whether to retry. When it does and
    the model error has a `retry_after`, the SDK suspends the execution for that long instead of the strategy's delay,
    rounded up to whole seconds. A strategy is a function with no maximum to read, so the wait is capped at the
    strategy's own delay for the attempt or a minute, whichever is longer.
    """
    step_config = config or StepConfig()
    strategy = step_config.retry_strategy or RetryPresets.default()

    def retry_strategy(error: Exception, attempts_made: int) -> RetryDecision:
        decision = strategy(error, attempts_made)
        if not decision.should_retry:
            return decision
        delay = model_error_retry_after(error, maximum=max(decision.delay_seconds, _MINIMUM_RETRY_AFTER_CAP))
        if delay is None:
            return decision
        return RetryDecision.retry(Duration(seconds=math.ceil(delay)))

    return replace(step_config, retry_strategy=retry_strategy)


def _parse_step_config(config: Mapping[str, Any] | None) -> StepConfig | None:
    """Validate and build a `StepConfig` from a step configuration mapping."""
    if not config:
        return None
    unknown = sorted(set(config) - _STEP_CONFIG_FIELDS)
    if unknown:
        raise UserError(
            f'Unknown {TOOL_CONFIG_KEY!r} step config {"key" if len(unknown) == 1 else "keys"} '
            f'{", ".join(repr(key) for key in unknown)}. Supported keys are '
            f'{", ".join(repr(field) for field in sorted(_STEP_CONFIG_FIELDS))}.'
        )
    for key, value in config.items():
        _validate_step_config_value(key, value)
    return StepConfig(**config)


def _validate_step_config_value(key: str, value: object) -> None:
    adapter_config = _STEP_CONFIG_VALUE_ADAPTERS.get(key)
    if adapter_config is None:
        return
    adapter, expected = adapter_config
    try:
        adapter.validate_python(value, strict=True)
    except ValidationError:
        raise UserError(
            f'Invalid {TOOL_CONFIG_KEY!r} step config value for {key!r}: expected {expected}, '
            f'got {type(value).__name__}.'
        ) from None


def _toolset_tool(value: object | None) -> ToolsetTool[Any] | None:
    if isinstance(value, ToolsetTool):
        return value  # pyright: ignore[reportUnknownVariableType]
    return None
