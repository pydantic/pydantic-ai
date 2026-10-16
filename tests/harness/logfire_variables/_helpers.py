"""Plain test helpers shared by the managed-variable capability test modules.

Fixtures live in `conftest.py`; this module holds importable functions (tool bodies, the
tool-capturing model, and the local-provider context manager) so the capability test files can
share them without copy-paste.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import logfire
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, VariableConfig, VariablesConfig
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset


def get_weather(city: str) -> str:
    return f'sunny in {city}'


def get_forecast(city: str) -> str:
    return f'forecast for {city}'  # pragma: no cover - advertised only; the test models never call it


def weather_toolset() -> FunctionToolset[object]:
    """A toolset with an `id` and instructions of its own, so `toolset:weather` is addressable."""
    toolset = FunctionToolset[object]([get_weather], id='weather')

    @toolset.instructions
    def call_weather_first(_ctx: RunContext[object]) -> str:
        return 'TOOLSET: call get_weather first.'

    return toolset


def capture_tools(seen: list[ToolDefinition]) -> FunctionModel:
    """A model that records the advertised function tools it is shown, then ends the run."""

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(info.function_tools)
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def advertised(seen: list[ToolDefinition]) -> dict[str, str | None]:
    """The advertised `{name: description}` of each captured tool definition."""
    return {td.name: td.description for td in seen}


def published_value(name: str, value: Any, *, label: str = 'production') -> VariablesConfig:
    """A variables config in which `name` holds `value`, serialized as JSON, under one label.

    The shape a Logfire project has once someone has saved a value in the UI: one labeled version at
    full rollout, which is what a capability with a matching `label` then resolves.
    """
    return VariablesConfig(
        variables={
            name: VariableConfig(
                name=name,
                labels={label: LabeledValue(version=1, serialized_value=json.dumps(value))},
                rollout=Rollout(labels={label: 1.0}),
                overrides=[],
            )
        }
    )


@contextmanager
def variables_provider(capfire: CaptureLogfire, variables_config: VariablesConfig) -> Generator[None]:
    """Reconfigure Logfire with a local variables provider for the duration of the block.

    Restores the module's baseline configuration on exit so the change does not leak into other
    tests in this module (or any module collected after it).
    """
    logfire.configure(
        send_to_logfire=False,
        console=False,
        variables=logfire.LocalVariablesOptions(config=variables_config),
        additional_span_processors=[SimpleSpanProcessor(capfire.exporter)],
    )
    try:
        yield
    finally:
        logfire.configure(send_to_logfire=False, console=False)


Publish = Callable[[str, Any], None]
"""The `publish` fixture: put a managed config in the project for `agent__<agent name>`."""
