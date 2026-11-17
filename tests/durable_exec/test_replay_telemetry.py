"""The engine-facing hooks that keep replayed durable work out of telemetry."""

from __future__ import annotations

from logfire.testing import CaptureLogfire

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.durable_exec import record_durable_unit, replay_scope
from pydantic_ai.models.test import TestModel


def _span_names(capfire: CaptureLogfire) -> list[str]:
    return [span['name'] for span in capfire.exporter.exported_spans_as_dict()]


async def test_record_durable_unit_outside_a_scope_does_nothing(capfire: CaptureLogfire) -> None:
    record_durable_unit(executed=True)
    record_durable_unit(executed=False)

    await Agent(TestModel(), name='a', capabilities=[Instrumentation()]).run('go')

    assert _span_names(capfire) == ['chat test', 'invoke_agent a']


async def test_is_replaying_drops_spans_that_end_while_replaying(capfire: CaptureLogfire) -> None:
    replaying = True
    agent = Agent(TestModel(), name='a', capabilities=[Instrumentation()])

    with replay_scope(is_replaying=lambda: replaying):
        await agent.run('go')
    assert _span_names(capfire) == []

    replaying = False
    with replay_scope(is_replaying=lambda: replaying):
        await agent.run('go')
    assert _span_names(capfire) == ['chat test', 'invoke_agent a']
