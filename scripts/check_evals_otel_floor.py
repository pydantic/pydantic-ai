"""Check online evaluation events with the minimum OTel API/SDK and no Logfire extra.

Run separately from pytest: its shared fixtures import Logfire, whose dependencies
can force a newer OTel version and hide an unsupported minimum.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib.metadata import version

from opentelemetry._logs import SeverityNumber, set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnErrorLocation, OnlineEvalConfig, wait_for_evaluations


@dataclass
class AlwaysTrue(Evaluator[int, int]):
    """Produce a successful evaluation result."""

    def evaluate(self, ctx: EvaluatorContext[int, int]) -> bool:
        return True


@dataclass
class FailingEvaluator(Evaluator[int, int]):
    """Produce an evaluator failure that must also emit an event."""

    def evaluate(self, ctx: EvaluatorContext[int, int]) -> bool:
        raise ValueError('evaluation failed')


async def main() -> None:
    """Verify both event paths through the public online evaluation API."""
    exporter = InMemoryLogExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    set_logger_provider(provider)
    errors: list[tuple[OnErrorLocation, Exception]] = []
    config = OnlineEvalConfig(on_error=lambda exc, ctx, ev, loc: errors.append((loc, exc)))

    @config.evaluate(AlwaysTrue(), FailingEvaluator())
    async def identity(x: int) -> int:
        return x

    try:
        assert await identity(1) == 1
        await wait_for_evaluations()
        assert not errors, errors
        records = [entry.log_record for entry in exporter.get_finished_logs()]
        assert len(records) == 2, records
        assert all(record.event_name == 'gen_ai.evaluation.result' for record in records)
        by_name = {record.attributes['gen_ai.evaluation.name']: record for record in records if record.attributes}
        result = by_name['AlwaysTrue']
        assert result.attributes is not None
        assert result.attributes['gen_ai.evaluation.score.value'] == 1.0
        failure = by_name['FailingEvaluator']
        assert failure.attributes is not None
        assert failure.attributes['error.type'] == 'ValueError'
        assert failure.severity_number == SeverityNumber.WARN
        print(f'OTel {version("opentelemetry-api")}: result and failure events emitted without sink errors')
    finally:
        provider.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
