"""Test that `CurrentTime` tells the model the date or hour of each request, and appends changes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import CurrentTime


def _agent(capability: CurrentTime[None], sent: list[ModelRequest]) -> Agent[None, str]:
    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        sent.append(request)
        return ModelResponse(parts=[TextPart('ok')])

    return Agent(FunctionModel(reply), deps_type=type(None), capabilities=[capability])


def _clock(monkeypatch: pytest.MonkeyPatch, now: datetime) -> None:
    # The run stamps each request with this clock, and the capability reads the time from that stamp.
    monkeypatch.setattr('pydantic_ai._agent_graph.now_utc', lambda: now)


class TestCurrentTime:
    @pytest.mark.parametrize(
        ('capability', 'expected'),
        [
            (CurrentTime[None](), "Today's date is Friday, 2026-01-02 (UTC)."),
            (CurrentTime[None](timezone='Asia/Tokyo'), "Today's date is Saturday, 2026-01-03 (Asia/Tokyo)."),
            (
                CurrentTime[None](timezone=timezone(timedelta(hours=-5)), precision='hour'),
                'The current time is within the hour starting 2026-01-02T18:00-05:00 (Friday, UTC-05:00).',
            ),
            (
                CurrentTime[None](timezone=ZoneInfo('Europe/Amsterdam'), precision='hour'),
                'The current time is within the hour starting 2026-01-03T00:00+01:00 (Saturday, Europe/Amsterdam).',
            ),
        ],
    )
    async def test_states_the_request_time(
        self, capability: CurrentTime[None], expected: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clock(monkeypatch, datetime(2026, 1, 2, 23, 30, tzinfo=UTC))
        sent: list[ModelRequest] = []
        await _agent(capability, sent).run('What day is it?')
        assert sent[0].instructions is not None
        assert f'\n{expected}\n' in sent[0].instructions

    async def test_a_new_date_is_appended_rather_than_rewriting_the_instructions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: list[ModelRequest] = []
        agent = _agent(CurrentTime[None](), sent)

        _clock(monkeypatch, datetime(2026, 1, 2, 23, 30, tzinfo=UTC))
        first = await agent.run('What day is it?')
        _clock(monkeypatch, datetime(2026, 1, 2, 23, 50, tzinfo=UTC))
        second = await agent.run('Still?', message_history=first.all_messages())
        _clock(monkeypatch, datetime(2026, 1, 3, 0, 10, tzinfo=UTC))
        await agent.run('And now?', message_history=second.all_messages())

        assert len({request.instructions for request in sent}) == 1
        assert [[part.content for part in request.parts if isinstance(part, UserPromptPart)] for request in sent] == (
            snapshot(
                [
                    ['What day is it?'],
                    ['Still?'],
                    [
                        'And now?',
                        """\
<system><pyai:context id="capability:current_time">
Today's date is Saturday, 2026-01-03 (UTC).
</pyai:context></system>\
""",
                    ],
                ]
            )
        )

    def test_states_nothing_before_the_first_request(self) -> None:
        instructions = CurrentTime[None]().get_instructions()
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), messages=[])
        assert instructions(ctx) is None

    def test_an_unknown_time_zone_fails_when_built(self) -> None:
        with pytest.raises(ZoneInfoNotFoundError):
            CurrentTime[None](timezone='Mars/Olympus_Mons')
