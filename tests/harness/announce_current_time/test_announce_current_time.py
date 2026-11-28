"""Test that `AnnounceCurrentTime` tells the model the date or hour of each request, and appends changes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import AnnounceCurrentTime


def _agent(capability: AnnounceCurrentTime[None], sent: list[ModelRequest]) -> Agent[None, str]:
    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        sent.append(request)
        return ModelResponse(parts=[TextPart('ok')])

    return Agent(FunctionModel(reply), deps_type=type(None), capabilities=[capability])


class _HostZone(tzinfo):
    """A `tzinfo` that isn't a `ZoneInfo`, named only by `tzname()`."""

    def __init__(self, name: str | None) -> None:
        self.name = name

    def utcoffset(self, dt: datetime | None) -> timedelta:
        return timedelta(hours=3)

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(0)  # pragma: lax no cover - only some Python versions ask for it

    def tzname(self, dt: datetime | None) -> str | None:
        return self.name


def _german_strftime(format: str, t: object) -> str:
    return format.replace('%A', 'Freitag')


class _CountedZoneInfo(ZoneInfo):
    built = 0

    def __new__(cls, key: str) -> _CountedZoneInfo:
        cls.built += 1
        return super().__new__(cls, key)


def _clock(monkeypatch: pytest.MonkeyPatch, now: datetime) -> None:
    # The run stamps each request with this clock, and the capability reads the time from that stamp.
    monkeypatch.setattr('pydantic_ai._agent_graph.now_utc', lambda: now)


class TestAnnounceCurrentTime:
    @pytest.mark.parametrize(
        ('capability', 'expected'),
        [
            (AnnounceCurrentTime[None](), "Today's date is Friday, 2026-01-02 (UTC)."),
            (AnnounceCurrentTime[None](timezone='Asia/Tokyo'), "Today's date is Saturday, 2026-01-03 (Asia/Tokyo)."),
            (
                AnnounceCurrentTime[None](timezone=timezone(timedelta(hours=-5)), precision='hour'),
                'The current time is within the hour starting 2026-01-02T18:00-05:00 (Friday, UTC-05:00).',
            ),
            (
                AnnounceCurrentTime[None](timezone=ZoneInfo('Europe/Amsterdam'), precision='hour'),
                'The current time is within the hour starting 2026-01-03T00:00+01:00 (Saturday, Europe/Amsterdam).',
            ),
            (AnnounceCurrentTime[None](timezone=_HostZone('XST')), "Today's date is Saturday, 2026-01-03 (XST)."),
            (AnnounceCurrentTime[None](timezone=_HostZone(None)), "Today's date is Saturday, 2026-01-03 (UTC+03:00)."),
        ],
    )
    async def test_states_the_request_time(
        self, capability: AnnounceCurrentTime[None], expected: str, monkeypatch: pytest.MonkeyPatch
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
        agent = _agent(AnnounceCurrentTime[None](), sent)

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
        instructions = AnnounceCurrentTime[None]().get_instructions()
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), messages=[])
        assert instructions(ctx) is None

    @pytest.mark.parametrize('id', [None, ''])
    def test_an_empty_id_falls_back_to_the_default(self, id: str | None) -> None:
        assert AnnounceCurrentTime[None](id=id).id == 'current_time'

    @pytest.mark.parametrize('precision', ['date', 'hour'])
    def test_the_weekday_ignores_the_process_locale(
        self, precision: Literal['date', 'hour'], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `datetime` formats `%A` by calling the `time.strftime` attribute, which follows `LC_TIME`; stand in for
        # a German locale, which CI doesn't install, and check the stand-in is live.
        monkeypatch.setattr('time.strftime', _german_strftime)
        assert f'{datetime(2026, 1, 2):%A}' == datetime(2026, 1, 2).strftime('%A') == 'Freitag'
        instructions = AnnounceCurrentTime[None](precision=precision).get_instructions()
        request = ModelRequest(parts=[], timestamp=datetime(2026, 1, 2, 12, tzinfo=UTC))
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), messages=[request])
        part = instructions(ctx)
        assert part is not None
        assert 'Friday' in part.content

    async def test_the_time_zone_is_resolved_once_when_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Resolving a zone the cache has evicted reads tzdata from disk, which a request must not do in the event loop.
        monkeypatch.setattr('pydantic_ai_harness.announce_current_time._capability.ZoneInfo', _CountedZoneInfo)
        monkeypatch.setattr(_CountedZoneInfo, 'built', 0)
        capability = await anyio.to_thread.run_sync(lambda: AnnounceCurrentTime[None](timezone='Europe/Amsterdam'))
        agent = _agent(capability, [])
        await agent.run('What day is it?')
        await agent.run('And now?')
        assert _CountedZoneInfo.built == 1

    def test_agent_spec_schema_publishes_its_fields(self) -> None:
        definitions = AgentSpec.model_json_schema_with_capabilities([AnnounceCurrentTime])['$defs']
        properties = definitions['spec_params_AnnounceCurrentTime']['properties']
        assert set(properties) == {'id', 'description', 'defer_loading', 'timezone', 'precision'}
        assert properties['timezone']['type'] == 'string'

    async def test_from_spec_takes_a_zone_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [{'AnnounceCurrentTime': {'timezone': 'Asia/Tokyo', 'precision': 'hour'}}],
            },
            custom_capability_types=[AnnounceCurrentTime],
        )
        _clock(monkeypatch, datetime(2026, 1, 2, 23, 30, tzinfo=UTC))
        request = (await agent.run('What time is it?')).all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert request.instructions is not None
        assert 'within the hour starting 2026-01-03T08:00+09:00 (Saturday, Asia/Tokyo).' in request.instructions

    def test_an_unknown_time_zone_fails_when_built(self) -> None:
        with pytest.raises(ZoneInfoNotFoundError):
            AnnounceCurrentTime[None](timezone='Mars/Olympus_Mons')
