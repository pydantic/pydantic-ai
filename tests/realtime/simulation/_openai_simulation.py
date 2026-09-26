"""The simulation of a session with an OpenAI Realtime protocol model (OpenAI, Azure OpenAI, xAI)."""

from __future__ import annotations as _annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from hypothesis import strategies as st
from hypothesis.stateful import precondition, rule

from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.xai import XaiProvider
from pydantic_ai.realtime import RealtimeModel
from pydantic_ai.realtime.azure import AzureRealtimeModel
from pydantic_ai.realtime.openai import OpenAIRealtimeConnection, OpenAIRealtimeModel, OpenAIRealtimeModelSettings
from pydantic_ai.realtime.settings import RealtimeModelSettings
from pydantic_ai.realtime.xai import XaiRealtimeModel

from ._machine import TICKS, ManualTurnMachine
from ._openai_server import Dialect, OpenAIServer, ServerSession
from ._simulation import SessionOptions, Simulation, step
from ._truth import GroundTruth
from ._wire import SendFault


@dataclass
class OpenAIOptions:
    """Connect-time configuration of the simulated OpenAI-protocol session."""

    turn_detection: Literal['server_vad', 'manual'] = 'server_vad'
    transcription: bool = True
    dialect: Dialect = 'openai'
    """Which provider's model the session connects to: OpenAI, Azure OpenAI, or xAI Grok Voice."""


class OpenAISimulation(Simulation):
    """A session with an OpenAI Realtime protocol model over the simulated server."""

    def __init__(
        self,
        *,
        seed: int = 0,
        options: SessionOptions | None = None,
        openai: OpenAIOptions | None = None,
        strict: bool | None = None,
    ) -> None:
        super().__init__(seed=seed, options=options, strict=strict)
        self.openai = openai or OpenAIOptions()
        self.provider = self.openai.dialect
        self.server = OpenAIServer(
            dialect=self.openai.dialect, model='grok-voice-latest' if self.openai.dialect == 'xai' else 'gpt-realtime'
        )
        self.deferred_requests = 0
        """How many requests for a response the connection deferred behind an active one."""
        if self.options.latency:
            self.server.network.latency = lambda: self.rng.choice((0, 0, 0, 1, 2, 4))

    @property
    def failed_sends(self) -> list[tuple[str | None, str | None, SendFault]]:
        return self.server.network.failed_sends

    @property
    def truth(self) -> GroundTruth:
        return self.server.truth

    def observe_connection(self, connection: Any) -> None:
        """Count the requests for a response the connection held back behind an active one (it folds them)."""
        assert isinstance(connection, OpenAIRealtimeConnection)
        request_response = connection._request_response  # pyright: ignore[reportPrivateUsage]

        async def counted(input_indexes: Sequence[int]) -> None:
            if connection._response_active:  # pyright: ignore[reportPrivateUsage]
                self.deferred_requests += len(input_indexes)
            await request_response(input_indexes)

        connection._request_response = counted  # pyright: ignore[reportPrivateUsage]

    def build_model(self) -> RealtimeModel:
        if self.openai.dialect == 'azure':
            return AzureRealtimeModel(
                'gpt-realtime',
                provider=AzureProvider(
                    azure_endpoint='https://simulated.openai.azure.com', api_version='2026-04-10', api_key='simulated'
                ),
            )
        if self.openai.dialect == 'xai':
            return XaiRealtimeModel('grok-voice-latest', provider=XaiProvider(api_key='simulated'))
        return OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='simulated'))

    def model_settings(self) -> RealtimeModelSettings:
        settings: OpenAIRealtimeModelSettings = {
            'turn_detection': self.openai.turn_detection == 'server_vad',
            'input_transcription_model': 'gpt-4o-mini-transcribe' if self.openai.transcription else None,
        }
        settings['reconnect'] = {'max_attempts': 2, 'base_delay': 0.1, 'jitter': False}
        return settings

    @contextmanager
    def transport(self) -> Iterator[None]:
        with self.server.network.patch():
            yield

    # --- provider behavior --------------------------------------------------------------------------

    def _after_server(self, deliver: bool, ticks: int | None) -> None:
        if deliver:
            self.server.network.deliver()
        self._advance(ticks)

    @step
    def deliver(self, count: int | None = None, ticks: int | None = None) -> None:
        """Let up to `count` in-flight server frames (all by default) reach the client."""
        self.server.network.deliver(count)
        self._advance(ticks)

    @step
    def speak(self, chunks: int = 1, deliver: bool = True, ticks: int | None = None) -> None:
        """The active response says a word, with `chunks` × 100 ms of audio."""
        self.server.speak(chunks)
        self._after_server(deliver, ticks)

    @step
    def call_tool(self, name: str = 'lookup', deliver: bool = True, ticks: int | None = None) -> str:
        """The active response calls a tool."""
        call_id = self.server.call_tool(name)
        self._after_server(deliver, ticks)
        return call_id

    @step
    def finish(
        self,
        status: Literal['completed', 'incomplete', 'failed'] = 'completed',
        late: bool = False,
        deliver: bool = True,
        ticks: int | None = None,
    ) -> None:
        """The active response ends (as cancelled, if the client asked for that)."""
        self.server.finish(status, late=late)
        self._after_server(deliver, ticks)

    @step
    def release_late_done(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.release_late_done()
        self._after_server(deliver, ticks)

    @step
    def repeat_done(self, deliver: bool = True, ticks: int | None = None) -> None:
        """The server sends the latest `response.done` again."""
        self.server.repeat_done()
        self._after_server(deliver, ticks)

    @step
    def speech_start(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.speech_start()
        self._after_server(deliver, ticks)

    @step
    def speech_stop(self, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.speech_stop()
        self._after_server(deliver, ticks)

    @step
    def transcribe(self, fail: bool = False, deliver: bool = True, ticks: int | None = None) -> None:
        self.server.transcribe(fail=fail)
        self._after_server(deliver, ticks)

    @step
    def reject_next(self, kind: Literal['content', 'response'] = 'content', ticks: int | None = None) -> None:
        """The provider will refuse the next user content item, or the next response request."""
        self.server.arm_rejection(kind)
        self._advance(ticks)

    @step
    def fail_next_send(self, fault: SendFault = 'lost', ticks: int | None = None) -> None:
        """The next frame the client sends hits a dead socket."""
        self.server.network.send_faults.append(fault)
        self._advance(ticks)

    @step
    def drop(self, refuse_dials: int = 0, ticks: int | None = None) -> None:
        """The connection drops; the next `refuse_dials` re-dials fail."""
        self.server.network.dial_failures += refuse_dials
        self.server.drop()
        self._advance(ticks)

    # --- settling -----------------------------------------------------------------------------------

    def drive_server_to_rest(self) -> bool:
        server = self.server
        session = server.session
        if session is None or not session.socket.alive:
            return False
        if server.network.in_flight():
            server.network.deliver()
        elif session.late_done is not None:
            server.release_late_done()
        elif session.active is not None:
            if not session.active.truth.words and not session.active.truth.tool_calls:
                server.speak(1)
            server.finish()
        elif session.speaking is not None:
            server.speech_stop()
        elif session.pending_transcripts:
            server.transcribe()
        else:
            return False
        server.network.deliver()
        return True


def _options(dialect: Dialect) -> st.SearchStrategy[OpenAIOptions]:
    return st.builds(
        OpenAIOptions,
        turn_detection=st.sampled_from(['server_vad', 'manual']),
        transcription=st.booleans(),
        dialect=st.just(dialect),
    )


class OpenAIMachine(ManualTurnMachine):  # pragma: lax no cover (driven only by randomized exploration)
    """Randomized traces against the simulated OpenAI Realtime server."""

    provider_options = _options('openai')

    @staticmethod
    def make_simulation(seed: int, options: SessionOptions, provider_options: OpenAIOptions) -> Simulation:
        return OpenAISimulation(seed=seed, options=options, openai=provider_options)

    @property
    def openai_sim(self) -> OpenAISimulation:
        sim = self.s
        assert isinstance(sim, OpenAISimulation)
        return sim

    def server_session(self) -> ServerSession | None:
        """The live server session, if the simulation is still running and connected."""
        session = self.openai_sim.server.session if self.live() else None
        return session if session is not None and session.socket.alive else None

    def active(self) -> bool:
        session = self.server_session()
        return session is not None and session.active is not None

    def can_drop(self) -> bool:
        session = self.server_session()
        if session is None:
            return False
        # What xAI's native resumption does with a response in flight has never been recorded, so its drops
        # are kept to the moments the recording covers: between responses, with nothing on the wire.
        return self.openai_sim.openai.dialect != 'xai' or (
            session.active is None and session.late_done is None and not self.openai_sim.server.network.in_flight()
        )

    @precondition(lambda self: self.active())
    @rule(chunks=st.integers(min_value=1, max_value=2), deliver=st.booleans(), ticks=TICKS)
    def speak(self, chunks: int, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.openai_sim.speak(chunks=chunks, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.active())
    @rule(deliver=st.booleans(), ticks=TICKS)
    def call_tool(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.openai_sim.call_tool(deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.active())
    @rule(
        status=st.sampled_from(['completed', 'completed', 'completed', 'incomplete', 'failed']),
        late=st.booleans(),
        deliver=st.booleans(),
        ticks=TICKS,
    )
    def finish(
        self, status: Literal['completed', 'incomplete', 'failed'], late: bool, deliver: bool, ticks: int | None
    ) -> None:
        self.run(lambda: self.openai_sim.finish(status=status, late=late, deliver=deliver, ticks=ticks))

    @precondition(lambda self: (session := self.server_session()) is not None and session.late_done is not None)
    @rule(deliver=st.booleans(), ticks=TICKS)
    def release_late_done(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.openai_sim.release_late_done(deliver=deliver, ticks=ticks))

    @precondition(lambda self: (session := self.server_session()) is not None and bool(session.ended_responses))
    @rule(deliver=st.booleans(), ticks=TICKS)
    def repeat_done(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.openai_sim.repeat_done(deliver=deliver, ticks=ticks))

    @precondition(
        lambda self: (
            (session := self.server_session()) is not None
            and session.server_vad
            and session.speaking is None
            and session.audio_ms > 0
        )
    )
    @rule(deliver=st.booleans(), ticks=TICKS)
    def speech_start(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.openai_sim.speech_start(deliver=deliver, ticks=ticks))

    @precondition(lambda self: (session := self.server_session()) is not None and session.speaking is not None)
    @rule(deliver=st.booleans(), ticks=TICKS)
    def speech_stop(self, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.openai_sim.speech_stop(deliver=deliver, ticks=ticks))

    @precondition(lambda self: (session := self.server_session()) is not None and bool(session.pending_transcripts))
    @rule(fail=st.booleans(), deliver=st.booleans(), ticks=TICKS)
    def transcribe(self, fail: bool, deliver: bool, ticks: int | None) -> None:
        self.run(lambda: self.openai_sim.transcribe(fail=fail, deliver=deliver, ticks=ticks))

    @precondition(lambda self: self.server_session() is not None and bool(self.openai_sim.server.network.in_flight()))
    @rule(count=st.one_of(st.none(), st.integers(min_value=1, max_value=3)), ticks=TICKS)
    def deliver(self, count: int | None, ticks: int | None) -> None:
        self.run(lambda: self.openai_sim.deliver(count=count, ticks=ticks))

    @precondition(lambda self: self.live())
    @rule(kind=st.sampled_from(['content', 'response']), ticks=TICKS)
    def reject_next(self, kind: Literal['content', 'response'], ticks: int | None) -> None:
        self.run(lambda: self.openai_sim.reject_next(kind=kind, ticks=ticks))

    # A fault armed now can hit a send made mid-response, so it is left out where drops are restricted.
    @precondition(lambda self: self.can_drop() and self.openai_sim.openai.dialect != 'xai')
    @rule(fault=st.sampled_from(['lost', 'ambiguous']), ticks=TICKS)
    def fail_next_send(self, fault: SendFault, ticks: int | None) -> None:
        self.run(lambda: self.openai_sim.fail_next_send(fault=fault, ticks=ticks))

    @precondition(lambda self: self.can_drop())
    @rule(refuse_dials=st.integers(min_value=0, max_value=2), ticks=TICKS)
    def drop(self, refuse_dials: int, ticks: int | None) -> None:
        self.run(lambda: self.openai_sim.drop(refuse_dials=refuse_dials, ticks=ticks))


class AzureMachine(OpenAIMachine):  # pragma: lax no cover (driven only by randomized exploration)
    """Randomized traces against the simulated server speaking as Azure OpenAI."""

    provider_options = _options('azure')


class XaiMachine(OpenAIMachine):  # pragma: lax no cover (driven only by randomized exploration)
    """Randomized traces against the simulated server speaking as xAI Grok Voice."""

    provider_options = _options('xai')
