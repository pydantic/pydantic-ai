"""The realtime session simulator's tests: randomized exploration, pinned findings, and baseline conversations.

- `test_exploration` runs each provider's state machine through randomized interleavings of client
  operations, provider behavior, and faults, checking every invariant after every step. It tolerates the
  known findings in `_findings.py` and fails on anything else. In CI it runs a small, derandomized batch;
  see `__init__.py` for running it long.
- The `test_known_*` tests pin each known finding to a minimal scenario, as a strict expected failure:
  when the fix lands, the test starts passing and fails as `XPASS(strict)`, which is the prompt to remove
  the finding from `_findings.py` and drop the `xfail` mark, keeping the scenario as a regression test.
- The `test_baseline_*` tests are conversations every provider must get through without a single
  violation, strictly: they keep the simulator itself honest.
- The `test_scenario_*` tests walk each simulated server through its faults and edge behaviors
  deterministically, tolerating the known findings they run into like exploration does.
"""

from __future__ import annotations as _annotations

import os
from collections.abc import Callable
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai.messages import ModelResponse, SpeechPart, ToolCallPart, ToolReturnPart

from ...conftest import try_import

with try_import() as imports_successful:
    from hypothesis import HealthCheck, settings
    from hypothesis.stateful import (
        RuleBasedStateMachine,
        run_state_machine_as_test,  # pyright: ignore[reportUnknownVariableType]
    )

    from ._elevenlabs_agents import ElevenLabsAgentsMachine, ElevenLabsAgentsOptions, ElevenLabsAgentsSimulation
    from ._findings import FINDINGS_BY_ID, KNOWN_FINDINGS
    from ._gemini import GeminiBehavior, GeminiMachine, GeminiSimulation
    from ._live import LiveMachine, LiveSimulation
    from ._machine import KNOWN_HIT_COUNTS
    from ._openai_server import Dialect
    from ._openai_simulation import (
        AzureMachine,
        ElevenLabsMachine,
        OpenAIMachine,
        OpenAIOptions,
        OpenAISimulation,
        XaiMachine,
    )
    from ._simulation import FindingReproduced, SessionOptions, Simulation
    from ._wire import SendFault

pytestmark = pytest.mark.skipif(not imports_successful(), reason='realtime provider SDKs or hypothesis not installed')


def exploration_settings() -> settings:
    """A small, derandomized batch by default; `REALTIME_SIMULATION_EXAMPLES` for a long, random run."""
    examples = os.environ.get('REALTIME_SIMULATION_EXAMPLES')
    return settings(
        max_examples=int(examples) if examples else 50,
        stateful_step_count=int(os.environ.get('REALTIME_SIMULATION_STEPS', '25')),
        derandomize=examples is None,
        database=None,
        deadline=None,
        print_blob=True,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )


@pytest.mark.parametrize(
    'machine',
    [
        pytest.param(OpenAIMachine, id='openai'),
        pytest.param(AzureMachine, id='azure'),
        pytest.param(XaiMachine, id='xai'),
        pytest.param(ElevenLabsMachine, id='elevenlabs'),
        pytest.param(GeminiMachine, id='gemini'),
        pytest.param(LiveMachine, id='gpt-live'),
        pytest.param(ElevenLabsAgentsMachine, id='elevenlabs-agents'),
    ]
    if imports_successful()
    else [],
)
@pytest.mark.slow(reason='explores 50 randomized sessions of up to 25 steps against each provider simulator')
def test_exploration(machine: type[RuleBasedStateMachine]) -> None:
    KNOWN_HIT_COUNTS.clear()
    run_state_machine_as_test(machine, settings=exploration_settings())
    if os.environ.get('REALTIME_SIMULATION_EXAMPLES'):  # pragma: no cover (a long run, by hand)
        print(f'\nknown findings hit ({machine.__name__}):')
        for hit, count in KNOWN_HIT_COUNTS.most_common():
            print(f'  {count:6} {hit}')


# --- known findings, pinned ------------------------------------------------------------------------


PINNED: set[str] = set()


def known(finding_id: str) -> pytest.MarkDecorator:
    """Mark a scenario as reproducing a known finding: it must raise `FindingReproduced` until the fix lands."""
    PINNED.add(finding_id)
    if not imports_successful():  # pragma: lax no cover (the module is skipped)
        return pytest.mark.xfail(reason=finding_id)
    return pytest.mark.xfail(raises=FindingReproduced, strict=True, reason=str(FINDINGS_BY_ID[finding_id]))


def reproduce(finding_id: str, sim: Simulation, scenario: Callable[[Any], object]) -> None:
    # Other known findings the scenario meets on the way are tolerated, even in a strict run.
    sim.strict = False
    with sim.enforcing(finding_id) as s:
        scenario(s)


def test_merged_requests_release_their_reservations() -> None:
    """Two turns typed while the first is answered: the connection merges their requests into one (OR3, #8765).

    A wait taken then is owed both turns' replies, which the one merged response answers.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.send_text()
        sim.send_text()
        sim.wait_for_reply()

    run_clean(OpenAISimulation(), scenario)


def test_a_response_taken_for_ours_before_the_server_echoed_settles_nothing_twice() -> None:
    """Server VAD answers a spoken turn while our request is outstanding, before the server has echoed metadata.

    The connection takes that response for the one it asked for (it can't tell yet), and the response that
    really answers the request then settles nothing a second time (`lifecycle.input_settled_twice`).
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.reject_next(kind='response')
        sim.create_response()
        sim.create_response()
        sim.send_audio(chunks=1)
        sim.speech_start(deliver=False)
        sim.speech_stop(deliver=False)
        sim.finish(deliver=False)

    run_clean(OpenAISimulation(openai=OpenAIOptions(transcription=False)), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai', 'elevenlabs'])
def test_raising_tool_ends_the_exchange(dialect: Dialect) -> None:
    """A tool that raises ends the exchange, and nothing is left waiting (OR8, #8765).

    The turn ends with it: its failure doesn't go out, so no reply is coming to end it, and the turn completes as
    the tool raises, once. A reply asked for afterwards completes a turn of its own.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.finish()
        assert sim.turn_completes == []
        sim.finish_tool(outcome='error')
        sim.settle()
        assert len(sim.turn_completes) == 1
        sim.send_text()
        sim.wait_for_reply()
        sim.settle()
        assert len(sim.turn_completes) == 2

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect)), scenario)


def test_pruned_tool_round_leaves_the_server_conversation_and_stays_in_history() -> None:
    """With `openai_prune_tool_items`, an answered tool round is deleted from the server conversation, not history.

    A reconnect after it replays the conversation without the round, and nothing it deleted was still needed.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.finish()
        sim.finish_tool()
        sim.settle()  # (the server answers the result)
        assert sim.truth.pruned, 'the round was deleted from the server conversation'
        sim.drop()
        sim.settle()  # (reconnected)
        session = sim.server.session
        assert session is not None and not set(sim.truth.pruned) & session.conversation, 'the replay leaves them out'
        sim.send_text()

    run_clean(OpenAISimulation(openai=OpenAIOptions(prune_tool_items=True)), scenario)


def test_kept_tool_round_stays_in_the_server_conversation() -> None:
    """A tool named in `openai_keep_tool_items` keeps its round in the server conversation while pruning."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.finish()
        sim.finish_tool()
        sim.settle()  # (the server answers the result)
        assert not sim.truth.pruned, 'the kept round stays'

    run_clean(
        OpenAISimulation(openai=OpenAIOptions(prune_tool_items=True, keep_tool_items=frozenset({'lookup'}))), scenario
    )


@known('SIM-10')
def test_known_late_cancel_drops_a_finished_reply() -> None:
    """The reply finished on the server before the cancel reached it, but the client hadn't read it yet."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.send_audio()
        sim.speak(deliver=False)
        sim.finish(deliver=False)
        sim.interrupt(mode='cancel')
        sim.settle()

    reproduce('SIM-10', OpenAISimulation(openai=OpenAIOptions(turn_detection='manual')), scenario)


def test_turn_spoken_before_a_reply_but_committed_after_it_is_filed_after_it() -> None:
    """The user started talking, the model answered a typed turn, and only then was the spoken turn committed (SIM-11, fixed by the session core)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.send_text()
        sim.settle()
        sim.commit_audio()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(turn_detection='manual')), scenario)


def test_second_commit_keeps_the_turn_of_earlier_audio_in_place() -> None:
    """SIM-11, fixed by the session core.

    Audio streamed before each of two replies, committed as one turn after them, then a second turn committed:
    the session gives the second turn the place of the audio before the second reply.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.send_text()
        sim.settle()
        sim.send_audio()
        sim.send_text()
        sim.settle()
        sim.commit_audio()
        sim.send_audio()
        sim.commit_audio()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(turn_detection='manual')), scenario)


def test_live_utterance_transcribed_after_a_reply_is_filed_after_it() -> None:
    """On GPT-Live, audio streamed before the model's reply, transcribed only after it, is a turn after it (SIM-11, fixed by the session core)."""

    def scenario(sim: LiveSimulation) -> None:
        sim.send_audio()
        sim.speak()
        sim.settle()
        sim.user_says()
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_refused_tool_results_request_ends_the_wait() -> None:
    """SIM-12, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.reject_next('response')
        sim.finish()
        sim.settle()

    run_clean(OpenAISimulation(), scenario)


def test_turn_committed_by_hand_under_server_vad_is_filed_where_the_provider_put_it() -> None:
    """Server VAD hears the user start; the app commits the buffer by hand before VAD commits the rest (OR9, fixed by the session core)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.commit_audio()
        sim.settle()

    run_clean(OpenAISimulation(), scenario)


def test_gemini_cut_off_tool_turn_leaves_the_wait_to_the_reply() -> None:
    """SIM-13: a typed turn cuts off a turn waiting on tool results, whose calls Gemini cancels.

    The wait is owed the reply to the typed turn, which the next turn gives, not the cut-off one (seen live on
    `gemini-3.1-flash-live-preview`: `tool_call_cancellation`, `interrupted`, `turn_complete`, then a new turn).
    """

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(deliver=False)
        sim.send_text()
        sim.wait_for_reply()
        sim.settle()

    run_accepting(GeminiSimulation(behavior=GeminiBehavior(text_cuts_off_reply=True)), scenario, 'SIM-39')


def async_gemini() -> GeminiSimulation:
    """Gemini 2.5 native audio with asynchronous tool calls: the model keeps talking after a call."""
    return GeminiSimulation(behavior=GeminiBehavior(async_tool_calls=True))


def test_gemini_async_filler_recorded_before_the_result() -> None:
    """S0: the model narrates after the call and finishes; the result then cuts in, and the model answers.

    Recorded `[call, filler] {return} [answer]`, as it was said.
    """

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools()
        sim.speak()
        sim.finish()
        sim.settle()

    run_clean(async_gemini(), scenario)


def test_gemini_async_result_cuts_into_speech() -> None:
    """S2: the result arrives while the model is still narrating; what it said before belongs to the call."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools()
        sim.speak()
        sim.finish_tool()
        sim.settle()

    run_clean(async_gemini(), scenario)


def test_turn_heard_before_a_cancelled_reply_ended_is_filed_after_it() -> None:
    """Server VAD hears the user start while a reply the app cancelled (after it called a tool) is still ending (SIM-11, fixed by the session core)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.interrupt(mode='cancel')
        sim.call_tool()
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.settle()

    run_clean(OpenAISimulation(), scenario)


def test_gemini_raising_tool_completes_the_turn_once() -> None:
    """A tool that raises ends the turn once on Gemini Live too, and a reply asked for afterwards completes its own."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools()
        sim.finish_tool(outcome='error')
        sim.settle()
        assert len(sim.turn_completes) == 1
        sim.send_text()
        sim.wait_for_reply()
        sim.settle()
        assert len(sim.turn_completes) == 2

    run_clean(GeminiSimulation(), scenario)


def test_gemini_tool_batch_answer_lost_to_a_drop_is_settled() -> None:
    """Every result of the batch went out, then the connection dropped before the answer (SIM-1, fixed by the session core).

    On OpenAI the reconnect asks for that answer again; Gemini doesn't resume a generation after a re-dial, so the
    answer is lost with the connection.
    """

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(count=2)
        sim.finish_tool()
        sim.finish_tool()
        sim.drop()
        sim.settle()

    run_clean(GeminiSimulation(), scenario)


def test_refused_context_leaves_the_spoken_turn_in_place() -> None:
    """SIM-14, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.reject_next('content')
        sim.speech_start(deliver=False)
        sim.send_image()
        sim.settle()

    run_clean(OpenAISimulation(), scenario)


def test_raising_tool_ends_the_wait_for_a_deferred_request() -> None:
    """SIM-15, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.create_response()
        sim.call_tool()
        sim.finish_tool(outcome='error')
        sim.settle()

    run_clean(OpenAISimulation(), scenario)


def test_tool_result_over_request_limit_ends_the_wait() -> None:
    """SIM-15, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_image(respond=True)
        sim.call_tool(deliver=False)
        sim.send_image(respond=True)
        sim.settle()

    run_clean(OpenAISimulation(options=SessionOptions(request_limit=2)), scenario)


@pytest.mark.parametrize('ending', ['request_limit', 'tool_error'])
def test_turn_held_behind_a_reply_is_recorded_when_a_session_ended_on_its_own_closes(ending: str) -> None:
    """A turn committed while a reply streams is held behind that reply, which a session that ended on its own settles as it closes.

    Here the session ends on the reply's tool call: the request its result would make passes `request_limit`, or the
    tool raises.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.send_image(respond=True)
        sim.commit_audio()
        sim.call_tool()
        if ending == 'tool_error':
            sim.finish_tool(outcome='error')
        sim.settle()

    run_clean(
        OpenAISimulation(
            options=SessionOptions(request_limit=1 if ending == 'request_limit' else None),
            openai=OpenAIOptions(turn_detection='manual', dialect='azure'),
        ),
        scenario,
    )


def test_cleared_barge_in_releases_the_dropped_request() -> None:
    """SIM-16, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.create_response()
        sim.create_response()
        sim.speech_start(deliver=False)
        sim.clear_audio()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(transcription=False)), scenario)


def test_extended_thinking_parallel_calls_are_answered_once() -> None:
    """SIM-17, fixed by the session core."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_audio()
        sim.user_speaks()
        sim.call_tools(count=2)
        sim.settle()

    run_clean(GeminiSimulation(behavior=GeminiBehavior(stalls_in_progress=True)), scenario)


def test_gemini_async_second_call_in_the_same_turn() -> None:
    """The model makes a second asynchronous call in the turn of the first: both are in the one response."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(deliver=False)
        sim.call_tools()

    run_clean(async_gemini(), scenario)


def _live_history(sim: LiveSimulation) -> list[list[str]]:
    """Each recorded message's parts: what was said, and the tool calls and returns, by call id."""
    session = sim.session
    assert session is not None
    return [
        [
            part.transcript or ''
            if isinstance(part, SpeechPart)
            else f'{part.part_kind} {part.tool_call_id}'
            if isinstance(part, ToolCallPart | ToolReturnPart)
            else part.part_kind
            for part in message.parts
        ]
        for message in session.all_messages()
    ]


def test_live_reply_goes_on_across_a_delegated_round() -> None:
    """SIM-18: Live goes on speaking while the backend waits for its tool's result ("Okay, checking ... now.", seen
    live): what it says is the reply that called the tool, ahead of the result, and only the answer follows it."""

    def scenario(sim: LiveSimulation) -> None:
        sim.speak()
        sim.delegate()
        sim.backend_call()
        sim.backend_finish()
        sim.advance_time(0.1)
        sim.finish_tool()
        sim.speak()
        sim.settle()
        history.extend(_live_history(sim))

    history: list[list[str]] = []
    run_clean(LiveSimulation(), scenario)
    assert history == snapshot([['r1w1', 'tool-call call_1', ' r1w2'], ['tool-return call_1'], ['r2w1']])


def test_live_continuation_call_is_a_response_of_its_own() -> None:
    """SIM-18: a call the backend makes in a continuation, with the results it was continued with, follows them."""

    def scenario(sim: LiveSimulation) -> None:
        sim.speak(deliver=False)
        sim.delegate(deliver=False)
        sim.backend_call(deliver=False)
        sim.backend_finish(deliver=False)
        sim.advance_time(0.1)
        sim.finish_tool()
        sim.backend_call(deliver=False)
        sim.settle()
        history.extend(_live_history(sim))

    history: list[list[str]] = []
    run_clean(LiveSimulation(), scenario)
    assert history == snapshot(
        [['r1w1', 'tool-call call_1'], ['tool-return call_1'], ['tool-call call_2'], ['tool-return call_2'], ['r3w1']]
    )


def test_live_delegation_settling_while_another_is_owed_ends_the_reply() -> None:
    """A delegation that settles while another one's tool results are still owed ends the reply that called the tools.

    What Live says next is the settled delegation's outcome, a reply of its own, ahead of the other one's answer.
    """

    def scenario(sim: LiveSimulation) -> None:
        sim.speak()
        sim.delegate()
        sim.backend_call()
        sim.backend_finish()
        sim.delegate()
        sim.backend_finish()
        sim.speak()
        sim.advance_time(0.1)
        sim.finish_tool()
        sim.settle()
        history.extend(_live_history(sim))

    history: list[list[str]] = []
    run_clean(LiveSimulation(), scenario)
    assert history == snapshot([['r1w1', 'tool-call call_1'], ['tool-return call_1'], ['r2w1'], ['r3w1']])


def test_live_speech_before_a_continuation_call_stays_in_the_reply() -> None:
    """SIM-18: what Live says after the round, before the backend calls again, is the reply that called the tools."""

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.backend_call()
        sim.backend_finish()
        sim.advance_time(0.1)
        sim.finish_tool()
        sim.speak()
        sim.backend_call()
        sim.settle()
        history.extend(_live_history(sim))

    history: list[list[str]] = []
    run_clean(LiveSimulation(), scenario)
    assert history == snapshot(
        [['tool-call call_1', 'r1w1'], ['tool-return call_1'], ['tool-call call_2'], ['tool-return call_2'], ['r3w1']]
    )


def test_gemini_cut_off_unstarted_turn_leaves_the_wait_to_the_reply() -> None:
    """SIM-13: two spoken turns the model hadn't started answering, then a typed one that cuts them off.

    The cut-off turn said nothing, so it answers nothing: the typed turn's reply is the next turn.
    """

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_audio()
        sim.user_speaks(deliver=False)
        sim.send_audio()
        sim.user_speaks(deliver=False)
        sim.send_text()
        sim.wait_for_reply()
        sim.settle()

    run_accepting(GeminiSimulation(behavior=GeminiBehavior(text_cuts_off_reply=True)), scenario, 'SIM-39')


def test_gemini_async_batch_of_three_is_answered_once() -> None:
    """SIM-19, fixed by the session core."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(count=3)
        sim.settle()

    run_clean(async_gemini(), scenario)


@known('G3b')
def test_known_typed_turn_during_reconnect_fails() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.drop(ticks=0)
        sim.send_text()
        sim.settle()

    reproduce('G3b', OpenAISimulation(), scenario)


def test_barge_in_on_a_tool_round_releases_the_deferred_request() -> None:
    """SIM-20, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.create_response()
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.settle()

    run_clean(OpenAISimulation(), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai'])
def test_late_terminal_of_a_barged_in_reply_stays_its_own(dialect: Dialect) -> None:
    """The `response.done` of a reply the user cut off arrives after the next reply started (#8801).

    Each reply is assembled under its own id: the late terminal and its usage end the reply it names, not the one
    streaming.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.send_audio()
        sim.speech_start(late=True)
        sim.speech_stop()
        sim.speak()
        sim.release_late_done()
        sim.speak()
        sim.finish()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect)), scenario)


def test_late_terminal_of_a_barged_in_reply_lands_while_the_next_calls_a_tool() -> None:
    """The cut-off reply's terminal arrives between the next reply's tool call and its own terminal (#8801).

    It neither closes the tool-call reply early nor lands its usage on it.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.send_audio()
        sim.speech_start(late=True)
        sim.speech_stop()
        sim.speak()
        sim.call_tool()
        sim.release_late_done()
        sim.finish()

    run_clean(OpenAISimulation(), scenario)


def test_late_terminal_of_a_barged_in_reply_lands_after_the_next_ended() -> None:
    """The cut-off reply's terminal arrives only after the next reply ended (#8801): it records the cut-off reply."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.send_audio()
        sim.speech_start(late=True)
        sim.speech_stop()
        sim.speak()
        sim.finish()
        sim.release_late_done()

    run_clean(OpenAISimulation(), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'xai'])
def test_request_whose_refusal_is_lost_is_settled_by_the_reconnect(dialect: Dialect) -> None:
    """SIM-21, fixed by the session core: on xAI too, now that its reconnect replays history instead of resuming."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.reject_next('response')
        sim.create_response()
        sim.send_text()
        sim.drop()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect)), scenario)


@known('SIM-23')
def test_known_close_cuts_off_a_terminal_mid_frame() -> None:
    """The reply's `response.done` sends the deferred request, and the close lands while that send is in flight."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak(deliver=False)
        sim.create_response()
        sim.finish(ticks=0)
        sim.close()
        sim.settle()

    # Seeded, since each send draws a latency, which is what holds the deferred request's send open.
    reproduce('SIM-23', OpenAISimulation(seed=5, options=SessionOptions(latency=True)), scenario)


@known('SIM-22')
def test_known_terminal_read_as_the_connection_drops_loses_its_usage() -> None:
    """The cancelled reply's `response.done` is read right before the drop (found by exploration on the refactor)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.clear_audio()
        sim.create_response()
        for _ in range(15):  # (each send draws a latency, which is what lines the drop up with the read)
            sim.clear_audio()
        sim.interrupt(mode='cancel')
        sim.send_audio()
        sim.create_response()
        sim.speech_start(ticks=0)
        sim.drop()
        sim.settle()

    reproduce(
        'SIM-22',
        OpenAISimulation(options=SessionOptions(latency=True), openai=OpenAIOptions(transcription=False)),
        scenario,
    )


@known('SIM-4')
def test_known_failed_deferred_create_after_a_refusal_keeps_its_reservation() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.reject_next('response')
        sim.create_response()
        sim.create_response()
        sim.fail_next_send()
        sim.settle()

    reproduce('SIM-4', OpenAISimulation(), scenario)


@known('SIM-4')
def test_known_xai_held_audio_failing_after_a_reply_drops_its_terminal() -> None:
    """Audio sent during a reply is held back, and goes out from the receive loop once the reply ends."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.fail_next_send()
        sim.send_audio()
        sim.settle()

    reproduce(
        'SIM-4',
        OpenAISimulation(openai=OpenAIOptions(dialect='xai', turn_detection='manual', transcription=False)),
        scenario,
    )


@known('SIM-4')
def test_known_xai_held_audio_failing_before_a_deferred_request_loses_it() -> None:
    """The held audio goes out ahead of the request deferred behind the reply, which the failure takes with it."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.create_response()
        sim.fail_next_send()
        sim.send_audio()
        sim.settle()

    reproduce(
        'SIM-4',
        OpenAISimulation(openai=OpenAIOptions(dialect='xai', turn_detection='manual', transcription=False)),
        scenario,
    )


@known('SIM-4')
def test_known_xai_clear_before_a_deferred_request_failing_loses_it() -> None:
    """The clear xAI needs before a deferred request goes out from the receive loop, once the reply ends."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.clear_audio()
        sim.create_response()
        sim.fail_next_send()
        sim.send_audio()
        sim.clear_audio()
        sim.settle()

    reproduce(
        'SIM-4',
        OpenAISimulation(openai=OpenAIOptions(dialect='xai', turn_detection='manual', transcription=False)),
        scenario,
    )


def test_barge_in_keeps_a_request_vad_will_not_answer() -> None:
    """SIM-32, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.create_response()
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(vad_responds=False, transcription=False)), scenario)


def test_turn_spoken_while_a_cancelled_reply_ends_is_filed_after_it() -> None:
    """SIM-11, fixed by the session core.

    The user starts talking after the app cancelled a reply, before the provider has ended it.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_image(respond=True)
        sim.interrupt(mode='cancel')
        sim.speak()
        sim.send_audio()
        sim.settle()
        sim.commit_audio()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(turn_detection='manual', dialect='azure')), scenario)


def test_gemini_async_result_cut_in_does_not_end_the_wait() -> None:
    """The model finishes speaking after an async call; the result then cuts in, before the model answers it (SIM-33, fixed by the session core)."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools()
        sim.wait_for_reply()
        sim.finish(deliver=False)
        sim.settle()

    run_clean(async_gemini(), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai'])
@pytest.mark.parametrize('transcription', [True, False])
@pytest.mark.parametrize('ending', ['close', 'drop', 'drop_for_good'])
@pytest.mark.parametrize('acknowledged', [True, False])
def test_turn_committed_before_the_end_is_recorded(
    dialect: Dialect, transcription: bool, ending: str, acknowledged: bool
) -> None:
    """SIM-25 and SIM-34, fixed: a turn `commit_audio()` committed is recorded, without the transcript or reply
    still to come, when the session closes or the connection drops, even before the client read the provider's
    commit."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.commit_audio()
        if acknowledged:
            sim.deliver()
        if ending == 'close':
            sim.close()
        else:
            sim.drop(refuse_dials=2 if ending == 'drop_for_good' else 0)
        sim.settle()

    run_clean(
        OpenAISimulation(openai=OpenAIOptions(dialect=dialect, turn_detection='manual', transcription=transcription)),
        scenario,
    )


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai'])
@pytest.mark.parametrize('ending', ['close', 'drop'])
def test_hand_commit_unacknowledged_under_vad_ends_the_turn_under_way(dialect: Dialect, ending: str) -> None:
    """A hand commit server VAD heard the start of is that one turn, not a second, when the session ends first."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.commit_audio()
        if ending == 'close':
            sim.close()
        else:
            sim.drop()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect, transcription=False)), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai'])
@pytest.mark.parametrize('ending', ['close', 'drop'])
def test_hand_commit_of_nothing_after_a_vad_commit_is_no_turn(dialect: Dialect, ending: str) -> None:
    """Server VAD committed the audio already, so a hand commit unacknowledged when the session ends commits nothing."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop()
        sim.settle()
        sim.commit_audio()
        if ending == 'close':
            sim.close()
        else:
            sim.drop()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect, transcription=False)), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai'])
def test_hand_commit_of_unheard_audio_after_a_vad_commit_is_no_turn(dialect: Dialect) -> None:
    """Retired SIM-40, as first pinned: under server VAD, the provider commits only speech its VAD heard (checked
    live), so a hand commit of audio sent after VAD committed the turn before it commits nothing, acknowledged or not.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.speech_stop(deliver=False)
        sim.send_audio()
        sim.commit_audio()
        sim.deliver(3)
        sim.close()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect, transcription=False)), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai'])
@pytest.mark.parametrize('ending', ['close', 'drop'])
def test_unacknowledged_hand_commit_of_silence_under_vad_is_no_turn(dialect: Dialect, ending: str) -> None:
    """Under server VAD, a hand commit sent after VAD committed the turn (audio streaming on in the silence after it)
    commits nothing, so the session ending before the provider refuses it records no turn for it."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop()
        sim.settle()
        sim.send_audio()
        sim.commit_audio()
        if ending == 'close':
            sim.close()
        else:
            sim.drop()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect, transcription=False)), scenario)


@pytest.mark.parametrize('ending', ['close', 'drop'])
def test_scenario_unacknowledged_commit_of_speech_not_yet_heard_starting_is_accepted(ending: str) -> None:
    """SIM-42: a commit of speech whose start the client hadn't read yet, unacknowledged when the session ends, can't
    be told from a commit of silence, so it makes no turn (shrunk from exploration)."""

    with OpenAISimulation(
        openai=OpenAIOptions(dialect='azure', vad_interrupts=False, vad_responds=False), strict=False
    ) as sim:
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.commit_audio()
        if ending == 'close':
            sim.close()
        else:
            sim.drop()
        sim.settle()
        assert ('SIM-42', 'history.turn_missing') in sim.checker.known_hits


@pytest.mark.parametrize('dialect', ['openai', 'azure'])
def test_hand_commit_mid_speech_after_an_empty_one_keeps_its_turn(dialect: Dialect) -> None:
    """A hand commit sent as VAD stopped is refused as empty, the user goes on, and a second hand commit mid-speech
    commits that speech; the session closes before its acknowledgement is read. Each turn is recorded once (the
    probe frames this follows are pinned in `test_openai_lifecycle.py`)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop(deliver=False)
        sim.commit_audio()
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.commit_audio()
        # VAD's stop, commit and item, the refusal of the first hand commit, and the second turn's start: not its
        # commit.
        sim.deliver(5)
        sim.close()
        sim.settle()

    run_clean(
        OpenAISimulation(openai=OpenAIOptions(dialect=dialect, transcription=False, vad_responds=False)), scenario
    )


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai'])
@pytest.mark.parametrize('ending', ['close', 'drop'])
def test_hand_commit_of_nothing_raced_by_a_vad_stop_is_no_turn(dialect: Dialect, ending: str) -> None:
    """A hand commit that went out as server VAD committed the same audio, with neither acknowledgement read when the
    session ends, isn't taken for a second turn: it may well have committed nothing."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.speech_stop(deliver=False)
        sim.commit_audio()
        sim.deliver(3 if dialect == 'xai' else 2)
        if ending == 'close':
            sim.close()
        else:
            sim.drop()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect, transcription=False)), scenario)


@pytest.mark.parametrize('transcription', [False, True])
@pytest.mark.parametrize('reopens', [1, 2])
def test_xai_reopening_a_turn_after_its_own_stop(transcription: bool, reopens: int) -> None:
    """SIM-41, recorded live (October 2026): xAI hears the user stop and starts its reply, they speak again before it
    commits the turn, and it reports `speech_started` for the same item and drops that reply without a
    `response.done`. It then stops, starts another reply, and commits the turn once. The turn is recorded once,
    each dropped reply ends cancelled, and the reply that follows is recorded (also with the frames of two re-opens
    read together)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        for _ in range(reopens):
            sim.speech_stop(commit=False, deliver=reopens == 1)
            sim.send_audio()
            sim.speech_start(deliver=reopens == 1)
        sim.speech_stop(commit=False)
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect='xai', transcription=transcription)), scenario)


def test_xai_reopening_a_turn_it_started_no_reply_for() -> None:
    """A reply asked for while the user spoke is under way when xAI hears them stop, so it starts none of its own
    then: the turn it re-opens has no reply to drop."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.send_text()
        sim.speech_stop(commit=False)
        sim.send_audio()
        sim.speech_start()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect='xai', transcription=False)), scenario)


def test_xai_commits_a_stopped_turn_nobody_reopens() -> None:
    """The simulated xAI commits a turn whose stop it didn't commit at once, once nobody speaks again (as it comes to
    rest)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop(commit=False)
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect='xai', transcription=False)), scenario)


def test_turn_committed_after_a_racing_clear_is_recorded() -> None:
    """SIM-26, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio(chunks=3, ticks=0)
        sim.clear_audio()
        sim.commit_audio()
        sim.settle()

    run_clean(
        OpenAISimulation(
            options=SessionOptions(latency=True), openai=OpenAIOptions(turn_detection='manual', transcription=False)
        ),
        scenario,
    )


def test_hand_commit_under_server_vad_keeps_both_turns() -> None:
    """SIM-27, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.commit_audio()
        sim.settle()

    run_clean(
        OpenAISimulation(
            openai=OpenAIOptions(dialect='azure', transcription=False, vad_interrupts=False, vad_responds=False)
        ),
        scenario,
    )


def test_input_held_behind_a_lost_reply_is_replayed() -> None:
    """SIM-28, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.send_text(respond=False)
        sim.drop()
        sim.settle()
        sim.send_text()
        sim.settle()

    run_clean(OpenAISimulation(), scenario)


def test_clear_after_an_unread_vad_commit_keeps_the_turn() -> None:
    """SIM-35, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop(deliver=False)
        sim.clear_audio()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(transcription=False)), scenario)


def test_xai_clear_after_a_request_committed_speech_keeps_the_turn() -> None:
    """On xAI push-to-talk, the request a text turn sends commits the speech in the buffer before the clear (SIM-35)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.settle()
        sim.send_text()
        sim.clear_audio()
        sim.settle()

    run_clean(
        OpenAISimulation(openai=OpenAIOptions(dialect='xai', turn_detection='manual', transcription=False)), scenario
    )


@known('SIM-36')
def test_known_xai_push_to_talk_audio_after_a_repeated_terminal_is_held() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.create_response()
        sim.send_audio()
        sim.speak(deliver=False)
        sim.finish(deliver=False)
        sim.repeat_done()
        sim.send_audio()
        sim.settle()

    reproduce(
        'SIM-36',
        OpenAISimulation(openai=OpenAIOptions(dialect='xai', turn_detection='manual', transcription=True)),
        scenario,
    )


def test_xai_push_to_talk_turn_spoken_over_is_recorded() -> None:
    """SIM-29, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.create_response()
        sim.send_audio()
        sim.settle()

    run_clean(
        OpenAISimulation(openai=OpenAIOptions(dialect='xai', turn_detection='manual', transcription=False)),
        scenario,
    )


@known('SIM-30')
def test_known_xai_push_to_talk_request_deferred_behind_an_answer_is_dropped() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.send_text()
        sim.create_response()
        sim.settle()

    reproduce(
        'SIM-30',
        OpenAISimulation(openai=OpenAIOptions(dialect='xai', turn_detection='manual', transcription=False)),
        scenario,
    )


def test_xai_push_to_talk_turn_spoken_during_a_reply_is_filed_after_it() -> None:
    """SIM-31, fixed by the session core.

    xAI push-to-talk: the user talks during the reply to a held commit; the request after it commits that turn.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.commit_audio()
        sim.send_audio()
        sim.create_response()
        sim.settle()
        sim.create_response()
        sim.settle()

    run_clean(
        OpenAISimulation(openai=OpenAIOptions(dialect='xai', turn_detection='manual', transcription=True)),
        scenario,
    )


def test_late_transcript_holds_back_the_reply_instead_of_being_inserted() -> None:
    """E, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop()
        sim.speak()
        sim.finish()
        sim.transcribe()

    run_clean(OpenAISimulation(), scenario)


def test_repeated_terminal_is_not_recorded_as_a_new_response() -> None:
    """A robustness fault, not recorded provider behavior: the server sends a `response.done` twice (#8801).

    The repeat is about a response already recorded, so it records nothing, and its usage isn't counted twice.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.repeat_done()

    run_clean(OpenAISimulation(), scenario)


def test_started_reply_lost_to_a_drop_is_settled_openai() -> None:
    """The response had started (`response.created` read) when the socket dropped, so it is not asked for again (SIM-1, fixed by the session core)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.deliver()
        sim.drop()
        sim.settle()

    run_clean(OpenAISimulation(), scenario)


def test_server_vad_reply_whose_tool_result_was_lost_to_a_drop_does_not_hold_back_the_next() -> None:
    """A server-VAD reply calls a tool whose result is lost with the connection; the reply asked for after the reconnect is answered.

    The failed send is raised from `async for`, and the session goes on receiving over the re-dial (SIM-1 was a
    simulator that stopped reading at the first error the iterator raised).
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.fail_next_send(fault='lost')
        sim.speech_stop(deliver=False)
        sim.call_tool(deliver=False)
        sim.settle()
        sim.create_response()
        sim.wait_for_reply()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(transcription=False)), scenario)


def test_request_sent_after_a_tool_raised_is_answered_xai() -> None:
    """A tool raises; a request sent after that is answered, and waited for, as usual.

    The failure is raised from `async for`, and a consumer that keeps iterating keeps a working session (SIM-15 was
    a simulator that stopped reading at the first error the iterator raised).
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.call_tool()
        sim.finish_tool(outcome='error')
        sim.create_response()
        sim.wait_for_reply()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(transcription=False, dialect='xai')), scenario)


def test_live_call_that_raises_is_answered_with_its_failure() -> None:
    """A delegated round of three calls, one of which raises while the backend response that asked is still running.

    The backend resumes only from every output of its round, and Live can't cancel the delegation, so the failure
    history records goes out as the call's output: the round continues, and a reply asked for afterwards is answered
    and ends. (SIM-43: each result asked for a reply of its own that never came, and the delegation suspended the
    turn clock for good.)
    """

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate(deliver=False)
        sim.backend_call(deliver=False)
        sim.backend_call(deliver=False)
        sim.backend_call()
        failed = sim.finish_tool(outcome='error')
        sim.settle()
        assert sim.server.truth.tool_calls[failed].output_received
        sim.send_text()
        sim.wait_for_reply()
        sim.settle()

    sim = LiveSimulation()
    run_clean(sim, scenario)
    # The round's results continued the delegation: the backend answered twice, and then once more for the text.
    assert len(sim.server.truth.usage_read) == 3


def test_live_wait_for_reply_waits_for_the_answer_to_a_failed_call() -> None:
    """A tool that raised still gets its failure sent on GPT-Live, so the exchange goes on to the answer.

    `wait_for_reply()` waits for the reply that answers the failure, as it does for a result that succeeded, rather
    than returning as the tool raises, with the answer then coming as a reply nobody asked for.
    """

    def scenario(sim: LiveSimulation) -> None:
        sim.send_text()
        sim.backend_call()
        sim.backend_finish()
        waiter = sim.wait_for_reply()
        sim.finish_tool(outcome='error')
        assert waiter.returned is None
        sim.settle()
        assert waiter.returned is not None

    sim = LiveSimulation()
    run_clean(sim, scenario)
    assert sim.exchanges_abandoned_at == []


def test_live_call_that_raises_after_its_siblings_answered_is_answered_with_its_failure() -> None:
    """The backend response has ended and the other results are out, waited for, when the last call raises (SIM-43)."""

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate(deliver=False)
        sim.backend_call(count=3)
        sim.backend_finish()
        sim.finish_tool()
        sim.finish_tool()
        sim.wait_for_reply()
        sim.finish_tool(outcome='error')
        sim.settle()
        sim.send_text()
        sim.wait_for_reply()
        sim.settle()

    run_clean(LiveSimulation(), scenario)


@known('SIM-37')
def test_known_turn_cleared_while_spoken_loses_its_transcript_xai() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.clear_audio()
        sim.settle()

    reproduce('SIM-37', OpenAISimulation(openai=OpenAIOptions(dialect='xai')), scenario)


@known('SIM-38')
def test_known_stop_server_vad_takes_back_holds_history() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop(commit=False)
        sim.speech_start()
        sim.speech_stop()
        sim.settle()

    reproduce('SIM-38', OpenAISimulation(), scenario)


def test_gemini_spoken_turn_is_not_inserted_into_recorded_history() -> None:
    """E, fixed by the session core."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_audio()
        sim.send_text(respond=False)
        sim.user_speaks(finished=False, deliver=False)
        sim.settle()

    run_clean(GeminiSimulation(), scenario)


def test_live_spoken_turn_is_not_inserted_into_recorded_history() -> None:
    """E, fixed by the session core."""

    def scenario(sim: LiveSimulation) -> None:
        sim.send_audio()
        sim.send_text(respond=False)
        sim.user_says(deliver=False)
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_gemini_turn_spoken_after_a_reply_is_filed_after_it() -> None:
    """SIM-11, fixed by the session core."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.send_audio()
        sim.settle()
        sim.user_speaks(finished=False, deliver=False)
        sim.settle()

    run_clean(GeminiSimulation(), scenario)


def test_live_text_sent_after_a_delegation_is_recorded_in_order() -> None:
    """SIM-2a, fixed by the session core."""

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate(deliver=False)
        sim.settle()
        sim.send_text(respond=False)
        sim.user_says(deliver=False)
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_gemini_turn_sent_during_a_tool_round_is_recorded_after_it() -> None:
    """SIM-2a, fixed by the session core: an input sent while a reply is owed is placed after that reply."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(deliver=False)
        sim.send_text(respond=False)
        sim.settle()

    run_clean(GeminiSimulation(), scenario)


@pytest.mark.parametrize('finished', [True, False])
def test_gemini_turn_heard_before_a_drop_is_kept(finished: bool) -> None:
    """The user spoke, and the connection dropped before the model replied: the turn stays, its reply is lost."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_audio()
        sim.user_speaks(finished=finished)
        sim.drop()
        sim.settle()

    run_clean(GeminiSimulation(), scenario)


def test_gemini_close_in_the_middle_of_a_tool_call_message_returns_every_call() -> None:
    """The session closes after running the first of two calls in one message: the second still gets a return."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_audio()
        sim.user_speaks(deliver=False)
        sim.call_tools(count=2, ticks=0)
        sim.close()
        sim.settle()

    run_clean(GeminiSimulation(), scenario)


def test_gemini_typed_turn_lost_to_a_drop_is_settled() -> None:
    """A typed turn the model hadn't answered when the connection dropped is settled (SIM-1, fixed for it by #8763)."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.drop()

    run_clean(GeminiSimulation(), scenario)


def test_turn_sent_after_a_reply_ended_unread_is_recorded_after_it() -> None:
    """The first reply ended empty before the second turn was sent, but the client hadn't read it yet (SIM-2a, fixed by the session core)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.finish(deliver=False)
        sim.send_text(respond=False)
        sim.settle()

    run_clean(OpenAISimulation(), scenario)


def test_wait_waits_for_a_started_vad_reply() -> None:
    """SIM-2b, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop()
        sim.wait_for_reply()

    run_clean(OpenAISimulation(), scenario)


def test_wait_waits_for_a_delegated_reply() -> None:
    """SIM-2b, fixed by the session core."""

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.wait_for_reply()

    run_clean(LiveSimulation(), scenario)


def test_reconnect_asks_again_for_an_unstarted_reply() -> None:
    """SIM-3, fixed by the session core."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.send_text()
        sim.drop()
        sim.settle()

    run_clean(OpenAISimulation(), scenario)


@known('SIM-4')
def test_known_failed_deferred_create_drops_the_terminal_frame() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.send_text()
        sim.fail_next_send()
        sim.settle()

    reproduce('SIM-4', OpenAISimulation(), scenario)


def test_gemini_parallel_calls_are_one_response_answered_once() -> None:
    """The calls of one `tool_call` message are one response, and their batch one reply (G2a/G2b, #8765)."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(count=2)

    run_clean(GeminiSimulation(), scenario)


def test_gemini_tool_turn_boundary_does_not_end_the_wait() -> None:
    """Vertex `gemini-live-2.5-flash` closes the tool-call turn before the answer (8766, fixed by #8766)."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools()
        sim.wait_for_reply()
        sim.finish_tool()
        sim.speak()
        sim.finish()

    run_clean(GeminiSimulation(behavior=GeminiBehavior(closes_tool_turn_separately=True)), scenario)


def test_gemini_resumed_session_settles_a_forgotten_tool_call() -> None:
    """The resumption handle predates the call the second turn makes (G6, fixed by #8763)."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.send_text()
        sim.call_tools()
        sim.drop()
        sim.advance_time(1)

    run_clean(GeminiSimulation(), scenario)


def test_send_during_reconnect_survives_openai() -> None:
    """Audio sent while the link is re-dialed is dropped, not raised (G3, fixed by #8806)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.drop(ticks=0)
        sim.send_audio()

    run_clean(OpenAISimulation(), scenario)


def test_send_during_reconnect_survives_gemini() -> None:
    """Audio sent while the link is re-dialed is dropped, not raised (G3, fixed by #8806)."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.drop(ticks=0)
        sim.send_audio()

    run_clean(GeminiSimulation(), scenario)


def test_live_parallel_calls_are_answered_once() -> None:
    """SIM-6, fixed by the session core."""

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.backend_call(count=2)
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_live_results_of_delegations_in_a_row_answered_together() -> None:
    """Two delegations each call a tool, and the model answers both results with one reply (SIM-6, fixed by the session core)."""

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.backend_call()
        sim.backend_finish()
        sim.delegate()
        sim.backend_call()
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_live_queued_text_answered_together() -> None:
    """SIM-7, fixed by the session core."""

    def scenario(sim: LiveSimulation) -> None:
        sim.send_text()
        sim.send_text()
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_live_drop_raises_a_realtime_error() -> None:
    """SIM-8: a dropped GPT-Live connection ends the session with a `RealtimeError`, as on the other providers."""

    def scenario(sim: LiveSimulation) -> None:
        sim.drop()

    sim = LiveSimulation()
    run_clean(sim, scenario)
    assert [repr(error) for error in sim.consumer_errors] == snapshot(
        [
            "RealtimeError('OpenAI GPT-Live connection closed: received 1006 (abnormal closure [internal]) simulated drop; no close frame sent')"
        ]
    )


def test_live_abandoned_calls_owe_nothing() -> None:
    """SIM-9, fixed by the session core."""

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.backend_call()
        sim.backend_finish(status='failed')
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_live_abandoned_call_still_running_doesnt_hold_the_exchange_open() -> None:
    """A tool call whose backend gave up still runs, but its result gets no reply: the exchange ends with the reply.

    Found live on Eleven Realtime, which abandons a call the same way when a new response starts: the turn only
    completed once the abandoned tool finished, if at all.
    """

    def scenario(sim: LiveSimulation) -> None:
        sim.send_text()
        sim.backend_call()
        sim.backend_finish(status='failed')
        # The reply that delegated is over, and the abandoned call owes nothing, though it still runs.
        assert sim.tools.pending()
        assert len(sim.turn_completes) == 1
        sim.send_text()
        waiter = sim.wait_for_reply(ticks=0)
        sim.backend_finish()
        sim.speak()
        sim.advance_time(1.0)
        # The reply to the interjection ends its exchange as it ends, not once the call finishes.
        assert sim.tools.pending()
        assert waiter.returned is not None
        assert len(sim.turn_completes) == 2
        sim.finish_tool()
        # Its result goes nowhere, and ends no exchange of its own.
        assert len(sim.turn_completes) == 2

    run_clean(LiveSimulation(), scenario)


def test_live_text_sent_as_a_delegation_starts_is_answered_by_its_reply() -> None:
    """8763c #3 on GPT-Live: text sent while a delegation the client hasn't read yet opens the reply.

    Live speaks to text in the reply under way (seen live, in every run), so that reply answers it, as the client
    takes it to.
    """

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate(deliver=False)
        sim.send_text()
        sim.wait_for_reply(ticks=0)
        sim.backend_finish()
        sim.speak()
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_live_text_sent_while_a_tool_round_is_half_answered_gets_a_delegation_of_its_own() -> None:
    """A typed turn's `response.create` while the round's other result is still out runs the backend on the text.

    It doesn't continue the round: the backend resumes only from every result of it.
    """

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.backend_call(count=2)
        sim.backend_finish()
        sim.finish_tool()
        sim.send_text()
        sim.wait_for_reply()
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_live_tool_result_content_goes_with_the_result_not_as_a_typed_turn() -> None:
    """A `ToolReturn`'s content follows its result as a message item, which isn't a typed turn of its own.

    Taken for one, it was queued for the next typed turn's `response.create`, which then took it for its own.
    """

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.backend_call()
        sim.backend_finish()
        sim.finish_tool(outcome='media')
        sim.settle()
        sim.send_text()
        sim.wait_for_reply()
        sim.settle()

    sim = LiveSimulation()
    run_clean(sim, scenario)
    assert [input_.kind for input_ in sim.server.truth.inputs] == ['tool_output', 'text']


def test_live_image_sent_for_a_response_is_answered_by_its_delegation() -> None:
    """`send(image, respond=True)` queues the image for the backend, and its `response.create` has Live run it."""

    def scenario(sim: LiveSimulation) -> None:
        sim.send_image(respond=True)
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_live_text_sent_while_a_reply_goes_on_is_answered_by_it() -> None:
    """Text sent mid-reply, which Live takes up there, is answered by that reply, which goes on to speak to it (seen live)."""

    def scenario(sim: LiveSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.send_text()
        sim.wait_for_reply(ticks=0)
        sim.speak()
        sim.settle()

    run_clean(LiveSimulation(), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai'])
def test_text_sent_as_server_vad_starts_a_reply_gets_its_own(dialect: Dialect) -> None:
    """8763c #3 on the OpenAI protocol: text sent just as server VAD starts a reply of its own.

    The reply started before the text joined the conversation, so it doesn't answer it. OpenAI refuses the request
    (after that reply is done) and the connection asks again; xAI cancels its own reply for ours (both seen live).
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop(deliver=False)
        sim.send_text()
        sim.wait_for_reply(ticks=0)
        # OpenAI's own reply, then ours; xAI's own reply is cancelled for ours.
        for _ in range(1 if dialect == 'xai' else 2):
            sim.speak()
            sim.finish()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(transcription=False, dialect=dialect)), scenario)


# --- baseline conversations -----------------------------------------------------------------------


def run_clean(sim: Simulation, scenario: Callable[[Any], object]) -> None:
    """Run a scenario that must hit no violation at all, known or not, then hang up and check it all again."""
    sim.strict = True
    with sim as s:
        scenario(s)
        s.settle()
        s.close()
        s.settle()
        s.check_handoff()


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai', 'elevenlabs'])
def test_baseline_openai_protocol_conversation(dialect: str) -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.wait_for_reply(ticks=0)
        sim.speak(chunks=2)
        sim.call_tool()
        sim.finish()
        sim.finish_tool()
        sim.speak()
        sim.finish()
        sim.play(chunks=3)
        sim.send_audio(chunks=2)
        sim.speech_start()
        sim.speech_stop()
        sim.transcribe()
        sim.speak()
        sim.finish()
        sim.settle()
        sim.send_text(respond=False)
        sim.send_image()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect)), scenario)  # pyright: ignore[reportArgumentType]


def test_baseline_openai_barge_in_and_reconnect() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak(chunks=3)
        sim.play()
        sim.interrupt(mode='played_bytes')
        sim.finish()
        sim.settle()
        sim.drop()
        sim.settle()
        sim.send_text()
        sim.speak()
        sim.finish()

    run_clean(OpenAISimulation(), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai'])
@pytest.mark.parametrize('cut', ['speech', 'cancel'])
def test_baseline_reply_cut_off_without_a_truncation_and_reconnect(dialect: Dialect, cut: str) -> None:
    """A reply cut off before any of it was played (by the user speaking, or a cancel) has nothing to truncate:
    a reconnect replays its transcript marked as interrupted, which restores it."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        if cut == 'speech':
            sim.speech_start()
        else:
            sim.interrupt(mode='cancel')
        sim.settle()
        sim.drop()
        sim.settle()
        sim.send_text()
        sim.settle()
        session = sim.server.session
        assert session is not None and 'assistant:r1w1 [Interrupted]' in session.conversation

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect)), scenario)


@pytest.mark.parametrize('fault', ['ambiguous', 'lost'])
@pytest.mark.parametrize(
    'mode,finished',
    [
        pytest.param('played_bytes', False, id='played_bytes-speaking'),
        pytest.param('played_bytes', True, id='played_bytes-finished'),
        # (A reply that finished keeps what history recorded under `played_ms`: see `interrupt()`.)
        pytest.param('played_ms', False, id='played_ms-speaking'),
    ],
)
def test_interrupt_whose_send_fails_records_the_cut(mode: str, finished: bool, fault: SendFault) -> None:
    """The link dies while the truncation is sent, so `interrupt()` raises, and the session reconnects.

    The truncation may have reached the provider (`ambiguous`) or not (`lost`): the session can't tell which. The
    listener heard only what they heard either way, so history records the reply as cut off there, rather than as
    heard in full or interrupted with no position, and the reconnect replays it as cut off.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.speak(chunks=2)
        if finished:
            sim.finish()
            sim.settle()
        sim.play()
        sim.fail_next_send(fault=fault)
        sim.interrupt(mode=mode)  # pyright: ignore[reportArgumentType]
        sim.settle()
        sim.send_text()

    # (The failed interrupt is known finding G3b, `send.failed_across_reconnect`, tolerated here.)
    run_tolerant(OpenAISimulation(openai=OpenAIOptions(turn_detection='manual', transcription=False)), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'azure'])
def test_interrupt_racing_the_end_of_the_reply_records_the_cut(dialect: str) -> None:
    """The reply ends on the provider while `interrupt(played_ms=...)` is on its way: the truncation still applies.

    The cancel comes too late to stop the reply, which the provider reports as completed, but the listener heard only
    what they heard, so history records it as cut off, as it does a reply that had ended before the interrupt.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.speak()
        sim.finish(deliver=False)
        sim.interrupt(mode='played_ms')

    run_clean(
        OpenAISimulation(openai=OpenAIOptions(dialect=dialect, turn_detection='manual', transcription=False)),  # pyright: ignore[reportArgumentType]
        scenario,
    )


@pytest.mark.parametrize('dialect', ['openai', 'azure'])
def test_cancel_after_a_cut_keeps_the_cut(dialect: str) -> None:
    """A plain `interrupt()` after one that truncated the reply: the reply stays cut where the first one cut it."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.speak()
        sim.interrupt(mode='played_ms')
        sim.interrupt(mode='cancel')

    run_clean(
        OpenAISimulation(openai=OpenAIOptions(dialect=dialect, turn_detection='manual', transcription=False)),  # pyright: ignore[reportArgumentType]
        scenario,
    )


def test_baseline_openai_barge_in_after_the_reply_finished_and_reconnect() -> None:
    """Generation outruns playback: the user cuts off a reply that already finished, which history records as cut
    off, and a reconnect replays it marked as such rather than as heard in full."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak(chunks=3)
        sim.finish()
        sim.settle()
        sim.play(chunks=2)
        sim.interrupt(mode='played_bytes')
        sim.settle()
        sim.drop()
        sim.settle()
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.settle()
        session = sim.session
        assert session is not None
        cuts.extend(
            (message.state, [part.interrupted_at_ms for part in message.parts if isinstance(part, SpeechPart)])
            for message in session.all_messages()
            if isinstance(message, ModelResponse)
        )
        truncations.extend(sim.truth.truncations)

    cuts: list[tuple[str, list[int | None]]] = []
    truncations: list[tuple[str, int]] = []
    run_clean(OpenAISimulation(), scenario)
    assert truncations == snapshot([('item_msg_2', 100)])
    assert cuts == snapshot([('interrupted', [100]), ('complete', [None])])


def test_baseline_openai_manual_turns() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.commit_audio()
        sim.transcribe()
        sim.create_response()
        sim.speak()
        sim.finish()
        sim.send_image(respond=True)
        sim.speak()
        sim.finish()

    run_clean(OpenAISimulation(openai=OpenAIOptions(turn_detection='manual', transcription=True)), scenario)


def test_baseline_gemini_conversation() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.speak(chunks=2)
        sim.call_tools()
        sim.finish_tool()
        sim.speak()
        sim.finish()
        sim.send_audio()
        sim.user_speaks(finished=True)
        sim.speak()
        sim.finish()
        sim.settle()
        sim.send_text(respond=False)
        sim.issue_handle()
        sim.drop()
        sim.settle()
        sim.send_text()
        sim.speak()
        sim.finish()

    run_clean(GeminiSimulation(), scenario)


def test_gemini_extended_thinking_filler_followed_by_more_speech() -> None:
    """The model stalls `IN_PROGRESS` and then speaks again instead of calling a tool: two responses, one exchange."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish(in_progress=True)
        sim.wait_for_reply()
        sim.speak()
        sim.finish()
        sim.settle()

    run_clean(GeminiSimulation(behavior=GeminiBehavior(stalls_in_progress=True)), scenario)


def test_gemini_user_cuts_in_on_a_held_filler() -> None:
    """The model stalls `IN_PROGRESS`, and the user speaks over the pause: the exchange is over."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish(in_progress=True)
        sim.wait_for_reply()
        sim.send_audio()
        sim.user_speaks()
        sim.settle()

    run_clean(GeminiSimulation(behavior=GeminiBehavior(stalls_in_progress=True)), scenario)


def test_scenario_gemini_context_sent_while_a_reply_is_owed() -> None:
    """SIM-39: context sent right after a typed turn is placed after the reply, which the server may not have begun."""

    with GeminiSimulation(strict=False) as sim:
        sim.send_text()
        sim.send_text(respond=False)
        sim.settle()
        assert ('SIM-39', 'history.order') in sim.checker.known_hits


def test_baseline_gemini_extended_thinking() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish(in_progress=True)
        sim.wait_for_reply()
        sim.call_tools()
        sim.finish_tool()
        sim.speak()
        sim.finish()

    run_clean(GeminiSimulation(behavior=GeminiBehavior(stalls_in_progress=True, handles_at_turn_start=True)), scenario)


def test_live_backend_response_read_while_closing_counts_its_request() -> None:
    """The session closes right after a delegation's backend reported its usage, before reading it: closing reads it."""

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate(deliver=False)
        sim.backend_finish(deliver=True, ticks=0)
        sim.close()
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_baseline_live_conversation() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.send_audio()
        sim.user_says()
        sim.speak()
        sim.delegate()
        sim.backend_call()
        sim.wait_for_reply()
        sim.finish_tool()
        sim.backend_finish()
        sim.backend_finish()
        sim.speak()
        sim.advance_time(1.0)
        sim.bill()
        sim.settle()
        sim.send_text()
        sim.speak()

    run_clean(LiveSimulation(), scenario)


# --- fault and behavior scenarios ---------------------------------------------------------------


def run_accepting(sim: Simulation, scenario: Callable[[Any], object], *accepted: str) -> None:
    """Run a scenario that may hit no violation but the accepted limitations named, then hang up and check it all again."""
    run_tolerant(sim, scenario)
    assert {finding_id for finding_id, _ in sim.checker.known_hits} <= set(accepted), sim.checker.known_hits


def run_tolerant(sim: Simulation, scenario: Callable[[Any], object]) -> None:
    """Run a scenario through faults and edge behaviors: known findings are tolerated, anything else fails."""
    sim.strict = False
    with sim as s:
        scenario(s)
        s.settle()
        s.close()
        s.settle()
        s.check_handoff()


def test_scenario_later_turn_takes_the_place_of_earlier_audio() -> None:
    """SIM-11 also covers the later of two turns, placed where the session saw audio the earlier turn took."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.create_response()
        sim.settle()
        sim.send_audio()
        sim.send_text()
        sim.settle()
        sim.commit_audio()
        sim.send_audio()
        sim.commit_audio()

    run_tolerant(
        OpenAISimulation(
            openai=OpenAIOptions(dialect='azure', transcription=True, vad_interrupts=False, vad_responds=False)
        ),
        scenario,
    )


def test_scenario_live_later_utterance_takes_the_place_of_earlier_audio() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.send_audio()
        sim.speak(deliver=False)
        sim.settle()
        sim.send_audio()
        sim.user_says(deliver=False)
        sim.settle()
        sim.user_says(deliver=False)

    run_tolerant(LiveSimulation(), scenario)


_XAI_PUSH_TO_TALK: dict[str, Callable[[OpenAISimulation], object]] = {
    'talk during a reply': lambda s: [s.send_text(), s.speak(), s.send_audio(), s.commit_audio(), s.create_response()],
    'commit silence': lambda s: [s.send_audio(voiced=False), s.commit_audio(), s.create_response()],
    'clear after commit': lambda s: [s.send_audio(), s.commit_audio(), s.clear_audio(), s.create_response()],
    'type during the answer to committed audio': lambda s: [
        s.send_text(),
        s.send_audio(),
        s.create_response(),
        s.finish(),
        s.send_text(),
    ],
    'ask again after the answer': lambda s: [
        s.send_audio(),
        s.commit_audio(),
        s.create_response(),
        s.speak(),
        s.finish(),
        s.settle(),
        s.create_response(),
    ],
}


@pytest.mark.parametrize('transcription', [True, False])
@pytest.mark.parametrize('name', list(_XAI_PUSH_TO_TALK))
def test_scenario_xai_push_to_talk(name: str, transcription: bool) -> None:
    """xAI with turn detection off, as recorded (#9070): it reports speech, answers a commit of speech at once, and
    drops a request with nothing new after answering committed audio."""
    run_tolerant(
        OpenAISimulation(openai=OpenAIOptions(dialect='xai', turn_detection='manual', transcription=transcription)),
        _XAI_PUSH_TO_TALK[name],
    )


@pytest.mark.parametrize('typed', [True, False])
def test_gemini_async_pair_cut_off_by_a_user_turn(typed: bool) -> None:
    """Two asynchronous calls whose turn a user turn cut off, answered together with it (SIM-19, fixed by the session core)."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(count=2)
        if typed:
            sim.send_text()
        else:
            sim.user_speaks(finished=True)
        sim.settle()

    run_clean(async_gemini(), scenario)


def test_baseline_server_vad_answers_each_turn_heard_during_a_reply() -> None:
    """With `interrupt_response` off, every turn VAD commits while a reply plays is answered after it, in order."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        for _ in range(2):
            sim.send_audio()
            sim.speech_start()
            sim.speech_stop()
        sim.finish()
        sim.settle()
        assert [response.answers for response in sim.truth.responses.values()] == [['t1'], ['u1'], ['u2']]

    run_clean(
        OpenAISimulation(openai=OpenAIOptions(dialect='azure', transcription=False, vad_interrupts=False)), scenario
    )


def test_xai_hand_commit_while_vad_hears_the_user_is_filed_in_place() -> None:
    """SIM-27 on xAI, fixed by the session core: the hand-committed turn goes where xAI committed it, after the reply
    that ended first."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.send_audio()
        sim.settle()
        sim.speech_start(deliver=False)
        sim.commit_audio()
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect='xai', transcription=True)), scenario)


def test_scenario_turn_without_its_transcript_kept_at_a_reconnect() -> None:
    """A turn whose transcript the drop cut off is recorded without one, and the reply after the reconnect too.

    xAI's own resumption used to lose it (SIM-25); the reconnect now settles it, as on OpenAI.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.send_audio()
        sim.commit_audio()
        sim.speak(deliver=False)
        sim.finish(deliver=False)
        sim.deliver()
        sim.drop()
        sim.settle()
        sim.create_response()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect='xai', transcription=True)), scenario)


def test_scenario_late_cancel_drops_a_tool_call_and_ends_the_wait() -> None:
    """SIM-10 also drops a tool call the finished response made, so the wait ends without the round's answer."""
    with OpenAISimulation(strict=False, openai=OpenAIOptions(turn_detection='manual', transcription=False)) as s:
        s.send_text()
        s.wait_for_reply()
        s.call_tool(deliver=False)
        s.speak(deliver=False)
        s.finish(deliver=False)
        s.interrupt(mode='cancel')
        s.settle()
        assert ('SIM-10', 'wait.early') in s.checker.known_hits


def test_scenario_wait_after_a_drop_cut_into_a_tool_round() -> None:
    """A wait begun after a drop cut into a tool round owes nothing for it: the reconnect settles the round."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.call_tool()
        sim.fail_next_send()
        sim.finish()
        sim.clear_audio()
        sim.wait_for_reply()
        sim.advance_time(0.1)

    run_tolerant(OpenAISimulation(openai=OpenAIOptions(turn_detection='manual', transcription=False)), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'azure'])
def test_scenario_wait_for_a_turn_whose_tool_round_a_drop_cut_into(dialect: Dialect) -> None:
    """The same for a wait on a turn's reply: the drop cut into the tool round it called, which the reconnect settles."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.fail_next_send()
        sim.finish()
        sim.clear_audio()
        sim.wait_for_reply()
        sim.advance_time(0.1)

    run_tolerant(OpenAISimulation(openai=OpenAIOptions(dialect=dialect)), scenario)


def test_scenario_openai_server_vad_edges() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.speak()
        sim.play()
        sim.interrupt(mode='played_ms')
        sim.finish()
        sim.send_text()
        sim.finish(status='incomplete')
        sim.send_text()
        sim.speak()
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop()
        sim.speak()
        sim.interrupt(mode='cancel')
        sim.finish(late=True)
        sim.settle()
        sim.send_audio()
        sim.speech_start()
        sim.create_response()
        sim.speech_stop()
        sim.transcribe(fail=True)
        sim.finish(status='failed')
        sim.speak()
        sim.call_tool()
        sim.interrupt(mode='cancel')
        sim.finish(late=True)
        sim.finish_tool(outcome='retry')
        sim.send_text()
        sim.release_late_done()
        sim.settle()
        sim.send_text()
        sim.speak()
        sim.settle()
        sim.send_text()
        sim.settle()
        sim.send_text()
        sim.speak()
        sim.send_audio()
        sim.speech_start(late=True)
        sim.speech_stop()
        sim.release_late_done()
        sim.settle()
        sim.send_audio()
        sim.speech_start()
        sim.tick(ticks=1)
        sim.tick()

    run_tolerant(
        OpenAISimulation(options=SessionOptions(latency=True), openai=OpenAIOptions(transcription=True)), scenario
    )


def test_scenario_openai_refusals() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.interrupt(mode='played_ms')
        sim.reject_next('content')
        sim.send_text()
        sim.wait_for_reply()
        sim.reject_next('response')
        sim.send_text()
        sim.commit_audio()
        sim.send_audio()
        sim.clear_audio()
        sim.send_audio()
        sim.commit_audio()
        sim.send_image(respond=True)
        sim.speak()
        sim.finish()
        sim.close()
        sim.close()
        sim.play()

    run_tolerant(OpenAISimulation(openai=OpenAIOptions(turn_detection='manual', transcription=False)), scenario)


def test_scenario_openai_connection_faults() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.call_tool()
        sim.finish()
        sim.settle()
        sim.fail_next_send(fault='ambiguous')
        sim.send_text()
        sim.settle()
        sim.send_text()
        sim.speak()
        sim.finish(deliver=False)
        sim.wait_for_reply()
        sim.drop(refuse_dials=1)
        sim.settle()
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop()

    run_tolerant(OpenAISimulation(openai=OpenAIOptions(transcription=False)), scenario)


def test_scenario_failed_audio_commit_after_an_unacknowledged_commit() -> None:
    """A failed second commit removes only its own bookkeeping before the socket reconnects."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.commit_audio()
        sim.send_audio()
        sim.fail_next_send()
        sim.commit_audio()
        commits = [operation for operation in sim.operations if operation.name == 'commit_audio']
        assert len(commits) == 2
        assert commits[0].done and commits[0].error is None
        assert commits[1].done and commits[1].error is not None
        assert sim.server.network.failed_sends[-1][0] == 'input_audio_buffer.commit'

    run_tolerant(OpenAISimulation(openai=OpenAIOptions(turn_detection='manual', transcription=False)), scenario)


def test_scenario_xai_reconnect_replays_history() -> None:
    """xAI reconnects like OpenAI, by replaying the history: its own resumption drops user text (checked live)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.settle()
        sim.drop()
        sim.settle()
        sim.send_text()

    run_tolerant(OpenAISimulation(openai=OpenAIOptions(dialect='xai')), scenario)


@pytest.mark.parametrize('clear', [False, True])
def test_scenario_xai_reconnect_after_a_turn_is_discarded(clear: bool) -> None:
    """xAI adds a spoken turn's item at speech start: once the turn is discarded, by a drop mid-speech or a clear
    before one, the reconnect doesn't discard it a second time for the transcript it was awaiting."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        if clear:
            sim.clear_audio()
        sim.drop()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect='xai')), scenario)


def test_push_to_talk_turn_filed_before_its_answer() -> None:
    """The transcript of a committed turn arrives after its answer is recorded (OR9, fixed by #8764).

    The turn is still filed by inserting it into recorded history, which is known finding E.
    """

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.commit_audio()
        sim.create_response()
        sim.speak()
        sim.finish()
        sim.transcribe()
        sim.settle()

    run_tolerant(OpenAISimulation(openai=OpenAIOptions(turn_detection='manual')), scenario)


def test_scenario_gemini_barge_in_and_faults() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools()
        sim.send_audio()
        sim.user_speaks()
        sim.speak(deliver=False)
        sim.deliver(count=1)
        sim.send_image()
        sim.settle()
        sim.fail_next_send(fault='ambiguous')
        sim.send_text()
        sim.settle()
        sim.fail_next_send()
        sim.send_text()
        sim.settle()
        sim.send_text()
        sim.speak()
        sim.drop(refuse_dials=1)
        sim.settle()
        sim.send_text()
        sim.send_text()
        sim.speak()
        sim.send_audio()
        sim.user_speaks()
        sim.speak(deliver=False)

    run_tolerant(
        GeminiSimulation(
            options=SessionOptions(latency=True),
            behavior=GeminiBehavior(handles_at_turn_start=True, input_transcription=False),
        ),
        scenario,
    )


def test_scenario_gemini_stall_answered_by_speech() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish(in_progress=True)
        sim.speak()
        sim.finish()
        sim.send_text()
        sim.speak()
        sim.finish(in_progress=True)

    run_tolerant(GeminiSimulation(behavior=GeminiBehavior(stalls_in_progress=True)), scenario)


def test_scenario_live_edges() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.send_text(respond=False)
        sim.send_text()
        sim.speak()
        sim.speak(deliver=False)
        sim.deliver(count=1)
        sim.advance_time(1.0)
        sim.speak()
        sim.delegate()
        sim.backend_call()
        sim.finish_tool()
        sim.backend_finish(status='failed')
        sim.settle()
        sim.fail_next_send(fault='ambiguous')
        sim.send_text()
        sim.settle()

    run_tolerant(LiveSimulation(options=SessionOptions(latency=True)), scenario)


def test_scenario_live_drop_mid_reply() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.drop()

    run_tolerant(LiveSimulation(), scenario)


def test_baseline_gemini_async_user_turn_during_the_held_round() -> None:
    """The user asks something else while an async tool runs, and the model answers it before the result.

    #8760's N1a: the answer to the user must not join the calling response, and the user's turn goes after the
    tool's return, not before the call.
    """

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.call_tools()
        sim.finish()
        sim.send_audio()
        sim.user_speaks(finished=True)
        sim.speak()
        sim.finish()

    run_clean(async_gemini(), scenario)


def test_baseline_gemini_async_user_turn_outlasts_the_hold() -> None:
    """#8760's N1b: the tool outlasts the 5 s the session holds a user turn for; the turn still goes after the call."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.call_tools()
        sim.finish()
        sim.send_audio()
        sim.user_speaks(finished=True)
        sim.advance_time(6)

    run_clean(async_gemini(), scenario)


def test_baseline_gemini_async_typed_turn_during_the_held_round() -> None:
    """#8760's N2: a typed turn while an async tool runs is recorded at once, and its wait returns with its reply."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.call_tools()
        sim.finish()
        sim.send_text()
        sim.wait_for_reply()
        sim.speak()
        sim.finish()

    run_clean(async_gemini(), scenario)


def test_scenario_gemini_async_parallel_calls() -> None:
    """Two async calls in one message, with speech before, between, and after their results.

    Speech between the two results goes after the second (each result must directly follow its call), but what was
    said before the first result belongs to the calling response.
    """

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(count=2)
        sim.speak()
        sim.finish_tool()
        sim.speak()

    run_accepting(async_gemini(), scenario, '8760-accepted-between-results')


def test_scenario_gemini_async_call_in_flight_at_the_result_is_accepted() -> None:
    """A second call the model made before the first's result went out, but which the session hadn't read yet, is
    recorded in a response of its own: the same accepted limitation as speech in flight."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_audio()
        sim.user_speaks(deliver=False)
        sim.call_tools()
        sim.call_tools(deliver=False)

    run_accepting(async_gemini(), scenario, '8760-accepted')


def test_scenario_openai_tool_retries_exhausted() -> None:
    """A tool that asks for a retry twice in a row ends the session, as it ends a standard run."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.finish()
        sim.finish_tool(outcome='retry')
        sim.call_tool()
        sim.finish()
        sim.finish_tool(outcome='retry')

    run_tolerant(OpenAISimulation(), scenario)


def test_scenario_gemini_async_speech_in_flight_at_the_result_is_accepted() -> None:
    """Speech still on its way to the session when the result goes out is split there: an accepted limitation."""
    sim = async_gemini()
    sim.strict = False
    with sim as s:
        s.send_text()
        s.call_tools()
        s.speak()
        s.speak(deliver=False)
        s.finish_tool()
        s.settle()
        assert '8760-accepted' in {finding_id for finding_id, _ in s.checker.known_hits}


def test_tool_result_awaiting_its_reply_when_barged_in_on_without_a_vad_reply() -> None:
    """SIM-32, fixed by the session core: the reply a cancelled response's tool results wait for is still owed."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_image(respond=True)
        sim.call_tool()
        sim.tick(ticks=0)
        sim.tick(ticks=0)
        sim.speech_start(deliver=False)
        sim.settle()

    run_clean(
        OpenAISimulation(
            openai=OpenAIOptions(dialect='azure', transcription=False, vad_interrupts=True, vad_responds=False)
        ),
        scenario,
    )


def test_scenario_gemini_tool_result_sent_on_a_dropped_connection() -> None:
    """G3b: the tool result the session sends hits the dropped socket, and that failure is raised from `async for`.

    The session goes on receiving over the re-dial, so a turn sent afterwards is answered.
    """

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.speak(deliver=False)
        sim.call_tools(deliver=False)
        sim.deliver()
        sim.drop()
        sim.settle()
        sim.send_text()
        sim.settle()

    sim = GeminiSimulation()
    run_clean(sim, scenario)
    assert [repr(error) for error in sim.consumer_errors] == snapshot(
        [
            "RealtimeError('Realtime connection failed while sending: sent 1006 (abnormal closure [internal]) simulated drop; no close frame received')"
        ]
    )


@known('8763c #3')
def test_known_gemini_text_sent_while_an_unread_reply_speaks() -> None:
    """Text sent while a reply the client hasn't read yet is under way: that reply is taken for the text's own."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.speak(deliver=False)
        sim.send_text(respond=True)
        sim.settle()

    reproduce('8763c #3', GeminiSimulation(), scenario)


def test_live_text_sent_while_an_unread_reply_speaks_is_answered_by_it() -> None:
    """Text sent while a reply the client hasn't read yet is under way is taken for that reply's, and it is (8763c #3).

    Live opens a delegation for the text in the reply under way, and goes on to speak its outcome there.
    """

    def scenario(sim: LiveSimulation) -> None:
        sim.speak(deliver=False)
        sim.send_text(respond=True)
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_live_delegation_during_a_tool_round_joins_the_reply() -> None:
    """A delegation the model starts while a delegated tool round is owed its answer is the reply's (8763c #3, once).

    The reply that called the tools goes on until the round's answer, so the wait for that answer isn't ended by it.
    """

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate(deliver=False)
        sim.backend_call(deliver=False)
        sim.wait_for_reply()
        sim.advance_time(0.1)
        sim.wait_for_reply()
        sim.backend_finish(deliver=False)
        sim.delegate(deliver=False)
        sim.settle()

    run_clean(LiveSimulation(), scenario)


def test_scenario_a_provider_reply_before_our_item_is_added_is_not_ours() -> None:
    """SIM-24, for a request with an item of its own: a server VAD reply that starts before the server adds our
    item can't be the one our `response.create` asks for, metadata echoed or not."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.speech_stop(deliver=False)
        sim.finish(deliver=False)
        sim.send_text(respond=True)
        sim.settle()

    run_clean(OpenAISimulation(openai=OpenAIOptions(transcription=False, dialect='xai')), scenario)


def test_scenario_a_call_the_session_refuses_over_a_limit_is_left_out() -> None:
    """The session refuses a call it counts over `request_limit` (here, after a repeated terminal): no core keeps it."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.send_image(respond=True)
        sim.settle()
        sim.create_response()
        sim.call_tool(deliver=False)
        sim.repeat_done(deliver=False)
        sim.call_tool(deliver=False)

    run_tolerant(
        OpenAISimulation(options=SessionOptions(request_limit=3), openai=OpenAIOptions(transcription=False)), scenario
    )


def test_scenario_results_of_an_abandoned_tool_batch_owe_no_reply() -> None:
    """A tool round over `request_limit` abandons the batch: the results still sent after that ask for nothing."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.settle()
        sim.create_response()
        sim.call_tool(deliver=False)
        sim.call_tool(deliver=False)

    run_tolerant(
        OpenAISimulation(options=SessionOptions(request_limit=2), openai=OpenAIOptions(transcription=False)), scenario
    )


def test_scenario_a_request_orphaned_by_a_stray_terminal_is_lost_with_the_connection() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_image(respond=True)
        sim.reject_next(kind='content')
        sim.play(chunks=1)
        sim.send_image(respond=False)
        sim.send_image(respond=False)
        sim.send_image(respond=False)
        sim.finish(deliver=False, ticks=0)
        sim.reject_next(kind='content')
        sim.send_image(respond=True)
        sim.reject_next(kind='response', ticks=0)
        sim.repeat_done(deliver=True)
        sim.drop(refuse_dials=0)

    run_tolerant(OpenAISimulation(openai=OpenAIOptions(transcription=False)), scenario)


def test_scenario_a_spoken_turn_whose_transcript_is_never_read() -> None:
    """xAI adds the turn at speech start; the session stops reading before its transcript comes."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio(chunks=1)
        sim.speech_start(deliver=False)
        sim.create_response()

    run_tolerant(
        OpenAISimulation(
            options=SessionOptions(request_limit=1), openai=OpenAIOptions(transcription=True, dialect='xai')
        ),
        scenario,
    )


def test_scenario_live_backend_report_read_while_closing_an_ended_session() -> None:
    """A session that ended on its request limit accounts nothing more: not the backend report its close reads."""

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.settle()
        sim.delegate()
        sim.settle()
        sim.delegate()
        sim.backend_finish()

    run_tolerant(LiveSimulation(options=SessionOptions(request_limit=1)), scenario)


def test_every_finding_is_pinned() -> None:
    """Every known bug has a scenario that fails with it until the fix lands; accepted limitations have none."""
    assert {finding.id for finding in KNOWN_FINDINGS if not finding.accepted} == PINNED


def test_gemini_tool_call_abandoned_by_a_drop_ends_the_wait() -> None:
    """A call the resumed session doesn't know is abandoned (#8763), so nothing more is owed to the turn that made it."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools()
        sim.drop()
        sim.wait_for_reply()

    run_clean(GeminiSimulation(), scenario)


@pytest.mark.parametrize('dialect', ['openai', 'azure'])
def test_baseline_openai_tool_result_with_media(dialect: str) -> None:
    """A tool result with an image: the image follows the output as a user message (xAI takes no images)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.finish()
        sim.finish_tool(outcome='media')
        sim.speak()
        sim.finish()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect)), scenario)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('dialect', ['openai', 'azure'])
def test_baseline_reconnect_replays_the_conversation(dialect: Dialect) -> None:
    """A re-dial starts an empty conversation: the local replay is what gives it the tool round back."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.call_tool()
        sim.finish()
        sim.finish_tool()
        sim.speak()
        sim.finish()
        sim.settle()
        sim.drop()
        sim.settle()
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.settle()
        first, second = sim.server.sessions
        assert first.conversation <= second.conversation
        assert 'function_call:call_1' in second.conversation

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect=dialect)), scenario)


def test_baseline_xai_reconnect_replays_a_tool_round() -> None:
    """The history replayed into xAI's new conversation includes the tool call and its output, which the session
    must not run again."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.call_tool()
        sim.finish()
        sim.finish_tool()
        sim.speak()
        sim.finish()
        sim.settle()
        sim.drop()
        sim.settle()
        replayed = [
            frame['item']['type']
            for frame in sim.server.network.sockets[-1].received
            if frame['type'] == 'conversation.item.added'
        ]
        # The user's message comes first: xAI's own resumption left it out (checked live).
        assert replayed == ['message', 'message', 'function_call', 'function_call_output', 'message']
        sim.send_text()
        sim.speak()
        sim.finish()

    run_clean(OpenAISimulation(openai=OpenAIOptions(dialect='xai')), scenario)


def test_baseline_server_vad_without_interrupting_or_responding() -> None:
    """Server VAD with `interrupt_response` and `create_response` off: the user's speech neither cuts the reply
    off nor gets one of its own, until the app asks for it."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop()
        sim.speak()
        sim.finish()
        sim.settle()
        assert len(sim.truth.responses) == 1
        sim.create_response()
        assert sim.truth.responses_by_number[2].answers == ['u1']
        sim.speak()
        sim.finish()

    run_clean(
        OpenAISimulation(openai=OpenAIOptions(vad_interrupts=False, vad_responds=False, transcription=False)), scenario
    )


def test_scenario_terminal_cut_off_by_hanging_up() -> None:
    """A `response.done` whose deferred request a close cuts off loses its usage: SIM-23."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_image(respond=True)
        sim.clear_audio()
        sim.create_response()
        sim.call_tool(deliver=False)
        sim.deliver()
        # The tool's result would pass `request_limit`: the session raises that and goes on receiving.
        sim.finish_tool()
        sim.deliver()
        sim.clear_audio()
        sim.finish(ticks=0)
        sim.close()

    run_tolerant(
        OpenAISimulation(
            seed=20,
            options=SessionOptions(request_limit=2, latency=True),
            openai=OpenAIOptions(turn_detection='manual', transcription=False),
        ),
        scenario,
    )


def test_xai_server_vad_answers_every_turn() -> None:
    """xAI echoes `create_response: False` back but answers the turn anyway (`XaiRealtimeModelSettings`)."""
    from ._openai_server import ServerSession

    socket = cast(Any, None)
    vad: dict[str, Any] = {'type': 'server_vad', 'create_response': False}
    assert not ServerSession(index=0, socket=socket, turn_detection=vad).create_response
    assert ServerSession(index=0, socket=socket, turn_detection=vad, answers_every_turn=True).create_response


# --- ElevenLabs Agents ------------------------------------------------------------------------------


def test_baseline_elevenlabs_agents_conversation() -> None:
    def scenario(sim: ElevenLabsAgentsSimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.wait_for_reply()
        sim.finish_tool()
        sim.speak(chunks=2)
        sim.finish()
        sim.report_usage()
        sim.ping()
        sim.send_audio()
        sim.user_says()
        sim.speak()
        sim.finish()
        sim.report_usage()
        sim.send_text(respond=False)
        sim.settle()

    run_clean(ElevenLabsAgentsSimulation(), scenario)


def test_elevenlabs_agents_barge_in_keeps_the_truncated_reply_with_the_reply_it_cut() -> None:
    """Recorded: the reply still being generated is cut to what was heard, and that `agent_response` ends it."""

    def scenario(sim: ElevenLabsAgentsSimulation) -> None:
        sim.send_text()
        sim.speak(chunks=2)
        sim.send_audio()
        sim.barge_in()
        sim.user_says()
        sim.speak()
        sim.finish()
        sim.settle()

    run_clean(ElevenLabsAgentsSimulation(), scenario)


@pytest.mark.parametrize('voiced', [False, True], ids=['being-voiced', 'voiced-in-full'])
def test_elevenlabs_agents_barge_in_on_a_finished_reply(voiced: bool) -> None:
    """Cut in on while its audio still comes, the reply ends with what was heard; voiced in full, nothing changes."""

    def scenario(sim: ElevenLabsAgentsSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish()
        if voiced:
            sim.report_usage()
        sim.barge_in()
        sim.user_says()
        sim.settle()

    run_clean(ElevenLabsAgentsSimulation(), scenario)


@pytest.mark.parametrize('calls_ended', [False, True], ids=['reply-under-way', 'reply-ended'])
def test_elevenlabs_agents_raising_tool_completes_the_turn_once(calls_ended: bool) -> None:
    """A client tool that raises ends the turn once, whether or not the reply that called it had ended by then.

    A reply still under way ends the turn as it ends; once the calls have ended it (after a quiet stretch), nothing
    else will, so the turn completes as the tool raises.
    """

    def scenario(sim: ElevenLabsAgentsSimulation) -> None:
        sim.send_text()
        sim.call_tool()
        if calls_ended:
            sim.advance_time(2.0)
        sim.finish_tool(outcome='error')
        sim.settle()
        assert len(sim.turn_completes) == 1

    run_clean(ElevenLabsAgentsSimulation(), scenario)


def test_elevenlabs_agents_tool_calls_end_after_a_quiet_stretch() -> None:
    """With nothing after them, a reply's calls are over once no more come for a moment."""

    def scenario(sim: ElevenLabsAgentsSimulation) -> None:
        sim.send_text()
        sim.call_tool(count=2)
        sim.advance_time(0.1)
        sim.finish_tool(call=0)
        sim.finish_tool(call=0)
        sim.speak()
        sim.finish()
        sim.report_usage()

    run_clean(ElevenLabsAgentsSimulation(), scenario)


def test_elevenlabs_agents_hold_audio_while_queued_is_no_reply() -> None:
    def scenario(sim: ElevenLabsAgentsSimulation) -> None:
        sim.deliver()
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.settle()

    run_clean(ElevenLabsAgentsSimulation(elevenlabs=ElevenLabsAgentsOptions(queued=True)), scenario)


def test_elevenlabs_agents_drop_ends_the_session() -> None:
    def scenario(sim: ElevenLabsAgentsSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.drop()
        sim.settle()

    run_clean(ElevenLabsAgentsSimulation(), scenario)


def test_scenario_elevenlabs_agents_edges() -> None:
    def scenario(sim: ElevenLabsAgentsSimulation) -> None:
        sim.send_text()
        sim.speak(deliver=False)
        sim.send_text(ticks=1)
        sim.deliver()
        sim.barge_in(deliver=False)
        sim.user_says(deliver=False)
        sim.deliver(count=1)
        sim.speak()
        sim.finish()
        sim.call_tool(deliver=False)
        sim.deliver()
        sim.send_text()
        sim.fail_next_send()
        sim.finish_tool()
        sim.settle()

    run_tolerant(ElevenLabsAgentsSimulation(options=SessionOptions(latency=True)), scenario)


def test_elevenlabs_agents_reply_voiced_after_its_text() -> None:
    """Recorded: the whole text arrives once the LLM is done, while TTS voices the rest; the reply ends with its audio."""

    def scenario(sim: ElevenLabsAgentsSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.voice(chunks=2)
        sim.report_usage()
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.voice()
        sim.barge_in()
        sim.user_says()
        sim.speak()
        sim.finish()
        sim.advance_time(2.0)

    run_clean(ElevenLabsAgentsSimulation(), scenario)


def test_elevenlabs_agents_parallel_tool_calls_are_one_response() -> None:
    """A reply's parallel tool calls are one response, and the agent answers once every result is in."""

    def scenario(sim: ElevenLabsAgentsSimulation) -> None:
        sim.send_text()
        sim.call_tool(count=2)
        sim.finish_tool(call=0)
        sim.finish_tool(call=0)
        sim.speak()
        sim.finish()
        sim.report_usage()

    run_clean(ElevenLabsAgentsSimulation(), scenario)


def test_scenario_vertex_cut_off_reply_is_taken_for_the_text_that_cut_it() -> None:
    """8763c #3 on Vertex's `gemini-live-2.5-flash`: it cuts a reply off for text, but reports no speech, so the cut-off
    reply can't be told from one the user spoke over: it keeps answering the text, and the wait ends with it."""
    with GeminiSimulation(strict=False, behavior=GeminiBehavior(closes_tool_turn_separately=True)) as sim:
        sim.send_text()
        sim.speak(deliver=False)
        sim.send_text()
        sim.wait_for_reply()
        sim.speak()
        sim.finish()
        sim.settle()
        assert '8763c #3' in {finding_id for finding_id, _ in sim.checker.known_hits}
