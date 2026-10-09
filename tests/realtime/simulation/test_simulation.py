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

from ...conftest import try_import

with try_import() as imports_successful:
    from hypothesis import HealthCheck, settings
    from hypothesis.stateful import (
        RuleBasedStateMachine,
        run_state_machine_as_test,  # pyright: ignore[reportUnknownVariableType]
    )

    from ._findings import FINDINGS_BY_ID, KNOWN_FINDINGS
    from ._gemini import GeminiBehavior, GeminiMachine, GeminiSimulation
    from ._live import LiveMachine, LiveSimulation
    from ._machine import KNOWN_HIT_COUNTS
    from ._openai_server import Dialect
    from ._openai_simulation import AzureMachine, OpenAIMachine, OpenAIOptions, OpenAISimulation, XaiMachine
    from ._simulation import FindingReproduced, SessionOptions, Simulation

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
        pytest.param(GeminiMachine, id='gemini'),
        pytest.param(LiveMachine, id='gpt-live'),
    ]
    if imports_successful()
    else [],
)
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


def test_raising_tool_ends_the_exchange() -> None:
    """A tool that raises ends the session, and nothing is left waiting (OR8, #8765)."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.finish()
        sim.finish_tool(outcome='error')

    run_clean(OpenAISimulation(), scenario)


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


@known('SIM-11')
def test_known_turn_spoken_before_a_reply_filed_before_it() -> None:
    """The user started talking, the model answered a typed turn, and only then was the spoken turn committed."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.send_text()
        sim.settle()
        sim.commit_audio()
        sim.settle()

    reproduce('SIM-11', OpenAISimulation(openai=OpenAIOptions(turn_detection='manual')), scenario)


@known('SIM-11')
def test_known_second_commit_takes_the_turn_of_earlier_audio() -> None:
    """Audio streamed before each of two replies, committed as one turn after them, then a second turn committed:
    the session gives the second turn the place of the audio before the second reply."""

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

    reproduce('SIM-11', OpenAISimulation(openai=OpenAIOptions(turn_detection='manual')), scenario)


@known('SIM-11')
def test_known_live_utterance_spoken_before_a_reply_filed_before_it() -> None:
    """On GPT-Live, audio streamed before the model's reply is transcribed as a turn only after that reply."""

    def scenario(sim: LiveSimulation) -> None:
        sim.send_audio()
        sim.speak()
        sim.settle()
        sim.user_says()
        sim.settle()

    reproduce('SIM-11', LiveSimulation(), scenario)


@known('SIM-12')
def test_known_refused_tool_results_request_leaves_wait_hanging() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.reject_next('response')
        sim.finish()
        sim.settle()

    reproduce('SIM-12', OpenAISimulation(), scenario)


@known('OR9')
def test_known_turn_committed_by_hand_under_server_vad_filed_late() -> None:
    """Server VAD hears the user start; the app commits the buffer by hand before VAD commits the rest."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.commit_audio()
        sim.settle()

    reproduce('OR9', OpenAISimulation(), scenario)


@known('SIM-13')
def test_known_gemini_cut_off_tool_turn_ends_the_wait_early() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(deliver=False)
        sim.send_text()
        sim.wait_for_reply()
        sim.settle()

    reproduce('SIM-13', GeminiSimulation(), scenario)


def async_gemini() -> GeminiSimulation:
    """Gemini 2.5 native audio with asynchronous tool calls: the model keeps talking after a call."""
    return GeminiSimulation(behavior=GeminiBehavior(async_tool_calls=True))


@known('8760')
def test_known_gemini_async_filler_recorded_after_the_result() -> None:
    """S0: the model narrates after the call and finishes; the result then cuts in, and the model answers.

    Expected: `[call, filler] {return} [answer]`. Recorded: `[call] {return} [filler] [answer]`.
    """

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools()
        sim.speak()
        sim.finish()
        sim.settle()

    reproduce('8760', async_gemini(), scenario)


@known('8760')
def test_known_gemini_async_result_cuts_into_speech() -> None:
    """S2: the result arrives while the model is still narrating; what it said before belongs to the call."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools()
        sim.speak()
        sim.finish_tool()
        sim.settle()

    reproduce('8760', async_gemini(), scenario)


@known('SIM-11')
def test_known_turn_heard_before_a_cancelled_reply_ended_filed_before_it() -> None:
    """Server VAD hears the user start while a reply the app cancelled (after it called a tool) is still ending."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.interrupt(mode='cancel')
        sim.call_tool()
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.settle()

    reproduce('SIM-11', OpenAISimulation(), scenario)


@known('SIM-1')
def test_known_gemini_tool_batch_answer_lost_to_a_drop() -> None:
    """Every result of the batch went out, then the connection dropped before the answer: it stays owed.

    On OpenAI the reconnect asks for that answer again; Gemini doesn't resume a generation after a re-dial.
    """

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(count=2)
        sim.finish_tool()
        sim.finish_tool()
        sim.drop()
        sim.settle()

    reproduce('SIM-1', GeminiSimulation(), scenario)


@known('SIM-14')
def test_known_refused_context_misfiles_the_spoken_turn() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.reject_next('content')
        sim.speech_start(deliver=False)
        sim.send_image()
        sim.settle()

    reproduce('SIM-14', OpenAISimulation(), scenario)


@known('SIM-15')
def test_known_raising_tool_leaves_a_deferred_request_owed() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.create_response()
        sim.call_tool()
        sim.finish_tool(outcome='error')
        sim.settle()

    reproduce('SIM-15', OpenAISimulation(), scenario)


@known('SIM-15')
def test_known_tool_result_over_request_limit_leaves_wait_hanging() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_image(respond=True)
        sim.call_tool(deliver=False)
        sim.send_image(respond=True)
        sim.settle()

    reproduce('SIM-15', OpenAISimulation(options=SessionOptions(request_limit=2)), scenario)


@known('SIM-16')
def test_known_cleared_barge_in_keeps_the_dropped_request() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.create_response()
        sim.create_response()
        sim.speech_start(deliver=False)
        sim.clear_audio()
        sim.settle()

    reproduce('SIM-16', OpenAISimulation(openai=OpenAIOptions(transcription=False)), scenario)


@known('SIM-17')
def test_known_extended_thinking_parallel_calls_leave_a_reservation() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_audio()
        sim.user_speaks()
        sim.call_tools(count=2)
        sim.settle()

    reproduce('SIM-17', GeminiSimulation(behavior=GeminiBehavior(stalls_in_progress=True)), scenario)


@known('8760')
def test_known_gemini_async_second_call_in_the_same_turn() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(deliver=False)
        sim.call_tools()

    reproduce('8760', async_gemini(), scenario)


@known('SIM-18')
def test_known_live_reply_split_by_a_delegated_round() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.speak(deliver=False)
        sim.delegate(deliver=False)
        sim.backend_call(deliver=False)
        sim.backend_finish(deliver=False)
        sim.advance_time(0.1)
        sim.finish_tool()
        sim.backend_call(deliver=False)
        sim.settle()

    reproduce('SIM-18', LiveSimulation(), scenario)


@known('SIM-13')
def test_known_gemini_cut_off_unstarted_turn_ends_the_wait_early() -> None:
    """Two spoken turns the model hadn't started answering, then a typed one that cuts them off."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_audio()
        sim.user_speaks(deliver=False)
        sim.send_audio()
        sim.user_speaks(deliver=False)
        sim.send_text()
        sim.wait_for_reply()
        sim.settle()

    reproduce('SIM-13', GeminiSimulation(), scenario)


@known('SIM-19')
def test_known_gemini_async_batch_of_three_leaves_a_reservation() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(count=3)
        sim.settle()

    reproduce('SIM-19', async_gemini(), scenario)


@known('G3b')
def test_known_typed_turn_during_reconnect_fails() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.drop(ticks=0)
        sim.send_text()
        sim.settle()

    reproduce('G3b', OpenAISimulation(), scenario)


@known('SIM-20')
def test_known_barge_in_on_a_tool_round_keeps_the_deferred_request() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.create_response()
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.settle()

    reproduce('SIM-20', OpenAISimulation(), scenario)


@known('8801')
def test_known_late_terminal_of_a_barged_in_reply_lands_on_the_next() -> None:
    """#8801's case: the `response.done` of a reply the user cut off arrives after the next reply started."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.send_audio()
        sim.speech_start(late=True)
        sim.speech_stop()
        sim.speak()

    reproduce('8801', OpenAISimulation(), scenario)


@known('SIM-21')
def test_known_request_whose_refusal_is_lost_keeps_its_reservation() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.reject_next('response')
        sim.create_response()
        sim.send_text()
        sim.drop()
        sim.settle()

    reproduce('SIM-21', OpenAISimulation(), scenario)


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


@known('SIM-32')
def test_known_barge_in_drops_a_request_vad_will_not_answer() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.create_response()
        sim.create_response()
        sim.send_audio()
        sim.speech_start(deliver=False)
        sim.settle()

    reproduce('SIM-32', OpenAISimulation(openai=OpenAIOptions(vad_responds=False, transcription=False)), scenario)


@known('SIM-11')
def test_known_turn_spoken_while_a_cancelled_reply_ends_filed_before_it() -> None:
    """The user starts talking after the app cancelled a reply, before the provider has ended it."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_image(respond=True)
        sim.interrupt(mode='cancel')
        sim.speak()
        sim.send_audio()
        sim.settle()
        sim.commit_audio()
        sim.settle()

    reproduce('SIM-11', OpenAISimulation(openai=OpenAIOptions(turn_detection='manual', dialect='azure')), scenario)


@known('SIM-33')
def test_known_gemini_async_result_cut_in_ends_the_wait_early() -> None:
    """The model finishes speaking after an async call; the result then cuts in, before the model answers it."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools()
        sim.wait_for_reply()
        sim.finish(deliver=False)
        sim.settle()

    reproduce('SIM-33', async_gemini(), scenario)


@known('SIM-25')
def test_known_turn_committed_before_closing_is_lost() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.commit_audio()
        sim.close()
        sim.settle()

    reproduce('SIM-25', OpenAISimulation(openai=OpenAIOptions(turn_detection='manual', transcription=True)), scenario)


@known('SIM-26')
def test_known_turn_committed_after_a_racing_clear_is_lost() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio(chunks=3, ticks=0)
        sim.clear_audio()
        sim.commit_audio()
        sim.settle()

    reproduce(
        'SIM-26',
        OpenAISimulation(
            options=SessionOptions(latency=True), openai=OpenAIOptions(turn_detection='manual', transcription=False)
        ),
        scenario,
    )


@known('SIM-27')
def test_known_hand_commit_under_server_vad_loses_a_turn() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.commit_audio()
        sim.settle()

    reproduce(
        'SIM-27',
        OpenAISimulation(
            openai=OpenAIOptions(dialect='azure', transcription=False, vad_interrupts=False, vad_responds=False)
        ),
        scenario,
    )


@known('SIM-28')
def test_known_input_held_behind_a_lost_reply_is_not_replayed() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.send_text(respond=False)
        sim.drop()
        sim.settle()
        sim.send_text()
        sim.settle()

    reproduce('SIM-28', OpenAISimulation(), scenario)


@known('SIM-34')
def test_known_core_loses_a_hand_commit_unanswered_at_a_drop() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.commit_audio()
        sim.drop()
        sim.settle()

    reproduce('SIM-34', OpenAISimulation(openai=OpenAIOptions(turn_detection='manual', transcription=False)), scenario)


@known('SIM-35')
def test_known_clear_after_an_unread_vad_commit_drops_the_turn() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop(deliver=False)
        sim.clear_audio()
        sim.settle()

    reproduce('SIM-35', OpenAISimulation(openai=OpenAIOptions(transcription=False)), scenario)


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


@known('SIM-29')
def test_known_xai_push_to_talk_turn_spoken_over_is_lost() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.create_response()
        sim.send_audio()
        sim.settle()

    reproduce(
        'SIM-29',
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


@known('SIM-31')
def test_known_xai_push_to_talk_turn_spoken_during_a_reply_filed_before_it() -> None:
    """xAI push-to-talk: the user talks during the reply to a held commit; the request after it commits that turn."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.commit_audio()
        sim.send_audio()
        sim.create_response()
        sim.settle()
        sim.create_response()
        sim.settle()

    reproduce(
        'SIM-31',
        OpenAISimulation(openai=OpenAIOptions(dialect='xai', turn_detection='manual', transcription=True)),
        scenario,
    )


@known('E')
def test_known_late_transcript_inserted_into_recorded_history() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop()
        sim.speak()
        sim.finish()
        sim.transcribe()

    reproduce('E', OpenAISimulation(), scenario)


@known('8801')
def test_known_repeated_terminal_recorded_as_a_new_response() -> None:
    """A robustness fault, not recorded provider behavior: the server sends a `response.done` twice."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.repeat_done()

    reproduce('8801', OpenAISimulation(), scenario)


@known('SIM-1')
def test_known_reply_lost_to_a_drop_keeps_its_reservation_openai() -> None:
    """The response had started (`response.created` read) when the socket dropped, so it is not asked for again."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.deliver()
        sim.drop()
        sim.settle()

    reproduce('SIM-1', OpenAISimulation(), scenario)


def test_gemini_typed_turn_lost_to_a_drop_is_settled() -> None:
    """A typed turn the model hadn't answered when the connection dropped is settled (SIM-1, fixed for it by #8763)."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.drop()

    run_clean(GeminiSimulation(), scenario)


@known('SIM-2a')
def test_known_turn_sent_before_reply_content_recorded_ahead_of_it() -> None:
    """The first reply ended empty before the second turn was sent, but the client hadn't read it yet."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.finish(deliver=False)
        sim.send_text(respond=False)
        sim.settle()

    reproduce('SIM-2a', OpenAISimulation(), scenario)


@known('SIM-2b')
def test_known_wait_returns_before_a_started_vad_reply() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop()
        sim.wait_for_reply()

    reproduce('SIM-2b', OpenAISimulation(), scenario)


@known('SIM-2b')
def test_known_wait_returns_before_a_delegated_reply() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.wait_for_reply()

    reproduce('SIM-2b', LiveSimulation(), scenario)


@known('SIM-3')
def test_known_reconnect_does_not_ask_again_for_an_unstarted_reply() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.send_text()
        sim.drop()
        sim.settle()

    reproduce('SIM-3', OpenAISimulation(), scenario)


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


@known('SIM-6')
def test_known_live_parallel_calls_leak_reservations() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.backend_call(count=2)
        sim.settle()

    reproduce('SIM-6', LiveSimulation(), scenario)


@known('SIM-6')
def test_known_live_results_of_delegations_in_a_row_answered_together() -> None:
    """Two delegations each call a tool; the model answers both results with one reply, which leaves one owed."""

    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.backend_call()
        sim.backend_finish()
        sim.delegate()
        sim.backend_call()
        sim.settle()

    reproduce('SIM-6', LiveSimulation(), scenario)


@known('SIM-7')
def test_known_live_queued_text_answered_together() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.send_text()
        sim.send_text()
        sim.settle()

    reproduce('SIM-7', LiveSimulation(), scenario)


@known('SIM-8')
def test_known_live_drop_raises_a_raw_websocket_error() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.drop()

    reproduce('SIM-8', LiveSimulation(), scenario)


@known('SIM-9')
def test_known_live_abandoned_calls_keep_reservations() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.backend_call()
        sim.backend_finish(status='failed')
        sim.settle()

    reproduce('SIM-9', LiveSimulation(), scenario)


@known('8763c #3')
def test_known_reply_already_under_way_takes_a_turns_reservation() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.delegate(deliver=False)
        sim.send_text()
        sim.wait_for_reply()
        sim.settle()

    reproduce('8763c #3', LiveSimulation(), scenario)


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


@pytest.mark.parametrize('dialect', ['openai', 'azure', 'xai'])
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
def test_scenario_gemini_async_pair_cut_off_by_a_user_turn(typed: bool) -> None:
    """SIM-19 also covers two asynchronous calls whose turn a user turn cut off, answered together with it."""
    sim = GeminiSimulation(strict=False, behavior=GeminiBehavior(async_tool_calls=True))
    with sim as s:
        s.send_text()
        s.call_tools(count=2)
        if typed:
            s.send_text()
        else:
            s.user_speaks(finished=True)
        s.settle()
        assert ('SIM-19', 'wait.hang') in s.checker.known_hits


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


def test_scenario_xai_hand_commit_while_vad_hears_the_user_is_misplaced() -> None:
    """SIM-27 on xAI: the hand-committed turn is recorded where its audio began, before a reply that ended first."""
    with OpenAISimulation(strict=False, openai=OpenAIOptions(dialect='xai', transcription=True)) as s:
        s.create_response()
        s.send_audio()
        s.settle()
        s.speech_start(deliver=False)
        s.commit_audio()
        s.settle()
        assert ('SIM-27', 'history.order') in s.checker.known_hits


def test_scenario_core_holds_a_reply_behind_a_turn_without_its_transcript() -> None:
    """SIM-25 in the new core: the reply after a reconnect waits behind the untranscribed turn, so it goes missing."""
    with OpenAISimulation(strict=False, openai=OpenAIOptions(dialect='xai', transcription=True)) as s:
        s.create_response()
        s.send_audio()
        s.commit_audio()
        s.speak(deliver=False)
        s.finish(deliver=False)
        s.deliver()
        s.drop()
        s.settle()
        s.create_response()
        s.settle()
        assert s.checker.shadow is not None
        assert ('SIM-25', 'shadow.response.missing') in s.checker.shadow.known_hits


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


def test_scenario_xai_resumption() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.settle()
        sim.drop()
        sim.settle()
        sim.send_text()

    run_tolerant(OpenAISimulation(openai=OpenAIOptions(dialect='xai')), scenario)


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

    Accepted on #8760: speech between the two results goes after the second (each result must directly follow
    its call), but what was said before the first result belongs to the calling response.
    """

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(count=2)
        sim.speak()
        sim.finish_tool()
        sim.speak()

    run_tolerant(async_gemini(), scenario)


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


def test_scenario_tool_result_awaiting_its_reply_when_barged_in_on_without_a_vad_reply() -> None:
    """SIM-32 also drops the reply a cancelled response's tool results were waiting for."""
    sim = OpenAISimulation(
        strict=False,
        openai=OpenAIOptions(dialect='azure', transcription=False, vad_interrupts=True, vad_responds=False),
    )
    with sim as s:
        s.send_image(respond=True)
        s.call_tool()
        s.tick(ticks=0)
        s.tick(ticks=0)
        s.speech_start(deliver=False)
        s.settle()
        assert ('SIM-32', 'wait.hang') in s.checker.known_hits


def test_scenario_gemini_tool_result_sent_on_a_dropped_connection() -> None:
    """G3b: the tool result the session sends hits the dropped socket, which ends the event stream over the re-dial."""
    with GeminiSimulation(strict=False) as s:
        s.send_text()
        s.speak(deliver=False)
        s.call_tools(deliver=False)
        s.deliver()
        s.drop()
        s.settle()
        s.send_text()
        s.settle()
        assert ('G3b', 'wait.hang') in s.checker.known_hits


def test_scenario_live_delegation_ends_the_wait_on_a_tool_round() -> None:
    """8763c #3: a delegation the model starts on its own ends a wait owed the tool round's answer."""
    with LiveSimulation(strict=False) as s:
        s.delegate(deliver=False)
        s.backend_call(deliver=False)
        s.wait_for_reply()
        s.advance_time(0.1)
        s.wait_for_reply()
        s.backend_finish(deliver=False)
        s.delegate(deliver=False)
        s.settle()
        assert ('8763c #3', 'wait.early') in s.checker.known_hits


def test_scenario_a_provider_reply_before_any_metadata_echo_is_accepted() -> None:
    """Before the server has echoed any request metadata, a server VAD reply read while our request is outstanding
    is taken for its answer: an accepted limitation of the lifecycle tracker's inference."""
    sim = OpenAISimulation(openai=OpenAIOptions(transcription=False, dialect='xai'))
    sim.strict = False
    with sim as s:
        s.send_audio()
        s.speech_start(deliver=False)
        s.speech_stop(deliver=False)
        s.finish(deliver=False)
        s.send_text(respond=True)
        s.settle()
        assert s.checker.shadow is not None
        assert ('SIM-24', 'shadow.history.order') in s.checker.shadow.known_hits


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


def test_baseline_xai_resumption_replays_a_tool_round() -> None:
    """xAI's resumed conversation includes the tool call and its output, which the session must not run again."""

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
        assert replayed == ['message', 'function_call', 'function_call_output', 'message']
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
        sim.settle()
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


def _run_to_the_end(finding_id: str, sim: Simulation, scenario: Callable[[Any], object]) -> None:
    """Run a pinned scenario with its finding tolerated, so the shadow core is judged to the end of it."""
    run_tolerant(sim, scenario)


@pytest.mark.parametrize('scenario', sorted(name for name in dict(globals()) if name.startswith('test_known_')))
def test_the_core_gets_the_known_findings_right(scenario: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every pinned scenario again, run to its end: the shadow core breaks no invariant outside `SHADOW_PENDING`."""
    monkeypatch.setitem(globals(), 'reproduce', _run_to_the_end)
    globals()[scenario]()
