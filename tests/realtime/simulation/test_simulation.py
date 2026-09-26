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
from typing import Any

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from hypothesis import HealthCheck, settings
    from hypothesis.stateful import RuleBasedStateMachine, run_state_machine_as_test

    from ._findings import FINDINGS_BY_ID
    from ._gemini import GeminiBehavior, GeminiMachine, GeminiSimulation
    from ._live import LiveMachine, LiveSimulation
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
        suppress_health_check=list(HealthCheck),
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
    run_state_machine_as_test(machine, settings=exploration_settings())


# --- known findings, pinned ------------------------------------------------------------------------


def known(finding_id: str) -> pytest.MarkDecorator:
    """Mark a scenario as reproducing a known finding: it must raise `FindingReproduced` until the fix lands."""
    finding = FINDINGS_BY_ID[finding_id] if imports_successful() else finding_id
    return pytest.mark.xfail(raises=FindingReproduced, strict=True, reason=str(finding))


def reproduce(finding_id: str, sim: Simulation, scenario: Callable[[Any], object]) -> None:
    with sim.enforcing(finding_id) as s:
        scenario(s)


@known('OR3')
def test_known_merged_requests_leak_reservations() -> None:
    """Two turns typed while the first is answered: the connection merges their requests into one."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.send_text()
        sim.send_text()
        sim.settle()

    reproduce('OR3', OpenAISimulation(), scenario)


@known('OR8')
def test_known_raising_tool_leaves_wait_hanging() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.call_tool()
        sim.finish()
        sim.finish_tool(outcome='error')
        sim.settle()

    reproduce('OR8', OpenAISimulation(), scenario)


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


@known('SIM-1')
def test_known_reply_lost_to_a_drop_keeps_its_reservation_gemini() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.drop()
        sim.settle()

    reproduce('SIM-1', GeminiSimulation(), scenario)


@known('SIM-2')
def test_known_turn_sent_before_reply_content_recorded_ahead_of_it() -> None:
    """The first reply ended empty before the second turn was sent, but the client hadn't read it yet."""

    def scenario(sim: OpenAISimulation) -> None:
        sim.send_text()
        sim.finish(deliver=False)
        sim.send_text(respond=False)
        sim.settle()

    reproduce('SIM-2', OpenAISimulation(), scenario)


@known('SIM-2')
def test_known_wait_returns_before_a_started_vad_reply() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop()
        sim.wait_for_reply()

    reproduce('SIM-2', OpenAISimulation(), scenario)


@known('SIM-2')
def test_known_wait_returns_before_a_delegated_reply() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.wait_for_reply()

    reproduce('SIM-2', LiveSimulation(), scenario)


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


@known('G2b')
def test_known_gemini_parallel_calls_split_into_responses() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(count=2)

    reproduce('G2b', GeminiSimulation(), scenario)


@known('G2a')
def test_known_gemini_batch_answer_leaks_reservations() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools(count=2)
        sim.settle()

    reproduce('G2a', GeminiSimulation(), scenario)


@known('8766')
def test_known_gemini_tool_turn_boundary_ends_the_wait_early() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.call_tools()
        sim.wait_for_reply()
        sim.finish_tool()
        sim.speak()

    reproduce('8766', GeminiSimulation(behavior=GeminiBehavior(closes_tool_turn_separately=True)), scenario)


@known('G6')
def test_known_gemini_resumed_session_forgets_a_tool_call() -> None:
    """The resumption handle was issued after the first turn, before the call the second turn makes."""

    def scenario(sim: GeminiSimulation) -> None:
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.send_text()
        sim.call_tools()
        sim.drop()
        sim.advance_time(1)
        sim.finish_tool()
        sim.settle()

    reproduce('G6', GeminiSimulation(), scenario)


@known('G3')
def test_known_send_during_reconnect_fails_openai() -> None:
    def scenario(sim: OpenAISimulation) -> None:
        sim.drop(ticks=0)
        sim.send_audio()
        sim.settle()

    reproduce('G3', OpenAISimulation(), scenario)


@known('G3')
def test_known_send_during_reconnect_fails_gemini() -> None:
    def scenario(sim: GeminiSimulation) -> None:
        sim.drop(ticks=0)
        sim.send_audio()
        sim.settle()

    reproduce('G3', GeminiSimulation(), scenario)


@known('SIM-6')
def test_known_live_parallel_calls_leak_reservations() -> None:
    def scenario(sim: LiveSimulation) -> None:
        sim.delegate()
        sim.backend_call(count=2)
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
