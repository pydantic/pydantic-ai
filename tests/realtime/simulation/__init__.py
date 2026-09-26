"""A deterministic simulator for `RealtimeSession`: randomized interleavings checked against invariants.

The realtime session's bugs live in orderings: a send racing a frame, a reconnect landing mid-response,
a tool finishing while a reply streams. Cassette tests replay one recorded ordering each; this package
explores the rest. Each test drives a *real* `RealtimeSession`, with a real connection class for its
provider, against a simulated server over a fake socket:

- `_loop.py`: an asyncio loop on a virtual clock. Timers (reconnect backoff, GPT-Live's turn clock)
  fire only when a step moves time on, and a step can let exactly `n` loop iterations pass, so a trace
  decides which task wins each race, and the same trace always runs the same way.
- `_wire.py`: the fake WebSocket and network shared by the OpenAI-protocol and GPT-Live servers, with
  send latency, send faults (lost, or delivered but reported failed), drops, and refused re-dials.
  Server frames stay in flight until the trace delivers them.
- `_openai_server.py` (OpenAI Realtime, and the Azure OpenAI and xAI dialects), `_gemini.py` (Gemini
  Live), `_live.py` (GPT-Live): small models of each provider's server, built from the recorded
  cassettes and the live stress runs. Each keeps `_truth.py`'s `GroundTruth`: what the provider
  received, generated, billed, and when the client read it, independent of anything the session says.
- `_simulation.py`: the `Simulation` a trace drives. Every public method is a step (a client operation,
  a provider behavior, a fault, or letting time pass), recorded as the Python call that replays it.
- `_invariants.py`: the properties checked after every step (history is append-only, usage matches what
  was billed, `wait_for_reply()` returns neither early nor never, ...) and once the trace comes to rest
  (every answered input is recorded, in order, with its whole reply). The module docstring lists them.
- `_conformance.py`: the codec event lifecycle contract, checked on the simulated traces and on every
  recorded WebSocket cassette (`test_conformance.py`, via `_cassette_replay.py`).
- `_machine.py` and the `*Machine` classes: Hypothesis state machines that pick the steps, shrink a
  failure to a minimal trace, and report it as code you can paste into a test.
- `_findings.py`: the violations current main is known to have, each tied to the PR or redesign phase
  that fixes it.

Running it
----------

In CI, `test_exploration` runs a small, derandomized batch per provider. For a long, random run:

```bash
REALTIME_SIMULATION_EXAMPLES=5000 REALTIME_SIMULATION_STEPS=40 \\
    uv run pytest tests/realtime/simulation/test_simulation.py -k exploration
```

Known findings are tolerated there (`--hypothesis-show-statistics` counts them), so only something new
fails; set `REALTIME_SIMULATION_STRICT=1` to fail on those too. A failure prints the replayable trace:

```python
sim = OpenAISimulation()
sim.send_text()
sim.send_text()
sim.drop()
sim.settle()
```

Adding a finding, and retiring one
----------------------------------

A new violation that's a real bug gets an entry in `_findings.py` (the invariant codes it shows up as, the
providers, and a predicate that recognizes its trigger, so it doesn't mask anything else) and a pinned
`test_known_*` scenario in `test_simulation.py`: the shrunk trace, run with `sim.enforcing(<id>)` under a
strict `xfail`. When the fix lands, the pinned scenario passes and fails as `XPASS(strict)`: delete the
finding, drop the `xfail` mark, and keep the scenario as a regression test.

A violation that's the simulator's fault (a server doing something the real provider doesn't) is fixed in
the server model, ideally against a cassette that shows what the provider does.
"""
