# Three experiments: what feels great vs janky

All three are built on the facade (zero eval logic of its own) and run offline.

## A. Agent/tool matchers  — VERDICT: GREAT ✅
`calls_tool='get_weather'` (maps to `HasMatchingSpan`) and structured
`expect={'name': 'Ada', 'age': 36}` (maps to a new `HasFields` evaluator).

What feels great:
- These are the evals agent-builders actually want ("did it call the tool?",
  "are the structured fields right?") and they're one kwarg each.
- `expect=` transparently switches between text (ci-substring) and structured
  (partial field match) based on the value type — no new vocabulary.
- The facade auto-configures OpenTelemetry + instrumentation, so `calls_tool`
  "just works" with no setup. This is the whole game: span evals are powerful
  but normally need OTel boilerplate; hiding that is what makes it non-janky.

Jank / caveats:
- Auto-tracing sets a global `TracerProvider` and calls `Agent.instrument_all()`
  — a process-wide side effect. Fine for an eval run; should be documented and
  ideally scoped.
- The span query hardcodes `name='running tool'` + `gen_ai.tool.name`. That's
  the stable gen_ai OTel attribute, but the facade is now coupled to a span
  shape; worth a tiny helper in core so it can't drift.

Recommendation: ship `calls_tool=` and structured `expect={...}`. Make
"the facade owns tracing setup for span matchers" a real, documented feature.

## B. pytest plugin — VERDICT: GREAT CONCEPT, HACKY PROTOTYPE ⚠️
`test_agent = as_pytest(suite)` → the whole suite runs ONCE (concurrently) and
you still get one pytest item per case.

What feels great:
- You get per-case PASS/FAIL in the normal pytest output AND a single concurrent
  run (not N sequential agent calls). That's genuinely best-of-both.
- Writing it is tiny: build a suite, hand it to `as_pytest`.

Jank / caveats:
- The "run once" uses a lazy module-level cache — a hack. Under `pytest-xdist`
  each worker would re-run the whole suite (cache isn't shared across processes).
- An error in the run itself (not a case assertion) surfaces on whichever case
  ran first. A real plugin should use a session-scoped fixture +
  `pytest_generate_tests`/collection hook so the run is a first-class session
  step.
- `test_agent = as_pytest(suite)` is a slightly magical idiom; auto-generated
  case IDs are ugly unless you pass `name=`.

Recommendation: pursue, but as a proper pytest plugin (session fixture +
collection hook), not the lazy-cache shim. The call-site feel is right.

## C. Logfire-hosted datasets & evaluators — VERDICT: SLICK AT THE CALL SITE ✅ (mock)
`cases=hosted_cases('qa-regression')`, `judge=hosted_rubric('is-a-haiku')`,
`suite.run(experiment='nightly')` → prints a Logfire link.

What feels great:
- Pulling a dataset or a judge rubric BY NAME reads beautifully and mirrors the
  managed-prompt mental model: non-engineers edit cases/rubrics in the Logfire
  UI; code just references them by name. `judge=hosted_rubric('...')` needs no
  facade change (it resolves to a string).
- `run(experiment=...)` returning a UI link closes the loop nicely.

Jank / caveats:
- Entirely mocked here; real value depends on a Logfire backend.
- Names are placeholders. To truly mirror managed prompts, the pull should be a
  classmethod consistent with the existing `Dataset.from_file` family, e.g.
  `Dataset.from_logfire('qa-regression', label='production')`, with the facade
  accepting that dataset. (Could not confirm the exact managed-prompt API in the
  repo — likely a Logfire-side feature — so the precise method name should be
  aligned before building.)

Recommendation: strongest "wow" for the managed-prompt parallel, but it's
blocked on backend + exact API alignment. Design `Dataset.from_logfire` /
`to_logfire` to mirror `from_file`, and a hosted-rubric pull for judges.

## Overall
- A is ready and high-value — do first.
- C feels the slickest but needs the Logfire backend + API alignment — design now,
  build when unblocked.
- B is the right idea; redo the mechanism as a real plugin before shipping.
