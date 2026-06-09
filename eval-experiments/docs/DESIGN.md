# Easy evals — design notes (iteration 2)

Builds on FINDINGS.md. This captures the decisions from exploring: more eval
types, lambda-free matchers, run() vs immediate/pytest, naming (crowd-sourced
from sub-agents), the facade principle, and the Logfire-hosted vision.

## Guiding principle (from the maintainer): FACADE, not a second engine
The ergonomic layer must be a thin translation over the existing
`pydantic_evals` primitives — friendly kwargs map to existing `Evaluator`
instances; cases are real `Case`s; runs go through the real `Dataset.evaluate`;
pass/fail is read from the real `EvaluationReport`. The facade holds **zero
evaluation logic**.

Where a beginner matcher has no existing built-in, the answer is **add a small,
broadly-useful built-in `Evaluator`** (a shared primitive) — not hide logic in
the facade. Candidates implemented in `proposed_evaluators.py`: `NotContains`,
`OneOf`, `Matches`, `MaxLength`.

Translation table (the entire facade):

| friendly kwarg | evaluator | status |
|---|---|---|
| `expect=` (forgiving, ci-substring) | `Contains(value, case_sensitive=False)` | exists |
| `equals=` (strict) | `Equals(value)` | exists |
| `contains=` (str or list) | `Contains(...)` | exists |
| `excludes=` | `NotContains(...)` | **propose** |
| `one_of=` | `OneOf(...)` | **propose** |
| `matches=` (regex) | `Matches(pattern)` | **propose** |
| `max_words=` / `max_chars=` | `MaxLength(...)` | **propose** |
| `max_duration=` | `MaxDuration(seconds)` | exists |
| `judge=` (plain English) | `LLMJudge(rubric=...)` | exists |
| `passes=` (function escape hatch) | tiny `_Check` wrapper | facade-only |

## Eval types a beginner actually wants (and how they map)
- "the answer is Paris" → `expect='Paris'` (forgiving) / `equals=` (strict)
- "mentions X" / "doesn't mention X" → `contains=` / `excludes=`
- classification "one of {yes,no}" → `one_of=[...]`
- format "looks like an email / starts with {" → `matches=r'...'`
- "keep it short" → `max_words=` / `max_chars=`
- "fast enough" → `max_duration=`
- subjective "is polite / is a haiku / is factual" → `judge='...'`
- agent-specific "called the weather tool" → maps to the existing
  `HasMatchingSpan` evaluator; a future `calls_tool='get_weather'` sugar is the
  natural next matcher (span-based, very valuable for agent builders)
- structured output field checks (Pydantic-native) → future `expect={...}`
  partial-dict match for agents with `output_type`
- anything else → `passes=<function>` or drop to explicit `evaluators=[...]`

## Avoiding lambdas
Almost every common case is now declarative (`contains='4'` instead of
`check=lambda o: '4' in o`). `passes=<function>` remains only as the rarely-used
escape hatch. Beginners never need a lambda for the basics.

## run() vs immediate vs pytest — offer THREE altitudes, one vocabulary
All three share the same matchers and the same underlying engine.

1. **Immediate / pytest** — `check(agent, prompt, expect=...)` runs now and
   asserts (raises `AssertionError` with a friendly message read from the real
   report). Works in a plain script or as an ordinary pytest test. Lowest floor.
2. **Suite** — `eval_suite(agent)` + `.case(...)` + `.run()`. Batch run;
   concurrency (`max_concurrency`) and one report handled for you. Best for
   exploration / notebooks.
3. **Explicit** — drop to `Case` / `Dataset` / `Evaluator` / `LLMJudge` for full
   control, serialization, generation, span-based evals, report evaluators.

### On concurrency + pytest (the user's question)
- The **suite** path already gives in-process async concurrency for free via
  `Dataset.evaluate(max_concurrency=...)`. This is the right tool when you want
  many cases to run fast.
- The **pytest** path gives per-case PASS/FAIL, CI integration, and familiar
  tooling. Two patterns both work today through the facade:
  - one `check(...)` per `def test_*`, or
  - `@pytest.mark.parametrize` over a list of matcher-dicts calling `check`.
  Cross-test concurrency in pytest is process-level (pytest-xdist); for in-process
  concurrency use the suite. A future pytest plugin could run the whole dataset
  once concurrently in a session fixture and surface one test item per case —
  best of both — but that's a "grow into it" addition, not needed for v1.
- Verdict: **keep `run()`** (it owns concurrency + the aggregate report) AND ship
  `check()` for immediate/pytest. They serve different moments; neither replaces
  the other.

## Naming (decisions from the sub-agent survey: 2 beginners + 1 pythonic taste)
Strong consensus:
- One-shot function → **`check`** (unanimous #1; `expect`/`grade`/`assert_eval`
  rejected; `expect` collides with the kwarg).
- Run method → **`.run()`** (unanimous; `report` implies print-only).
- Collection → **`EvalSuite`** via **`eval_suite(agent)`** factory. Avoid
  `Eval`/`Evals` — "eval" is badly overloaded (`pydantic_evals`, `eval_suite`,
  `run_evals`, builtin `eval`) and beginners lose track.
- **Reject methods-on-Agent (`agent.check`, `agent.evals()`)**: beginners love
  the discoverability, but evals apply to *any* task/callable, not just agents,
  and the framework prizes a slim core — don't weld eval methods onto the hottest
  class or invert the core→evals dependency.
- **Don't ship a lowercase `case()` factory** next to capital `Case` (differs
  only by case = unguessable). Add cases via the `suite.case(...)` *method*
  (method-vs-class doesn't collide); keep capital `Case` as the one canonical
  class for explicit/serialized construction.
- Matcher renames for consistency + readability:
  - `not_contains=` → **`excludes=`**
  - `max_seconds=` → **`max_duration=`** (mirrors `MaxDuration`, accept
    float seconds or `timedelta`)
  - `check=` (kwarg) → **`passes=`** (avoids echoing the `check()` function;
    beginners were unsure `check=` took a function)
  - keep `contains`/`judge` (mirror `Contains`/`LLMJudge`)
- `expect=` should be **forgiving (ci-substring) by default** — the single most
  valuable bit of magic for this audience (kills the exact-match trap) — with a
  separate honest **`equals=`** for strict equality. No magical boolean flag.

Rule of thumb adopted: **a kwarg mirrors its evaluator's name** when one exists
(`contains`→`Contains`, `max_duration`→`MaxDuration`, `judge`→`LLMJudge`,
`equals`→`Equals`). Guessable in both directions.

### Recommended surface
```python
from pydantic_evals import check, eval_suite

# one-shot, assert-style (great in pytest or a script)
check(agent, 'Capital of France?', expect='Paris')     # forgiving: ci-substring
check(agent, '2 + 2?', equals='4', max_duration=2)     # strict + latency

# a suite: run together, reported together, concurrency handled
suite = eval_suite(agent)
suite.case('Capital of France?', expect='Paris')
suite.case('Summarize this', judge='is concise and factual', excludes='lorem ipsum')
suite.case('Sentiment of "I love this"', one_of=['positive', 'negative', 'neutral'])
report = suite.run()        # -> rich report; report.assert_passed() for CI
```

## Growth path (so it can expand without regret)
- New matcher = new built-in `Evaluator` + one line in the translation table.
- `judge=` can grow `score=`/`rubric` config by accepting an `LLMJudge` directly.
- `calls_tool=` / span matchers reuse `HasMatchingSpan`.
- Structured `expect={...}` partial match for typed agent outputs.
- Everything is convertible to `Case`/`Dataset`, so serialization (YAML/JSON),
  `generate_dataset`, retries, and Logfire all come along for free.

## Logfire-hosted datasets & evaluators (mirror "managed prompts")
Vision: make hosted eval assets as slick as managed prompts — pull a dataset or
a judge rubric by name from Logfire, run locally, results stream back to the UI.
(API shape to mirror managed prompts exactly — see LOGFIRE.md once the
managed-prompt API is confirmed.)
