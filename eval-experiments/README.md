# easy_evals

The easiest way to write and run evals for a Pydantic AI agent — a thin,
beginner-friendly **facade over `pydantic_evals`** that adds *no* evaluation
logic of its own. Built to explore "what would make evals so easy that people
actually write them?" (and to demo a Logfire-hosted story).

## The 30-second version
```python
from easy_evals import check, eval_suite

# one assert-style check (great in a script or as a pytest test)
check(agent, 'What is the capital of France?', expect='Paris')

# a suite: batch run, concurrency + a report handled for you
suite = eval_suite(agent)
suite.case('What is the capital of France?', expect='Paris')        # forgiving match
suite.case('What is 2 + 2?', equals='4', max_words=10)              # strict + length
suite.case('Sentiment of "I love this"', one_of=['positive', 'negative'])
suite.case('Write a haiku', judge='is a haiku (three lines)')      # LLM judge
suite.run()
```

## Why it's easy
- **One import**, colocated with the agent you already have. No `Case`/`Dataset`/
  `Evaluator` vocabulary required to start.
- **`expect=` is forgiving** (case-insensitive substring) so beginners don't hit
  the exact-match trap; `equals=` is there when you want strict.
- **Declarative matchers, no lambdas** for the common cases.
- **Plain-English `judge=`** LLM-as-judge, batteries included.
- Everything is a real `pydantic_evals` object underneath, so you can always drop
  down to the full library (serialization, generation, retries, span evals).

## Matcher vocabulary (every kwarg → a real evaluator)
| kwarg | maps to | notes |
|---|---|---|
| `expect='Paris'` | `Contains(case_sensitive=False)` | forgiving text match |
| `expect={'name': 'Ada'}` | `HasFields` *(proposed)* | partial match on structured output |
| `equals='4'` | `Equals` | strict |
| `contains=` / `excludes=` | `Contains` / `NotContains` *(proposed)* | str or list |
| `one_of=['a','b']` | `OneOf` *(proposed)* | classification |
| `matches=r'...'` | `Matches` *(proposed)* | regex |
| `max_words=` / `max_chars=` | `MaxLength` *(proposed)* | length budget |
| `max_duration=2` | `MaxDuration` | latency |
| `judge='is polite'` | `LLMJudge` | plain-English rubric |
| `calls_tool='get_weather'` | `HasMatchingSpan` | did the agent call the tool? (auto-instruments tracing) |
| `passes=fn` | tiny function wrapper | escape hatch |

## Three altitudes, one vocabulary
1. **Immediate** — `check(agent, prompt, ...)` runs now and asserts. Works in a
   script or as an ordinary pytest test.
2. **Suite** — `eval_suite(agent).case(...).run()`. Batch run; owns concurrency
   (`max_concurrency`) and the aggregate report.
3. **Explicit** — drop to `Case` / `Dataset` / `Evaluator` for full control.

## pytest
```python
from easy_evals import eval_suite
from easy_evals.pytest_plugin import as_pytest

suite = eval_suite(agent)
suite.case('Capital of France?', expect='Paris')
suite.case('Write a haiku', judge='is a haiku')

test_agent = as_pytest(suite)   # one concurrent run, one green/red pytest item per case
```

## Logfire-hosted datasets & managed-variable rubrics
Grounded in the **real** Logfire API; falls back to an offline simulation with
identical call sites when there's no `LOGFIRE_TOKEN` / `logfire[datasets]`.

```python
from easy_evals.hosted import push, pull, hosted_rubric

# grade with a rubric stored as a Logfire managed variable -- your team edits it
# in the UI (versions + production/canary labels), no code change, no redeploy
suite.case('Summarize', judge=hosted_rubric('prompt__summary_rubric', label='production'))

push(suite, 'qa-regression')          # -> logfire.experimental.api_client push_dataset
dataset = pull('qa-regression')       # -> real pydantic_evals.Dataset, ready to evaluate
report = dataset.evaluate_sync(task)  # results stream to the Logfire Evals UI via OTel
```
Real API used under the hood: `LogfireAPIClient.push_dataset` / `get_dataset`
(https://pydantic.dev/docs/logfire/evaluate/datasets/sdk/) and
`logfire.var(name).get(label=...).value`
(https://pydantic.dev/docs/logfire/manage/managed-variables/).

## Iterating on results
Three conveniences over `pydantic_evals` for the eval loop:
```python
# 1. Beat the blank page: let an LLM generate starter cases
suite.generate(n=10, about='customer-support questions about refunds')

# 2. Trust the result: run each case N times (LLM output is flaky) -> pass-rate
suite.run(repeat=5)

# 3. Close the loop: did my change help? Diff against an earlier run
before = suite.report()
# ...change the prompt / model...
suite.run(baseline=before)        # prints a side-by-side diff
```
All three are thin wrappers: `generate` over `pydantic_evals.generation.generate_dataset`,
`repeat` over `Dataset.evaluate(repeat=...)`, and `baseline` over `report.print(baseline=...)`.

## What we'd propose upstream
- **New built-in evaluators** in `pydantic_evals.evaluators`: `NotContains`,
  `OneOf`, `Matches`, `MaxLength`, `HasFields` (each a small, standalone,
  serializable `Evaluator` — usable without the facade).
- **The `easy_evals` facade** (`check`, `eval_suite`, the matcher kwargs) as the
  beginner front door that builds plain `pydantic_evals` objects.
- A **real pytest plugin** (session-scoped fixture, one item per case).
- First-class **Logfire hosted** helpers (`push`/`pull`/`hosted_rubric`) so the
  hosted-datasets + managed-variable story is one obvious import.

## Demo & tests
The hosted integration's *live* path and the strict typecheck of `hosted.py` need
**`logfire >= 4.35`** (hosted datasets + managed variables). The workspace
lockfile pins 4.16, so install a newer logfire into the venv locally (this does
**not** touch the lockfile — bumping it there would change Pydantic AI's
telemetry-snapshot tests and must be a separate, reviewed PR). A `Makefile`
wraps everything:
```bash
make setup       # uv pip install --no-deps 'logfire>=4.35' into the venv (local only)
make check       # ruff + pyright (strict) + pytest   (runs with --no-sync)
make demo        # the slide-by-slide demo
```
Everything except the live Logfire path and `hosted.py`'s typecheck also works on
plain logfire 4.16 — the facade itself imports OpenTelemetry, not logfire.

## Types
The facade is **generic over the agent's output type** and passes `pyright`
**strict** with **no errors or warnings**, using the **same flags as upstream
pydantic-ai** (`typeCheckingMode=strict`, `reportUnnecessaryTypeIgnoreComment`,
etc. — see `pyrightconfig.json`). `expect=`, `passes=` and structured checks are
type-checked against what the agent actually returns — e.g.
`check(person_agent(), passes=lambda p: p.age > 0)` type-checks, while
`p.upper()` is a type error. The hosted helpers are fully typed against the real
`logfire.experimental.api_client` and `logfire.var` APIs — **no `# type: ignore`
anywhere**. The only `Any` is the agent's *deps* slot (`AbstractAgent[Any,
OutputT]`, matching pydantic-ai's own convention — evals don't care about deps)
and the in-process offline dataset store (a dynamic boundary mirroring the real
client's `Dataset | dict` return).

## Caveats (prototype)
- `as_pytest` runs the suite once per session via a real session-scoped fixture;
  under `pytest-xdist` each worker session runs it once (inherent to xdist).
- `calls_tool` auto-sets a global OTel `TracerProvider` and calls
  `Agent.instrument_all()` — a process-wide side effect (fine for an eval run).
- Hosted features need `logfire >= 4.35` and a `LOGFIRE_TOKEN` to go *live*;
  without them the call sites run in offline-simulation. The datasets client is
  still `logfire.experimental.*` and may change.

## Layout
```
easy_evals/          the package
  core.py            the facade (kwargs -> real evaluators), check, eval_suite
  evaluators.py      proposed new built-ins: NotContains, OneOf, Matches, MaxLength, HasFields
  pytest_plugin.py   as_pytest(suite)
  hosted.py          push / pull / hosted_rubric (real Logfire API + offline fallback)
demo.py              slide-by-slide demo
baseline_today.py    the "before": the same eval with today's pydantic_evals API
fake_models.py       offline agent + offline LLM judge (no API keys)
tests/               test_matchers, test_facade, test_hosted, test_pytest_plugin
docs/                DESIGN.md, FINDINGS.md, EXPERIMENTS-FEEL.md (the thinking)
```
