# Making Pydantic AI evals beginner-friendly — experiment findings

## The question
"Nobody uses evals because of how hard they are." Can we make writing evals,
datasets, and cases for a Pydantic AI agent trivial — for someone almost using a
computer for the first time?

## What I did
1. Read the current `pydantic_evals` API + docs end to end.
2. Researched the DX of 9 other eval/test frameworks (Vitest, Braintrust,
   OpenAI Evals, promptfoo, Vercel AI SDK, DeepEval, Inspect AI, Ragas,
   LangSmith).
3. Built the **same eval two ways** against an offline agent:
   - `baseline_today.py` — current `pydantic_evals` API.
   - `easy_evals.py` + `demo_new.py` — a prototype ergonomic façade.
4. Ran a **blind usability test**: 6 sub-agents role-playing near-beginners,
   each given ONE API reference doc and the same task (check 3 trivia answers +
   "is this a haiku"). 2 on today's API, 4 on the prototype.

## What the research said (cross-framework synthesis)
The loved tools all converge on the same shape, and the disliked one (OpenAI
Evals) does the opposite:
1. **Plain-English LLM judge as the default** for fuzzy output (DeepEval
   `GEval`, Ragas `AspectCritic`, promptfoo `llm-rubric`). Biggest
   friction-killer for non-experts: write a *sentence*, get a score + reason.
2. **A scorer is just a function returning bool/number** (LangSmith, Braintrust,
   Inspect). No mandatory metric classes.
3. **pytest-native** + reads-like-English naming (DeepEval, Vitest).
4. **Three-noun model, sub-5-line hello world**: dataset → task → scorers, with
   everything else defaulted (Braintrust `Eval(data, task, scores)`).
5. **Colocation + zero-config + model-at-the-edge** (Vitest in-source, Inspect
   `--model`, Vercel mock model). Don't scatter data/config into magic dirs.

## The current API's friction (felt while writing `baseline_today.py`)
- Separate vocabulary to learn before writing one assertion: `Case`, `Dataset`,
  `Evaluator`, `EvaluatorContext`, and *which* of ~7 evaluators to pick.
- You must hand-write a **task wrapper** around the agent
  (`async def task(q): return (await agent.run(q)).output`).
- The **exact-match trap**: `EqualsExpected()` is listed first and reads like
  "the answer should be X", but LLMs answer in full sentences, so it silently
  fails. You must know to reach for `Contains`.
- Per-case `evaluators` must be a **tuple** (`(x,)`) — non-obvious.
- `LLMJudge` needs `rubric` + `model` + an understanding of score/assertion.
- `pydantic_evals` is a different package from `pydantic_ai`, so agent builders
  don't discover it.

## The prototype (`easy_evals.py`)
A thin façade over `pydantic_evals` (reuses its runner, judge, and report):
```python
from easy_evals import eval_suite

evals = eval_suite(agent)
evals.case('What is the capital of France?', expect='Paris')
evals.case('What is 2 + 2?', check=lambda out: '4' in out)
evals.case('Write a haiku about the sea', judge='is a haiku about the sea')
evals.run()
```
Three ideas do the heavy lifting:
- **Colocated with the agent** — no task wrapper; one import.
- **`expect=` is forgiving** — case-insensitive substring match (kills the
  exact-match trap), with optional escalation to a semantic LLM judge on a miss.
- **`judge='...'` / `check=fn`** — plain-English LLM judge (batteries included)
  and plain-function scorers, the two patterns the research flagged as the most
  beginner-friendly.

## Usability results (6 blind beginners, 1 doc each)
| API | Beginners | Ran first try | Conceptual blockers reported |
|-----|-----------|---------------|------------------------------|
| Today | 2 | 2/2* | tuple syntax, `Contains` vs `EqualsExpected`, async-def-but-`evaluate_sync`, what `IsInstance` does, writing a rubric |
| Prototype | 4 | 4/4** | none reported; "trivial" |

\* Both "today" beginners succeeded **only because the doc pre-warned them**
about the tuple and the `EqualsExpected` trap — both wrote "I got lucky the doc
flagged this." Without the warning they'd have hit the silent exact-match
failure.

\** One earlier run failed on a missing-module *path artifact* of the
experiment (the prototype isn't a real installed package); after fixing the
path, every beginner wrote essentially identical, correct code in one shot and
rated it "trivial." One even tried `pip install easy_evals` — a reminder that in
the product this must be a properly shipped, discoverable API.

Prototype beginners' code was near-identical and confident; today's beginners
all carried lingering "I trusted it but didn't understand it" confusion.

## Recommendation
1. **Ship the ergonomic façade inside `pydantic_evals`** (which already
   hard-depends on `pydantic-ai-slim`). This honours the "just point it at your
   agent" instinct **without** putting eval methods on the slim `Agent` class
   (keeps core lean per the project philosophy).
2. **Auto-wrap an `Agent` as the task** so no task function is needed; let
   `evaluate(agent, ...)` / a suite accept an agent directly.
3. **Make `expect=` forgiving by default** (substring / normalized match). This
   single change removes the most common silent failure beginners hit. Offer
   semantic matching explicitly via `judge=` (and/or opt-in escalation).
4. **Keep `judge='<sentence>'` and `check=fn`** as the two fuzzy/custom paths;
   default the judge model so it works with zero config.
5. **Return a clear pass/fail + friendly summary** from `run()`, alongside the
   existing rich report.

Net: the capability already exists in `pydantic_evals`. The unlock is almost
entirely **defaults + surface area** — a forgiving `expect`, an agent-aware
entry point, plain-English judge/`check`, and colocation. That is what turned a
"fiddly, trust-but-don't-understand" task into a "trivial, one-shot" one for
genuine beginners.

## Files
- `baseline_today.py` — eval with today's API (runs offline).
- `easy_evals.py` — prototype façade.
- `demo_new.py` — same eval with the prototype.
- `fake_models.py` — offline agent + offline LLM-judge so everything runs with
  no API keys.
- `usability/today/`, `usability/new/` — the beginner exercise (doc, setup,
  and each beginner's submitted `my_eval*.py`).
