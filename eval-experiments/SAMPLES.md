# Samples: with vs. without `easy_evals`

Each scenario has two runnable files in `samples/`:
`NN_<scenario>_today.py` (today's `pydantic_evals` API) and `NN_<scenario>_easy.py`
(the `easy_evals` facade). They're functionally equivalent and run offline against
the fake models in `fake_models.py`.

Run them all: `make samples` (plus `pytest samples/06_pytest_*.py` for the pytest pair).

## At a glance (lines of code, excluding the module docstring)

| # | Scenario | today | easy | What the facade removes |
|---|----------|------:|-----:|-------------------------|
| 01 | Basic Q&A correctness | 33 | 12 | task wrapper, `Case`/`Dataset`, choosing `Contains` over `EqualsExpected` |
| 02 | Several checks on one case | 51 | 11 | **two hand-written `Evaluator` classes** (`excludes`, `max_words`) |
| 03 | LLM-as-judge | 25 | 11 | `Case`/`Dataset` ceremony; rubric is plain English |
| 04 | Structured-output fields | 40 | 11 | a hand-written partial-field `Evaluator` (`expect={...}`) |
| 05 | "Did it call the tool?" | 43 | 11 | **OTel setup + instrumentation + exact span query** |
| 06 | Evals as pytest tests | 19 | 15 | manual agent-wrapping + by-hand asserts; one concurrent run |
| 07 | Repeat (flaky) + compare | 23 | 12 | task wrapper, `Case`/`Dataset` |
| 08 | Logfire-hosted + managed rubric | 33 | 13 | client boilerplate, schema args, manual rubric fetch |

The deltas that matter most are **02, 04, 05** — today they require you to *write
evaluator classes or wire up OpenTelemetry*; with the facade they're a keyword.

---

## 01 — Basic Q&A correctness

**today**
```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, IsInstance
from fake_models import qa_agent

agent = qa_agent()

async def task(question: str) -> str:
    return (await agent.run(question)).output

dataset = Dataset(
    name='qa',
    cases=[
        Case(name='france', inputs='What is the capital of France?', expected_output='Paris',
             evaluators=(Contains(value='Paris'),)),
        Case(name='japan', inputs='What is the capital of Japan?', expected_output='Tokyo',
             evaluators=(Contains(value='Tokyo'),)),
    ],
    evaluators=[IsInstance(type_name='str')],
)
dataset.evaluate_sync(task).print(include_input=True, include_output=True)
```

**easy**
```python
from easy_evals import eval_suite
from fake_models import qa_agent

evals = eval_suite(qa_agent())
evals.case('What is the capital of France?', expect='Paris')
evals.case('What is the capital of Japan?', expect='Tokyo')
evals.run()
```

---

## 02 — Several checks on one case

**today** — no built-in `not-contains` / `max-words`, so you write them:
```python
@dataclass
class NotContains(Evaluator[object, object, object]):
    value: str
    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        return self.value.lower() not in str(ctx.output).lower()

@dataclass
class MaxWords(Evaluator[object, object, object]):
    limit: int
    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        return len(str(ctx.output).split()) <= self.limit

dataset = Dataset(name='qa', cases=[
    Case(name='japan', inputs='What is the capital of Japan?',
         evaluators=(Contains(value='Tokyo'), NotContains(value='Paris'), MaxWords(limit=20))),
])
```

**easy**
```python
evals.case('What is the capital of Japan?', contains='Tokyo', excludes='Paris', max_words=20)
```

---

## 04 — Structured-output fields

**today**
```python
@dataclass
class HasFields(Evaluator[object, object, object]):
    fields: dict[str, object] = field(default_factory=dict)
    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        return all(getattr(ctx.output, k, None) == v for k, v in self.fields.items())

dataset = Dataset(name='people', cases=[
    Case(name='ada', inputs='Make a person named Ada aged 36',
         evaluators=(HasFields(fields={'name': 'Ada', 'age': 36}),)),
])
```

**easy**
```python
evals.case('Make a person named Ada aged 36', expect={'name': 'Ada', 'age': 36})
```

---

## 05 — "Did it call the tool?"

**today** — you own the OpenTelemetry setup and must know the span shape:
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from pydantic_ai import Agent

trace.set_tracer_provider(TracerProvider())   # else no spans recorded
Agent.instrument_all()                         # else the agent emits nothing

Case(name='weather', inputs='What is the weather in Paris?', evaluators=(
    HasMatchingSpan(
        query={'name_equals': 'running tool', 'has_attributes': {'gen_ai.tool.name': 'get_weather'}},
        evaluation_name='calls_get_weather'),
))
```

**easy** — tracing is configured for you:
```python
evals.case('What is the weather in Paris?', expect='sunny', calls_tool='get_weather')
```

---

## 06 — Evals as pytest tests

**today**
```python
@pytest.mark.parametrize('question,expected', [
    ('What is the capital of France?', 'Paris'),
    ('What is the capital of Japan?', 'Tokyo'),
])
def test_qa(question, expected):
    output = agent.run_sync(question).output
    assert expected.lower() in output.lower()      # forgiving match by hand
```

**easy** — one concurrent run, one pytest item per case:
```python
suite = eval_suite(qa_agent())
suite.case('What is the capital of France?', expect='Paris')
suite.case('What is the capital of Japan?', expect='Tokyo')
test_qa = as_pytest(suite)
```

---

## 07 — Repeat (flaky output) + compare to a baseline

**today**
```python
baseline = dataset.evaluate_sync(task, repeat=3)
latest = dataset.evaluate_sync(task, repeat=3)
latest.print(baseline=baseline, include_input=True)
```

**easy**
```python
baseline = evals.report(repeat=3)
evals.run(repeat=3, baseline=baseline)
```

---

## 08 — Logfire-hosted dataset + managed-variable rubric

**today**
```python
rubric = logfire.var('prompt__haiku_rubric', default='is a haiku').get(label='production').value
dataset = Dataset(name='qa-regression', cases=[
    Case(name='haiku', inputs='Write a haiku about the sea.',
         evaluators=(LLMJudge(rubric=f'The output {rubric}.'),))])
with LogfireAPIClient() as client:
    client.push_dataset(dataset, name='qa-regression')
    fetched = client.get_dataset('qa-regression', input_type=str, output_type=str, metadata_type=type(None))
fetched.evaluate_sync(task).print()
```

**easy**
```python
evals.case('Write a haiku about the sea.', judge=hosted_rubric('prompt__haiku_rubric', label='production'))
push(evals, 'qa-regression')
evals.run(experiment='nightly-2026-06-09')
```

---

## Two more (facade-only conveniences)

**Classification** — today needs a custom `OneOf` evaluator; easy:
```python
evals.case('Sentiment of "I love this"', one_of=['positive', 'negative', 'neutral'])
```

**Generate cases** — beat the blank page (today: `generate_dataset(Dataset[str, str, None], ...)`); easy:
```python
evals.generate(n=10, about='customer-support questions about refunds')
```
