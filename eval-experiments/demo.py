"""easy_evals -- the demo (slide by slide, with talking points).

Runs fully offline with fake models. With LOGFIRE_TOKEN + `logfire[datasets]`,
slide 5 talks to real Logfire with no code change.

    uv run --project /home/user/pydantic-ai python /home/user/eval-experiments/demo.py
"""

from easy_evals import check, eval_suite
from easy_evals.hosted import experiment_url, hosted_rubric, pull, push
from fake_models import judge_model, person_agent, qa_agent, tool_agent

judge = judge_model()
agent = qa_agent()


def slide(n: int, title: str, *talking_points: str) -> None:
    """Print a slide header with talking points."""
    print(f'\n\033[1m{"━" * 72}\n  SLIDE {n}: {title}\033[0m')
    for point in talking_points:
        print(f'  • {point}')
    print('\033[1m' + '━' * 72 + '\033[0m')


# ─────────────────────────────────────────────────────────────────────────────
slide(
    1,
    'The 30-second eval',
    'One line. No Dataset, no Case, no boilerplate. Reads like an assert.',
    "expect= is forgiving: '...is Paris.' still matches 'Paris' (no exact-match trap).",
    'Raises AssertionError on failure, so it works as-is inside pytest or a script.',
)
check(agent, 'What is the capital of France?', expect='Paris')
print('  >>> passed  (expect= is forgiving, so the full sentence still matches)')

# ─────────────────────────────────────────────────────────────────────────────
slide(
    2,
    'A suite: the whole matcher vocabulary, no lambdas',
    'expect / equals / contains / excludes / one_of / matches / max_words / judge ...',
    'Each kwarg maps to a real pydantic_evals evaluator -- this is a facade, not a 2nd engine.',
    'run() handles concurrency and prints the familiar report; returns True if all passed.',
)
suite = eval_suite(agent, judge_model=judge)
suite.case('What is the capital of France?', expect='Paris')                  # forgiving
suite.case('What is 2 + 2?', equals='2 + 2 = 4.', max_words=10)               # strict + length
suite.case('Sentiment of "I love this"', one_of=['positive', 'negative'])    # classification
suite.case('What is the capital of Japan?', excludes='Paris')                 # negative
suite.case('Write a haiku about the sea.', judge='is a haiku (three lines)')  # LLM judge
suite.run()

# ─────────────────────────────────────────────────────────────────────────────
slide(
    3,
    'Agent-aware matchers (the part beginners cannot do today)',
    'calls_tool= asserts the agent actually invoked a tool -- we auto-configure OTel for you.',
    'expect={...} does a partial field match on structured (Pydantic) output.',
)
check(tool_agent(), 'Weather in Paris?', expect='sunny', calls_tool='get_weather')
print('  >>> passed  (answer matched AND the get_weather tool was actually called)')
check(person_agent(), 'Make a person named Ada aged 36', expect={'name': 'Ada', 'age': 36})
print('  >>> passed  (structured output: name + age fields matched)')

# ─────────────────────────────────────────────────────────────────────────────
slide(
    4,
    'The same suite as pytest tests',
    'test_agent = as_pytest(suite)  ->  one green/red item per case.',
    'The suite runs ONCE per session (concurrently); per-case results map to pytest items.',
)
print('  See tests/test_pytest_plugin.py')
print('  Run:  uv run python -m pytest tests/test_pytest_plugin.py -v')

# ─────────────────────────────────────────────────────────────────────────────
slide(
    5,
    'Iterate: repeat for flakiness, diff against a baseline, generate cases',
    'repeat=N runs each case N times (LLM output is flaky) and reports the pass-rate.',
    'baseline= diffs this run against an earlier report -- did my change help?',
    "suite.generate(n=10, about='...') LLM-generates starter cases (needs a model).",
)
iter_suite = eval_suite(agent, judge_model=judge)
iter_suite.case('What is the capital of France?', expect='Paris')
iter_suite.case('What is 2 + 2?', contains='4')
baseline = iter_suite.report()  # capture a baseline, then re-run with repeat + diff
iter_suite.run(repeat=3, baseline=baseline)
print("  (to generate cases:  suite.generate(n=10, about='refund questions'))")

# ─────────────────────────────────────────────────────────────────────────────
slide(
    6,
    'Logfire-hosted datasets + managed-variable rubrics',
    'judge=hosted_rubric(...) grades with a rubric your team edits in the Logfire UI (no deploy).',
    'push() -> LogfireAPIClient.push_dataset;  pull() -> get_dataset (a real pydantic_evals.Dataset).',
    'Results stream to the Evals UI via OTel. Offline here; identical call sites go live with a token.',
)
hosted = eval_suite(agent, judge_model=judge)
hosted.case('What is the capital of France?', expect='Paris')
hosted.case('Write a haiku about the sea.', judge=hosted_rubric('prompt__haiku_rubric', label='production'))
print('  pushed ->', push(hosted, 'qa-regression', description='QA smoke set'))

dataset = pull('qa-regression')


async def task(question: str) -> str:
    """The system under test: just run the agent."""
    return (await agent.run(question)).output


dataset.evaluate_sync(task, name='nightly').print(include_input=True, include_durations=False)
print('  experiment ->', experiment_url('nightly'))
