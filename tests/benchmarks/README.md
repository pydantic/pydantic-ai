# Benchmarks

```sh
uv sync --frozen
uv run pytest tests/benchmarks --codspeed --codspeed-mode=walltime
```

Run these commands from the repository root. For a correctness check without timing, omit the CodSpeed flags.

The [CodSpeed workflow](../../.github/workflows/benchmark.yml) measures wall time on CodSpeed's dedicated ARM64 Graviton runners, with Python 3.14. It runs on pull requests and pushes to `main`. Fixed hardware avoids the CPU differences of GitHub-hosted runners, while isolation reduces timing noise.

You need [CodSpeed macro-runner access for public repositories](https://codspeed.io/docs/integrations/ci/github-actions/macro-runners#public-repositories). After changing the runner or measurement mode, record a fresh `main` baseline before comparing performance. Walltime results are not comparable to the previous CPU-simulation results. The job has a ten-minute timeout to bound runner usage. Superseded PR runs are canceled; `main` baseline runs are kept.

The agent-run benchmarks use `TestModel` to avoid network latency. Their fixtures warm up a reused agent before measurement. BlockBuster is disabled for all benchmarks in this module because its blocking-call instrumentation changes the workload.

The synthetic-history benchmark supplies 1,000 or 5,000 consecutive assistant-response fragments without provider identity metadata. The agent must merge these into one response. Fixtures construct the history and warm up the history-processing path outside the measured test.

The replay benchmark captures a `FunctionModel` stream of 1,000 or 5,000 chunks, each containing 256 characters. Capture happens outside measurement. The test measures `CompletedStreamedResponse` replay through completion and checks its final response, not live generation or network latency. Each case allows 15 seconds of measurements so the slower replay produces more samples.

## Evaluation curves

```sh
uv run pytest tests/benchmarks/test_eval_curves.py --codspeed --codspeed-mode=walltime
```

`test_eval_curves.py` calls the public precision-recall and ROC evaluators separately. The fixtures construct and warm up real report contexts outside measurement. The tests measure extraction, threshold counting, full-resolution AUC, downsampling, and chart construction together.

The workloads contain 256 cases with 32 score buckets and 4,096 cases with 2,048 buckets. Every bucket contains equal positive and negative counts. Rows are shuffled deterministically. Each evaluator displays 16 points, while AUC uses every threshold. Assertions check the known AUC and displayed point count, not timing thresholds.

## Keyword tool search

```sh
uv run pytest tests/benchmarks/test_tool_search.py --codspeed --codspeed-mode=walltime
```

`test_tool_search.py` drives local keyword search through `Agent.run()`, `ToolSearch`, and `FunctionModel`, without provider traffic. It pins the asyncio backend used by CodSpeed; the agent capability lifecycle currently schedules asyncio tasks. Fixtures build the callable catalog and reuse one public function schema outside measurement. Each measured run includes tool preparation, search-corpus collection, ranking, discovery, and result handling.

The workloads contain 256 or 8,192 deferred tools and perform one or five sequential searches. Repeated searches must return successive pages of undiscovered tools. Stable-corpus cases keep metadata unchanged. Changing-corpus cases update one prepared description using a new dependency value on every run. The first tool alternates between matching and not matching the query, and assertions verify the resulting page shift. A unique revision suffix prevents a cache from treating the changing corpus as two warmed snapshots. This includes any future index construction or invalidation in the measured operation rather than assuming a prebuilt index is free.

Fixture warmup removes startup costs. It does not move the measured run's tool preparation or search calls outside the benchmark. The five-search workload includes its first search as well as subsequent searches.

The parallel-search cases send five or twenty `search_tools` calls in one model response. They measure two model requests, the first search and subsequent searches over one prepared corpus, tool execution, and result handling. Every call sees the same pre-discovery snapshot, so assertions require identical first-page matches for all calls. These cases complement sequential discovery and keep any future index construction inside the measured run.

## Stress testing

```sh
for run in $(seq 1 10); do
    uv run pytest tests/benchmarks --codspeed --codspeed-mode=walltime || exit 1
done
```

Run this before and after an optimization. Check that every repetition passes and that the measured change exceeds the variation between runs. Keep timing thresholds out of assertions; CodSpeed tracks performance, while assertions check correctness.

```sh
uv run python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path('.codspeed').glob('results_*.json')):
    result = json.loads(path.read_text())
    if result['instrument']['type'] == 'walltime':
        for benchmark in result['benchmarks']:
            print(f"{path.name}: {benchmark['uri']}: {benchmark['stats']['median_ns'] / 1e6:.3f} ms")
PY
```

Compare the saved `median_ns` values. The locked `pytest-codspeed` 5.0.3 scales `Time (best)` twice in its console table, so that column is not reliable for local comparisons.
