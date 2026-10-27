# Benchmarks

```sh
uv sync --frozen
uv run pytest tests/benchmarks --codspeed --codspeed-mode=simulation
```

Run these commands from the repository root. For a correctness check without timing, omit the CodSpeed flags. Without CodSpeed, each benchmark runs only its smallest workload: the larger workloads are marked `codspeed_only` and deselected by [`conftest.py`](conftest.py), because they reach the same code paths and would only slow down the regular test jobs, which collect this directory too.

The [CodSpeed workflow](../../.github/workflows/benchmark.yml) uses CPU simulation on GitHub-hosted runners with Python 3.14. It runs on pull requests and pushes to `main`. Simulation produces hardware-independent measurements, so results remain comparable across GitHub-hosted CPUs.

After changing the measurement mode, record a fresh `main` baseline before comparing performance. Simulation results are not comparable to the previous wall-time results. The job has a thirty-minute timeout. Superseded PR runs are canceled; `main` baseline runs are kept.

The agent-run benchmarks use `TestModel` to avoid network latency. Their fixtures warm up a reused agent before measurement. BlockBuster is disabled for all benchmarks in this module because its blocking-call instrumentation changes the workload.

The synthetic-history benchmark supplies 1,000 or 5,000 consecutive assistant-response fragments without provider identity metadata. The agent must merge these into one response. Fixtures construct the history and warm up the history-processing path outside the measured test.

The replay benchmark captures a `FunctionModel` stream of 1,000 or 5,000 chunks, each containing 256 characters. Capture happens outside measurement. The test measures `CompletedStreamedResponse` replay through completion and checks its final response, not live generation or network latency.

## Evaluation curves

```sh
uv run pytest tests/benchmarks/test_eval_curves.py --codspeed --codspeed-mode=simulation
```

`test_eval_curves.py` calls the public precision-recall and ROC evaluators separately. The fixtures construct and warm up real report contexts outside measurement. The tests measure extraction, threshold counting, full-resolution AUC, downsampling, and chart construction together.

The workloads contain 256 cases with 32 score buckets and 4,096 cases with 2,048 buckets. Every bucket contains equal positive and negative counts. Rows are shuffled deterministically. Each evaluator displays 16 points, while AUC uses every threshold. Assertions check the known AUC and displayed point count, not timing thresholds.

## Keyword tool search

```sh
uv run pytest tests/benchmarks/test_tool_search.py --codspeed --codspeed-mode=simulation
```

`test_tool_search.py` drives local keyword search through `Agent.run()`, `ToolSearch`, and `FunctionModel`, without provider traffic. It pins the asyncio backend used by CodSpeed; the agent capability lifecycle currently schedules asyncio tasks. Fixtures build the callable catalog and reuse one public function schema outside measurement. Each measured run includes tool preparation, search-corpus collection, ranking, discovery, and result handling.

The workloads contain 256 or 8,192 deferred tools and perform one or five sequential searches. Repeated searches must return successive pages of undiscovered tools. Stable-corpus cases keep metadata unchanged. Changing-corpus cases update one prepared description using a new dependency value on every run. The first tool alternates between matching and not matching the query, and assertions verify the resulting page shift. A unique revision suffix prevents a cache from treating the changing corpus as two warmed snapshots. This includes any future index construction or invalidation in the measured operation rather than assuming a prebuilt index is free.

Fixture warmup removes startup costs. It does not move the measured run's tool preparation or search calls outside the benchmark. The five-search workload includes its first search as well as subsequent searches.

The parallel-search cases send five or twenty `search_tools` calls in one model response. They measure two model requests, the first search and subsequent searches over one prepared corpus, tool execution, and result handling. Every call sees the same pre-discovery snapshot, so assertions require identical first-page matches for all calls. These cases complement sequential discovery and keep any future index construction inside the measured run.

## CodSpeed dependency filter

Pull requests run CodSpeed when they change a benchmark dependency. Pushes to `main` and manual dispatch always run the full suite. The classifier stays inline in the workflow: its token-bearing job reads file names and statuses from the GitHub API without checking out or executing pull-request code. Both rename paths are checked. API failures, incomplete file lists, unknown statuses, and unknown paths run benchmarks.

The workflow's [`affects_benchmarks` function](../../.github/workflows/benchmark.yml) is the dependency map; its case branches contain the group rationale and exact paths. Imports during collection, fixtures, warmup, measured execution, and teardown all count, even outside the measured operation.

Other test suites and their data, provider implementations, and harness/CLI source skip because the isolated benchmark run never loads them. Installation and test configuration still trigger benchmarks even in skipped packages. New/copied/renamed Python source in core, graph, or evals runs until its dependencies can be audited. Unknown paths and statuses remain conservative triggers.

### Checking dependency drift

After an isolated benchmark run, the conftest checks loaded repository files against the workflow's actual `affects_benchmarks` function. It fails only when a loaded dependency would skip CodSpeed, or the classifier cannot run. Included new modules need no inventory update. Mixed or unrelated suites skip the guard so their collection imports cannot contaminate this audit. Directory paths, trailing slashes, absolute paths, individual files, and test-node selections are supported.

This executes the extracted function only in the benchmark/test process. The token-bearing classifier job still executes no pull-request files.

Check the dependencies with the workflow's installation and a correctness run:

```sh
uv sync --frozen
uv run pytest tests/benchmarks
```

If the guard reports a path, trace its import through collection, fixtures, or runtime and add its trigger before the matching exclusion in `affects_benchmarks`. Run `uv run pytest .github/scripts/test_ci_content_classifier.py` to check scheduling. The ordinary correctness command uses the smallest workloads; the full CodSpeed commands above also audit lazy imports in larger workloads. Changes to benchmark code, loaded dependencies, installation/configuration, or new core/graph/evals source trigger the full CodSpeed job, where the guard checks its actual benchmark run.
