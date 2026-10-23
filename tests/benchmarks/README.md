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

The workflow's `affects_benchmarks` function is the dependency map. It includes collection, fixtures, warmup, measured execution, and teardown: an import matters even when the imported module is outside the measured operation.

| Dependency path | Why it is retained |
| --- | --- |
| `Agent.run`, `_agent_graph`, `run`, `messages`, `usage`, context/cancellation/history helpers | Measured agent execution, synthetic-history merging, and tool dispatch. |
| `models/__init__.py`, `models/function.py`, `models/test.py`, continuation and request-timeout helpers | Measured `TestModel` requests, `FunctionModel` search responses, and `CompletedStreamedResponse` replay. |
| `ToolSearch`, `_tool_search`, tool manager, tools, function/combined/prepared toolsets | Measured preparation, keyword ranking, sequential discovery, and parallel result handling. |
| `pydantic_graph` | Measured graph builder execution; every current graph module is also imported. |
| Evals `evaluators/report_common.py` | Measured score extraction, threshold counting, AUC, downsampling, and chart construction. |
| Agent, capability, toolset, native-tool exports; other core root modules | Transitive public re-exports import all current agent, capability, toolset, and native-tool modules. Retained even where only class definitions execute at collection. |
| Selected models, profiles, durable-execution primitives, embedding/image bases, workspaces | Transitive imports, listed as explicit file exceptions in the classifier. `profiles/decision.py` and `providers/__init__.py` are included. |
| Evals dataset, lifecycle, task-run, evaluator, reporting, and OTel modules | Transitive imports and report-context construction. |
| `tests/__init__.py`, root `conftest.py`, `_inline_snapshot.py`, `cost_guards.py`, `cassette_hooks.py`, `cassette_utils.py` | Collection and autouse fixtures: event loop, instrumentation/cache resets, HTTP client tracking, VCR checks, SDK cache/GC workarounds, and test-cost hooks. |
| `tests/benchmarks/*` except `README.md` and `AGENTS.md` | Workloads, local fixtures/collection hook, and future benchmark data, including Markdown fixtures. The two audited documentation files are not runtime inputs. |

Provider implementations, provider profiles, live-provider model adapters, realtime/UI adapters, durable-engine integrations, external tools, workspace conformance tests, and the unimported online/generation/judge eval modules skip when modified. `infer_model` receives model instances and returns them directly; its string dispatch and `infer_provider_class` branches do not load provider implementations. The imported wrappers and model utilities remain triggers.

Other test suites and their data skip: the workflow only collects `tests/benchmarks`, no benchmark uses a VCR cassette, and binary-asset fixtures are not requested. Examples, harness, CLAI, and CLAI 2 source skip because they are not imported. Their Python installation/test configuration still triggers conservatively: `uv sync --frozen` resolves the workspace, default extras, and dev/lint groups. Manifests, lockfiles, Python-version files, requirements, and Python build/test configuration retain full runs. Pytest's third-party plugins are supplied by these locked dependencies; the repository's `cost_guards` plugin is included explicitly. JavaScript manifests in the unimported CLI UI do not affect this Python-only install or command. `Makefile` is not called by the benchmark workflow.

New/copied/renamed Python source files in core, graph, or evals run even inside an excluded directory until their dependencies can be audited. Unrecognized paths remain conservative triggers. Shell `case` globs match slashes; explicit imported-file exceptions precede directory exclusions.

### Checking dependency drift

After an isolated benchmark run, the conftest checks loaded repository files against the workflow's actual `affects_benchmarks` function. It fails only when a loaded dependency would skip CodSpeed, or the classifier cannot run. Included new modules need no inventory update. Mixed or unrelated suites skip the guard so their collection imports cannot contaminate this audit. Directory paths, trailing slashes, absolute paths, individual files, and test-node selections are supported.

This executes the extracted function only in the benchmark/test process. The token-bearing classifier job still executes no pull-request files.

Check the dependencies with the workflow's installation and a correctness run:

```sh
uv sync --frozen
uv run pytest tests/benchmarks
```

If the guard reports a path, trace its import through collection, fixtures, or runtime and add its trigger before the matching exclusion in `affects_benchmarks`. Run `uv run pytest .github/scripts/test_ci_content_classifier.py` to check scheduling. The ordinary correctness command uses the smallest workloads; the full CodSpeed commands above also audit lazy imports in larger workloads. Changes to benchmark code, existing dependencies, installation/configuration, or new core/graph/evals source already trigger that full run, so this audit does not need a second benchmark subprocess in the scripts job.
