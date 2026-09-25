# Benchmarks

```sh
uv sync --frozen
uv run pytest tests/benchmarks --codspeed --codspeed-mode=walltime
```

Run these commands from the repository root. For a correctness check without timing, omit the CodSpeed flags.

The [CodSpeed workflow](../../.github/workflows/benchmark.yml) measures wall time on CodSpeed's dedicated ARM64 Graviton runners, with Python 3.14. It runs on pull requests and pushes to `main`. Fixed hardware avoids the CPU differences of GitHub-hosted runners, while isolation reduces timing noise.

You need [CodSpeed macro-runner access for public repositories](https://codspeed.io/docs/integrations/ci/github-actions/macro-runners#public-repositories). After changing the runner or measurement mode, record a fresh `main` baseline before comparing performance. Walltime results are not comparable to the previous CPU-simulation results. The job has a ten-minute timeout to bound runner usage. Superseded PR runs are canceled; `main` baseline runs are kept.

The agent benchmarks use `TestModel` to avoid network latency. Their fixtures warm up a reused agent before measurement. BlockBuster is disabled for these benchmarks because its blocking-call instrumentation changes the workload.

The synthetic-history benchmark supplies 1,000 or 5,000 consecutive assistant-response fragments without provider identity metadata. The agent must merge these into one response. Fixtures construct the history and warm up the history-processing path outside the measured test.

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
