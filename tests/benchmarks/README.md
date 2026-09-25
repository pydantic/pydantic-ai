# Benchmarks

```sh
uv sync --frozen
uv run pytest tests/benchmarks --codspeed --codspeed-mode=walltime
```

Run these commands from the repository root. For a correctness check without timing, omit the CodSpeed flags.

The [CodSpeed workflow](../../.github/workflows/benchmark.yml) uses instrumentation on Linux to compare CPU work across commits, without relying on noisy wall-clock thresholds. It runs on pull requests and pushes to `main`.

The agent benchmark uses `TestModel` to avoid network latency. Its fixture warms up a reused agent before measurement. BlockBuster is disabled for this benchmark because its blocking-call instrumentation changes the workload.

## Stress testing

```sh
for run in $(seq 1 10); do
    uv run pytest tests/benchmarks --codspeed --codspeed-mode=walltime || exit 1
done
```

Run this before and after an optimization. Check that every repetition passes and that the measured change exceeds the variation between runs. Keep timing thresholds out of assertions; CodSpeed tracks performance, while assertions check correctness.
