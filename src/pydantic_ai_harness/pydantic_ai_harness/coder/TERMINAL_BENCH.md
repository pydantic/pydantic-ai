# Run Coder on Terminal-Bench 2.1 with Harbor

This playbook runs the exported `pydantic_ai_harness.coder:coder_agent`
(`Agent(capabilities=[Coder()])`) inside Harbor's disposable task containers.
Harbor installs the agent, supplies the task instruction, runs the verifier,
and collects results. Installing harness on the host does not change the
harness installed inside those containers.

## 1. Check prerequisites and pin the adapter

You need Git, [uv](https://docs.astral.sh/uv/), a running Docker daemon,
network access for installation/model calls, and a provider credential with
sufficient quota. Use an isolated benchmark machine without sensitive host
mounts. Coder can execute arbitrary commands inside its task environment.

The adapter is currently proposed in
[Harbor PR #3208](https://github.com/harbor-framework/harbor/pull/3208), following
[the installed-agent pattern in #3199](https://github.com/harbor-framework/harbor/pull/3199).
Do not assume the agent is available in a released Harbor package. These
commands select the adapter revision inspected for this playbook:

```bash
git clone https://github.com/harbor-framework/harbor.git harbor-harness-bench
cd harbor-harness-bench
git fetch origin refs/pull/3208/head
git checkout --detach fa64ce158719bf193f5649ea4ef44b221256d673
uv sync
uv run harbor run --help
docker info
```

When adopting a newer adapter, record its commit and recheck its options.
The pinned adapter supports `version`, `pip_packages`, `max_tokens`, and
`tool_retries` as agent kwargs. It accepts Harbor's `provider/model` syntax
and translates it to Pydantic AI's `provider:model` syntax.

## 2. Select the actual 2.1 dataset

At the time this playbook was written, Harbor's public
[registry](https://github.com/harbor-framework/harbor/blob/main/registry.json)
listed `terminal-bench@2.0`, not `terminal-bench@2.1`. Do not substitute 2.0
and report it as 2.1. Obtain the intended 2.1 task release from its publisher
and use its local Harbor-format task directory:

```bash
export TB21_DIR='/absolute/path/to/terminal-bench-2.1/tasks'
test -d "$TB21_DIR"
find "$TB21_DIR" -name task.toml | head
```

The directory must contain task directories with `task.toml`, instructions,
environment definitions, and verifier tests. Preserve the publisher's release
identifier and checksum, or the dataset repository URL and commit SHA.
If you do not have a verifiable 2.1 release, stop here rather than guessing a
registry identifier. Once 2.1 is registered, `-d terminal-bench@2.1` can replace
`-p "$TB21_DIR"` in the commands below after checking the registry entry.

## 3. Select credentials, model, and harness version

Load the provider credential into your shell from your normal secret manager
(for example `ANTHROPIC_API_KEY`). Do not put its value in a job file, commit,
or command-line argument. The adapter forwards the selected provider's
credentials to the agent process. Treat collected logs as potentially sensitive.

```bash
: "${ANTHROPIC_API_KEY:?Load this from your secret manager first}"
export BENCH_MODEL='anthropic/claude-opus-4-1'
export HARNESS_VERSION='REPLACE_WITH_PUBLISHED_VERSION'
```

Choose an available model and an exact published harness version before
running. The adapter's `version` option constructs
`pydantic-ai-harness==<version>`; it does not accept a Git SHA or a local
checkout path. For the six-tool Coder, select a release containing
[harness PR #870](https://github.com/pydantic/pydantic-ai-harness/pull/870).
Until that code is released, this release-pinned recipe cannot benchmark it.
Record which Coder generation you used when comparing scores.

Install the matching `[coder]` extra via `pip_packages` so the agent venv also
gets ripgrep. Omit `allowed_commands`: current Coder has no command allowlist,
and the older adapter's override path rebuilds the agent with legacy options.
Do not use that path to configure the current Coder.

## 4. Smoke-test one task, then run the suite

Run from the Harbor checkout. The model calls below incur provider charges.
Start with one task and one concurrent trial so installation, credentials,
and verifier failures are easy to distinguish:

```bash
uv run harbor run \
  -p "$TB21_DIR" \
  -a pydantic-ai-harness \
  -m "$BENCH_MODEL" \
  --ak "version=$HARNESS_VERSION" \
  --ak "pip_packages=[\"pydantic-ai-harness[coder]==$HARNESS_VERSION\"]" \
  --ak max_tokens=32000 \
  --ak tool_retries=5 \
  --n-tasks 1 \
  -n 1 \
  --job-name coder-tb21-smoke
```

Check that installation succeeded, the agent made tool calls, and the verifier
produced a reward. A completed agent process is not necessarily a solved task.
After the smoke test, run the full local release:

```bash
uv run harbor run \
  -p "$TB21_DIR" \
  -a pydantic-ai-harness \
  -m "$BENCH_MODEL" \
  --ak "version=$HARNESS_VERSION" \
  --ak "pip_packages=[\"pydantic-ai-harness[coder]==$HARNESS_VERSION\"]" \
  --ak max_tokens=32000 \
  --ak tool_retries=5 \
  -n 1 \
  --job-name coder-tb21-full
```

Use a new job name for each experiment. Raise concurrency only after checking
CPU/RAM, Docker capacity, provider rate limits, and budget. Keep the task
selection, attempt count, timeout policy, model, and agent settings fixed when
comparing harness versions. For official submissions, follow the benchmark's
current evaluation rules rather than treating this smoke-test configuration
as a leaderboard prescription.

## 5. Collect evidence and diagnose failures

Harbor writes jobs under `jobs/` by default. Keep the job configuration,
aggregate and per-trial results, verifier logs/rewards, and agent logs.
The adapter also writes `pydantic-ai-harness-result.json` in each trial's agent
logs, containing output, usage, and an error when available. Inspect it with:

```bash
find jobs/coder-tb21-smoke -name pydantic-ai-harness-result.json -print
```

- **Unknown agent:** verify you are running Harbor from the adapter checkout.
- **Installation failure:** check the chosen release exists and supports the
  requested extra; inspect the trial's setup logs.
- **Authentication or quota failure:** check the selected provider's host
  credential and account quota, without printing the credential into logs.
- **Agent error or timeout:** inspect the adapter result and trial logs before
  changing retries or token limits. Record any configuration change as a new run.
- **Verifier failure:** distinguish an incorrect solution from infrastructure
  failure; preserve failed trials instead of excluding them from the score.

Archive the Harbor SHA, harness version, resolved container dependencies,
dataset identity, exact model ID, configuration, host architecture, and raw
results alongside any reported score. Report completed, failed, and errored
trial counts, not just the aggregate reward. This playbook adds no telemetry
or runtime behavior; it uses the adapter's existing result and usage reporting.

The commands and adapter interfaces were source-checked; no paid full-suite
run or Terminal-Bench 2.1 score is claimed here.
