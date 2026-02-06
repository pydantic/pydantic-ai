---
name: pytest-vcr
description: Record, rewrite, and debug VCR cassettes for HTTP recordings. Use when running tests with --record-mode, verifying cassette playback, or inspecting request/response bodies in YAML cassettes.
allowed-tools: Bash(uv run pytest *), Bash(uv run python .claude/skills/pytest-vcr/parse_cassette.py *), Bash(.claude/skills/pytest-vcr/run-vertex-tests.sh *), Bash(source .env *), Bash(git diff *)
---

# Pytest VCR Workflow

Use this skill when recording or re-recording VCR cassettes for tests, or when debugging cassette contents.

## Prerequisites

- Verify the required API key env var is set (do NOT read `.env` - it contains secrets):
  ```bash
  source .env && printenv | grep -q '^OPENAI_API_KEY=' && echo 'ok' || echo 'missing'
  ```
- Tests must be using VCR for HTTP recording

## Important flags
- `--lf` : Run only the last failed tests
- `--record-mode=new_episodes` : Record new cassettes only
- `--record-mode=rewrite` : Rewrite existing cassettes (deletes and re-records)
- `-vv` : Verbose output
- `--tb=line` : Short traceback output
- `-k=""` : Run tests matching the given substring expression

## Recording Cassettes

### Step 1: Record cassettes

**For NEW cassettes** (tests that don't have recordings yet):
```bash
source .env && uv run pytest path/to/test.py::test_function_name -v --tb=line --record-mode=new_episodes
```

**To REWRITE cassettes** (tests with updated expectations):
```bash
source .env && uv run pytest path/to/test.py::test_function_name -v --tb=line --record-mode=rewrite
```

Multiple tests can be specified:
```bash
# we use `rewrite` here because it works with both new and existing cassettes
# and because we're selecting a specific set of tests there's no danger of overwriting unrelated cassettes
source .env && uv run pytest path/to/test.py::test_one path/to/test.py::test_two -v --tb=line --record-mode=rewrite
```

### Step 2: Verify recordings

Run the same tests WITHOUT `--record-mode` to verify cassettes play back correctly:
```bash
source .env && uv run pytest path/to/test.py::test_function_name -vv --tb=line
```

### Step 3: Review snapshots

If tests use [`snapshot()`](https://github.com/15r10nk/inline-snapshot) assertions:
- The test run in Step 2 auto-fills snapshot content
- Review the generated snapshot files to ensure they match expected output
- You only review - don't manually write snapshot contents
- Snapshots capture what the test actually produced, additional to explicit assertions

## Parsing Cassettes

Parse VCR cassette YAML files to inspect request/response bodies without dealing with raw YAML.

### Usage

```bash
uv run python .claude/skills/pytest-vcr/parse_cassette.py <cassette_path> [--interaction N]
```

### Examples

```bash
# Parse all interactions in a cassette
uv run python .claude/skills/pytest-vcr/parse_cassette.py tests/models/cassettes/test_foo/test_bar.yaml

# Parse only interaction 1 (0-indexed)
uv run python .claude/skills/pytest-vcr/parse_cassette.py tests/models/cassettes/test_foo/test_bar.yaml --interaction 1
```

### Output

For each interaction, shows:
- Request: method, URI, parsed body (truncated base64)
- Response: status code, parsed body (truncated base64)

Base64 strings longer than 100 chars are truncated for readability.


## Vertex AI Tests

Vertex tests use the `skip_unless_vertex` fixture from `tests/conftest.py` — they only run in CI or when `ENABLE_VERTEX=1` is set. `ENABLE_VERTEX=1` is only needed when recording/rewriting cassettes locally; during playback, cassettes replay without live auth. Add `skip_unless_vertex: None` as a parameter to any new vertex test.

Vertex tests require special auth setup. Use the provided script:

```bash
# Record new Vertex cassettes
.claude/skills/pytest-vcr/run-vertex-tests.sh tests/path/to/test.py -v --tb=line --record-mode=new_episodes

# Verify playback
.claude/skills/pytest-vcr/run-vertex-tests.sh tests/path/to/test.py -vv --tb=line
```

The script auto-detects project from gcloud and checks auth. If auth fails:
```bash
gcloud auth application-default login
gcloud config set project <your-project-id>
```

## Full Workflow Example

```bash
# 1. Record new cassette
source .env && uv run pytest tests/models/test_openai.py::test_chat_completion -v --tb=line --record-mode=new_episodes

# 2. Verify playback and fill snapshots
source .env && uv run pytest tests/models/test_openai.py::test_chat_completion -vv --tb=line

# 3. Review test code diffs (excludes cassettes)
git diff tests/ -- ':!**/cassettes/**'

# 4. List new/changed cassettes (name only - use parse_cassette.py to inspect)
git diff --name-only tests/ -- '**/cassettes/**'

# 5. Inspect cassette contents if needed
uv run python .claude/skills/pytest-vcr/parse_cassette.py tests/models/cassettes/test_openai/test_chat_completion.yaml
```
