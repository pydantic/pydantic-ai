---
name: pytest-vcr
description: Use this skill to record VCR cassettes and parsing them for debugging. Use when tests need HTTP recordings created/updated or when debugging cassette contents.
---

# Pytest VCR Workflow

Use this skill when recording or re-recording VCR cassettes for tests, or when debugging cassette contents.

## Prerequisites

- Ensure `.env` file exists with required API keys
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
source .env && uv run pytest path/to/test.py::test_one path/to/test.py::test_two -v --tb=line --record-mode=new_episodes
```

Do NOT use `-v` flag during recording.

### Step 2: Verify recordings

Run the same tests WITHOUT `--record-mode` to verify cassettes play back correctly:
```bash
source .env && uv run pytest path/to/test.py::test_function_name -vv --tb=line
```

### Step 3: Review snapshots

If tests use `snapshot()` assertions:
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


## Full Workflow Example

```bash
# 1. Record new cassette
source .env && uv run pytest tests/models/test_openai.py::test_chat_completion -v --tb=line --record-mode=new_episodes

# 2. Verify playback and fill snapshots
source .env && uv run pytest tests/models/test_openai.py::test_chat_completion -vv --tb=line

# 3. Review any snapshot changes in the diff
git diff tests/

# 4. Debug cassette contents if needed
uv run python .claude/skills/pytest-vcr/parse_cassette.py tests/models/cassettes/test_openai/test_chat_completion.yaml
```
