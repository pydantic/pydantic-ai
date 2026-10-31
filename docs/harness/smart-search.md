---
title: Smart Search
description: "Give a Pydantic AI agent code search by behavior: describe it in plain English and get ranked source excerpts, judged by any model, such as a decision model over the System One API or TypeSafe's Jev."
---

# Smart Search

`SmartSearch` gives an agent code search by behavior: the model describes the behavior it is looking for in plain
English ("where do we reject expired sessions?") and gets back ranked source excerpts with file and line
ranges, without first having to guess the symbol names a text search needs. Relevance is decided by a judge
model you choose: any Pydantic AI model works, and a [decision model](../models/decision.md), served over the
[System One API](../models/system-one.md) or as TypeSafe's Jev, is the recommended kind.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/smart_search/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

![The model calls smart_search with a plain-English query. SmartSearch lists the files in the workspace with rg --files, chunks them into snippets, shortlists up to 128 snippets with BM25, and sends each one to the judge model you chose. Snippets whose relevance reaches the threshold come back to the model as a SmartSearchResult with file, line range and a short excerpt, plus coverage figures and warnings.](../img/harness/smart-search.svg)

## The problem

A coding agent dropped into an unfamiliar repository spends its first turns locating code: guessing
identifiers, chaining exploratory `grep` calls, and listing and reading whole files to find the one function
that matters. Every miss costs a round trip and fills the context with code it will throw away. Text search
only finds what you can already name.

## Usage

Pass the judge as `model`, which is required (see [Choosing the judge](#choosing-the-judge)). `SmartSearch` searches the run's
[workspace](../workspace.md), so attach one next to it:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import SmartSearch

agent = Agent(
    '{{ openai:large }}',
    capabilities=[LocalWorkspace('.'), SmartSearch(model='system-one:clm-latest')],
)

result = agent.run_sync('Where do we retry failed webhook deliveries?')
print(result.output)
```

The workspace must be able to run commands and have [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`)
on its `PATH`, which lists the files to search. The `coder` extra installs `rg` for a local workspace; a
sandbox image needs it installed. The tool is not offered on a workspace that cannot run commands, and a run
without a workspace fails at its start.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `model` | required | The judge: any model or model name. A judge given by name times out each request after 30 seconds; a model instance keeps its own `settings`. |
| `threshold` | `0.5` | Relevance (the product of the two judgments, 0 to 1) a snippet needs to be returned. Tuned for Jev; see [Choosing the judge](#choosing-the-judge). |
| `concurrency` | `8` | Snippets judged at once. |
| `cache_index` | `False` | Keep each searched directory's index in memory between searches; see [Large codebases](#large-codebases). |
| `guidance` | `None` | Discovery instructions added whenever the tool is offered: `None` for the default, which has the model reach for `smart_search` before exploratory text search, `''` for none. |

## Tool

| Tool | Purpose |
|---|---|
| `smart_search` | Find code by what it does: `query` in plain English, optionally scoped by `directory` and a ripgrep `glob`, returning up to `limit` matches after judging up to `candidates` snippets. |

A search runs in four steps:

1. **List** the files under `directory` with `rg --files`, which honors `.gitignore` and skips hidden
   files. Binary, non-UTF-8 and minified files are skipped and counted.
2. **Chunk** each file into snippets: overlapping 60-line windows of plain text, except for Python, which is
   parsed with the standard library so that functions, methods and classes become snippets, with the blocks of
   a long function as snippets of their own.
3. **Shortlist** the `candidates` snippets (128 by default, at most 256) that best match the query with
   local [BM25](https://en.wikipedia.org/wiki/Okapi_BM25), weighting paths and symbol names and expanding
   common programming synonyms.
4. **Judge** each shortlisted snippet with the judge model, which rates how well the snippet acts on what the
   query names and how well it performs the action the query describes. A snippet's relevance is the
   product of the two, and those at or above `threshold` are returned, best first, each with a focused
   excerpt of at most 12 lines.

The result is a `SmartSearchResult`: the matches, how many passed but were cut by `limit`, coverage figures
(files seen and skipped, snippets built and judged, and whether the shortlist was complete) and warnings, so
the model can tell an empty result from a complete search that found nothing. Matches in test files are labeled
`kind='test'` rather than demoted.

A failed judgment fails the whole search, and the error is reported to the model as a tool failure, so it
can fall back to regular search instead of acting on half-judged results.

## Large codebases

A search is capped at 200,000 files (counted after `glob`), 50 million lines and 2 GiB of source, enough for
the Linux kernel; past that it asks the model to narrow `directory` or `glob`. Without caching, every search
reads, chunks and indexes its directory from scratch.

Set `cache_index=True` to keep each searched directory's index between searches, for the life of the agent:

```python
from pydantic_ai_harness import SmartSearch

capability = SmartSearch(model='system-one:clm-latest', cache_index=True)
```

Later searches of the same directory still read every file, to catch changes by content hash on any
workspace, but only chunk and index the files that changed. Parallel `smart_search` calls on one directory
take turns, so its index is built once rather than by each call. On the Pydantic AI repository (4,500 files,
2 million lines, 126,000 snippets), the first search spent about 9 seconds building the index and later ones
under a second refreshing it, against 10 to 14 seconds for every search without the cache. Up to four
directories (each with its `glob`) are kept, least recently searched evicted first, and memory grows with the
source they hold.

## Choosing the judge

The judge is required, and is any Pydantic AI model or model name, passed as `model`. It decides which provider
receives the searched source, so it is never picked for you.

A [decision model](../models/decision.md) is the recommended judge: it answers each relevance question with a
probability in a fraction of the time a language model takes to write one, which is what 128 judgments per
search need. There are two ways to use one:

- **Any decision model served over the [System One API](../models/system-one.md)**, such as CLM, Laya, or one
  [Ollama](../models/system-one.md#ollama) runs locally. Set `SYSTEM_ONE_BASE_URL`, and `SYSTEM_ONE_API_KEY` if
  the API needs a key, then pass `model='system-one:<name>'`, or a
  [`SystemOneModel`][pydantic_ai.models.system_one.SystemOneModel]. This needs nothing beyond `pydantic-ai-slim`.
- **[Jev](../models/typesafe.md) through TypeSafe's SDK**, as `model='typesafe:jev-latest'`, with
  `TYPESAFE_API_KEY` set.

To install TypeSafe's SDK:

```bash
pip/uv-add "pydantic-ai-slim[typesafe]"
```

A judge given by name times out each request after 30 seconds; a model instance keeps its own `settings`, so set
`settings={'timeout': ...}` on it for a slower judge, such as a local one.

Any language model works as well, since the two ratings are ordinary structured output. As one search makes
up to `candidates` judge requests, prefer a small, fast model to the run's own when you are not using a
decision model.

`threshold` (0.5 by default) is tuned for Jev's probabilities. Another decision model's probabilities, and
even more so a language model's ratings, can be calibrated differently, so check a few searches and adjust it
if results are too sparse or too noisy. Pin a model version (for example `'typesafe:jev-1.13.0'`) once you
have tuned `threshold` against it.

## Usage and limits

Each judged snippet counts as one request in the run's [usage](../agent.md#usage-limits), and the judge's tokens
are added to it, so `result.usage` and the run's [`UsageLimits`][pydantic_ai.usage.UsageLimits] cover the search. So that one search cannot use up
the run, it judges at most half of the requests the run's `request_limit` has left, after one kept for the model's
next turn. The shortlist's best-ranked snippets are judged first, and a warning says how many were left unjudged.
Under the default `request_limit` of 50, a run's first search judges 24 snippets. Raise the limit for full
128-snippet searches, for example with `agent.run(..., usage_limits=UsageLimits(request_limit=500))`. When no
request is left to spend, the search fails and the model is told to use regular search.

## Telemetry

Each search opens a `smart search` span under its tool call, on the run's tracer. The span has these
attributes:

- `smart_search.judge_model`: the judge model.
- `smart_search.files`, `.skipped_files`, `.snippets` and `.evaluated`: the coverage figures.
- `smart_search.indexed_files`: how many files the search chunked and indexed. With `cache_index`, only
  new and changed files are, so this shows how much the cache saved.
- `smart_search.selection_complete`: whether the coverage was complete.
- `smart_search.matches` and `.omitted_matches`: how many matches were returned and cut.
- `smart_search.query`: the query, only when the run includes content in traces.

A failed search records no exception on this span; the enclosing tool span reports it, redacted by the run's
`include_content`. The judge runs themselves are not instrumented. Their prompts carry source snippets, so global
instrumentation must not export them past the host agent's own `instrument` or `include_content` settings. Their
usage is still added to the run's.

## Agent spec (YAML/JSON)

`SmartSearch` works with Pydantic AI's [agent spec](../agent-spec.md):

```yaml
# agent.yaml
model: {{ openai:large }}
capabilities:
  - SmartSearch:
      model: system-one:clm-latest
      threshold: 0.6
```

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import SmartSearch

agent = Agent.from_file('agent.yaml', custom_capability_types=[SmartSearch])
```

Attach the workspace when running the agent, with `agent.run(..., workspace=...)`.

## API reference

::: pydantic_ai_harness.smart_search.SmartSearch

::: pydantic_ai_harness.smart_search.SmartSearchResult

::: pydantic_ai_harness.smart_search.SmartSearchMatch

::: pydantic_ai_harness.smart_search.SmartSearchCoverage
