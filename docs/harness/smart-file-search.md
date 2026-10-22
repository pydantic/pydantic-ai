---
title: Smart File Search
description: "Give a Pydantic AI agent code search by behavior: describe it in plain English and get ranked source excerpts, judged by any model, such as a decision model over the System One API or TypeSafe's Jev."
---

# Smart File Search

`SmartFileSearch` gives an agent code search by behavior: the model describes the behavior it is looking for in plain
English ("where do we reject expired sessions?") and gets back ranked source excerpts with file and line
ranges, without first having to guess the symbol names a text search needs. Relevance is decided by a judge
model you choose: any Pydantic AI model works, and a [decision model](../models/decision.md), served over the
[System One API](../models/system-one.md) or as TypeSafe's Jev, is the recommended kind.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/smart_file_search/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

A coding agent dropped into an unfamiliar repository spends its first turns locating code: guessing
identifiers, chaining exploratory `grep` calls, and listing and reading whole files to find the one function
that matters. Every miss costs a round trip and fills the context with code it will throw away. Text search
only finds what you can already name.

## Usage

Pass the judge as `model` (see [Choosing the judge](#choosing-the-judge)). `SmartFileSearch` searches the run's
[workspace](../workspace.md), so attach one next to it:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import SmartFileSearch

agent = Agent(
    '{{ openai:large }}',
    capabilities=[LocalWorkspace('.'), SmartFileSearch(model='system-one:clm-latest')],
)

result = agent.run_sync('Where do we retry failed webhook deliveries?')
print(result.output)
```

The workspace must be able to run commands and have [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`)
on its `PATH`, which lists the files to search. The `coder` extra installs `rg` for a local workspace; a
sandbox image needs it installed. The tool is not offered on a workspace that cannot run commands, and a run
without a workspace fails at its start.

## Tool

| Tool | Purpose |
|---|---|
| `smart_file_search` | Find code by what it does: `query` in plain English, optionally scoped by `directory` and a ripgrep `glob`, returning up to `limit` matches after judging up to `candidates` snippets. |

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

The result is a `SmartFileSearchResult`: the matches, how many passed but were cut by `limit`, coverage figures
(files seen and skipped, snippets built and judged, and whether the shortlist was complete) and warnings, so
an empty result is never mistaken for proof that the code does not exist. Matches in test files are labelled
`kind='test'` rather than demoted.

A failed judgment fails the whole search, and the error is reported to the model as a tool failure, so it
can fall back to regular search instead of acting on half-judged results.

## Large codebases

A search is capped at 200,000 files (counted after `glob`), 50 million lines and 2 GiB of source, enough for
the Linux kernel; past that it asks the model to narrow `directory` or `glob`. Without caching, every search
reads, chunks and indexes its directory from scratch.

Set `cache_index=True` to keep each searched directory's index between searches, for the life of the agent:

```python {test="skip"}
from pydantic_ai_harness import SmartFileSearch

capability = SmartFileSearch(model='system-one:clm-latest', cache_index=True)
```

Later searches of the same directory still read every file, to catch changes by content hash on any
workspace, but only chunk and index the files that changed. Parallel `smart_file_search` calls on one directory
take turns, so its index is built once rather than by each call. On the Pydantic AI repository (4,500 files,
2 million lines, 126,000 snippets), the first search spent about 9 seconds building the index and later ones
under a second refreshing it, against 10 to 14 seconds for every search without the cache. Up to four
directories (each with its `glob`) are kept, least recently searched evicted first, and memory grows with the
source they hold.

## Choosing the judge

The judge is any Pydantic AI model, set with `model`:

- **Unset** (the default): TypeSafe's Jev (`'typesafe:jev-latest'`) when the `typesafe` SDK is installed and
  `TYPESAFE_API_KEY` is set, and the run's own model otherwise.
- **A model or model name**: that model, always. Jev is never forced on you.

A [decision model](../models/decision.md) is the recommended judge: it answers each relevance question with a
probability in a fraction of the time a language model takes to write one, which is what 128 judgments per
search need. There are two ways to use one:

- **Any decision model served over the [System One API](../models/system-one.md)**, such as CLM, Laya, or one
  [Ollama](../models/system-one.md#ollama) runs locally. Set `SYSTEM_ONE_BASE_URL`, and `SYSTEM_ONE_API_KEY` if
  the API needs a key, then pass `model='system-one:<name>'`, or a
  [`SystemOneModel`][pydantic_ai.models.system_one.SystemOneModel]. This needs nothing beyond `pydantic-ai-slim`.
  It is never picked by default, as the model name depends on the server.
- **[Jev](../models/typesafe.md) through TypeSafe's SDK**, which becomes the default once the SDK is installed
  and `TYPESAFE_API_KEY` is set:

  ```bash
  pip/uv-add "pydantic-ai-slim[typesafe]"
  ```

Any language model works as well, since the two ratings are ordinary structured output. As one search makes
up to `candidates` judge requests, prefer a small, fast model to the run's own when you are not using a
decision model. The judge's token usage is not counted in the run's usage or checked against its usage
limits, so bound its cost with `candidates`.

`threshold` (0.5 by default) is tuned for Jev's probabilities. Another decision model's probabilities, and
even more so a language model's ratings, can be calibrated differently, so check a few searches and adjust it
if results are too sparse or too noisy. Pin a model version (for example `'typesafe:jev-1.13.0'`) once you
have tuned `threshold` against it.

## Privacy

`smart_file_search` sends the query, the shortlisted snippets and their paths to the judge model's provider.
It only searches inside the workspace's working directory (symlinks followed), so the model can't send source
from elsewhere on the machine. The default instructions tell the model to honor requests to keep code local by
using local search instead. If no source may leave the machine, judge with a local model, such as a decision
model [Ollama](../models/system-one.md#ollama) serves over the System One API.

## Instructions

`SmartFileSearch` adds discovery guidance to the system prompt: reach for `smart_file_search` before exploratory text
search when locating unfamiliar behavior, use exact search for known symbols and exhaustive references, and
fall back to regular search when results are weak or the tool fails. Pass `guidance` to replace it, or
`guidance=''` to add none.

## Telemetry

`SmartFileSearch` adds no spans of its own. With instrumentation on, each judgment is an agent run named
`smart_file_search` under the tool call's span, so a search's judge cost is grouped under that name.

## Agent spec (YAML/JSON)

`SmartFileSearch` works with Pydantic AI's [agent spec](../agent-spec.md):

```yaml
# agent.yaml
model: {{ openai:large }}
capabilities:
  - SmartFileSearch:
      model: system-one:clm-latest
      threshold: 0.6
```

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import SmartFileSearch

agent = Agent.from_file('agent.yaml', custom_capability_types=[SmartFileSearch])
```

Attach the workspace when running the agent, with `agent.run(..., workspace=...)`.

## API reference

::: pydantic_ai_harness.smart_file_search.SmartFileSearch

::: pydantic_ai_harness.smart_file_search.SmartFileSearchResult

::: pydantic_ai_harness.smart_file_search.SmartFileSearchMatch

::: pydantic_ai_harness.smart_file_search.SmartFileSearchCoverage
