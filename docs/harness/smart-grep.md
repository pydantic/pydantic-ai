---
title: Smart Grep
description: "Give a Pydantic AI agent semantic code search: describe behavior in plain English and get ranked source excerpts, judged by any model, with TypeSafe's Jev as the recommended judge."
---

# Smart Grep

`SmartGrep` gives an agent semantic code search: the model describes the behavior it is looking for in plain
English ("where do we reject expired sessions?") and gets back ranked source excerpts with file and line
ranges, without first having to guess the symbol names a text search needs. Relevance is decided by a judge
model you choose: any Pydantic AI model works, and TypeSafe's Jev decision model is the recommended one.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/smart_grep/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

A coding agent dropped into an unfamiliar repository spends its first turns locating code: guessing
identifiers, chaining exploratory `grep` calls, and listing and reading whole files to find the one function
that matters. Every miss costs a round trip and fills the context with code it will throw away. Text search
only finds what you can already name.

## Usage

```bash
pip/uv-add "pydantic-ai-harness[smart-grep]"
```

`SmartGrep` searches the run's [workspace](../workspace.md), so attach one next to it:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import SmartGrep

agent = Agent(
    'openai:gpt-6-luna',
    capabilities=[LocalWorkspace('.'), SmartGrep(model='anthropic:claude-haiku-4-5')],
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
| `smart_grep` | Find code by what it does: `query` in plain English, optionally scoped by `directory` and a ripgrep `glob`, returning up to `limit` matches after judging up to `candidates` snippets. |

A search runs in four steps:

1. **List** the files under `directory` with `rg --files`, which honors `.gitignore` and skips hidden
   files. Binary, non-UTF-8 and minified files are skipped and counted.
2. **Chunk** each file into snippets along its syntax: functions, methods and classes, with the blocks of a
   long function as snippets of their own. Python is parsed with the standard library; the `smart-grep`
   extra adds [tree-sitter](https://tree-sitter.github.io/) grammars for JavaScript, TypeScript, Go, Rust,
   Java, C, C++, C#, Ruby, PHP, Kotlin, Swift, Scala, Bash and Lua. Other files, and any file without its
   grammar installed, are cut into overlapping 60-line windows.
3. **Shortlist** the `candidates` snippets (128 by default, at most 256) that best match the query with
   local [BM25](https://en.wikipedia.org/wiki/Okapi_BM25), weighting paths and symbol names and expanding
   common programming synonyms.
4. **Judge** each shortlisted snippet with the judge model, which rates how well the snippet acts on what the
   query names and how well it performs the action the query describes. A snippet's relevance is the
   product of the two, and those at or above `threshold` are returned, best first, each with a focused
   excerpt of at most 12 lines.

The result is a `SmartGrepResult`: the matches, how many passed but were cut by `limit`, coverage figures
(files seen and skipped, snippets built and judged, and whether the shortlist was complete) and warnings, so
an empty result is never mistaken for proof that the code does not exist. Matches in test files are labelled
`kind='test'` rather than demoted.

A failed judgment fails the whole search, and the error is reported to the model as a tool failure, so it
can fall back to regular search instead of acting on half-judged results.

## Choosing the judge

The judge is any Pydantic AI model, set with `model`:

- **Unset** (the default): TypeSafe's Jev (`'typesafe:jev-latest'`) when the `typesafe` SDK is installed and
  `TYPESAFE_API_KEY` is set, and the run's own model otherwise.
- **A model or model name**: that model, always. Jev is never forced on you.

[Jev](../models/typesafe.md) is the recommended judge: it is a [decision model](../models/decision.md)
that answers each relevance question with a calibrated probability in a fraction of the time a language
model takes to write one, which is what 128 judgments per search need. To use it, install the SDK and set
the key:

```bash
pip/uv-add "pydantic-ai-harness[smart-grep]" "pydantic-ai-slim[typesafe]"
```

Any language model works as well, since the two ratings are ordinary structured output. As one search makes
up to `candidates` judge requests, prefer a small, fast model to the run's own when you are not using Jev.
The judge's token usage is not counted in the run's usage or checked against its usage limits, so bound
its cost with `candidates`.

`threshold` (0.5 by default) is tuned for Jev's probabilities. A language model's ratings are less
calibrated, so check a few searches and adjust it if results are too sparse or too noisy. Pin a Jev version
(for example `'typesafe:jev-1.13.0'`) once you have tuned `threshold` against it.

## Privacy

`smart_grep` sends the query, the shortlisted snippets and their paths to the judge model's provider. It only searches
inside the workspace's working directory (symlinks followed), so the model can't send source from elsewhere
on the machine. The
default instructions tell the model to honor requests to keep code local by using local search instead; pick a
local judge model if no source may leave the machine.

## Instructions

`SmartGrep` adds discovery guidance to the system prompt: reach for `smart_grep` before exploratory text
search when locating unfamiliar behavior, use exact search for known symbols and exhaustive references, and
fall back to regular search when results are weak or the tool fails. Pass `guidance` to replace it, or
`guidance=''` to add none.

## Telemetry

`SmartGrep` adds no spans of its own. With instrumentation on, each judgment is an agent run named
`smart_grep` under the tool call's span, so a search's judge cost is grouped under that name.

## Agent spec (YAML/JSON)

`SmartGrep` works with Pydantic AI's [agent spec](../agent-spec.md):

```yaml
# agent.yaml
model: openai:gpt-6-luna
capabilities:
  - SmartGrep:
      model: anthropic:claude-haiku-4-5
      threshold: 0.6
```

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import SmartGrep

agent = Agent.from_file('agent.yaml', custom_capability_types=[SmartGrep])
```

Attach the workspace when running the agent, with `agent.run(..., workspace=...)`.

## API reference

::: pydantic_ai_harness.smart_grep.SmartGrep

::: pydantic_ai_harness.smart_grep.SmartGrepResult

::: pydantic_ai_harness.smart_grep.SmartGrepMatch

::: pydantic_ai_harness.smart_grep.SmartGrepCoverage
