# Docs Conventions

How the harness's user-facing docs stay correct, consistent, and discoverable. Read this before
touching `README.md`, `docs/`, or a capability README — and before adding a
capability, since every capability lands with docs or fails CI.

## Writing Style

Follow Samuel's [writing-style skill](https://github.com/pydantic/monty/blob/main/.agents/skills/writing-style/SKILL.md)
(in the Monty repo). The short version: the reader is an engineer looking for a fact — state
mechanism, not significance; no throat-clearing, no "powerful"/"seamless"-class adjectives, no
"not just X, but Y" reveals; plain verbs over industry metaphors; lead with the fact; one idea
per sentence.

## The Parity Contract (enforced by `tests/test_docs_parity.py`)

Every capability package must, or CI fails:

1. Have its own `README.md` (purpose-first lead, source-module link, spaced-words H1).
2. Be linked from the top-level `README.md` capability tables.
3. Have a `docs/<slug>.md` page registered in `_CAPABILITY_PAGE_META` (source module + exact H1).
4. Have a partner PR that adds or updates the page in pydantic-ai's
   `docs/navigation.yml`.

## Sidebar Source Of Truth Lives In pydantic-ai (since 2026-08-12)

Since unified-docs #179, the harness sidebar renders from **pydantic-ai's
`docs/navigation.yml`** (the "Pydantic AI Harness" section; entries carry
`source: "harness"`). This repository has no local navigation file. Adding,
renaming, removing, or regrouping a harness docs page therefore requires a partner
PR that updates pydantic-ai's `docs/navigation.yml`. Keep slugs
`harness/<page>` and preserve existing `aliases:` when renaming.

## Page Conventions

- **Purpose-first lead**: the opening paragraph says what the capability is for and when to use
  it. Lifecycle hook names (`before_model_request`, …) never appear in the lead — mechanism goes
  below the purpose.
- **H1 = the capability's spaced name** (`Code Mode`, not `CodeMode`), matching filename and nav
  label. Allowlisted ClassName exceptions live in the parity test.
- **Every page links its own source module** on GitHub.
- **Experimental framing only on ACP.** Every other capability page carries the standard
  version-promise blockquote ("While Pydantic AI Harness is on 0.x releases, the API may change
  between minor releases — …") directly before its first `##` heading, linking the
  [version policy](../docs/index.md#version-policy). Copy it verbatim from an existing page;
  don't improvise variants.
- **Example style**: comments one line max; outputs elided (`#> ...`) rather than long canned
  text; models are the current generation (verify against core's `_known_model_names.py` — as of
  2026-08 that means `claude-fable-5` / `gpt-5.6-sol`, not older defaults) unless a page
  intentionally pins an older one.

## The Capability Index Spans Two Repos

`README.md` and `docs/index.md` present **one capability index across core and Harness** (the
"Ships in" column): core capabilities (Web Search, MCP, Tool Search, Thinking, tool approval, …)
appear here linked to the pydantic-ai docs, and pydantic-ai's docs mirror harness entries in the
other direction.

**This mirroring is a maintenance contract.** If you add, rename, re-categorize, or remove a
capability here, check whether pydantic-ai's docs (its capabilities overview and front pages)
reference it — and open a companion PR there when they do. The same applies in reverse when core
capabilities change. Until an automated docs-review workflow enforces this, the reminder lives
here: a rename PR is not done until both repos agree.

## Categories

Capabilities are grouped by what they give the agent, not by implementation detail:
**Harnesses** (complete combined capabilities like `Coder`) · **Execution environments** (the
workspace the agent acts in) · **Tools & native abilities** · **Web & research** ·
**Reasoning, planning & delegation** · **Context management** (how the agent spends the context
window — Code Mode lives here, not under execution: it changes *how* the agent executes, not
*where*) · **Knowledge & memory** · **Control & safety** · **Self-extension** ·
**Execution runtime** (durable execution, persistence, observability plumbing). The same scheme
orders the sidebar in pydantic-ai's `docs/navigation.yml`. A new capability goes in the category
matching its user-facing benefit; if none fits, raise it in the PR rather than inventing an
eleventh silently. Keep every table's "Package" column and one-line description style intact.

## Harness Pages Specifically

A harness page (`Coder`, `Researcher`, …) must state near the top that it is a **regular combined
capability composing other capabilities** and show the blown-out equivalent `capabilities=[...]`
list — the transparency promise is part of the product. Harness classes are named like the agent
they create (`Coder`), not like capabilities; the word "preset" is not used.

The blown-out equivalent — on the docs page and in the harness's `examples/` counterpart — writes
the harness's defaults out literally (instructions, command allowlists) instead of importing the
constants: the reader is meant to see the entire picture and copy-tweak it. It mirrors the
exported agent (`coder_agent`, …) exactly, including its `name=` and identity `instructions=`.

The Coder blown-out block lives on **five surfaces** that must stay identical:
`docs/coder.md`, `docs/index.md`, `README.md`, `pydantic_ai_harness/coder/README.md`, and (as a
parameterized module) `examples/coding_agent.py`. `tests/test_docs_parity.py` compares the four
markdown copies byte-for-byte and checks the written-out allowlist against
`DEFAULT_ALLOWED_COMMANDS`, so a change to any one of them fails CI until all move together —
change the implementation, the pages, and the example in the same PR. Each block carries a
keep-in-sync comment naming the surfaces.

Style inside blown-out blocks: `capabilities=[...]` lists are always written one entry per line
(never collapsed to a single line, including sub-agents'), and every entry in the main agent's
list carries a short trailing comment saying what it contributes.

## No em dashes

Do not use em dashes anywhere in docs, READMEs, code comments, commit messages, or PR text. Replace them with a colon, semicolon, comma, parentheses, or a sentence break, choosing whichever reads best case by case. This includes the version-promise blockquote: its canonical wording uses a semicolon ("may change between minor releases; when it does, ...").
