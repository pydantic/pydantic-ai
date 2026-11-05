# Docs Conventions

How the harness's user-facing docs stay correct, consistent, and discoverable. Read this before
touching `src/pydantic_ai_harness/README.md`, `docs/harness/`, or a capability README — and before adding a
capability, since every capability lands with docs or fails CI.

## Writing Style

Follow Samuel's [writing-style skill](https://github.com/pydantic/monty/blob/main/.agents/skills/writing-style/SKILL.md)
(in the Monty repo). The short version: the reader is an engineer looking for a fact — state
mechanism, not significance; no throat-clearing, no "powerful"/"seamless"-class adjectives, no
"not just X, but Y" reveals; plain verbs over industry metaphors; lead with the fact; one idea
per sentence.

## The Parity Contract (enforced by `tests/harness/test_docs_parity.py`)

Every capability package must, or CI fails:

1. Have its own `README.md` (purpose-first lead, source-module link, spaced-words H1).
2. Be linked from the `src/pydantic_ai_harness/README.md` capability tables.
3. Have a `docs/harness/<slug>.md` page registered in `_CAPABILITY_PAGE_META` (source module + exact H1).

Also add the page to `docs/navigation.yml`, in its category subsection under "Capabilities". A
durable-execution backend goes in the "Durable Execution" section instead (like
`harness/aws-lambda.md`). An interface goes in the "Interfaces" section instead (like
`harness/acp.md`). `tests/test_docs_navigation.py` fails for a page that has no entry and is not
listed in that test's `UNPUBLISHED_PAGES`.

## Semantic Parity

Update both the capability README and `docs/harness/<capability>.md` for every user-facing change:
public classes, parameters, defaults, tool names, extras, and safety semantics.
ACP's README stays under `pydantic_ai_harness/experimental/acp/`; its docs page remains flat under `docs/harness/`.
The two docs must agree with each other and the source. Document composition constraints and safety implications.
Treat README, docs page, or example drift as a defect to fix before merge.

Keep the packaged `pydantic-ai-harness` agent skill in sync too:
`src/pydantic_ai_harness/pydantic_ai_harness/.agents/skills/pydantic-ai-harness/`.
Add new capabilities to its routing table and matching `references/` file.
Update changed parameters, defaults, tool names, and extras there.

Every snippet in both docs must run with the declared extras and all imports present.
Check class and parameter names against the source; preserve the source's model choices.
Use canonical module paths, without `experimental` for graduated capabilities.
Follow the rendering-specific model-name rules under "Page Conventions" below.

Use relative `.md` links for harness and Pydantic AI docs links on docs pages (`../toolsets.md` from `docs/harness/`).
Use reference-style links for API elements (`[RunContext][pydantic_ai.tools.RunContext]`).
CI's offline link check verifies target files and anchors.

`tests/harness/test_docs_parity.py` enforces mechanical checks, not semantic agreement or runnable snippets.
The [docs-parity-reviewer skill](../../../.agents/skills/docs-parity-reviewer/SKILL.md) covers those checks.

## Sidebar Source Of Truth

The harness sidebar renders from **`docs/navigation.yml`**: mostly the category
subsections under "Capabilities", shared with core capability pages, plus a few
pages under "Durable Execution" and "Interfaces". Harness entries carry
`path: "harness/<page>.md"`. Adding, renaming, removing, or regrouping a harness
docs page updates `docs/navigation.yml` in the same PR. Keep each harness page's
slug as `harness/<page>`. When renaming a page, keep its existing `aliases:`.

## Page Conventions

- **Installation commands**: on docs pages, put `pip/uv-add <packages>` on its own
  line in a `bash` fence. The [unified-docs preprocessor](https://github.com/pydantic/unified-docs/blob/main/src/integrations/docs-sync.ts)
  expands this shorthand into pip / uv tabs. Use a separate `bash` fence with
  `py-cli <command>` for follow-up executables; it generates the bare command and
  its `uv run` equivalent. Do not hand-write installation tabs.
  In READMEs, use executable commands under plain `uv:` / `pip:` labels, uv first,
  because GitHub does not run the preprocessor. Keep package arguments identical,
  including extras. Put inline dependency-install examples in fenced blocks too.
  Use `uv run` for README follow-up executables in the uv project environment.
  Workflow YAML and repository-development commands retain
  their required execution environment rather than becoming `uv add` examples.
- **Purpose-first lead**: the opening paragraph of each docs page and README says what the capability is for and when to use
  it. Lifecycle hook names (`before_model_request`, …) never appear in the lead — mechanism goes
  below the purpose. Mention class names only after stating the purpose.
- **Capability name**: the docs filename, page H1, README H1, and nav label match the capability's descriptive name.
  H1s use spaced words (`Code Mode`, not `CodeMode`). Allowlisted ClassName exceptions live in the parity test.
- **Every page links its own source module** on GitHub. A page exposing a public class may end
  with `## API reference` and `::: pydantic_ai_harness.<Class>` autodoc blocks.
- **Experimental framing only on ACP.** ACP keeps its experimental warning and possible-removal notice.
  Every other capability page carries the standard version-promise blockquote
  ("While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, ...")
  directly before its first `##` heading, linking the
  [version policy](../../../docs/harness/index.md#version-policy). Copy it verbatim from an existing page;
  don't improvise variants.
- **Example style**: comments one line max; outputs elided (`#> ...`) rather than long canned
  text. On `docs/harness/` pages, name models with the aliases from
  `docs/model_aliases/aliases.json` (`{{ anthropic:large }}`), never model IDs; see the "Model names"
  section of `docs/AGENTS.md`.
  READMEs, which GitHub renders without resolving aliases, keep concrete model IDs.

## The Capability Index Spans Core And Harness

`src/pydantic_ai_harness/README.md` and `docs/harness/index.md` present **one capability index across core and
Harness** (the "Package" column): core capabilities (Web Search, MCP, Tool Search, Thinking, tool
approval, …) appear there linked to the Pydantic AI docs, and the Pydantic AI docs mirror harness
entries in the other direction.

**This mirroring is a maintenance contract.** If you add, rename, re-categorize, or remove a
harness capability, check whether the Pydantic AI docs (the capabilities overview and the front
pages) reference it, and update them in the same PR when they do. The same applies in reverse
when core capabilities change. Until an automated docs-review workflow enforces this, the
reminder lives here: a rename PR is not done until both indexes agree.

## Categories

Capabilities are grouped by what they give the agent, not by implementation detail:
**Harnesses** (complete combined capabilities like `Coder`) · **Execution environments** (the
workspace the agent acts in) · **Tools & native abilities** · **Integrations** (one third-party
service each, like `GitHub` or `Slack`; also list a new one on `docs/harness/integrations.md`) ·
**Web & research** ·
**Reasoning, planning & delegation** · **Context management** (how the agent spends the context
window — Code Mode lives here, not under execution: it changes *how* the agent executes, not
*where*) · **Knowledge & memory** · **Control & safety** · **Self-extension** ·
**Execution runtime** (durable execution, persistence, observability plumbing). The same scheme
orders the subsections under "Capabilities" in `docs/navigation.yml`. That file adds a core-only
"Loop Customization" subsection after them. A new capability goes in the category matching its
user-facing benefit. If no category fits, say so in the PR instead of silently adding a new
category. Keep every table's "Package" column and one-line description style intact.

## Harness Pages Specifically

A harness page (`Coder`, `Researcher`, …) must state near the top that it is a **regular combined
capability composing other capabilities** and show the blown-out equivalent `capabilities=[...]`
list — the transparency promise is part of the product. Harness classes are named like the agent
they create (`Coder`), not like capabilities; the word "preset" is not used.

Update the implementation, docs page, and example together in the same PR when composition or defaults change.

The blown-out equivalent — on the docs page and in the harness's `src/pydantic_ai_harness/examples/` counterpart — writes
the harness's defaults out literally (instructions, command allowlists) instead of importing the
constants: the reader is meant to see the entire picture and copy-tweak it. It mirrors the
exported agent (`coder_agent`, …) exactly, including its `name=` and identity `instructions=`.

The Coder blown-out block lives on **five surfaces** that must stay identical:
`docs/harness/coder.md`, `docs/harness/index.md`, `src/pydantic_ai_harness/README.md`,
`src/pydantic_ai_harness/pydantic_ai_harness/coder/README.md`, and (as a parameterized module)
`src/pydantic_ai_harness/examples/coding_agent.py`. `tests/harness/test_docs_parity.py` compares the four
markdown copies byte-for-byte and checks the written-out allowlist against
`DEFAULT_ALLOWED_COMMANDS`, so a change to any one of them fails CI until all move together —
change the implementation, the pages, and the example in the same PR. Each block carries a
keep-in-sync comment naming the surfaces.

Style inside blown-out blocks: `capabilities=[...]` lists are always written one entry per line
(never collapsed to a single line, including sub-agents'), and every entry in the main agent's
list carries a short trailing comment saying what it contributes.

## No em dashes

Do not use em dashes anywhere in docs, READMEs, code comments, commit messages, or PR text. Replace them with a colon, semicolon, comma, parentheses, or a sentence break, choosing whichever reads best case by case. This includes the version-promise blockquote: its canonical wording uses a semicolon ("may change between minor releases; when it does, ...").
