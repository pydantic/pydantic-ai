---
name: add-new-model
description: Add support for a newly-released language or image generation model in pydantic-ai (e.g. openai:gpt-5.6, anthropic:claude-sonnet-5, openai:gpt-image-2). Use when a provider ships a new model id and you need to wire literals, profile flags, adapters, and tests to recognize it. Handles SDK-lag, gateway list conventions, capability probing, and direct image-model geometry.
user-invocable: true
allowed-tools: Bash, Read, Edit, Write, Glob, Grep, WebFetch, WebSearch, AskUserQuestion
---

# Add New Model

Wire a newly-released provider model into pydantic-ai. Optimized for the common case (mirror an existing sibling); flags the cases where it's *not* a mirror and needs deeper work.

## Reference docs (read once before scoping)

- `agent_docs/pydantic-ai-slim.md` — the **Ownership** section, plus `pydantic_ai_slim/pydantic_ai/native_tools/AGENTS.md`, for the user-visible surface this model needs to land on.
- `pydantic_ai_slim/pydantic_ai/profiles/AGENTS.md`, `providers/AGENTS.md`, `models/AGENTS.md`, and `pydantic_ai_slim/pydantic_ai/AGENTS.md` (the capability-flag and `Provider.model_profile()` rules), plus the **Design Rules** section of `agent_docs/pydantic-ai-slim.md`. These tell you where capability facts belong (profile vs. provider vs. model class) when the new id has non-mirror behavior.

## Inputs

User invokes with `provider` + `model id` (e.g. `openai gpt-5.6`). If missing, ask via `AskUserQuestion`.

## Curation rule

The model-name `Literal` aliases (`KnownModelName`, each provider's `*ModelName`, and the image, embedding and
advisor aliases) list the models a user would pick. They are not catalogs. Apply this rule to every id you add:

- A **line** is a tier name: GPT `sol`/`terra`/`luna`/`astra`, Claude `opus`/`sonnet`/`haiku`/`fable`, Gemini
  `pro`/`flash`/`flash-lite`.
- A **generation** is the major version, counted per provider.
- Per line, list the newest id of the newest generation and the newest id of the previous generation. Remove
  the older id the new one replaces, in the provider alias and in `KnownModelName`.
- Leave out dated snapshots that have an undated alias, previews that a GA id superseded, floating `-latest`
  aliases (Mistral and TypeSafe excepted: their `-latest` ids are canonical), and non-chat models.
- On Bedrock, list one id per model: the `global.` profile when it exists, else `us.`, else the bare id.
- Keep every id that a model alias in `docs/model_aliases/aliases.json` resolves to, even when the rule above
  would drop it. `tests/test_docs_model_aliases.py` fails otherwise. Changing an alias target is a separate
  decision under `docs/model_aliases/AGENTS.md`.

Profiles match any id, so removing an id from a literal changes autocomplete only. Never remove a profile branch
because its id left a literal.

## Image generation models

Image-only models use a separate public surface from conversational models. If the model is consumed by `ImageGenerator`, update `KnownImageGenerationModelName` in `pydantic_ai_slim/pydantic_ai/images/__init__.py`, the relevant direct provider adapter, and its tests; do not also change conversational `KnownModelName`, profiles, gateway aliases, `ImageGenerationTool`, or `models/<provider>.py` unless that surface is explicitly supported and in scope. Add only the public model IDs the project intends to support, and do not infer or automatically add dated snapshots.

Keep common, provider-agnostic controls in `images/settings.py`, but import provider-specific setting types from the official SDK. Put model-specific size and aspect-ratio validation or mapping in the private `images/_<provider>_geometry.py` helper, and update the public support matrix in `docs/image-generation.md`. Verify geometry against official documentation; if the provider does not publish exact output shapes, probe every documented aspect-ratio and resolution combination for every supported model and record the evidence. Prefer deterministic table tests for the full matrix, adding one representative VCR cassette only when the new model or wire behavior needs integration coverage rather than recording every image combination.

## Step 1 — Verify the model exists at the provider

Never trust marketing names, news articles, or guesses. Hit the provider's model-listing endpoint:

| Provider  | Verification call |
|-----------|-------------------|
| OpenAI    | `curl -s https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"` |
| Anthropic | `curl -s https://api.anthropic.com/v1/models -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"` |
| xAI       | `curl -s https://api.x.ai/v1/models -H "Authorization: Bearer $XAI_API_KEY"` |
| Google    | `curl -s https://generativelanguage.googleapis.com/v1beta/models -H "x-goog-api-key: $GOOGLE_API_KEY"` |
| Groq      | `curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"` |
| Bedrock   | `aws bedrock list-foundation-models --region "$AWS_REGION"` |

Load credentials from the repo-root `.env` with `source .env && <cmd>`. `list-foundation-models` is region-scoped, so query the region your models are actually deployed in (not a hard-coded default). List **every** id the provider exposes for this release — base, dated snapshot, `-pro`, `-mini`, `-nano`, `-codex`, `-chat-latest`. Add only what actually exists; do not extrapolate sibling variants.

If the user-given id is **not** in the listing, stop and confirm with the user before proceeding.

## Step 2 — Mirror the most recent add-model PR for this provider

```bash
git log --all --oneline --grep="<previous-version-pattern>" -20
# e.g. for openai: --grep="gpt-5\.4\|gpt-5\.3"
# e.g. for anthropic: --grep="claude-opus-4\|claude-sonnet-4"
```

Pick the smallest, most recent "add model X" PR for the same provider. Pull its file list with `gh pr view <num> --json files --jq '.files[].path'`. **That file list is the floor of what you'll touch.** It is rarely the ceiling.

## Step 3 — Enumerate (load-bearing step)

For every variable, tuple, and literal you're about to touch, grep its readers across the repo. **This step is what catches the snapshot/enumeration tests that ratchet on every model add.** Skipping it pushes work onto CI and produces broken PRs.

Specifically, for a typical model add, grep for:

- The previous model id literal you're mirroring (e.g. `gpt-5.4`, `claude-opus-4-5`) — `rg '<prev-id>' --glob '!**/*.yaml' --glob '!**/cassettes/**'`
- Search for the previous model's display name in docs and public docstrings; inspect capability rosters for the same features.
- Every prefix/membership key in the profile module you're editing (e.g. OpenAI's `_REASONING_SUPPORT_BY_PREFIX` keys, Anthropic's inline `model_name.startswith((...))` tuples, xAI's `_GROK_43_REASONING_MODELS`)
- `KnownModelName` and its provider-block neighbours
- Snapshot test files: `tests/models/test_model_names.py`, `tests/test_capability_spec.py`
- Public API compatibility: the release comparison reports a model-name `Literal` change as an attribute-value change. The gate passes added and removed names on an alias listed under `model_name_literals` in `.github/api-compatibility-allowlist.json`. Do not add a waiver for that change: the gate fails on an unused waiver. If the gate fails on a model-name alias that is not listed, add its `path` and `symbol` to `model_name_literals`. Any other change to the alias, such as adding a non-`Literal` arm, needs an exact waiver.

Classify each hit:
- **must update** — model-name lists, dispatch tuples
- **snapshot to refresh** — `inline_snapshot` blocks needing `pytest --inline-snapshot=fix`
- **skip** — VCR cassettes, docs about an unrelated model

If `rg` output looks mangled (unicode/regex artifacts), drop to `grep -n` — don't push past garbled output.

## Step 3b — Verify pricing and context data

Pricing additions belong to the automated workflows in `pydantic/genai-prices`.
The [Update prices workflow](https://github.com/pydantic/genai-prices/blob/main/.github/workflows/update-prices.yml)
imports provider data hourly and aggregator data weekly.
[Genie Pricer](https://github.com/pydantic/genai-prices/blob/main/.github/workflows/GENIE_PRICER.md)
handles targeted model additions through the
[Model Price Check workflow](https://github.com/pydantic/genai-prices/blob/main/.github/workflows/agentic-model-price-check.md).
Do not open or implement a manual genai-prices counterpart PR during a model add.
Do not trigger a pricing request unless the user authorizes that request.

Verify the exact API IDs and hosting providers independently of pricing data.
Do not infer availability from an older sibling or a pricing match.
Do not block model-name support on pricing-feed lag.

Cost and `context_window` data come from `pydantic/genai-prices` through `_genai_prices.py`.
Bundled data requires a `genai-prices` release.
Check both the installed package and the live merged feed for the exact model.
Update the Pydantic AI lock when adopting a package release with the entry.
The live merged feed can supply entries through `pydantic_ai.prices.update_in_background()`.
Check the updater for an entry before claiming a package release blocks support.
`Model.profile` only consults genai-prices when nothing else set `context_window`.
Without a price entry in either source, `ModelResponse.cost()` raises `LookupError` and a
`cost_limit` cannot be enforced — the run warns `CostNotFoundWarning` at the end instead.
Without a `context_window` value from any source, `RunContext.context_window_used` is `None`.
Compare matched prices against the provider's official rates; a broad sibling match can silently return stale prices.
Report missing or incorrect data and link an existing automation proposal when available.

## Step 4 — SDK pin check

The OpenAI, Anthropic and xAI aliases are curated `Literal`s, not the SDK's model literal. `tests/models/test_model_names.py`
ties the two together through `SDK_MODEL_LITERALS`:

- `test_sdk_model_literal` snapshots each SDK's full model literal at the version in `uv.lock`. It skips on CI legs
  that install another version (lowest-versions, the upgrade canary).
- `test_curated_model_names_are_sdk_ids_or_bridges` requires every curated id to be in the locked SDK's literal or in
  that entry's `bridges`, and every bridge to be absent from the SDK.

Check whether the locked SDK lists the new id:

```bash
uv run python -c "from openai.types import AllModels; from typing import get_args; print([m for m in get_args(AllModels) if '<new-version>' in m])"
```

Anthropic reads `anthropic.types.ModelParam`, xAI reads `xai_sdk.types.model.ChatModel`, and the image and embedding
entries read the SDK types `SDK_MODEL_LITERALS` names. If the locked SDK lacks the id, add the id to that entry's
`bridges`. Remove the bridge in the PR that bumps the SDK past the release that lists it.

When a `uv.lock` upgrade changes an SDK's model literal, `test_sdk_model_literal` fails. Review the new ids against
the curation rule, update the curated alias and `KnownModelName`, drop any bridge the SDK now lists, then run
`uv run pytest tests/models/test_model_names.py --inline-snapshot=fix`.

If a release lists the id but sits inside the 7-day `exclude-newer` window, bridge it. For Anthropic, bump the SDK
through the quarantine exemption its landmine section describes.

## Step 5 — Probe capabilities (only if not a pure mirror)

If the new model is just another sibling in an existing family (e.g. `gpt-5.5` after `gpt-5.4`), skip to Step 6 — the existing profile branch covers it once you add the prefix to the dispatch tuple.

If the model is a new family or has unclear capabilities, write a small comparison script (`local-notes/probe_<model>.py`) that hits the new model AND its closest neighbour with:
- `temperature` / `top_p` (does the API reject sampling params?)
- `reasoning.effort` values (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) — note which the API accepts. OpenAI's 400 lists the accepted values. Probe `max` on both the Responses and Chat Completions APIs: they accept different sets, recorded in `openai_responses_supports_max_reasoning_effort` (set in `OpenAIProvider.model_profile`) and `openai_chat_supports_max_reasoning_effort` (set in each OpenAI-compatible provider's `model_profile`)
- New parameters mentioned in the release notes
- Streaming / tool calls if the family is new

Diff the responses. Anything that diverges from the neighbour belongs in the profile.

### Gateway parity

Where the gateway serves a model, it must behave the same as the provider's canonical API. Step 3 only
gets the id recognized; this is about behavior, and nothing enumerates it for you.

The gateway reaches the canonical API through an ordinary SDK client carrying a proxy base URL. So:

- **Narrow a capability by client class, never by base URL.** Bedrock, Vertex and Foundry are separate
  transports and earn their own gates. A proxied client is the canonical API, and must keep every
  capability the unproxied one has.
- **A `base_url` test inside a capability decision is the defect, not the fix.** It splits the gateway
  off from the transport it actually reaches. No capability in `models/` or `profiles/` is decided that
  way — if you are about to be the first, you are answering the wrong question.
- **Probe the gateway leg rather than reasoning about it.** `Model('<id>', provider='gateway')`, then
  exercise whatever capability you gated. If `PYDANTIC_AI_GATEWAY_BASE_URL` is set in the environment,
  check it points at the gateway root: a provider-specific proxy path 404s every other provider.
- **Probe Bedrock Gateway profile regions individually.** A successful `gateway/bedrock:` inference-profile ID
  does not establish support for another region prefix or the bare ID. Probe each candidate; exclude only IDs
  Gateway rejects.

A model the gateway genuinely does not serve is the other case entirely: it belongs in
`UNSUPPORTED_GATEWAY_MODEL_NAMES`, on evidence that the gateway rejects the id. Never leave the id
advertised and quietly degraded by a capability carve-out instead.

## Step 6 — Edit (minimal diff matching the mirrored PR)

Make only the changes the enumeration step surfaced. Resist scope creep. If you suspect a pre-existing bug in a sibling model's profile, reproduce it live first. Fold a reproduced bug on the same profile gate into this PR, with a test and one PR-body line. Flag an unreproduced or larger one in the PR description instead.

Docs examples name models through the aliases in `docs/model_aliases/aliases.json`, not model ids. Check the placement rules in `docs/model_aliases/AGENTS.md`. When they place the new model in an alias, repoint that alias in `aliases.json` instead of editing doc pages. Then update the README mirrors that `tests/test_docs_parity.py` and `tests/harness/test_docs_parity.py` flag.

After edits:

```bash
make format && make lint
PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run pyright <changed-python-files>
```

Run the tests directly touching the changed surface — the profile test plus any enumeration tests you updated. CI is the safety net for the long tail; locally you only need to verify the surface area of your change.

If snapshot tests changed: `uv run pytest <file> --inline-snapshot=fix` then verify the diff is the expected literal addition only.

## Step 7 — VCR / integration tests

**Default for mirror-only adds:** skip recording a new VCR. Repo convention uses one representative model per family for VCR (e.g. `gpt-5.2` covers the gpt-5.x reasoning family). The profile unit test added in Step 6 is sufficient.

**When the new model introduces meaningful changes** to `pydantic_ai_slim/pydantic_ai/models/<provider>.py` (new request shape, new response field, new handler branch):

1. Look for an existing parametrized VCR test that covers the changed feature. `rg -l '<feature-name>' tests/models/`. If one exists and it parametrizes over model ids, **tag the new id onto the parametrize list** rather than writing a new test.
2. If no parametrized coverage exists and you need a new VCR test, place it:
   - **Prefer** `tests/models/<provider>/test_<feature>.py` **only if the file already exists** (e.g. `tests/models/anthropic/test_output.py`).
   - Otherwise add it to `tests/models/test_<provider>.py`. **Do not create a new `tests/models/<provider>/` subdirectory** if one doesn't already exist for this provider.
3. Record using the `testing-skill` skill workflow.

## Step 8 — PR

Follow the `pushing-commits-to-the-repo` skill for the title, body, template, and final metadata
check. Keep the model-specific evidence concise:

- One sentence: what model(s) were added.
- "Verified via probe / mirror of #NNNN" — explicit about which changes were API-verified vs assumed-by-mirror.
- Flag pre-existing latent bugs found but deliberately not fixed.
- Link the prior add-model PR for context.

## Provider-specific landmines

### OpenAI

- **`_REASONING_SUPPORT_BY_PREFIX`** in `pydantic_ai_slim/pydantic_ai/profiles/openai.py` — a dict keyed by model-name prefix (`'gpt-5.6'`, `'gpt-5.3-chat'`, `'gpt-5'`, `'o'`, …) → `_ReasoningSupport(enabled_by_default, can_be_disabled, supports_mode, supports_context)`, resolved **first-match-wins** by `_reasoning_support()`. A new `gpt-5.N` family MUST be added here, and **ordering matters**: a more specific prefix (`'gpt-5.3-chat'`) must precede the broader one it would otherwise shadow (`'gpt-5.3'`), and every newer `gpt-5.x` family must precede the plain `'gpt-5'` catch-all. Miss it and the model falls through to the `_NO_REASONING` default (`thinking_always_enabled=False`, `openai_supports_reasoning_effort_none=False`) — wrong defaults, no error. The resolved matrix is pinned in `tests/profiles/test_openai.py`.
- **`KnownModelName` lives in `pydantic_ai_slim/pydantic_ai/models/_known_model_names.py`** (a `TypeAliasType`), **not** `models/__init__.py`. Add the id to its `openai:`, `openai-chat:` and `gateway/openai:` entries, and remove the ids the curation rule retires.
- **Most `gpt-5.x-chat` variants DO reason** (`_ALWAYS_ON_REASONING`: reason at a fixed effort, reject `reasoning_effort='none'` and sampling parameters). The non-reasoning exception is the original `gpt-5-chat`/`gpt-5-chat-latest` (`_NO_REASONING`). Verify each `-chat`/`-chat-latest` variant against the live Responses API; don't copy a sibling's reasoning class blindly.
- **`-pro` variants map to `_ALWAYS_ON_REASONING`** (`gpt-5.2-pro`, `gpt-5.4-pro`, `gpt-5.5-pro`) — they reason and reject `effort='none'`. `_ReasoningSupport` doesn't encode per-effort-*value* rejection, so if a new `-pro` rejects a specific value (e.g. `'low'`), flag it rather than assuming the enum covers it.
- **`tests/models/test_model_names.py::test_known_model_names`** asserts `known_model_names()` equals the set the curated aliases generate, gateway mirror included. Step 4's bridge check reads `AllModels` (the broad union, not the chat-only `ChatModel`).
- **`tests/test_capability_spec.py::test_model_json_schema_with_capabilities`** is a snapshot test enumerating every `KnownModelName`. Refresh with `--inline-snapshot=fix`.
- **A dotted point release matches none of its family's prefixes.** `gpt-6.1-sol` does not start with `'gpt-6-sol'`. Before you add it, it resolves to `_NO_REASONING` and misses every `startswith` gate in `openai_model_profile()`. Add it to `_REASONING_SUPPORT_BY_PREFIX` and to the generation's gate tuple (`_GPT_6_MODEL_PREFIXES`).
- **Probe a point release against every sibling, not only its namesake.** GPT-6.1 Sol rejects `effort='none'` like GPT-6 Astra; GPT-6 Sol accepts it. The accepted-effort list decides `can_be_disabled`. Decide `supports_image_output` by forcing the tool (`tool_choice={'type': 'image_generation'}`), not from the model page's tool list.
- **Chat Completions rejects function tools while reasoning is on** for the GPT-6 family: `Function tools with reasoning_effort are not supported`. A model that rejects `effort='none'` therefore has no Chat Completions tool calling. Document the limit in `docs/models/openai.md`.
- **A gateway 404 `No cost data available for model` means genai-prices has no entry yet.** The Gateway refuses the id until the entry lands, so a user who picks it from autocomplete gets an error. Add the `gateway/` id to `UNSUPPORTED_GATEWAY_MODEL_NAMES` with a comment naming the missing price data, keep the direct-provider id, and follow Step 3b to report the gap. Remove the exclusion once the Gateway serves the id.
- **clai2 keeps a curated Codex menu**: `CODEX_MODELS` in `src/pydantic_clai2/pydantic_clai2/model_catalog.py`. Add the id when Codex offers it. Check `openai/codex`'s `codex_tui__chatwidget__tests__model_selection_popup.snap`, then run `Agent('openai-codex:<id>')` with a local Codex login. Update the Codex model lists in `src/pydantic_clai2/README.md` (two of them), `src/pydantic_clai2/PLUGINS.md` and `docs/harness/clai2.md`: `rg 'openai-codex:gpt-|gpt-5.6-luna' src/pydantic_clai2 docs/harness`.

OpenAI image-model landmines:

- **Probe `quality` with a valid `size`.** The Images API validates `size` before `quality`. An unknown `quality` returns the same `low, medium, high, auto` list for every model, so it tells you nothing. Only a request that passes size validation shows the per-model rejection: `The model 'gpt-image-2' does not support quality 'xhigh'.`
- **Probe geometry with out-of-bounds sizes.** The API rejects a size before it generates anything. When the new model rejects each boundary size with the same message as a known family, add its ids to that family's tuple in `images/_openai_geometry.py`.
- **A model page's "Responses: Not supported" means the top-level `model`.** GPT Image 2.6 pages say it, yet the Responses `image_generation` tool accepts both 2.6 ids. Probe the tool with `tool_choice={'type': 'image_generation'}` before deciding `ImageGenerationModelName`.

### Anthropic

- **TWO literal lists, not one.** Add the id to BOTH:
  1. `pydantic_ai_slim/pydantic_ai/models/_known_model_names.py` — the `anthropic:` AND `gateway/anthropic:` blocks (the `KnownModelName` alias moved here from `models/__init__.py` in #5803; older add-model PR diffs that edit `__init__.py` are stale on this point).
  2. `LatestAnthropicModelNames` in `models/anthropic/__init__.py` — see the SDK-lag bridge below.
- **Anthropic names are enumeration-tested.** `LatestAnthropicModelNames` is the curated `Literal`; `test_known_model_names` ties it to `KnownModelName`.
- **SDK-lag bridge (the Step 4 mechanism for Anthropic).** When the locked SDK's `anthropic.types.ModelParam` doesn't list the new id, add it to the `anthropic` entry's `bridges` in `SDK_MODEL_LITERALS`. `ModelParam` ≠ `anthropic.types.model.Model`; check `ModelParam`.
- **Capability flags live as `startswith` prefix tuples in `profiles/anthropic.py`** inside `anthropic_model_profile()` (+ the module-level `_ANTHROPIC_CODE_EXECUTION_20260120_MODEL_PREFIXES`). A new family is NOT a literal-only add — it almost always needs at least one profile override (a literal-only add is only right when the family truly inherits every default branch, which is rare). Probe and set each independently: `models_that_support_json_schema_output`, `supports_adaptive`, `supports_effort`, `supports_xhigh_effort`, `disallows_budget_thinking`, `disallows_sampling_settings`, `supports_task_budgets`, `supports_tool_search`, code-exec version, `anthropic_supports_fast_speed`. Default-`False` flags (e.g. fast speed) are subtractive — just omit the id from that tuple.
- **A point release inherits every flag of its base id silently.** The tuples are `startswith` prefixes, so `'claude-opus-5'` already matches `claude-opus-5-5` (as `'claude-fable-5'` matches `claude-fable-5-1`): before you touch anything, the new id resolves to the base model's profile. Tests stay green and nothing warns, so the only way to find a divergence is to read the model's migration guide and probe side by side with the base id. Opus 5.5 looked like an Opus 5 mirror and broke default `output_type` runs with a 400 until it opted out of forcing. Where a flag must *not* carry over, carve the id out explicitly (`startswith('claude-opus-5') and not startswith('claude-opus-5-5')`).
- **Read the migration guide's "breaking changes" before probing.** Anthropic's `platform.claude.com/docs/en/models/<id>/migration-guide` and `whats-new-<id>` pages list every divergence from the previous model and name which other models share it (e.g. "the first three also apply on Claude Fable 5.1"). Those map straight onto profile flags, and they tell you what to probe.
- **Bump the SDK through the 7-day quarantine rather than bridging, once the SDK lists the id.** `exclude-newer = "7 days"` in the root `pyproject.toml` keeps a same-day `anthropic` release out of the lock; admit exactly that release with a timestamp cutoff under `[tool.uv.exclude-newer-package]` (one second past its last artifact's PyPI `upload_time`, plus a `TODO` to remove it once the global window covers it — past that date it turns into a ceiling), raise the floor under `tool.hatch.metadata.hooks.uv-dynamic-versioning.optional-dependencies` in `pydantic_ai_slim/pyproject.toml` (move the exact `uv add --optional`-generated requirement there, add a comment above it naming what the code now needs from that version, as the `anthropic` and `xai-sdk` floors there have, and remove its conflicting `[project.optional-dependencies]` table), run `uv lock --upgrade-package anthropic`, and refresh the gh-aw runner's own lock with `uv lock --script .github/scripts/pydantic-ai-runner`. File a tracking issue for removing the cutoff and link it from the `TODO`. Precedents: #7989 (1.3.0), #8637 (1.8.0). The `bridges` entry above is the fallback for when no SDK release lists the id yet.
- **An SDK bump is a real change: pyright `models/anthropic/` and the Anthropic tests against it.** 1.8.0 renamed the citations request TypedDict to `BetaCitationsConfigParamParam` (`BetaCitationsConfigParam` became a response model, and passing it into a request param broke a dict assertion) and widened `BetaInputTransformation` to a union with `thinking_mismatch_allowed`.
- **Opus 5.5 skips thinking on trivial prompts at its default `medium` effort.** A cassette test that needs a thinking block (e.g. `stale_thinking_block_history`) has to raise `anthropic_effort` for it.
- **A point release falls into its base model's price entry.** The base entries' prefix and `contains` clauses (`starts_with: claude-opus-5`) also capture `claude-opus-5-5`. So until the new entry exists, `calc_price` returns the *old* model's price with no error: genai-prices 0.1.7 priced Opus 5.5 at Opus 5's $5/$25 instead of $4/$20. Check `calc_price(..., model_ref='<new-id>')` against the published price. Report an incorrect sibling match to the pricing automation owner per Step 3b. Link genai-prices #671 and #709 for the match-boundary regression pattern.
- **Forced `tool_choice` is a real per-model divergence worth probing.** Most Anthropic models accept `tool_choice` `{'type':'any'}`/`{'type':'tool'}` and only reject forcing alongside *thinking*; Claude Fable 5.1, Claude Mythos 5.1, and Claude Opus 5.5 reject it **unconditionally** (400 `tool_choice forces tool use is not compatible with this model`). That's modeled by `AnthropicModelProfile.anthropic_supports_forced_tool_choice` (default `True`) threaded into `_support_tool_forcing` in `models/anthropic/__init__.py`. Probe `tool_choice={'type':'any'}` against the new id AND its neighbour to tell a genuine divergence from a thinking-only constraint.
- **Probe the stale-thinking-block retry shape on every new binding model.** When the request set no thinking, the retry sends a `thinking` object through `extra_body`. Claude Fable 5.1 and Claude Opus 5.5 accept `{'block_binding': ...}` alone; Claude Sonnet 5.5 rejects it with `thinking.type: Field required`, so the retry fills in `'type': 'adaptive'`. Probe `extra_body={'thinking': {'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}}` against the new id. Also probe any new `thinking.type` the release adds with `block_binding`: Sonnet 5.5's `between_tools` rejects it (`Extra inputs are not permitted`), so the retry skips that type.
- **Probe a mid-conversation `system` entry against the `<system>`-tagged fallback before touching `_INLINE_SYSTEM_PROMPT_MODEL_PREFIXES`.** Ask each rendering to lift a restriction the top-level prompt set; a plain non-conflicting instruction lands on every model and discriminates nothing. Claude Opus 5 obeys the entry 6/6 and the tagged text 0/6. Claude Sonnet 5.5 refuses both 6/6, so Anthropic's published support decides and it stays in: the entry keeps operator authority at no measured cost. Leave an id out only when the tagged fallback does measurably better, as it did on Claude Sonnet 5. `_TOOL_AVAILABILITY_DELTA_MODEL_PREFIXES` is a separate gate: probe a `tool_addition` by reference and check that the model calls the added tool.
- **List Bedrock inference profiles, not only foundation models.** Claude Sonnet 5.5 launched with `global.anthropic.claude-sonnet-5-5` and no `us.` profile. Run `aws bedrock list-inference-profiles` and add only the profiles it returns.
- **Tests:** profile-flag unit tests go in `tests/profiles/test_anthropic.py` (NOT `tests/models/test_anthropic.py`). Forced-tool-choice / `_prepare_tools_and_tool_choice` fallback tests go in `tests/models/test_tool_choice_unit.py`. The capability behaviors keyed on shared flags (sampling drop, budget-thinking reject, xhigh) are already covered by the opus-4-7/4-8 parametrized tests — adding the new id to those lists is redundant once a dedicated profile test asserts the flags.
- **`tests/test_capability_spec.py::test_model_json_schema_with_capabilities`** snapshots the whole `KnownModelName` enum. Refresh it by running THAT TEST ALONE with `--inline-snapshot=fix` — running the whole file can pull in unrelated `snapshot()` blocks and abort the fix.
- **`providers/bedrock.py` `bedrock_structured_output_unsupported`**: only relevant if the new id is actually served on Bedrock. A direct-API-only model (not in Bedrock's foundation-model list) doesn't belong there; don't add it speculatively just because the mirrored PR did.

### xAI (Grok)

- **`XaiModelName` is a curated `Literal`.** `test_known_model_names` ties its ids to the `xai:` entries in `models/_known_model_names.py`.
- **SDK-lag bridge.** `xai_sdk`'s `ChatModel` frequently lags a release, so bumping the floor often won't help (check newer wheels first: download from PyPI and grep `xai_sdk/types/model.py` for `ChatModel: TypeAlias = Literal[`). Add the id to the `xai` entry's `bridges` in `SDK_MODEL_LITERALS`.
- **A new `grok-4.x` is NOT a pure mirror.** Reasoning-effort support lives in `profiles/grok.py` as membership sets (`_GROK_43_REASONING_MODELS` + a per-family effort frozenset), not startswith prefixes. The `grok-4` prefix auto-grants `grok_supports_builtin_tools=True` but leaves `grok_reasoning_efforts` **empty** (→ `supports_thinking=False`) unless you add the id to a reasoning-models set. Forgetting this silently ships a reasoning model with thinking off. Add a `_GROK_<ver>_REASONING_MODELS` set + effort frozenset and an `elif` branch in `grok_model_profile`.
- **Probe reasoning efforts via the OpenAI-compatible REST endpoint**, comparing against the closest neighbour: `POST https://api.x.ai/v1/chat/completions` with `{"model":..., "reasoning_effort": <val>, "max_tokens":1}`. A rejected value returns 400 `This model does not support 'reasoning_effort' value '<val>'`. **Whether `none` is accepted decides `thinking_always_enabled`** (rejected → always-on). CAVEAT: REST silently accepts `xhigh`/`minimal` even though the gRPC `ReasoningEffort` (in `xai_sdk/types/chat.py`) is `Literal['none','low','medium','high']` — don't over-read REST acceptance; `GrokReasoningEffort` is those four and `_map_reasoning_effort` collapses `xhigh`→`high`, `minimal`→`low`. Grok 4.5 example: accepts `low/medium/high`, rejects `none` → always-on; Grok 4.3 accepts `none` too.
- **Floating aliases** (`grok-latest`, `grok-build-latest`) go in the profile reasoning-models set (so passing them resolves the right behavior) but are **NOT** added as `KnownModelName` literals: the curation rule leaves out `-latest` aliases.
- **xai is NOT a gateway provider** (`'xai'` absent from `providers/gateway.py`'s `ModelProvider`) — no `gateway/xai:` entries in `_known_model_names.py`.
- **Snapshot that ratchets:** `tests/test_capability_spec.py::test_model_json_schema_with_capabilities` embeds the full `KnownModelName` enum. It's a plain sorted string list — hand-add the new ids in sorted position (deterministic, no need for `--inline-snapshot=fix`). Profile-flag tests go in `tests/providers/test_xai.py` (see `test_xai_model_profile`); the parametrized `tests/test_thinking.py::test_grok_43_profile_thinking_support` asserts the *4.3* effort set specifically — don't add a different-effort model to it.
- **env / probing:** `XAI_API_KEY` lives in the repo-root `.env` (not in every worktree). Run probes with `source .env && <script>` so `$XAI_API_KEY` is exported; put any `curl` referencing it in a script file rather than passing the key inline. Verify enumeration/profile logic with a plain `uv run python` snippet (recurse `get_args(XaiModelName)`, compare to `known_model_names()`; call `grok_model_profile(...)` directly) rather than a full `uv run pytest tests/` run.

### Bedrock

- **Bedrock Mantle is a separate provider from Bedrock Runtime.** `bedrock:` (the `BedrockProvider`, boto3-only) talks to the Converse API; `bedrock-mantle:` (the `BedrockMantleProvider`, an `openai`-backed `Provider[AsyncOpenAI]` built on `AsyncBedrockOpenAI`) talks to Mantle's OpenAI-compatible API. They have separate model catalogs and separate optional extras (`bedrock` vs `bedrock-mantle`); don't fold Mantle deps into the `bedrock` group.
- **Mantle model families use different endpoints, keyed off the profile.** `BedrockMantleProvider.model_profile` stamps `bedrock_mantle_interface: Literal['chat','responses','openai-responses']` on the profile (GPT-5.4+ → `openai-responses` at `/openai/v1`; GPT-OSS → `responses` at `/v1`; GPT-OSS Safeguard → `chat` at `/v1`). `infer_model` reads that (via the profile, not a separate interface method) to pick `BedrockMantleResponsesModel` vs `BedrockMantleChatModel`, and the Responses model overrides `client` to pick the base URL. Add a family only after verifying its endpoint against the AWS model card + a live request.
- **`bedrock:` stays on Converse; it does NOT auto-route to Mantle.** `bedrock_openai_model_profile` allows Converse only for the GPT ids it lists; any other proprietary GPT id on `bedrock:` raises, pointing users to `bedrock-mantle:`. Add a `bedrock:` GPT id to that allowed set and to `LatestBedrockModelNames` only after a successful Converse call.
- **Response-scoped tool-call IDs are a profile flag, not a Mantle-wide behavior.** `openai_responses_tool_call_ids_are_response_scoped` (on `OpenAIModelProfile`) is enabled only for Mantle GPT-5.6 Responses; `OpenAIResponsesModel` qualifies call IDs with the response ID in both request and streaming ingestion so history stays uniquely keyed (#6536).

### Google (Gemini)

- **TWO places for the id, FOUR `KnownModelName` blocks.** Add to:
  1. `LatestGoogleModelNames` in `models/google.py` (`GoogleModelName = str | LatestGoogleModelNames` — the `str` arm is permissive at typecheck time, but the enumeration test only walks the `Literal` arm).
  2. `models/_known_model_names.py` — **four** blocks: `gateway/google-cloud:`, `gateway/google:`, `google-cloud:`, `google:` (older add-model PRs that only edit three blocks or `models/__init__.py` are stale; KnownModelName moved in #5803).
- **No SDK-lag bridge needed.** `google-genai` does not ship a model-id Literal the enumeration test consumes — the local `LatestGoogleModelNames` Literal *is* the source of truth. Adding the id lands green immediately.
- **Profile is substring-gated, with a per-model level table.** `profiles/google.py::google_model_profile` gates by an older-Gemini negative list (`gemini-1`, `gemini-2.x`, `gemma`): every other id, `gemini-4-*` included, gets the modern branch (thinking level, tool combination, server-side tool invocations, MIME types in tool returns). `'pro' in model_name and 'flash' not in model_name` sets always-on thinking. The exception is `_MODEL_THINKING_LEVELS`, a `startswith` table mapping id prefixes to their documented level sets that already holds both pro previews, the 3.7 and 3.8 flash ids, and `gemini-3.1-flash-lite-image` — so probe every new id rather than assuming the Gemini-3 branch covers it. Probe all four levels with `generateContent` and `thinkingConfig.thinkingLevel` (`MINIMAL`, `LOW`, `MEDIUM`, `HIGH`) on the Gemini API and on Vertex separately. A 400 on both means the id needs an entry in the table carrying exactly the levels it accepts (non-contiguous sets like `minimal, high` are fine — unsupported unified efforts snap to the nearest documented level). A 400 on only one of them means a level set in `GoogleModel.profile` gated on the client's transport (`_is_google_cloud`), as `gemini-3.1-flash-image` has for the Gemini API, so the other API keeps the levels it accepts. Reach Vertex with application-default credentials, `GOOGLE_PROJECT`, and `location='global'`, as `tests/conftest.py::vertex_provider` does. If you can probe only the Gemini API and it 400s, use the transport-gated branch too, so Vertex keeps its current behavior. Cite Google's documented level sets in the code comment: the Gemini API [thinking table](https://ai.google.dev/gemini-api/docs/thinking) (for image models, the [image-generation page](https://ai.google.dev/gemini-api/docs/image-generation) instead) and the Vertex [thinking table](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/thinking). The probe results decide the entry, even where they diverge from those tables. Probe too when the release notes claim any other capability divergence (no thinking, image-only, Pro always-on).
- **API verification:** `curl -s "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200&key=$GOOGLE_API_KEY"` (key is often in the main worktree `.env`, not every linked worktree). Confirm exact ids; do **not** invent dated snapshots or `-preview` suffixes. Specialized / limited-access models (e.g. Flash Cyber via CodeMender) are out of scope unless they appear in that public listing.
- **A Vertex 404 at the default `us-central1` does not mean Vertex lacks the id.** Probe `location='global'` before you conclude the id is Gemini-API-only.
- **Gateway support is opt-out, not opt-in.** The enumeration test generates `gateway/{google,google-cloud}:*` for every `LatestGoogleModelNames` entry **except** those listed in `UNSUPPORTED_GATEWAY_MODEL_NAMES` in `tests/models/test_model_names.py`. Only add to `UNSUPPORTED_GATEWAY_MODEL_NAMES` when the gateway actually rejects the id. `test_unsupported_gateway_model_names_are_curated` fails on an entry whose id left the curated aliases; remove that entry.
- **Snapshots / tests:** hand-add the new ids in sorted position in `tests/test_capability_spec.py::test_model_json_schema_with_capabilities` (plain sorted string list). Mirror-only adds skip new VCR by default; #5527 recorded one for `gemini-3.5-flash` but that is not required for a pure name add.
- **Docs:** doc pages name models through the tiered aliases in `docs/model_aliases/aliases.json`. Leave pages that pin a model on purpose (`docs/models/google.md`, `docs/capabilities/thinking.md`) alone.

Google image-model landmines:

- Direct image generation has a separate public literal, `KnownImageGenerationModelName` in `pydantic_ai_slim/pydantic_ai/images/__init__.py`. When the task is scoped to `ImageGenerator`, update and test this literal independently; do not automatically widen the change to conversational `KnownModelName`, gateway aliases, profiles, and capability snapshots unless those surfaces are explicitly in scope.
- `Client().models.list()` returns a lazy pager. Keep the client in a named variable until iteration finishes; constructing it inline can let it be closed before the pager sends its request. The endpoint can still list deprecated preview image IDs, so cross-check the official deprecation page and add only current IDs.
- Probe image settings on the exact model and API surface. For `gemini-3.1-flash-image`, the minimum `generateContent` value is `ImageConfigDict(image_size='512')`; the superficially similar literal `'0.5K'` is invalid and returns HTTP 400. `gemini-3.1-flash-lite-image` supports only 1K output. Do not transfer value spellings between model families or API examples without a live check.

### MoonshotAI (Kimi)

- `MoonshotAIModelName` in `providers/moonshotai.py` is a local literal, independent of the OpenAI SDK's model enum. Update it and `models/_known_model_names.py` together, then refresh `tests/test_capability_spec.py::test_model_json_schema_with_capabilities`.
- The `kimi-k3` prefix already covers `kimi-k3.1-pro` and `kimi-k3.1-pro-highspeed` in both family and provider profiles. Probe before adding another prefix or changing profile flags.
- Verify the API at `https://api.moonshot.ai/v1`. The separate Kimi Code subscription uses different model IDs. Read the [canonical API guide](https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart) and compare live behavior with the nearest sibling; do not transfer subscription semantics.

### Others

Not yet documented here. **When you add the next model for one of these providers, add the landmines you encountered to this section before closing the session** (see Step 9).

## Step 9 — Update this skill

After completing the model-add, before closing the session: if anything came up that isn't already documented in this skill — a new test that ratcheted, a provider-specific dispatch tuple, a misleading SDK behavior, a corrected misconception, an iteration the user had to walk you through — **add it to this SKILL.md**.

Specifically:
- Provider-specific landmines → the matching subsection (or create it).
- Generic process gaps → the relevant numbered step.
- Workflow shape errors → restructure the steps.

This skill exists to compound learnings. A model-add that surfaced new friction and didn't update this file wasted that friction.
