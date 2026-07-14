---
name: add-new-model
description: Add support for a newly-released LLM model in pydantic-ai (e.g. openai:gpt-5.6, anthropic:claude-sonnet-5). Use when a provider ships a new model id and you need to wire literals, profile flags, and tests to recognize it. Handles SDK-lag, gateway list conventions, and capability probing.
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
- Every prefix/membership key in the profile module you're editing (e.g. OpenAI's `_REASONING_SUPPORT_BY_PREFIX` keys, Anthropic's inline `model_name.startswith((...))` tuples, xAI's `_GROK_43_REASONING_MODELS`)
- `KnownModelName` and its provider-block neighbours
- Snapshot test files: `tests/models/test_model_names.py`, `tests/test_capabilities.py`

Classify each hit:
- **must update** — model-name lists, dispatch tuples
- **snapshot to refresh** — `inline_snapshot` blocks needing `pytest --inline-snapshot=fix`
- **skip** — VCR cassettes, docs about an unrelated model

If `rg` output looks mangled (unicode/regex artifacts), drop to `grep -n` — don't push past garbled output.

## Step 4 — SDK pin check

Snapshot/enumeration tests in this repo often tie `KnownModelName` to a literal set defined in the provider SDK. **The provider SDK frequently lags the model release by days.**

For OpenAI, check the broad union the repo actually consumes (`OpenAIModelName = str | AllModels`), **not** the chat-only `ChatModel` Literal — `AllModels` also carries Responses-API-only and embeddings ids that the enumeration test walks:

```bash
uv run python -c "from openai.types import AllModels; from typing import get_args; print([m for m in get_args(AllModels) if '<new-version>' in m])"
```

Anthropic and xAI do **not** follow this OpenAI flow — the repo bridges their SDK lag with a local `Literal` and lands green immediately, no split. See the SDK-lag bridge notes in their landmine sections below (Anthropic checks `ModelParam`, not `Model`).

If a provider with no bridge (e.g. OpenAI) doesn't yet list the new id, **the literals PR cannot land green on CI**. Surface this to the user with the choice:

1. **Split the PR** — land the profile/handler change now (capability flip is harmless without `KnownModelName` literals because runtime accepts plain strings). Open a separate draft PR for the literals; promote it once the SDK ships and the pin is bumped.
2. **Hold the whole PR** — wait for SDK release, bump pin, refresh snapshots with `pytest --inline-snapshot=fix`, push.
3. **Bump SDK pin now** — only if the new SDK is already released.

Default recommendation: option 1 (split). Use `AskUserQuestion`.

## Step 5 — Probe capabilities (only if not a pure mirror)

If the new model is just another sibling in an existing family (e.g. `gpt-5.5` after `gpt-5.4`), skip to Step 6 — the existing profile branch covers it once you add the prefix to the dispatch tuple.

If the model is a new family or has unclear capabilities, write a small comparison script (`local-notes/probe_<model>.py`) that hits the new model AND its closest neighbour with:
- `temperature` / `top_p` (does the API reject sampling params?)
- `reasoning.effort` values (`none`, `low`, `medium`, `high`, `xhigh`) — note which the API accepts
- New parameters mentioned in the release notes
- Streaming / tool calls if the family is new

Diff the responses. Anything that diverges from the neighbour belongs in the profile.

## Step 6 — Edit (minimal diff matching the mirrored PR)

Make only the changes the enumeration step surfaced. Resist scope creep. If you discover a pre-existing bug in a sibling model's profile, **flag it in the PR description; do not fix it in this PR.**

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

Terse, not botty. Structure:

- One sentence: what model(s) were added.
- Bulleted file list with one-line "what changed" per file.
- "Verified via probe / mirror of #NNNN" — explicit about which changes were API-verified vs assumed-by-mirror.
- Flag pre-existing latent bugs found but deliberately not fixed.
- Link the prior add-model PR for context.

Include the [PR template](.github/pull_request_template.md), fill in the issue number, and check the "AI generated code" box in the GitHub UI yourself — `gh pr create` cannot set it.

## Provider-specific landmines

### OpenAI

- **`_REASONING_SUPPORT_BY_PREFIX`** in `pydantic_ai_slim/pydantic_ai/profiles/openai.py` — a dict keyed by model-name prefix (`'gpt-5.6'`, `'gpt-5.3-chat'`, `'gpt-5'`, `'o'`, …) → `_ReasoningSupport(enabled_by_default, can_be_disabled, supports_mode)`, resolved **first-match-wins** by `_reasoning_support()`. A new `gpt-5.N` family MUST be added here, and **ordering matters**: a more specific prefix (`'gpt-5.3-chat'`) must precede the broader one it would otherwise shadow (`'gpt-5.3'`), and every newer `gpt-5.x` family must precede the plain `'gpt-5'` catch-all. Miss it and the model falls through to the `_NO_REASONING` default (`thinking_always_enabled=False`, `openai_supports_reasoning_effort_none=False`) — wrong defaults, no error. The resolved matrix is pinned in `tests/profiles/test_openai.py`.
- **`KnownModelName` lives in `pydantic_ai_slim/pydantic_ai/models/_known_model_names.py`** (a `TypeAliasType`), **not** `models/__init__.py`. It has split `openai:` and `gateway/openai:` blocks. Don't assume the gateway block omits `-pro`/`-chat-latest` — for the `gpt-5.x` series it enumerates them (`gateway/openai:gpt-5.2-pro`, `gateway/openai:gpt-5.3-chat-latest`, …). Mirror the exact enumeration of the most recent series across both blocks rather than guessing a convention.
- **Most `gpt-5.x-chat` variants DO reason** (`_ALWAYS_ON_REASONING`: reason at a fixed effort, reject `reasoning_effort='none'` and sampling parameters). The non-reasoning exception is the original `gpt-5-chat`/`gpt-5-chat-latest` (`_NO_REASONING`). Verify each `-chat`/`-chat-latest` variant against the live Responses API; don't copy a sibling's reasoning class blindly.
- **`-pro` variants map to `_ALWAYS_ON_REASONING`** (`gpt-5.2-pro`, `gpt-5.4-pro`, `gpt-5.5-pro`) — they reason and reject `effort='none'`. The three-fact `_ReasoningSupport` model doesn't encode per-effort-*value* rejection, so if a new `-pro` rejects a specific value (e.g. `'low'`), flag it rather than assuming the enum covers it.
- **`tests/models/test_model_names.py::test_known_model_names`** asserts `known_model_names()` equals the set generated from `_PROVIDER_TO_MODEL_NAMES['openai']`, i.e. `OpenAIModelName = str | AllModels` (the broad union, not the chat-only `ChatModel`). A literal missing from `AllModels` fails this test — Step 4's SDK check is mandatory and must query `AllModels`.
- **`tests/test_capabilities.py::test_model_json_schema_with_capabilities`** is a snapshot test enumerating every `KnownModelName`. Refresh with `--inline-snapshot=fix`.

### Anthropic

- **TWO literal lists, not one.** Add the id to BOTH:
  1. `pydantic_ai_slim/pydantic_ai/models/_known_model_names.py` — the `anthropic:` AND `gateway/anthropic:` blocks (the `KnownModelName` alias moved here from `models/__init__.py` in #5803; older add-model PR diffs that edit `__init__.py` are stale on this point).
  2. `AnthropicModelName` in `models/anthropic.py` — see the SDK-lag bridge below.
- **Anthropic names ARE enumeration-tested**, unlike what you might assume from the hand-maintained look of the list. `tests/models/test_model_names.py::test_known_model_names` asserts `known_model_names()` (i.e. `KnownModelName`) equals the set generated from `_PROVIDER_TO_MODEL_NAMES['anthropic']`, which is `AnthropicModelName` = `ModelParam` (from the installed `anthropic` SDK) `| Literal[...bridge...]`. A new id missing from BOTH the SDK's `ModelParam` and the local bridge fails this test with `Extra names: {...}`.
- **SDK-lag bridge (the Step 4 mechanism for Anthropic).** When the installed SDK's `anthropic.types.model_param.ModelParam` doesn't yet list the new id (check: `get_args` it and grep), bridge it with a local `Literal`:
  ```python
  AnthropicModelName = LatestAnthropicModelNames | Literal['claude-fable-5']
  ```
  plus a docstring note to drop the literal once the `anthropic` pin is bumped past the release that adds it. This is the in-repo pattern (PR #5709, commit `f29854015` added the bridge; later commits dropped it + bumped the floor to `>=0.105.0`). The bridge lands green immediately — no need to split the PR for Anthropic. NOTE: `ModelParam` ≠ `anthropic.types.model.Model`; check `ModelParam` (it's the superset the repo actually consumes, and may carry ids `Model` doesn't).
- **Capability flags live as `startswith` prefix tuples in `profiles/anthropic.py`** inside `anthropic_model_profile()` (+ the module-level `_ANTHROPIC_CODE_EXECUTION_20260120_MODEL_PREFIXES`). A new family is NOT a literal-only add (that's only right when it truly inherits the default profile, like `claude-mythos-preview`). Probe and set each independently: `models_that_support_json_schema_output`, `supports_adaptive`, `supports_effort`, `supports_xhigh_effort`, `disallows_budget_thinking`, `disallows_sampling_settings`, `supports_task_budgets`, `supports_tool_search`, code-exec version, `anthropic_supports_fast_speed`. Default-`False` flags (e.g. fast speed) are subtractive — just omit the id from that tuple.
- **Forced `tool_choice` is a real per-model divergence worth probing.** Most Anthropic models accept `tool_choice` `{'type':'any'}`/`{'type':'tool'}` and only reject forcing alongside *thinking*; some (Claude Fable 5) reject it **unconditionally** (400 `tool_choice forces tool use is not compatible with this model`). That's modeled by `AnthropicModelProfile.anthropic_supports_forced_tool_choice` (default `True`) threaded into `_support_tool_forcing` in `models/anthropic.py`. Probe `tool_choice={'type':'any'}` against the new id AND its neighbour to tell a genuine divergence from a thinking-only constraint.
- **Tests:** profile-flag unit tests go in `tests/profiles/test_anthropic.py` (NOT `tests/models/test_anthropic.py`). Forced-tool-choice / `_prepare_tools_and_tool_choice` fallback tests go in `tests/models/test_tool_choice_unit.py`. The capability behaviors keyed on shared flags (sampling drop, budget-thinking reject, xhigh) are already covered by the opus-4-7/4-8 parametrized tests — adding the new id to those lists is redundant once a dedicated profile test asserts the flags.
- **`tests/test_capabilities.py::test_model_json_schema_with_capabilities`** snapshots the whole `KnownModelName` enum. Refresh it by running THAT TEST ALONE with `--inline-snapshot=fix` — a blanket `pytest tests/test_capabilities.py --inline-snapshot=fix` errors out on unrelated `_DeprecatedCallableRequestUsage` usage snapshots elsewhere in the file (`TypeError: can not call CustomCode('_DeprecatedCallableRequestUsage')`), which can abort the fix.
- **`providers/bedrock.py` `bedrock_structured_output_unsupported`**: only relevant if the new id is actually served on Bedrock. A direct-API-only model (not in Bedrock's foundation-model list) doesn't belong there; don't add it speculatively just because the mirrored PR did.

### xAI (Grok)

- **Strict enumeration despite `XaiModelName = str | ChatModel`.** The `str` arm looks permissive but the enumeration test's `get_model_names` recurses into the union and yields nothing for a bare `str` type — so `KnownModelName`'s `xai:` block is strictly enforced against the SDK's `ChatModel` Literal, exactly like OpenAI. `tests/models/test_model_names.py::test_known_model_names` fails with "Extra/Missing names" on any mismatch. Confirm parity: `xai:` + `get_args(ChatModel)` must equal the `xai:` entries in `models/_known_model_names.py`.
- **SDK-lag bridge (Anthropic-style, and it's needed for xAI too).** `xai_sdk`'s `ChatModel` frequently lags a release — as of 1.17.0 it still lacked `grok-4.5`, so **bumping the floor won't help** (check newer wheels first: download from PyPI and grep `xai_sdk/types/model.py` for `ChatModel: TypeAlias = Literal[`). Bridge with a local Literal: `XaiModelName = str | ChatModel | Literal['grok-4.5', 'grok-4.5-latest']`, docstring-note to drop it when the floor is bumped past the release that adds the id. This makes the enumeration test's generated side include the new id, matching the hand-added `_known_model_names.py` literal — lands green immediately. (Historically xAI *bumped the SDK floor* — commits `e3f6e3c54`/`58f394aea` — but that only works when the SDK already ships the id.)
- **A new `grok-4.x` is NOT a pure mirror.** Reasoning-effort support lives in `profiles/grok.py` as membership sets (`_GROK_43_REASONING_MODELS` + a per-family effort frozenset), not startswith prefixes. The `grok-4` prefix auto-grants `grok_supports_builtin_tools=True` but leaves `grok_reasoning_efforts` **empty** (→ `supports_thinking=False`) unless you add the id to a reasoning-models set. Forgetting this silently ships a reasoning model with thinking off. Add a `_GROK_<ver>_REASONING_MODELS` set + effort frozenset and an `elif` branch in `grok_model_profile`.
- **Probe reasoning efforts via the OpenAI-compatible REST endpoint**, comparing against the closest neighbour: `POST https://api.x.ai/v1/chat/completions` with `{"model":..., "reasoning_effort": <val>, "max_tokens":1}`. A rejected value returns 400 `This model does not support 'reasoning_effort' value '<val>'`. **Whether `none` is accepted decides `thinking_always_enabled`** (rejected → always-on). CAVEAT: REST silently accepts `xhigh`/`minimal` even though the gRPC `ReasoningEffort` (in `xai_sdk/types/chat.py`) is `Literal['none','low','medium','high']` — don't over-read REST acceptance; `GrokReasoningEffort` is those four and `_map_reasoning_effort` collapses `xhigh`→`high`, `minimal`→`low`. Grok 4.5 example: accepts `low/medium/high`, rejects `none` → always-on; Grok 4.3 accepts `none` too.
- **Floating aliases** (`grok-latest`, `grok-build-latest`) go in the profile reasoning-models set (so passing them resolves the right behavior) but are **NOT** added as `KnownModelName` literals — mirror the SDK, which lists only stable ids like `grok-4.3`/`grok-4.3-latest`.
- **xai is NOT a gateway provider** (`'xai'` absent from `providers/gateway.py`'s `ModelProvider`) — no `gateway/xai:` entries in `_known_model_names.py`.
- **Snapshot that ratchets:** `tests/test_capabilities.py::test_model_json_schema_with_capabilities` embeds the full `KnownModelName` enum. It's a plain sorted string list — hand-add the new ids in sorted position (deterministic, no need for `--inline-snapshot=fix`). Profile-flag tests go in `tests/providers/test_xai.py` (see `test_xai_model_profile`); the parametrized `tests/test_thinking.py::test_grok_43_profile_thinking_support` asserts the *4.3* effort set specifically — don't add a different-effort model to it.
- **env / probing:** `XAI_API_KEY` lives in the repo-root `.env` (not in every worktree). Run probes with `source .env && <script>` so `$XAI_API_KEY` is exported; put any `curl` referencing it in a script file rather than passing the key inline. Verify enumeration/profile logic with a plain `uv run python` snippet (recurse `get_args(XaiModelName)`, compare to `known_model_names()`; call `grok_model_profile(...)` directly) rather than a full `uv run pytest tests/` run.

### Google, Bedrock, others

Not yet documented here. **When you add the next model for one of these providers, add the landmines you encountered to this section before closing the session** (see Step 9).

## Step 9 — Update this skill

After completing the model-add, before closing the session: if anything came up that isn't already documented in this skill — a new test that ratcheted, a provider-specific dispatch tuple, a misleading SDK behavior, a corrected misconception, an iteration the user had to walk you through — **add it to this SKILL.md**.

Specifically:
- Provider-specific landmines → the matching subsection (or create it).
- Generic process gaps → the relevant numbered step.
- Workflow shape errors → restructure the steps.

This skill exists to compound learnings. A model-add that surfaced new friction and didn't update this file wasted that friction.
