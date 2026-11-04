# Model aliases

[`aliases.json`](aliases.json) maps each model alias that doc pages use to a concrete model. This file
covers choosing those models. How pages write aliases is in the "Model names" section of
[`docs/AGENTS.md`](../AGENTS.md).

## Chat model tiers

- Place a model in a tier by its price per token, on the scale of Anthropic's lineup: `small` is
  Haiku-priced, `medium` Sonnet-priced, `large` Opus-priced, and `xlarge` Fable-priced. Anthropic is
  the scale because it offers a model at all four prices.
- Compare output prices per token, by ratio: a model belongs to the tier whose Anthropic output price
  is the smallest multiple away. One price keeps placement unambiguous. Keep each provider's tiers in
  price order: a cheaper model never sits above a pricier one.
- Tier only general-purpose chat models. Coding, research, multi-agent, and other specialized models
  stay out of the tiers.
- Never rank a provider's models only against each other. Each provider's top model would then share
  one tier name across providers at very different prices.
- Read prices from the `genai-prices` package (`genai_prices.data`), which Pydantic AI already
  depends on.
- Give a tier the provider's newest model at that price.
- A newer model that matches an older one's capability at the same or lower price replaces it. Drop
  the older model from the lineup, then fill any tier it leaves with the gap rule below. Never keep
  both in neighboring tiers.
- Fill a tier that has no model at its price with the nearest model above it. At the top of the
  lineup, use the nearest model below. Every chat provider defines all four tiers, so a page can
  switch providers without checking which tiers exist.
- Mirror the `anthropic:` tiers in the `bedrock:` tiers, with the same Claude models.

## Realtime, image, and embedding models

- Point `<provider>:realtime`, `<provider>:image`, and `<provider>:embedding` at the model the
  provider recommends for general use, not at its largest one. Readers copy examples as written.

## Known models

- Point aliases at models in Pydantic AI's known model names. `tests/test_docs_model_aliases.py`
  checks each alias against them.
- A model that its provider serves but Pydantic AI does not list yet goes in that test's
  `_NOT_YET_KNOWN`. Remove it from `_NOT_YET_KNOWN` when a model PR adds it. The test fails until you do.

## Current placements

Update this table whenever a change to `aliases.json` fills, opens, or moves a gap.

| Alias | Model | Reason |
|---|---|---|
| `openai:large`, `openai:xlarge` | `gpt-6-astra` | Astra is Fable-priced. `gpt-6.1-sol` replaced the Opus-priced GPT-5.4 to 5.6 models at a lower price, so `large` takes Astra, the nearest model above. |
| `google:large`, `google:xlarge` | `gemini-4-argon` | Google's most capable model. `genai-prices` has no Argon price yet; re-place it when it does. |
| `google:small` | `gemini-3.1-flash-lite` | Haiku-priced, and Google's newest model at that price. `gemini-3.5-flash-lite` is Sonnet-priced by output, so it can't replace it. |
| `xai:small` | `grok-4.3` | xAI's cheapest general chat model. Its output price is nearer Sonnet's than Haiku's, so `small` takes it as the nearest model above. |
| `xai:medium`, `xai:large`, `xai:xlarge` | `grok-4.7` | Sonnet-priced and newer than `grok-4.3`. It replaced `grok-4.6` at the same price. Nothing sits above it. |
| `openai:image` | `gpt-image-2.5-flare` | Same price as `gpt-image-2`, newer. Sunburst is OpenAI's slower premium editing model. |
| `openai:embedding` | `text-embedding-3-small` | OpenAI's general-purpose embedding model. `text-embedding-3-large` costs 6.5 times as much. |
