"""The models Anthropic, OpenAI and Google have deprecated on their own APIs, for `ModelProfile.deprecation`.

Compiled from the providers' deprecation pages as fetched on 2026-11-19, Anthropic's again on 2026-11-24, and Google's on
2026-11-26. Google's tables hold only `gemini-2.5-pro` and `gemini-2.5-flash` so far: the rest of Google's pages is
https://github.com/pydantic/pydantic-ai/issues/14054. This file is the source of truth: edit it by hand when a provider
announces a deprecation, following `.agents/skills/add-new-model/SKILL.md`. Each entry follows these rules:

- Add a model only when the provider's notice gives it a deprecation or retirement date. A model the notice doesn't
  date gets no entry, such as the bare `gpt-4o` alias, whose dated snapshots are listed instead.
- Cover only the schedule the provider sets for its own API: the Claude API for Anthropic, Chat Completions and
  Responses for OpenAI, and for Google both the Gemini API and Google Cloud, which date the same model differently. A
  platform that serves the same models on a schedule of its own, such as Amazon Bedrock or Azure, is not covered.
- Key the entry by every id the API accepts for the model, aliases included.
- Set `source_url` to the section of the provider's page that lists the model.
- Set `replacement` to the first model id the notice recommends, or leave it `None` when the notice names none. A
  qualifier the notice adds, such as a reasoning mode, stays on the linked page.
- Set `rerouted_to` to the model the notice says the API answers the id with from its retirement date, or leave it
  `None` when the notice says nothing of the kind.
- Keep an entry after its retirement date: the warning then explains why the request fails, or, where the provider
  routes the id to another model, why another model answers it.
"""

from __future__ import annotations as _annotations

from datetime import date

from ..profiles import ModelDeprecation

ANTHROPIC_MODEL_DEPRECATIONS: dict[str, ModelDeprecation] = {
    model_id: deprecation
    for model_ids, deprecation in (
        (
            ('claude-1.0',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2024, 11, 6),
                replacement='claude-haiku-4-5-20251001',
            ),
        ),
        (
            ('claude-1.1',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2024, 11, 6),
                replacement='claude-haiku-4-5-20251001',
            ),
        ),
        (
            ('claude-1.2',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2024, 11, 6),
                replacement='claude-haiku-4-5-20251001',
            ),
        ),
        (
            ('claude-1.3',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2024, 11, 6),
                replacement='claude-haiku-4-5-20251001',
            ),
        ),
        (
            ('claude-instant-1.0',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2024, 11, 6),
                replacement='claude-haiku-4-5-20251001',
            ),
        ),
        (
            ('claude-instant-1.1',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2024, 11, 6),
                replacement='claude-haiku-4-5-20251001',
            ),
        ),
        (
            ('claude-instant-1.2',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2024, 11, 6),
                replacement='claude-haiku-4-5-20251001',
            ),
        ),
        (
            ('claude-2.0',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2025, 7, 21),
                replacement='claude-opus-4-8',
            ),
        ),
        (
            ('claude-2.1',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2025, 7, 21),
                replacement='claude-opus-4-8',
            ),
        ),
        (
            ('claude-3-sonnet-20240229',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2025, 7, 21),
                replacement='claude-sonnet-5-5',
            ),
        ),
        (
            ('claude-3-5-sonnet-20240620',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2025, 10, 28),
                replacement='claude-sonnet-5-5',
            ),
        ),
        (
            ('claude-3-5-sonnet-20241022',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2025, 10, 28),
                replacement='claude-sonnet-5-5',
            ),
        ),
        (
            ('claude-3-opus-20240229',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2026, 1, 5),
                replacement='claude-opus-4-8',
            ),
        ),
        (
            ('claude-3-5-haiku-20241022',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2026, 2, 19),
                replacement='claude-haiku-4-5-20251001',
            ),
        ),
        (
            ('claude-3-7-sonnet-20250219',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2026, 2, 19),
                replacement='claude-sonnet-5-5',
            ),
        ),
        (
            ('claude-3-haiku-20240307',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2026, 4, 20),
                replacement='claude-haiku-4-5-20251001',
            ),
        ),
        (
            ('claude-opus-4-20250514',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2026, 6, 15),
                replacement='claude-opus-4-8',
            ),
        ),
        (
            ('claude-sonnet-4-20250514',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2026, 6, 15),
                replacement='claude-sonnet-5-5',
            ),
        ),
        (
            ('claude-opus-4-1-20250805',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2026, 8, 5),
                replacement='claude-opus-4-8',
            ),
        ),
        (
            (
                'claude-sonnet-4-5-20250929',
                'claude-sonnet-4-5',
            ),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2026, 11, 30),
                replacement='claude-sonnet-5-5',
            ),
        ),
        (
            (
                'claude-opus-4-5-20251101',
                'claude-opus-4-5',
            ),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#deprecation-history',
                retirement_date=date(2027, 1, 22),
                replacement='claude-opus-5-5',
            ),
        ),
        (
            ('claude-mythos-preview',),
            ModelDeprecation(
                source_url='https://platform.claude.com/docs/en/about-claude/model-deprecations#model-status'
            ),
        ),
    )
    for model_id in model_ids
}
"""Models deprecated on the Claude API: https://platform.claude.com/docs/en/about-claude/model-deprecations."""

OPENAI_MODEL_DEPRECATIONS: dict[str, ModelDeprecation] = {
    model_id: deprecation
    for model_ids, deprecation in (
        (
            ('code-cushman-001',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-03-20-codex-models',
                retirement_date=date(2023, 3, 23),
                replacement='gpt-4o',
            ),
        ),
        (
            ('code-cushman-002',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-03-20-codex-models',
                retirement_date=date(2023, 3, 23),
                replacement='gpt-4o',
            ),
        ),
        (
            ('code-davinci-001',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-03-20-codex-models',
                retirement_date=date(2023, 3, 23),
                replacement='gpt-4o',
            ),
        ),
        (
            ('ada',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='babbage-002',
            ),
        ),
        (
            ('babbage',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='babbage-002',
            ),
        ),
        (
            ('code-davinci-002',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='gpt-3.5-turbo-instruct',
            ),
        ),
        (
            ('code-davinci-edit-001',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='gpt-4o',
            ),
        ),
        (
            ('curie',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='davinci-002',
            ),
        ),
        (
            ('davinci',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='davinci-002',
            ),
        ),
        (
            ('text-ada-001',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='gpt-3.5-turbo-instruct',
            ),
        ),
        (
            ('text-babbage-001',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='gpt-3.5-turbo-instruct',
            ),
        ),
        (
            ('text-curie-001',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='gpt-3.5-turbo-instruct',
            ),
        ),
        (
            ('text-davinci-001',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='gpt-3.5-turbo-instruct',
            ),
        ),
        (
            ('text-davinci-002',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='gpt-3.5-turbo-instruct',
            ),
        ),
        (
            ('text-davinci-003',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='gpt-3.5-turbo-instruct',
            ),
        ),
        (
            ('text-davinci-edit-001',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-07-06-gpt-and-embeddings',
                retirement_date=date(2024, 1, 4),
                replacement='gpt-4o',
            ),
        ),
        (
            ('gpt-3.5-turbo-0301',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-11-06-chat-model-updates',
                retirement_date=date(2024, 9, 13),
                replacement='gpt-3.5-turbo',
            ),
        ),
        (
            ('gpt-3.5-turbo-0613',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-11-06-chat-model-updates',
                retirement_date=date(2024, 9, 13),
                replacement='gpt-3.5-turbo',
            ),
        ),
        (
            ('gpt-3.5-turbo-16k-0613',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2023-11-06-chat-model-updates',
                retirement_date=date(2024, 9, 13),
                replacement='gpt-3.5-turbo',
            ),
        ),
        (
            ('gpt-4-1106-vision-preview',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models',
                retirement_date=date(2024, 12, 6),
                replacement='gpt-4o',
            ),
        ),
        (
            ('gpt-4-vision-preview',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models',
                retirement_date=date(2024, 12, 6),
                replacement='gpt-4o',
            ),
        ),
        (
            ('gpt-4-32k',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models',
                retirement_date=date(2025, 6, 6),
                replacement='gpt-4o',
            ),
        ),
        (
            ('gpt-4-32k-0314',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models',
                retirement_date=date(2025, 6, 6),
                replacement='gpt-4o',
            ),
        ),
        (
            ('gpt-4-32k-0613',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models',
                retirement_date=date(2025, 6, 6),
                replacement='gpt-4o',
            ),
        ),
        (
            ('gpt-4.5-preview',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-04-14-gpt-45-preview',
                retirement_date=date(2025, 7, 14),
                replacement='gpt-4.1',
            ),
        ),
        (
            ('o1-preview',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-04-28-o1-preview-and-o1-mini',
                retirement_date=date(2025, 7, 28),
                replacement='o3',
            ),
        ),
        (
            ('gpt-4o-audio-preview-2024-10-01',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-06-10-gpt-4o-audio-preview-2024-10-01',
                retirement_date=date(2025, 10, 10),
                replacement='gpt-audio-1.5',
            ),
        ),
        (
            ('o1-mini',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-04-28-o1-preview-and-o1-mini',
                retirement_date=date(2025, 10, 27),
                replacement='o4-mini',
            ),
        ),
        (
            ('codex-mini-latest',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-11-17-codex-mini-latest-model-snapshot',
                retirement_date=date(2026, 2, 12),
                replacement='gpt-5-codex-mini',
            ),
        ),
        (
            ('chatgpt-4o-latest',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-11-18-chatgpt-4o-latest-snapshot',
                retirement_date=date(2026, 2, 17),
                replacement='gpt-5.1-chat-latest',
            ),
        ),
        (
            (
                'gpt-4-0125-preview',
                'gpt-4-turbo-preview',
                'gpt-4-turbo-preview-completions',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-09-26-legacy-gpt-model-snapshots-march-2026-shutdown',
                retirement_date=date(2026, 3, 26),
                replacement='gpt-5',
            ),
        ),
        (
            ('gpt-4-0314',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-09-26-legacy-gpt-model-snapshots-march-2026-shutdown',
                retirement_date=date(2026, 3, 26),
                replacement='gpt-5',
            ),
        ),
        (
            ('gpt-4o-audio-preview',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-09-15-gpt-4o-realtime-preview-models',
                retirement_date=date(2026, 5, 7),
                replacement='gpt-audio-1.5',
            ),
        ),
        (
            ('gpt-4o-mini-audio-preview',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-09-15-gpt-4o-realtime-preview-models',
                retirement_date=date(2026, 5, 7),
                replacement='gpt-audio-mini',
            ),
        ),
        (
            (
                'computer-use-preview-2025-03-11',
                'computer-use-preview',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-terra',
            ),
        ),
        (
            ('gpt-4o-mini-search-preview-2025-03-11',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-terra',
            ),
        ),
        (
            ('gpt-4o-search-preview-2025-03-11',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-terra',
            ),
        ),
        (
            ('gpt-5-chat-latest',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            ('gpt-5-codex',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            ('gpt-5.1-chat-latest',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            ('gpt-5.1-codex',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            ('gpt-5.1-codex-max',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            ('gpt-5.1-codex-mini',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-terra',
            ),
        ),
        (
            ('gpt-5.2-codex',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            ('gpt-audio-mini-2025-10-06',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-audio-1.5',
            ),
        ),
        (
            (
                'o3-deep-research-2025-06-26',
                'o3-deep-research',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            (
                'o4-mini-deep-research-2025-06-26',
                'o4-mini-deep-research',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots-july-2026-shutdown',
                retirement_date=date(2026, 7, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            ('gpt-5.2-chat-latest',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-05-08-gpt-52-chat-latest-and-gpt-53-chat-latest-model-snapshots',
                retirement_date=date(2026, 8, 10),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            ('gpt-5.3-chat-latest',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-05-08-gpt-52-chat-latest-and-gpt-53-chat-latest-model-snapshots',
                retirement_date=date(2026, 8, 10),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            ('babbage-002',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-09-26-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 9, 28),
                replacement='gpt-5.6-terra',
            ),
        ),
        (
            ('davinci-002',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-09-26-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 9, 28),
                replacement='gpt-5.6-terra',
            ),
        ),
        (
            ('gpt-3.5-turbo-1106',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-09-26-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 9, 28),
                replacement='gpt-5.6-terra',
            ),
        ),
        (
            ('gpt-3.5-turbo-instruct',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2025-09-26-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 9, 28),
                replacement='gpt-5.6-terra',
            ),
        ),
        (
            ('gpt-5.4-cyber',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-09-11-gpt-54-cyber',
                retirement_date=date(2026, 10, 1),
            ),
        ),
        (
            ('ft-o4-mini-2025-04-16',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 10, 23),
                replacement='gpt-5.6-terra',
            ),
        ),
        (
            (
                'gpt-3.5-turbo-0125',
                'gpt-3.5-turbo',
                'gpt-3.5-turbo-completions',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 10, 23),
                replacement='gpt-5.6-terra',
            ),
        ),
        (
            (
                'gpt-4-0613',
                'gpt-4',
                'gpt-4-0613-completions',
                'gpt-4-completions',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 10, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            ('gpt-4-1106-preview',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 10, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            (
                'gpt-4-turbo',
                'gpt-4-turbo-2024-04-09',
                'gpt-4-turbo-completions',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 10, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            (
                'gpt-4.1-nano',
                'gpt-4.1-nano-2025-04-14',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 10, 23),
                replacement='gpt-5.6-luna',
            ),
        ),
        (
            ('gpt-4o-2024-05-13',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 10, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            (
                'o1-2024-12-17',
                'o1',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 10, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            (
                'o1-pro-2025-03-19',
                'o1-pro',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 10, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            (
                'o3-mini-2025-01-31',
                'o3-mini',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 10, 23),
                replacement='gpt-5.6-sol',
            ),
        ),
        (
            (
                'o4-mini-2025-04-16',
                'o4-mini',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-04-22-legacy-gpt-model-snapshots',
                retirement_date=date(2026, 10, 23),
                replacement='gpt-5.6-terra',
            ),
        ),
        (
            (
                'gpt-5',
                'gpt-5-2025-08-07',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-06-11-gpt-5-and-o3-model-deprecations',
                retirement_date=date(2026, 12, 11),
                replacement='gpt-6.1-sol',
            ),
        ),
        (
            (
                'gpt-5-mini',
                'gpt-5-mini-2025-08-07',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-06-11-gpt-5-and-o3-model-deprecations',
                retirement_date=date(2026, 12, 11),
                replacement='gpt-6.1-terra',
            ),
        ),
        (
            (
                'gpt-5-nano',
                'gpt-5-nano-2025-08-07',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-06-11-gpt-5-and-o3-model-deprecations',
                retirement_date=date(2026, 12, 11),
                replacement='gpt-6-luna',
            ),
        ),
        (
            (
                'gpt-5-pro',
                'gpt-5-pro-2025-10-06',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-06-11-gpt-5-and-o3-model-deprecations',
                retirement_date=date(2026, 12, 11),
                replacement='gpt-6.1-sol',
            ),
        ),
        (
            (
                'o3',
                'o3-2025-04-16',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-06-11-gpt-5-and-o3-model-deprecations',
                retirement_date=date(2026, 12, 11),
                replacement='gpt-6.1-sol',
            ),
        ),
        (
            (
                'o3-pro',
                'o3-pro-2025-06-10',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-06-11-gpt-5-and-o3-model-deprecations',
                retirement_date=date(2026, 12, 11),
                replacement='gpt-6.1-sol',
            ),
        ),
        (
            ('gpt-4.1-2025-04-14',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-10-14-gpt-4-series-model-deprecations',
                retirement_date=date(2027, 1, 14),
                replacement='gpt-6-luna',
            ),
        ),
        (
            ('gpt-4.1-mini-2025-04-14',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-10-14-gpt-4-series-model-deprecations',
                retirement_date=date(2027, 1, 14),
                replacement='gpt-6-luna',
            ),
        ),
        (
            ('gpt-4o-2024-08-06',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-10-14-gpt-4-series-model-deprecations',
                retirement_date=date(2027, 1, 14),
                replacement='gpt-6-luna',
            ),
        ),
        (
            ('gpt-4o-2024-11-20',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-10-14-gpt-4-series-model-deprecations',
                retirement_date=date(2027, 1, 14),
                replacement='gpt-6-luna',
            ),
        ),
        (
            ('gpt-4o-mini-2024-07-18',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-10-14-gpt-4-series-model-deprecations',
                retirement_date=date(2027, 1, 14),
                replacement='gpt-6-luna',
            ),
        ),
        (
            ('gpt-4o-audio',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-07-20-legacy-audio-realtime-and-transcription-models',
                retirement_date=date(2027, 1, 20),
                replacement='gpt-audio-1.5',
            ),
        ),
        (
            ('gpt-4o-mini-audio',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-07-20-legacy-audio-realtime-and-transcription-models',
                retirement_date=date(2027, 1, 20),
                replacement='gpt-audio-1.5',
            ),
        ),
        (
            (
                'gpt-audio',
                'gpt-audio-2025-08-28',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-07-20-legacy-audio-realtime-and-transcription-models',
                retirement_date=date(2027, 1, 20),
                replacement='gpt-audio-1.5',
            ),
        ),
        (
            (
                'gpt-audio-mini',
                'gpt-audio-mini-2025-12-15',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-07-20-legacy-audio-realtime-and-transcription-models',
                retirement_date=date(2027, 1, 20),
                replacement='gpt-audio-1.5',
            ),
        ),
        (
            ('gpt-5.5-cyber-preview',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-11-05-gpt-55-cyber',
                retirement_date=date(2027, 2, 1),
            ),
        ),
        (
            (
                'gpt-5.1',
                'gpt-5.1-2025-11-13',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-10-01-gpt-53-codex-gpt-51-gpt-54-nano',
                retirement_date=date(2027, 4, 1),
                replacement='gpt-6-sol',
            ),
        ),
        (
            ('gpt-5.3-codex',),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-10-01-gpt-53-codex-gpt-51-gpt-54-nano',
                retirement_date=date(2027, 4, 1),
                replacement='gpt-6-sol',
            ),
        ),
        (
            (
                'gpt-5.4-nano',
                'gpt-5.4-nano-2026-03-17',
            ),
            ModelDeprecation(
                source_url='https://developers.openai.com/api/docs/deprecations#2026-10-01-gpt-53-codex-gpt-51-gpt-54-nano',
                retirement_date=date(2027, 4, 1),
                replacement='gpt-6-luna',
            ),
        ),
    )
    for model_id in model_ids
}
"""Models deprecated on the OpenAI API: https://developers.openai.com/api/docs/deprecations."""

GEMINI_API_MODEL_DEPRECATIONS: dict[str, ModelDeprecation] = {
    'gemini-2.5-pro': ModelDeprecation(
        source_url='https://aistudio.google.com/docs/deprecations#gemini-2.5-pro-models',
        retirement_date=date(2026, 12, 1),
        replacement='gemini-3.8-flash',
        rerouted_to='gemini-3.8-flash',
    ),
    'gemini-2.5-flash': ModelDeprecation(
        source_url='https://aistudio.google.com/docs/deprecations#gemini-2.5-flash-models',
        retirement_date=date(2026, 12, 1),
        replacement='gemini-3.5-flash-lite',
        rerouted_to='gemini-3.5-flash-lite',
    ),
}
"""Models deprecated on the Gemini API: https://aistudio.google.com/docs/deprecations."""

GOOGLE_CLOUD_MODEL_DEPRECATIONS: dict[str, ModelDeprecation] = {
    model_id: ModelDeprecation(
        source_url='https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions#gemini-models',
        retirement_date=date(2026, 10, 20),
        replacement='gemini-3.8-flash',
    )
    for model_id in ('gemini-2.5-pro', 'gemini-2.5-flash')
}
"""Models deprecated on Google Cloud: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions."""
