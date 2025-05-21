from typing import Annotated, Literal

import pydantic
from typing_extensions import NotRequired, TypedDict


class GeminiModalityTokenCount(TypedDict):
    """See <https://ai.google.dev/api/generate-content#modalitytokencount>."""

    modality: Annotated[
        Literal['MODALITY_UNSPECIFIED', 'TEXT', 'IMAGE', 'VIDEO', 'AUDIO', 'DOCUMENT'], pydantic.Field(alias='modality')
    ]
    token_count: Annotated[int, pydantic.Field(alias='tokenCount', default=0)]


@pydantic.with_config(pydantic.ConfigDict(populate_by_name=True))
class GeminiUsageMetaData(TypedDict, total=False):
    """See <https://ai.google.dev/api/generate-content#UsageMetadata>.

    The docs suggest all fields are required, but some are actually not required, so we assume they are all optional.
    """

    prompt_token_count: Annotated[int, pydantic.Field(alias='promptTokenCount')]
    candidates_token_count: NotRequired[Annotated[int, pydantic.Field(alias='candidatesTokenCount')]]
    total_token_count: Annotated[int, pydantic.Field(alias='totalTokenCount')]
    cached_content_token_count: NotRequired[Annotated[int, pydantic.Field(alias='cachedContentTokenCount')]]
    thoughts_token_count: NotRequired[Annotated[int, pydantic.Field(alias='thoughtsTokenCount')]]
    tool_use_prompt_token_count: NotRequired[Annotated[int, pydantic.Field(alias='toolUsePromptTokenCount')]]
    prompt_tokens_details: NotRequired[
        Annotated[list[GeminiModalityTokenCount], pydantic.Field(alias='promptTokensDetails')]
    ]
    cache_tokens_details: NotRequired[
        Annotated[list[GeminiModalityTokenCount], pydantic.Field(alias='cacheTokensDetails')]
    ]
    candidates_tokens_details: NotRequired[
        Annotated[list[GeminiModalityTokenCount], pydantic.Field(alias='candidatesTokensDetails')]
    ]
    tool_use_prompt_tokens_details: NotRequired[
        Annotated[list[GeminiModalityTokenCount], pydantic.Field(alias='toolUsePromptTokensDetails')]
    ]


gemini_usage_metadata_ta = pydantic.TypeAdapter(GeminiUsageMetaData)


def parse_usage_details(metadata: GeminiUsageMetaData) -> dict[str, int]:
    details: dict[str, int] = {}
    if cached_content_token_count := metadata.get('cached_content_token_count'):
        # 'cached_content_token_count' left for backwards compatibility
        details['cached_content_token_count'] = cached_content_token_count  # pragma: no cover
        details['cached_content_tokens'] = cached_content_token_count  # pragma: no cover

    if thoughts_token_count := metadata.get('thoughts_token_count'):
        details['thoughts_tokens'] = thoughts_token_count

    if tool_use_prompt_token_count := metadata.get('tool_use_prompt_token_count'):
        details['tool_use_prompt_tokens'] = tool_use_prompt_token_count  # pragma: no cover

    detailed_keys_map: dict[str, str] = {
        'prompt_tokens_details': 'prompt_tokens',
        'cache_tokens_details': 'cache_tokens',
        'candidates_tokens_details': 'candidates_tokens',
        'tool_use_prompt_tokens_details': 'tool_use_prompt_tokens',
    }

    details.update(
        {
            f'{detail["modality"].lower()}_{suffix}': detail['token_count']
            for key, suffix in detailed_keys_map.items()
            if (metadata_details := metadata.get(key))
            for detail in metadata_details
        }
    )

    return details
