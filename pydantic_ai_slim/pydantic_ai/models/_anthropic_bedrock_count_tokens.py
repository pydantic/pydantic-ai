"""Token counting for Anthropic models served via Amazon Bedrock.

The Anthropic SDK refuses to count tokens on Bedrock: its high-level
`client.beta.messages.count_tokens()` routes to `/v1/messages/count_tokens`, which
`anthropic/lib/bedrock/_client.py` rejects with
`AnthropicError('Token counting is not supported in Bedrock yet')`.

Bedrock *does* support token counting, but only through its own Bedrock Runtime endpoint
`/model/{model}/count-tokens`, which wraps an `InvokeModel`-style body. So we bypass the
high-level method and issue the low-level request ourselves via the client's `.post()`.
Don't "simplify" this back to `count_tokens()` — it will start raising again.
"""

from __future__ import annotations

import base64
import urllib.parse
from typing import TYPE_CHECKING, Literal

from anthropic import NotGiven, Omit
from anthropic.types.beta import BetaMessageTokensCount
from pydantic_core import to_json

from .._utils import is_str_dict
from ..exceptions import UnexpectedModelBehavior

if TYPE_CHECKING:
    from anthropic import (
        AsyncAnthropicBedrock,  # pyright: ignore[reportPrivateImportUsage]
        RequestOptions,
        Timeout,
    )
    from anthropic.types.anthropic_beta_param import AnthropicBetaParam
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaContextManagementConfigParam,
        BetaMessageParam,
        BetaOutputConfigParam,
        BetaRequestMCPServerURLDefinitionParam,
        BetaTextBlockParam,
        BetaThinkingConfigParam,
        BetaToolChoiceParam,
        BetaToolUnionParam,
    )


async def count_tokens_via_bedrock(
    client: AsyncAnthropicBedrock,
    model: str,
    *,
    system: str | list[BetaTextBlockParam] | Omit,
    messages: list[BetaMessageParam],
    max_tokens: int,
    tools: list[BetaToolUnionParam] | Omit,
    tool_choice: BetaToolChoiceParam | Omit,
    mcp_servers: list[BetaRequestMCPServerURLDefinitionParam] | Omit,
    betas: list[AnthropicBetaParam] | Omit,
    output_config: BetaOutputConfigParam | Omit,
    cache_control: BetaCacheControlEphemeralParam | Omit,
    thinking: BetaThinkingConfigParam | Omit,
    context_management: BetaContextManagementConfigParam | Omit,
    timeout: float | Timeout | None | NotGiven,
    speed: Literal['standard', 'fast'] | Omit,
    extra_headers: dict[str, str],
    extra_body: object | None,
) -> BetaMessageTokensCount:
    """Count input tokens via Bedrock Runtime's `/model/{model}/count-tokens` endpoint.

    Mirrors the parameters the regular Messages request sends, so the count matches what
    inference would actually be billed. API errors should be mapped by the caller (the
    `.post()` raises the SDK's `APIStatusError` on non-2xx).
    """
    body: dict[str, object] = {
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': max_tokens,
        'messages': messages,
    }
    for key, value in (
        ('system', system),
        ('tools', tools),
        ('tool_choice', tool_choice),
        ('mcp_servers', mcp_servers),
        ('output_config', output_config),
        ('cache_control', cache_control),
        ('thinking', thinking),
        ('context_management', context_management),
        ('speed', speed),
    ):
        if not isinstance(value, Omit):
            body[key] = value
    if not isinstance(betas, Omit):
        body['anthropic_beta'] = betas
    if is_str_dict(extra_body):
        body.update(extra_body)

    options: RequestOptions = {'headers': {**extra_headers, 'Content-Type': 'application/json'}}
    if not isinstance(timeout, NotGiven):
        options['timeout'] = timeout

    # Bedrock CountTokens only accepts BASE foundation-model ids (e.g.
    # `anthropic.claude-sonnet-4-20250514-v1:0`). Cross-region inference profile ids (the
    # `us.`/`eu.`/`global.` prefixes) 400 with "The provided model doesn't support counting
    # tokens", and end-of-life model versions 404. We deliberately don't translate those —
    # Bedrock's own error message is clearer than anything we'd substitute.
    quoted_model = urllib.parse.quote(model, safe=':')
    encoded_body = base64.b64encode(to_json(body)).decode()
    content = to_json({'input': {'invokeModel': {'body': encoded_body}}})
    # `cast_to=object` (not `dict[str, object]`): the SDK passes `cast_to` to `issubclass()`, which
    # raises `TypeError` on a subscripted generic under Python 3.10. `object` returns the raw parsed
    # JSON body, which we validate explicitly below.
    response = await client.post(
        f'/model/{quoted_model}/count-tokens',
        cast_to=object,
        content=content,
        options=options,
    )

    if is_str_dict(response) and isinstance(input_tokens := response.get('inputTokens'), int):
        return BetaMessageTokensCount(input_tokens=input_tokens)
    raise UnexpectedModelBehavior('Unexpected Bedrock count tokens response')
