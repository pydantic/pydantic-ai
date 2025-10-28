from collections.abc import Sequence
from typing import cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    TextPart,
    ThinkingPart,
    UnexpectedModelBehavior,
)
from pydantic_ai.direct import model_request

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice

    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


async def test_openrouter_with_preset(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash-lite', provider=provider)
    settings = OpenRouterModelSettings(openrouter_preset='@preset/comedian')
    response = await model_request(model, [ModelRequest.user_text_prompt('Trains')], model_settings=settings)
    text_part = cast(TextPart, response.parts[0])
    assert text_part.content == snapshot(
        """\
Why did the train break up with the track?

Because it felt like their relationship was going nowhere.\
"""
    )


async def test_openrouter_with_native_options(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)
    # These specific settings will force OpenRouter to use the fallback model, since Gemini is not available via the xAI provider.
    settings = OpenRouterModelSettings(
        openrouter_models=['x-ai/grok-4'],
        openrouter_transforms=['middle-out'],
        openrouter_provider={'only': ['xai']},
    )
    response = await model_request(model, [ModelRequest.user_text_prompt('Who are you')], model_settings=settings)
    text_part = cast(TextPart, response.parts[0])
    assert text_part.content == snapshot(
        """\
I'm Grok, a helpful and maximally truthful AI built by xAI. I'm not based on any other companies' models—instead, I'm inspired by the Hitchhiker's Guide to the Galaxy and JARVIS from Iron Man. My goal is to assist with questions, provide information, and maybe crack a joke or two along the way.

What can I help you with today?\
"""
    )
    assert response.provider_details is not None
    assert response.provider_details['downstream_provider'] == 'xAI'
    assert response.provider_details['native_finish_reason'] == 'stop'


async def test_openrouter_with_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    request = ModelRequest.user_text_prompt(
        "What was the impact of Voltaire's writings on modern french culture? Think about your answer."
    )

    model = OpenRouterModel('z-ai/glm-4.6', provider=provider)
    response = await model_request(model, [request])

    assert len(response.parts) == 2

    thinking_part = response.parts[0]
    assert isinstance(thinking_part, ThinkingPart)
    assert thinking_part.id == snapshot(None)
    assert thinking_part.content is not None
    assert thinking_part.signature is None


async def test_openrouter_preserve_reasoning_block(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('openai/o3', provider=provider)

    messages: Sequence[ModelMessage] = []
    messages.append(
        ModelRequest.user_text_prompt(
            "What was the impact of Voltaire's writings on modern french culture? Think about your answer."
        )
    )
    response = await model_request(model, messages)
    messages.append(response)

    openai_messages = await model._map_messages(messages)  # type: ignore[reportPrivateUsage]

    assistant_message = openai_messages[1]
    assert 'reasoning_details' in assistant_message
    assert assistant_message['reasoning_details'] == snapshot(
        [
            {
                'id': None,
                'format': 'openai-responses-v1',
                'index': 0,
                'type': 'reasoning.summary',
                'summary': """\
**Exploring Voltaire's impact**

The user seeks a thoughtful answer about Voltaire's influence on modern French culture. I should summarize his significance by discussing Enlightenment values like liberté and human rights, his role in shaping French Revolution ideas, and his distinctive use of satire. It’s also important to address his anti-clerical stance that encouraged secularism, the promotion of freedom of speech, his legacy in literature, and presence in institutions. Finally, I want to touch on his enduring impact on contemporary discussions around laïcité and cultural narrative.**Structuring the response**

I’m thinking about how to organize the response. I want to start with an introduction that highlights Voltaire as a central figure of the Enlightenment, noting how his writings shaped essential French values like rationality, secularism, and tolerance. After that, I should break the discussion into categories such as political thought, secularism, literature, education, and popular culture. I aim to develop this into a thoughtful answer of around 800 to 1000 words, which will cover all these important aspects.\
""",
            },
            {
                'id': 'rs_06569cab051e4c78016900b1eb409c81909668d636e3a424f9',
                'format': 'openai-responses-v1',
                'index': 0,
                'type': 'reasoning.encrypted',
                'data': 'gAAAAABpALH0SrlqG_JSYqQT07H5yKQO7GcmENK5_dmCbkx_o6J5Qg7kZHHNvKynIDieAzknNQwmLf4SB96VU59uasmIK5oOyInBjdpczoYyFEoN8zKWVOolEaCLXSMJfybQb6U_CKEYmPrh_B4CsGDuLzKq7ak6ERC0qFb0vh6ctchIyzWN7MztgnrNt85TReEN3yPmox0suv_kjEc4K5nB6L8C5NOK8ZG4Y3X88feoIvteZq0u2AapGPAYJ-tqWqbwYBBBocX7nYfOw3mVGgHb1Ku7pqf13IoWtgR-hz0lmgsLuzGucmjaBKE881oodwUGKUkDWuUbIiFIxGLH5V6cR53XttM91wAKoUgizg0HuFHS_TEYeP2rJhVBv-8OpUmdKFs-lIrCqVBlJDeIwQS_jGSaY4_z-6Prjh797X3_mSDtXBNqaiAgRQkMHHytW6mrVfZVA-cXikTLX5CRc266KNY6MkaJRAS7-tOKxjMwE-IyvmrIIMW4YTdnoaTfVcZhE5YpbrqilZllTW2RtU4lLFh4DFmBRplJsh2F4the_VXm1LITRYrZlrkLB3qTkA_oPslxFxGk_BApWRmbpCxs9mNgwzqqDCsYyvkGqUNAqCTdgPZMApWwJyRNURu_s8yHo-wcLS42zgPvC64E2GvNaO5G5xPFApbHyN950seaSiivqLc-TysXpk6RxNwKm2l1EJDPvMk0G6sZnLlQVPSXQQsCcSfZmJFSHUNSk7u99o5JsuHWsW5oco2cD111Ghd2EAujdimTRGbhjhTTt1SOGl0DL7EgYVWFiYXxgB7XsXy6PgzuIXBuJkJRn4qpk6VeRHpHujbntbVlxlt5ah-lcvRqka8JEew5NXv4qL5zuMQiSIhmHdw_zVucZv7TqknUPJSovsFte40pYwVIeQP23HMekTqsAwEjc4S28Kg300IchGuEi9ihEL9-5_kgrFTgGOOQhNYo28ftnTD7LtoS5m3Av9CraHdbK9Y4bs1u7-qFfCopcamLXJPQe1ZQ0xqR3_zGQJtK24N_oi2Et5g4o_flqzyVwrd83B5nrcbUuayJL3C9SQg4NR2VD8eS96c3qIl_FxCsD6SoTQu22VbrRngvkM_WP1EtvBSKwMtYHnHlQSufV1bkv4E3JXfHg2UJZdvJ0MtfNMTY9qx39YlI1A1Ds4ctMjCF4qAS2XPkUvvgIpwFq4JzH3v2d-f57itMmqamINLmxP2Pv1J69kj7M_shl_FWTJrWn_MtKLsS77Awxc3NdhXhvA2ketiLp_wOE8CED-o4j_Yh0NKy2AVNqeQcmZvJ3FK2vysB2oAjRqTemcad_B2fHkdceoMvSqAYk26gGm8Nvu8GK_atpKOfi1akGKQBRoERZmPT2wyDpXXS4GdVMyC8m5MUa7xJHwUsRDn4ucW792Pt_5skKrBK_So2pGhmoZa8nJYZ8x7O9ZNEXF6a4OIRgbGKnkVpP95YzlQAsVxR31YXkE1pcdM4nRqCpPjdoQjZ0Twr0ute5v4J6Lhb1F3FsNrg3Sm9YRkJ9h-yfUfvyt1bK1V4nFMtRFt120WjfIvlZZ-1qyenToySK8doSSUZ6VOQWG_ieBkf-IRAN3eONC0n6BfGogsVlPXhXHLznouLnzapC4pGWWBIDsGlTvZj_o7UpHMPr_20PDC5d2jSGGtXf6kJvtfsAnJjtQPHs41VfDLyT-yQIlnUd8QvdwUlQ22A79I-rg38C8BWJNqg3sbOtzMMpt6R8Cvyp4dmB1ksS24tpiEZZ42aH8JIgoqs1sRbFPsC1v3kDPd3XRbbKpliQxseR_xWMNZkGj2F8q9HH1lgLkkCod_97OYrYBROxn9K79wlkZBUFjrNXA3EuiBf-IDOvQeKtDRypAaTKnHybIEOIypTNOWjhGT6oQutKSFswfvSeJGA0fF26FAgxnVmzFS7eAyzSHDqygQfhB7Yp7N2yEbD0eFLUs8qgete-eDIn6eM5E5eMnT1JeP6LD8ku5iR30sDdU8O6BrsGvUypMSID-hoBDytF1_GS6yOhMsU4pXZHTJ4yYNUOFyMH3ReE3SeAuFFohR9aXTpUA5YeLy6-Xo0_ZA9FuFMDVK4Bp1F5f-2BJ3FXRc1aqtyROpMdBtY4ehEqm-FKqbYd4VlaIMb9adG1LnOgWpnWCr9ciOP-c75rxX885yZLXO8rHJ_wbg04JzobGFnKdZHPtCYiTgkpnFavesiy9iI_bRO0Mu7SaDwXne6u4NY4YIHGRRCKR7o98lvSCOw7PT3SgWPoHEML6Na0QnycJeiPayB7megnFGfQZn_lSzDDeAiKgOBJ42LJZf3ysH1Dueqb7icX6xn4HlrMJdDLhMCgvwry4QQkrgrsIhyrTFNt2j--0IO7hX4RbwU5v5yudb0QRQovbmjoPRk4qeZKOyH-YSW-J1lu2MJcrk9Z-Sc4d875cZ6B3HxUuJFYQWqMoJ6EkZjNRLpp3XBkFEu5ip4md7yu_FYK7SInsuVI0igMVRx5i9vURIiGnVf60yBfWpqJac0Jp_7V3ftXsdXk3pSE4_GF9QgKM4l9chXH-frUEExxXS13BRQJP1b29-0B7ciG-48c18uSktRItBmXv2_baSiyo7_nvnPWVUgpig9qOuiFFmVPPvFRGTQS6jh8dPZR8PCFnpxuMhVrrJDJUNu8wmVGdVDMZP6kO2PYhNpz35RrX25SSgzjbl6V4uFDb7KQdWQAv-78QkLibUeB7w47I_2G47TGxbUsmnTt_sss1LW0peGrmCMKncgSQKro8rSBQ==',
            },
        ]
    )


async def test_openrouter_errors_raised(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)
    agent = Agent(model, instructions='Be helpful.', retries=1)
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Tell me a joke.')
    assert str(exc_info.value) == snapshot(
        "status_code: 429, model_name: google/gemini-2.0-flash-exp:free, body: {'code': 429, 'message': 'Provider returned error', 'metadata': {'provider_name': 'Google', 'raw': 'google/gemini-2.0-flash-exp:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations'}}"
    )


async def test_openrouter_validate_non_json_response(openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        model._process_response('This is not JSON!')  # type: ignore[reportPrivateUsage]

    assert str(exc_info.value) == snapshot(
        'Invalid response from OpenRouter chat completions endpoint, expected JSON data'
    )


async def test_openrouter_validate_error_response(openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    choice = Choice.model_construct(
        index=0, message={'role': 'assistant'}, finish_reason='error', native_finish_reason='stop'
    )
    response = ChatCompletion.model_construct(
        id='', choices=[choice], created=0, object='chat.completion', model='test', provider='test'
    )
    response.error = {'message': 'This response has an error attribute', 'code': 200}  # type: ignore[reportAttributeAccessIssue]

    with pytest.raises(ModelHTTPError) as exc_info:
        model._process_response(response)  # type: ignore[reportPrivateUsage]

    assert str(exc_info.value) == snapshot(
        'status_code: 200, model_name: test, body: This response has an error attribute'
    )


async def test_openrouter_validate_error_finish_reason(openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.0-flash-exp:free', provider=provider)

    choice = Choice.model_construct(
        index=0, message={'role': 'assistant'}, finish_reason='error', native_finish_reason='stop'
    )
    response = ChatCompletion.model_construct(
        id='', choices=[choice], created=0, object='chat.completion', model='test', provider='test'
    )

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        model._process_response(response)  # type: ignore[reportPrivateUsage]

    assert str(exc_info.value) == snapshot(
        'Invalid response from OpenRouter chat completions endpoint, error finish_reason without error data'
    )


async def test_openrouter_map_messages_reasoning(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-3.7-sonnet:thinking', provider=provider)

    user_message = ModelRequest.user_text_prompt('Who are you. Think about it.')
    response = await model_request(model, [user_message])

    mapped_messages = await model._map_messages([user_message, response])  # type: ignore[reportPrivateUsage]

    assert len(mapped_messages) == 2
    assert mapped_messages[1]['reasoning_details'] == snapshot(  # type: ignore[reportGeneralTypeIssues]
        [
            {
                'id': None,
                'type': 'reasoning.text',
                'text': """\
This question is asking me about my identity. Let me think about how to respond clearly and accurately.

I am Claude, an AI assistant created by Anthropic. I'm designed to be helpful, harmless, and honest in my interactions with humans. I don't have a physical form - I exist as a large language model running on computer hardware. I don't have consciousness, sentience, or feelings in the way humans do. I don't have personal experiences or a life outside of these conversations.

My capabilities include understanding and generating natural language text, reasoning about various topics, and attempting to be helpful to users in a wide range of contexts. I have been trained on a large corpus of text data, but my training data has a cutoff date, so I don't have knowledge of events that occurred after my training.

I have certain limitations - I don't have the ability to access the internet, run code, or interact with external systems unless given specific tools to do so. I don't have perfect knowledge and can make mistakes.

I'm designed to be conversational and to engage with users in a way that's helpful and informative, while respecting important ethical boundaries.\
""",
                'signature': 'ErcBCkgICBACGAIiQHtMxpqcMhnwgGUmSDWGoOL9ZHTbDKjWnhbFm0xKzFl0NmXFjQQxjFj5mieRYY718fINsJMGjycTVYeiu69npakSDDrsnKYAD/fdcpI57xoMHlQBxI93RMa5CSUZIjAFVCMQF5GfLLQCibyPbb7LhZ4kLIFxw/nqsTwDDt6bx3yipUcq7G7eGts8MZ6LxOYqHTlIDx0tfHRIlkkcNCdB2sUeMqP8e7kuQqIHoD52GAI=',
                'format': 'anthropic-claude-v1',
                'index': 0,
            }
        ]
    )
