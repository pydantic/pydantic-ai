from typing import cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    ModelHTTPError,
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

    model = OpenRouterModel('openai/o3', provider=provider)
    response = await model_request(model, [request])

    assert len(response.parts) == 3

    thinking_summary_part = response.parts[0]
    thinking_redacted_part = response.parts[1]
    assert isinstance(thinking_summary_part, ThinkingPart)
    assert isinstance(thinking_redacted_part, ThinkingPart)
    assert thinking_summary_part.id == snapshot(None)
    assert thinking_summary_part.content is not None
    assert thinking_summary_part.signature is None
    assert thinking_redacted_part.id == snapshot('rs_068633cde4ea68920168ff95bdf3d881969382f905c626cbcd')
    assert thinking_redacted_part.content == ''
    assert thinking_redacted_part.signature == snapshot(
        'gAAAAABo_5XHkkwCuk-f0waWF42hzBg4R9rD9YUpiXWCgX81P6W2mXf-FLIDTmdvRxm3ctZjMD8Uw4Om_8HIu4TCVHd56avFGbdKVUHf7xNzSBJqYxlGLsp7OB3LKukXWjekw9i9dotrHHZQkXyRRt5esuDGujsquFbI8WYFhMCEhPZaAIJ-IKrnxiS2f2MJxVyGWv9eRWRzZFJmJUr8MHZpIbJS6tLumThbN1fGhn0hes-OWWxfNKfclSoZz86qego4k0Zo8PF2tYqX1uKvLOBr-SSPwplUU798j3DFxMQo6pdAZRT4pJGd-L19nMlrn8DQ5LyIFEV7hMIRD-ieJThuQ3OBei5xJaH1fmZwDFKJHQn_agcZDflY36HlCbIrt1ab-sLgsB4D0TCRq4j42cH0xc3qXC1wrMGuPUOO8CsvbDssJgdXSmTJKhrmsCMH4pPKh0PY983sFlGp_WeRT7RX--NA7JD7sUe7ZlVAWeaQdkXtNmLcvlMl8GdAUrErpUCLvvxYSgD5skISESjgY_gMKCi7NHPaOdgKvRgTc6S5aW2J_xzUyJWDGfPwzIWirVlesEjtUsloeieWRwa-C9YNDi9ZrDhSTqdoAHBW6J6sm1cDGOVN9GqtZ_SmOMGYrYVQZxkvV4nheM6lShDVUHqxh7P5IPazkWGjQBGTccje6RDFDLpBJ2x_gP9qR9aS0VWRieVN6swzpln--lDiKXNvK2RP_5sm0wiCiR14yjxsLibSudCnZWj3f3PWbqcT0xXoqJc0sERzwSrNldt9my7hN8dgWwG2q-atczccNNLSwut7dgiGSazaXHz0SsGvQRi2Gw5bAYfh2mJSyVbA0ZyRYv6nFpilNrQlpeXCoqbvBGbsDDZSfOqnO_OxfNr4yKSeP6JeJnY1-DMZ-zl6eN80ipjlTlJ60opdsJmSa3hGQxsGDVqIL2Ep3sHGT78MZXj2bPEtU5kEhfyD4f4ghfHZeKczJ5TvYKNFEfS9kUnoIbP0uiB1udDOz5mij_GwlvqeqnHSdOaaOoSxxDviPbcbPZgDTVPhADwpAGVkOK3TzysnQZjijAmOzcLp6-LpBG2LBrLatzHH4_wJUEWXFi-ORzvnxTaVGDR8Dn2UuKF4v81blN2-j14xp_2DaojIqIg2XhyDq6Q2a6s6a0rdqtnWC0QhiBC6-TCyDlBHTbENIOsGpmAykGD4_-hnx9Pp69CvfCmwjpgsRF319nNL_2awDWpQyIfxQ-smC-v_ljCF7JDwianEyKsM30tIqNsAcJwtf5_f98TUpbSHzDbw_7tGdzIZKIJQkqteMyKJXOtaZ_XxDcwCs9cswsDwutvTxdqtIXZN526l59zQ9uNYx9roLdN8n6WbhuZUSkQK7xrS-KgqdxZPlMg-ImqlwgWkDM8G8DLLcHuAzrb1r7F2tXjGjUDSDWS1zSCScg79WezPcL4bQ_zTAMDOQ0w4OMCxCJKwRye4KYY7c4QnCVdW3JiFrHByo2oVZOtknTiJOllsY6OkziVeiRrMwiRMhwgGKAisTosFcqzHILptzApWYIb8Jdx3glaJStoWbdgV80Rne_u333z3FfCajpwsfvkGM_yss5jAgcN_eNDXKBTZxx8NUu3d3Kz2u0tr_5MBOILwuItUNqWLhc1oMqkFnHrXwna874t9bgOvxR3ve552BjXL54XU7aGjTPKAGTAcWsxnvS2tx-wpTO8vZ9tsCnbDItpu9JDZnd-JqAfIr37qqeygxt0iIwPhRLz6Jlrjd4aqnpAhsFTb3CPkUX0hOeiCWYRkuTuqLXCUsycCjc_6Y5oznlPd6Pf2_IEpGBn819ob32Z8vbXsO7eTtBz3Yxc0CkuizkQQa_Efgz5kFFJoWwmzlMbl-SlArEgvNaiXv1WFWTR9jrUFZ1GRscczFbTjYWanmLawtR9NFyTj4G7XjB4Ikc4cyZIIsqEtRJH7IMd-3HVSnJX5jICyHagugShAPpwnyA_dn4_kiyXl821_nLCyWWMrQQNxQvoKA9EDKaXRLJ6RpnKWB5vaOm4el2v8rIgpE7OAhNW4wowVTdnn9lk3FcYa2arv_4415X07VY03njlmCV025HRxMNbc9ay4B6nndEBLHbP5TpfJHOzF6o57keP3LauKOzyVLN9YOsuc2Ht9vd4JZiyCuVzAaj4Z_G-ZvcSvRvXkTCS-bjyGH2FPw7MAQWDBw6ArBi-WTSYFwX4_k_bb0QjGmMjrLrbn4Vt_ClGKaapUajENwcnPBIpQ-p1yQBq0lSEQeVPwX76zIiktgiBFOD0MkJiWZSgybnwSdM3sN1kr7mP6Lph9DP7JiTHLakd-htJAyVmJbvRuT2_vpH9ywMddWpeHQiwBGhBjYxVWv0AdYUw7Gs7pVCC9ccPM1A727NGs-ecXFvxutO3lyR16zgw-e2dEt6eITYS4e1IsCe05r01WDUbR8B6IsMFl0sd7qG69X8nVLbK-m8sfURYLSrLsiXvsrmWNBaqraVfj1y6ALTKuJ657heQsF0BuNYVJldK_SgmmefTExc_t6ApkbkokWWMLZxk3J9wtCs4xrUzFkHX3AkqLpiAdi3yUyTjr-vPA5KHXLcBoDuj883w4yfwmKN_hGphWAcjv3N99_ao9UGYfNVIJmxcg7vMGlA1uEyawY3WjA5xlSrM6k--lph3PrT9Ukm8ojiZCMaMiJDVNjKORUJUyiSW8qJTcZEvKmfoju9KDuVfPbf0zT8vmQXWmAzWuH0QMi1KXjQEqtVoqgetV-YwzaZ-i7m8KPWxkRjV4t8aM1P6k71fA7DnOunbySPlEG-jNqxIrY5HNTbinDBDF_zp52JpL0saMKUfnY2EHL8gWXoXG4OguxzNFofp3tPk_uadv8rbmdno3RVPB7KrJqZFizoQ35F7MahgHCunKr9oK4uJ82sWQEa-tXgX8GI7a_rp-O5U6faibRjFSZODU-WXukzoSMhQrcJDpXT_1s5imdkJDV0wM20e-f18fjniMaSaCmgXOA3RdnPlZc26c6giZ7InttDaNRZCr-RCsDjQQVN4AKwnE5XM3yHL2usRx8ILmXZYWfTDNn-UDeueocDcuPhx9aMFf2rRMcw=='
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
