"""OpenAI Responses API tool_choice tests.

These tests verify that tool_choice settings are correctly handled for the OpenAI Responses
API (OpenAIResponsesModel). Each test class focuses on a specific tool_choice option.

Tests are recorded as VCR cassettes against the live OpenAI API.

Note: `tool_choice='required'` and `tool_choice=[list]` are designed for direct model
requests, not agent runs, because they exclude output tools.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings, ToolsPlusOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, UsageLimits

from ...conftest import IsNow, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


# =============================================================================
# Tool definitions (declared at module level, not using decorator pattern)
# =============================================================================


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f'Sunny, 22C in {city}'


def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f'14:30 in {timezone}'


def get_population(city: str) -> str:
    """Get the population of a city."""
    return f'{city} has 1 million people'


# =============================================================================
# Helper functions for direct model requests
# =============================================================================


def make_tool_def(name: str, description: str, param_name: str) -> ToolDefinition:
    """Create a ToolDefinition for testing direct model requests."""
    return ToolDefinition(
        name=name,
        description=description,
        parameters_json_schema={
            'type': 'object',
            'properties': {param_name: {'type': 'string'}},
            'required': [param_name],
        },
    )


# =============================================================================
# Structured output model
# =============================================================================


class CityInfo(BaseModel):
    """Information about a city."""

    city: str
    summary: str


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def openai_responses_model(openai_api_key: str) -> OpenAIResponsesModel:
    """Create an OpenAI Responses model for testing."""
    return OpenAIResponsesModel('gpt-5-mini', provider=OpenAIProvider(api_key=openai_api_key))


# =============================================================================
# Test Classes
# =============================================================================


class TestToolChoiceAuto:
    """Tests for tool_choice=None and tool_choice='auto'.

    When tool_choice is None or 'auto', the model decides whether to use tools.
    """

    async def test_auto_with_function_tools_uses_tool(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model uses a function tool when tool_choice='auto' and tools are available."""
        agent: Agent[None, str] = Agent(openai_responses_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            "What's the weather in Paris?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=1000),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="What's the weather in Paris?",
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0098495564d87959006945ad69e2788195a88242375e92e1dd',
                            signature='gAAAAABpRa1rgqM-6eFKT4ZrmTJCCO_uzbJ01y667ybC7JKmefEKU7lxIYMFNurVQ6Vtb3_w2Qw_BEDzVmaJba6PWguLc0ocVXjz2DpznW9ZKEROso2fnFSGWWPa7G7VDSMfx84U-78vUk-t_dPzaadIARY7kx00fM5W1w39SguSFjTcqyXIKpOimYngtXhqdQN4hxBf66gnNIJtUTlXlKYwbK35UdQJ_F50Nz1MRMhrFtr8lLeNh2K5loN2E3q_A-UK7X3q85VEQ5Bgkfna_WIdkKISh2cm6DzJL45tul8wkOrHec4xFUOHxWiHnQeAd2Mjri9DizNuZoQSjLhBH7mlmeVdSuUaPeI4nrDAgORGQnNH1Nqlg6Sp74P6zoS1stXU2v43QVwhgrq6HdAizo-3kmVLfkFEKxtn1DuMD0tzMvtOkYcbZxuFeCsHWaqRgQRKTrHa-wXthkCRmBw5cXK6cw8WdzbQUY2s226inuMAjmL67IA00nz1puFb58oofRDRfiOaFoeekAhwA_S-TcP8GY5rpwwCvgLMjsSH5TaMt2_t1M0WpreJdSDzPCs_OQAo5Gtbvow57BUaeBy_UcWTSWvWyNyXclXYQxAN2EKoUNRrdZtNyzDOTpsD5VM-2TAOJYis02tBnn86o28-Sv98VQ8dnm2sTqHuR0eIPFQiMKIQlvTFAdYMNh5wRLPp8ixHx1wKh5GOqwNrMokr6mrUj0PGNZVcuV13y5cDOshj260fPPu-xf2k5AOVms5HrLZUb1QSGS4Z2OEctc2FvXoY_VKH6-UzG-6aIBck6Mbcjw6PIaoJZGA1SIhBTUKRrhGnAtBRKtI83WDzuyCwESpMYKxuJhhXHqWt4BOdVkFu2EBUhj-qx0gpRzvsNKHyHQrBSir7Gz1aIO6ndf8ijE5hu2Y0oQTZZVRNCxiZ7HzBo8Xcdog_RdLyOnUgKbJ81JhhPC38xN1IX3CFcrszrp2NmHA6UgcVWvohQPWSq8PUKE64MxiIUxQWvvFP0w9p6R5eh7myGYq1IkMj4gh1VmJKSl6jGHy7GR-Nm6Swpxm3c4YTAaKV4N7fGP9zgqkmUJ-PQANaNUPeqS_aIOGe_MfGOg09Uwp46T9C73wtwEkNHsLPaxHbBUfnsrQAtphyXRztjftfYvnHRrZW7lEvbPhEJjyLGE_-xHxakBj9IAdnkDzaE2lArQM1PB3qO9t09Qqnu3fy4uITvl-jGv7k7MnweirpaifMEY_DdNTsxOTqNXqmS4OA1zxjau-55WEZsWS3w1Nm00pa',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Paris"}',
                            tool_call_id='call_xPQM0uDwo0Vtqsu6P38OJgX6',
                            id='fc_0098495564d87959006945ad6adb9c819589ca1aa682b00807',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=84, details={'reasoning_tokens': 64}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 17, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0098495564d87959006945ad6953ac819586d5cb744edcd065',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Paris',
                            tool_call_id='call_xPQM0uDwo0Vtqsu6P38OJgX6',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="It's sunny in Paris with a temperature of 22°C.",
                            id='msg_0098495564d87959006945ad6c14108195979a0f5cc2f73449',
                        )
                    ],
                    usage=RequestUsage(input_tokens=155, output_tokens=16, details={'reasoning_tokens': 0}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 19, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0098495564d87959006945ad6b96c481958c457fe976f58d9a',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_function_tools_can_respond_directly(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model can respond without tools when tool_choice='auto'."""
        agent: Agent[None, str] = Agent(openai_responses_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            'Say hello in one word',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=1000),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Say hello in one word',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0593209bf6ea67fc006945ad6d6c2c8195b5e3489f68f680ed',
                            signature='gAAAAABpRa1vf1WqNBNJFvWks_-QlTOf9xNzR4y4pTIYxZ0Rn5n7zdEgxudGluN38yDXTTbJ2SFCsO4fmNL81v9XjfDxXtgRHxHDK5NeRkvRPigjytCkBGwcDr3GkcdQviigWOxP9Cd8bGtLg3_VdsbvsGa9hmi4o28CLXr-vkblkeYsb1hP5kKyK_EJxgJzs8BT8nS4k6LDOnYRrFwEo6BcE0JEWaD-BpavW8CSbW7h5ZLQM12P2KA7JsDQdwGy5E5gC2cXfQP_PqQgx4qLpS1VGP6gvuAWD7VU6ZNm9UpYmUx60SdJhpLANEbECXHQo3-IwDaT-A0y8Kr3_g4IT2viItLvr2_7mFdh58ZALtPXjxZ_V_VAYq0XHx2yVxp47zRlRehxHVYy6GprujETYrL1AbZ4zHEYkD8YTb7tKyicHcyedQfBv1e3xMyprMZ4pq8Pp59trius4nIyRW-PIWm0oDagYZf8h1kpL1sz7iy5fkj6Bc2QZkvmAwjIi7r7TYHNVFHqTJ6xclDcJ6oFXEnfyTSDUIsetovRj1Aeb7B8MC8PosfjyX8rdJJSmgBRNM2WEgIexv_54W-SD-d7UIJbpvBBTWl2jc8wtp_xz5C4K_ZneYoRY3LZX0-JzPwBOm-XgaQONNdan1FDQ-GsCUptIt7pGlAiW4Vt7iU6-AwpgPFrkNjd4CJ8AWeQEjUdI_kynkmB9AfdMWSOjMQZTHw_5NOVkyy8MeUrVIwRs4E2nhUghVZggIhn25ELX2ZVWDEPWaVB7Kj1OsuiyfKyNNs9kAPz-mMNg5WsTui7DEAuTIldimcJ4yUCFu1BhJjwCyeRgyLxddJG_QKR-J33qf5gfZeGm7W6rnTq9GPkqnUOiLvnLJlc8iEGeyQt1e1Vxp0NAAVTXyQaNOnwUJ8woOkctIdZ28fk1cR4ZN86q9DztVuZWSShDaecohL0AnnfaqgC3HUkjm24qmT9eKqCIL_bwct42-VVHe6h4L0yCY78yfprvJ_8Vdh0wxT2UmfhN-LyziOBwPMv1mpZvja0sH1mpkTEpnzU7fyH2bRfw8O-BNuZZzwzpei5oNg9RUFW2RyTX6bSX0IxC1hN2RInTMrqM7Tffb4p18Zrg8MGXFodC40vp-Sw5YMj-j5Fq12xfVBcOlV_9UY_yBGAX1Pgl2Ve5PruxCt1a7POFe1O_8c2mhvUSRsh0xVOcpTSfKQP9nw1OqG4ADX2UaoKEhnysPXKox0KVO_2ma98-ApElQDbygGyV5Z8o_bTe52_1fMMy2A8C_tk3yR3ZZEZvTxpY28cCHZ_dLVrwPcQQirnNwLQd8V2I0SUqLNHeuMHmVmrtx0JMlXdRtpmcmWrhvngWYQjAtca9zTTbX1bfI51Amh0MEHU8LrvH49L1AOC6aZW4czJWqt7IgWLMF2asZNRvmTsrfeaSYUj0XMX-mhXKYkCfTAhbIh9M24=',
                            provider_name='openai',
                        ),
                        TextPart(content='Hello', id='msg_0593209bf6ea67fc006945ad6ec3188195955ff420ad7c6518'),
                    ],
                    usage=RequestUsage(input_tokens=49, output_tokens=71, details={'reasoning_tokens': 64}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 21, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0593209bf6ea67fc006945ad6d108081959596a5aca76c4888',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_defaults_to_auto_behavior(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """When tool_choice is not set (None), behaves like 'auto'."""
        agent: Agent[None, str] = Agent(openai_responses_model, tools=[get_weather])

        result = await agent.run(
            "What's the weather in London?",
            usage_limits=UsageLimits(output_tokens_limit=1000),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="What's the weather in London?",
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0debae64eefe64ca006945ad6fb554819d860cda2b51f59f38',
                            signature='gAAAAABpRa1x3D_OMH5aIi99BcoAgI5l1dr_FvNsUknds-32L3NF9hcMPS4Z1MUNCK6Ai4QXxKHXkmFdR1_qUP8EA4kJL3K4nYeTzyq1PvKba6bKV0hP2-388e0Vi7Ct2ndZfPAza2NXGqufkJGynTukPy_-WpV2OKaCvHWUF9eUfhISmvO5mRhTiEJvQsa0WofdjNidLY7Fh1eLgtrR4n3ER0y7VjYfwAzysjXDDF3u9Irz4fJ5csMlYSoIYSqEutRjTIHtSZFtjvdCLN4GyXIDa0qNG1ZumEaJUf55KapmPxlbBc1uyWN81C9PmZtAVioJwGX9E_t8flD87V2J0rVks_z9Cq7V4O-mLSD3g6ZfzNBJn9lWRHKM99SdQTanZ5Q6Zy9qUAsOyQW4pqxNHe1NCcM8LhjEif4aLtB4o7muaGkZlO6lfAjueaMRNxfnVq91m-D-xPaFUwne1a5AzyqFMrPBSG8PjCu2x9Uv8qQbGMitqwZTcK1sZZRy5FDj2a39d4t0cnSL5FbG6BPUM1S5IkEz4yhspUumXw3tAWtZs7gVjK9Z0HvqRogQz7hQOTe5kYNhsbHMMx2McalCH8yD_8Qp_ds1G747F49Myb5GrmP8vbruVh3DU6LNKCkKcXWrSKISd-694MtjbhNZNeio6aSTo-kLwyUGMPRQ6k-MUJrf_kek-cbK6pswIzvAq-975hdtyFOgRXmM3-gu6mJPT1FZf4BlEeKYyhXXLh-wL97pL4TfEe52MLMFgHuwXHR-MtawEeUR9lojgYp141Nl5YmTn9ec0GuzNN7hFErnOStmt8jQPlgHOy9ajwmHhrKDZ-16PnbRgJkGjNCw3WloBVGuEm_IcRJHrNR117m_BCaFrl22T6qHT_JVbMv0JevxPYOgmXbKbo3WtA7A6URKu3f_q_R6gYrdfk61fq526Ueqse6PjRfEKy8hNs5yRw94xVOdhZYfuNH1_rIQE_yVdHbvc5gu1R5ZXvqvk363gdZrnAYqaHy-9DvtwMid99jbe6SfKbZGYJcu92v7pz4faKrBvZkOriacHOkMz0Gst4Fkuh5GVkEvJN54BZVSeOUJNR3-PEHE58yny1vZ8MvQcr9MSRCA1ZqQVd6cs9tw5T1-rX3D-cqDYA0610CW1_WTH2zByQQYjjK04JlkrnTkRFBij0AONXvRnPvflYJPDB3wP_LNu1vOW9St03HHXBB8DH1g7DWiQnKjB_-Y6CBZCuiyvn09WWOV4Upcx_zObLZM9wgYYKpwt8E6CBvjsirRtzA5Am9LT9TJ5KU1IqxJh8vNPhKwV9m72ltY_y7u3S64F_8hsQsGSKM0yLgkPRLCfW3PkxkksNOv3lt4ynugdhWbhTKTpxltlZlGwJBuMklqNVxorcdothfa00Z2WR9BziV3A4J9',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"London"}',
                            tool_call_id='call_RHlw05OaMVowjRif4li7Cixu',
                            id='fc_0debae64eefe64ca006945ad71478c819d883def68dc0b7387',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=84, details={'reasoning_tokens': 64}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 23, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0debae64eefe64ca006945ad6f6d08819d9173972acf734cb6',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in London',
                            tool_call_id='call_RHlw05OaMVowjRif4li7Cixu',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="It's sunny in London and about 22°C.",
                            id='msg_0debae64eefe64ca006945ad727d3c819daf303010091b94e4',
                        )
                    ],
                    usage=RequestUsage(input_tokens=176, output_tokens=14, details={'reasoning_tokens': 0}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 25, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0debae64eefe64ca006945ad71dd40819dab36d33878d66f07',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_auto_with_structured_output(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model uses output tool when tool_choice='auto' with structured output."""
        agent: Agent[None, CityInfo] = Agent(openai_responses_model, output_type=CityInfo, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run('Get weather for Tokyo and summarize', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get weather for Tokyo and summarize',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_04d4446a114c6f4a006945ad73e0cc8192aa6ae0d717242cda',
                            signature='gAAAAABpRa12tssRxVQKkhlilPudBgRz0O5uXQPcp15Ya7HOMevn1Mdg_NELXKqjBFvTAglNgvYmm84ux6fQy3JPTuMPRUBdD0WmxW50Di6OBRHfmZf_NT508vdI9BDtABpTf1BjGDI2GUX8U68_gPC0LrKIcENzAjtfsbinYMZdnf5NYnIpyBr1QRSLdUf7NwxjUZIj2T3enJRxs9J6Exkxp1eQtBQkOgDCbPUW6hEGF8wi84QHYop2w1gBMwVf-AJ4JIYo5ptr8fCxqCsFGiOCDEPLP3cILO3B4cS4CQDMyew9VqEgfTxP1DWgdSE2_p03vxpE4s23Q3XuMdkWXkLq8bewffRa97NB6jvHqSLfH2YueP_pO0tf_bRl8ZQkWc8QKOk6smpu4eWVETshP8VfOQqOWk6G9SipBlHghAjfxWcVJ9tigiNRChDcQzZUhIaM6CQMHsYqahsRBJI5bLL3Dc_Tclafk4IN0V5ms2WurhK1khxL-6QaMEmCidOajsDAEczrWnqjbyE6yy0MynQT-AqeMwcVY-f_2nNjUX0kgS5aooqioSyfNYHXU7Xseit97R0vwimHq4yr2ZtXHnA7qmnZTdZf-u5fwWq0p2SAAlsNwkYaW0WLIAN1aqsqRVia-wZbToxaRjn6klRESAZiNwg5aEUlZVe7GDL0PGu8MqV_laTuIEsxTA8Mzu39oTiPN0w4FeapW5bFVoji88q1RVmr9dPS_5qM2uX4FVX2h7abFbEv9sA13AL4ORvALeXPIKvdWeCqs3qoEGuIYX7kxvDhh_5e7NH8ia_klDKXKksh9J55K1WfqL3YvKOcDa5nNX9Twe3NY5767V9kWQUy1UYWJ4Vh6xDWQZttH-ABnr2Jqk-FdbJWvpJiiIyqYxftt-ykZui3sjN2ebuQohUgQOntEafIe4tdYJioa_EPY9ZzHkBhSVM2A_abNDch70dOgpHCvZCbrp8vHCsVUbQz1fR9GV1ORfhY2LP6GZi2DAF4q-cKEUwsuXAD4SClBj43KIpb3XCfDr0tJvhJhxYW6QDX7KoDcLYEcQiIsf9XK7HKFfbF3q_FqM4QjrSwnEmtY_kE1D5UyBmtdNyeRads4WlnkQI1eIH3GtRgYE0D7oHFxrzbY2YleVP51MqkT_aM81CTmw8fuLr3JBaJfm-SzmN8xTtgzOqgh-gA6CFJ9pjM8_Xwzyqv8qcZt3E_JsgDbe0q-cl-1NIASsLk02OZpQRVjTOF3iEbiVkhjC9CjRaO_7fPhA8WZ9xuMjDomVDJrWVMjGl95cEN84MbJvvzsdBKz02taubyqoeYaYAsCbTUXJN3IzlI_dKxqcZTVladhhUaZnitaiqoyG_4trX5o0sDicEyu0ZjxR5ulS1gku8htBilMpw6pKWB-2kVsYgLxK_FLMHwxAutDNwLFsdOtjk3nvET87tFllrAPUvGMyxEaIbSCRQfRN2F8ffk_ypTup4q1iT6VzxtpkQREtBNaWVKoTX10ngjYtJGRTXmpH9tlewQ5Nj8_xnHOfQeH58FBncQSBOX3i4znWmY_c8DESMmjiIs-WqYGFr50iTDv0errbMFKZFPHiAhbv3f7hsA-hugUQDFsEuw911vgD2SM6PX0dg6gFIH_jG6sRhxjbefzxUPOvHmIdd0tPrxEzJJuEmwoBrHd9m2yMCUbKD4CXeMblp6xIEIoEEq7KKZ3MBwZZEXmuJ9DQGDLbsABaPRHK-MISU4wzo3uzgO_Bh4UUaLooC7on1J5q-3qA8G8ep8YEP6Y655KLoNJK41NnPbFkz8QkgtSAyFrBm_N4w1yrhkpGCGid5XYX0kA5bLtk7mHiNTCDjl4FrpjA0p9C4bRR92cPYbf-saGUow8HDV7jb-ClxphpPerTQBH83wy7uCUG76wg7Vavws-dlL5azM9QrkZXTcBWig4eYi6hgDEq57fVqKiAHu7Zkmd0WzKCdJQDM_ILx3K69kWyuYTXE3_Fzwa2zLvQNj0pd9zfXxuZ_4Q9Ve9fID_NTWk1EYiYYExoFrS7iqvFgJbIvKujEPNhh8DqI_Dpj3g8wACbd0r7Fhvw1SaveE49iu6FC3l_iwCiQLXyhsMkq42abXsowQbAI-0L_a',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Tokyo"}',
                            tool_call_id='call_feBPBYdjmBHdDh4zMVgTUCg7',
                            id='fc_04d4446a114c6f4a006945ad765d9c81929c7d63cce482bab1',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=73, output_tokens=148, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 27, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_04d4446a114c6f4a006945ad73674c8192b1e5d9a2edda2808',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Tokyo',
                            tool_call_id='call_feBPBYdjmBHdDh4zMVgTUCg7',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_04d4446a114c6f4a006945ad7768c881928bd69a27d03681e7',
                            signature='gAAAAABpRa18lZTQE2XJ57ZOAd9a3BMOcWtmwQGsBo1PbuZtSlr9nbKi1_-G0v5f5pH13Ki3HWyByP3Y5qcvs53b9JhqtwHZI_9MR2psAjzN5uZeFfwc4-17YQg5ugjMpy8Axjt_igclvpeUQ-yWFGIznB07bCpWelI9JA0ETZUkBgdUWTzSTREpWiXR77AwOb31A3--10GjrBTIwQauwdfpouvxKc878P6IZ3K1Wl6GPQhOBINxGgX03LwAhoT76Vp7re7LxuWYnhCZZPF85lAsMHJHsIH3FkTtkrwdpbf3sumcTu9_WB22lRuOj-EJBqlgj79FvGBhyFPpSBBIIeDxpI3gzOIwzt5eq1H1xb7tCksVu2dKxZd2CmOXXqcIekaVKQ_DkBnz9QYTMPq1oiIIUrmg4WkZ7JX3HtmI1cANDusnl8tdptYRKMkECQvn1iX_p0Nc-3CTITNPFL3g-hZ5-x2CIaI8hDWePKg0o-g8btuDOt_NCeV_H9WJ8wstMSeaMKb8PGzReG2iY1Rndgs1GYH9XIrakLeUMNozf52Xmrkszean1VGyDGaZFKYzP4DdWLGPXV_SZ0SI-xbhLYaV0ZoJupnl-1prqZLsQC4nIBm0157Rq9ybwJqurfqjU6KbOwz3yCYT_b_x5y5XyJ2npUxBaP-9MZl1ZlLCU-YmYVfMMDiPwTCR8N34pnc7tFpS89Jlp-db_Yi3Y9c8JeWN3uGZlHcZ78vkMYWQ7f-XDOY_91YCMz2ofL0ItiwFAxryvqMJqVvqY427IubXZ6l_DNZhRSqGR2f6zXyi8al2hE5HCvezxT8O6EkrnDhfA0HeD8bl6Irnwu9SFup8qhtdCR9SGsPMQQLPUfGFdLhxQrQxL-_WPuTK1Z2m28WhuBkUbkKfI-DYg04-PUe3Y_9nvsdfVInBhc5698N43QQ45eP-KIkx0J8UPoLBOufI313yvcJrKAUJaCaoKVKxCjxYJAPYr4Cl4KoL_WmS8jp6VwcDKSfvrpyvAZ8X6V5rAQgIa7tlZVkBHYuY0RakX1EOeqcyhvDd-dLToB5WiK8Jp-B37mvRwOnb-JTknciv99duQ1el335zGIB5Ilq0vuivfaShPPp-3FzWr_XM_yUZLYXWx9j-p0sW4mM8dxW4chV6c9TK7SVR2sgbYd0KCHNY2HrkzrecW2P5nedxsKGdsiDEHvWLonttCMS8ocJRPR73rCyXU4WZinWelgKxvusJzL7V5q5FJqYuwUByUV-FY-W-0F2-y69ok-YYjkwhc48AYLRi6tHLJdir5LkX4Uae7c-qR-76B9eTIZDWPygguShgTdqrOTrbfyqe6LcTywhjo9-qUJIyUe7-eBw3xndFMK9Klngvrl-Nehs9m-hrO-HSR1xsvu5ACwR5JTSs6xTdzOG8kuNgp6-ZPJR78uPDVaOh2KOwOKldpT-Vn1yKfFhl0Xz6EFFzjU545CfywPcMXLM2WEmcq3M6qqcBTZRV5lC01Xw1aQxmhggSu1WAiX-O4yQnKN3faqsET-eGFnlZIuivMyHGXRFYHdCir28LYtE4SIUv-l6OSaf9h7DDAXoTcY5P5YiSKVTV51nLaqd0scBFjd8lUCrsS_Fmr85t7YK8XvVHquZ6YpHW0LhHuqEyMAH7aZwXjI31lydJf27uDWXVjB671Q0j5V4mSwNLokfUQBbz9s_9cgn5NKloGzUmgvDosnr9i7OslRwBLOAVRHDqBYymsfG340VAItZ4oJ_NjpdpON6PWqy3R0CWLx9wwfDgoibjdosyqtX18uuKwMpr_DBl0pNzA9dV4Wc6R3iA-1LkjxPQKB29_sBpUmCoNQRgZcl3o2nmcd7kkyBy39bFBmtBSPekS5N8h8Vv32Jr1eGutrzR01jkk5cpT066xxJe-brgs8mdyeqkMryLWYmlHtExw4TUQYg2lE4_uEVx5bs6fZFp_5MUrSrIUVL38lI9_51lqqhFFelRsC4U-uzx__KZbE4w-wYn90-06Caikty7VVuEFDKtEuHwX30NtSKki_Va7qCgXLM2yRXpmK_Qe31o9Gyph2J3QJ6_KgFGPg2-bsvkeRM9iCWF-ySNIbMFvhnko0Z0yDM0XXE8IFdHOGUtZo7vDPLAMPyl8aRN10RMSVVDOIDtEwh_pMzJghzdXLwC1v35YieeQ_Oze5LjlhOdFD6SbOLtcPPscZUPav6ksK9h2ej7myzAyG56gyczd4kmx2wUgYdVBuNmShaMwaoB7itcihdLg7sY39bhPcnZf80QZ2fU25Yd8swHTnMDvP1Gy7aQJaX6-YK0Z4sBa3pipHS9YKrs30P_jn6CH8mVCrcrOgKOhFqrkIs7wLQG6oaduNRF6FwnDbUovB-G_BvMiHAGGgYOSJsbtyqlsvat9LilhSZq4M1YKGazgw_FpBeQP3ttZV3iPExaMMg9865rjsJ5B7ol9iplPj7FvpdBuvvDAUJMmhAVdjzCjS1WrELDRm1I3_KQ69s0AwriQKG4ynv39GngaErY6ML6f_Hbi4dSxo0p0E8KQC05C4qqCIwmgmtiBiyxn-zIEahERRfxKM8zQP2rtr9vSuizJW4slIkjTqtOdpbDKM50We22EkZCmvbCTsA3P0kZrAnRjFCmiku3InpgdAQHRxJCyFOhuw3SsHlBq8AgiRJd1qgWozlhKpJRx3l68uRmk1tLqbQWJLyhzKRbsuNmFZOwhbVwiMmWgr3zBPmn8v9GwE7YQfM=',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Tokyo","summary":"Current weather in Tokyo: Sunny, 22°C."}',
                            tool_call_id='call_4kZueRSDVZZfXi0xGLr0cGb6',
                            id='fc_04d4446a114c6f4a006945ad7be1e8819285a05650a14dec9e',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=295, output_tokens=290, details={'reasoning_tokens': 256}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 54, 31, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_04d4446a114c6f4a006945ad772260819283bf81f129520f37',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_4kZueRSDVZZfXi0xGLr0cGb6',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )


class TestToolChoiceNone:
    """Tests for tool_choice='none' and tool_choice=[].

    When tool_choice is 'none' or [], function tools are disabled but output tools remain.
    """

    async def test_none_prevents_function_tool_calls(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model responds with text when tool_choice='none', even with tools available."""
        agent: Agent[None, str] = Agent(openai_responses_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        result = await agent.run(
            "What's the weather in Berlin?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=1000),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="What's the weather in Berlin?",
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_039a5882b50f882b006945afe42624819192ba6b0b7c201b70',
                            signature='gAAAAABpRa_1bzlTZWcADyX8yPJgEqa299P0iHgxIKIrjmqgOxuWmknGIEmRkfJL5hvTIaMGhWy3TB7N5ZAJiy05k8N-Cgi4KOIJSnR6NOwyrF2qfWKrblFtmBaxloM4KSKlfudiBEkzJFg5OuMcNBPqnvO_ya7iw4veWnDPdnJ9FKiZUsYAYMC6JOzKolAnVklYQlLrLT3yIrblt_S7C1SQyN0EENZsA0yopC0bBF6WBv61_4xTFh24PMWiNfx4pibEs6eFNegArtD3QtaFsx-aY3hsz9DxNtMD0fOGMp8BLRoie69YMJBBd-cstwWaf6kCK1H2pDkB5vee8LSpUkuCjgwS13tMg0YxLAnvJlF98g6cYLnMwIQEDdFO7kpKifATyebXSedZmmN2h6jdYeJdwuBgOMib2L5XyX4R3ViXOXvUgiZ7Z13ZIUdbVMSWelYTsDHiHLhJuho6e9n2MwAMuivOl1qlMAdFDe7A_4EaK_SLlbpr89njkqlDZ47IfRdmWt9-0Y5aCZmr0fqsfVy621sWXoNf6b-3ZB3PKA6vls-p_qa8yvJduXVphhitklPoxqW4I2bMqzpWiNwMNLkfTmRB5Vq6lsW7BROaU3w_M46nLEK45_Jgku9y0EwVbRFMUHCRbYRlJedXwn1yJf1-5hUXpUqxFuZmyiKVUzcWVggbXhyqhCPWnJxqQCyDbggqQypFvOd1YgZdifh1TPPkv1e8kfHiuhOfX3SDeGTdGN11MMOXB8DKJeULtBoK85U304kq_9qvwFihIPaTp6ynnW5wSZlZIvNUD2Bb5ZiMzYioE25hRuZKps2FJGHFforKn0yrVNbtEGaQY1Cv1VgpsaNbWi3R4ecHNi5WWLmXKPBkEPtAVv1_WYmPtA7dEKle5X280NkiXIRgSAtWE-kPhDEyQzMVL2M6m-24aWKOaKi8-LdXGqfOdUaZJfaPj9jw0KAHpQslUgFDHUO6Ywv28PLFRqwcnVY8j10EIQfhTWBGU8zISwSVdPx4dYIUs00-WNA26M3W92D8wxM2riB9mMkIvBJH1z_Nbykts1vY_OcSy9BnXOB22hcPG2XoXVMU33XyXPzSSSp21ZKr1AK0ZtWIofHUReQA8H5FDlQ1O2nxySUC4m9H-K1SihoZ4vOyaSCyBgB3CG2VNgT6sivDrEAkCBdKVMPiKMI3XcMBEkFMId_PTmFrbX4fnwkuyQ2RNKc2mokEMi6qt14hZ8lBE6hzbsLMdfDyKM__6AfT7PayFAos9vruUp1z_DozLlpxlYNfi7p9J6aDTz5d09cMMx_qHYWe_YGaR1LfmEK9AVCBM-_FyV0nzlb7zoNtYPV-eFeANStxDq_decHja8COEDNL5DgVxwDp-EnEgwPppZSrcwCXE6lwZkGJ-ShmL2Fa-RIEqCkS_5nClQkbtqtf_mtxcWvut_hRYdpclT3PsJl8rJ1-82_9e8DlYNkNowSVgR4GpgGL4uVqUjq25lBk2mrq_4dM4lHhMzpjl1jjwzV2IwS-iwKlyQRKV1VEaZASnflVXBvxFS0aSR8APmovlMmcYlQVIv3jN_5vG4UMwe2O6E2WffcXjG6yzbkzBf8HVVRdIVwYnt98BDA2k-jt09hbmQuzUhThcpFivJQ-1Al4UT4B734sDDlkpn1juOC1eC2RClYspcK8S-S0YsjksvQExgpcamW-QbukDYMBMKVdULeUEt2pri600RDI1ZjXeSoAE7Y_Xowbse9CgiL2O60bQPiduXVC9BD3o7QzW19WQpYi66e8N87XgmaRt6HXkwvZAmTl6NLGNvFWrgIGG_jpjNo7zEI2NEA_7bXg8xVKkAKiYpu5uhjpcNFEYSMSY43CZXK78iDNFGyalnbQlYXrKpxm_1CNlm9adUO0mWdYthVki15OyLwaCl_PouzmJZliQ01S-0KnvBNyf_aLVGZ3-A2E-cHcYwQiNMVblUekZDSIvFFRs_V7V_U24ES3cqIMioSR6Alfp9cWeuzicXrqYv6xZdeOU8N7Q_P3_7JHODL_AhfZlk0K6CIp8W-1uh0hZfUyTh2CMh96_h9RY5rJVZZiQnbQCTPgSd9XyLAlSrqTQYjmM_lLzR9dq-9FchYBKiH1uIBDE0DhiA_D5mefXzWARD-TOPDftO4xGeW_6kOwYTQaRvFdjX6k071q_OckRvscQPtcF3Kw7TD92QJOaPnn8Pi8GBiK1g49LI-Z8qazGTcAi7fgAsVzIROlYTxRJxdGAi96614zjgPUTYM8WQPrXciuiWaqWo3l8It_Mfi4lZ8fRI0y6jBqiUtGRR3S66i4N2BSXZWOzQvrfUpPe2-uMQLNqvbgtLZahQsqqnebegh3PRh0TrXV_wleZpoNdTzmPavCPzaxBU1X-KLy-V2TqMQN2OEYUlh6LLpQ8XmMFSc5-aVHhxSYGacvAiXrMzXasusE84OpzxYrUmxz_l3HHlfvN3NVSkMrpY7TZwzoi2fIMTFCnfIqL4HdzhMiHytMwufPMow3zkJWKsayc3v_-LPXIpEPeL0nq9fUclpPC8zzQ48CnDejkJcS6E2aI0QLLwwUzcd3ZK9KdW4cGtTmDF0SqOOgb5o0JbAH9Siazo1GADI07xAbHFRV3JxjrzhFjsYCbiKUBGCpHUGriXROo2zF9AWdARpCcOsCBebqs-so9xhF00CNQJ4UQt4ssI6TzWxTy0_0wP0f1QK5MXle2RgEVwc052eTFLe8LOueI5kgWn2_1gbT1ty0_UHXIxZfeFzN8J7lRktrV9uT5fdplKqABlu9Wpk4S90dnKoUTqAu9auisjeByRNQjmWPGpRfpTeDT62SwPw73L6tMQaNUWJ7Dp7fL8qNSGOnnM-rojJoF5DNqJNW3GyMVvab3tK5JAJYFV47BVaXGGubheUO-tVM1T4I0ej8a7-kjLEvjGuCWWY42lkELn9dCqIo8LnpLjYXpJLZtFNOXfk5UcWQrhbhqLzRlPOYkt7o-IknzHG4mT4HrW6xnsv1J0DFiH7aCgaKK9fsmM-8Rr-swsysBnq8CKKHhLsrAgjYmJuIa0YUgTWJhw6G7vFxRg95MrlQZc9Bmn3FJk_i9RhnN6Lmg5fuwX5A7c4Ju-TzvsiTYmsOCeOXvLDSapLJSBzsnpTYLWf_rLRESmfRWtOXcC-l9CXMeRQ8eYJvXRXqSXiQPftFZJVwaudkZbMyXxlycwQJAgMFebEwTPKYEnUHU6gNfdEm6GNcTbJo0i_3AA0EQbSYsS6WUAKM_aiw1R6xSOLDZE9sfGAW3ApjeAfEksYTVVz0y2GywR3f-Cyi7uMYbNb-4juCzXOPh0RUcMf_lZlt84iO_RZy3NS-XOtJS0ufBXp7b05Vp3bztJ5lGEUFlHpY9jjjC5eyMVHiZhddxCnYNr8a2crakDHzh_NtlRM38g4O_ykKeSBAZTgc1iWhm9Q7MT-TY_83_AaUPNY1dgErx8x2GKjzHmXJQamrNF9hBJYFtT8Jh2VdfqiR7mfc7o5R7NKaO9mkt5J9mtc51UrSbPllZQ3FludIBPt3PhfRV6HCvBsHpAYxNRWVOgZwlOtXdoKlTq_2SN94AlhG7tZijLuXYlyMutk_a7H202G9sgnSoiNAIA6YH2r66wgNKYm6ipL9EacCHLxTlxZjTyif-iBECpWtc2GcxhZ-4yx0cKgkJnm3mERY3v5BGaST1TIzJOU1BApGunBBoNAWsMIajO7s9L5xwi5yxz1tuic0NkD78ria983vrp5UQ6OVDjpzhJ2taxEV67HjYA1rJ2GPGS3VxsQT-fmBRqaSPTyTgj8FxU4Y_6hnbU2Mk2aXRPkVN3e37Wcqeo6IMRxvy4Yc9BgTGKoOfH8Ne7_KXQzA_5LX909W4yfmBUc-5atn_aXUql7-7VHHw4a1GjZbh72hOS0tt6qLKl933pbTZqfduqkodg6Y640Uv7mZ_LLc1D2SOd67LsW_2pDE7UfPKD9St-n4aC8abkcYgX1bbGUzOlVHaUNg-Z0xEeWAqD0be-DtYiNf_sn-3pNAoMIs5sf2nr7DRhO6FQuDqsRqVmm6mmnaAkjtMnzKqC0pVYwXaekilQqOCXxi1UNiXIGy-TS5tt6Zcp5VpG2FNU0o0BnmDCzcZIHOp9hYGqcFM7q2nT5XI5fTHtuMuS6k0F5gVmcanYuSa6sK4Wkqztmg_U6bm82Xvw1wjIGHVzzYw5I84wfxtcCp7fy5jdAp50kjRRVOJrjczEjjM5jkOpfTxNTPVJr_HKxYj079mGZw0mvzLQaocMGH-9e4nEG2RJm56gxjvOrLFd18nIoNu5sMaqeVP-_-h3evSbtRZVH1Nx9Yf9nrOPwYT8Mb0kCm1D-88VO3e-fDSsym3x0t8-_mWnMfPyubZj9ufYu_dYJXkS83OapkjVwcf5wWtFswYcwVCiW1QnmuVVwWRVtgn7CHs-V-S806F9xKnqLXppr9wkVrDBC5LizYvt0gLBpyP76VNMJ3HWaSmkhcM3UBwRILPtIr9fdaQi8eaACeXXZjayejwpl5B9AIpYx7JzHr949j6NGn5Se2aPBLt7M2cRbGZGhCiwvK6D2Y8EKJZTyGLLmYU2OK_M8IZiC1HkmJe51xL5wPNCg8m6FSqxZ3cKYFhVCva2jwvIys2aIOh6UbCXa1KvkO7QEccTI_KDD9li3s1ygatFkxVw5PxsKSYXuEEPLJnbPI_jGTK85S06Dd_4bpCqLeZexTEC46JxokXPFtAHOaDQbtc-Me60695ngNC7wE_bPnIXtkrrSUq05YPdnqG14EHOCB3XiJbCu1KOohrIFMOOvIYuRBA4D4ofMzCT_pO4QRG-x8n8ENctZd5CpjWVMU2cy59t8BqGnHc6tlII85wT8ZcuwTG5JDIX-QkBO5Wg==',
                            provider_name='openai',
                        ),
                        TextPart(
                            content="""\
Sorry—I can’t fetch live weather data at the moment. I can, however, help with general conditions and how to get an up-to-the-minute forecast.

Quick summary for Berlin in late December:
- Typical temperatures: around 0–5 °C (low often near or below freezing at night).
- Conditions: often cloudy, damp, with rain, sleet or occasional snow; it can be windy.
- Time zone: Central European Time (CET, UTC+1) at this time of year.
- Practical advice: wear warm layers, a waterproof outer layer and closed shoes; keep a hat/gloves handy.

How to get the current weather now:
- Check your phone’s weather app or search “Berlin weather” in a web search.
- Trusted sources: the German Weather Service (DWD), Weather.com, BBC Weather, or AccuWeather.

If you want, tell me whether you need the current temperature, an hourly forecast, or a 7-day outlook and I’ll fetch the exact live data for you if you’d like me to.\
""",
                            id='msg_039a5882b50f882b006945afedd1cc819181c805111b252ba8',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=862, details={'reasoning_tokens': 640}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 20, 4, 51, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_039a5882b50f882b006945afe3de88819189fb0f06961ef7fc',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_empty_list_same_as_none(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Empty list [] behaves the same as 'none'."""
        agent: Agent[None, str] = Agent(openai_responses_model, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': []}

        result = await agent.run(
            "What's the weather in Rome?",
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=1000),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="What's the weather in Rome?",
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0362cfe909ecac94006945aff6c3288192bf7f0cb8bf75a1fa',
                            signature='gAAAAABpRa__-6dJXTKqLMHpj6cPVV-adhzweoPIcrSHYYCjUZYySGEUDp8f1dFEqz5z6b_5Rpf4nmvQrQ9uglZe9Dtg_QYvBRn-42UhbPJD-eozg9p_URuYj1JidurtI2rfvNx40ft4uYX_cf8e0RATTOvjFM4rm9HWBI4vdcRUoReUcCE3Nh-hI3Bdk5ShOyoT9Pcqba_Mrlp3t2Sik_TUArIIckyC9HQHEjw5QyPcHRZDYXqnVzfasEX63aD3D89BGnzK03BQLH4I2HMvU732pYmsulzYPO3gQYXxeRinLRFnQqpga7iEqfCOBjWlpvLLo8XfKxPnjn900Wl0u2aD2DWrYvA4XndyzmegosB4k9Q2rpgKBSzw4AbR41IxA7bBMhPexEuG0WVlNr3PEe5F0eTfTNP27nR8h5BxyUxi5iChiC5UWQ7P19NjKgDmq-kJvFtp1lbJMUG482XzLbu1wuPQO61ZNhMTaIwHmaBjgP6R0C3EoJvbgiMcGWz5zmmljUIjqDBouMJDHqLtWQRXSBDH3zL1OaZNv5NmnYhG6cTv2KMcwkF0fjGA7JgUjZmjm_wh5ri3KXTtDjLUbege1WDIkJJwC7qAeek3drL9v0ZfKsOvwbJaOr9AxuhbIh0EcJ9DeZOfgwBn-C5NOA0LH_KwYXNXivnK8xKMQWS52sg1a1cBN80aoClkD9pmj9h2xbLf7EcGvePzhREJd4tlhydWlo4sxZPFHG2nY6ky5-fNgMUqMJsSqwr_AtAMM_uUTSDe4eG_uhQU4ofxW_LaB1fLGOx6rr8xvb18bfBRb4_0E7lDohM8c8oqA9BrK5Ux6UbH-XIaiQyZUhUZugSS3DGN-L6z2ZeHhP5VE0lR8ft6h4519a-UTkqaXHUTClERi4bfElRzG94F86uw-1hH7SkKzaVpeAWmndYj8cJFK9pB659TSliKzr6CYUrkU8xlSeD5R8NpL3M_PP_I8uATYBAaLhtKG0tG5trEXzNxHnX5BA5Qd-SSJLan5j71RWUonwF_h1G_8retoUinef-Cq0sOBBjR_CQ1yxfJ5azDr0rIH5FTmgJMejBOfB15gMjZhfrBBivs6d5PAAU1VGwELKCdKjwbrCBx8E1vEYwyDylcltfPas3RjObECy36BoIPFS92v4NDTLu_dJZFIzpZdPc69TF4KWOX4l49uyb8v_k3pFhfANmFoH2Jhzrm_ossdUfMVEJzgTAW_G_TV3AIXaq3n8hEsJ24ufjPqzw7sbvXxoTmT-MM37Eqj_nigO_iUhktwZjIgI0cdbe9l2lldnlnxQc7ivQwYEVyVfTwP4SP-MTVU0JfgMaYq1qZvpYACPPlddtMRpkG73M1eFZZSKaGSELzKMOV_5Q0EFI-QeiNOea9fHnXR8tmAXJrXr6smafJZBB3Am63s4eXiiGdYB76mMJyvUssPEADcDHunOI3_a7KF3k8nytlTZp9P4WM1UJLMkuuOrsryKy3gNW_odnbKIm31aeNW_p98Fd_J9FtpptEfnlONSJ2I8BhJG203lenLJslWuqerEzFBt4LXVF4gIzOgnhADblbR_6bRhcgSVdWJWRV-GBAMjov4w8xGXr8p_sWKxmGfBsOkY0twgHpnv6gzmUnd2HSDC93qQEE3CdOT3w2a2fzsaUpCy2Rh0yXl4NL7Lhkb_g_vbb-tFIK4GtB778kz1dkgW9yFJ0sx8-zklBqGmZdBF_vEJ8p6HPLWnieEXJS_lOXmRVIp8lbbiiqpG0ZJXOOosQNPCYzfkBoM1czvB2Vu65jD65spqgq533-VtE93nsiSn2rmaUbkIMi4rnPy7Yy434VvHJeR_6woWT1HvtDtRVJ1N3ZkfiBv7QHpH16hqb-jgQq8-OREM3LoMYDYGHugBJo9fMfLN6ZC3OWEH8EU-ySA66mD1sNRL2pH_SKzkUfv47gSwVwyUdkmkkVfA912IL-pW_TSTorcx1IWppTSPI5zqaf9A9lYp5EZCZS265qtysqVzBpIzwNWwBitqs4FxSs-47gSgmWDNHVtoqUusK90H71_QfuwW3cHMPM5JrgbjN1KVzm7TrFburR1Er6rW1UOJos3K6UShKlQLruEwQjsolp-2ysvTGN25pol9wTk-9XAfPC7ZZfcYSYCx5WEYHe1wKFsDR20mhQYGzBKL10NJSgIEE2GlQWHy9kNk3_L1a3bL2DYQAEqevoeNzw-9Gr6t4bG01BAmlLcgEzcvq_CVZ2K3JtPgaxMsqlR3_nxKbLbEb_z5usvZjs6SdhquxhdIDp_a3j1hoEZr6bab9sog3uYAMD6k23ruv9WAVDc20_oewEj8RZ8AB_5em9hzPFWVSO1zPW26nLcUyXzJzhMvHeNld1DTsHx-w3rIpKsb37xVKursvHmBGwIod73Tm9Mg2mTLJkuny3QvOrAdEKsCeRZG1kfZ6gd_WwMqpv3fl7x7YA4-Ruipar96tV2vkhBOYo2QpZqY8zMLp27Bf1pqABGN_Y38hubFpDoSwdExcm-wjn4IQdDrADfUjIJoem4AzCG6P9FrxXy5kJY9jUT5ZRzTymZD8Vkzk9MMk1D3IrptvTV2Okn0wyEOnEaK3uBwume0biu7Sce7-r4hbcpsstVPnyfaawJDs3MShmUdcScvevQX-zDG-v7t5J_0UpIK7cU9KIkWp-1PNcRzrJ9-B-y0jnp7C0cxzWmnHn-taTdZNGu2I6lKBw4Yr2tsmPidr5iwoo7-tX5vJww9qQ3sJu4e3hWAPqvvbLUYxpcTMQdcm6XDeWn0p9uHIJ7mQPMnvqKDsxubRS6uDUb6Yn-sHi2Tg7SOpjRlSQYc_8bGBgNhlSlPqlTnMmRKFyQnVzc3QL77BWNZmHuORslPy60_S_3kEUAAnw-Pw4L7FikV1TDluSXVRahA3ky1KdkwPyJQewORKNBb6XGFYzqWVEC2W-ouWUzbHEiI6yCeoEZaK8lCKSsfVIMNHE6zxB4snTiUu8hhfWF5GIRnReFR5ttGNkuAnLgEWU-rD9Y-kpFuOjM3mqNH9aY_UHoh1n2BZAhHN7Lr7e8d_qEVlTSCn7i7YnAkJ310ngEKlBqAZGpDGvOlA4rLQcEhnpdvbKvcAVcDYkzP3dZ1XJnVvo0nksjWXwAW_8fsiGY0KIuYvtcKNp_BCERLexJNOrEh6fRDUtKmxSR-ka4nmDKWHQSN4oSxZaqTZ8EMn1E6hg_0vI4skN9MQvTXNgrsSVdR1qEylzfYOjM-DO3V_vI4cR_nu4G3-ehQlUarEF625wNJp_Bxo7fyp7QJ4pgxFimX0tQ8rdnz8OWMLL8D3GJfGop_Bp6LR6RbJMHH5Z0e1MXvI_OZ_fyjlpD9NYdX88qmJCqWKi-1mO4M4Ne1vN0xKUqOcFvVDClZrJAIdmEWK0SDiQMrZ248xtQdM8mIFIuzxioD20QgG8pO57OwOJA8zpZJOjmNNMrRujChH4hAa774-n2xc3J8L2sC9KKBSSBMuAtb07KNSv85j1sG99Ik72L1oF7VzAPFrFt5jyvEYzLYsKs_1t-K6_icLYFubMcuT4EbI04WhDgGIYqWZEXfQI7uk1A-OfGPA3O9HzeIPKrrs5TxS5L5I-SUXv4WgqNI1iaz-uoVm3An2gNDIrJPU2gKkvOhxw2pLGKwPJkldMst_6aG317IIqjC669e_6_HK9DR7ceBOxtevnBdH9aukPbTq3LIt3NmPNDeHAJD6qj1jE3nCZqVzrKkQw_RN8n9fyUyDRqeIXk-KxfTcAYFOCz9ib9Rst0pLpusDXIDcXimsJzVMbsB08SC8obhUGUZGqC5v-zKieLbmkStOvt-v0H3m3ijogzB_8_WK7LJwziTomVSLEVYlPSR9diDg8d9FlR3wc7sn3_XKSOyQic4tAdHd7lrqt1PGv7RFKe4mY2En7rmtghTljxVuD7VKaVCzk2ZP5-WYZtVpCj4rX_9VfcPWxNqRNWP7VfvxlsqXiRGCTkxdgPqtHuP52YRNe30xeMimbz6UmgZsroWyKxRyblFCN_136wQAybUvVPBgjd_hVQKYO-s9MfC4KxG7UcBMsCM6HCO-KOn-QnJt_FQ3QtSJm-SzojrQb6Ynx1w2rCKiph_k2cRoh_LwVNX9TD_ttY_KlYdJz1hFDZVeastZ_OJoO6vaVL0TwnxplgNbZumGy--ZvrpfNi4xLFeMLW682YVpIQ-S658CRAcL-9VS8z4mLeWJrzWgk4P9-i2AAKm-_X8E65SHejcTHzmXoQrliSrRxXc6RCxgIkzpYyWrt9XDWGCEmikPqRcFccRT1O5QWfT3Z1R8OsHIxJ-HfvcJTuuZuAo4JLrubY5UDPBFiTLG9zYKsTh3jjNJxdpNxSeKnEDyYm1lx0lnxPeAqymwNKBUqPAMIHpwcBbhECfz332QWHM_E7n3ivafSbhY3RkvYlBAxEL9TLXNWV8Sb2Jg84NNmSzyObaPfpgXFvG0JyHHD-41iimqDCIz4Gf5aW622hqKjbHOd7V-HFFKMgfSDyM0exw3xT0D8y_4RtvzmhHoYeR7uw2WxqUTdOYqfNYzOq4LhUuMb5K7TvPZYy2Q6bRRChU1EtXwYhqd3b3_d6BQ-sGW1h5sR-JNPMrp2I_bMflac4QXIscpSpRO5S5yfR6B6DCyPwdm1G0cWEPQrenVskvE7U5TmLQhycJOSxLIfhnOGGAJ8gf7igg==',
                            provider_name='openai',
                        ),
                        TextPart(
                            content="""\
I can’t fetch live weather right now, but I can tell you typical conditions for Rome this time of year and how to get a current forecast.

Right now (mid‑December) Rome’s usual climate:
- Average high: about 12–14°C (54–57°F)  \n\
- Average low: about 4–7°C (39–45°F)  \n\
- Rain: December often has several rainy days; expect occasional showers or overcast skies  \n\
- Wind: generally light to moderate; occasional chilly breezes or cold fronts

Clothing/packing tip: layers (sweater + light coat), a waterproof jacket or umbrella, and a scarf for cooler nights.

If you want the actual current conditions or an hourly forecast, open your phone’s weather app or search “Rome weather” on the web. If you paste the current temperature/forecast you find here, I can help interpret it and give recommendations. Do you want that?\
""",
                            id='msg_0362cfe909ecac94006945affd7d708192b50284bba0b1d2df',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=774, details={'reasoning_tokens': 576}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 20, 5, 10, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0362cfe909ecac94006945aff6544c8192a2170f3cb40cf6f1',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_structured_output_still_uses_output_tool(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Output tools are still available when tool_choice='none' with structured output."""
        agent: Agent[None, CityInfo] = Agent(openai_responses_model, output_type=CityInfo, tools=[get_weather])
        settings: ModelSettings = {'tool_choice': 'none'}

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = await agent.run('Tell me about Madrid', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Tell me about Madrid',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0f49e2c2e5f4f700006945ad9b7e608193907cfc65989f5e82',
                            signature='gAAAAABpRa2mMxGK3tawIijrznKCmIcLmTMxDYNTAnUfp4eNf5hqtTjviEdOqCAL8FDErcS4s0sORbtYVV2ZBmMKfQm_xmIFR5GrOe7aiBSi1rPFupdH983N5YtCa5LSEUufcpIWjo-laRpAFpt8f69PrMDXx1FUw_GynN1kEFSAhgXC6RyKjUirI9XMSrACxDo145OHiYAwE9V5VfYd09AsCpMEaXi6I34gCaWVVjC_XLsa-krAuCI-MGUgiyLEGSnnear5DFmYXf7VLvTLlYXVpT3jzXMKAOOG1Tb5QqLrgXAGlRrkwwDSBM_MtGdUu161I9uYzK5n9VbKWSiw7aDt4Bmmhcevx1wP4dU8461Ln0__TEZst9UT0iBJ0JU8Uz9xMUte6oSmNGVGw1tDnRoQhmio-umQYdF8WfELdI1D6DRNUAOee2caYM6L9wmAPTJdjLqgGZ_93hqo-8RtTKUkjz19xUWXtxNuRJm4GGCditteDOGx7jqneRDcbnyGK9aZ3vpUHc5UG0QzHNCWqElHzJ1eYcRtCFJmQnLuhSUKGuaWEfrfh0IoXSuccIfh0CoxAuuzKEhgwEnoyf88ABYDMotcLvNyZAdCA9dglIw2rQz_emvBljdONEwjGaC8JX_-Rk0GWEYO4qDt_f7ZMVcHkuSI19zm2U3z8zFhrEWvS_axjivKdNz5Sk66AeuucxYqNgWzIwUeeARU_Vlbzbl-jV3f0PtHbC8QV3TJaihoeX-7ZsP1MjdDHLo0CSvGTnzdxKa6rFpSYiSsupgm04pPnIY-JuKoE7KdO-KfQHURFvpozsrhhRcO525D8vSCXyr47gJLkeHseuBVz4YMZKKRtUNBBse-brGZ80TXWbue3r6gds_hlMjCs04US7TVqIEcqfRrFcNZYbo2E8491CKMmbIHbysQ3Oi_CkpnM1uyeMnpcQE2BXObMlL-zxLEPDiJE_3yw0nGsE3zWwLxq45Kz7VyBHPcz6PsHv_fPOtrLcgQ430cRBGwGISHy6y22GX41o5V5GRb8n7HteE2XHc1UhxCwCLODoB7CGF0ZUAxA1JpEIPAcYwX2ohmUToVzQpVIcoqGBDz7Re_F1EbAHyexIzAFZTxHoDCPobTJKooaVQrakDjaKx4DYGUbNIQiQxh4jPm50j0oSUT9L5KgBF0hhbTJr02hrbJu83WPufdY01sxyH6znaUnrOjUXYGyEV6HklL_patQnSxjGh3L5SwOWz0lYUXsB0r-J9SBCmZvDiQjb2OFR_rjMgoS70jEorD4uND5Jo_AGH3kWtsyffyg_qn2gkaOTugCnam5dW1i9dCaXiTqZExTGml23CfEIeUclBDG-2XiddjqN8fpf8YYgNovsSIAFd2EEFUeCT0uY-IDkJAdoLTO7cRNtLWfVXnLsz1SKw5juSW9acdSutUMkTwBEz9ckngEJI63-lvhuFAqfFd8XE5_LTdFQTGsJH4HUqx4cX-FUp0joNqCdJUg0GdszhtpfUOelCF5UJisyAK1Zi7mAiRlK92MVAn656b9tiXT_BQCsrkzxyfEtahCw-DOf0QkZRgWSi40Rud1ATYi1soiWvKb9DBnHTo2Hz7wIxKQWNwKBiLFnJZprXHKLlJ_MShZESSOg2bUwjSEnOsbDhD_nfoo2hvmTF9aHtzOpZ_ib1R05oY9TEBHoIXY0nVs7CBgIjn7e85SCVmpJ47_4B2sb2zuZkMUDVhJvYQn2NKHIGrP9B8Wdcf3icske-5nyXcxQXB2-wfePKY9bWfvu4-K5MKS7h_AtmMLVmhZWkzB2Nlt_EQbMcW8DPHuVShroAWVpSt7RrakIE1DYt1iouxt20cQJtP50lfT-6QtBUlrfEDDzT3aasEtlGY3OlfCTdbZHi1w25NnyVEYGyYoyZrO2AFUJzh507h1Th1Rp7yTVxuxsBLT6YXnKZDMcmqTpJ2xa5qSihQWH4UkXRfHh04o7RD0ADgUHnooDGpfTw2RC_Syc16N3tS21ZSQK4j8XmKabdAt45y-wUGh_3ucBHg_0lk0eVcOnX6k1zIZXH1xw__-4dSk6NEdoviu-N8AGmaWMaHwJXxCQhQBKqs-EDc89LIsXu-G9TP9bE1SBDdDm809FAGzn4flcntCS9W4VoPIFXwg7aA44gDZ1ZOcMtzbL9MCOYiQ8TH0OCD4lxhw-IiG_B1kCtmcEhSGkJTFxQAHY3bXmOHzrumzjgiMhRS1v3nCJLLmrFyoMqysGyJhZ73C2RA2C1wX5mDC4QGgi5gG35fE7tGJPqvv8tbDKaYy4MEUAMj1QVRwHI2oxdm80XSW-WYo1ZnfRiqtDJR5GNG7DbyJgv9jRtxxTC2LrhQdh4hPvN-2wtStGPSX59nKNHA2Xc945I34J-O4dop--xLgclLuoVLgceT8ngvISN4YyNSMz6qagaC7iecUsKFz5gcre-XVeexLK3v-vGwdo1pn-Q8_GoaAKh8ZaDh3wXPlrJscFrAvi3kWHD-WBfHSRhTzI4vYFsQVvZJqjsgwKpMIGtjiWdWZ6hMSGn4QOMdOwgwtuaz9RitUJWrlUiX6Hss3nfshIImyX2fVBNmBLwwerHS_YmnSk71ZlexxOVJ74uLESohvG454OhgV48TwQ8RNXbc9de8aQNNBYMhjn4dlrPhwMMDQgh0XvZRK4vaRScHMrair6Kxwqf43gaABZkp_LsyMULpQEUZxfY9-pszDgSoZpai0WirVhzXiCEwyoDmJ1_i1Hx5EyZhIT0SfRZk6MEJX91M1vU_ovk9Oft8DXgy1QLWdN3ERLA8XomdVN7NcTBFZceaxvllqIPwepVD8TMkf7LSb5K72AAD7uchYbXmjh2gwe1NDBtwLDkt1SRqzW1mI8oqCchXg8HAOnrp0WN5mqSjuhJo8MjOZzjqe8vJnPL7NhJXhJayy915Psy7-Gc82D2rBNeUFrcZDxVu3-WRtapfiDuS1NPLsv-xSTdJ6pSJGTNFHl48-3pRbDN-sMJ1gOZQgIm9AqITFcP8wdJhR5kfhrzeFTBn4gNPwKrWQ_f3OGobGbAaQkLOQM1SdMNQi9GK4h55awpHGgG9sZs7Q-s3ZWlActTwBd0IqVxeJVIiOLD0OixL-ZDGrR2zesygUetf8lrdyP8SdSDJZEfBxRio89DnAY_d50YUWKTzcMzIXGcas49kmBW933A0OH_o8y3qCoodApoViqBsnqm26qWsvj-tOsZztdcUplqls9shD4CpahIXhCPhwUtA5kIl90Gg3WqrCcCj3aQDRBwoBqmP-dR80pesXRFTKhcw_G75Tbb6uKSHzn99ooAIvKvl-gZ0977P2eorLxVHxSoz7-Q8TQo1vHmO8Ld7vJ-c7pJU184xrkv3qhAauCLfY0qBVodC_bzk2AgxftVZL4DsZWWOG1HVQnh9t4cIhl7lPvK1mxZNGuJ2F4Ldrn1oqPO8L_v-hVcdP7_r5zy3oR38hVjShC04m_TEXPbQgbcyzL4z60kiajQdUkU9e0oOxZvazmFChENgVfvVe8AsdO-2qELSLUSPJizf2wONNqxfRgbEl6WLPuAsjx9KfsEuHGqrxZIgqVxmEsoNeg95k2QNvJS2XdNZxqQ0U3O_H_-sEC01EEkVB1GcnO7DahjNPBK9TrSCPtE5nac3HvOKEG-N9FNrdC_VaJp3BQd-KEARw4NurSTVAgLAeohp94pWGTQCH9V9XDAjUm9YZnlI1tIFWBG7tQNW2a-Ejz_bteyCMbZuGoiofor7mH3gxiiQMjHXtt15iy9QZYXZjnLKOaN6ynqezRvPb4OQGFvo4Vd-i7SsvQTOsJZ9WuTui1ZuikMKYxw3xqpTAWcJlXH-ze36u0bMVc6gLtDLLToMmhYGTeicYp9d0wUjqoT_OdfHFzzkAs682yh7Ai_ARSYmQzwIRu-wTqT1roUvfDfYzGm3X0E3uLZqFUa22EK0ZV4nwus_xTIemeZ4hqct38_t',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Madrid","summary":"Madrid is Spain’s capital and largest city, located near the geographic center of the Iberian Peninsula. It’s famous for world‑class art museums (Prado, Reina Sofía, Thyssen), grand boulevards (Gran Vía), historic plazas (Plaza Mayor, Puerta del Sol), the Royal Palace, large parks like El Retiro, a lively tapas and nightlife scene, and easy day‑trip access to historic towns such as Toledo and Segovia."}',
                            tool_call_id='call_ZKcJLIGBda4NlWJ5no0xPy6d',
                            id='fc_0f49e2c2e5f4f700006945ada47e4881938c882570c14c40a2',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=71, output_tokens=566, details={'reasoning_tokens': 448}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 55, 6, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0f49e2c2e5f4f700006945ad9ae4608193bfc109a07362027e',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_ZKcJLIGBda4NlWJ5no0xPy6d',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )


class TestToolChoiceRequired:
    """Tests for tool_choice='required'.

    When tool_choice is 'required', the model must use a function tool.
    Output tools are NOT included - this is for direct model requests.
    These tests use direct model.request() calls instead of agent.run().
    """

    async def test_required_forces_tool_use(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model is forced to use a tool when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool],
            allow_text_output=True,
        )

        response = await openai_responses_model.request(
            [ModelRequest.user_text_prompt('Hello')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ThinkingPart(
                    content='',
                    id='rs_0a8ab955cd861bf0006945ada750548191a99d652183eca617',
                    signature='gAAAAABpRa2qk8Ao3j_vhzoQGjcYVMyg31oVq7qswpwhEBnLj-43H3kSfpCUKZDpr5oJsOCwRYud-SWZVD24Jq8cajT8IUv_ytOeoNGW3MxZipq3vuicH9Xmx6q-zsFqUZUnC5uEXoWpnRuTwqmUKWG2JZOR8MhuelP8BX1hNxU1gJNxESwPYYb3J3KmqQSfxvwgFLBcWjSOI23bqKrRiEYndTNXZXHNRu_PY45Y2PdJaKAosO3OMjamxGKz8NyqmsDLDT3KymMtBWtKFGm2PYGjpwuZXAdvv13nxUQKEFEummH0EkZaBBxDmtidAjZtJzpPYAjncBbwB75nV0lstqP_4KcoF7ZMZLU_0YdO7QR7OWOqgciQhp1FYL4OuRCi0LlL7yJSrBAZK3RVfr879RApme-YyHarHCjnkq7-ouvPc9Nluur9UdnZSQnMZTPrmXIkZzqF_aiSTJEXeTDfaozev6As4Sm82v3hsrNFjmUGIzkng9NmkUdHZIqk9unNv-3IXMdyyVWnvR3VPVwp4FketEKtqujR5zFlOdpTfbbwfduCAGRwckOCk6bfO19Sfn4dybhcexIgiFw5SAtu71GpeUC5YYzvaNfw7wzo2H5VjxLUU0QsLfoVYQBRDDS5xFAq4bv95GlrAXiLF-Vd4Y22vqQZrK2Q5nYUMkrsB69xMjC5lwXfkOPYTElaFcAhqQmDq3fSuoOM0eJ57M8h6zOYjqFcl3u7m7e6HdHXfcAllwGWQbeQytPdPZJVbYauP5h9N8GqvozAgkoqB-haUQgSrIwmYrpHPyOidNrq7H1FC17Y1NLXAEjunde_L6583QUqTSNYgaB55XUTt-b0VN99Sj4nodMdEDAj74wjYeBbroL3Jq38pqITFRNn4fKh1LKciAOTUUQ_JO2wDZMa1mH37xGMBYFocctq9bNp6iPscS-0WyxwCCrtkjtIMFNTEp0tUuLd7QSdHatOTWEyoPr6QL4gc0CXyP4MnLz2OReCy0UnTGNaR2Z-mEE4mrGcwwiNqgAwwg_kzruu52tpJEQVBn6PMkNFW_Rn0knoYkL3hFtD7Cm3_0sRh7tIF31b9rtnYSoG0sznnkWOcaN7W4cIO12KrvQrLzP_gmum8tD5B9X76Au_-hMQiZMV-8VaZWYNk4rvjAbT8QyFzvFmM8_2hE7T0-S59K13UfHKNZ2F6efofsZIeUTBgSBZSIpZRYuWGPBX_GPU2cXlISQD9_oz7ISykBSEpkVJGCKhMc4kf3fJ2o0YV_RqdSm_hgh_Hw5mK5pztxkaKRqTFIHEgbsCtXUmgwnmnomFZLEaXOYQYKO7AIMp9CQCRL8a6gNlZ4V8WXrnzdLu-eeN627vzq1EKCYey4Ynz3FEeq86GZ44gAJnrSR_llaTUku_jru_Kfh-FUsG3uV3JTT9lYkSx7P3wQfGi5vqs7xsCiOVpYMSQeRBWxjssI_0tIVLkxQKCg0MEzG7hfoW793STp6uaUX3pIg16xH225FoIR2_grBtMU3ptg_f5OSVm1v4WtQFcjdojjz2GlBnXCbinSS-63WSri7oR6e5Suw3phxgpfLDsawqaqUn8gHyf6-OUrvlHexMD1JSw48Hgg1nt0D_CHT_CSzVG8PbBwnhdOlN_eYEzGu9goCtTatzVnbHP8z_4lV55E6_swF_-TZjfjumrogkE4Bic53uE9yWce0-lLl3Sc9GhcKMKfvcgx-mBV6rEoMyvzX4ab2b00zPCiQneymLybd4WhaIeFo_O7ZknxMkNWPisVelAjx3Kua6D0TV6QBqmWF-oUyOU1WsumB32gGkG-NcbXZCyzrRcsQrO7he8pXeVg7L_AsIPaMydwigFscVCwYr2cDnifJzLZRqdlXSgQq6KCCaSXMl_IhygfTZTTcD12XdULo0GeYD5PzQ6-R7FFqsm2P9dxVhZ8cd-X-yvQCmGCIKQQdcZ-4UMVvS3roKHqBTv5k=',
                    provider_name='openai',
                ),
                ToolCallPart(
                    tool_name='get_weather',
                    args='{"city":"San Francisco"}',
                    tool_call_id='call_TAUogdpeEGrCrNEjaghnmMoO',
                    id='fc_0a8ab955cd861bf0006945adaa137c819183ab43621a285b28',
                ),
            ]
        )

    async def test_required_with_multiple_tools(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model must use one of the available tools when tool_choice='required'."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        settings: ModelSettings = {'tool_choice': 'required'}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool],
            allow_text_output=True,
        )

        response = await openai_responses_model.request(
            [ModelRequest.user_text_prompt('Give me any information')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ThinkingPart(
                    content='',
                    id='rs_0f1df6a79361234d006945adab4b88819db764b555415861a7',
                    signature='gAAAAABpRa2vJ0_OgvEYf6vlD91z_yLW98ID5S_kSVm5MVstj4bD9BDY75hrkvWMmACdV3c-LsPoYj505RK09c8vFqkgVT_CT5UPsThEOU7iI4MUknYTklhCPRhqXf3IUw6sYuiwQNyhiJu4xb1tBU0Ao0HroSNYMWc14ovjvEscgUwaezuLacKGvJOaG6mHfcmk6m2ZtVK3zRCL0VITN4lTpBW8-SdJANLRqxWsOF4bwLy3k1dsZsZbsHRec-YkKH_DSsHZdM50wpYpPXEefQXsQ_nEIru3R2BV6fae7VTAOEsn_AUCqXpbEdbBqyEHLwi-MZuV7BFxnPba7ure7vJIyrIsPglswIXhDC-hdNIflLT0uO3V2hy8qifXZZRktMFYPVmfIXUl1XETlnZglnE4t0kGBpnYRsVsGPFOwc2r3-upXXtcHkrxD5COtfgkzAmMXcbqt57JOfpZ-xhtuM5FQjaaGTlu_YEFjaKjQAoaPomkJlZoCqToaZL8O9ec4oKaBClMPkaOncpTWUfo7FVezHJq3u_iGB0mp3Op-QNeJXN0d84kxebJVref1YRdWuDUMdv87JP1zV3xN8Uffq3qxNMR6gujUMHtcxlr03dV9WJblrSRBdT0OX4QU645Lveo8gZenHYaXg10s-RPmKjgHwUhTylFHwZtbDzL2zns93D3-hfI22BrtykUBQbXk2hBXLWI3aJvdpsLil_FFCu6J7UftYwiBs7hA9Dw9PNmnCRL5gh8pEhRZ9JereFFwcGc2NIn2TOS5GwB4sBTN242KSWXCyvhhzgAWTXGT51vjGsn6aY5GWinVB8y5BcEvmWLyucdobefZxNbkmv1qnSSw00i047pOUYRAofozL_B8OJpv9XWaix4tGbmnwOJQ0-jE4lj0LMlKf6vGK7sThRiDiqTGGH1ngbSw7wnx0avbtH9BEQZkNCmkoyYEpO6Hh0KmC4QP2i9jK5EDrFBjbKWpnHvxt4wx55-p3jGoEOvwR-RDSAlAQw-4512ST2LIve1-3gMbhY56GzfAiF-KyflET4HXlDwbRIAuPxzUWyMOqYJo1ykoeojvCMmlc878Z8UgzBQXNBpNNabB3oGb9GmudM2qdZaaqE4B7ulpeQWVBD9B5YJ9uIcNkrKcAM1_BMNIqTjO2fwXr7seGsq4a-9TGVrrLgzhZ8x4T-M6rkbWF92sBo62BfwhcIC1kiVGrceel93Dk-pFjVAE6Umx_RTVLv61k2VXwh06IOb9CGYAK8EIEe6KUhY7ulTu_pgidl3H9yHavdfNhHRbD7xGjhdPAyABILkHxe9trqB2Uqwr7oyalcQkciz_eTDkhNbKK8FuM9CM6v1XPbjF6ZP4GPQjFVRc0fQM4ETDQRYPxatUD4x044V5Xv9Flyd_qS7J1jpZW8bfCivlYniE_QAfatJp9RJzAQGOT9d65xao4_zYbv8KCMXMEQz7QSZshynk8vW-ELa-9X3airJMfCZzQcHAZNUfplwG38xjYHGWvxEUxFgvNv4H2eTaCd44rF2oaLWdojrWv4GwTfts7C27bA2rj52PnVB7Z2ELYIiFiPMBVHke0-VrWQKduMQO-SLGG0RpnsIL3MEkfgk9ZHNQs_WNUZ2DEeco4JKmYdP8-Zp76kfzg1Fe7VajvVQ8UG1IMtbE2fpQkkiFxAL5iz4ofpig6G0RrKr9bB9AbtHI4V8080gMvqZ44FO0ESa8u0tVn77xnvR7Y9rGHC6rVlA-1EbK0VI9iuTbVfXAEIB9MFZjAnx-fhTO4S6jhh-nlmfYkl_U54J1r-1nmspY4nCQKJf3fed2k5VFxuD_MKzym_ijFq4tql2-2fd1jdKM1qgryS5Id3UdCLN-AWz8T2tMtlDNBVDGxGTkJ4cQLTOcj5oAI8I4DxXX5uhJm0pYhbJnhX_fBtEvvWTaCK9O9b3fOkiN4Gl0TKpty0bv-8OgqY4XW7neBlHBML15FYaUY0zDcx2TYotThQFKTnTmKZiT-z0jzAgvO0tfXMfr6l3ZCKMZYyMZ1mqMWgScZuOiMJfsbfZer2Nv-WKFVbCDNN126IsLQqSJkP8qYWPvt0wO9Oh0U9vR9OK4HXsVTACJphR2dh4uhWOSjgiM6-cjQWE2swLMXXwnirfXcGSqqDODQstwVcZtr4pyiQ=',
                    provider_name='openai',
                ),
                ToolCallPart(
                    tool_name='get_time',
                    args='{"timezone":"UTC"}',
                    tool_call_id='call_uhQDpmAxwngXvpVwszNO7bqj',
                    id='fc_0f1df6a79361234d006945adaea37c819dbe9df62aa6e6c34b',
                ),
            ]
        )


class TestToolChoiceList:
    """Tests for tool_choice=[tool_names].

    When tool_choice is a list of tool names, only those tools are available
    and the model must use one of them. Output tools are NOT included.
    These tests use direct model.request() calls instead of agent.run().
    """

    async def test_single_tool_in_list(self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None):
        """Model uses the specified tool when given a single-item list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await openai_responses_model.request(
            [ModelRequest.user_text_prompt('Give me some info about Paris')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ThinkingPart(
                    content='',
                    id='rs_00ee39221530168b006945adb11fac81958e2abf8faeaaaa44',
                    signature='gAAAAABpRa2zcVTXyhVh2yUSPiRReb688yunfYzpsA_8hV3D7Bc_iZiLymkbxCNC_uyd2Mqcwj090t6TnB0moqWsjj5HasgFmPSKIpc4el9vMDgb5m-liyqc_NQA1B3r46ZLYbks5ehsOMr5oTWzMSichAeXSydz3cKvAacVCszDv5YASPZYlqw2v-vrlSzOeEkeemXP9CPLhtF-KenAoa-lr7kUOL6JkvDGhvfH1pwg5Cvj1cT7iPlRePlGIsKh70LWV1EygHl4zlGWx19ZktWapCE2d_0Q7NhjlItW-7UKJnoRU0VAUjA5QeIrHAu45unD7dj2EJoIIL2k2pgF3-o4q-k8LEBeXmvkrB6I8xlLXl6LHyfHfoR0fMOWAb3VHcECLye2HHqbN7KAJUW1v_GziVQ3htLebK4whE9oT171sH_5YOvOJvvGnSOh0izSkFQNESUvpsDMxLDESpcvQi1aiUQJTXFGENyVfxgzI0-ywRxCKhxdRAY-Jy5MwqTuC3fTGQasFiLzbhZXLpoVeLnhW66GZIi0WTN4Hzk64Xzgi-yyS5rq4H3E7Phb8ToTbh4o5GqB07zW6oZobO9jXVfQX1cBLSy2XZww509s1rMYJPYH2vD8S478_qC1EuPAsUZqW-9g9Pib-NGdAKt9pvRl8Ra2oC7QmFr6F0lKpz-Pi1hn6WT3RGFlVNe7o5514JiLVVwBVWkvkdFXbI2V6BmLfpVIJQ9kk8raeQR20neRU5GFjI3PmT_FI5mzsT038Wc1QwLHIXY8gZoZIIJokqTwbkGykMEu8QcOr4ETbbo6jUXfG-Bgt7iZwC_oz50tU0EjHa7-7GJ2TLaG4oxPRRFLolVhM4NThabXGXqqQ_dbjR5teTtcoNiWuzfzp1wQri9JX9wzX0l57nsLQMXESL4oMwvkh80Cu2e2yHMVOXTUFFCdjGTWftO94L1AbidivAqxsZoJpgXnHgAmPJLYKZNfKltadJYTdCrVfiHOVpAuPzi84c7PnFfe0wc1TjMx3g5x_sbvpyM_KKe-boex9LaCrT6vpTuf4d8n2yySkEwjXw6W7Z5mJ3p-VEIGRipZZI_0lTpSiSGQg4sDs5xb0vO406rPvbDrHCx5Jsklqdaj-Qr1bCxNKnD8gOjUBuZ3qKdQX_Acq5sv57ropjCZH4RZB2Xe7nwQ05g9oIVkB9n8cFxHK1XKzvRTc0x46W-N_pU9HzEphzVbUdwbuRwAz45V9P6opNIT_leon99dNMvZEy50YBeM2kt5Q_0vgG9pmDq6alXmS0sU9D_MgNwrqhhaU44mTwT84hfq7Y1TlFDXEtxmLK_bol7WoD1uMdVJt9Lp8GreF3ooCLtKRfqOtKu7x712LTgyG7tEAjjcH0TpopSbx4i6biwlHvE83LTPW8-D-vJ-_jzvTPEuNYMblO8hLQ31hmXPxHsFMSKP5Te3QAotF8Tq0sLLh11F0bENkd73VBNKzq2ymsBMhNx2kJ_p2aCpgJwffcPBtSnQxghsUV_MSxihTpqGelesys-BfoYPJgJonb6OJo9FgSWi5_1kivZmW1vKsa7t8iqB_gWluOpyB6nxNnFgeFuli8TtPyefRK1ZioT2SHomRD5Va0beMH1tKftagLhEMMoUDiqPEMaolUxQujHtYseNnUWLeR01NESyBvjuZPvCWYPMjLwKwuUi80wgXdk9vJuWvo0TEY-8T_DZlO1xePz4sIB9xZWTQVU2-GO8MwilDc-27nXlztjsHPlETQIDkq0pNJmHfhrOj5lxCBGn2tRgpBBZ4iHaMedI5WlsPHyUGWTJgmU_x4P1IAApbwcZxiAGGYYD4el8sUZLS3Y=',
                    provider_name='openai',
                ),
                ToolCallPart(
                    tool_name='get_weather',
                    args='{"city":"Paris, France"}',
                    tool_call_id='call_uvLqbR8iSddLL7gmBEff7JM4',
                    id='fc_00ee39221530168b006945adb37b748195b5aad06091b60df6',
                ),
            ]
        )

    async def test_multiple_tools_in_list(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model can use any tool from the specified list."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        time_tool = make_tool_def('get_time', 'Get current time in a timezone', 'timezone')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather', 'get_time']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, time_tool, population_tool],
            allow_text_output=True,
        )

        response = await openai_responses_model.request(
            [ModelRequest.user_text_prompt('What time is it in Tokyo?')],
            settings,
            params,
        )

        assert response.parts == snapshot(
            [
                ThinkingPart(
                    content='',
                    id='rs_08ac1b945ca54380006945adb4f78481a385c65812c2058dce',
                    signature='gAAAAABpRa22eKGTAbHa7qOBygeZQiCgMq8M9q4h_j7TMUNw4oO9Wj9VPsZoyuE_Yg_ZunvBST-6ksTRfDQKzZuX7yEcmYunjMzD8IgCp9O3ghz4O1HQzyvJndw77M34SW7IbOszElEnmntLPgtgbkwYSoz4OHIu7jQDFehq-UDbZ6kN_Pvu9MFCoqsLVTxJs8dKoT9myC8r3a2qmExJ93I9veDz7ndV736UBkLJS04CmjHIjg3QdW2xnvk6FtBiUuOEu_s26ng6I48dIlJHjLLwD0HBdPDsClcDPWAdOKDfADZARbmABCjN8YtscRPe-GS44bkLdygxyd21i6X7mg8X7P4W_ZON2cjjhjMNBZXaPBcrWAltDmpqc0GSnQ-4QB0V9Hg5oXyAyJ7E0R4RwOZceGVBpqh_-Dw2_TfvXDz1Yqf0zVLtxxmhgSDPI5TIBNmJNJzVgEJFJFpjeDmO1X6kzfW9463XDnhKJRYIbQ6AOZ9x8BEolsNnbiVf-tMNaXI8HbXe3W3duUJP27BDhSLIt5GPyROJpKUruOa4T1EWBZH0NBnooGbO0AJJSfnJRh8T0OIn3uU8kTgg1lqhNQEmCZzeLHhAsCUaZ_W2Rw5T4YSniX1nlrI8YCr6NmqbcftiOK8pV3j1yIuDvibqIKNHS0RiDiCvbxVO-fHpoibeCnFJ5Kerlw05iEqmXwUVxy3Qtd6qlsrUOFGf6ATpp4rKVBPmSP7KXrvPd8gdNzyUJIovMjJYcOFLmSVmsvqFu-imolZ477excP1ZCpTMg8kIOOKs3zLPo5BcF-MQC7jwgc5fC_D-YpxqcsiScSHMYB1_k-GjS3Mwoa-yHHZpU2HQy_YGqKARAQ5RN9dApwTT8Neb6qFyA3pAx09JKvsUK7clQV3QOSOPxaoc0Mxy9HTiNcMRQNffEP7buTL1_vXCtcIDYucGo6A0POsXvAHOmvjtFvmeGudzry_jRumS5QF6jazzIJE5c4mMSUvzMIvBFlDu6KYlzEWR9gWS5kNRtboH2nqUZTpxpA1ZE-r62THbpPTF0WjSOM4lPJo8O_yFhkTTRwUlWPSttUDFAv6wALtJsb8ZkQTBcbFboB24dJnD3OcbsjmSJLbukSOEb7UUreWdh9kP80KZEMeKO-rqWHH4KEdoE98P_dflZVJ8gPGN23H0QnE_LeIBA_DH4QQNdW7g9EqQDZLNnDsRQaFTZX8fQkCLyYKlMdcdlLuz6u1Fcro3lcWYfO8sH8pNr5EYcpq8-P6hhHty5jxEE8_yrL-qv5uy-SSzwHXwX24l9TAEDnI8fPXEqxqcSx56JrY1CzcMX5_yyaCP9IW6SerBcFV3sxjeMYKa2m9-sbVBLPTpEhDke2hp9ejkO8tJVZWrGn5vc7nHi6zIUDkw4jtCHDzV-BMX_-ky',
                    provider_name='openai',
                ),
                ToolCallPart(
                    tool_name='get_time',
                    args='{"timezone":"Asia/Tokyo"}',
                    tool_call_id='call_18iOaNrHTxQiS2WUrtwpFn3L',
                    id='fc_08ac1b945ca54380006945adb6481081a3a7b6fcb63318393f',
                ),
            ]
        )

    async def test_excluded_tool_not_called(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Tools not in the list are not called."""
        weather_tool = make_tool_def('get_weather', 'Get weather for a city', 'city')
        population_tool = make_tool_def('get_population', 'Get population of a city', 'city')
        settings: ModelSettings = {'tool_choice': ['get_weather']}
        params = ModelRequestParameters(
            function_tools=[weather_tool, population_tool],
            allow_text_output=True,
        )

        response = await openai_responses_model.request(
            [ModelRequest.user_text_prompt('What is the population of London?')],
            settings,
            params,
        )

        # Model should use get_weather since get_population is excluded
        assert response.parts == snapshot(
            [
                ThinkingPart(
                    content='',
                    id='rs_053b747a065d0b09006945adb79c7481a38c9c9a3de7113df6',
                    signature='gAAAAABpRa28zbIg_mdRTOYLOWfPdBeRS4ygnT7tUSvL61wC9IiyMq2p8pUDldnyKvmv9J2AhVjDTOPznDK1BxfsrANNWg2e-izKsxvDNMOKBoMuO1iPC35lzyDloOqQqYya1uhLFydF4QSSLDzXsouqVRxGiy9_LBfRaPqScQrYnEeWn8W-EPW310LrcW4agtSk4hMzNhd10uWQcmUTo5ryG1-UUnxKjajOaW0nm9zVD4oysnD808LgfUiR-DdByKuQ6Nediy3MykEvXmCLcHmTsvoAoKsN-cYDQLiV8tb05tRGxLpz705zccHuAbsUUqCUh1C1KYtCXOJrl6j1JNKzlUK0iUoF3TcvmPs1xIoDbUgGB-1e56LXc7-gP-T30mb1YIsajKjTtwZQRkwWGLpL1aiDl0BpP_RiSLB4srG6RbDikwrHRm5Tm_ewpEpL3Bj8lG3IkylLTP9epAEFPoX_8VO-GovdacodH718_2ZEnqDQPX_zT22NinpwuNZ4R2QyomdIy4jpobtFAKNejEh9YBUjCaFeHgIsqwy6Iu9KRRal9DWof3Gn1_gOskKSqe0RlLgMexUnFJcIrQmoR43hnMpRQz3ejunI92xt7BRVrxcoNRZ-1Rg8H1_sscO39H7RTVl-VeDgvPmmPZbdPfA-u2nN7ZfWpTNbSTzmFRL8J6Uy9nYebZZMxdWPI2N83Cc7htkE_oy3oG7AjyEOgutZuQPdJWZypNcv7aSIKuXoxr9Zjvg2Ki4jStNG_2GSO-JS-BRIZ3K_mAsLtLQHCwMWLJDvylJXJMbSiAIFExoMAdaHMYs5MAFv6EOsFQxcTtLRdUWY-6UTv4QKnSnsawGjby9gWfHyVqtCTxJfK9GTDR_vYpA5ZfDo3yuAq5i7HfKS4AFGW4oPhJX4C6a8DUPHOw8tg98ZUDQS6SiILvpmkk8OvkCA4J2XA7siGhnt6e01_8Fazedg7kViXQFotGpL2GFxH6ROWYlOy_L5wSA46O2icUm28GM_3zNYBY7_3VyxqWLnYdmf1lbJnLbgXH5gg1ZH1G40O5rhXOi1lzX71YZjSdK9Ebm0NSLuJ959l0YGUl00NldQPRaN17_zo3sZ0-tFgBExkxvq14usjyFC-ahYqon7m-fAphQjxfMULlHVoCJi0rR7UjWItLFWp3yHj7ZVyhUqhEea0v9PBlP446xKSRCswBTCAPX1FsWvDV6vZt6KLC2qhy4s-8jdpeuz3B-BV_vYbiDjAFaLMSSJHT_rZOUOhuSJ9OFU29s4vv8X6VPx4Vnzo5wnyQR3vzxMw9LkW8K_fhgiOIR787KgX9hp3oc6VNkiulwPVlIHyIRYjikEmjC4hdkNkqu-myxe97ayze6zLwi10XeJnCcHSyeMQhH05rkkwE_NnFSrSgmjxoGNlLbxFEh9aWQj3CJvarbHPoSWrVtr5GlKdfn7MyxomCvCUeVSjRFrpmHRRbeOoqqvQnnajNd_nmTWfMpC9N3U2qOVBIclmFdrYaIQVYEo-Rqi6NPJnju7ZqCUSyHmFOZVfeAK4hUdqQc8-L5gB1z6tqz5vINRT0mElgv41Wevz3bw-CHcJg-vyCyvgAxvLGDVtbvTNMjZGHjUFGNjxtV2ZSvgtzCV3cOiobE-ypWVwNm0fKYFaG2kIJcl_YfON_CH3nAfvpfI0OILBU5EJXiLSUTlvn4hHJnNsGTHOgg9p4MYtZgBTit7-3OaiEWcGtITmxWjuPi39o2Whi2XjVJmhnXrg6pXldqbxPN6WdOJmk4i9Cg7z2AOXxoktyyIBThvjLJyzVv9d1_pVUJ5Woo9z8OAulrkeu7bQMyF99IBF0wzv_BugCdup2brrjdqK1YgmfxGdsmX1923g5M5fVt4dI2giz7ykz7ZWELi7QhQ08hMlFhHq0EqTMk2hXAnwtIWjk43D7461pT9V7MMJQ7_TNbkMVGyjLDYa2J2VanvG7tU6kAb8V8jru6kxItA4HMNbu8Pt_LYNakAtCGeBd09k9Ub4GvkOGsU_eF6AJE3v_pVxXGzmCST6Kaq2b-sNJ5_F37as_IpD-Ofu3Z0o186ZO0BpUIl-2M8bNbZmWr8dYunXuKVMR5FMYvDZXkCYAB4XJVaRlZTFQRp5tQnlJMYdZccQoayjl1iG87iG64I9Rqn4fxFR3M8WVBTq2SvJ0nW91hP0qnpumm41POHUZdnUAZOsEJFOYoFk-wZ37Eo4mHFZyqPCXVDHwYv-0KrtkHwCGwjycmI0cqEwUntwKSBijD08dVHN_SPJQI5qtKKEhNpYlf-t9iMRSytTKC0g4V0x-3WMhS3ZIWcG9DYgb3kIEnL9ZU8TpqBStrqbgEz259X0pLV_bLuw_JrKy-brbvDIt5uhSw--KxtRHIY2zllWnrjlK62CksXAeKS8Co0Ozc-ysxiVA_aCN4_BtoStznTBgN9PpQT5iC2TzpGqlE92k88wvv4MU4YGy_MfoyzswU7TrPW7SRr1mr9RsT7jV1qGJV84-ZVN-w9cacJrQx7sqeIMDr0uj_j7gvJSUz-q7Gnw9e5GYVG4fNY9CTIaLMNVrg4BqSKhradjk4U2xgybHEL23O3sfBLyfMr9--qsP76Mc4mBx8oipl-Kt3ve5h8Ig6pzQ2e8ZEsnio95MPU1nKrEEmoucm06gM2qRUe9b8AEUbbWQlsUSSHDZfBWSvQSzjgr3hCDONVIMgM2_14XGp7VPryJl3Q0RTc4XnUzSDMKUE=',
                    provider_name='openai',
                ),
                ToolCallPart(
                    tool_name='get_weather',
                    args='{"city":"London"}',
                    tool_call_id='call_241hJ2RGL4zYBF4q7j8Jy3wh',
                    id='fc_053b747a065d0b09006945adbbacf481a3a955318160772077',
                ),
            ]
        )


class TestToolsPlusOutput:
    """Tests for tool_choice=ToolsPlusOutput(...).

    ToolsPlusOutput allows specifying function tools while keeping output tools available.
    This is for agent use where structured output is needed alongside specific function tools.
    """

    async def test_tools_plus_output_with_structured_output(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Combines specified function tools with output tools for structured output."""
        agent: Agent[None, CityInfo] = Agent(
            openai_responses_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
        )
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather'])}

        result = await agent.run('Get weather for Sydney and summarize', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get weather for Sydney and summarize',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_02930ea85df92c07006945adbce16481a3952b06cd588f2b38',
                            signature='gAAAAABpRa2_U_ZIK2RaQbq3TjYjEtFWVi0nEpUZA31bO66Xja18bIgeL1Y3-SkryTx_53GZ50iODKl2Io9ipWX3j3J1859ZG9wK7edofHhMR9D66wrMkutQzDfX0VGJ8hvns_x-w26drVwXU7q0oRQB3djisO6cqMYvo4RMFlLDN-qio0Uwxf0FeI3f5CPJ6zo44CXyaERFCHDmy1gkMwsvbT2sVGeAzD5Pw1djLDdkp_7VKpU0ZXWL7LMg_iXILZPFyc0utzJs3LoBLHisOeNqF0dYAHmJOFOVQ2TrkKw5wrfd7QoHNqdwg6wqgsp6ML0JiaN_gLqY1_x4gH4Oeca0r9JK9zRmgdneexn4nl1YC-NlawwGUMThTKQulhfkNH7W3EXXs47q99Ao6q7rDMWaQssyw8Ks8ycOEw_x4JCDpfVF_nB9I1moqHRXFDrsvOskZzgz9RaHxgA23fyeWuZ1gAa6gX4yaMJtchJl0dA4fAR2Yp4zDpYWrYUXsRaVmMInzBIv_WN5y3z_6nhLHSTp-69BZU9DQs4Yva3CfTZmqKOKV2OUnd8wrJyqa0rQd_Q9wT8MsREKtZUEm8NEMVHvgPiaZrnzp_DLBFsoxttPEk-IcqbN4J6sOVacg3mMwKxSTtILkhbht8kmgX9JqnbuL_CM15GGVpkcdAxB_euLKramymMgoNTZ5KDHeo0IcnxToTl-3p5td5vEVcOd1lA9_76LBzDbutPczG62Ok99nYlrFj8o7eKiDfo384z3eO-McGLR1RSaiK7ydjP1S4GKmj1HcYcsySiaY67DAiWgfOTkR9CtIWA4tADOPeTux4pMSv9u8nxHVSl5CsLFoCB8oCi9u_mvKf4ktEUZOrRDlkZ22F1Dhuaw7nezRrh_k7_o2cVh8krMqx3TQ57eXS92gnSU4uFC5WA26VRI3SisnugTM4TenopE3CFrbdQ10jEx4zgokQHQnAYg--i_DLj_vvZW6ls4QYeILPjqG7-GhkoFosbj2Yj_1PE9_4ck3aQH8guTkWtojutFM3pxETqbGvZcwX-w9yTZF5IQhKCy4NKTpSp-w-LYHeI5jd1fVq6xyTMHH9jQQSN4aah2iqdUcEPMEaOxdZ5pTg3RQO7N4K5QnpKc9MGhyr76dooivM9lUQkpuRPIThuIbVmTSEpfPqbAXKeq0BwKd58Ibs2l0-DSCP4BaYR7MWvCrc0rjvKII9cw11gJiZ76TMPCtiK0WW9nSMuD5TTgCMH-tx-l3F4Gpy2KWWSgDtnXcqFInwcP1rlWhST9OL2q54Q5y3Nopy7rK1EYSyfoBMdEaYQAAnm1F7hkfLWTvIcqiI4CEGhMUI9bgvmv7KTi_c0-NmB2fI8_4mr2k6O0d0xX2YT7m6hKgH_X1AECTfhDu_Yh2pjuGf5k-wbcH-_qrf27uBYhkdG21yJ4BInOCGN5qYiKpytkON7KR4Y_yPYk0oOM9bdkrbVSJXIMOcyzeCzj1Hoz4x85qjQoKjho3NdXl7fa0RHmp5J-8A4Ovnn9xBdbid-YPHABp9M444TEHxfLR1lQCtRDYfTGvR3yb0r4uIkIcH_U0rzdjJXXbBH10OLnZXpalZjrxMRYknrUxVilbNRdb1gbkhAJpGGbO2Ir5R53357aIbBRkbXpa84d-wBUIZ-UcEIZnHE17DM5ytMksTznkfG-xaEQaw==',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Sydney"}',
                            tool_call_id='call_VHaUOnJSRlDiWZxoTwnKXJwX',
                            id='fc_02930ea85df92c07006945adbea6e881a3ae47c2f6bf1e729e',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=116, output_tokens=148, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 55, 40, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_02930ea85df92c07006945adbc87b481a380f77ad811650cac',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Sydney',
                            tool_call_id='call_VHaUOnJSRlDiWZxoTwnKXJwX',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_02930ea85df92c07006945adbf9b9c81a389c5865b779abc87',
                            signature='gAAAAABpRa3HTpK93lZ8syK7wIbo6KvylddxQAaUiX3JhwX-EkDb0Hyxm-T_M82GwuWNuROBGXpsmgmSv4GGyNuo3wScaNRbC99JhlBEuDLKDx1J2AbckJ-3GUzl0V_0BwCuiP6sFoHq8Z6Gw9vD7s-mNTUkK1JdgUZv2jbh78fMC5V49kQAjPeindHckdZPgvT30OppSa5Cmi-dyMLlZhYnstRZLE8nmQj7DqiE9VB38tcTROmATnvHLEikntcTu0DLu_xw7h2DLK7DkfYOsKOEqbtuhmlksFa7HOmvfgyblp8L3-Zm7_oGXxv969Pktoc5qxkSGsa_tbplyjShp7F1mi6OhBCmWyZd8KfCDXMk2MY-lTXNwG69DtYeXMWL3HzDNx_l8iOY_x5pR1nrHfLFzow_YQ1LQbJ3MKdmYh5mVfmVgXMl_VNJf2vYmqzCJGuQRa3Nfn0gaOsVMQbTx8p41VtcSgaV_bPAaRgud_9Jfnstmfzwsv7nlfJncXU_cgLn5Pjl-X4j8FddkrIIQaz1gw-WcY5i4PMXQCHvCZLsM6kSBXtpIBv9BQyh40hsQQlN5BISWRYJHfRkNXUEhJp81mKsU12IhotVw-iwVU4xhsiTX75f2ZxIYWgXnbXttw5BrmWwrsYT2XUK1y73JQDxjXubKjcIS7CyrQ8hwxKeP5iiNQrc4C-HoyD_AXdAxSHyRgMoHMarcs6b0HHjQXitZIOTrtw2nDpFe5uDOXAuFDNPIS6zIYogT2hKRR3312zYwogpsdxyLuIZzOlnaWK7SsJJnnWa09cWWzmB4-BxkS5VAAP0VHSZnd072EOGy4RJNyOSmLYN73RaHFZ3jKT9_7fuVtruQ4iiMxCScZitr1yC7LLRyUQ-jSCVXqnHf-rgMBJle0zVnttusMJNRMUcEAkcQEplbcRpXjGcgmbhjCABNw9UqPIR7ovm4X0L-ioZR6HPluf-3wfpmxdkpAf9ish5jar8o36jCHTmqFkZm6noLPKylmeLhxTWtWuqMbUFmEH34NomqdCsSSLI3OokqH6-9-KXJdGvyKvIcjf7twvYJthzmn2aqaHACKrQfxy6QvkK8_8bcj2WfTb-Gwd6EtzY2_pPbxQeemUwqw4x5JAHjSiuzcCW1gC7OVb6tHnUY4uCggP1vvNWUa-z7RFT38ixICXbYRjrmhkREYJWprknaXbDhKCFyPJnom4CVsWWA9VWApU559lo7VY9rM3V1SljH97X12nEs4fBN0FZUfLmPMCfqTKWYscLcVLbFyALO-R6y1YcJDhEKyXJCIIvr0oK73HF91_-GOgFVszWWvGDgfWwMmjzeJ8jfYill9nXer7kuTv0o1DmL7ypiefY72pleo-FzjcLiApnZpXCx6gEGpTmG4Pwn3Gh0bf8HxDKEm1lRXAx30fMbE3SvZm2t6hqJsY_Jn4eMoavootTOxuW_xgCPWzW_ZX6lQLTQMHRoFESnHf3My5pwXkmvEYD_HgsXDA8nTX8Palow_qlgHpvegR7rKd1PTNEKbqhltRB7NsL0vgNyxGnnvOXOw0wpPBiFxYVCuMXkNALcbXOImMZMqDHKgtEahLrKJqEsnKbQaPGRxLPT1HJDLl38bnvRLUFA-Ic8ukTdABdSJzIVvtRZ58UTUyWRbkhGjLXdhpe_RUn-1NLwqsW0ShBCAg__rOVId39ezuXUeTNl3o3ANGfqRtDFBETcBMnWtykT2s6MymkM1EXbu3xBVraZcXTHhz0bXSVLLUkj2VOuDJTCtLsJfYNN_SD1o3zcNTJkkXAF-fUWvquytWoxE3_4vp3Vd_GEjJL3zSUSvTaDx1ic6Jwgp6O-4jS_J74HvvF2hOblCz3gfKacdNVgGFy9A7gZmErxNTdRLm3Gzv7tFJlJEeVV1kvfvRKrovc1B11xt5wSlcCEmU_c6l3RQLqyiWcFXejyYHG9-gpYjq-T0Iw0mhqEx0OIOoo0szBo3gLk8gClTMM1u_jX3hN5QeaVAKmo3J9GbM6vD2Xeueq-T-Q8XdltSKkBFtCfM78Af5jGk7D4eEatVlJw-wq5thwh8iklZcPmJnEeYoMIWZl-PGTI29069epbmLjkNgZSHg9jS0c-qxtDipjXqV4jdIzXBND5nTWohLazyvIcsjgpzsTSGjTgrGUt868pmnVPYWMD1qDEmL5-RcsD75cvg27hrmA0KTqPTD7ueqlFHD3TtniiR1x_oYJeaPzaM7LwokOyWSwBWU2P3CLTw3FE3VKv4ZCHwM1yVqs86jE6tzmsxcYowO6aG0g13IbIkRWEsWAqqNfC4N7xTDBtCQFPgn60WWS3VVTgxYLCZJX_D6PsZvXPmPwMIfpc2EaXNfBZZzl2okxmL0aBxwdZrq15QXjF_uF1zxtWhmoARObQZOwBMPICZYjzGcuOF3lIL2jaMLmsgGD1B-x-WS2r0UKr4DjZeh0QpYNqVDGmBmUY74prw5OYbKAQEa6ID8p1rK59n3y6SJ6NGSSrp3TRZcHGZVtxuGdTgMW-lhvbR97T0CRT3RpKsL2abE1wGOhA7nG2kaGh_DlcNxwQbK71kQC14My25oLr4q-iT-MtJZiYavqrTdDLntnaR_-4tFZT0mbzABCZEJP0t2cJJuPDlkwNHsFNedSf6aEEYe96DZyM-oK_pie1KzIXtqaBzzY0hgDpGqCBq0-nyqSYJvX4N1t-XipAfno6XL2t1JC1J8JNlTBJUwvb4y9UW54i_ixG5Dvh_mjA5wJIeFuIDSEcrYtE3Fxx0BZE6gye3DhJtOtypcfTUf-NaBqQGMDZN0g4e-WnnLKtoT73r516QyXnKJ_dUt6mpfVQmgkgP248GKRryF2E9nnJ4jwAFAt7a5tiqodP1VPSH4Kb3PoPWY6tp29nahnoOSiv0njHccZsxJL_gWa7QQVSvo7uYgLM3EMnN4k0INErJjEhTPvO04BwuSiWJUadvXU83zrr7sWkBftj5bg4ttruvkNlz_HeKw=',
                            provider_name='openai',
                        ),
                        ThinkingPart(
                            content='',
                            id='rs_02930ea85df92c07006945adc4aa7081a39b9caa32e16dc26b',
                            signature='gAAAAABpRa3H0iNAr3AhGjsou4Qz5Tx6OFIok0ogB4uzbBZBCn_slzsYyE1ECQYmOaRCjMt7r8wyUgMUsUWLrWNO3WaCA-Nz4FS60MFyupIZnA-L6fuKu4HoWUqiBLX0qFTGjm3NFw48TWkuq2UAlDhexCDKfjbpAsucYq_zAKZdjAj8xajHdLmdHru_YKpxHmQL5ennRHSy-zoUb4q4nYdSFl0t6y3Jj1IIwTf0RR98pg1teTRD290Y720p7xpBrKkB3lhrpNwRHDbKaKpJmJ9uPrboTP1cWtnGOm6UrevY0glEyGP9xA7aL1mJkQqnE0eEUbPUopY-e8md_PRQ4JMikjHaz9Z7Xdj09GYEX-w0Ray14eIX4dD0XoVFHIOd19Mg3RKYKAYGz3pymfzSIA39Cs9cO4nW_1acn8s0XxfZKhW-CA3_c2dIk_Bk2EHvoYnkxPvk05lwCmHvRTuskJiFh-uqTa2lacy3nCzumuCuuOdkAGLBexYWUOZDW0E594OZG60m_tMSNZCyX_nh4fAgFHCtHLkOFt9WvMIMiKWZZrwMmMRBOUkeJe_xqt66SbHGajgWw4DLvz1M5OLdqEL5cW858rXwGzzDIWdnM0wpn-pB-qUoK3Zy9KqCpJsN1bDmzT-3X9vdXsdSOJ6A7VhUnid8qrzCJWWSYZdAS_zDAlMKTPsL7QfYpHiiVC0WwoO3Dl_9kMkOvolhLiMa6P9ogM0jUewAS4LSlas6GCX_RbMd32knXc3Hy3uLsFEH0929Wb5_cchq3fmpnuKFdXSla3_AyNkaDKq6mfjTpM3P7vS4EOtVdNE6XjtOHDPMIL9lrXONcFZ2KPpMK4sUlJTGlB26vtEtJA==',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Sydney","summary":"Sunny, 22°C in Sydney. Clear skies and mild — pleasant for outdoor activities. Recommend sunglasses and sunscreen for daytime; a light layer for cooler mornings/evenings. Would you like an hourly forecast or multi-day outlook?"}',
                            tool_call_id='call_lCfU4P0xSJSWrNBiBGpeKBdh',
                            id='fc_02930ea85df92c07006945adc4ec7081a392bfae053d0d27bb',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=284, output_tokens=389, details={'reasoning_tokens': 320}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 55, 43, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_02930ea85df92c07006945adbf4aec81a381ae9fe000c17478',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_lCfU4P0xSJSWrNBiBGpeKBdh',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_tools_plus_output_multiple_function_tools(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Multiple function tools can be specified with ToolsPlusOutput."""
        agent: Agent[None, CityInfo] = Agent(
            openai_responses_model, output_type=CityInfo, tools=[get_weather, get_time, get_population]
        )
        settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather', 'get_population'])}

        result = await agent.run('Get weather and population for Chicago', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get weather and population for Chicago',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_048b20bcdb65249e006945adc8820881a2acfaaa6407b139f0',
                            signature='gAAAAABpRa3SesFJPWsmOlY6Ci_ijunRL24IqPcY1YNVt5OGTzo9eUjLfdClmUea-Cno5dHbm5zmDOC5EgJBPKmyoyBlDd7-JS8RGYt3VFchlFv1DhRvpANMogOqfpHfh8Q4uFbiA4upICbHt021xBqk2yPsxLchJWB8E3-pB60maZd4phAU9dO7kGJUaC51rT979nfSDZnVyHCiYwquFf9HNlftk-TEiIBrQxez6x6pXt1c1cHnvRyb-s2OPcqv13nBP0IY7PudVHPNn4HEY65LlvPJ2EgNSe5rME0Rh9L82TOBsD8pdyzA3WLoUDisTXxYyqjBucEB-VLJi_KOG-MdT1PdJ84Z46AavAfNn6lLQaH-LJjRP0Xj-V6P9K0zE1P2NwWc-HQvgvPtdeJPgo5RZdQpgYRXnO1iRUtTG95NgMjftD9AV5vqPoOn62qgWrG9yoWH1sqH586YfDK4zFowQDSsn-tx-rN87Ub3PHlFAnBJumEFlYtSRd4qMO4rqrgO2P5_qPJCix2xOxKElYK8HQYQJ9yHTCrB5vLtexDXfuFbTH8eOF-6SZhcSM_JCRhvJccwsPEwlNBzjIEzwuJkxyEou0VQhS8xjrdFRgNL8ZpyPsalTI6Pd81aqlt43Jiv8-OOHcVAWcYV9QXxbxXHV_MOIpeKNL1uAFNpQAj4BeeoNGl5Ev29r3AX70eoLTBVwUIsfTHTakmuIjZiU5Pkn9y1gN8ZtD2A_1FBV5FCbxD1hoWdsSDP2LLONGeoyTdr18PRusJG4Wi48UlE27L06zS_cv-6YvjLck0hmHFNaz9IB0gJ-eMFD0BidgffKULaX_--HIjt1EaJ3rkBauT-NS7OA5iRGlEUfLify3ZpDgLnsFNFe0josM6wgDtkehAQQc6PPQ8ZOO14NulJ0Wcs4guJB5k4otpmYsrWOdZxc_wLbUTvqPjYd5VVsj1RZTFGSOxlex2p9nE92vkVETDtWoFcgET_T3BDXpHfYvXNPs1Km8nsNnHXKPrquj5tAYWIdbKmRVt7CZyEMTO2ttHrv9fuyInd-XSj-VUVK1DULB8pVkMeZB_ERKnJGpvPvwwsmeEiE9HUqiTqoBun3ZLiZsWyHFmOdHUQxB0vEePrABqiev6Px7ZO24xWb0D8XRhwC7GQ4aoy_K4Acle-jxVpl7BbDDQ0usM-iUE3dbae5xX8PjrB-h8U-sEojOHxIOYxJeyCoI2VDksV0LrU0V-p3xlH13zbMVHgqCC-kJOO6i4NrrgKeT48pNYhLRrlVqTillEg6MwYGPR9-zzVTaTgYXC2tCI31V6f4GA7z2gT8goq6dy6so6ViI2vYR3ZGgDN_oq4iyry30TTCd__OJ1hymAZFnZR1zyvzSr2KivDo0xZa2-_AYIGMf3myHQQJtkRTXi9TfD4WUb6wKnH1XTU6_LXGUG6CuncqtqbkIdkids170Gh_FZ3lny9_B8xpm_x2g-bUpx6FSZH-xPyH9oIVyI51zkCuuGZKfswnf7HnQaWEEN-rRyCKY-gv7s7E71NwZ8er8_NxPTTp0hLzyiF-X8MX0tAp8PvKuTiMqE9tczMLyQ77RhubQ85jOzjk6GvMSwGEjNqAeDhVWm50_tm3PVQGgnp3j01VB1-pQ8nFB8_e9Gg8mlMXDfunNsW8yaSF3mPPKFAxa-sj2pZSpqS3imKsyTHEnG_6WrUbHR9QeQ0UbSUZ4AguG6u9GjAKFOyuCC8BLRp1UCwGgr4OMBEIgV6GnrwNODNdYd2-JwgNnIsFqpuS8Jffg9RgEAm7vXUq6nK64vFVrPxXvuIoHcy4J0r8boyDbO9nY7BNkyB4IY4nv1vncVDXff8gC8dR_4rqBSJCIxRvEiZIOXPhv7Dzt2_cI1c88zM4PSv3oQ1e64fulXfyLdQFVPkPgvs2yefCajqBkCoOAZFHHG35H88MbWgtGA_76jXHGkYl8k2-VL5-GrjWq6N7aEFe4vY79uA3JhT8eQAvzU1hLQoS5G63lgwdrycmU87KfvgYe3gzhLEJW45A24g5gMjwiykyVll6x4_zF3cdErUmcVTRIditqZjqJc8SwfP0R3WhSsVfaxHjJh1H88PQWjj8ztPCsI6Hu5BU8BI6xHLKX-AlWRAJeW59Ccb2xGdtUYdsOzc-Os7b0FDSibz-g6_ad2F_tGvBZVGF9RiTYtVOiPdaC-1Yawx_BZ5nH1-a3r4Xkd2vxNO51z14tR5wnDZoKUWPMW0krsQlY2lZvCWwZkGd6tkZNOSvpE7AaRyUqALx-P-a5QjD0nJCioaN2B6mxaQQmkuaqEO9EUNtre4RYIYtVEZqTxUP2Tk3UzcCpnliz5LS-H6ddQI1akekoc_Va90hvXzICzIn1yyrRs0m3pBkRRai_3npRIMCi-fDk1WLK40QqI8KWKr7Irurx3TWWNdmfphw1VOxoJS8pJltP0tcckygcQBVPY3e-3qDBUHmmuR1K-q6z7OnnrsTMc3vzN5tcyS6QmFumRUQG5m3-OPzAtyF82wErvzOxjiFlJxSXhMjhXD3qFd3gZSJZ1N7pegtUIZEZRI4JWhH3zadugYARNldrG8JHLNh07sQunGqpVU8SFvpB4TL3T1dgyYPN9_yho6LQv3dbzUQDg1gdevRrscDEFS9C-XLkNJcqLobGACLwv_E-wV9tvy6RQv6mbyd6lxl1hGtLPUF9I7BQ6ukEYILTLJSa9u1b40NtKC64N3KR4zz9VHtRexisRQJjf6oyryqBkvTFxBbXkQH3WlrPOmimMqLV7wNvSBD6KxXcnMhmZAuwxgOCTbgsS2l1xfF2arDiIQDv2GA8YIIohns-MwOy0rm8o-zIQTZTjcTEg32Qg2U9SA7yAoZUkMe5l6H744-ay-fbL-zqdWK7Mcmrwr9XsyAQcKJM0_9aSfirnNpBCR7ai-_sZrY_8VnOTpwSAMdH8JXNAjZypoH6lV8HyB7SRffpbhO3x5UhzuYih_cQ6KRdWb-vJD_NXiExHOnQn9kMR0aaPeKIekVoy81VrPqrfmfgJC_dmxYRIG-00nyuHSJRPTr9CI71F2X4zX_uZv-PTWJAwV3W5N8ADytJorMhL4Ja3Hm9knKZTEveMLwpO8kAZgpc7xChLVUxEEkFL6c-6elBfOFMhJGj_NS_tePVq1xpw05IaZjMKWilUfFveGj2VLyBzW-6mjm0XQWy9wB4eLe3Jx6V-5TtRy_GTiFQjhhP931yijy8aWNwsP690FHaNFEe5EoSvEFxUcpo8b-XBNEUhX4m9gM36VLWj1TmBXzM74TPL1Vj6DEjAvM0z__b5cKjuMnHnOVvofu1bi0_fUZ-NNCaYCNUjAziIHZouYJoLIG1L2KR3JH8waMzBeMJGF0heQuGHjRfM3hVnLOaCxQphCadzYISOe7-v1mzf3nbFUyyfhuaPQ-_AG7jCHUBcwB2SP7HNZH3SFISjRNlFMHAC4FGBdA9itfIbC6hvtYpMBWtUgtjG4WMe25au2TMkFv9NND1VAvlzSnfTK0M1f8F8YIUXBRUp7RZoT8lpRoKwUzqNfp2N4BB6a0qTtW0urs_8OA6c3cp194oTHMcg9l1RYzCKL7XROGHd_QhgmAjsDNA==',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Chicago"}',
                            tool_call_id='call_SLfTLXh0mXdyO00yhf7HrzOh',
                            id='fc_048b20bcdb65249e006945add14e1081a2b307ee869796c949',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=116, output_tokens=404, details={'reasoning_tokens': 384}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 55, 52, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_048b20bcdb65249e006945adc8388c81a2952779bea5ccd478',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Chicago',
                            tool_call_id='call_SLfTLXh0mXdyO00yhf7HrzOh',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_048b20bcdb65249e006945add2de4481a2b2fe03b588cbbc9d',
                            signature='gAAAAABpRa3U8izC0dAFRVLQ_RZLWCfpZcWmdMoyBFdJedKbmBADxneagC9pSENyUnAtmIv2HWvnHwMxz15qtH2j5um02H02rsspkZfyE1ankXqHk1Xy6iHGyl2s4W4l7k5bOttrVK0hUUykeTFBApeO5NkWxD2bHacbmOaxmv-M5uaSXqsihcwhffI-e3mIMQ-3H-yzoK7mYsoiunc-YBXCubp0AsbnayJk3WIin4cGSHFkv0CPED0qnckT3UeTNmuW3QY1eqYTdN2TTo3ZO6Gj8i5S2qLE7yZo8UUkKnTmV8wg_YZ-13QyvEfpsJB5gW5_zgejtb4pbvcDsMN0H8UhXlvD4xUckFlZpgfPP_BxsbNMx026GrjbJvTOcIP13q7iQ6Ep2GfQD4haXhqfo9cg-c_4dC4AYKzu9JjUgfpDjTmx9iacMag3IbrMsqwT1UnKkVyvfVcoNTXkwe4lNoJRul9sdJEXKnLHhpxAunTwCLt2nr65BD8ebzxLeJM0Yi7rO_jOImCPm-WwtnreA2NWirdO4haBfHv27mD7V9JZZryXn7MSdLLwZR4dWPGnMupWVERuR0AoFZOnhvQpA-m2axEg3AsQ3O37xmAe_Oxpn5VKo_5kCoFnhWQ6hFet5ier_MirP6PrYMvuhXSp5YPLtMJfXCHXCaMYz-uz3phxpp7CD06w75gaVT5gWl1H2EHmUfhdiKEq7UhfVXozXUK1HHBIv0yDaJSgNcSsb2bCamcQpSmo8DQYMTTVIMvGo_Z2dW1GTDE8z03-KhdjDjzaI_oYOKHsdFT-IZBqwyC7pkZd-CADM9oe6Cj9wEGWasdTyz8I_W7MigeuaahS52m-cJFDXvkUO-ntpcwEa3fqzU22lMKVXnzL8IENG-mQfpOqfZKInAvtiP6GVB3xwnamJvxxRaS3YlY78ExQAFQwqNedPlrPpIP4JvWq0HayaIReSHCjFf_lWspEKFv8yvSzrl7SF0J1-h0NKtMfcLVLkfrVXdfJj0fJt86Gduo44Rr9l61d26NDaGRLB7S_Q09nhB4v8CoI8mw4DSEeOvMa5mDpOOiFNFwX1rwcEkshIX3w4rbAMoPP',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_population',
                            args='{"city":"Chicago"}',
                            tool_call_id='call_4et2CYELlIo2iwfr6uyPz1rw',
                            id='fc_048b20bcdb65249e006945add3c19881a2b3dcef2c3656d282',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=539, output_tokens=20, details={'reasoning_tokens': 0}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 2, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_048b20bcdb65249e006945add26da081a2b4e1506d98f2a740',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_population',
                            content='Chicago has 1 million people',
                            tool_call_id='call_4et2CYELlIo2iwfr6uyPz1rw',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_048b20bcdb65249e006945add56db881a2aba434418ef6cc56',
                            signature='gAAAAABpRa3Zq5mpySGzhsOH3onu7FPNKOmE3cZKG-wmNnCoCgXDc2ly5gtq9HSWaTdjszQ654G7_OUDJouTr-S8hPNQmnn5Kj87hMibSJsdKDRfRYRREn_g4X4JK6JYcl3BBCI1mPHD4wfntlTIi-n7M9PDkgtAGZX72MvBu4dMby40jTZ_kODM7gcCGUbTSerPneGzpG3FNIkGFatbllv2fG3X1Y4dqocrg_FeXxxqHcWOYlRqGD8f5ZL0kI-j9Fkv-k9f9gYSUIjg4Pyen6m-IEUUXFKyjHuFMGZDBNDrik3AD56dD8GweaT7Efde8I1Kc2BSQ1DV9k_eo1cS5KMoEyaEndVCJ_e7L5f8UYTjIR0IQ_lX4osO82_y_6RbOqqie3BYuJbY5qMrGwkcfFeerb0Bqz7iNM1_ZUYIVwyzcoGOnbp3Zpc9_uE5I9njDVB8hdH6Mr7Rk74IxGrV2XNEdrRrpN3GGViIFi3vNLW1BxjIm8Ki7p6GH5AofONYgzpCv_rad4czarK4kmELQzPAdAOtNxcmmGE6HgnW219erCLYmWW_NHyjkNrownSJLC6yQc-XudeIvOsXCUK3uXWpnRWfIRJCcK9zvU9zAYa2Ww3eRZFmYB0kKdbYCs7aP8t8hjgeH9ZRaWBiH9qaHmDmct3ZKc7_saWBENgduuEZ5mqS9MkFdpkZDlgDWEXi0_b56Z3ICRiSTvM1k53-MZaLg5J0C5wH6mLH8elnqDLECN2lJvuCl7RrgcUt0gpLXIZpKLi7YnzYRfv5N77wFvGUT5I5tCWW8ItcS9wi2Y3BygNPPhgnaMuhDhrcQxUMK5ATtJhogi5oI6jv8VxbiU8mio49RZJdetokrU9sTZqyTKffn76ququumu6nsvl04dk9KSprRm-HdjJCJ0AIfLU_6rhb7slmtBGUhjMxAJp_M71FMPW1lkVZweLHYkQrCPtr850sG1px3hhbvmdFaZQNoWnc8uRAg7JQf9aWOh2S4uxx36r6ZuFD1ZJgRRA7bga_xQ9waTkyT2BAUJc2vLF0AxtHs9iz171x7WmOB0cMHmF7qzCm7dBhYrnxrSOzsnBpfRb94qC32whNjjqgtkm012zF_E5HAucjIskiB2uZlYN__or1ZYvCA50JnV6nz3Q3Lq27fECROnBlinQp81CxFLB9yvyp2pqkD_UAZaXRavWW5ClLUj6Mo_zr_QJYKo66zrwnvNB7id9GcavJm58fXS8arjIIxGwsozrHG8eFoh2AbVSSLZztU5xTfhda1ewP5PR4Ntg__uf-YhyM-3qW27FTEfkqLuHsLI9LHiw5qCHJGbRrBTlDcmg7i3DImHFNQ3YtF94YQ94ePkHNIlCuVam3K4foCOqgKKyJgLsl2XQNsXcHsTSVnhO9UcNqCdJ22Q71AXFRF5ywHaIH1J6QkoYH-Dh5Sklht0yURbxqwBz0TDLNNZQde3fJMjB7HAgtMGmhb11HGXavBZ3dnwaPMZIklslE-5KDxwUmxK3RsrvZDVC0pcQ_-YMwo0IC5AnfRFQowwMaRieVTKHAAlkUuZH1EpEEV-KEki_OyJYfiyV93ZT_4xlJy6cfGPewwnARB5G74NOhPoQmhZw_NamXorYnpsnvcA_wK0gw0wqsNo-tpQGOwKWWmuvc-MwSd6XNc1zM2ss0JkNWVjupdCujVWknt6eSmdtI5_qr9F6D2BJF_6hOC-1wOvp8U3A1GOApr63wBpTx0UY9tOYFGvRbVLD1DCWq1k8xnese-Rx-MJZagJroJeBrhCMKrowENKCJpBMreqP7KYBhQTCCkV8mfJxfJPMiZoxtv1anmyD8Xdt7YaHXaWmIhqB0_jCAOwhXC3FT_eBi-Q1OIbGpexBJaqm_M9qAcvMOZglCQjQ89_d0GcBGr3Rg9bgHKA9mHqAxxdl--Nar3VZxfQ9ydaliUKkhhf_U7B_mM0da_YFCeE9UL0PF4QMmbKUBtSbq2lbgHl4AgCrpVwpRxyYRU-VdYY3vkKz-07l56vua7o-LfVM0oA54yPszQ3Nial2uqn0YcuR7SHlWqgUkQtxg7jsHA3zc9vRqcmEViA1mNGnc_fHme8ubb1w=',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Chicago","summary":"Current weather in Chicago: Sunny, 22C. Population: 1 million people."}',
                            tool_call_id='call_IjvuT2OKlKY4fdSdZU7EZjG2',
                            id='fc_048b20bcdb65249e006945add8a91881a29aa0e8519736ba6e',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=617, output_tokens=169, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 5, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_048b20bcdb65249e006945add5176481a2acd101abbc4ef0aa',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_IjvuT2OKlKY4fdSdZU7EZjG2',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )


class TestNoFunctionTools:
    """Tests for scenarios without function tools.

    These tests verify tool_choice behavior when only output tools exist.
    """

    async def test_auto_with_only_output_tools(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model uses output tool when no function tools but structured output required."""
        agent: Agent[None, CityInfo] = Agent(openai_responses_model, output_type=CityInfo)

        result = await agent.run('Tell me about New York')

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Tell me about New York',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0e706bda8f41fb38006945adda7b6c8195832e7b5bf2482346',
                            signature='gAAAAABpRa3mGAubjW_qB5YJJz1L7N6XyHkwiwa_6quW3bdQ3T3m7jGHe_ESHVNxc6nqIU_WKqI3NqdGlO5B2-JZfBKCxY6XlUGGsNt1rc4hx0IALokqkvvAEOT6siL-gpYsAbfICmDdUAdcQ8tbIjs2rLrGJYvjcIIQPDwYaDGzvlwXDDe9KUke7aV3aMnII5_l-fxnTbqXSYcWPiB2hNuqOXgnbTCrjH9ZkhmcVihn6qMav_WCkIz8EfNV3WR8J6mgb8_z9GSn5AqajYgqYCRebL8fEXXrPwwnATX38lBxFXP2moos9pq7CQStlMHZZsfupwoP8BsSb81p1re9mYqa4fT0DPRoEoOBl1kUI19CDm_jqUoiK9PtWSv3M9mxE5ZQgZ6l1um5YR55CBqoV2MxOEICAjD2-yyjncvceG0fF98v7DpByWAZv1zjgTo2_rvYgIjdWo8c6un_52LYS86sKxi93AvokxuQsV7NDIM_69LOzXmgjemOnaNA8oseXohyFMe4MPaD8qqPntKvw-D-ZuwxWzuu_u7ul8rH9zOE8CpzX0ANXaavHYgQ1TvvYDQtPGY8ZJ6lTBsce7HQ_8Q_7SFSLCliBuuPLccsmh_RTo1ToCVMHHgGPCH77ugVIirAangGtMHa5ihUBKALKhw0oN1vDKEDtCKmtg79Nbt_YF66riu2burKixFvmjyWWbb2LxMosweKsBgZ3phtGSZ5YTDeh03unEB1WO8JkkzqCTcfNFkiHBL_iinFIqQh_C3V9sfKIEV2Ah8bu6dMbCan7VPMdH-_Cw5jzeKhVKv52YJcvML2-iesBv0Q5SXe7f547heGH7ueBBDVNpzJ9rwBoUhRiyqEaET-j53Vp90RPRnX8zAf6Kb5MHkpYxq8_iF3tmucuSoryflX4Z44voNXg2p8bSzKWwBq0UZHRcQeVOC-oJ1BVjZiRRpsFcquuZAqEI5IJwfyvG7HI7aHIKZQ5UYzE9dwdC3rb9OxFwjQ07r_x__BErktd65zL7yLE-IQFZGX7g6QSQQktc00AtikJ4EbRXYy5eYYpeh2ftfy6YuHh-W9jprQfhn-D1HzuLn1IWO1K8Bbmh-nSgOgnjphQj8u-7Yc-t9xLC-QZoTLKbm-YVl7u2o5OIxAEFxXtSUcv5WGvcDrcL43DUIU5hNamXZT3RnF5fkyB0tJW-ZN7sTtlDN8KCSyuhqL0j-RPBrdVp1wldx3umCWUzTZmPrYH4K0x0MGnymDSgh6Qw1IPxHtlPVAGD48Bb5ZH7Jy3NRHkAqRGur2fH6_KnUMP_bWxA5T4_z2slJneRLLv5pDfZfMD2OVCgO59Yx2a3mCyikNHHq4xp-re8_-XlxSRx-lCmPOaE94wXd9MbtNEH4IcGyiHy1LNbdD_3U3z-0QHjNvPWFClfZAzQSBfBObA-j5wNH81y20NLyDWTTPPRlFgqebiMLkFaR8lwKZWUhrugNTIoCP-wN6yZxxOcKJuqK1Jk2s0FtHRR9Qa6lel9Ijt3qYvHY3dwWYIj5ziRFvY7wvnMiOtvyz_x-7coUPpx2iTLQAAgSEh1YdbnWsXf-A0J-y_xVLTLddz3F588s3Al0njgsjvkhHesvq1qJE3QuDF0sNAOTr-8Lao3dSygVSzUBvBxtyG_EU9qnKljyi4ftoTy1nljWnSySIQ1e61FSvSuHZULcwLNlFJBoPTe_QPUjYN21NQZFNrutLhovczRKMJcpjWO-INIWsrC7ib-cHG8UTDuN9Wvj8VTC9Vj5jeT5kAaUnfFoSv_KWcvmdYjHqHfujjCPsUMqRB3Q3aLN6IBkAHxtlsFmLiW52mpABvhLpsc-K3Bg8hIvHhT62qrzqdxf_Bvo3hXnJIEidZYVJ7C5IlfhymEILo51FfjYFABkSz-1E3qHn3TsF0yy8TTXlNTLyHwgzQeq-A4ZjXQaidWLu7Dc8YjQaFg6tVEjF8H4WJDWy5Kdq4WqKiA53ikYayVHE4ymR4W1kvw0lTBrYWGLHtZ53v52-nWns72_sh2hff51QozmZImI83b7jcHWr5BE3K4gvmt0Tp4J5RuRV33Gshb85VDhiSXwAyVHHikIz92HEgPqqn44gpeVmZwCJ2qUO600q8RL1tJNhdShL2t0lSYltNy-exEa4nOIjRU6axHdOB05NcuQdI-ap5TxDO0E64ktAqqKXoD9KJcAOcCgkE0cPUHDKUCrquDl6l5dHFspFmasHbcjFJomwkrI1-SOkLomQTlL9ATo0I7B18VF-OVoPwue6AY0pUg-m3OVj96YkGI0Hly_H7dyQXY9upwHQEpCbX4dE9FCsGCRyF0F62stdVFySuPK-vZ7Y_UtmTmeGD6C-RYwX60mFkdMQm56MKeZm8LkU4JKsQl29TWXzddZlMKZX40luuQTPb8TknPdqMR7IShqADJGH3NEl22z3UF-0ZzvNuYER58KZzhyr3wzhlnDez0gzARwdYURX7j-SMU55BEhs1bSc4uFywlRgwibgjfAH7Jy2k1K_xTiaanEuWZku7ae1KK4mywG8UvVpvMBlZwUMW8SwtO7R-MyCWCKah1FKZRZrzOR8ZeRgeKVTMh2X9wVQFn7nruKIQ5Ey2VDUPcxiycdeP0GVAltgszVP1A0tD1Rbcv09c9X52WDy6e9DI0G8JFUA_SxKMQzKrjf8f8VWN9lyZd8S9JExl_CYD4HU6nObbxEeY9jbUUZKAoXnO32aU-fhQ1yIlT1Du1o-LKraaOPZ9y79iEGOAsvv_VhV4bSfWyFhbn8Sdjz8015Z2Si51YU273uxJSpAxRQEvW8r-ojrKzBnwVNxFJGBhbktXl2_cNZrINAlrFX8gSh5h9YLmGYBDGTtt5aD-x-raRMaKiUbaMriGjJ0EsJ-Mxneu-fhwf0h8QV9XffSkB3GwHSltNnbULTalAFvYR9XzztudYdeyLL6gD_XV4wlJDTt83S6-i8DCaUYUKRU1o3wo6ulRInOk1MPYjRiFkfxiA7T8fEJCuTtxuYV2-M4kkSZvoAUMe4c6AqR-8RgbGaHI80lZ-m1sPehrcq06r0L9VO9lVj-RZoxafvdwqR11nIrlm_3AzJcZ05rcgdgmwy-Q-czXbDuy2srKwuj5SAuMM6QJ8CR_J9Elkavrcg9rCI4_SwjfHZPFI5jAndeWdCzspuj9qHJiHNHkQGQqHWPQh09P6_9BTw6qI160toEetwgyaB604P3mEa5dwgvGhjcITlSgCZiPsSJngQobJkgAjwhDGZgGiyJyuzr2mm2oKrffjXmPAzi4Ovs7vxC9BfEKXiuTm3Gt8MTYkE0oXbQY7hVIImmv73fMTSffy7oFEpt5yM9Sj0QvqJ8EJVPPtWQBcdmuEF6S79VIyyoWhci0OGkUi1gAej40ryr3ZQG-6VaC1MbBpC4CrQHkumUoFi7Brt6Cg2g6opx-t0c7ZVWnbHZERTkhhzqKMaIXfcd9x_5ioRay_ml5W5z9SRniG7ckcMAEgZbUZ5TMxsZGG0isJld3mesbCk4Ge1QcBEvfBWmrBprXJbQDpdFTqL9Zogz51TCwdJ1SSoxVSX1N1rJizz2lL55MQ_KCPBcblRxCFo1LvtMhrafJz4-QiVe0Er_KsPfjXifKHKWw9fTSL6iAwBXa12-Pl70k88yMhdHhQJ_tCM4hZe_tjqc0vqreZ2Lvp64OiCjn4Evr4_ZL_fk0meXBvc14ZyVNcQEcsXVbIYxRp2_ewsaCfPyhC0eZVI79i3vHWPh4xTVzzHigKC6GEqoI6hzvZCSK3M7QJ-RFvRlFAsRifVlIZd7qroaqPnvb2InM5Zhgyw1O9uIu9vk5z-JPfFQ4VFcNSei198PH0RXOj-A-rAzMGrSPdF-VXDX1K2lRjR-XWPP4pgo-dt-INfIT0wa-6RBpxHErKoVCIpkHE3XXL6VmWMCLEj4vzZAaDgDvyu0RZGi2YmTg9NctyA9-eIyKmhDrU4Hc-Rp85Q5HmIe3d-GBxo6wi2HnmXk9VCTFRFbKq-VajvHWVk6Nv54rL0CsDcqJc4h4ZM-70-OQYCw5oYDfBlEwG6sOxcqLzt0Fo7LgmrGFYMyssG5ezZAjIu2bzkP1SUOgQ8HIfIolTLN0OxYZxEaQNmpmZ8t8cmtZyuLB-KPltnh7UF8cuJd-1qPT9eMIogDTN365qNysBXuphDHfoN0QxtY9eqkhds4oKVTno1tpaZq15Lt_RxQ3UsLf-gdEgnPXfIOBHdBqHV0_UXt1db_5hn4Fkd0JrcPZfAkZB8DyprgcfZbnvqmhPNxMsepLnr6tXYdEr6efH0rbXAwlinwgv7xPHpU1CZ14M6ZbE2hOFNLl2PAU7yK9uyGXeVxZ7Y1NgHPECKitW_RQEee7L2mwpyOjA==',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"New York","summary":"Overview of New York City: location, population, five boroughs, major landmarks (Statue of Liberty, Central Park, Times Square, Empire State Building, Brooklyn Bridge), culture and diversity, economy (finance, media, tech), transport (subway, airports), climate, visitor tips (best times, transit, safety, tipping)."}',
                            tool_call_id='call_ruWSF42LyQm6DD0SEzmOxG8i',
                            id='fc_0e706bda8f41fb38006945ade45a8c81958d19729cc9babf44',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=604, details={'reasoning_tokens': 512}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 10, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0e706bda8f41fb38006945add9fee48195a51c2a5d74af5d46',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_ruWSF42LyQm6DD0SEzmOxG8i',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_only_output_tools(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Output tools still work when tool_choice='none' with no function tools."""
        agent: Agent[None, CityInfo] = Agent(openai_responses_model, output_type=CityInfo)
        settings: ModelSettings = {'tool_choice': 'none'}

        # No warning when there are no function tools to disable
        result = await agent.run('Tell me about Boston', model_settings=settings)

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Tell me about Boston',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0a3de6f9c3c28af7006945ade6e8048194bc3e6c78157ee9c8',
                            signature='gAAAAABpRa3xoqrkFJ7z7A2vk1rD45NZ4OdhPBqpuQL-vVRLIntjyyWQn8JIoUcM00pudk4lY3CBAINx2ZbYyMfN-B7McnQ-vUPYk6EYdOXp_4uJ0bQHHnbJFGKscnhFYVdMtF9xUpiQNCgY4eyPDfr7YpwCzZFNh8L7iddQGGXOJxM7w9akSqVHjqrkInVsQtLTtP225tS-zwnRq6Cb9X2fHOB7RwiDpvVd6ULrVYqs0adCY6LqZITQlxQwls7sLo2aJ7aKyxpjHd-TrRsxqGNEpC88zAItP9xtVPmDzToqkqFkf6odtE8N7LQzAEYcn8KIjz7TXe3dRBPVjcbtcE7JVFO6lWi4Fcb_qnhBv85ZdX-kNOTE2Vvb9WLr_pz4xUZ1LCWcz1mhV-d4c-HF6oh_GZlrNTIN3z7f9h3bDE6OzvYtRw5G2rd6hk0wegfCdEmA4DO7RTKZ8nU5qE5hQtY8gjSNtRzB8Bm__TU_fqaH7DugHhOGhFHQ8ooUNUxsto8kw_cBhhHltl86LShQcMMkTIjCOlQrwcjIK8Pk9GSNR9dRapZkrmqQvmeN3OfRGh8Wz-STDKQtSjiH-4hP6hN7VX_LNn42LdRMkJhszgBWd6CTLgHYuPXrqTwPP6qOTRqZrCQQ5ULI95PhGXfv1I36ZbAUCEvk1I6q4EQosTm_XIU_sdkknoHFnhGH6xVegOQEJEePyDWcUQek8ZmYF3hoC1WuAWqu2Akhxisl7lbzaRViS-F-9RbAmO_SP4FeUTSYkQh50PG1AdJDQm7XLvOVwAgJ7uO6e5m4B0ZLmrdyUa5NiilyUh1Y1fK-cl9e09QSa2DDY1iHPSUilnQO9pZitB5eUhunoNOgGOstQSn4S_GRS7cTSxg0Qi25eF3zLHu0c2jEG_h89KDpYa1YwNJleuV8dKwgkj20zshGN0AcGw5UIXuzUM3wBIFfo4_gMC68c30SWHU5nh5CKN7IAYBRqxdkOOrBcR9bgj4MkuV-p0FhjZ6vL9nGOVgblh7cAVj06DYErTI29s0_7Cpb4yMimFx5X_IXFbyBzfN8-drpN9jlhMpSU1vhzwITbxh9bEWhZc6TIGol5Lt2aZiRevJtmrzOLov7en-lqbF8fDbDuceaTgguAM4wKu7gunCZT-B_k0LtnM9PFXlB-N4EKl2B1iH8GBecL93oK3CFpwyKL0Rv2TXYHfgAtoZabW-537SbGQuzUoP_N3Hkn9TKgeursTyC2FvatwsRishYczIrCvK5_OmwU7K-Xiytwy9_B32vcZyccn1mTGjSRKdBV3vazbkCFmUwo4L--eYl0f_DTb7pXMjAkK7uqR2-yE2hHN-AGckyzGNvofql4WaZ3DR2z2v7aYNXTnpnhbEYEQjzt_Bx7iCohA4ZJkyEhlb9FmxCvimCWRtQ318qN016wA9T8UIJnD2ypO5mKaNR0g4a8hV2FbG9uoOsEDSjAOzIf-PgynMjaRh2_ZqDEbhAlDCj7A-qcpygKchKujoz5ZlSHhnhFDNQEOnA7DDDjE6wZbERTm-RknXinP3CdNUGIHJLYS0S-6gwLYsvaBYGtaqAhM7G5BXt2PKreO_u4qck0vx6T-itPr0pBBDQpPnoc2God21DUUmnWPCBm2GoZwBBekP4RnGaVbaYqR9vPElvbyeEQXP1NZjasfQrH9yW_wMrCqerU8IBebWcJMv3TKLnjuMAZd-SkWk_N083mF8SrqSFw3uFn54SpDu9lSjDg2qfPnGU_0xxXyRv5_9AOtrhY6JkjZxlBDxXH8vAShLxBxMVuC3vMuPTzePERPpbjsq9sAPe4Ixls2AVstv-ZVpAC32dO2o07eMDPMABaFA1duGnnDxlT3qwb0iCp7Y6JKCQ5I12nB1hyx-p1Tg9nCZkkxjXUP5qT9fhooyTc8_HncMnCPMG7Jtnq5UUl9xFHxZa8j-M3g0ncSv54Dc_dz3J4cKJD8LxljZW__V7fWF-jCZ-wkTI4oQFyFCpHjqhp41Uqpx7IEf3SVJeUGxAw-DbB7_6By0tDbFQR_3bT19U3jsuYsurUvp0R3eRGHCvOCbY83N91mHljyihtj1du8d2LNBss9Ss6n29-0cKZBXNz7kL0N0-t3d_v7fY5Yy12OCXaF_cdWUGdMDbeS9ZmthMgOdKkZrSSPAcQSyQtnUugDslJa9WkdWlo5VYR89rMVYHt1XI-p2iX4TVxViTd3ChPli74FAF0sf_vqkblks_qnYEJfIehNqN2y8mNyRZPCcXpO15usYfDApsfDrG-9JlIQbsXUnLBMeavOQD5JaCrCM-Y5GbSDyK9rUdRc8pLZMSY_iBQsto3w==',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Boston","summary":"Boston is the capital and largest city of Massachusetts, founded in 1630 and a major center of American history, culture, education, and innovation. Key historic sites include the Freedom Trail, Boston Common, Faneuil Hall, and the North End. Prominent neighborhoods: Back Bay (Victorian brownstones and Newbury Street), Beacon Hill (historic streets), South End (restaurants and galleries), Fenway/Kenmore (Fenway Park), and the Seaport District (modern waterfront development). Major institutions: Harvard and MIT (just across the Charles River in Cambridge), numerous hospitals and universities (e.g., Boston University, Northeastern, Tufts). Economy: strong in education, healthcare, finance, biotech, and technology. Transportation: Logan International Airport, MBTA subway (the “T”), commuter rail and ferries; Boston is compact and walkable. Culture and sports: world-class museums (Museum of Fine Arts, Isabella Stewart Gardner), vibrant music and theater scenes, passionate sports fans (Red Sox, Celtics, Bruins, Patriots played in nearby Foxborough). Climate: humid continental — cold snowy winters, warm summers. Travel tips: bring comfortable shoes for walking the city and cobblestones, use the T or walk instead of driving (traffic and parking are difficult), book Fenway tours or Red Sox tickets in advance, visit in spring or fall for milder weather and fewer tourists."}',
                            tool_call_id='call_IUcMIrMvIy80BPfbBf6gzo9c',
                            id='fc_0a3de6f9c3c28af7006945adec0c588194be9a95467592db7a',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=49, output_tokens=496, details={'reasoning_tokens': 192}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 22, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0a3de6f9c3c28af7006945ade67f1081949397585675ead991',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_IUcMIrMvIy80BPfbBf6gzo9c',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )


class TestTextAndStructuredUnion:
    """Tests with union output types (str | BaseModel).

    When output_type is a union including str, allow_text_output is True.
    """

    async def test_auto_with_union_output(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """Model can return either text or structured output with union type."""
        agent: Agent[None, str | CityInfo] = Agent(
            openai_responses_model, output_type=str | CityInfo, tools=[get_weather]
        )
        settings: ModelSettings = {'tool_choice': 'auto'}

        result = await agent.run(
            'Get weather for Miami and describe it',
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=500),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get weather for Miami and describe it',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0f54edbaf2148c84006945adf248b08196a95096f6caea59c5',
                            signature='gAAAAABpRa31rlFZ90Z86Fnfft-T3vhzL5qgOeTjofY2gtYMOI5S5z7vj1bbATjWJaCumz8IctA40XZzUkmAKTG5bPqJf3l0HkJih6-Fj9Tkrk1OsivJkEArMFnDLfX0o9Mlv-LY0w5w0kEG9zu9SPchjAwpoSFiq1b29BqLLZAVfRcqPcvKY6Y7Ua6ajPqwa63G6ReViehVgzO0notH_IZBTXAyHcwnHdpIZpfx3KarnN8SY8x8aGFN1TxbPjuVJ51LZJhCIt-pKg04Ya81SkPbLfDeGQEQT6VAP6uWjEPU9qGmij7TjK-_wzbxSLobjL0kX2vaVOZCpWCwOXbbwlAmsOPTWAVoZzbI6fzk0Yakzd8a_VDIB22wM-auEbsv0mSQPG1gHbpTk0uqZGhreIyjQxTSYtdF7gJweUQ0u3L7EaJyNXk05oqSzmDTEIi1LaEc1qAvG4Rv27wJw5KFAwvQwjN9uY4xWyZHpJ-PHvUhqSu0RK3VUq8Riss8v2a6lmkjKUjeHK6YEBArQk11oCD0ctFV9ZX96SZzqG-M2PQ_9y_QTHJxFDwj9jyuq7mwBLvOAkx37MEaenuEhPARGiyQfEjK49_mdUFOpYdgZQMhHVW6WWUQCpCJJn9fmNVst-7eV8ueI4lQyx5vGGx2Ze6gfxFw04Ne3UgyqcXKL1YG37KOkB6YGgJLlPEr-MOx2WAQlJwuJXohw-7WZhv8Yp7i8_HG9XUThr8RGsEWCrJVgOoP8vaOb3m1cMlIQx95eqNCfFkS9l4wAJqhFjtetIZoq_69B6ol38vm8TDyKMujrsozUsyEPak6bU8jDXYbO0hJevvb2oPyOOQ9WJRQSEdxuyB49BVWPq3l1Y5UEfSwsB9Oa65Ot-7_ciEkn5G_e-lSq-092YpB5_x8B0j5Ghte50jn-y_DXyLeLjlursloS4YNKc1JAgNyCfQVikUoSOEJ_vNfR9ixvqSEWEbzYW7N8zFXfaEPOBDv-OkACp97_EwgtQZmYSbxd_REBCwQsfSk2TWgVqy9ga0i97bpyMgN_O2-hdG9aXc3vG_Nz1Fz9i7aCvDJwOJ-NhQ_iNVbHYIGslYgPNckxwGMRwFG-rQhzji3sLZMgUn1RSQXs8HgFhL11jimSOJBwZ-LSZE_bU-Cae80MtbEUM6fIDD_dc1hwXQkciewRGiP_ugs5lebxPv8e6z5He_bbnzr5GvhaW0e9bzPTjOV2sAJpOUv2IDWkxA2EzZTrJYKI09RmKE2n3cdEn7Du43yi3Lcj7Gc3zrjDaGx-pERQRdFEaZs9iqlGgKjNXVeh8m2RipMTZVG5jVN6iLemfL5FnSFjFM1m3vmnVU8VKt0LsK9mSmz2zbCOu-AS8gj9qLMYvRWucYIBj_h9ntV1M6fADxaoYO8oU3YXp6PwcW1S_h9IDvblZoyj-T-tQTHbGRDQhD0n5KRPL1IamhjaU_rpBuSGQvN-DNtvIBEIyJ_1VehaIGNkpHunfNHMpKbyaDyfh9wdrQkmBQnu7tPMKXysKw0Ng2ARbNZBjuVtX4xq5IEs2dG6WSa-EHFoMQcANqYohcHSy6oUG2s9u3RbzHS4dba_Rw04SiU1yHbPbE2D1Xqkb4YHmgasjy3SnU1OxtTikGeNL5iWxFTvNQovcfm1GGOGVCgelkyv0s1w48Ayzxbw01R_Qw8oodrl7ukP21rtaVlylTLuHEIuTPgyUCdj1ns5CDua5gIGOUklv4uyTvqHfkFityrw7NMRhPBUxP7_aIjntuYnO6qwudbSqk-1yJYeCcdU7q4B1WieyPXOhBPvnFI6t07fYObCIsc32A6dcl_3s_GABPKYMgnJr4=',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='get_weather',
                            args='{"city":"Miami"}',
                            tool_call_id='call_lYXStgN1Yy9d1dZ7tF9Nt0OM',
                            id='fc_0f54edbaf2148c84006945adf5169c8196a7e7eade1e5dafd0',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=74, output_tokens=148, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 33, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0f54edbaf2148c84006945adf1f0308196b8cd861f30dc61f9',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather',
                            content='Sunny, 22C in Miami',
                            tool_call_id='call_lYXStgN1Yy9d1dZ7tF9Nt0OM',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0f54edbaf2148c84006945adf62c08819698137632dff26c86',
                            signature='gAAAAABpRa37i8wqJdIAqVcAKndMVR6rKPbfawEvZn4uy1kvcO05BHaho3NTzFl9DhnieGt_IAWiIsXA3HTWTzYPM93NROyK1CzRCeBkSsMf_75rAy_s6i-SFezN5kX7Cse2qYQyWImPtD03BH4jdCGi22ST7GLO9i87Swb8gORNKK6OqMqKx_IyhcMXMQQLexl67Oz7AJUR70rUJ9Gep1u1FupmKTYuXJZmKGM0RwXwmG2fm5EBiHNJpTJGrgwAaOcL0BN7kqF7Y_2Bu8_LExbgbGSSSLigiqj_Qee9MItaKNATMyqyBA0tHgMmci0XNIxO52Qino0Ei81oRGFFd_Vmh9RKSz1j3iKdC78PCeEGhyQydKwGDQC3FxDyZfnKo-OhcvGbT_8J7GT611WsjP6Zo6N5hH8f0PPVEmLXS95qNV05c7z4aAcMnudnNKOltALTDGs1I3bmPvMu4nJQiUhel-azYa58IQTyf85CuLd9iNaLeiGDRXFpvBUQBo-MCeVguQKZiZdPdqrCJmYthBNjeLMojf9WrzN26SFx5DMkBnvQHB09Ji-Y1DyQ96_zNDoUBmhpB1VP8mhV-7f7EiKbpDIhC4vpBZTLLlXKRjUL2upsX4K-JYofx3-R7zvZBqN2jxSyI0s2xJYI3X31576aS4fVgxKzNnONj-G-K-qeUG7YBHX-NVlDLVDQHN82JS2w8PqlF4KELZPQAaAEMFJsgGWNrdI_94nStjwZ74dsZL65YeAPZLrm1wR6iPlpMIxCDckiU8LnOCOf_PCgTeYBSTrWpGgeSxcE5Pqkq0RUyAhFa0XyquRS0hy8kjP1S_RYtqp1Un33ET708YSummuTeUHFSctsa6fyswm5LvaDfPinXWkpy_1eY39kC77UyYDfYKmH41WI3VuCqtU0jQgK-l8jvyzmM7JyR8sDOpn6BiyKKr8ARppPXW2XCK_Ow2QREzbpSAoZ1ugg7Bt_Q1AC0BoQTM86IXWiQHKw-sYbLGr_d2XdFbSJlgwDWHtHkwERxShEleI9QErHshk4Jz-Mt08td-VADV37l7pMbFeLbLEZyj_L9O4UBo6dsod3bdPwGVELYghOPdvmfeBSOwCDHeKtpGkGpqgGTHnKtJhDa6AUk7am_QSSgG4IoVuwnWsA0nkg9nzVer3YcN5JUmZ7KgDUwg27eYHi8Zt-oGx1iAnAj5lUpHAxEYbnxXdoEESxcGaCecA7XLoUfpNhxkn9iM9MDBiVUCJU-iQ4GkyB_Cmx_tCgWdY9w1IFSuwuemlnrmlAZCLUof59p88B5X5KbYILsViaSP9ieLlNA9O5vLYt4ZdEyd52bVs41YS8esJ69FVNdKpCiQSSmrkWt7Bix98oEoWUmnjX-JE2El5hAtsqHdU-bAkbGEhzJAaHs6UwhTTFkPzbhq8Kq1KiiOqxjgcy2nQpV41ov85s9UHHWlpkpEFaEeQ1uCk1TW5iz6NgATm20PnjcyVwIucjjwyEWfHgUFbrPgzFQBESZvMnFswm19gEuz2yU1bCZscJxmt7D1Jus7LauGm--lFsZoCrY1LayUpLvhQhvx-HWieAnOKwwzJ5Gadyb65WSQBl5NuRJgwYhm7VaNk6ZCJyG2nHELhL5PZxXPWuxk4W_gSaQ-2EjeUNVvaqgWkmgjeFtvUsOqCcZ68QCFOR5YtqfbtJmNANIpMJw0n4_1LSz0TG3u2bCmJ-6sH51SvTblnAYLLSkbNYJ47N',
                            provider_name='openai',
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city":"Miami","summary":"Sunny, 22°C — clear skies and pleasant warmth. Expect bright sun; light layers or short sleeves comfortable. Use sunscreen and sunglasses if spending time outdoors; mornings/evenings might be slightly cooler. Overall great weather for outdoor activities."}',
                            tool_call_id='call_3cHnRoHfPOeJqJHc1iuPX0Aj',
                            id='fc_0f54edbaf2148c84006945adf8ca648196b667c5d79dd1b368',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=255, output_tokens=199, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 37, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0f54edbaf2148c84006945adf5b6988196af5e345c95e1e96c',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_3cHnRoHfPOeJqJHc1iuPX0Aj',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_none_with_union_output(
        self, openai_responses_model: OpenAIResponsesModel, allow_model_requests: None
    ):
        """With union type and tool_choice='none', model can still use output tools."""
        agent: Agent[None, str | CityInfo] = Agent(
            openai_responses_model, output_type=str | CityInfo, tools=[get_weather]
        )
        settings: ModelSettings = {'tool_choice': 'none'}

        with pytest.warns(UserWarning, match="tool_choice='none' but output tools"):
            result = await agent.run(
                'Describe Seattle briefly',
                model_settings=settings,
                usage_limits=UsageLimits(output_tokens_limit=1000),
            )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Describe Seattle briefly',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_0b9be64cebe1cb0c006945adfc8f78819e833f5eb13a7fa4ed',
                            signature='gAAAAABpRa4BpGevTAROF1ODuf2HDwoWDRJvMODu97893EXV_Ix5wVl_kmJeRooxrIyd6wLW4sK2dq096AQRipI4Fr2bMSSzxlMzIXh0doSqvF3q1MafSw1GqnIsWrU4yyEisVnDHTh-QMN52k0yWqkrKudIy_mF2xyDJE00dJftIqe34MEua-urtRxfte4E3PHwZ1y_7_GbFg9YeKJBHOhv3RIlgKxBMkX6Yothimf7CxJK0t3U1Bu7muVyOKMbmrp3QBo8iImKJlBrYAL2yy9QiUiqk5b2cTunsuc_6Vuh8BTW5Gtzkp_fKDwR-Ty8OJBXjqtxEGkgrYhpJgK7r8d5Vj0sCtgTKlo4VSheKzEYCiLN4iEcIT4AHLU2uUtMf7MkTNDJ1xBgcBaeXrMoqRk1pEVgVzndJvkDYTZcFbSt-YI42yKyQso70y7N_rH7iANYOnzn3uuTJJx-hnGtbY-RznDBeKncAWFKiFU8LNVZLDPB5q-1n4RSl_2EP8oQz01sSI3XgjVAEbCLq9lTF5MLUVDm8FtRttgokl29JhjqBfNd2PNU1idhZ0hmlF1GaRWFTCex4v9SjVidaO_5sFy13LHfdpl6_akCrCuJo8nVUJmTuVto8lib1iHvUVaclGW_6yy33YD8rMVnv6x28dMnCe1jgt01OpR19r-shjXT0g8adu4ykdsuMkh1_RpClIUQjnA_F3THJlLUKgNPDjt9ohcrUUEA37pyF0dboSxgpAh7Q5T750a0mkoF6NeqSv177udK-TDSWHqkU0SIyL7j6ArVpczTnZBediDhcORIrXCp0JNhHcT_JoU05TCFAGI2405BMDj60MP2csBfJ4KmaHYQUH-11UibjtRLN4hBjUXPikbstLlQBwcLKIRIDiS0LYyhyJdgjHAImUVhaI90RHmXuu-hQ09EoCgK2s8Asc-FChy9bx5NNrAi6kjbXd4T_ozkS5s8Y-8v_W1S9fDeoBwzUBuekNB_FhuOWGzu_vlFV1PHopY4oYIXbSlVw_qZSho7jyl4aTbs3Gs8Zw2J2wYubl7N4DeTilHEZKijbTpJSWsBBOJJ-mUpqLNZ1qSwTH9WpcBj8qDMiM0fgfvEkNhBBG8ehKSbPVyrzmtwa4au4frvx8IvUh87wZyq-dzUWtAzYSpCabJhdJyjUb-fjF_lPWmpY7sKg_fNoJCLYm-SsHd2gMFL9Ohy-ADiSVEWn7hh6ehyG-Yv5UXcHbbEB7ILV-mracBKYWPnasHQuGv7V50I06um_O-Cj2ps-xaWdQLzV3OM9cVlBHL9kz2NtaVpRXnDWn60bZ1wqaI4vFojDY5WRvO_C7S-HBLkAs3XGtnZlPMBmB3agZ0wqXhNPk3tszjyQJCtfmGq3kIhb1KYW2n-Jdnn2OIvUzrZAWNICSdXadurjGOPqrc-OuObdU24IPhTQS_avwWfC5mAEqRYniWdN8UsbvyO3fK1avqsIz3GoideRQycP6VsI2yOpbSNAHj-SDKnPv3AsqALf51rbjkERwvv6SyNx7WX8J3RJp_7r-BjFYtebL3m-MvEofcPm2MZy0qLzISh2vYm46nPpQetInVILLehwdn19uwtEglRAWanI7AqWtzHgV4SKJwuSFG2nUcKMFBXB-lpKstRO1zFQGk=',
                            provider_name='openai',
                        ),
                        TextPart(
                            content='Seattle is the largest city in Washington state, located on a narrow isthmus between Puget Sound and Lake Washington in the Pacific Northwest. Known as the "Emerald City," it has a mild, maritime climate with wet winters and relatively dry summers. Landmarks include the Space Needle, Pike Place Market, and waterfront ferries, with frequent views of Mount Rainier on clear days. Seattle is a major tech and aerospace hub (Amazon, Microsoft nearby, Boeing), and it has a strong music, coffee, and outdoor culture. The city proper has roughly 700–800k residents, with a metro area of about 3.5–4 million.',
                            id='msg_0b9be64cebe1cb0c006945adfe9a4c819e9e69fa478fb9f57b',
                        ),
                    ],
                    usage=RequestUsage(input_tokens=70, output_tokens=267, details={'reasoning_tokens': 128}),
                    model_name='gpt-5-mini-2025-08-07',
                    timestamp=datetime(2025, 12, 19, 19, 56, 44, tzinfo=timezone.utc),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed'},
                    provider_response_id='resp_0b9be64cebe1cb0c006945adfc0c04819ebf1d64404e56d07b',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )
