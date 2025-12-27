"""Code execution tests for OpenAI Responses model."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    BinaryImage,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPartDelta,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.usage import RequestUsage

from ...conftest import IsBytes, IsDatetime, IsNow, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import (
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_openai_responses_code_execution_return_image(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel(
        'gpt-5',
        provider=OpenAIProvider(api_key=openai_api_key),
        settings=OpenAIResponsesModelSettings(openai_include_code_execution_outputs=True),
    )

    agent = Agent(model=model, builtin_tools=[CodeExecutionTool()], output_type=BinaryImage)

    result = await agent.run('Create a chart of y=x^2 for x=-5 to 5')
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            _identifier='653a61',
            identifier='653a61',
        )
    )
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Create a chart of y=x^2 for x=-5 to 5',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_68cdc38812288190889becf32c2934990187028ba77f15f7',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'container_id': 'cntr_68cdc387531c81938b4bee78c36acb820dbd09bdba403548',
                            'code': """\
import numpy as np\r
import matplotlib.pyplot as plt\r
\r
# Data\r
x = np.arange(-5, 6, 1)\r
y = x**2\r
\r
# Plot\r
plt.figure(figsize=(6, 4))\r
plt.plot(x, y, marker='o')\r
plt.title('y = x^2 for x = -5 to 5')\r
plt.xlabel('x')\r
plt.ylabel('y')\r
plt.grid(True, linestyle='--', alpha=0.6)\r
plt.xticks(x)\r
plt.tight_layout()\r
\r
# Save and show\r
plt.savefig('/mnt/data/y_equals_x_squared.png', dpi=200)\r
plt.show()\r
\r
'/mnt/data/y_equals_x_squared.png'\
""",
                        },
                        tool_call_id='ci_68cdc39029a481909399d54b0a3637a10187028ba77f15f7',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            _identifier='653a61',
                            identifier='653a61',
                        ),
                        id='ci_68cdc39029a481909399d54b0a3637a10187028ba77f15f7',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed', 'logs': ["'/mnt/data/y_equals_x_squared.png'"]},
                        tool_call_id='ci_68cdc39029a481909399d54b0a3637a10187028ba77f15f7',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68cdc398d3bc8190bbcf78c0293a4ca60187028ba77f15f7',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2973, cache_read_tokens=1920, output_tokens=707, details={'reasoning_tokens': 512}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 19, 20, 56, 34, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68cdc382bc98819083a5b47ec92e077b0187028ba77f15f7',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run('Style it more futuristically.', message_history=messages)
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            _identifier='81863d',
            identifier='81863d',
        )
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Style it more futuristically.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_68cdc39f6aa48190b5aece25d55f80720187028ba77f15f7',
                        signature='gAAAAABozcPV8NxzVAMDdbpqK7_ltYa5_uAVsbnSW9OMWGRwlnwasaLvuaC4XlgGmC2MHbiPrccJ8zYuu0QoQm7jB6KgimG9Ax3vwoFGqMnfVjMAzoy_oJVadn0Odh3sKGifc11yVMmIkvrl0OcPYwJFlxlt2JhPkKotUDHY0P2LziSsMnQB_KaVdyYQxfcVbwrJJnB9wm2QbA3zNZogWepoXGrHXL1mBRR3J7DLdKGfMF_7gQC5fgEtb3G4Xhvk8_XNgCCZel48bqgzWvNUyaVPb4TpbibAuZnKnCNsFll6a9htGu9Ljol004p_aboehEyIp6zAm_1xyTDiJdcmfPfUiNgDLzWSKf-TwGFd-jRoJ3Aiw1_QY-xi1ozFu2oIeXb2oaZJL4h3ENrrMgYod3Wiprr99FfZw9IRN4ApagGJBnWYqW0O75d-e8jUMJS8zFJH0jtCl0jvuuGmM5vBAV4EpRLTcNGOZyoRpfqHwWfZYIi_u_ajs_A6NdqhzYvxYE-FAE1aJ89HxhnQNjRqkQFQnB8sYeoPOLBKIKAWYi3RziNE8klgSPC250QotupFaskTgPVkzbYe9ZtRZ9IHPeWdEHikb2RP-o1LVVO_zFMJdC6l4TwEToqRG8LaZOgSfkxS8eylTw7ROI2p8IBSmMkbkjvEkpmIic0FSx23Ew_Q-Y6DPa9isxGZcMMS0kOPKSPSML2MGoVq5L3-zIVj6ZBcFOMSaV5ytTlH-tKqBP9fejMyujwQFl5iXawuSjVjpnd2VL83o-xKbm6lEgsyXY1vynlS2hT52OYUY3MMvGSCeW5d7xwsVReO0O1EJqKS0lLh8thEMpJvar9dMgg-9ZCgZ1wGkJlpANf2moQlOWXKPXcbBa2kU0OW2WEffr4ecqg1QwPoMFLmR4HDL-KknuWjutF5bo8FW0CAWmxObxiHeDWIJYpS4KIIwp9DoLdJDWlg8FpD6WbBjKQN6xYmewHaTLWbZQw8zMGBcnhAkkyVopjrbM_6rvrH4ew05mPjPRrq9ODdHBqDYEn1kWj9MBDR-nhhLrci_6GImd64HZXYo0OufgcbxNu5mcAOsN3ww13ui8CTQVsPJO20XHc4jfwZ2Yr4iEIYLGdp0Xgv8EjIkJNA1xPeWn9COgCRrRSVLoF6qsgZwt9IRRGGEbH6kvznO_Y7BTTqufsORG6WNKc_8DDlrczoZVy0d6rI1zgqjXSeMuEP9LBG-bJKAvoAGDPXod8ShlqGX3Eb9CmBTZtTOJZYdgAlsZHx9BZ6zHlrJDjSDhc8xvdUAn9G3JvTI3b5JWSNX0eEerZ4c0FVqlpR-mSG201qnFghtoGHTLJhlIf9Ir8Daio_AYxUTRarQbcKnJuyKHPOz1u0PX2zS0xegO-IZhFbzNaB8qwQgeBiHfP-1dP9mkttqIRMt-hMt9NMHXoGIvFxgQ-xUVw7GRWx-ffKY7nPAbZD8kwVP3i4jTVj8phhwQcDy9UmbaPjm4LBgJkfdwNfSpm3g_ePK4aLa_l7iF2WSSfy2wObb7VatDzYDcNRG0ZTMGsiHy8yzZAcec18rG7uE6QCKx32G8NI5YvcN1kbnrZEuoKTBuSb2B_ZAhvED9HxbG8mH4ZEHHioVuH3_-b2TesVUAbORab_-rG9CU6qyy_eAqP54FYiXXSWtBWNo4baVdqCzgSCiNxgpxx64WPw8y2M1bOMoV6KPGwDOjcNwbO9nQwztqTWPW0Ot_Llf0HV0p-RPC1Uy8uBB5flhJ3p5uqxCPV3kDRzXgjh28EaBEkaSw_6SZkJNvwbD_7VihlHGaO89TwlqSIYUT_gc72NZKRrj4f-Y-0NwxjaSVVGuWCoeG-TMjG6uXpSozo2J47_x_a0lr4KCT8NDYlksajyuPUbYhC7jhQ9uJakmAc7ay_VHn_LYlAWRdAA7wYvqw7aYIuSIYg2OfL6NlggCpBnhsUPEXmMRHcfj1Ctc1aeUjBcpLFVmTZ82lB0FdcKRe3bBsKRckbdKalehoK0NJtrWqNQQH7xPrS-r7or_oOWhA4EDIkRUOG9eZhdsvTXBUamxGwutJ97SdDkgppVC4M7DMK2ZGGBzQsE-JMilERvFQ8JqwVWPxExWmE_-H2-bYe-T-CguCin-mTqhLYswHVtXjtruoHBmDs2SdnkD3intwSpqxsltscCfRaoRYWTCTbchCdbctSEIc39ECpc5tL1Gnav0bwSkMYkxyaRVBiYBbmIG9JftkKIYtdZ_Ddjmq8k29QflqrcigahsVLZPye3dxVTuviqbQjRd2SPMv8RxgSebgm5RZZIpP4WposryghYZFvuA1WImRzsImnAJI9J-8dv6IhHpHsWOw9K-Neg8GlnDU1mGHUElMUbqHiLojmXqPGfhBI3iSR0Ugs7ErpeRUrSk3il2o3rysG1Fn7ePuP5qNJUt2NyBUxf3TExMOwG_zqvpIPr2V_ARr3PsfeD0IcY83Bh428S8KPzc7ASOjT9dGQtVVrdjSxHi8o5ANxGx6z3bHC5dJvDCXg8a7FIJHAd5CUqJxrBi-K4p21jf1BNqgO5JAJO1JrvtdTk4GOVe8YEfhxmGWW9oeuRg8crsIWCCCoxr2XJKgPCj2TTPkBDZ1O3Yw3_nuWaBU5sB09uEB5lTKMd0OfSHbPF4c50RWAFgQB-tHjIUss3oEcAUaZHC77r6sIYoAEBlU8Dgly983fFD0HCqtpIpKS_B_K1fTXYpWRM3uUZpPKEgbfw1Kiqp5cweKTeRKNvjlau6VxhPyVi66xPdHUCC_BcX1eeFe-zcxe6fczcJWqGZGtYyVS_S_GlWZcdA6AHvGU6c4KjG0oU_9q-pdHSRtpnrhqFu2L884m64A_HsFU71Dj34AxhmXO1Am-zSL3j9nEPPUe6lJSGyhHU9k8ApDadWagvlODdXYWaWiMCXGXcYtl_iUAm24IJozlLJ1IW9HW6RoTfKrxwQwND3pX9CLNewuPV776pVtRjvUMbLaYg8nzOu1eNT2IW9dUdzc7wqOjiT1gHuVd6RzJyTCWJb9yPwDTkB_NKkjfUPmJ9Id924xtxy6H0eDYRq-SqsSSEklr6KJc88PV35QqvaMUW1dt_tGynHgYy9PXlWXQLKw-Xphku3FS_R4BLUhJbXDsMOQq332yhizP3qQ7vjEmPm8KB4DMIWBNn_D9xFuDuTCMNPAA9AGYWgC39-L4wPbpBHpqWjDwMzijFpm0CEViPD9ghyyV8syT1uLscxJVVDlBx90u_qWLSzMnFrVWmZ60OyWa9EqG44ZU8ELLHlEDRO_yHuTVpSafCLeDe5baOG2mI6tZnDBmm_ysbYdaC2N_zNBK9rhx7g7BNLQPevl0vtZm7GVLYXiVaO5ZinHxeTyJ6dRU5b0HmSw8r7EpdgORfjUuMkUfWPwhXgTU8SbvjTZg1gJowyNDYCvacrgnmnpBG9BgNjsfWlGTwz19AcEP_GjCWRWoE-uE_5fIyq5eFEefCBUKU0Ejs0IB-Re5h8bbdc6bNV3Tnx4UfGDU6FbQrJmPzrw5wp_wCeVYjtNGRbO2MKr_m52km5xMpVMMHtthVbQ9Zsa9F9zB6Dkr-R4F7o0dITMhG3qaREHKc8mXIGoHND-WSGPZLntB43JmRIWwjlJNstv7VlVc-dU89oh6Z1biH9B88SENI1ao2wMQV-BB17E6cmfzm1JsSR-HkzSf3yoUJWwvIu4CaR4jeMZohuoNqfGvQWIJSfyyUNzq5uY5__04QUmNcRVspOTH4EOHAoXLfCV3VI7fodj4FppiIuIXKwS3N03-Qt4sQ__XQWuyDdORvhRJeCvYcK5kkyOQILcABxDItxLmk8AgdT0Hz0BAo_u1U71srS-T8a8O0-fXWsJAHxDg_rJn0LUm6zq2vXNl8zmOKwEayyb0YySbMRxI-LwLyOXGRDyAVvm_7KKJu1HHqMntLyY2G1xowFpwMVLYXlGxDbsSpE-g5kFnHWhj13FiekLxaFgMRNsMA-r5_rWbEjRa6H328FKsUJcYe9qsp2LlzdJmYZDTIMgzxupFwQ-R5F6QjWOudMBsRszb4YqnOPJ8P9YnY2WYd0B7srb5Gh7T6r6mcCl-HAb2z9QDeXOc2Lu7ujuSvGj7_Gk7PkZH-LzoAEaGG9Z-7IVJlV_hOBPif3GlJUSUhTlIwWxn75gOyoOFuMak-rQqkb0SaL5anfXS_NUTVgSh5G5JQIoykLxbVlGiyeq0M_oEvTw2wMZcWT2hhaudcQ6L912pntcD-WF2tfppgp6sN5-cq-D8Y39N5Txvs-wo-H7-vYKPozTNUKCfnzgXfvt5fOi3RBR4MZU3eHT8OZ7d1d3otho_4GVMNIFa6mxjW1BC_J42Hn27-vrNDLZI_BXdF1t2CCq9VeRwxIW1R9vadd04HzAXyhap95BAYacmbULR6BkX97TvY3hv5cMiaQFkzxg-tf-nGC_VCknvwKxu4ocoB14p9w5TPSKcJz4J26XvyQbi6AdaXbOk625ajB_clv3VJvXYz7DgvWZd408tMykYQLMEyv5lnS7qwQokeM4ilIXwM7EugiakhfefTM9ZdxaWVcvQdqGerx98wlhifCSv0FqFRpJdkqgHmV1qzrAjPDEKT5HJOjsvs5hb7gKBqHR-bYlgS94pvDUpPArQXYcGYGum6vFsCAJypefMTF3D7Zhu4hhWQQv-DzSmfcZOxSeVJFrgVeqJnIbZPtd59HCBXNIRXJa42wUYE4szNli8wKWX0rYSIhiX-ig2YYZz3ZoBE1KDOpzheuk9OMYg7tQG2UlmVq27ggaKJ2gEGuVv-GI7uD7vKxPQ97QwCf38gWKU95CjMEBm_EvmLs9eubNpSpz8Yoek8hWWgrCXUSwRsYnF-lGdG0nIkCClvzqqAGOjyPxG4qfrCXJ-4rVc4DQiJUj71_I0EAhOgxb5WYBt4a7C1aUxC__qeOTAecof-UjzNlUPTo91JgOh5xvZkRkgGFNsq1OFqOcRrrKV8U8brizYkIhDjzjwCIzScSYvEfY4S6st-oJBv5fwTqwICSs59hf6WR8GXsPFR4v3UtF0Rkt-Nrek-X6V7BCui1M5HeFRN7lcTYs1Qw2bIwu4Td5PIkZ16oHdCk9u5pEZce-n_MIwj2Yoq_Lq1BBY9f1rpG9IuaycwabFnd2MOj89-xdgC197DAij5WjZjXahooyAl0Mt3p9MrHCit7LYbxqd_dGBOmg9YRfGPhsoZ17oAmHyg_gvpooOsu21T_06ynhvySjOG0yUcphquvtHJWqQdcT6BBX0X-kGE4nA41VdMhepLhDRDXtR4HJ1m_dPFpkHeAAFIefjt5Kb782TDLFE3KuHFWqSU2K2UmlY12P21dpRvyUNz8ss_AA3rl5jFpcnC2IyJNDIZbqdJPd2z0SNlwNyBq7Vl6poenR-j2X3xzIGlCDQ9zRgs50wdWtZ3ZRWLVWMrVkhkddoVKuh1W9rlwsvxmlZbOeRk_Uh0BymAa0-4-n0jI4_-O8jqpL-YzL1Y191brY4ywLUrQXpln41UK76pxc34FojI1Nymw523SNYxAHSlpj01gNmcjPrBTFxQ9SDY7AlrSFwJia_KvWnsZ53qt6fiDHV7p62KzlG_rpz_dQSQoj-z1hZBoUxi4nqzeCIzPcB_3JqeqD6x1O-Vh3uk-6NxN_qCE8cRsizB5vV-Ur-4tqau6LIrdfIB3Db12vpgiCmD_BD4xCxOijDn-97edRZw__xYfhx9_MBEB6gYl1ZBtLJfxDN54N5UION2tiZ2U8THD_h4d8-c26H7NQv44kYppbaseMckhpVOBDh52P5gxWFwp4VGqAIkZ7KU10qAD6M3GTFx7vGth8cT8YS1s2gPDW-WcVQGlAF94gT-FE6vzAjxwRJ4m7B1rJZfYReDvMrAoLroayOVmfB8pOKVQLQEF5dUmlzAIIpeh1NAiTg4n3FXW7OXQhzhU8bmo0e2FuSEOUVimGw9Nk_Wor3kQFp-9kj_iazSC4p5VURnyY_lAirPfyw4nskpZzCjSg_EAU8Au5vvOqrdDEPjrbeT8ks0wi2rsB0AxQxhgf6jUWzp0apeZOIl9dJFH_OnyJfvwrV4YHpee3174WKYhOJIOy2-8FJbMw1MpQtVV49yWmZsIyjRNj2uLbqY7jWBo2UEeOVW5n1tdk5zAVF-RFPKyh9150MnJz_RQtgoNdUD4iLBwlHYHVGLyH4a3GJmOJP6ZC-A-8RiUjvhu5co0yC8M83aVFjLe-yob3sNgJQgdVJnEOfPz4-1DVORoDgIRrRBcZQZqvkZwADFUkyy9jy5oXdEJ5XzthnizbrOZkHk6sQsNXrP4Uadqo9w99uy7TUh62l5AMWBFcaaQhuAuFkUZCavIqoO-2k4oXIDoTeBYzbyo_HH6caMk0D0_zgEg_5i-NhT3EUPdoCBNmjbOKmN2wzf6kqEyc8-nunjfq6HOjC6B6SE6VgOVJgBrhB4cBto4CxO45eqeuCi_WCjRtSS43Bh0QFZi6xK8rRjItyQRIfBpomETElbng3mAmBLPNb_7CzfsBdhBhJQLKu9KZ__uL3YVGtrCaLcOsfwP7BXRNQJH0yN_JWfMZH3y3B8z1O__xGhR63ugExWJZyUn55KAEiODbX35_PcftWXjslq-wzsK4J2fO_HFNU8Pi4egk6ibvCUDFRUelukaAy_YHdb0VTSB6XCymTo96jK0HGjG8FaVwvQaesaUE-e0_JpdMXN3KstKFeTlDUx1o3Ny93-VxLB5rkOSd6cRjEnFRA7Q6HnturEjwPAeJjR2Ll5dsisVrdjqHMbSfSObkpd2dZ0T3LP4-_ug7qRJF60DJTjTPpx7YxeARzuwiu02TlVW0J0PrdXT8EpISHneKc1VWhtRcdD0R0spuAMzJLwELaOemihL1TJSIMBqFikbpulZCZ1k1kA_5D7I5c7pOF1g4uYBW-gJNTenfC9wYmDJAOCcnwk1W4=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'container_id': 'cntr_68cdc387531c81938b4bee78c36acb820dbd09bdba403548',
                            'code': """\
import numpy as np\r
import matplotlib.pyplot as plt\r
import matplotlib.patheffects as pe\r
\r
# Data\r
x_smooth = np.linspace(-5, 5, 501)\r
y_smooth = x_smooth**2\r
x_int = np.arange(-5, 6, 1)\r
y_int = x_int**2\r
\r
# Futuristic styling parameters\r
bg_color = '#0b0f14'          # deep space blue-black\r
grid_color = '#00bcd4'        # cyan\r
neon_cyan = '#00e5ff'\r
neon_magenta = '#ff2bd6'\r
accent = '#8a2be2'            # electric purple\r
\r
plt.style.use('dark_background')\r
plt.rcParams.update({\r
    'font.family': 'DejaVu Sans Mono',\r
    'axes.edgecolor': neon_cyan,\r
    'xtick.color': '#a7ffff',\r
    'ytick.color': '#a7ffff',\r
    'axes.labelcolor': '#a7ffff'\r
})\r
\r
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)\r
fig.patch.set_facecolor(bg_color)\r
ax.set_facecolor(bg_color)\r
\r
# Neon glow effect: draw the curve multiple times with increasing linewidth and decreasing alpha\r
for lw, alpha in [(12, 0.06), (9, 0.09), (6, 0.14), (4, 0.22)]:\r
    ax.plot(x_smooth, y_smooth, color=neon_cyan, linewidth=lw, alpha=alpha, solid_capstyle='round')\r
\r
# Main crisp curve\r
ax.plot(x_smooth, y_smooth, color=neon_cyan, linewidth=2.5)\r
\r
# Glowing integer markers\r
ax.scatter(x_int, y_int, s=220, color=neon_magenta, alpha=0.10, zorder=3)\r
ax.scatter(x_int, y_int, s=60, color=neon_magenta, edgecolor='white', linewidth=0.6, zorder=4)\r
\r
# Grid and spines\r
ax.grid(True, which='major', linestyle=':', linewidth=0.8, color=grid_color, alpha=0.25)\r
for spine in ax.spines.values():\r
    spine.set_linewidth(1.2)\r
\r
# Labels and title with subtle glow\r
title_text = ax.set_title('y = x^2  •  x ∈ [-5, 5]', fontsize=16, color=neon_cyan, pad=12)\r
title_text.set_path_effects([pe.withStroke(linewidth=3, foreground=accent, alpha=0.35)])\r
\r
ax.set_xlabel('x', fontsize=12)\r
ax.set_ylabel('y', fontsize=12)\r
\r
# Ticks\r
ax.set_xticks(x_int)\r
ax.set_yticks(range(0, 26, 5))\r
\r
# Subtle techy footer\r
footer = ax.text(0.98, -0.15, 'generated • neon-grid',\r
                 transform=ax.transAxes, ha='right', va='top',\r
                 color='#7fdfff', fontsize=9, alpha=0.6)\r
footer.set_path_effects([pe.withStroke(linewidth=2, foreground=bg_color, alpha=0.9)])\r
\r
plt.tight_layout()\r
\r
# Save and show\r
out_path = '/mnt/data/y_equals_x_squared_futuristic.png'\r
plt.savefig(out_path, facecolor=fig.get_facecolor(), dpi=200, bbox_inches='tight')\r
plt.show()\r
\r
out_path\
""",
                        },
                        tool_call_id='ci_68cdc3be6f3481908f64d8f0a71dc6bb0187028ba77f15f7',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            _identifier='81863d',
                            identifier='81863d',
                        ),
                        id='ci_68cdc3be6f3481908f64d8f0a71dc6bb0187028ba77f15f7',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'status': 'completed',
                            'logs': [
                                """\
/tmp/ipykernel_11/962152713.py:40: UserWarning: You passed a edgecolor/edgecolors ('white') for an unfilled marker ('x').  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.
  ax.scatter(x_int, y_int, s=60, color=neon_magenta, edgecolor='white', linewidth=0.6, zorder=4)
""",
                                "'/mnt/data/y_equals_x_squared_futuristic.png'",
                            ],
                        },
                        tool_call_id='ci_68cdc3be6f3481908f64d8f0a71dc6bb0187028ba77f15f7',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
I gave the chart a neon, futuristic look with a dark theme, glowing curve, and cyber-style markers and grid.

Download the image: [y_equals_x_squared_futuristic.png](sandbox:/mnt/data/y_equals_x_squared_futuristic.png)

If you want different colors or a holographic gradient background, tell me your preferred palette.\
""",
                        id='msg_68cdc3d0303c8190b2a86413acbedbe60187028ba77f15f7',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=4614, cache_read_tokens=1792, output_tokens=1844, details={'reasoning_tokens': 1024}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 19, 20, 57, 1, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68cdc39da72481909e0512fef9d646240187028ba77f15f7',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_code_execution_return_image_stream(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel(
        'gpt-5',
        provider=OpenAIProvider(api_key=openai_api_key),
        settings=OpenAIResponsesModelSettings(openai_include_code_execution_outputs=True),
    )

    agent = Agent(model=model, builtin_tools=[CodeExecutionTool()], output_type=BinaryImage)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Create a chart of y=x^2 for x=-5 to 5') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            _identifier='df0d78',
            identifier='df0d78',
        )
    )
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Create a chart of y=x^2 for x=-5 to 5',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_06c1a26fd89d07f20068dd936ae09c8197b90141e9bf8c36b1',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args="{\"container_id\":\"cntr_68dd936a4cfc81908bdd4f2a2f542b5c0a0e691ad2bfd833\",\"code\":\"import numpy as np\\r\\nimport matplotlib.pyplot as plt\\r\\n\\r\\n# Data\\r\\nx = np.linspace(-5, 5, 1001)\\r\\ny = x**2\\r\\n\\r\\n# Plot\\r\\nfig, ax = plt.subplots(figsize=(6, 4))\\r\\nax.plot(x, y, label='y = x^2', color='#1f77b4')\\r\\nxi = np.arange(-5, 6)\\r\\nyi = xi**2\\r\\nax.scatter(xi, yi, color='#d62728', s=30, zorder=3, label='integer points')\\r\\n\\r\\nax.set_xlabel('x')\\r\\nax.set_ylabel('y')\\r\\nax.set_title('Parabola y = x^2 for x in [-5, 5]')\\r\\nax.grid(True, alpha=0.3)\\r\\nax.set_xlim(-5, 5)\\r\\nax.set_ylim(0, 26)\\r\\nax.legend()\\r\\n\\r\\nplt.tight_layout()\\r\\n\\r\\n# Save image\\r\\nout_path = '/mnt/data/y_eq_x_squared_plot.png'\\r\\nfig.savefig(out_path, dpi=200)\\r\\n\\r\\nout_path\"}",
                        tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            _identifier='df0d78',
                            identifier='df0d78',
                        ),
                        id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed', 'logs': ["'/mnt/data/y_eq_x_squared_plot.png'"]},
                        tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_06c1a26fd89d07f20068dd937ecbd48197bd91dc501bd4a4d4',
                    ),
                ],
                usage=RequestUsage(input_tokens=2772, output_tokens=1166, details={'reasoning_tokens': 896}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 20, 47, 35, tzinfo=timezone.utc),
                },
                provider_response_id='resp_06c1a26fd89d07f20068dd9367869c819788cb28e6f19eff9b',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_06c1a26fd89d07f20068dd936ae09c8197b90141e9bf8c36b1',
                    signature=IsStr(),
                    provider_name='openai',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_06c1a26fd89d07f20068dd936ae09c8197b90141e9bf8c36b1',
                    signature='gAAAAABo3ZN28TIB2hESP9n7FpWJJ4vj1KEPIVHYTNh64J3S9rOSRfmmTK_uSNB79wwlv3ur6X9Yl9sPe6moHK4nud8jgeScuOeCDq70JGXZ6xH_NBdiDWzeMis1WIDsyJrADdADGQRhjb8sXi6lz3nNvjeqXD-oZJkxTJ9FeJsCNNPBHX-ZYRIYZ7vGKLPfmi5qNS7V6VVGvwEWOBwW75ptObu5E8g2TqhPlUzsVoZsIZiczRXq6zQpDtMPAtv6Mz8puaq-o65P5-vZMywmEjyi0Dd2M9ozUfhWfhpEhCsAiItesA802-TSBQCKeP62riRAMJvfD3PEGLYL9d_7mUvJYSsiOADU0K6wfI6y8bRL-UaWUvn60KfPvqfBFm9-hwP1NS77OKoZABIuGz5sc3BuAh6ebKrJkfNHq7W0BA09S2gt3wLPzflpVl-wJ74L9UGnaKpmG3XRFogff_SNgDhO0_Cb4-1PYJi2NpqnCwTG2c8EFxXiP4trdynbpgRD5hKDj65FU46cBjR0g00bCShqwsseAzw_lAxbawcjF0zmAyz68Km2jCRKHRGgeMpbT-YQWs04IizKYsWfF-8pXX2vwSqk3Kb51OysuPN0K3gihF9v2tPnK2qFzkvNics__CDabCmafEKQlLp6TDRc5RY4ZcSHNwUM_dybJStzoH6qed1GQNt05wBhDZg39N7pJ8_dG7wXCSGHY5CRORZm19UGTd9DoZMzr8JmtxmRgJoKCHW_gavpt4__zifPVxqLUWj6GBaQRT8pR_Tym27HcsC0GbHLR1nel9hC6RzydTU5y7LWY_NoGUE4WZX5rHe5t73lFNSMwd9-6i9Qlj60_rBZ5z9oTAl_Ksywgo68AG7dFdSeI3VLnOyzhqeePn0ywaMp3HqO-FIXW3fjqtM2XMMMMn2Cje5rZhJ9JNmMqnxpltITkVdHMo7Yr1WFTkwLByEOb3M4LCq5B3dM1s1pVmqWAc9YNjpB7Fbi6fG90EAYFNEM4ubOE7y2d5E4hco0MbEKg-Fh0ubh1I2Y1kthZFEmPQLm6fFaljJKPtYojEZZ2cZ7sN3UaVg8Zpf3A7WS9kM2--lL5LuBnVDebf8Xrzv9dTmJvOtwWzJsY4RxWdnzfl_ZokHmg_HDNbeZpHsVI0gqHGr7YTlFJ0NUXW9mzZMx9e_VTrrf34XwRue3xVCqzsspRMjMIlAoDp0Rp0L2tJWAbKs_btqVpqjz8p-64CzSRq65BmSP6i86G0cJ9WLSD3gL3wR-Zt2HyvUvecHVmgKhXgY3F-RchYRO7TarJgyZY5bP2EEpHUwSWx4uWjYfzXMGYn8gNwgwl89qog-inK88qSG0DbqJQPwYNuRjS7Mu01O6eV39Zu7Njsn2io-kPc5HLRrbbhN7qCSki8yPWE_7yPtbIKlwWKOlEYx8_SGgE7waBFRem7ElsE9wvCX5KknilmN5_d9L4Sos0oT5NHAhApvVVDcygz9VGYBAmWfMOynDnOiTIpsAdjHmuZG7GJNAtUEYx7U7pNqbD2FJMIeN0L-3uqhxisRzeX64JZkVHWYL8HjeC1zHiUMZXKW1KXIvIU2_BCtqay22FtBskeMXZAReKhv3eX2oQlWL2Ps9VOk2imzjqBbFLzJgDq0iFoaHdOXGqo54GYZIxfWi10uo65s-3gOGmqPPE02FHEMjK7VHFjMh91FPhh8TmpWjOfa9QEcpEHSZJ6ipUMTVfRHHHshB6Sb74x-Jfr6Ioq2RnWd3E32GpE3kd1poqOssBi5jCqsA86tIMt0m8p_CDu_ANvMNKTiGTQdejm2rUhccpdbp8uLBPnqWxyGOCTlREglHPeh2EzjEMbtIaFp2NhHE6UlJ_nw40CDa5PA7C4lgUkn-4KtPy6rSaMu0mWM4vPO-5ksdtB3E5PkCdIB8j7htbhZH_MTv9RL7loDNkRVlJRSBiAC_qCGgVPyP4l1w4imdey-_HuVCKBD2vaXUz2l2efn-jLSlhty5vBOR-kr0EsU02_NYZtOKgBR1zIslAlnhM8lTxJWH4osSXHa4fIx9O9tyALjvxhooYww_Die_8iCH4u5cF53z3mvoK3Knzeada3jglwQyL3_uUQegcFKpvZwVAcguVMvrsbNgdR9VeKmYq8U7yBvziP-_vpj1UZcf3QxlNK_oOgDg9lxP3vsSKzxliW422svFDiyPkWPh1DWmry1xBD4Pldemf8OEvgSHSDAlegWoBnfOHljDcPf6kT0PaC-jHrKn8t1cQgWk1-1oxiW4zKIlKGoRvmo4lCcUfqGXb5EPuZM1qRFWxv4roAVoxdLV0Pz53L_Q-grQWvbKH_Rl6Dw1BysU55Klt8vn_XBL5Zw_UlbT9FrszDRjJ56F7zElzqVYunI5uJaPWTwQyO-4dvM94CqiUU59iFkfZqaSulYktZrgZeXe0lw59ecQnL_pR2xwkialTgDoqtPksIjTuWVzkiW9hIL5t9sHyCdJ9nqmwZRZU-JuTPXswmrJEJ23GhvtH9kWsswLd0qvmY5mV3cwr7hlFNWEf8_5e3LoCa9uHQgIa0uquekJ3St9dLOXpkcRv74nCpxkcjems_2ZC71DRU63NILFjKC5ffsUPOZ4NfevDMUDbYHdeyVV6E2f-_1yMYCWI_sws69fWQkWUIv33hk7Gm55NaNgLD4RYCUBTO7v1FtEZiVYAU5ab7NvvnTJ3FaEHo9G9eTzN1I_MmPzqlYX539YF_DDedh0ThnSoJl7PYD-7LhRRG1215KmsTWbqDGmtTsHePAVRSh464XHgiZ6cNPNogtMl4ym6r6nsMbzFP2krBR1f-u0tHfQFxAeLyBWij01Z1WBz4GBh3bpdLrB85AlvFeY7R46PPydAHxwwanYVyxpS0UmS7Y2S37EVRdFzai1izvoy3-wA05YKcnRiUKR-oMcLf-BmB3HHZnY77YOuqQBUZNI7OR8B6lvTARQuoJbK26ONmXEsH-VoBJR7C-hNiXMVh1jHfhuaBAj6Dg9g1Vs2kGxfoJUXB5dlFmR42mnyGcT96N8ZAIdIoQSrBzai6bQbuvOb3OAcG2lEhOZHZiwFRCzpHMfu5dctZ_wcTUhYZwgOcBNIo4WELyjv0Yx22AHSHcrUzFezOwibs-heUF_ciKWkGv9OaabaAGTaTVncfCnS7rOcD3Xum89EAVegpYiQzK0DZ_VKooPoddgHs6diYOEn4iJyvE54vaVi72NAy0Tf9poRlidKaM009FImefEtZqwD1MmaeVbjcClv5Xwyh-KCQ2hCZmrnJ2P_e0bWIsE0MAJOK8iU6Q3zxbntbZAQAKZHqqauT8kkRYxk6oBicV5BS-whqDN_GoNZrnRLTNkjk1a8mnqg_kucvC1mCQRbvP367DYqZGuAd2EQWVLSBQibHoVIUcYAFbsfRHfsQ-uiZVZsjZ-xGM-ZcTzCJ6p-hFi9IQXKqOioM_xzRl4TSY-AEbGja_RY0puxi8BeZXvSxx8eYsJ0TRtIIQwloZzKpbx1OwyK-Ibfj01PU5NIurJL10PKXcnc7ImXN-b_p8wfzEVN12lSbQ8m-Rs0tx32jfvviXyHtWYfHuNqP0eL3Xjuka6FGnuDOeOAIzy4xj1vqhXd8UN2tiFOObl4Rza5pKzF-0IcEsKX36v4iN8oYxOoCxCxLwvFw3znYiAKe6CVky4e46LxZOI3bGM6MSrypwblPMA2gC_ogfMiYViJe8gsgld9UvgQaFfj0EEgfc0BWfxVw2i6Yv3OcH3T1jaHnCVgvcDpTXI4-ZeeWKl6fhH9ukYAG4-Y2mGiJhxJ7cjSg8CwU0KDmNRwoXGB2FT0bKWovkcFYM5ueMbXFTZ4FFcgfWcOzXFZka82HFB_iqD1XvOYMFQNiz3jdtuOr8o66rtCVAjJnuoTQDmbSrWPU0-utUMJx-4QAlZM8hdtXGfNBp0JRxctMZdxR4BAzF7JH_ETYi3itZkgDLEs9JBdty6gUiM0NdR6F_7mxsHCik3rpb5bauJKP89gV03mnBQuSUQTauNxdzXqw55SPDAHMBWg8QwyffzWwmyTAjl_R1QiFsTOv31U-HditYAeYMhLAP0mIs97T0inLsTUri1s2b1s7j6-I-NLXuT4VKiBO8lqVicTbQdQwiXehHQsi18e0H6T9XM0xBQK2t1dd4Jz2oLUGroSB3XuNbcaaxsffqRQgk43KIMEw9VsUA3FOTEpdM_xYIYEFM_-ApjDQJ15JyMRspfmu7HDdd-ybcXZ-C8WASJUPV8tFEfP4xgUcZeu-mExkryebbdMExq78yj7GlwWaeqBYfEXsvG6FIOqL9iFVcc3iIelrly0oM_xJmLOB_CCkGylDmHLxZZydf5v0RDh0KOXd7J-QYepcALXYoXmToj2JPrJPkaznH-2tI5xwp_M-mktoYNOhWrOepFjceXDSF5G5ILomGd9mHLnkq514ayZJCeE437I2geH4s6upgSAaqc07IVvdU3WjorhBw9fvefI5NnYwMiUSk_LC-JiQZDJ0bMLttvwKDx0TmOnMDJqxDr06_MWXn3i0zLQlAjItS2foksr6EMeK2InZznVZtgjcbD0exqZuzjCAqKz4PLQl62xyuJx8trJe0uHbQk-NweJthN5xcj41kJTcDuXbA1bA9HerCBWMX0RW3RXAKTvltGaqyMyUsJ_uOb40D0m56SqOmxnyA-mauiV2R11KC5Hh7YSS587NxkWUx2t7G9uio6WgWyx-HvhXYVi8wejyZw51z70YEa-aUDS2G_N0e6BV2B6dMGyd3lzTkMY6Ncs127IwQmXkV4VGL0stfchFf7rhXc1CZmFm7NZOMQPgb3_Heb39gZfMa4EYUVLuvfSpuM8wHZcQa57_uj6wmGp7NBBVpcgTee9ADvJXxjlmAj6gm9TiCl_GYbBLCdoTRAgsgsy1r4WijYr2sA_zch6EbDpTjQy6ER5GINZ4zi0VDy9avZcxhGmOEHYvKzcLB5PANOAW-8FLFHGgDWvf0cEMCD0UpSLAJVIX6rMjMJC3N_cgWmmv_zbllaW-vDVNFPyZOW32zU-l7r46_5IuF9Vc5choUlWOGLADSnXReau9WC4rfGF05CAvLe5Q0dex4K14SHJTEJuBWhGTaaXzONQSGtU9LJexoI1ijcnz9X59VvXxFX0oHmLvgTAim6nN96X5kllHFvrdDjMOiZKQTXtodUI-3ZcjfA5booJk7tnFeni0H2L1sqvpGy8JDlfl0fds8hST0vtXscfD5jDC-i6btLnRgpOpRDQMebCkqRlisZScBXb0nxoHK7CHtnQy4aCQq4oCBgMXdbwHOnbBygBSAg-HCpK53YoT-R5NUdESGmGCX5uJ0qlmGaXSshFbNW_NpQItJIrD7NW3VmqfWvSB1VL-nyVLOmc_wPmUhY7dSGArYKYQFKL4cBOSfHHHuftrRXy356_mTcDeFsHzqH3RXPaXhiad_lmQ9Bcw0OD_BotHvYfvVCaETpweH3eHl3RPBiUHlc5Da4nprHbXrvQL675qwVLiwLwOvPULU4VdGU-jIfSMkRUbJhSt349C1poj4aM-aD3s5iJy-3YDRYzmqMmFFr9CoKMah6hmn6n0oKSwg0YpLOc9JRDhBfp87_NNsWdRkpNw_DC7OaIF6VNxc6o2t9jExqmAiAbyRSkW2x-UiZl6kbB3uqffgAYWNylgJDZ-UPQNki30zURQFl1anKa8xhIGOgH7piVerG2LO8X7pFxa3DlYxFm37HC6irFtBwsFbvNGicua6MfUD3dV2MhE9x-sOlG9O08DKObUwBTpTzfAe-P_jGWHnyOsLXbaiV_cwxgWkEw9rKuFpI1SPuPrdO8_iSYdH36TqIREPLVbRcSJvHrsWP2Bf-Bb04SIonHV4Olu9KEYWVCOltRx7JFjp3eVQZLAGwjtxG_vDlublMpybM6TZdg1UYaCU4ZqLKss3iWO3wBNwC2usITNSjaiiLSH96fOHpAyXMhhodFDS9X-frLB46hilqE3PwoIyiR5R1dAdM7oiWa5qD6KH_dISw5H-uO6ZrUFo6i14E4RcCtRBBKALvVnApLxA_lcpnFR9_TZkstK-6klIEiSttNhxhHhv36XJw_J6jUTHnxRBr4JyXLL3-NmDZy8mplsbS4OXl7gg0vuIOBBHarKFvCEdvZv8ikxbDeftTz2je9mrCNCAHKTeNQWKf7Q7HFfPcza_BwhSqrd64DndvGVkfLlYBrbVSZp5nxPF13qBWIw9bbXTU5z8Wna72Lh4HqL-cUDsKbKBpst1VuBgaA7Va',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    provider_name='openai',
                ),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='{"container_id":"cntr_68dd936a4cfc81908bdd4f2a2f542b5c0a0e691ad2bfd833","code":"',
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='import', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' numpy', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' as', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' np', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='import', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' matplotlib', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.pyplot', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' as', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' plt', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='#', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' Data', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' np', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.linspace', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(-', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='100', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='1', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='**', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='2', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='#', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' Plot', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='fig', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' plt', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.subplots', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(figsize', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='=(', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='6', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='4', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='))\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.plot', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' label', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="='", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='^', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='2', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="',", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' color', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="='#", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='1', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='f', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='77', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='b', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='4', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="')\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='xi', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' np', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.arange', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(-', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='6', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='yi', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' xi', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='**', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='2', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.scatter', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='i', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' yi', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' color', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="='#", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='d', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='627', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='28', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="',", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' s', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='=', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='30', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' z', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='order', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='=', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='3', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' label', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="='", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='integer', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' points', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="')\\r\\n\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.set', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_xlabel', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="('", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="')\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.set', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_ylabel', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="('", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="')\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.set', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_title', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="('", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='Par', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ab', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ola', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='^', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='2', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' for', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' in', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' [-', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=']', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="')\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.grid', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(True', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' alpha', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='=', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='0', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='3', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.set', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_xlim', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(-', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='5', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.set', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_ylim', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='0', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='26', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='ax', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.legend', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='()\\r\\n\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='plt', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.tight', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_layout', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='()\\r\\n\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='#', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' Save', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' image', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='out', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_path', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=" '/", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='mnt', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='/data', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='/y', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_eq', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_x', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_squared', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_plot', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.png', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta="'\\r\\n", tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='fig', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='.savefig', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(out', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_path', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' dpi', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='=', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='200', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\r\\n\\r\\n', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='out', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='_path', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='"}', tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7'
                ),
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args="{\"container_id\":\"cntr_68dd936a4cfc81908bdd4f2a2f542b5c0a0e691ad2bfd833\",\"code\":\"import numpy as np\\r\\nimport matplotlib.pyplot as plt\\r\\n\\r\\n# Data\\r\\nx = np.linspace(-5, 5, 1001)\\r\\ny = x**2\\r\\n\\r\\n# Plot\\r\\nfig, ax = plt.subplots(figsize=(6, 4))\\r\\nax.plot(x, y, label='y = x^2', color='#1f77b4')\\r\\nxi = np.arange(-5, 6)\\r\\nyi = xi**2\\r\\nax.scatter(xi, yi, color='#d62728', s=30, zorder=3, label='integer points')\\r\\n\\r\\nax.set_xlabel('x')\\r\\nax.set_ylabel('y')\\r\\nax.set_title('Parabola y = x^2 for x in [-5, 5]')\\r\\nax.grid(True, alpha=0.3)\\r\\nax.set_xlim(-5, 5)\\r\\nax.set_ylim(0, 26)\\r\\nax.legend()\\r\\n\\r\\nplt.tight_layout()\\r\\n\\r\\n# Save image\\r\\nout_path = '/mnt/data/y_eq_x_squared_plot.png'\\r\\nfig.savefig(out_path, dpi=200)\\r\\n\\r\\nout_path\"}",
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    provider_name='openai',
                ),
                next_part_kind='file',
            ),
            PartStartEvent(
                index=2,
                part=FilePart(
                    content=BinaryImage(data=IsBytes(), media_type='image/png', _identifier='df0d78'),
                    id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartStartEvent(
                index=3,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed', 'logs': ["'/mnt/data/y_eq_x_squared_plot.png'"]},
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='file',
            ),
            PartStartEvent(
                index=4,
                part=TextPart(content='Here', id='msg_06c1a26fd89d07f20068dd937ecbd48197bd91dc501bd4a4d4'),
                previous_part_kind='builtin-tool-return',
            ),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' chart')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' y')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' =')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' x')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='^')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' x')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' from')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='Download')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' image')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' [')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='Download')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' chart')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='](')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='sandbox')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=':/')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='mnt')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='/data')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='/y')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='_eq')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='_x')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='_squared')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='_plot')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='.png')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=')')),
            PartEndEvent(
                index=4,
                part=TextPart(
                    content=IsStr(),
                    id='msg_06c1a26fd89d07f20068dd937ecbd48197bd91dc501bd4a4d4',
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args="{\"container_id\":\"cntr_68dd936a4cfc81908bdd4f2a2f542b5c0a0e691ad2bfd833\",\"code\":\"import numpy as np\\r\\nimport matplotlib.pyplot as plt\\r\\n\\r\\n# Data\\r\\nx = np.linspace(-5, 5, 1001)\\r\\ny = x**2\\r\\n\\r\\n# Plot\\r\\nfig, ax = plt.subplots(figsize=(6, 4))\\r\\nax.plot(x, y, label='y = x^2', color='#1f77b4')\\r\\nxi = np.arange(-5, 6)\\r\\nyi = xi**2\\r\\nax.scatter(xi, yi, color='#d62728', s=30, zorder=3, label='integer points')\\r\\n\\r\\nax.set_xlabel('x')\\r\\nax.set_ylabel('y')\\r\\nax.set_title('Parabola y = x^2 for x in [-5, 5]')\\r\\nax.grid(True, alpha=0.3)\\r\\nax.set_xlim(-5, 5)\\r\\nax.set_ylim(0, 26)\\r\\nax.legend()\\r\\n\\r\\nplt.tight_layout()\\r\\n\\r\\n# Save image\\r\\nout_path = '/mnt/data/y_eq_x_squared_plot.png'\\r\\nfig.savefig(out_path, dpi=200)\\r\\n\\r\\nout_path\"}",
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed', 'logs': ["'/mnt/data/y_eq_x_squared_plot.png'"]},
                    tool_call_id='ci_06c1a26fd89d07f20068dd937636948197b6c45865da36d8f7',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
        ]
    )
