import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RequestUsage,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

pytest.importorskip('ddgs')
pytest.importorskip('markdownify')

from inline_snapshot import snapshot
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

from pydantic_ai_harness.researcher import Researcher

if TYPE_CHECKING:

    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsInstance(expected_type: type[RequestUsage], **kwargs: Any) -> RequestUsage: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
else:
    from dirty_equals import IsDatetime, IsInstance, IsStr

pytestmark = pytest.mark.anyio


@pytest.mark.vcr
async def test_researcher_completes_task(
    tmp_path: Path, allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY', 'replay-key'))
    # Replay recomputes tool returns while model responses come from the cassette, and the recorded
    # `read_tool_result` calls reference spill handles keyed by the recording run's `run_id` — so the
    # run below pins it. Rooting the overflow store in `tmp_path` keeps spills left behind by earlier
    # runs out of the lookup.
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    agent = Agent(
        'openai:gpt-5.6-sol',
        capabilities=[Researcher()],
        model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort='none'),
    )

    result = await agent.run(
        'Compare the free-threaded Python support in CPython 3.13 and 3.14 using official sources. '
        'Delegate the 3.13 investigation to the researcher sub-agent while you investigate 3.14, '
        'then synthesize the maturity status and main limitations. Search for and read the sources '
        'needed for each version, and include direct links.',
        run_id='01a0021a-aa83-7252-8a64-d7108e706f29',
    )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Compare the free-threaded Python support in CPython 3.13 and 3.14 using official sources. Delegate the 3.13 investigation to the researcher sub-agent while you investigate 3.14, then synthesize the maturity status and main limitations. Search for and read the sources needed for each version, and include direct links.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- researcher: Research a focused sub-question on the web and report back with findings and source links\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='delegate_task',
                        args='{"agent_name":"researcher","task":"Investigate free-threaded Python support specifically in CPython 3.13 using only official Python sources (docs.python.org, peps.python.org, python.org, GitHub python/cpython if necessary). Search broadly and read the supporting sources. Report: maturity/experimental status, how to obtain/build/enable it, runtime identification and GIL toggling behavior, extension compatibility requirements and GIL fallback, ABI/wheel tagging, important limitations/performance caveats, and any guarantees or lack thereof. Include direct source URLs for every factual claim, and distinguish any inference."}',
                        tool_call_id='call_ThjTLUpjfRlxSV43zSkDc4qF',
                        id='fc_0e58f8341a296874006a7f83c2c3dc819699d5d5c3b5a14098',
                        provider_name='openai',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='gpt-5.6-sol',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_0e58f8341a296874006a7f83c1b21081969141bc089fa42e84',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='delegate_task',
                        content=IsStr(
                            regex="""\
\\[Tool\\ output\\ too\\ large\\ \\(20,719\\ chars\\);\\ stored\\ to\\ handle\\ '01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_ThjTLUpjfRlxSV43zSkDc4qF\\.0'\\.\\ Read\\ it\\ with\\ read_tool_result\\(handle='01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_ThjTLUpjfRlxSV43zSkDc4qF\\.0',\\ offset=0,\\ limit=200,\\ from_end=False,\\ pattern=None\\)\\.\\]\\
\\#\\ Free\\-threaded\\ CPython\\ 3\\.13\\
\\
\\#\\#\\ Executive\\ summary\\
\\
CPython\\ 3\\.13\\ introduced\\ a\\ \\*\\*separate,\\ experimental\\ free\\-threaded\\ build\\*\\*\\ in\\ which\\ the\\ GIL\\ is\\ normally\\ disabled\\.\\ It\\ is\\ not\\ the\\ default\\ CPython\\ build,\\ has\\ a\\ distinct\\ ABI,\\ requires\\ separately\\ built\\ native\\ extensions,\\ and\\ can\\ automatically\\ re\\-enable\\ the\\ GIL\\ when\\ an\\ extension\\ has\\ not\\ declared\\ free\\-threading\\ support\\.\\ CPython\\ 3\\.13’s\\ documentation\\ warns\\ of\\ bugs,\\ unsafe\\ edge\\ cases,\\ increased\\ memory\\ use,\\ and\\ roughly\\ \\*\\*40%\\ single\\-threaded\\ overhead\\ on\\ `pyp\\
\\.\\.\\.\\[19,719\\ chars\\ omitted\\]\\.\\.\\.\\
html\\#c\\.PyUnstable_Module_SetGIL>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/using/windows\\.html\\#installing\\-free\\-threaded\\-binaries>\\ \\ \\
<https://peps\\.python\\.org/pep\\-0703/>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
\\*\\*Bottom\\ line:\\*\\*\\ CPython\\ 3\\.13\\ free\\ threading\\ is\\ suitable\\ for\\ experimentation,\\ porting,\\ compatibility\\ testing,\\ and\\ carefully\\ controlled\\ workloads\\.\\ Its\\ own\\ official\\ documentation\\ does\\ not\\ present\\ it\\ as\\ a\\ drop\\-in\\ production\\ replacement\\ for\\ ordinary\\ CPython\\ 3\\.13\\.\
"""
                        ),
                        tool_call_id='call_ThjTLUpjfRlxSV43zSkDc4qF',
                        metadata={
                            'overflow_handle': IsStr(
                                regex='01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_ThjTLUpjfRlxSV43zSkDc4qF\\.0'
                            ),
                            'overflow_bytes': 20757,
                        },
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- researcher: Research a focused sub-question on the web and report back with findings and source links\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={
                            'type': 'search',
                            'queries': [
                                'site:docs.python.org/3.14 free-threading Python HOWTO limitations',
                                'site:docs.python.org/3.14 whatsnew 3.14 free-threaded Python officially supported',
                                'site:peps.python.org PEP 779 free-threading 3.14 supported',
                                'site:docs.python.org/3.14 C API extension modules free threading',
                            ],
                            'query': 'site:docs.python.org/3.14 free-threading Python HOWTO limitations',
                        },
                        tool_call_id='ws_0e58f8341a296874006a7f845146388196a71d0623a1466610',
                        id='ws_0e58f8341a296874006a7f845146388196a71d0623a1466610',
                        provider_name='openai',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e58f8341a296874006a7f845146388196a71d0623a1466610',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={
                            'type': 'open_page',
                            'url': 'https://docs.python.org/3.14/howto/free-threading-python.html',
                        },
                        tool_call_id='ws_0e58f8341a296874006a7f8453f2e48196b0e4c47fc64f90be',
                        id='ws_0e58f8341a296874006a7f8453f2e48196b0e4c47fc64f90be',
                        provider_name='openai',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e58f8341a296874006a7f8453f2e48196b0e4c47fc64f90be',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={
                            'pattern': 'Free-threaded Python is officially supported',
                            'type': 'find_in_page',
                            'url': 'https://docs.python.org/3.14/whatsnew/3.14.html',
                        },
                        tool_call_id='ws_0e58f8341a296874006a7f8457a21c8196ae93b0db202d8a91',
                        id='ws_0e58f8341a296874006a7f8457a21c8196ae93b0db202d8a91',
                        provider_name='openai',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e58f8341a296874006a7f8457a21c8196ae93b0db202d8a91',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='read_tool_result',
                        args=IsStr(
                            regex='\\{"handle":"01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_ThjTLUpjfRlxSV43zSkDc4qF\\.0","offset":0,"limit":120\\}'
                        ),
                        tool_call_id='call_H7yjfeTQBnxFNmNbiZimBniZ',
                        id='fc_0e58f8341a296874006a7f8459f5c4819680b271ba2abd79fa',
                        provider_name='openai',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='gpt-5.6-sol',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_0e58f8341a296874006a7f84501d548196acac86c0fa44e665',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='read_tool_result',
                        content=IsStr(
                            regex="""\
\\[handle\\ '01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_ThjTLUpjfRlxSV43zSkDc4qF\\.0':\\ 331\\ matching\\ line\\(s\\);\\ showing\\ 120\\]\\
\\#\\ Free\\-threaded\\ CPython\\ 3\\.13\\
\\
\\#\\#\\ Executive\\ summary\\
\\
CPython\\ 3\\.13\\ introduced\\ a\\ \\*\\*separate,\\ experimental\\ free\\-threaded\\ build\\*\\*\\ in\\ which\\ the\\ GIL\\ is\\ normally\\ disabled\\.\\ It\\ is\\ not\\ the\\ default\\ CPython\\ build,\\ has\\ a\\ distinct\\ ABI,\\ requires\\ separately\\ built\\ native\\ extensions,\\ and\\ can\\ automatically\\ re\\-enable\\ the\\ GIL\\ when\\ an\\ extension\\ has\\ not\\ declared\\ free\\-threading\\ support\\.\\ CPython\\ 3\\.13’s\\ documentation\\ warns\\ of\\ bugs,\\ unsafe\\ edge\\ cases,\\ increased\\ memory\\ use,\\ and\\ roughly\\ \\*\\*40%\\ single\\-threaded\\ overhead\\ on\\ `pyperformance`\\*\\*\\.\\
\\
The\\ feature’s\\ acceptance\\ was\\ deliberately\\ provisional:\\ the\\ Steering\\ Council\\ reserved\\ the\\ ability\\ to\\ roll\\ back\\ disruptive\\ parts—or\\ even\\ all—of\\ PEP\\ 703\\.\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\#free\\-threaded\\-cpython>\\ \\ \\
<https://peps\\.python\\.org/pep\\-0703/>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
\\*\\*Methodology\\ note:\\*\\*\\ I\\ treated\\ the\\ versioned\\ CPython\\ 3\\.13\\ documentation\\ as\\ authoritative\\ for\\ implemented\\ behavior\\.\\ The\\ 3\\.13\\ “What’s\\ New”\\ document\\ itself\\ warns\\ that\\ PEPs\\ are\\ usually\\ not\\ maintained\\ after\\ implementation,\\ so\\ I\\ use\\ PEP\\ 703\\ mainly\\ for\\ design\\ intent,\\ ABI\\ rationale,\\ and\\ the\\ Steering\\ Council’s\\ acceptance\\ proviso\\.\\ \\ \\
Source:\\ <https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\)\\)\\
\\
\\-\\-\\-\\
\\
\\#\\#\\ 1\\.\\ Maturity\\ and\\ experimental\\ status\\
\\
\\-\\ Python\\ 3\\.13\\ itself\\ is\\ a\\ stable\\ language\\ release,\\ but\\ its\\ free\\-threaded\\ mode\\ is\\ explicitly\\ \\*\\*experimental\\*\\*,\\ is\\ \\*\\*not\\ enabled\\ by\\ default\\*\\*,\\ and\\ uses\\ a\\ different\\ executable,\\ usually\\ `python3\\.13t`\\ or\\ `python3\\.13t\\.exe`\\.\\ The\\ documentation\\ says\\ to\\ expect\\ bugs\\ and\\ a\\ substantial\\ single\\-threaded\\ performance\\ hit\\.\\ \\ \\
\\ \\ Source:\\ <https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\#free\\-threaded\\-cpython>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\)\\)\\
\\
\\-\\ The\\ dedicated\\ 3\\.13\\ HOWTO\\ repeats\\ that\\ the\\ mode\\ remains\\ experimental\\ and\\ that\\ work\\ is\\ ongoing\\.\\ \\ \\
\\ \\ Source:\\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
\\-\\ Windows\\ installation\\ behavior\\ is\\ also\\ explicitly\\ marked\\ experimental\\ and\\ “should\\ be\\ expected\\ to\\ change\\ in\\ future\\ releases\\.”\\ \\ \\
\\ \\ Source:\\ <https://docs\\.python\\.org/3\\.13/using/windows\\.html\\#installing\\-free\\-threaded\\-binaries>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/using/windows\\.html\\)\\)\\
\\
\\-\\ PEP\\ 703\\ was\\ accepted\\ with\\ a\\ gradual\\-rollout\\ proviso:\\ changes\\ may\\ be\\ rolled\\ back\\ if\\ they\\ prove\\ too\\ disruptive,\\ potentially\\ including\\ all\\ of\\ PEP\\ 703\\.\\ This\\ is\\ a\\ policy\\-level\\ lack\\ of\\ permanence,\\ not\\ a\\ claim\\ that\\ rollback\\ is\\ expected\\.\\ \\ \\
\\ \\ Source:\\ <https://peps\\.python\\.org/pep\\-0703/\\#note>\\ \\(\\[peps\\.python\\.org\\]\\(https://peps\\.python\\.org/pep\\-0703/\\)\\)\\
\\
\\-\\-\\-\\
\\
\\#\\#\\ 2\\.\\ Obtaining,\\ building,\\ and\\ enabling\\ it\\
\\
\\#\\#\\#\\ Official\\ binaries\\
\\
\\-\\ Official\\ Python\\ 3\\.13\\ installers\\ for\\ \\*\\*macOS\\ and\\ Windows\\*\\*\\ optionally\\ install\\ free\\-threaded\\ binaries\\.\\ They\\ are\\ not\\ the\\ ordinary/default\\ interpreter\\ binaries\\.\\ \\ \\
\\ \\ Sources:\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\#installation>\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/using/mac\\.html\\#installing\\-free\\-threaded\\-binaries>\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/using/windows\\.html\\#installing\\-free\\-threaded\\-binaries>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
\\-\\ On\\ Windows,\\ choose\\ \\*\\*Customize\\ installation\\*\\*,\\ then\\ select\\ \\*\\*Download\\ free\\-threaded\\ binaries\\*\\*\\.\\ For\\ unattended\\ installation,\\ use\\ `Include_freethreaded=1`\\.\\ The\\ main\\ executable\\ is\\ `python3\\.13t\\.exe`;\\ the\\ launcher\\ can\\ select\\ it\\ with\\ `py\\.exe\\ \\-3\\.13t`\\.\\ \\ \\
\\ \\ Source:\\ <https://docs\\.python\\.org/3\\.13/using/windows\\.html\\#installing\\-free\\-threaded\\-binaries>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/using/windows\\.html\\)\\)\\
\\
\\-\\ Windows\\ NuGet\\ packages\\ are\\ named\\ `python\\-freethreaded`,\\ `pythonx86\\-freethreaded`,\\ and\\ `pythonarm64\\-freethreaded`\\.\\ \\ \\
\\ \\ Source:\\ <https://docs\\.python\\.org/3\\.13/using/windows\\.html\\#free\\-threaded\\-packages>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/using/windows\\.html\\)\\)\\
\\
\\#\\#\\#\\ Building\\ from\\ source\\
\\
On\\ platforms\\ using\\ the\\ normal\\ `configure`\\ build:\\
\\
```sh\\
\\./configure\\ \\-\\-disable\\-gil\\
make\\
make\\ install\\ \\ \\ \\ \\ \\ \\ \\#\\ or\\ make\\ altinstall\\
```\\
\\
`\\-\\-disable\\-gil`\\ defines\\ `Py_GIL_DISABLED`\\ and\\ adds\\ `t`\\ to\\ `sys\\.abiflags`\\.\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/using/configure\\.html\\#cmdoption\\-disable\\-gil>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\#installation>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/using/configure\\.html\\?utm_source=openai\\)\\)\\
\\
This\\ is\\ a\\ \\*\\*build\\ configuration\\*\\*,\\ not\\ a\\ switch\\ that\\ turns\\ an\\ ordinary\\ CPython\\ 3\\.13\\ binary\\ into\\ a\\ free\\-threaded\\ binary\\.\\ The\\ resulting\\ interpreter\\ has\\ its\\ own\\ executable\\ and\\ ABI\\.\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\#free\\-threaded\\-cpython>\\ \\ \\
<https://peps\\.python\\.org/pep\\-0703/\\#build\\-configuration\\-changes>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\)\\)\\
\\
\\#\\#\\#\\ Installing\\ packages\\
\\
`pip`\\ 24\\.1\\ or\\ newer\\ is\\ required\\ to\\ install\\ packages\\ containing\\ C\\ extensions\\ into\\ the\\ 3\\.13\\ free\\-threaded\\ build\\.\\ \\ \\
Source:\\ <https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\#free\\-threaded\\-cpython>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\)\\)\\
\\
\\-\\-\\-\\
\\
\\#\\#\\ 3\\.\\ Runtime\\ identification\\
\\
These\\ checks\\ answer\\ different\\ questions:\\
\\
```python\\
import\\ sys\\
import\\ sysconfig\\
\\
print\\(sys\\.version\\)\\
print\\(sysconfig\\.get_config_var\\("Py_GIL_DISABLED"\\)\\)\\
print\\(sys\\._is_gil_enabled\\(\\)\\)\\
```\\
\\
\\-\\ `python\\ \\-VV`\\ and\\ `sys\\.version`\\ contain\\ \\*\\*`experimental\\ free\\-threading\\ build`\\*\\*\\ when\\ the\\ executable\\ supports\\ free\\ threading\\.\\ \\ \\
\\-\\ `sysconfig\\.get_config_var\\("Py_GIL_DISABLED"\\)\\ ==\\ 1`\\ identifies\\ the\\ free\\-threaded\\ \\*\\*build\\ configuration\\*\\*\\ and\\ is\\ the\\ documented\\ recommended\\ test\\ for\\ build\\-dependent\\ decisions\\.\\ \\ \\
\\-\\ `sys\\._is_gil_enabled\\(\\)`\\ reports\\ the\\ current\\ process\\ state:\\ `True`\\ if\\ the\\ GIL\\ is\\ active,\\ `False`\\ otherwise\\.\\ It\\ is\\ a\\ CPython\\ implementation\\ detail\\ and\\ is\\ not\\ guaranteed\\ to\\ exist\\ in\\ every\\ Python\\ implementation\\.\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\#identifying\\-free\\-threaded\\-python>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/library/sys\\.html\\#sys\\._is_gil_enabled>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
For\\ C\\ compilation:\\
\\
```c\\
\\#ifdef\\ Py_GIL_DISABLED\\
\\ \\ \\ \\ /\\*\\ free\\-threaded\\-build\\-specific\\ code\\ \\*/\\
\\#endif\\
```\\
\\
In\\ a\\ free\\-threaded\\ build,\\ `Py_GIL_DISABLED`\\ is\\ defined\\ as\\ `1`;\\ in\\ a\\ regular\\ build\\ it\\ is\\ not\\ defined\\.\\ On\\ Windows\\ 3\\.13,\\ extensions\\ must\\ define\\ it\\ manually\\ when\\ compiling\\ against\\ the\\ free\\-threaded\\ installation\\.\\ \\ \\
Source:\\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\#identifying\\-the\\-free\\-threaded\\-build\\-in\\-c>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\)\\)\\
\\
On\\ Unix/configure\\ builds,\\ `sys\\.abiflags`\\ includes\\ `t`\\.\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/using/configure\\.html\\#cmdoption\\-disable\\-gil>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/library/sys\\.html\\#sys\\.abiflags>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/using/configure\\.html\\?utm_source=openai\\)\\)\\
\\
\\-\\-\\-\\
\\
\\#\\#\\ 4\\.\\ Runtime\\ GIL\\ toggling\\ and\\ extension\\ fallback\\
\\
A\\ free\\-threaded\\ executable\\ can\\ still\\ run\\ with\\ the\\ GIL\\ enabled:\\
\\
```sh\\
python3\\.13t\\ \\-X\\ gil=1\\ program\\.py\
"""
                        ),
                        tool_call_id='call_H7yjfeTQBnxFNmNbiZimBniZ',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- researcher: Research a focused sub-question on the web and report back with findings and source links\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='read_tool_result',
                        args=IsStr(
                            regex='\\{"handle":"01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_ThjTLUpjfRlxSV43zSkDc4qF\\.0","offset":120,"limit":140\\}'
                        ),
                        tool_call_id='call_Uc4auU7ikVIFweZ0PbbYlD4w',
                        id='fc_0e58f8341a296874006a7f845e4534819693c53cd9dd73725e',
                        provider_name='openai',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='gpt-5.6-sol',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_0e58f8341a296874006a7f845bb5a0819693d55bb5431d3cff',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='read_tool_result',
                        content=IsStr(
                            regex="""\
\\[handle\\ '01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_ThjTLUpjfRlxSV43zSkDc4qF\\.0':\\ 331\\ matching\\ line\\(s\\);\\ showing\\ 140\\]\\
PYTHON_GIL=1\\ python3\\.13t\\ program\\.py\\
```\\
\\
It\\ can\\ be\\ explicitly\\ forced\\ off\\ with:\\
\\
```sh\\
python3\\.13t\\ \\-X\\ gil=0\\ program\\.py\\
PYTHON_GIL=0\\ python3\\.13t\\ program\\.py\\
```\\
\\
`\\-X\\ gil=0,1`\\ forces\\ the\\ corresponding\\ state\\.\\ Forcing\\ `0`\\ is\\ available\\ only\\ in\\ a\\ build\\ configured\\ with\\ `\\-\\-disable\\-gil`\\.\\ The\\ command\\-line\\ option\\ takes\\ precedence\\ over\\ `PYTHON_GIL`\\.\\ \\ \\
Source:\\ <https://docs\\.python\\.org/3\\.13/using/cmdline\\.html\\#cmdoption\\-X>\\ and\\ <https://docs\\.python\\.org/3\\.13/using/cmdline\\.html\\#envvar\\-PYTHON_GIL>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/using/cmdline\\.html\\)\\)\\
\\
\\#\\#\\#\\ Automatic\\ fallback\\
\\
If\\ a\\ C\\ extension\\ does\\ not\\ explicitly\\ declare\\ that\\ it\\ is\\ safe\\ without\\ the\\ GIL:\\
\\
1\\.\\ importing\\ it\\ produces\\ a\\ warning;\\ and\\
2\\.\\ CPython\\ enables\\ the\\ GIL\\ at\\ runtime\\.\\
\\
The\\ module\\ declaration\\ defaults\\ to\\ `Py_MOD_GIL_USED`\\ when\\ no\\ declaration\\ is\\ present\\.\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\#module\\-initialization>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/c\\-api/module\\.html\\#c\\.Py_mod_gil>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\)\\)\\
\\
That\\ automatic\\ fallback\\ is\\ suppressed\\ if\\ the\\ user\\ explicitly\\ forced\\ the\\ GIL\\ off\\ with\\ `PYTHON_GIL=0`\\ or\\ `\\-X\\ gil=0`\\.\\ This\\ permits\\ loading\\ the\\ extension\\ without\\ fallback,\\ but\\ it\\ does\\ \\*\\*not\\*\\*\\ make\\ an\\ unsafe\\ extension\\ thread\\-safe\\.\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\#free\\-threaded\\-cpython>\\ \\ \\
<https://peps\\.python\\.org/pep\\-0703/\\#pythongil\\-environment\\-variable>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\)\\)\\
\\
\\-\\-\\-\\
\\
\\#\\#\\ 5\\.\\ Native\\-extension\\ compatibility\\ requirements\\
\\
\\#\\#\\#\\ Declaring\\ support\\
\\
For\\ a\\ multi\\-phase\\ initialized\\ extension,\\ add:\\
\\
```c\\
static\\ PyModuleDef_Slot\\ slots\\[\\]\\ =\\ \\{\\
\\#if\\ PY_VERSION_HEX\\ >=\\ 0x030D0000\\
\\ \\ \\ \\ \\{Py_mod_gil,\\ Py_MOD_GIL_NOT_USED\\},\\
\\#endif\\
\\ \\ \\ \\ \\{0,\\ NULL\\}\\
\\};\\
```\\
\\
`Py_MOD_GIL_NOT_USED`\\ asserts\\ that\\ the\\ module\\ is\\ safe\\ without\\ an\\ active\\ GIL\\.\\ If\\ the\\ slot\\ is\\ absent,\\ the\\ default\\ is\\ `Py_MOD_GIL_USED`\\.\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\#multi\\-phase\\-initialization>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/c\\-api/module\\.html\\#c\\.Py_mod_gil>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\)\\)\\
\\
For\\ single\\-phase\\ initialization:\\
\\
```c\\
\\#ifdef\\ Py_GIL_DISABLED\\
\\ \\ \\ \\ PyUnstable_Module_SetGIL\\(module,\\ Py_MOD_GIL_NOT_USED\\);\\
\\#endif\\
```\\
\\
`PyUnstable_Module_SetGIL\\(\\)`\\ exists\\ only\\ in\\ free\\-threaded\\ builds\\ and\\ is\\ part\\ of\\ the\\ Unstable\\ API;\\ its\\ signature\\ or\\ behavior\\ may\\ change\\ without\\ warning\\ in\\ minor\\ releases\\.\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\#single\\-phase\\-initialization>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/c\\-api/module\\.html\\#c\\.PyUnstable_Module_SetGIL>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\)\\)\\
\\
\\#\\#\\#\\ Being\\ thread\\-safe\\ in\\ practice\\
\\
The\\ declaration\\ is\\ only\\ an\\ assertion;\\ extension\\ authors\\ must\\ actually\\ remove\\ assumptions\\ that\\ the\\ GIL\\ protects\\ their\\ state\\.\\ Important\\ requirements\\ include:\\
\\
\\-\\ protect\\ global\\ caches\\ and\\ mutable\\ global/module\\ state\\ with\\ explicit\\ locks\\ or\\ thread\\-local\\ storage;\\
\\-\\ do\\ not\\ directly\\ access\\ concurrently\\ mutable\\ C\\-API\\ struct\\ fields;\\
\\-\\ do\\ not\\ use\\ unchecked\\ macros\\ such\\ as\\ `PyList_GET_ITEM`\\ against\\ concurrently\\ mutable\\ containers;\\
\\-\\ prefer\\ strong\\-reference\\ APIs\\ such\\ as\\ `PyList_GetItemRef\\(\\)`\\ and\\ `PyDict_GetItemRef\\(\\)`\\ where\\ another\\ thread\\ could\\ remove\\ the\\ underlying\\ item;\\
\\-\\ protect\\ `PyDict_Next\\(\\)`\\ iteration\\ with\\ a\\ Python\\ critical\\ section\\ when\\ concurrent\\ mutation\\ is\\ possible;\\
\\-\\ use\\ the\\ object\\-memory\\ domain\\ only\\ for\\ Python\\ objects,\\ and\\ do\\ not\\ allocate\\ ordinary\\ buffers\\ with\\ `PyObject_Malloc\\(\\)`;\\
\\-\\ continue\\ using\\ thread\\-state\\ APIs\\ such\\ as\\ `PyGILState_Ensure\\(\\)`,\\ `PyEval_SaveThread\\(\\)`,\\ and\\ `Py_BEGIN_ALLOW_THREADS`,\\ even\\ when\\ the\\ GIL\\ is\\ disabled\\.\\
\\
Source:\\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\)\\)\\
\\
On\\ Windows\\ 3\\.13,\\ an\\ extension\\ build\\ must\\ manually\\ pass\\ `Py_GIL_DISABLED=1`\\ to\\ the\\ compiler\\ because\\ the\\ official\\ installer\\ does\\ not\\ arrange\\ this\\ automatically\\.\\ \\ \\
Source:\\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\#windows>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\)\\)\\
\\
\\-\\-\\-\\
\\
\\#\\#\\ 6\\.\\ ABI\\ and\\ wheel\\ tagging\\
\\
\\-\\ The\\ free\\-threaded\\ and\\ normal\\ CPython\\ 3\\.13\\ builds\\ are\\ \\*\\*not\\ ABI\\-compatible\\*\\*\\.\\ Native\\ extensions\\ must\\ be\\ rebuilt\\ specifically\\ for\\ the\\ free\\-threaded\\ build\\.\\ \\ \\
\\ \\ Sources:\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\#building\\-extensions\\-for\\-the\\-free\\-threaded\\-build>\\ \\ \\
\\ \\ <https://peps\\.python\\.org/pep\\-0703/\\#backwards\\-compatibility>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\)\\)\\
\\
\\-\\ The\\ free\\-threaded\\ ABI\\ uses\\ a\\ `t`\\ suffix\\.\\ This\\ appears\\ in\\ `sys\\.abiflags`,\\ extension/shared\\-library\\ naming,\\ executables,\\ and\\ wheel\\ ABI\\ identification\\.\\ POSIX\\ `pkg\\-config`\\ files\\ are\\ likewise\\ named,\\ for\\ example,\\ `python\\-3\\.13t\\.pc`\\.\\ \\ \\
\\ \\ Sources:\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/using/configure\\.html\\#cmdoption\\-disable\\-gil>\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\#building\\-extensions\\-for\\-the\\-free\\-threaded\\-build>\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\#build\\-changes>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/using/configure\\.html\\?utm_source=openai\\)\\)\\
\\
\\-\\ CPython\\ 3\\.13’s\\ free\\-threaded\\ build\\ does\\ \\*\\*not\\*\\*\\ support\\ the\\ Limited\\ C\\ API\\ or\\ Stable\\ ABI\\ for\\ free\\-threaded\\ native\\ extensions\\.\\ A\\ project\\ using\\ `py_limited_api=True`\\ must\\ opt\\ out\\ for\\ this\\ build,\\ and\\ separate\\ free\\-threaded\\ wheels\\ are\\ required\\.\\ \\ \\
\\ \\ Source:\\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\#limited\\-c\\-api\\-and\\-stable\\-abi>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\)\\)\\
\\
\\-\\ `Py_mod_gil`\\ itself\\ is\\ listed\\ as\\ part\\ of\\ the\\ Stable\\ ABI\\ since\\ 3\\.13,\\ but\\ that\\ does\\ not\\ make\\ a\\ CPython\\ 3\\.13\\ `abi3`\\ extension\\ binary\\-compatible\\ with\\ the\\ free\\-threaded\\ interpreter\\.\\ The\\ free\\-threaded\\-build\\ limitation\\ above\\ still\\ applies\\.\\ \\ \\
\\ \\ Sources:\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/c\\-api/module\\.html\\#c\\.Py_mod_gil>\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\#limited\\-c\\-api\\-and\\-stable\\-abi>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/c\\-api/module\\.html\\)\\)\\
\\
\\#\\#\\#\\ Wheel\\ filename\\ —\\ explicit\\ inference\\
\\
\\*\\*Inference\\ from\\ the\\ official\\ ABI\\ and\\ wheel\\-tag\\ specifications:\\*\\*\\ a\\ platform\\-specific\\ CPython\\ 3\\.13\\ free\\-threaded\\ wheel\\ conventionally\\ has\\ compatibility\\ tags\\ of\\ the\\ form:\\
\\
```text\\
cp313\\-cp313t\\-<platform>\\
```\\
\\
For\\ example:\\
\\
```text\\
package\\-1\\.0\\-cp313\\-cp313t\\-manylinux_2_17_x86_64\\.whl\\
```\\
\\
Reasoning:\\
\\
1\\.\\ CPython’s\\ free\\-threaded\\ ABI\\ includes\\ `t`;\\
2\\.\\ CPython’s\\ version/implementation\\ Python\\ tag\\ is\\ `cp313`;\\
3\\.\\ wheel\\ filenames\\ encode\\ `\\{python\\ tag\\}\\-\\{abi\\ tag\\}\\-\\{platform\\ tag\\}`\\.\\
\\
The\\ CPython\\ 3\\.13\\ HOWTO\\ directly\\ guarantees\\ the\\ `t`\\-suffixed\\ free\\-threaded\\ ABI\\ and\\ separate\\ wheels,\\ but\\ does\\ not\\ itself\\ spell\\ out\\ this\\ complete\\ example\\ filename\\.\\ \\ \\
Sources:\\ \\ \\
<https://peps\\.python\\.org/pep\\-0703/\\#build\\-configuration\\-changes>\\ \\ \\
<https://peps\\.python\\.org/pep\\-0425/\\#overview>\\ \\ \\
<https://peps\\.python\\.org/pep\\-0427/\\#file\\-name\\-convention>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\#building\\-extensions\\-for\\-the\\-free\\-threaded\\-build>\\ \\(\\[peps\\.python\\.org\\]\\(https://peps\\.python\\.org/pep\\-0703/\\)\\)\\
\\
Pure\\-Python\\ wheels\\ with\\ no\\ ABI\\ requirement,\\ such\\ as\\ `py3\\-none\\-any`,\\ are\\ not\\ subject\\ to\\ the\\ native\\-extension\\ ABI\\ split;\\ the\\ separate\\-build\\ requirement\\ specifically\\ concerns\\ C\\-API/native\\ extensions\\.\\ \\ \\
Sources:\\ \\ \\
<https://peps\\.python\\.org/pep\\-0425/\\#abi\\-tag>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html\\#building\\-extensions\\-for\\-the\\-free\\-threaded\\-build>\\ \\(\\[peps\\.python\\.org\\]\\(https://peps\\.python\\.org/pep\\-0425/\\)\\)\\
\\
\\-\\-\\-\\
\\
\\#\\#\\ 7\\.\\ Important\\ limitations\\ and\\ performance\\ caveats\
"""
                        ),
                        tool_call_id='call_Uc4auU7ikVIFweZ0PbbYlD4w',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- researcher: Research a focused sub-question on the web and report back with findings and source links\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='read_tool_result',
                        args=IsStr(
                            regex='\\{"handle":"01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_ThjTLUpjfRlxSV43zSkDc4qF\\.0","offset":260,"limit":71\\}'
                        ),
                        tool_call_id='call_WGPuExRtAPl7o0U4ZQnJiya0',
                        id='fc_0e58f8341a296874006a7f846069e4819682547ca97b99ee99',
                        provider_name='openai',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='gpt-5.6-sol',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_0e58f8341a296874006a7f845f3fdc8196a12485e126e3140c',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='read_tool_result',
                        content=IsStr(
                            regex="""\
\\[handle\\ '01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_ThjTLUpjfRlxSV43zSkDc4qF\\.0':\\ 331\\ matching\\ line\\(s\\);\\ showing\\ 71\\]\\
\\
\\#\\#\\#\\ Single\\-threaded\\ performance\\
\\
CPython\\ 3\\.13’s\\ free\\-threaded\\ build\\ has\\ approximately\\ \\*\\*40%\\ overhead\\ on\\ `pyperformance`\\*\\*\\ versus\\ the\\ normal\\ GIL\\ build\\.\\ Programs\\ dominated\\ by\\ I/O\\ or\\ C\\-extension\\ work\\ may\\ see\\ less\\ impact\\.\\ The\\ largest\\ stated\\ cause\\ is\\ that\\ the\\ specializing\\ adaptive\\ interpreter\\ is\\ disabled\\ in\\ the\\ 3\\.13\\ free\\-threaded\\ build\\.\\ \\ \\
Source:\\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\#single\\-threaded\\-performance>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
This\\ 40%\\ figure\\ from\\ the\\ implemented\\ 3\\.13\\ documentation\\ supersedes\\ PEP\\ 703’s\\ earlier\\ prototype\\ performance\\ estimates\\.\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\#single\\-threaded\\-performance>\\ \\ \\
<https://peps\\.python\\.org/pep\\-0703/\\#performance>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
\\#\\#\\#\\ Immortalization\\ and\\ memory\\ growth\\
\\
After\\ the\\ first\\ additional\\ thread\\ starts,\\ the\\ 3\\.13\\ free\\-threaded\\ build\\ immortalizes\\ several\\ kinds\\ of\\ objects,\\ including\\ module\\-level\\ functions,\\ method\\ descriptors,\\ code\\ objects,\\ modules\\ and\\ their\\ dictionaries,\\ and\\ classes\\.\\ Numeric\\ and\\ string\\ literals\\ and\\ strings\\ returned\\ by\\ `sys\\.intern\\(\\)`\\ are\\ also\\ immortalized\\.\\ Since\\ immortal\\ objects\\ are\\ not\\ deallocated,\\ applications\\ creating\\ many\\ such\\ objects\\ can\\ consume\\ more\\ memory\\.\\ \\ \\
Source:\\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\#immortalization>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
\\#\\#\\#\\ Frames\\
\\
Accessing\\ a\\ frame\\ object\\ from\\ a\\ different\\ thread\\ is\\ unsafe\\ and\\ may\\ crash\\ the\\ process\\.\\ Consequently,\\ `sys\\._current_frames\\(\\)`\\ is\\ generally\\ unsafe\\ in\\ this\\ build\\.\\ `inspect\\.currentframe\\(\\)`\\ and\\ `sys\\._getframe\\(\\)`\\ are\\ generally\\ safe\\ only\\ while\\ the\\ resulting\\ frame\\ remains\\ in\\ the\\ same\\ thread\\.\\ \\ \\
Source:\\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\#frame\\-objects>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
\\#\\#\\#\\ Iterators\\
\\
Sharing\\ one\\ iterator\\ object\\ between\\ threads\\ is\\ generally\\ unsafe:\\ elements\\ may\\ be\\ duplicated\\ or\\ omitted,\\ and\\ the\\ interpreter\\ may\\ crash\\.\\ \\ \\
Source:\\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\#iterators>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
\\#\\#\\#\\ Containers\\ are\\ protected,\\ but\\ not\\ a\\ synchronization\\ contract\\
\\
`dict`,\\ `list`,\\ and\\ `set`\\ use\\ internal\\ locks\\ in\\ the\\ current\\ free\\-threaded\\ implementation\\ to\\ provide\\ behavior\\ similar\\ to\\ GIL\\-enabled\\ CPython\\ under\\ concurrent\\ operations\\.\\ However,\\ Python\\ has\\ never\\ guaranteed\\ particular\\ results\\ for\\ concurrent\\ mutation\\ of\\ these\\ containers,\\ so\\ this\\ is\\ explicitly\\ an\\ implementation\\ description—not\\ a\\ present\\ or\\ future\\ language\\ guarantee\\.\\ Explicit\\ `threading\\.Lock`\\ synchronization\\ is\\ recommended\\.\\ \\ \\
Source:\\ <https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\#thread\\-safety>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
\\#\\#\\#\\ Parallelism\\ is\\ not\\ automatic\\
\\
The\\ build\\ permits\\ Python\\ threads\\ to\\ execute\\ in\\ parallel\\ on\\ multiple\\ cores,\\ but\\ the\\ documentation\\ expressly\\ says\\ that\\ not\\ all\\ software\\ benefits\\ automatically\\.\\ Benefits\\ depend\\ on\\ workload,\\ thread\\ design,\\ contention,\\ extension\\ behavior,\\ and\\ whether\\ an\\ imported\\ extension\\ causes\\ GIL\\ fallback\\.\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/whatsnew/3\\.13\\.html\\#free\\-threaded\\-cpython>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
\\-\\-\\-\\
\\
\\#\\#\\ 8\\.\\ Guarantees—and\\ what\\ is\\ not\\ guaranteed\\
\\
\\#\\#\\#\\ Reasonably\\ firm\\ 3\\.13\\ interfaces\\
\\
\\-\\ `\\-\\-disable\\-gil`,\\ `PYTHON_GIL`,\\ `\\-X\\ gil=0,1`,\\ `Py_GIL_DISABLED`,\\ `sys\\._is_gil_enabled\\(\\)`,\\ and\\ `Py_mod_gil`\\ were\\ added\\ and\\ documented\\ in\\ 3\\.13\\.\\ \\ \\
\\ \\ Sources:\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/using/configure\\.html\\#cmdoption\\-disable\\-gil>\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/using/cmdline\\.html\\#envvar\\-PYTHON_GIL>\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/library/sys\\.html\\#sys\\._is_gil_enabled>\\ \\ \\
\\ \\ <https://docs\\.python\\.org/3\\.13/c\\-api/module\\.html\\#c\\.Py_mod_gil>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/using/configure\\.html\\?utm_source=openai\\)\\)\\
\\
\\#\\#\\#\\ Explicitly\\ not\\ guaranteed\\
\\
\\-\\ The\\ overall\\ mode\\ is\\ experimental\\ and\\ may\\ contain\\ bugs\\ or\\ change\\ in\\ later\\ releases\\.\\ \\ \\
\\-\\ Windows/macOS\\ installation\\ layout\\ and\\ behavior\\ are\\ experimental\\.\\ \\ \\
\\-\\ Internal\\ locking\\ of\\ built\\-in\\ containers\\ is\\ not\\ a\\ language\\-level\\ concurrency\\ guarantee\\.\\ \\ \\
\\-\\ Cross\\-thread\\ frame\\ access\\ and\\ shared\\ iterators\\ are\\ not\\ generally\\ safe\\.\\ \\ \\
\\-\\ `sys\\._is_gil_enabled\\(\\)`\\ is\\ CPython\\-specific,\\ not\\ guaranteed\\ across\\ Python\\ implementations\\.\\ \\ \\
\\-\\ `PyUnstable_Module_SetGIL\\(\\)`\\ may\\ change\\ without\\ warning\\ in\\ minor\\ releases\\.\\ \\ \\
\\-\\ A\\ `t`\\-tagged\\ extension\\ binary\\ is\\ not\\ automatically\\ thread\\-safe\\ merely\\ because\\ it\\ was\\ compiled\\ for\\ the\\ free\\-threaded\\ ABI;\\ it\\ must\\ correctly\\ synchronize\\ its\\ own\\ state\\ and\\ accurately\\ declare\\ its\\ GIL\\ requirement\\.\\ \\ \\
\\-\\ There\\ is\\ no\\ guarantee\\ that\\ a\\ workload\\ will\\ become\\ faster;\\ substantial\\ single\\-thread\\ overhead\\ is\\ documented\\ for\\ 3\\.13\\.\\ \\ \\
\\-\\ PEP\\ 703’s\\ rollout\\ itself\\ retains\\ an\\ explicit\\ policy\\ escape\\ hatch\\ for\\ rollback\\.\\
\\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-extensions\\.html>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/c\\-api/module\\.html\\#c\\.PyUnstable_Module_SetGIL>\\ \\ \\
<https://docs\\.python\\.org/3\\.13/using/windows\\.html\\#installing\\-free\\-threaded\\-binaries>\\ \\ \\
<https://peps\\.python\\.org/pep\\-0703/>\\ \\(\\[docs\\.python\\.org\\]\\(https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html\\)\\)\\
\\
\\*\\*Bottom\\ line:\\*\\*\\ CPython\\ 3\\.13\\ free\\ threading\\ is\\ suitable\\ for\\ experimentation,\\ porting,\\ compatibility\\ testing,\\ and\\ carefully\\ controlled\\ workloads\\.\\ Its\\ own\\ official\\ documentation\\ does\\ not\\ present\\ it\\ as\\ a\\ drop\\-in\\ production\\ replacement\\ for\\ ordinary\\ CPython\\ 3\\.13\\.\
"""
                        ),
                        tool_call_id='call_WGPuExRtAPl7o0U4ZQnJiya0',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- researcher: Research a focused sub-question on the web and report back with findings and source links\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={'pattern': 'Limited C API and Stable ABI', 'type': 'find_in_page'},
                        tool_call_id='ws_0e58f8341a296874006a7f846a28a48196a9f3353186566c64',
                        id='ws_0e58f8341a296874006a7f846a28a48196a9f3353186566c64',
                        provider_name='openai',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e58f8341a296874006a7f846a28a48196a9f3353186566c64',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='web_fetch',
                        args='{"url":"https://docs.python.org/3.14/howto/free-threading-extensions.html"}',
                        tool_call_id='call_dmEMWZ9DEj1LpBAv20CBRl4f',
                        id='fc_0e58f8341a296874006a7f846cb90881969b89b1d5de82c56f',
                        provider_name='openai',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='gpt-5.6-sol',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_0e58f8341a296874006a7f8461704c81968b9c4059581c7f0b',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='web_fetch',
                        content=IsStr(
                            regex="""\
\\[Tool\\ output\\ too\\ large\\ \\(28,078\\ chars\\);\\ stored\\ to\\ handle\\ '01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_dmEMWZ9DEj1LpBAv20CBRl4f\\.0'\\.\\ Read\\ it\\ with\\ read_tool_result\\(handle='01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_dmEMWZ9DEj1LpBAv20CBRl4f\\.0',\\ offset=0,\\ limit=200,\\ from_end=False,\\ pattern=None\\)\\.\\]\\
shape:\\ \\{'url':\\ str,\\ 'title':\\ str,\\ 'content':\\ str\\}\\
\\{"url":"https://docs\\.python\\.org/3\\.14/howto/free\\-threading\\-extensions\\.html","title":"C\\ API\\ Extension\\ Support\\ for\\ Free\\ Threading\\ \\&\\#8212;\\ Python\\ 3\\.14\\.7\\ documentation","content":"C\\ API\\ Extension\\ Support\\ for\\ Free\\ Threading\\ —\\ Python\\ 3\\.14\\.7\\ documentation\\\\n\\\\n@media\\ only\\ screen\\ \\{\\\\ntable\\.full\\-width\\-table\\ \\{\\\\nwidth:\\ 100%;\\\\n\\}\\\\n\\}\\\\n\\\\nTheme\\\\nAuto\\\\nLight\\\\nDark\\\\n\\\\n\\#\\#\\#\\ \\[Table\\ of\\ Contents\\]\\(\\.\\./contents\\.html\\)\\\\n\\\\n\\*\\ \\[C\\ API\\ Extension\\ Support\\ for\\ Free\\ Threading\\]\\(\\#\\)\\\\n\\ \\ \\+\\ \\[Identifying\\ the\\ Free\\-Threaded\\ Build\\ in\\ C\\]\\(\\#identif\\
\\.\\.\\.\\[27,078\\ chars\\ omitted\\]\\.\\.\\.\\
hon\\ Software\\ Foundation\\ License\\ Version\\ 2\\.\\\\n\\ \\ \\\\nExamples,\\ recipes,\\ and\\ other\\ code\\ in\\ the\\ documentation\\ are\\ additionally\\ licensed\\ under\\ the\\ Zero\\ Clause\\ BSD\\ License\\.\\\\n\\ \\ \\\\nSee\\ \\[History\\ and\\ License\\]\\(/license\\.html\\)\\ for\\ more\\ information\\.\\ \\ \\\\n\\ \\ \\\\nThe\\ Python\\ Software\\ Foundation\\ is\\ a\\ non\\-profit\\ corporation\\.\\\\n\\[Please\\ donate\\.\\]\\(https://www\\.python\\.org/psf/donations/\\)\\\\n\\ \\ \\\\n\\ \\ \\\\nLast\\ updated\\ on\\ Aug\\ 14,\\ 2026\\ \\(13:12\\ UTC\\)\\.\\\\n\\[Found\\ a\\ bug\\]\\(/bugs\\.html\\)\\?\\\\n\\ \\ \\\\nCreated\\ using\\ \\[Sphinx\\]\\(https://www\\.sphinx\\-doc\\.org/\\)\\ 8\\.2\\.3\\."\\}\
"""
                        ),
                        tool_call_id='call_dmEMWZ9DEj1LpBAv20CBRl4f',
                        metadata={
                            'overflow_handle': IsStr(
                                regex='01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_dmEMWZ9DEj1LpBAv20CBRl4f\\.0'
                            ),
                            'overflow_bytes': 28155,
                        },
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- researcher: Research a focused sub-question on the web and report back with findings and source links\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='read_tool_result',
                        args=IsStr(
                            regex='\\{"handle":"01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_dmEMWZ9DEj1LpBAv20CBRl4f\\.0","pattern":"Stable\\ ABI","limit":50\\}'
                        ),
                        tool_call_id='call_9QGqlCO9vWmh9yKF1C730LaP',
                        id='fc_0e58f8341a296874006a7f846fa6188196bd76e08b904579c3',
                        provider_name='openai',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='gpt-5.6-sol',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_0e58f8341a296874006a7f846e25b08196845460b1804c2e50',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='read_tool_result',
                        content=IsStr(
                            regex="""\
\\[handle\\ '01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_dmEMWZ9DEj1LpBAv20CBRl4f\\.0':\\ 1\\ matching\\ line\\(s\\);\\ showing\\ 1\\]\\
\\{"url":"https://docs\\.python\\.org/3\\.14/howto/free\\-threading\\-extensions\\.html","title":"C\\ API\\ Extension\\ Support\\ for\\ Free\\ Threading\\ \\&\\#8212;\\ Python\\ 3\\.14\\.7\\ documentation","content":"C\\ API\\ Extension\\ Support\\ for\\ Free\\ Threading\\ —\\ Python\\ 3\\.14\\.7\\ documentation\\\\n\\\\n@media\\ only\\ screen\\ \\{\\\\ntable\\.full\\-width\\-table\\ \\{\\\\nwidth:\\ 100%;\\\\n\\}\\\\n\\}\\\\n\\\\nTheme\\\\nAuto\\\\nLight\\\\nDark\\\\n\\\\n\\#\\#\\#\\ \\[Table\\ of\\ Contents\\]\\(\\.\\./contents\\.html\\)\\\\n\\\\n\\*\\ \\[C\\ API\\ Extension\\ Support\\ for\\ Free\\ Threading\\]\\(\\#\\)\\\\n\\ \\ \\+\\ \\[Identifying\\ the\\ Free\\-Threaded\\ Build\\ in\\ C\\]\\(\\#identifying\\-the\\-free\\-threaded\\-build\\-in\\-c\\)\\\\n\\ \\ \\+\\ \\[Module\\ Initialization\\]\\(\\#module\\-initialization\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Multi\\-Phase\\ Initialization\\]\\(\\#multi\\-phase\\-initialization\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Single\\-Phase\\ Initialization\\]\\(\\#single\\-phase\\-initialization\\)\\\\n\\ \\ \\+\\ \\[General\\ API\\ Guidelines\\]\\(\\#general\\-api\\-guidelines\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Container\\ Thread\\ Safety\\]\\(\\#container\\-thread\\-safety\\)\\\\n\\ \\ \\ \\ \\ \\ \\*\\ \\[`PyDict_Next`\\]\\(\\#pydict\\-next\\)\\\\n\\ \\ \\+\\ \\[Borrowed\\ References\\]\\(\\#borrowed\\-references\\)\\\\n\\ \\ \\+\\ \\[Memory\\ Allocation\\ APIs\\]\\(\\#memory\\-allocation\\-apis\\)\\\\n\\ \\ \\+\\ \\[Thread\\ State\\ and\\ GIL\\ APIs\\]\\(\\#thread\\-state\\-and\\-gil\\-apis\\)\\\\n\\ \\ \\+\\ \\[Protecting\\ Internal\\ Extension\\ State\\]\\(\\#protecting\\-internal\\-extension\\-state\\)\\\\n\\ \\ \\+\\ \\[Critical\\ Sections\\]\\(\\#critical\\-sections\\)\\\\n\\ \\ \\ \\ \\-\\ \\[What\\ Are\\ Critical\\ Sections\\?\\]\\(\\#what\\-are\\-critical\\-sections\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Using\\ Critical\\ Sections\\]\\(\\#using\\-critical\\-sections\\)\\\\n\\ \\ \\ \\ \\-\\ \\[How\\ Critical\\ Sections\\ Work\\]\\(\\#how\\-critical\\-sections\\-work\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Deadlock\\ Avoidance\\]\\(\\#deadlock\\-avoidance\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Important\\ Considerations\\]\\(\\#important\\-considerations\\)\\\\n\\ \\ \\+\\ \\[Building\\ Extensions\\ for\\ the\\ Free\\-Threaded\\ Build\\]\\(\\#building\\-extensions\\-for\\-the\\-free\\-threaded\\-build\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Limited\\ C\\ API\\ and\\ Stable\\ ABI\\]\\(\\#limited\\-c\\-api\\-and\\-stable\\-abi\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Windows\\]\\(\\#windows\\)\\\\n\\\\n\\#\\#\\#\\#\\ Previous\\ topic\\\\n\\\\n\\[Python\\ support\\ for\\ free\\ threading\\]\\(free\\-threading\\-python\\.html\\ \\\\"previous\\ chapter\\\\"\\)\\\\n\\\\n\\#\\#\\#\\#\\ Next\\ topic\\\\n\\\\n\\[Remote\\ debugging\\ attachment\\ protocol\\]\\(remote_debugging\\.html\\ \\\\"next\\ chapter\\\\"\\)\\\\n\\\\ndocument\\.addEventListener\\('DOMContentLoaded',\\ \\(\\)\\ =>\\ \\{\\\\nconst\\ title\\ =\\ document\\.querySelector\\('meta\\[property=\\\\"og:title\\\\"\\]'\\)\\.content;\\\\nconst\\ elements\\ =\\ document\\.querySelectorAll\\('\\.improvepage'\\);\\\\nconst\\ pageurl\\ =\\ window\\.location\\.href\\.split\\('\\?'\\)\\[0\\];\\\\nelements\\.forEach\\(element\\ =>\\ \\{\\\\nconst\\ url\\ =\\ new\\ URL\\(element\\.href\\.split\\('\\?'\\)\\[0\\]\\.replace\\(\\\\"\\-nojs\\\\",\\ \\\\"\\\\"\\)\\);\\\\nurl\\.searchParams\\.set\\('pagetitle',\\ title\\);\\\\nurl\\.searchParams\\.set\\('pageurl',\\ pageurl\\);\\\\nurl\\.searchParams\\.set\\('pagesource',\\ \\\\"howto/free\\-threading\\-extensions\\.rst\\\\"\\);\\\\nelement\\.href\\ =\\ url\\.toString\\(\\);\\\\n\\}\\);\\\\n\\}\\);\\\\n\\\\n\\#\\#\\#\\ This\\ page\\\\n\\\\n\\*\\ \\[Report\\ a\\ bug\\]\\(\\.\\./bugs\\.html\\)\\\\n\\*\\ \\[Improve\\ this\\ page\\]\\(\\.\\./improve\\-page\\-nojs\\.html\\)\\\\n\\*\\ \\[Show\\ source\\]\\(https://github\\.com/python/cpython/blob/main/Doc/howto/free\\-threading\\-extensions\\.rst\\?plain=1\\)\\\\n\\\\n\\#\\#\\#\\ Navigation\\\\n\\\\n\\*\\ \\[index\\]\\(\\.\\./genindex\\.html\\ \\\\"General\\ Index\\\\"\\)\\\\n\\*\\ \\[modules\\]\\(\\.\\./py\\-modindex\\.html\\ \\\\"Python\\ Module\\ Index\\\\"\\)\\ \\|\\\\n\\*\\ \\[next\\]\\(remote_debugging\\.html\\ \\\\"Remote\\ debugging\\ attachment\\ protocol\\\\"\\)\\ \\|\\\\n\\*\\ \\[previous\\]\\(free\\-threading\\-python\\.html\\ \\\\"Python\\ support\\ for\\ free\\ threading\\\\"\\)\\ \\|\\\\n\\*\\ \\[Python\\]\\(https://www\\.python\\.org/\\)\\ »\\\\n\\\\n\\*\\ \\[3\\.14\\.7\\ Documentation\\]\\(\\.\\./index\\.html\\)\\ »\\\\n\\*\\ \\[Python\\ HOWTOs\\]\\(index\\.html\\)\\ »\\\\n\\*\\ C\\ API\\ Extension\\ Support\\ for\\ Free\\ Threading\\\\n\\*\\ \\|\\\\n\\*\\ Theme\\\\n\\ \\ Auto\\\\n\\ \\ Light\\\\n\\ \\ Dark\\\\n\\ \\ \\ \\|\\\\n\\\\nC\\ API\\ Extension\\ Support\\ for\\ Free\\ Threading\\[¶\\]\\(\\#c\\-api\\-extension\\-support\\-for\\-free\\-threading\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n=================================================================================================================\\\\n\\\\nStarting\\ with\\ the\\ 3\\.13\\ release,\\ CPython\\ has\\ support\\ for\\ running\\ with\\\\nthe\\ \\[global\\ interpreter\\ lock\\]\\(\\.\\./glossary\\.html\\#term\\-global\\-interpreter\\-lock\\)\\ \\(GIL\\)\\ disabled\\ in\\ a\\ configuration\\\\ncalled\\ \\[free\\ threading\\]\\(\\.\\./glossary\\.html\\#term\\-free\\-threading\\)\\.\\ This\\ document\\ describes\\ how\\ to\\ adapt\\ C\\ API\\\\nextensions\\ to\\ support\\ free\\ threading\\.\\\\n\\\\nIdentifying\\ the\\ Free\\-Threaded\\ Build\\ in\\ C\\[¶\\]\\(\\#identifying\\-the\\-free\\-threaded\\-build\\-in\\-c\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\\\n\\\\nThe\\ CPython\\ C\\ API\\ exposes\\ the\\ `Py_GIL_DISABLED`\\ macro:\\ in\\ the\\ free\\-threaded\\\\nbuild\\ it’s\\ defined\\ to\\ `1`,\\ and\\ in\\ the\\ regular\\ build\\ it’s\\ not\\ defined\\.\\\\nYou\\ can\\ use\\ it\\ to\\ enable\\ code\\ that\\ only\\ runs\\ under\\ the\\ free\\-threaded\\ build:\\\\n\\\\n```\\\\n\\#ifdef\\ Py_GIL_DISABLED\\\\n/\\*\\ code\\ that\\ only\\ runs\\ in\\ the\\ free\\-threaded\\ build\\ \\*/\\\\n\\#endif\\\\n```\\\\n\\\\nNote\\\\n\\\\nOn\\ Windows,\\ this\\ macro\\ is\\ not\\ defined\\ automatically,\\ but\\ must\\ be\\ specified\\\\nto\\ the\\ compiler\\ when\\ building\\.\\ The\\ \\[`sysconfig\\.get_config_var\\(\\)`\\]\\(\\.\\./library/sysconfig\\.html\\#sysconfig\\.get_config_var\\ \\\\"sysconfig\\.get_config_var\\\\"\\)\\ function\\\\ncan\\ be\\ used\\ to\\ determine\\ whether\\ the\\ current\\ running\\ interpreter\\ had\\ the\\\\nmacro\\ defined\\.\\\\n\\\\nModule\\ Initialization\\[¶\\]\\(\\#module\\-initialization\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\\\n\\\\nExtension\\ modules\\ need\\ to\\ explicitly\\ indicate\\ that\\ they\\ support\\ running\\ with\\\\nthe\\ GIL\\ disabled;\\ otherwise\\ importing\\ the\\ extension\\ will\\ raise\\ a\\ warning\\ and\\\\nenable\\ the\\ GIL\\ at\\ runtime\\.\\\\n\\\\nThere\\ are\\ two\\ ways\\ to\\ indicate\\ that\\ an\\ extension\\ module\\ supports\\ running\\ with\\\\nthe\\ GIL\\ disabled\\ depending\\ on\\ whether\\ the\\ extension\\ uses\\ multi\\-phase\\ or\\\\nsingle\\-phase\\ initialization\\.\\\\n\\\\n\\#\\#\\#\\ Multi\\-Phase\\ Initialization\\[¶\\]\\(\\#multi\\-phase\\-initialization\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\\\nExtensions\\ that\\ use\\ multi\\-phase\\ initialization\\ \\(i\\.e\\.,\\\\n\\[`PyModuleDef_Init\\(\\)`\\]\\(\\.\\./c\\-api/extension\\-modules\\.html\\#c\\.PyModuleDef_Init\\ \\\\"PyModuleDef_Init\\\\"\\)\\)\\ should\\ add\\ a\\ \\[`Py_mod_gil`\\]\\(\\.\\./c\\-api/module\\.html\\#c\\.Py_mod_gil\\ \\\\"Py_mod_gil\\\\"\\)\\ slot\\ in\\ the\\\\nmodule\\ definition\\.\\ If\\ your\\ extension\\ supports\\ older\\ versions\\ of\\ CPython,\\\\nyou\\ should\\ guard\\ the\\ slot\\ with\\ a\\ \\[`PY_VERSION_HEX`\\]\\(\\.\\./c\\-api/apiabiversion\\.html\\#c\\.PY_VERSION_HEX\\ \\\\"PY_VERSION_HEX\\\\"\\)\\ check\\.\\\\n\\\\n```\\\\nstatic\\ struct\\ PyModuleDef_Slot\\ module_slots\\[\\]\\ =\\ \\{\\\\n\\ \\ \\ \\ \\.\\.\\.\\\\n\\#if\\ PY_VERSION_HEX\\ >=\\ 0x030D0000\\\\n\\ \\ \\ \\ \\{Py_mod_gil,\\ Py_MOD_GIL_NOT_USED\\},\\\\n\\#endif\\\\n\\ \\ \\ \\ \\{0,\\ NULL\\}\\\\n\\};\\\\n\\\\nstatic\\ struct\\ PyModuleDef\\ moduledef\\ =\\ \\{\\\\n\\ \\ \\ \\ PyModuleDef_HEAD_INIT,\\\\n\\ \\ \\ \\ \\.m_slots\\ =\\ module_slots,\\\\n\\ \\ \\ \\ \\.\\.\\.\\\\n\\};\\\\n```\\\\n\\\\n\\#\\#\\#\\ Single\\-Phase\\ Initialization\\[¶\\]\\(\\#single\\-phase\\-initialization\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\\\nExtensions\\ that\\ use\\ single\\-phase\\ initialization\\ \\(i\\.e\\.,\\\\n\\[`PyModule_Create\\(\\)`\\]\\(\\.\\./c\\-api/module\\.html\\#c\\.PyModule_Create\\ \\\\"PyModule_Create\\\\"\\)\\)\\ should\\ call\\ \\[`PyUnstable_Module_SetGIL\\(\\)`\\]\\(\\.\\./c\\-api/module\\.html\\#c\\.PyUnstable_Module_SetGIL\\ \\\\"PyUnstable_Module_SetGIL\\\\"\\)\\ to\\\\nindicate\\ that\\ they\\ support\\ running\\ with\\ the\\ GIL\\ disabled\\.\\ The\\ function\\ is\\\\nonly\\ defined\\ in\\ the\\ free\\-threaded\\ build,\\ so\\ you\\ should\\ guard\\ the\\ call\\ with\\\\n`\\#ifdef\\ Py_GIL_DISABLED`\\ to\\ avoid\\ compilation\\ errors\\ in\\ the\\ regular\\ build\\.\\\\n\\\\n```\\\\nstatic\\ struct\\ PyModuleDef\\ moduledef\\ =\\ \\{\\\\n\\ \\ \\ \\ PyModuleDef_HEAD_INIT,\\\\n\\ \\ \\ \\ \\.\\.\\.\\\\n\\};\\\\n\\\\nPyMODINIT_FUNC\\\\nPyInit_mymodule\\(void\\)\\\\n\\{\\\\n\\ \\ \\ \\ PyObject\\ \\*m\\ =\\ PyModule_Create\\(\\&moduledef\\);\\\\n\\ \\ \\ \\ if\\ \\(m\\ ==\\ NULL\\)\\ \\{\\\\n\\ \\ \\ \\ \\ \\ \\ \\ return\\ NULL;\\\\n\\ \\ \\ \\ \\}\\\\n\\#ifdef\\ Py_GIL_DISABLED\\\\n\\ \\ \\ \\ PyUnstable_Module_SetGIL\\(m,\\ Py_MOD_GIL_NOT_USED\\);\\\\n\\#endif\\\\n\\ \\ \\ \\ return\\ m;\\\\n\\}\\\\n```\\\\n\\\\nGeneral\\ API\\ Guidelines\\[¶\\]\\(\\#general\\-api\\-guidelines\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\\\n\\\\nMost\\ of\\ the\\ C\\ API\\ is\\ thread\\-safe,\\ but\\ there\\ are\\ some\\ exceptions\\.\\\\n\\\\n\\*\\ \\*\\*Struct\\ Fields\\*\\*:\\ Accessing\\ fields\\ in\\ Python\\ C\\ API\\ objects\\ or\\ structs\\\\n\\ \\ directly\\ is\\ not\\ thread\\-safe\\ if\\ the\\ field\\ may\\ be\\ concurrently\\ modified\\.\\\\n\\*\\ \\*\\*Macros\\*\\*:\\ Accessor\\ macros\\ like\\ \\[`PyList_GET_ITEM`\\]\\(\\.\\./c\\-api/list\\.html\\#c\\.PyList_GET_ITEM\\ \\\\"PyList_GET_ITEM\\\\"\\),\\\\n\\ \\ \\[`PyList_SET_ITEM`\\]\\(\\.\\./c\\-api/list\\.html\\#c\\.PyList_SET_ITEM\\ \\\\"PyList_SET_ITEM\\\\"\\),\\ and\\ macros\\ like\\\\n\\ \\ \\[`PySequence_Fast_GET_SIZE`\\]\\(\\.\\./c\\-api/sequence\\.html\\#c\\.PySequence_Fast_GET_SIZE\\ \\\\"PySequence_Fast_GET_SIZE\\\\"\\)\\ that\\ use\\ the\\ object\\ returned\\ by\\\\n\\ \\ \\[`PySequence_Fast\\(\\)`\\]\\(\\.\\./c\\-api/sequence\\.html\\#c\\.PySequence_Fast\\ \\\\"PySequence_Fast\\\\"\\)\\ do\\ not\\ perform\\ any\\ error\\ checking\\ or\\ locking\\.\\\\n\\ \\ These\\ macros\\ are\\ not\\ thread\\-safe\\ if\\ the\\ container\\ object\\ may\\ be\\ modified\\\\n\\ \\ concurrently\\.\\\\n\\*\\ \\*\\*Borrowed\\ References\\*\\*:\\ C\\ API\\ functions\\ that\\ return\\\\n\\ \\ \\[borrowed\\ references\\]\\(\\.\\./glossary\\.html\\#term\\-borrowed\\-reference\\)\\ may\\ not\\ be\\ thread\\-safe\\ if\\\\n\\ \\ the\\ containing\\ object\\ is\\ modified\\ concurrently\\.\\ See\\ the\\ section\\ on\\\\n\\ \\ \\[borrowed\\ references\\]\\(\\#id2\\)\\ for\\ more\\ information\\.\\\\n\\\\n\\#\\#\\#\\ Container\\ Thread\\ Safety\\[¶\\]\\(\\#container\\-thread\\-safety\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\\\nContainers\\ like\\ \\[`PyListObject`\\]\\(\\.\\./c\\-api/list\\.html\\#c\\.PyListObject\\ \\\\"PyListObject\\\\"\\),\\\\n\\[`PyDictObject`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDictObject\\ \\\\"PyDictObject\\\\"\\),\\ and\\ \\[`PySetObject`\\]\\(\\.\\./c\\-api/set\\.html\\#c\\.PySetObject\\ \\\\"PySetObject\\\\"\\)\\ perform\\ internal\\ locking\\\\nin\\ the\\ free\\-threaded\\ build\\.\\ For\\ example,\\ the\\ \\[`PyList_Append\\(\\)`\\]\\(\\.\\./c\\-api/list\\.html\\#c\\.PyList_Append\\ \\\\"PyList_Append\\\\"\\)\\ will\\\\nlock\\ the\\ list\\ before\\ appending\\ an\\ item\\.\\\\n\\\\n\\#\\#\\#\\#\\ `PyDict_Next`\\[¶\\]\\(\\#pydict\\-next\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\\\nA\\ notable\\ exception\\ is\\ \\[`PyDict_Next\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_Next\\ \\\\"PyDict_Next\\\\"\\),\\ which\\ does\\ not\\ lock\\ the\\\\ndictionary\\.\\ You\\ should\\ use\\ \\[`Py_BEGIN_CRITICAL_SECTION`\\]\\(\\.\\./c\\-api/synchronization\\.html\\#c\\.Py_BEGIN_CRITICAL_SECTION\\ \\\\"Py_BEGIN_CRITICAL_SECTION\\\\"\\)\\ to\\ protect\\\\nthe\\ dictionary\\ while\\ iterating\\ over\\ it\\ if\\ the\\ dictionary\\ may\\ be\\ concurrently\\\\nmodified:\\\\n\\\\n```\\\\nPy_BEGIN_CRITICAL_SECTION\\(dict\\);\\\\nPyObject\\ \\*key,\\ \\*value;\\\\nPy_ssize_t\\ pos\\ =\\ 0;\\\\nwhile\\ \\(PyDict_Next\\(dict,\\ \\&pos,\\ \\&key,\\ \\&value\\)\\)\\ \\{\\\\n\\ \\ \\ \\ \\.\\.\\.\\\\n\\}\\\\nPy_END_CRITICAL_SECTION\\(\\);\\\\n```\\\\n\\\\nBorrowed\\ References\\[¶\\]\\(\\#borrowed\\-references\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\\\n\\\\nSome\\ C\\ API\\ functions\\ return\\ \\[borrowed\\ references\\]\\(\\.\\./glossary\\.html\\#term\\-borrowed\\-reference\\)\\.\\\\nThese\\ APIs\\ are\\ not\\ thread\\-safe\\ if\\ the\\ containing\\ object\\ is\\ modified\\\\nconcurrently\\.\\ For\\ example,\\ it’s\\ not\\ safe\\ to\\ use\\ \\[`PyList_GetItem\\(\\)`\\]\\(\\.\\./c\\-api/list\\.html\\#c\\.PyList_GetItem\\ \\\\"PyList_GetItem\\\\"\\)\\\\nif\\ the\\ list\\ may\\ be\\ modified\\ concurrently\\.\\\\n\\\\nThe\\ following\\ table\\ lists\\ some\\ borrowed\\ reference\\ APIs\\ and\\ their\\ replacements\\\\nthat\\ return\\ \\[strong\\ references\\]\\(\\.\\./glossary\\.html\\#term\\-strong\\-reference\\)\\.\\\\n\\\\n\\|\\ Borrowed\\ reference\\ API\\ \\|\\ Strong\\ reference\\ API\\ \\|\\\\n\\|\\ \\-\\-\\-\\ \\|\\ \\-\\-\\-\\ \\|\\\\n\\|\\ \\[`PyList_GetItem\\(\\)`\\]\\(\\.\\./c\\-api/list\\.html\\#c\\.PyList_GetItem\\ \\\\"PyList_GetItem\\\\"\\)\\ \\|\\ \\[`PyList_GetItemRef\\(\\)`\\]\\(\\.\\./c\\-api/list\\.html\\#c\\.PyList_GetItemRef\\ \\\\"PyList_GetItemRef\\\\"\\)\\ \\|\\\\n\\|\\ \\[`PyList_GET_ITEM\\(\\)`\\]\\(\\.\\./c\\-api/list\\.html\\#c\\.PyList_GET_ITEM\\ \\\\"PyList_GET_ITEM\\\\"\\)\\ \\|\\ \\[`PyList_GetItemRef\\(\\)`\\]\\(\\.\\./c\\-api/list\\.html\\#c\\.PyList_GetItemRef\\ \\\\"PyList_GetItemRef\\\\"\\)\\ \\|\\\\n\\|\\ \\[`PyDict_GetItem\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_GetItem\\ \\\\"PyDict_GetItem\\\\"\\)\\ \\|\\ \\[`PyDict_GetItemRef\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_GetItemRef\\ \\\\"PyDict_GetItemRef\\\\"\\)\\ \\|\\\\n\\|\\ \\[`PyDict_GetItemWithError\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_GetItemWithError\\ \\\\"PyDict_GetItemWithError\\\\"\\)\\ \\|\\ \\[`PyDict_GetItemRef\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_GetItemRef\\ \\\\"PyDict_GetItemRef\\\\"\\)\\ \\|\\\\n\\|\\ \\[`PyDict_GetItemString\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_GetItemString\\ \\\\"PyDict_GetItemString\\\\"\\)\\ \\|\\ \\[`PyDict_GetItemStringRef\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_GetItemStringRef\\ \\\\"PyDict_GetItemStringRef\\\\"\\)\\ \\|\\\\n\\|\\ \\[`PyDict_SetDefault\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_SetDefault\\ \\\\"PyDict_SetDefault\\\\"\\)\\ \\|\\ \\[`PyDict_SetDefaultRef\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_SetDefaultRef\\ \\\\"PyDict_SetDefaultRef\\\\"\\)\\ \\|\\\\n\\|\\ \\[`PyDict_Next\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_Next\\ \\\\"PyDict_Next\\\\"\\)\\ \\|\\ none\\ \\(see\\ \\[PyDict\\\\\\\\_Next\\]\\(\\#pydict\\-next\\)\\)\\ \\|\\\\n\\|\\ \\[`PyWeakref_GetObject\\(\\)`\\]\\(\\.\\./c\\-api/weakref\\.html\\#c\\.PyWeakref_GetObject\\ \\\\"PyWeakref_GetObject\\\\"\\)\\ \\|\\ \\[`PyWeakref_GetRef\\(\\)`\\]\\(\\.\\./c\\-api/weakref\\.html\\#c\\.PyWeakref_GetRef\\ \\\\"PyWeakref_GetRef\\\\"\\)\\ \\|\\\\n\\|\\ \\[`PyWeakref_GET_OBJECT\\(\\)`\\]\\(\\.\\./c\\-api/weakref\\.html\\#c\\.PyWeakref_GET_OBJECT\\ \\\\"PyWeakref_GET_OBJECT\\\\"\\)\\ \\|\\ \\[`PyWeakref_GetRef\\(\\)`\\]\\(\\.\\./c\\-api/weakref\\.html\\#c\\.PyWeakref_GetRef\\ \\\\"PyWeakref_GetRef\\\\"\\)\\ \\|\\\\n\\|\\ \\[`PyImport_AddModule\\(\\)`\\]\\(\\.\\./c\\-api/import\\.html\\#c\\.PyImport_AddModule\\ \\\\"PyImport_AddModule\\\\"\\)\\ \\|\\ \\[`PyImport_AddModuleRef\\(\\)`\\]\\(\\.\\./c\\-api/import\\.html\\#c\\.PyImport_AddModuleRef\\ \\\\"PyImport_AddModuleRef\\\\"\\)\\ \\|\\\\n\\|\\ \\[`PyCell_GET\\(\\)`\\]\\(\\.\\./c\\-api/cell\\.html\\#c\\.PyCell_GET\\ \\\\"PyCell_GET\\\\"\\)\\ \\|\\ \\[`PyCell_Get\\(\\)`\\]\\(\\.\\./c\\-api/cell\\.html\\#c\\.PyCell_Get\\ \\\\"PyCell_Get\\\\"\\)\\ \\|\\\\n\\\\nNot\\ all\\ APIs\\ that\\ return\\ borrowed\\ references\\ are\\ problematic\\.\\ For\\\\nexample,\\ \\[`PyTuple_GetItem\\(\\)`\\]\\(\\.\\./c\\-api/tuple\\.html\\#c\\.PyTuple_GetItem\\ \\\\"PyTuple_GetItem\\\\"\\)\\ is\\ safe\\ because\\ tuples\\ are\\ immutable\\.\\\\nSimilarly,\\ not\\ all\\ uses\\ of\\ the\\ above\\ APIs\\ are\\ problematic\\.\\ For\\ example,\\\\n\\[`PyDict_GetItem\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_GetItem\\ \\\\"PyDict_GetItem\\\\"\\)\\ is\\ often\\ used\\ for\\ parsing\\ keyword\\ argument\\\\ndictionaries\\ in\\ function\\ calls;\\ those\\ keyword\\ argument\\ dictionaries\\ are\\\\neffectively\\ private\\ \\(not\\ accessible\\ by\\ other\\ threads\\),\\ so\\ using\\ borrowed\\\\nreferences\\ in\\ that\\ context\\ is\\ safe\\.\\\\n\\\\nSome\\ of\\ these\\ functions\\ were\\ added\\ in\\ Python\\ 3\\.13\\.\\ You\\ can\\ use\\ the\\\\n\\[pythoncapi\\-compat\\]\\(https://github\\.com/python/pythoncapi\\-compat\\)\\ package\\\\nto\\ provide\\ implementations\\ of\\ these\\ functions\\ for\\ older\\ Python\\ versions\\.\\\\n\\\\nMemory\\ Allocation\\ APIs\\[¶\\]\\(\\#memory\\-allocation\\-apis\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\\\n\\\\nPython’s\\ memory\\ management\\ C\\ API\\ provides\\ functions\\ in\\ three\\ different\\\\n\\[allocation\\ domains\\]\\(\\.\\./c\\-api/memory\\.html\\#id1\\):\\ “raw”,\\ “mem”,\\ and\\ “object”\\.\\\\nFor\\ thread\\-safety,\\ the\\ free\\-threaded\\ build\\ requires\\ that\\ only\\ Python\\ objects\\\\nare\\ allocated\\ using\\ the\\ object\\ domain,\\ and\\ that\\ all\\ Python\\ objects\\ are\\\\nallocated\\ using\\ that\\ domain\\.\\ This\\ differs\\ from\\ the\\ prior\\ Python\\ versions,\\\\nwhere\\ this\\ was\\ only\\ a\\ best\\ practice\\ and\\ not\\ a\\ hard\\ requirement\\.\\\\n\\\\nNote\\\\n\\\\nSearch\\ for\\ uses\\ of\\ \\[`PyObject_Malloc\\(\\)`\\]\\(\\.\\./c\\-api/memory\\.html\\#c\\.PyObject_Malloc\\ \\\\"PyObject_Malloc\\\\"\\)\\ in\\ your\\\\nextension\\ and\\ check\\ that\\ the\\ allocated\\ memory\\ is\\ used\\ for\\ Python\\ objects\\.\\\\nUse\\ \\[`PyMem_Malloc\\(\\)`\\]\\(\\.\\./c\\-api/memory\\.html\\#c\\.PyMem_Malloc\\ \\\\"PyMem_Malloc\\\\"\\)\\ to\\ allocate\\ buffers\\ instead\\ of\\\\n`PyObject_Malloc\\(\\)`\\.\\\\n\\\\nThread\\ State\\ and\\ GIL\\ APIs\\[¶\\]\\(\\#thread\\-state\\-and\\-gil\\-apis\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\\\n\\\\nPython\\ provides\\ a\\ set\\ of\\ functions\\ and\\ macros\\ to\\ manage\\ thread\\ state\\ and\\ the\\\\nGIL,\\ such\\ as:\\\\n\\\\n\\*\\ \\[`PyGILState_Ensure\\(\\)`\\]\\(\\.\\./c\\-api/threads\\.html\\#c\\.PyGILState_Ensure\\ \\\\"PyGILState_Ensure\\\\"\\)\\ and\\ \\[`PyGILState_Release\\(\\)`\\]\\(\\.\\./c\\-api/threads\\.html\\#c\\.PyGILState_Release\\ \\\\"PyGILState_Release\\\\"\\)\\\\n\\*\\ \\[`PyEval_SaveThread\\(\\)`\\]\\(\\.\\./c\\-api/threads\\.html\\#c\\.PyEval_SaveThread\\ \\\\"PyEval_SaveThread\\\\"\\)\\ and\\ \\[`PyEval_RestoreThread\\(\\)`\\]\\(\\.\\./c\\-api/threads\\.html\\#c\\.PyEval_RestoreThread\\ \\\\"PyEval_RestoreThread\\\\"\\)\\\\n\\*\\ \\[`Py_BEGIN_ALLOW_THREADS`\\]\\(\\.\\./c\\-api/threads\\.html\\#c\\.Py_BEGIN_ALLOW_THREADS\\ \\\\"Py_BEGIN_ALLOW_THREADS\\\\"\\)\\ and\\ \\[`Py_END_ALLOW_THREADS`\\]\\(\\.\\./c\\-api/threads\\.html\\#c\\.Py_END_ALLOW_THREADS\\ \\\\"Py_END_ALLOW_THREADS\\\\"\\)\\\\n\\\\nThese\\ functions\\ should\\ still\\ be\\ used\\ in\\ the\\ free\\-threaded\\ build\\ to\\ manage\\\\nthread\\ state\\ even\\ when\\ the\\ \\[GIL\\]\\(\\.\\./glossary\\.html\\#term\\-GIL\\)\\ is\\ disabled\\.\\ For\\ example,\\ if\\ you\\\\ncreate\\ a\\ thread\\ outside\\ of\\ Python,\\ you\\ must\\ call\\ \\[`PyGILState_Ensure\\(\\)`\\]\\(\\.\\./c\\-api/threads\\.html\\#c\\.PyGILState_Ensure\\ \\\\"PyGILState_Ensure\\\\"\\)\\\\nbefore\\ calling\\ into\\ the\\ Python\\ API\\ to\\ ensure\\ that\\ the\\ thread\\ has\\ a\\ valid\\\\nPython\\ thread\\ state\\.\\\\n\\\\nYou\\ should\\ continue\\ to\\ call\\ \\[`PyEval_SaveThread\\(\\)`\\]\\(\\.\\./c\\-api/threads\\.html\\#c\\.PyEval_SaveThread\\ \\\\"PyEval_SaveThread\\\\"\\)\\ or\\\\n\\[`Py_BEGIN_ALLOW_THREADS`\\]\\(\\.\\./c\\-api/threads\\.html\\#c\\.Py_BEGIN_ALLOW_THREADS\\ \\\\"Py_BEGIN_ALLOW_THREADS\\\\"\\)\\ around\\ blocking\\ operations,\\ such\\ as\\ I/O\\ or\\\\nlock\\ acquisitions,\\ to\\ allow\\ other\\ threads\\ to\\ run\\ the\\\\n\\[cyclic\\ garbage\\ collector\\]\\(\\.\\./glossary\\.html\\#term\\-garbage\\-collection\\)\\.\\\\n\\\\nProtecting\\ Internal\\ Extension\\ State\\[¶\\]\\(\\#protecting\\-internal\\-extension\\-state\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\\\n\\\\nYour\\ extension\\ may\\ have\\ internal\\ state\\ that\\ was\\ previously\\ protected\\ by\\ the\\\\nGIL\\.\\ You\\ may\\ need\\ to\\ add\\ locking\\ to\\ protect\\ this\\ state\\.\\ The\\ approach\\ will\\\\ndepend\\ on\\ your\\ extension,\\ but\\ some\\ common\\ patterns\\ include:\\\\n\\\\n\\*\\ \\*\\*Caches\\*\\*:\\ global\\ caches\\ are\\ a\\ common\\ source\\ of\\ shared\\ state\\.\\ Consider\\\\n\\ \\ using\\ a\\ lock\\ to\\ protect\\ the\\ cache\\ or\\ disabling\\ it\\ in\\ the\\ free\\-threaded\\ build\\\\n\\ \\ if\\ the\\ cache\\ is\\ not\\ critical\\ for\\ performance\\.\\\\n\\*\\ \\*\\*Global\\ State\\*\\*:\\ global\\ state\\ may\\ need\\ to\\ be\\ protected\\ by\\ a\\ lock\\ or\\ moved\\\\n\\ \\ to\\ thread\\ local\\ storage\\.\\ C11\\ and\\ C\\+\\+11\\ provide\\ the\\ `thread_local`\\ or\\\\n\\ \\ `_Thread_local`\\ for\\\\n\\ \\ \\[thread\\-local\\ storage\\]\\(https://en\\.cppreference\\.com/w/c/language/storage_duration\\)\\.\\\\n\\\\nCritical\\ Sections\\[¶\\]\\(\\#critical\\-sections\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\\\n\\\\nIn\\ the\\ free\\-threaded\\ build,\\ CPython\\ provides\\ a\\ mechanism\\ called\\ “critical\\\\nsections”\\ to\\ protect\\ data\\ that\\ would\\ otherwise\\ be\\ protected\\ by\\ the\\ GIL\\.\\\\nWhile\\ extension\\ authors\\ may\\ not\\ interact\\ with\\ the\\ internal\\ critical\\ section\\\\nimplementation\\ directly,\\ understanding\\ their\\ behavior\\ is\\ crucial\\ when\\ using\\\\ncertain\\ C\\ API\\ functions\\ or\\ managing\\ shared\\ state\\ in\\ the\\ free\\-threaded\\ build\\.\\\\n\\\\n\\#\\#\\#\\ What\\ Are\\ Critical\\ Sections\\?\\[¶\\]\\(\\#what\\-are\\-critical\\-sections\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\\\nConceptually,\\ critical\\ sections\\ act\\ as\\ a\\ deadlock\\ avoidance\\ layer\\ built\\ on\\\\ntop\\ of\\ simple\\ mutexes\\.\\ Each\\ thread\\ maintains\\ a\\ stack\\ of\\ active\\ critical\\\\nsections\\.\\ When\\ a\\ thread\\ needs\\ to\\ acquire\\ a\\ lock\\ associated\\ with\\ a\\ critical\\\\nsection\\ \\(e\\.g\\.,\\ implicitly\\ when\\ calling\\ a\\ thread\\-safe\\ C\\ API\\ function\\ like\\\\n\\[`PyDict_SetItem\\(\\)`\\]\\(\\.\\./c\\-api/dict\\.html\\#c\\.PyDict_SetItem\\ \\\\"PyDict_SetItem\\\\"\\),\\ or\\ explicitly\\ using\\ macros\\),\\ it\\ attempts\\ to\\ acquire\\\\nthe\\ underlying\\ mutex\\.\\\\n\\\\n\\#\\#\\#\\ Using\\ Critical\\ Sections\\[¶\\]\\(\\#using\\-critical\\-sections\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\\\nThe\\ primary\\ APIs\\ for\\ using\\ critical\\ sections\\ are:\\\\n\\\\n\\*\\ \\[`Py_BEGIN_CRITICAL_SECTION`\\]\\(\\.\\./c\\-api/synchronization\\.html\\#c\\.Py_BEGIN_CRITICAL_SECTION\\ \\\\"Py_BEGIN_CRITICAL_SECTION\\\\"\\)\\ and\\ \\[`Py_END_CRITICAL_SECTION`\\]\\(\\.\\./c\\-api/synchronization\\.html\\#c\\.Py_END_CRITICAL_SECTION\\ \\\\"Py_END_CRITICAL_SECTION\\\\"\\)\\ \\-\\\\n\\ \\ For\\ locking\\ a\\ single\\ object\\\\n\\*\\ \\[`Py_BEGIN_CRITICAL_SECTION2`\\]\\(\\.\\./c\\-api/synchronization\\.html\\#c\\.Py_BEGIN_CRITICAL_SECTION2\\ \\\\"Py_BEGIN_CRITICAL_SECTION2\\\\"\\)\\ and\\ \\[`Py_END_CRITICAL_SECTION2`\\]\\(\\.\\./c\\-api/synchronization\\.html\\#c\\.Py_END_CRITICAL_SECTION2\\ \\\\"Py_END_CRITICAL_SECTION2\\\\"\\)\\\\n\\ \\ \\-\\ For\\ locking\\ two\\ objects\\ simultaneously\\\\n\\\\nThese\\ macros\\ must\\ be\\ used\\ in\\ matching\\ pairs\\ and\\ must\\ appear\\ in\\ the\\ same\\ C\\\\nscope,\\ since\\ they\\ establish\\ a\\ new\\ local\\ scope\\.\\ These\\ macros\\ are\\ no\\-ops\\ in\\\\nnon\\-free\\-threaded\\ builds,\\ so\\ they\\ can\\ be\\ safely\\ added\\ to\\ code\\ that\\ needs\\ to\\\\nsupport\\ both\\ build\\ types\\.\\\\n\\\\nA\\ common\\ use\\ of\\ a\\ critical\\ section\\ would\\ be\\ to\\ lock\\ an\\ object\\ while\\ accessing\\\\nan\\ internal\\ attribute\\ of\\ it\\.\\ For\\ example,\\ if\\ an\\ extension\\ type\\ has\\ an\\ internal\\\\ncount\\ field,\\ you\\ could\\ use\\ a\\ critical\\ section\\ while\\ reading\\ or\\ writing\\ that\\\\nfield:\\\\n\\\\n```\\\\n//\\ read\\ the\\ count,\\ returns\\ new\\ reference\\ to\\ internal\\ count\\ value\\\\nPyObject\\ \\*result;\\\\nPy_BEGIN_CRITICAL_SECTION\\(obj\\);\\\\nresult\\ =\\ Py_NewRef\\(obj\\->count\\);\\\\nPy_END_CRITICAL_SECTION\\(\\);\\\\nreturn\\ result;\\\\n\\\\n//\\ write\\ the\\ count,\\ consumes\\ reference\\ from\\ new_count\\\\nPy_BEGIN_CRITICAL_SECTION\\(obj\\);\\\\nobj\\->count\\ =\\ new_count;\\\\nPy_END_CRITICAL_SECTION\\(\\);\\\\n```\\\\n\\\\n\\#\\#\\#\\ How\\ Critical\\ Sections\\ Work\\[¶\\]\\(\\#how\\-critical\\-sections\\-work\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\\\nUnlike\\ traditional\\ locks,\\ critical\\ sections\\ do\\ not\\ guarantee\\ exclusive\\ access\\\\nthroughout\\ their\\ entire\\ duration\\.\\ If\\ a\\ thread\\ would\\ block\\ while\\ holding\\ a\\\\ncritical\\ section\\ \\(e\\.g\\.,\\ by\\ acquiring\\ another\\ lock\\ or\\ performing\\ I/O\\),\\ the\\\\ncritical\\ section\\ is\\ temporarily\\ suspended—all\\ locks\\ are\\ released—and\\ then\\\\nresumed\\ when\\ the\\ blocking\\ operation\\ completes\\.\\\\n\\\\nThis\\ behavior\\ is\\ similar\\ to\\ what\\ happens\\ with\\ the\\ GIL\\ when\\ a\\ thread\\ makes\\ a\\\\nblocking\\ call\\.\\ The\\ key\\ differences\\ are:\\\\n\\\\n\\*\\ Critical\\ sections\\ operate\\ on\\ a\\ per\\-object\\ basis\\ rather\\ than\\ globally\\\\n\\*\\ Critical\\ sections\\ follow\\ a\\ stack\\ discipline\\ within\\ each\\ thread\\ \\(the\\ “begin”\\ and\\\\n\\ \\ “end”\\ macros\\ enforce\\ this\\ since\\ they\\ must\\ be\\ paired\\ and\\ within\\ the\\ same\\ scope\\)\\\\n\\*\\ Critical\\ sections\\ automatically\\ release\\ and\\ reacquire\\ locks\\ around\\ potential\\\\n\\ \\ blocking\\ operations\\\\n\\\\n\\#\\#\\#\\ Deadlock\\ Avoidance\\[¶\\]\\(\\#deadlock\\-avoidance\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\\\nCritical\\ sections\\ help\\ avoid\\ deadlocks\\ in\\ two\\ ways:\\\\n\\\\n1\\.\\ If\\ a\\ thread\\ tries\\ to\\ acquire\\ a\\ lock\\ that’s\\ already\\ held\\ by\\ another\\ thread,\\\\n\\ \\ \\ it\\ first\\ suspends\\ all\\ of\\ its\\ active\\ critical\\ sections,\\ temporarily\\ releasing\\\\n\\ \\ \\ their\\ locks\\\\n2\\.\\ When\\ the\\ blocking\\ operation\\ completes,\\ only\\ the\\ top\\-most\\ critical\\ section\\ is\\\\n\\ \\ \\ reacquired\\ first\\\\n\\\\nThis\\ means\\ you\\ cannot\\ rely\\ on\\ nested\\ critical\\ sections\\ to\\ lock\\ multiple\\ objects\\\\nat\\ once,\\ as\\ the\\ inner\\ critical\\ section\\ may\\ suspend\\ the\\ outer\\ ones\\.\\ Instead,\\ use\\\\n\\[`Py_BEGIN_CRITICAL_SECTION2`\\]\\(\\.\\./c\\-api/synchronization\\.html\\#c\\.Py_BEGIN_CRITICAL_SECTION2\\ \\\\"Py_BEGIN_CRITICAL_SECTION2\\\\"\\)\\ to\\ lock\\ two\\ objects\\ simultaneously\\.\\\\n\\\\nNote\\ that\\ the\\ locks\\ described\\ above\\ are\\ only\\ \\[`PyMutex`\\]\\(\\.\\./c\\-api/synchronization\\.html\\#c\\.PyMutex\\ \\\\"PyMutex\\\\"\\)\\ based\\ locks\\.\\\\nThe\\ critical\\ section\\ implementation\\ does\\ not\\ know\\ about\\ or\\ affect\\ other\\ locking\\\\nmechanisms\\ that\\ might\\ be\\ in\\ use,\\ like\\ POSIX\\ mutexes\\.\\ Also\\ note\\ that\\ while\\\\nblocking\\ on\\ any\\ `PyMutex`\\ causes\\ the\\ critical\\ sections\\ to\\ be\\\\nsuspended,\\ only\\ the\\ mutexes\\ that\\ are\\ part\\ of\\ the\\ critical\\ sections\\ are\\\\nreleased\\.\\ If\\ `PyMutex`\\ is\\ used\\ without\\ a\\ critical\\ section,\\ it\\ will\\\\nnot\\ be\\ released\\ and\\ therefore\\ does\\ not\\ get\\ the\\ same\\ deadlock\\ avoidance\\.\\\\n\\\\n\\#\\#\\#\\ Important\\ Considerations\\[¶\\]\\(\\#important\\-considerations\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\\\n\\*\\ Critical\\ sections\\ may\\ temporarily\\ release\\ their\\ locks,\\ allowing\\ other\\ threads\\\\n\\ \\ to\\ modify\\ the\\ protected\\ data\\.\\ Be\\ careful\\ about\\ making\\ assumptions\\ about\\ the\\\\n\\ \\ state\\ of\\ the\\ data\\ after\\ operations\\ that\\ might\\ block\\.\\\\n\\*\\ Because\\ locks\\ can\\ be\\ temporarily\\ released\\ \\(suspended\\),\\ entering\\ a\\ critical\\\\n\\ \\ section\\ does\\ not\\ guarantee\\ exclusive\\ access\\ to\\ the\\ protected\\ resource\\\\n\\ \\ throughout\\ the\\ section’s\\ duration\\.\\ If\\ code\\ within\\ a\\ critical\\ section\\ calls\\\\n\\ \\ another\\ function\\ that\\ blocks\\ \\(e\\.g\\.,\\ acquires\\ another\\ lock,\\ performs\\ blocking\\\\n\\ \\ I/O\\),\\ all\\ locks\\ held\\ by\\ the\\ thread\\ via\\ critical\\ sections\\ will\\ be\\ released\\.\\\\n\\ \\ This\\ is\\ similar\\ to\\ how\\ the\\ GIL\\ can\\ be\\ released\\ during\\ blocking\\ calls\\.\\\\n\\*\\ Only\\ the\\ lock\\(s\\)\\ associated\\ with\\ the\\ most\\ recently\\ entered\\ \\(top\\-most\\)\\\\n\\ \\ critical\\ section\\ are\\ guaranteed\\ to\\ be\\ held\\ at\\ any\\ given\\ time\\.\\ Locks\\ for\\\\n\\ \\ outer,\\ nested\\ critical\\ sections\\ might\\ have\\ been\\ suspended\\.\\\\n\\*\\ You\\ can\\ lock\\ at\\ most\\ two\\ objects\\ simultaneously\\ with\\ these\\ APIs\\.\\ If\\ you\\ need\\\\n\\ \\ to\\ lock\\ more\\ objects,\\ you’ll\\ need\\ to\\ restructure\\ your\\ code\\.\\\\n\\*\\ While\\ critical\\ sections\\ will\\ not\\ deadlock\\ if\\ you\\ attempt\\ to\\ lock\\ the\\ same\\\\n\\ \\ object\\ twice,\\ they\\ are\\ less\\ efficient\\ than\\ purpose\\-built\\ reentrant\\ locks\\ for\\\\n\\ \\ this\\ use\\ case\\.\\\\n\\*\\ When\\ using\\ \\[`Py_BEGIN_CRITICAL_SECTION2`\\]\\(\\.\\./c\\-api/synchronization\\.html\\#c\\.Py_BEGIN_CRITICAL_SECTION2\\ \\\\"Py_BEGIN_CRITICAL_SECTION2\\\\"\\),\\ the\\ order\\ of\\ the\\ objects\\\\n\\ \\ doesn’t\\ affect\\ correctness\\ \\(the\\ implementation\\ handles\\ deadlock\\ avoidance\\),\\\\n\\ \\ but\\ it’s\\ good\\ practice\\ to\\ always\\ lock\\ objects\\ in\\ a\\ consistent\\ order\\.\\\\n\\*\\ Remember\\ that\\ the\\ critical\\ section\\ macros\\ are\\ primarily\\ for\\ protecting\\ access\\\\n\\ \\ to\\ \\*Python\\ objects\\*\\ that\\ might\\ be\\ involved\\ in\\ internal\\ CPython\\ operations\\\\n\\ \\ susceptible\\ to\\ the\\ deadlock\\ scenarios\\ described\\ above\\.\\ For\\ protecting\\ purely\\\\n\\ \\ internal\\ extension\\ state,\\ standard\\ mutexes\\ or\\ other\\ synchronization\\\\n\\ \\ primitives\\ might\\ be\\ more\\ appropriate\\.\\\\n\\\\nBuilding\\ Extensions\\ for\\ the\\ Free\\-Threaded\\ Build\\[¶\\]\\(\\#building\\-extensions\\-for\\-the\\-free\\-threaded\\-build\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\\\n\\\\nC\\ API\\ extensions\\ need\\ to\\ be\\ built\\ specifically\\ for\\ the\\ free\\-threaded\\ build\\.\\\\nThe\\ wheels,\\ shared\\ libraries,\\ and\\ binaries\\ are\\ indicated\\ by\\ a\\ `t`\\ suffix\\.\\\\n\\\\n\\*\\ \\[pypa/manylinux\\]\\(https://github\\.com/pypa/manylinux\\)\\ supports\\ the\\\\n\\ \\ free\\-threaded\\ build,\\ with\\ the\\ `t`\\ suffix,\\ such\\ as\\ `python3\\.14t`\\.\\\\n\\*\\ \\[pypa/cibuildwheel\\]\\(https://github\\.com/pypa/cibuildwheel\\)\\ supports\\\\n\\ \\ building\\ wheels\\ for\\ the\\ free\\-threaded\\ build\\ of\\ Python\\ 3\\.14\\ and\\ newer\\.\\\\n\\\\n\\#\\#\\#\\ Limited\\ C\\ API\\ and\\ Stable\\ ABI\\[¶\\]\\(\\#limited\\-c\\-api\\-and\\-stable\\-abi\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\\\nThe\\ free\\-threaded\\ build\\ does\\ not\\ currently\\ support\\ the\\\\n\\[Limited\\ C\\ API\\]\\(\\.\\./c\\-api/stable\\.html\\#limited\\-c\\-api\\)\\ or\\ the\\ stable\\ ABI\\.\\ If\\ you\\ use\\\\n\\[setuptools\\]\\(https://setuptools\\.pypa\\.io/en/latest/setuptools\\.html\\)\\ to\\ build\\\\nyour\\ extension\\ and\\ currently\\ set\\ `py_limited_api=True`\\ you\\ can\\ use\\\\n`py_limited_api=not\\ sysconfig\\.get_config_var\\(\\\\"Py_GIL_DISABLED\\\\"\\)`\\ to\\ opt\\ out\\\\nof\\ the\\ limited\\ API\\ when\\ building\\ with\\ the\\ free\\-threaded\\ build\\.\\\\n\\\\nNote\\\\n\\\\nYou\\ will\\ need\\ to\\ build\\ separate\\ wheels\\ specifically\\ for\\ the\\ free\\-threaded\\\\nbuild\\.\\ If\\ you\\ currently\\ use\\ the\\ stable\\ ABI,\\ you\\ can\\ continue\\ to\\ build\\ a\\\\nsingle\\ wheel\\ for\\ multiple\\ non\\-free\\-threaded\\ Python\\ versions\\.\\\\n\\\\n\\#\\#\\#\\ Windows\\[¶\\]\\(\\#windows\\ \\\\"Link\\ to\\ this\\ heading\\\\"\\)\\\\n\\\\nDue\\ to\\ a\\ limitation\\ of\\ the\\ official\\ Windows\\ installer,\\ you\\ will\\ need\\ to\\\\nmanually\\ define\\ `Py_GIL_DISABLED=1`\\ when\\ building\\ extensions\\ from\\ source\\.\\\\n\\\\nSee\\ also\\\\n\\\\n\\[Porting\\ Extension\\ Modules\\ to\\ Support\\ Free\\-Threading\\]\\(https://py\\-free\\-threading\\.github\\.io/porting/\\):\\\\nA\\ community\\-maintained\\ porting\\ guide\\ for\\ extension\\ authors\\.\\\\n\\\\n\\#\\#\\#\\ \\[Table\\ of\\ Contents\\]\\(\\.\\./contents\\.html\\)\\\\n\\\\n\\*\\ \\[C\\ API\\ Extension\\ Support\\ for\\ Free\\ Threading\\]\\(\\#\\)\\\\n\\ \\ \\+\\ \\[Identifying\\ the\\ Free\\-Threaded\\ Build\\ in\\ C\\]\\(\\#identifying\\-the\\-free\\-threaded\\-build\\-in\\-c\\)\\\\n\\ \\ \\+\\ \\[Module\\ Initialization\\]\\(\\#module\\-initialization\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Multi\\-Phase\\ Initialization\\]\\(\\#multi\\-phase\\-initialization\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Single\\-Phase\\ Initialization\\]\\(\\#single\\-phase\\-initialization\\)\\\\n\\ \\ \\+\\ \\[General\\ API\\ Guidelines\\]\\(\\#general\\-api\\-guidelines\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Container\\ Thread\\ Safety\\]\\(\\#container\\-thread\\-safety\\)\\\\n\\ \\ \\ \\ \\ \\ \\*\\ \\[`PyDict_Next`\\]\\(\\#pydict\\-next\\)\\\\n\\ \\ \\+\\ \\[Borrowed\\ References\\]\\(\\#borrowed\\-references\\)\\\\n\\ \\ \\+\\ \\[Memory\\ Allocation\\ APIs\\]\\(\\#memory\\-allocation\\-apis\\)\\\\n\\ \\ \\+\\ \\[Thread\\ State\\ and\\ GIL\\ APIs\\]\\(\\#thread\\-state\\-and\\-gil\\-apis\\)\\\\n\\ \\ \\+\\ \\[Protecting\\ Internal\\ Extension\\ State\\]\\(\\#protecting\\-internal\\-extension\\-state\\)\\\\n\\ \\ \\+\\ \\[Critical\\ Sections\\]\\(\\#critical\\-sections\\)\\\\n\\ \\ \\ \\ \\-\\ \\[What\\ Are\\ Critical\\ Sections\\?\\]\\(\\#what\\-are\\-critical\\-sections\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Using\\ Critical\\ Sections\\]\\(\\#using\\-critical\\-sections\\)\\\\n\\ \\ \\ \\ \\-\\ \\[How\\ Critical\\ Sections\\ Work\\]\\(\\#how\\-critical\\-sections\\-work\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Deadlock\\ Avoidance\\]\\(\\#deadlock\\-avoidance\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Important\\ Considerations\\]\\(\\#important\\-considerations\\)\\\\n\\ \\ \\+\\ \\[Building\\ Extensions\\ for\\ the\\ Free\\-Threaded\\ Build\\]\\(\\#building\\-extensions\\-for\\-the\\-free\\-threaded\\-build\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Limited\\ C\\ API\\ and\\ Stable\\ ABI\\]\\(\\#limited\\-c\\-api\\-and\\-stable\\-abi\\)\\\\n\\ \\ \\ \\ \\-\\ \\[Windows\\]\\(\\#windows\\)\\\\n\\\\n\\#\\#\\#\\#\\ Previous\\ topic\\\\n\\\\n\\[Python\\ support\\ for\\ free\\ threading\\]\\(free\\-threading\\-python\\.html\\ \\\\"previous\\ chapter\\\\"\\)\\\\n\\\\n\\#\\#\\#\\#\\ Next\\ topic\\\\n\\\\n\\[Remote\\ debugging\\ attachment\\ protocol\\]\\(remote_debugging\\.html\\ \\\\"next\\ chapter\\\\"\\)\\\\n\\\\ndocument\\.addEventListener\\('DOMContentLoaded',\\ \\(\\)\\ =>\\ \\{\\\\nconst\\ title\\ =\\ document\\.querySelector\\('meta\\[property=\\\\"og:title\\\\"\\]'\\)\\.content;\\\\nconst\\ elements\\ =\\ document\\.querySelectorAll\\('\\.improvepage'\\);\\\\nconst\\ pageurl\\ =\\ window\\.location\\.href\\.split\\('\\?'\\)\\[0\\];\\\\nelements\\.forEach\\(element\\ =>\\ \\{\\\\nconst\\ url\\ =\\ new\\ URL\\(element\\.href\\.split\\('\\?'\\)\\[0\\]\\.replace\\(\\\\"\\-nojs\\\\",\\ \\\\"\\\\"\\)\\);\\\\nurl\\.searchParams\\.set\\('pagetitle',\\ title\\);\\\\nurl\\.searchParams\\.set\\('pageurl',\\ pageurl\\);\\\\nurl\\.searchParams\\.set\\('pagesource',\\ \\\\"howto/free\\-threading\\-extensions\\.rst\\\\"\\);\\\\nelement\\.href\\ =\\ url\\.toString\\(\\);\\\\n\\}\\);\\\\n\\}\\);\\\\n\\\\n\\#\\#\\#\\ This\\ page\\\\n\\\\n\\*\\ \\[Report\\ a\\ bug\\]\\(\\.\\./bugs\\.html\\)\\\\n\\*\\ \\[Improve\\ this\\ page\\]\\(\\.\\./improve\\-page\\-nojs\\.html\\)\\\\n\\*\\ \\[Show\\ source\\]\\(https://github\\.com/python/cpython/blob/main/Doc/howto/free\\-threading\\-extensions\\.rst\\?plain=1\\)\\\\n\\\\n«\\\\n\\\\n\\#\\#\\#\\ Navigation\\\\n\\\\n\\*\\ \\[index\\]\\(\\.\\./genindex\\.html\\ \\\\"General\\ Index\\\\"\\)\\\\n\\*\\ \\[modules\\]\\(\\.\\./py\\-modindex\\.html\\ \\\\"Python\\ Module\\ Index\\\\"\\)\\ \\|\\\\n\\*\\ \\[next\\]\\(remote_debugging\\.html\\ \\\\"Remote\\ debugging\\ attachment\\ protocol\\\\"\\)\\ \\|\\\\n\\*\\ \\[previous\\]\\(free\\-threading\\-python\\.html\\ \\\\"Python\\ support\\ for\\ free\\ threading\\\\"\\)\\ \\|\\\\n\\*\\ \\[Python\\]\\(https://www\\.python\\.org/\\)\\ »\\\\n\\\\n\\*\\ \\[3\\.14\\.7\\ Documentation\\]\\(\\.\\./index\\.html\\)\\ »\\\\n\\*\\ \\[Python\\ HOWTOs\\]\\(index\\.html\\)\\ »\\\\n\\*\\ C\\ API\\ Extension\\ Support\\ for\\ Free\\ Threading\\\\n\\*\\ \\|\\\\n\\*\\ Theme\\\\n\\ \\ Auto\\\\n\\ \\ Light\\\\n\\ \\ Dark\\\\n\\ \\ \\ \\|\\\\n\\\\n©\\ \\[Copyright\\]\\(\\.\\./copyright\\.html\\)\\ 2001\\ Python\\ Software\\ Foundation\\.\\\\n\\ \\ \\\\nThis\\ page\\ is\\ licensed\\ under\\ the\\ Python\\ Software\\ Foundation\\ License\\ Version\\ 2\\.\\\\n\\ \\ \\\\nExamples,\\ recipes,\\ and\\ other\\ code\\ in\\ the\\ documentation\\ are\\ additionally\\ licensed\\ under\\ the\\ Zero\\ Clause\\ BSD\\ License\\.\\\\n\\ \\ \\\\nSee\\ \\[History\\ and\\ License\\]\\(/license\\.html\\)\\ for\\ more\\ information\\.\\ \\ \\\\n\\ \\ \\\\nThe\\ Python\\ Software\\ Foundation\\ is\\ a\\ non\\-profit\\ corporation\\.\\\\n\\[Please\\ donate\\.\\]\\(https://www\\.python\\.org/psf/donations/\\)\\\\n\\ \\ \\\\n\\ \\ \\\\nLast\\ updated\\ on\\ Aug\\ 14,\\ 2026\\ \\(13:12\\ UTC\\)\\.\\\\n\\[Found\\ a\\ bug\\]\\(/bugs\\.html\\)\\?\\\\n\\ \\ \\\\nCreated\\ using\\ \\[Sphinx\\]\\(https://www\\.sphinx\\-doc\\.org/\\)\\ 8\\.2\\.3\\."\\}\
"""
                        ),
                        tool_call_id='call_9QGqlCO9vWmh9yKF1C730LaP',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- researcher: Research a focused sub-question on the web and report back with findings and source links\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
## Summary

**CPython 3.13 introduced free threading as an experimental, separate build. CPython 3.14 moved it to PEP 703 phase II: officially supported, but still optional and still separate from the default GIL-enabled build.** “Supported” in 3.14 means the core design and APIs are considered stable enough for ongoing use; it does **not** mean that free threading is now the default, universally faster, or compatible with every extension package. ([3.13 What’s New](https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython), [3.14 What’s New](https://docs.python.org/3.14/whatsnew/3.14.html), [PEP 779](https://peps.python.org/pep-0779/))

| Area | CPython 3.13 | CPython 3.14 |
|---|---|---|
| Maturity | Experimental, PEP 703 phase I | Officially supported, PEP 703 phase II |
| Default build | No | No |
| Distribution | Separate `python3.13t` build | Separate `python3.14t` build |
| Single-thread overhead | About 40% on `pyperformance` | Roughly 1–8%, depending on platform in current 3.14 documentation |
| Extension ABI | Separate `t` ABI and wheels | Still separate `t` ABI and wheels |
| Stable/Limited ABI | Unsupported for free-threaded extensions | Still unsupported |
| Unsupported extension behavior | May warn and re-enable GIL | Same |
| Broad immortalization | Many object categories | Mostly eliminated; limited to code constants and interned strings |
| Frames and iterators | Serious cross-thread safety limitations | Limitations remain, but frame restriction is narrower |

## Maturity

### CPython 3.13: experimental

The 3.13 release documentation calls free threading **experimental**, says it is not enabled by default, and warns users to expect bugs and a substantial single-threaded performance penalty. It requires a different executable, generally `python3.13t` or `python3.13t.exe`. ([3.13 What’s New](https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython))

This phase was primarily appropriate for experimentation, extension porting, ecosystem compatibility work, and controlled workloads. PEP 703’s acceptance also retained the possibility of rolling back disruptive parts of the design. ([PEP 703](https://peps.python.org/pep-0703/))

### CPython 3.14: supported, but optional

PEP 779 advanced free-threaded CPython to **phase II** in 3.14: officially supported rather than experimental. The PEP says the design is sufficiently finalized, its APIs are usable and stable, and future changes are expected to follow the normal PEP 387 compatibility policy. It also notes that 3.14 added convenience and replacement APIs without breaking the 3.13 free-threading APIs. ([PEP 779](https://peps.python.org/pep-0779/))

Phase II is explicitly distinct from phase III. Free threading remains an optional build and has not replaced ordinary GIL-enabled CPython. A future PEP would be needed to make it the default. ([PEP 779](https://peps.python.org/pep-0779/), [3.14 What’s New](https://docs.python.org/3.14/whatsnew/3.14.html))

**Inference:** 3.14 is reasonable for production adoption when dependencies have been validated and the workload benefits from thread parallelism. It is still not a transparent, ecosystem-wide replacement for the ordinary interpreter.

## Installation and operation

Both versions use the same basic model:

- Official macOS and Windows installers can optionally install free-threaded binaries.
- Source builds use `./configure --disable-gil`.
- The free-threaded executable can still run with the GIL enabled using `PYTHON_GIL` or `-X gil`.
- `sysconfig.get_config_var("Py_GIL_DISABLED") == 1` identifies a free-threading-capable build.
- `sys._is_gil_enabled()` reports whether the GIL is active in the current process.
- Importing an extension that has not declared free-threading support can emit a warning and re-enable the GIL. ([3.13 HOWTO](https://docs.python.org/3.13/howto/free-threading-python.html), [3.14 HOWTO](https://docs.python.org/3.14/howto/free-threading-python.html))

Thus, running `python3.14t` does not guarantee that the process remains GIL-free: a dependency can trigger fallback unless the user explicitly forces the GIL off. Explicitly forcing it off does not make an unsafe extension thread-safe. ([3.14 HOWTO](https://docs.python.org/3.14/howto/free-threading-python.html#the-global-interpreter-lock-in-free-threaded-python))

## Major improvement: single-thread performance

In 3.13, the free-threaded interpreter is documented as approximately **40% slower** on the `pyperformance` suite. The main stated reason is that the specializing adaptive interpreter was disabled in that build. ([3.13 HOWTO](https://docs.python.org/3.13/howto/free-threading-python.html#single-threaded-performance))

The current 3.14 documentation reports average overhead ranging from approximately **1% on macOS AArch64 to 8% on x86-64 Linux**, depending on workload and hardware. ([3.14 HOWTO](https://docs.python.org/3.14/howto/free-threading-python.html#single-threaded-performance))

This is the most substantial practical improvement between the releases. Nevertheless, these are suite averages, not a guarantee for an individual application. Parallel speedup also depends on contention, workload design, native code, and whether dependencies leave the GIL disabled.

## Major improvement: object lifetime and memory behavior

### 3.13

When an additional thread starts, 3.13 immortalizes numerous categories, including:

- module-level functions,
- method descriptors,
- code objects,
- modules and module dictionaries,
- classes,
- numeric and string literals,
- strings interned with `sys.intern()`.

Such objects are never deallocated, so applications that dynamically create many modules, functions, or classes can experience continued memory growth. ([3.13 HOWTO](https://docs.python.org/3.13/howto/free-threading-python.html#immortalization))

### 3.14

In 3.14, immortalization is restricted to:

- code constants, including numeric, string and qualifying tuple literals;
- strings interned through `sys.intern()`.

The broad 3.13 immortalization of modules, classes and functions was therefore removed. ([3.14 HOWTO](https://docs.python.org/3.14/howto/free-threading-python.html#immortalization))

Memory use can still be higher in 3.14 because of larger headers for some objects, mimalloc and its separate heaps, quiescent-state-based memory reclamation, and biased, deferred, or per-thread reference-counting schemes that can delay deallocation. ([3.14 HOWTO](https://docs.python.org/3.14/howto/free-threading-python.html#increased-memory-usage))

## Native-extension limitations

Both releases require extensions to explicitly declare that they support GIL-free execution:

- Multi-phase modules use the `Py_mod_gil` slot with `Py_MOD_GIL_NOT_USED`.
- Single-phase modules use `PyUnstable_Module_SetGIL()`.
- Otherwise, importing the module warns and enables the GIL. ([3.13 extension HOWTO](https://docs.python.org/3.13/howto/free-threading-extensions.html#module-initialization), [3.14 extension HOWTO](https://docs.python.org/3.14/howto/free-threading-extensions.html#module-initialization))

Declaring support is not sufficient by itself. Extensions may need to:

- protect mutable global or module state;
- avoid direct access to concurrently mutable C-API structure fields;
- replace unsafe accessor macros with locking APIs;
- replace borrowed-reference access with strong-reference APIs where concurrent removal is possible;
- protect `PyDict_Next()` iteration with a critical section;
- obey stricter allocator-domain requirements;
- continue using thread-state APIs even though the GIL is disabled. ([3.14 extension HOWTO](https://docs.python.org/3.14/howto/free-threading-extensions.html))

The free-threaded build still has a separate ABI identified by the `t` suffix. Native extensions and wheels must be built specifically for it. The 3.14 free-threaded build still does **not** support the Limited C API or Stable ABI, so projects cannot yet use one `abi3` binary across ordinary and free-threaded CPython builds. ([3.13 extension HOWTO](https://docs.python.org/3.13/howto/free-threading-extensions.html#building-extensions-for-the-free-threaded-build), [3.14 extension HOWTO](https://docs.python.org/3.14/howto/free-threading-extensions.html#building-extensions-for-the-free-threaded-build))

## Python-level concurrency limitations

In both versions, built-in containers such as `dict`, `list`, and `set` use internal locking. However, CPython explicitly describes this as current implementation behavior, not a language guarantee for concurrent mutation. Programs should use `threading.Lock` or other explicit synchronization rather than depending upon incidental container locking. ([3.13 HOWTO](https://docs.python.org/3.13/howto/free-threading-python.html#thread-safety), [3.14 HOWTO](https://docs.python.org/3.14/howto/free-threading-python.html#thread-safety))

Sharing a single iterator concurrently remains unsafe in 3.14: threads may observe duplicate or missing elements. The 3.13 documentation additionally warns that this may crash the interpreter. ([3.13 HOWTO](https://docs.python.org/3.13/howto/free-threading-python.html#iterators), [3.14 HOWTO](https://docs.python.org/3.14/howto/free-threading-python.html#iterators))

Frame handling improved but remains hazardous:

- In 3.13, accessing frame objects from another thread is generally unsafe and may crash, making `sys._current_frames()` generally unsafe.
- In 3.14, the documented restriction is narrower: accessing `frame.f_locals` while that frame is executing in another thread may crash the interpreter. ([3.13 HOWTO](https://docs.python.org/3.13/howto/free-threading-python.html#frame-objects), [3.14 HOWTO](https://docs.python.org/3.14/howto/free-threading-python.html#frame-objects))

## Bottom line

- **3.13:** an experimental technology preview with severe single-thread overhead, broad immortalization, distinct extension requirements, and several known crash-prone edge cases.
- **3.14:** a supported optional configuration with much better single-thread performance, substantially improved object-lifetime behavior, and stabilized APIs.
- **Still unresolved in 3.14:** incomplete package compatibility, GIL fallback caused by unsupported extensions, separate native wheels, no free-threaded Stable ABI, greater memory consumption, unsafe concurrent iterator/frame operations, and the need to redesign or synchronize code rather than assuming existing threaded programs will automatically scale.\
""",
                        id='msg_0e58f8341a296874006a7f8471f45481968e63379073770a48',
                        provider_name='openai',
                        provider_details={'phase': 'final_answer'},
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='gpt-5.6-sol',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_0e58f8341a296874006a7f8470c6c48196a3c2cd136ecc9f8a',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert isinstance(result.output, str)
    assert len(result.output) > 200
    tool_names = {
        part.tool_name
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart | NativeToolCallPart)
    }
    assert 'delegate_task' in tool_names
    assert tool_names & {'web_search', 'web_search_preview'}
    assert tool_names & {'web_fetch', 'open_url', 'read_tool_result'}
