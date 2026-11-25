"""`AskUser` under `DBOSDurability`: the workflow waits for the answer, and the answerer isn't asked again on recovery."""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator

import anyio
import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import AskUser
from pydantic_ai_harness.ask_user import TOOL_NAME, AskUserAnswer, AskUserRequest, AskUserResponse
from tests._timeouts import READINESS_WAIT_TIMEOUT

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability, resolve_external_event
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

pytestmark = pytest.mark.xdist_group(name='harness-dbos')

_QUESTIONS = [{'header': 'Scope', 'question': 'Which files?', 'options': [{'label': 'All'}, {'label': 'Some'}]}]

_tokens: list[str] = []
_handed_on: list[anyio.Event] = []


def _ask_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart(TOOL_NAME, {'questions': _QUESTIONS}, tool_call_id='q-1')])
    returned = next(part for part in messages[-1].parts if isinstance(part, ToolReturnPart))
    return ModelResponse(parts=[TextPart(returned.model_response_str())])


async def _hand_on(ctx: RunContext[None], request: AskUserRequest, token: str) -> AskUserResponse | None:
    _tokens.append(token)
    _handed_on[-1].set()
    return None


ask_user_agent = Agent(
    FunctionModel(_ask_then_answer),
    name='dbos_ask_user',
    deps_type=type(None),
    capabilities=[DBOSDurability[None](), AskUser[None](answerer=_hand_on, timeout=600)],
)


@DBOS.workflow()
async def ask_user_workflow(prompt: str) -> str:
    return (await ask_user_agent.run(prompt)).output


@pytest.fixture(scope='module')
def dbos(tmp_path_factory: pytest.TempPathFactory) -> Generator[DBOS]:
    config: DBOSConfig = {
        'name': 'pydantic_ai_harness_ask_user_dbos',
        'system_database_url': f'sqlite:///{tmp_path_factory.mktemp("dbos") / "dbos.sqlite"}',
        'enable_otlp': False,
    }
    instance = DBOS(config=config)
    DBOS.launch()
    # On SQLite, DBOS polls for messages every second; poll more often so a resolved wait ends right away.
    instance._sys_db._notification_listener_polling_interval_sec = 0.01  # pyright: ignore[reportPrivateUsage]
    try:
        yield instance
    finally:
        DBOS.destroy()


async def test_a_question_waits_for_the_answer_and_its_answerer_is_not_asked_again(dbos: DBOS) -> None:
    _handed_on.append(anyio.Event())
    workflow_id = f'ask-user-{uuid.uuid4()}'
    with SetWorkflowID(workflow_id):
        handle = await DBOS.start_workflow_async(ask_user_workflow, 'tidy up')
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await _handed_on[-1].wait()
    [token] = _tokens

    response = AskUserResponse(answers=(AskUserAnswer(header='Scope', selected=('All',)),))
    await resolve_external_event(token, TypeAdapter(AskUserResponse).dump_python(response, mode='json'))
    assert json.loads(await handle.get_result()) == {'Scope': ['All']}

    # A fork from the last step replays the question's wait from the recorded steps.
    steps = await dbos.list_workflow_steps_async(workflow_id)
    fork = await DBOS.fork_workflow_async(workflow_id, steps[-1]['function_id'])
    assert json.loads(await fork.get_result(polling_interval_sec=0.01)) == {'Scope': ['All']}
    assert _tokens == [token], 'the replay does not hand the question on again'
