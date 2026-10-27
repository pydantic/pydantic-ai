"""The browser preserves the terminal's lower-bound cost and usage markers."""

from decimal import Decimal

import pytest

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.test import TestModel
from pydantic_clai2.builtin_plugins.web.protocol import PROTOCOL, Notice, SessionsChanged, TurnEnded
from tests.clai2.test_web import TOKEN, MakeWeb, make_web as make_web, receive_until


@pytest.mark.parametrize('unmeasured_requests', [0, 1])
def test_browser_preserves_omitted_usage_in_status_and_reports(make_web: MakeWeb, unmeasured_requests: int) -> None:
    class SetUsage(AbstractCapability[None]):
        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            response.usage.cost = Decimal('0.01')
            response.usage.unmeasured_requests = unmeasured_requests
            return response

    web = make_web(
        Agent(TestModel(call_tools=[], custom_output_text='answer'), deps_type=type(None), capabilities=[SetUsage()])
    )
    with web.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web.run()
        web.post('/api/input', {'text': 'hello'})
        receive_until(socket, TurnEnded)
        receive_until(socket, SessionsChanged)
        status = web.get('/api/state').json()['status']
        assert status['cost'] == '0.01'
        assert status['cost_is_lower_bound'] is bool(unmeasured_requests)
        for command in ('/usage', '/cost'):
            web.post('/api/input', {'text': command})
            report = receive_until(socket, Notice)[1].usage
            assert report is not None
            assert report.total.cost == '$0.0100'
            assert report.unmeasured_requests == unmeasured_requests
            assert report.unpriced == []
        web.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)
