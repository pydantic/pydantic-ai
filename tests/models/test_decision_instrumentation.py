"""The `decide` spans a decision model emits beneath its `chat` span, against the live TypeSafe API.

One `decide` span per Decisions request, so the snapshots show what was asked and answered on each: a turn
that picks a route and then fills it is two of them under one `chat`.
"""

from __future__ import annotations as _annotations

from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from pydantic_ai import Agent, ModelHTTPError, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.tools import ToolDefinition

from .._inline_snapshot import snapshot
from ..conftest import IsStr, try_import

with try_import() as imports_successful:
    from logfire.testing import CaptureLogfire

    from pydantic_ai.models.typesafe import TypeSafeModel
    from pydantic_ai.providers.typesafe import TypeSafeProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='typesafe-sdk or logfire not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

# Added to every span by Logfire rather than by the decision model: its own bookkeeping, and the run's baggage,
# whose ids are random.
_LOGFIRE_ATTRIBUTES = ('logfire.msg', 'logfire.span_type', 'gen_ai.agent.call.id', 'gen_ai.conversation.id')

# The response attributes that show the `chat` span's usage is the sum of its `decide` spans'.
_CHAT_ATTRIBUTES = ('gen_ai.response.model', 'gen_ai.usage.input_tokens', 'gen_ai.usage.output_tokens')


def span_tree(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    """The exported spans as a tree of names, with every `decide` span's attributes and a `chat` span's usage.

    JSON attributes are parsed, so a snapshot reads as the payload a trace backend receives.
    """
    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    nodes: dict[int, dict[str, Any]] = {}
    roots: list[dict[str, Any]] = []
    for span in sorted(spans, key=lambda span: span['start_time']):
        attributes: dict[str, Any] = span['attributes']
        node: dict[str, Any] = {'name': span['name']}
        operation = attributes.get('gen_ai.operation.name')
        if operation == 'decide':
            node['attributes'] = {key: value for key, value in attributes.items() if key not in _LOGFIRE_ATTRIBUTES}
            if events := span.get('events'):
                node['events'] = [{'name': event['name'], **event['attributes']} for event in events]
        elif operation == 'chat':
            node['attributes'] = {key: attributes[key] for key in _CHAT_ATTRIBUTES if key in attributes}
        nodes[span['context']['span_id']] = node
        if parent := span['parent']:
            nodes[parent['span_id']].setdefault('children', []).append(node)
        else:
            roots.append(node)
    return roots


@pytest.fixture
def jev(typesafe_api_key: str) -> TypeSafeModel:
    return TypeSafeModel('jev-latest', provider=TypeSafeProvider(api_key=typesafe_api_key))


class TicketPriority(BaseModel):
    """Triage an incoming support ticket."""

    priority: Literal['low', 'normal', 'urgent'] = Field(description='How quickly does this ticket need a response?')


async def test_one_field(allow_model_requests: None, jev: TypeSafeModel, capfire: CaptureLogfire):
    """A single field is one Decisions request: one `decide` span, whose usage is all of the `chat` span's."""
    agent = Agent(jev, output_type=TicketPriority, capabilities=[Instrumentation()], name='triage')
    result = await agent.run(
        'Our whole team has been locked out of the dashboard since this morning, and we have a client demo in an hour.'
    )
    assert result.output == TicketPriority(priority='urgent')
    assert span_tree(capfire) == snapshot(
        [
            {
                'name': 'invoke_agent triage',
                'children': [
                    {
                        'name': 'chat jev-latest',
                        'attributes': {
                            'gen_ai.response.model': 'jev-1.13.0',
                            'gen_ai.usage.input_tokens': 349,
                            'gen_ai.usage.output_tokens': 38,
                        },
                        'children': [
                            {
                                'name': 'decide jev-latest',
                                'attributes': {
                                    'gen_ai.operation.name': 'decide',
                                    'gen_ai.provider.name': 'typesafe',
                                    'gen_ai.system': 'typesafe',
                                    'server.address': 'api.typesafe.ai',
                                    'gen_ai.request.model': 'jev-latest',
                                    'pydantic_ai.decision.questions': {
                                        'priority': {
                                            'type': 'choice',
                                            'criteria': {'low': None, 'normal': None, 'urgent': None},
                                            'instructions': {
                                                'field': 'priority',
                                                'question': 'How quickly does this ticket need a response?',
                                                'goal': 'Triage an incoming support ticket.',
                                            },
                                        }
                                    },
                                    'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
                                    'pydantic_ai.decision.route': 'final_result',
                                    'pydantic_ai.decision.state': 'Our whole team has been locked out of the dashboard since this morning, and we have a client demo in an hour.',
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
                                            'pydantic_ai.decision.confidence': {'type': 'object'},
                                            'pydantic_ai.decision.answers': {'type': 'object'},
                                        },
                                    },
                                    'gen_ai.agent.name': 'triage',
                                    'gen_ai.response.model': 'jev-1.13.0',
                                    'pydantic_ai.decision.usage.input_tokens': 349,
                                    'pydantic_ai.decision.usage.output_tokens': 38,
                                    'gen_ai.response.id': 'req_01a0d06d1a6a76848732fb45d58fea6f',
                                    'pydantic_ai.decision.answers': {
                                        'priority': {
                                            'type': 'choice',
                                            'choice': 'urgent',
                                            'confidence': 1.0,
                                            'probabilities': {'low': 0.0, 'urgent': 1.0, 'normal': 0.0},
                                        }
                                    },
                                    'pydantic_ai.decision.confidence': {'priority': 1.0},
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    )


class Refund(BaseModel):
    """Refund the customer for a charge."""

    reason: Literal['duplicate_charge', 'service_outage', 'cancelled_order'] = Field(
        description='Why is the refund due?'
    )
    full_refund: bool = Field(description='Should the whole charge be refunded?')


class Escalation(BaseModel):
    """Hand the ticket to a specialist team."""

    team: Literal['billing', 'security', 'engineering'] = Field(description='Which team should handle it?')


async def test_union_route_and_fill(allow_model_requests: None, jev: TypeSafeModel, capfire: CaptureLogfire):
    """A union is picked, then filled: two `decide` spans under one `chat`, the second naming the route it fills."""
    agent = Agent(jev, output_type=[Refund, Escalation], capabilities=[Instrumentation()], name='support')
    result = await agent.run('I was charged twice for my March subscription. Please put the second charge back.')
    assert result.output == Refund(reason='duplicate_charge', full_refund=True)
    assert span_tree(capfire) == snapshot(
        [
            {
                'name': 'invoke_agent support',
                'children': [
                    {
                        'name': 'chat jev-latest',
                        'attributes': {
                            'gen_ai.response.model': 'jev-1.13.0',
                            'gen_ai.usage.input_tokens': 756,
                            'gen_ai.usage.output_tokens': 109,
                        },
                        'children': [
                            {
                                'name': 'decide jev-latest',
                                'attributes': {
                                    'gen_ai.operation.name': 'decide',
                                    'gen_ai.provider.name': 'typesafe',
                                    'gen_ai.system': 'typesafe',
                                    'server.address': 'api.typesafe.ai',
                                    'gen_ai.request.model': 'jev-latest',
                                    'pydantic_ai.decision.questions': {
                                        'tool': {
                                            'type': 'choice',
                                            'criteria': {
                                                'final_result_Refund': 'Refund the customer for a charge.',
                                                'final_result_Escalation': 'Hand the ticket to a specialist team.',
                                            },
                                            'instructions': 'Which of these does this call for?',
                                        }
                                    },
                                    'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
                                    'pydantic_ai.decision.route_question': 'tool',
                                    'pydantic_ai.decision.route_options': [
                                        'final_result_Refund',
                                        'final_result_Escalation',
                                    ],
                                    'pydantic_ai.decision.state': 'I was charged twice for my March subscription. Please put the second charge back.',
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
                                            'pydantic_ai.decision.route_options': {'type': 'array'},
                                            'pydantic_ai.decision.answers': {'type': 'object'},
                                        },
                                    },
                                    'gen_ai.agent.name': 'support',
                                    'gen_ai.response.model': 'jev-1.13.0',
                                    'pydantic_ai.decision.usage.input_tokens': 347,
                                    'pydantic_ai.decision.usage.output_tokens': 44,
                                    'gen_ai.response.id': 'req_01a0d06d1c17732ba445a7c0c48c9d7b',
                                    'pydantic_ai.decision.answers': {
                                        'tool': {
                                            'type': 'choice',
                                            'choice': 'final_result_Refund',
                                            'confidence': 1.0,
                                            'probabilities': {
                                                'final_result_Escalation': 0.0,
                                                'final_result_Refund': 1.0,
                                            },
                                        }
                                    },
                                    'pydantic_ai.decision.route_taken': 'final_result_Refund',
                                    'pydantic_ai.decision.route_reason': 'selected',
                                },
                            },
                            {
                                'name': 'decide jev-latest',
                                'attributes': {
                                    'gen_ai.operation.name': 'decide',
                                    'gen_ai.provider.name': 'typesafe',
                                    'gen_ai.system': 'typesafe',
                                    'server.address': 'api.typesafe.ai',
                                    'gen_ai.request.model': 'jev-latest',
                                    'pydantic_ai.decision.questions': {
                                        'reason': {
                                            'type': 'choice',
                                            'criteria': {
                                                'duplicate_charge': None,
                                                'service_outage': None,
                                                'cancelled_order': None,
                                            },
                                            'instructions': {
                                                'field': 'reason',
                                                'question': 'Why is the refund due?',
                                                'chosen': 'Refund',
                                                'goal': 'Refund the customer for a charge.',
                                            },
                                        },
                                        'full_refund': {
                                            'type': 'noul',
                                            'instructions': {
                                                'field': 'full_refund',
                                                'question': 'Should the whole charge be refunded?',
                                                'chosen': 'Refund',
                                                'goal': 'Refund the customer for a charge.',
                                            },
                                        },
                                    },
                                    'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
                                    'pydantic_ai.decision.route': 'final_result_Refund',
                                    'pydantic_ai.decision.state': 'I was charged twice for my March subscription. Please put the second charge back.',
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
                                            'pydantic_ai.decision.confidence': {'type': 'object'},
                                            'pydantic_ai.decision.answers': {'type': 'object'},
                                        },
                                    },
                                    'gen_ai.agent.name': 'support',
                                    'gen_ai.response.model': 'jev-1.13.0',
                                    'pydantic_ai.decision.usage.input_tokens': 409,
                                    'pydantic_ai.decision.usage.output_tokens': 65,
                                    'gen_ai.response.id': 'req_01a0d06d1cac7619a7c02643587517ea',
                                    'pydantic_ai.decision.answers': {
                                        'reason': {
                                            'type': 'choice',
                                            'choice': 'duplicate_charge',
                                            'confidence': 1.0,
                                            'probabilities': {
                                                'duplicate_charge': 1.0,
                                                'service_outage': 0.0,
                                                'cancelled_order': 0.0,
                                            },
                                        },
                                        'full_refund': {'type': 'noul', 'noul': 0.56},
                                    },
                                    'pydantic_ai.decision.confidence': {
                                        'reason': 1.0,
                                        'full_refund': 0.1200000000000001,
                                    },
                                },
                            },
                        ],
                    }
                ],
            }
        ]
    )


class Moderation(BaseModel):
    """Moderate a comment before it is published."""

    abusive: bool = Field(description='Does the comment insult or threaten someone?')
    action: Literal['publish', 'hold', 'remove'] = Field(description='What should happen to the comment?')


async def test_without_content(allow_model_requests: None, jev: TypeSafeModel, capfire: CaptureLogfire):
    """Without `include_content`, the state and answers are left out, and each question is recorded by type only."""
    agent = Agent(
        jev,
        output_type=Moderation,
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_content=False))],
        name='moderation',
    )
    result = await agent.run('Great write-up, thanks. The second chart finally made the latency spike click for me.')
    assert result.output == Moderation(abusive=False, action='publish')
    assert span_tree(capfire) == snapshot(
        [
            {
                'name': 'invoke_agent moderation',
                'children': [
                    {
                        'name': 'chat jev-latest',
                        'attributes': {
                            'gen_ai.response.model': 'jev-1.13.0',
                            'gen_ai.usage.input_tokens': 389,
                            'gen_ai.usage.output_tokens': 55,
                        },
                        'children': [
                            {
                                'name': 'decide jev-latest',
                                'attributes': {
                                    'gen_ai.operation.name': 'decide',
                                    'gen_ai.provider.name': 'typesafe',
                                    'gen_ai.system': 'typesafe',
                                    'server.address': 'api.typesafe.ai',
                                    'gen_ai.request.model': 'jev-latest',
                                    'pydantic_ai.decision.questions': {
                                        'abusive': {'type': 'noul'},
                                        'action': {'type': 'choice'},
                                    },
                                    'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
                                    'pydantic_ai.decision.route': 'final_result',
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
                                            'pydantic_ai.decision.confidence': {'type': 'object'},
                                            'pydantic_ai.decision.answers': {'type': 'object'},
                                        },
                                    },
                                    'gen_ai.agent.name': 'moderation',
                                    'gen_ai.response.model': 'jev-1.13.0',
                                    'pydantic_ai.decision.usage.input_tokens': 389,
                                    'pydantic_ai.decision.usage.output_tokens': 55,
                                    'gen_ai.response.id': 'req_01a0d06d1e447e9c9680f758b1884a95',
                                    'pydantic_ai.decision.answers': {
                                        'abusive': {'type': 'noul', 'noul': 0.01},
                                        'action': {'type': 'choice', 'confidence': 1.0},
                                    },
                                    'pydantic_ai.decision.confidence': {'abusive': 0.98, 'action': 1.0},
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    )


async def test_streamed_run(allow_model_requests: None, jev: TypeSafeModel, capfire: CaptureLogfire):
    """A streamed run makes the same request, and records the same `decide` span under its `chat` span."""
    agent = Agent(jev, output_type=TicketPriority, capabilities=[Instrumentation()], name='triage')
    async with agent.run_stream('Could you update the billing address on my account when you get a chance?') as result:
        output = await result.get_output()
    assert output == TicketPriority(priority='normal')
    assert span_tree(capfire) == snapshot(
        [
            {
                'name': 'invoke_agent triage',
                'children': [
                    {
                        'name': 'chat jev-latest',
                        'attributes': {
                            'gen_ai.response.model': 'jev-1.13.0',
                            'gen_ai.usage.input_tokens': 340,
                            'gen_ai.usage.output_tokens': 38,
                        },
                        'children': [
                            {
                                'name': 'decide jev-latest',
                                'attributes': {
                                    'gen_ai.operation.name': 'decide',
                                    'gen_ai.provider.name': 'typesafe',
                                    'gen_ai.system': 'typesafe',
                                    'server.address': 'api.typesafe.ai',
                                    'gen_ai.request.model': 'jev-latest',
                                    'pydantic_ai.decision.questions': {
                                        'priority': {
                                            'type': 'choice',
                                            'criteria': {'low': None, 'normal': None, 'urgent': None},
                                            'instructions': {
                                                'field': 'priority',
                                                'question': 'How quickly does this ticket need a response?',
                                                'goal': 'Triage an incoming support ticket.',
                                            },
                                        }
                                    },
                                    'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
                                    'pydantic_ai.decision.route': 'final_result',
                                    'pydantic_ai.decision.state': 'Could you update the billing address on my account when you get a chance?',
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
                                            'pydantic_ai.decision.confidence': {'type': 'object'},
                                            'pydantic_ai.decision.answers': {'type': 'object'},
                                        },
                                    },
                                    'gen_ai.agent.name': 'triage',
                                    'gen_ai.response.model': 'jev-1.13.0',
                                    'pydantic_ai.decision.usage.input_tokens': 340,
                                    'pydantic_ai.decision.usage.output_tokens': 38,
                                    'gen_ai.response.id': 'req_01a0d06d1faa7b7c91b00b759eeb9bdd',
                                    'pydantic_ai.decision.answers': {
                                        'priority': {
                                            'type': 'choice',
                                            'choice': 'normal',
                                            'confidence': 0.32,
                                            'probabilities': {'low': 0.45, 'normal': 0.55, 'urgent': 0.0},
                                        }
                                    },
                                    'pydantic_ai.decision.confidence': {'priority': 0.32},
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    )


async def test_api_error(allow_model_requests: None, typesafe_api_key: str, capfire: CaptureLogfire):
    """An error from the API is recorded on the `decide` span it happened in, as the `chat` span records it."""
    model = TypeSafeModel('jev-does-not-exist', provider=TypeSafeProvider(api_key=typesafe_api_key))
    agent = Agent(model, output_type=TicketPriority, capabilities=[Instrumentation()], name='triage')
    with pytest.raises(ModelHTTPError):
        await agent.run('The export button does nothing when I click it.')
    assert span_tree(capfire) == snapshot(
        [
            {
                'name': 'invoke_agent triage',
                'children': [
                    {
                        'name': 'chat jev-does-not-exist',
                        'attributes': {'gen_ai.response.model': 'jev-does-not-exist'},
                        'children': [
                            {
                                'name': 'decide jev-does-not-exist',
                                'attributes': {
                                    'gen_ai.operation.name': 'decide',
                                    'gen_ai.provider.name': 'typesafe',
                                    'gen_ai.system': 'typesafe',
                                    'server.address': 'api.typesafe.ai',
                                    'gen_ai.request.model': 'jev-does-not-exist',
                                    'pydantic_ai.decision.questions': {
                                        'priority': {
                                            'type': 'choice',
                                            'criteria': {'low': None, 'normal': None, 'urgent': None},
                                            'instructions': {
                                                'field': 'priority',
                                                'question': 'How quickly does this ticket need a response?',
                                                'goal': 'Triage an incoming support ticket.',
                                            },
                                        }
                                    },
                                    'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
                                    'pydantic_ai.decision.route': 'final_result',
                                    'pydantic_ai.decision.state': 'The export button does nothing when I click it.',
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
                                            'pydantic_ai.decision.confidence': {'type': 'object'},
                                            'pydantic_ai.decision.answers': {'type': 'object'},
                                        },
                                    },
                                    'gen_ai.agent.name': 'triage',
                                    'logfire.level_num': 17,
                                    'gen_ai.response.model': 'jev-does-not-exist',
                                },
                                'events': [
                                    {
                                        'name': 'exception',
                                        'exception.type': 'pydantic_ai.exceptions.ModelHTTPError',
                                        'exception.message': "status_code: 400, model_name: jev-does-not-exist, body: {'detail': {'error_type': 'api_usage_error', 'message': 'Unknown model: jev-does-not-exist'}}",
                                        'exception.stacktrace': "pydantic_ai.exceptions.ModelHTTPError: status_code: 400, model_name: jev-does-not-exist, body: {'detail': {'error_type': 'api_usage_error', 'message': 'Unknown model: jev-does-not-exist'}}",
                                        'exception.escaped': 'False',
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    )


def approve_refund() -> str:
    """Approve the refund as requested."""
    return 'Refund approved.'  # pragma: no cover - offered, but the history leaves only the other tool to call


def reject_refund() -> str:
    """Turn the refund request down."""
    return 'Refund rejected.'


async def test_forced_route_sends_nothing(allow_model_requests: None, jev: TypeSafeModel, capfire: CaptureLogfire):
    """With one route left and nothing to fill, nothing is sent, so the `chat` span has no `decide` span under it."""
    history = [
        ModelRequest(parts=[UserPromptPart('Please refund the duplicate charge on order 1042.')]),
        ModelResponse(parts=[ToolCallPart('approve_refund', {}, 'call_1')]),
        ModelRequest(parts=[ToolReturnPart('approve_refund', 'Refund approved.', 'call_1')]),
    ]
    agent = Agent(
        jev, output_type=[reject_refund], tools=[approve_refund], capabilities=[Instrumentation()], name='refunds'
    )
    result = await agent.run(message_history=history)
    assert result.output == 'Refund rejected.'
    assert span_tree(capfire) == snapshot(
        [
            {
                'name': 'invoke_agent refunds',
                'children': [
                    {'name': 'chat jev-latest', 'attributes': {'gen_ai.response.model': 'jev-latest'}},
                    {'name': 'execute_tool final_result'},
                ],
            }
        ]
    )


def send_sign_in_link(channel: Literal['email', 'sms']) -> str:
    """Send the customer a one-time link to sign in with."""
    return f'Sign-in link sent by {channel}.'


async def test_a_tool_wins_over_the_output_asked_beside_it(
    allow_model_requests: None, jev: TypeSafeModel, capfire: CaptureLogfire
):
    """With one output type and a tool, the output's fields are asked beside the route question, speculatively.

    When the tool wins, `route_taken` differs from the span's `route`, which says those answers were discarded;
    the tool's arguments are filled in a sibling `decide` span, and the output is filled on the next step, once
    the tool has returned and is no longer on offer.
    """
    agent = Agent(
        jev, output_type=TicketPriority, tools=[send_sign_in_link], capabilities=[Instrumentation()], name='support'
    )
    result = await agent.run("I can't sign in. Can you text me a sign-in link? My email is not working.")
    assert result.output == TicketPriority(priority='urgent')
    assert span_tree(capfire) == snapshot(
        [
            {
                'name': 'invoke_agent support',
                'children': [
                    {
                        'name': 'chat jev-latest',
                        'attributes': {
                            'gen_ai.response.model': 'jev-1.13.0',
                            'gen_ai.usage.input_tokens': 756,
                            'gen_ai.usage.output_tokens': 104,
                        },
                        'children': [
                            {
                                'name': 'decide jev-latest',
                                'attributes': {
                                    'gen_ai.operation.name': 'decide',
                                    'gen_ai.provider.name': 'typesafe',
                                    'gen_ai.system': 'typesafe',
                                    'server.address': 'api.typesafe.ai',
                                    'gen_ai.request.model': 'jev-latest',
                                    'pydantic_ai.decision.questions': {
                                        'priority': {
                                            'type': 'choice',
                                            'criteria': {'low': None, 'normal': None, 'urgent': None},
                                            'instructions': {
                                                'field': 'priority',
                                                'question': 'How quickly does this ticket need a response?',
                                                'goal': 'Triage an incoming support ticket.',
                                            },
                                        },
                                        'tool': {
                                            'type': 'choice',
                                            'criteria': {
                                                'final_result': 'Triage an incoming support ticket.',
                                                'send_sign_in_link': 'Send the customer a one-time link to sign in with.',
                                            },
                                            'instructions': 'Which of these does this call for?',
                                        },
                                    },
                                    'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
                                    'pydantic_ai.decision.route': 'final_result',
                                    'pydantic_ai.decision.route_question': 'tool',
                                    'pydantic_ai.decision.route_options': ['final_result', 'send_sign_in_link'],
                                    'pydantic_ai.decision.state': "I can't sign in. Can you text me a sign-in link? My email is not working.",
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
                                            'pydantic_ai.decision.confidence': {'type': 'object'},
                                            'pydantic_ai.decision.route_options': {'type': 'array'},
                                            'pydantic_ai.decision.answers': {'type': 'object'},
                                        },
                                    },
                                    'gen_ai.agent.name': 'support',
                                    'gen_ai.response.model': 'jev-1.13.0',
                                    'pydantic_ai.decision.usage.input_tokens': 416,
                                    'pydantic_ai.decision.usage.output_tokens': 73,
                                    'gen_ai.response.id': 'req_01a0d55501497a479430a06d216259a5',
                                    'pydantic_ai.decision.answers': {
                                        'priority': {
                                            'type': 'choice',
                                            'choice': 'urgent',
                                            'confidence': 0.62,
                                            'probabilities': {'low': 0.01, 'normal': 0.25, 'urgent': 0.74},
                                        },
                                        'tool': {
                                            'type': 'choice',
                                            'choice': 'send_sign_in_link',
                                            'confidence': 0.5,
                                            'probabilities': {'final_result': 0.25, 'send_sign_in_link': 0.75},
                                        },
                                    },
                                    'pydantic_ai.decision.confidence': {'priority': 0.62},
                                    'pydantic_ai.decision.route_taken': 'send_sign_in_link',
                                    'pydantic_ai.decision.route_reason': 'selected',
                                },
                            },
                            {
                                'name': 'decide jev-latest',
                                'attributes': {
                                    'gen_ai.operation.name': 'decide',
                                    'gen_ai.provider.name': 'typesafe',
                                    'gen_ai.system': 'typesafe',
                                    'server.address': 'api.typesafe.ai',
                                    'gen_ai.request.model': 'jev-latest',
                                    'pydantic_ai.decision.questions': {
                                        'channel': {
                                            'type': 'choice',
                                            'criteria': {'email': None, 'sms': None},
                                            'instructions': {
                                                'field': 'channel',
                                                'chosen': 'send_sign_in_link',
                                                'goal': 'Send the customer a one-time link to sign in with.',
                                            },
                                        }
                                    },
                                    'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
                                    'pydantic_ai.decision.route': 'send_sign_in_link',
                                    'pydantic_ai.decision.state': "I can't sign in. Can you text me a sign-in link? My email is not working.",
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
                                            'pydantic_ai.decision.confidence': {'type': 'object'},
                                            'pydantic_ai.decision.answers': {'type': 'object'},
                                        },
                                    },
                                    'gen_ai.agent.name': 'support',
                                    'gen_ai.response.model': 'jev-1.13.0',
                                    'pydantic_ai.decision.usage.input_tokens': 340,
                                    'pydantic_ai.decision.usage.output_tokens': 31,
                                    'gen_ai.response.id': 'req_01a0d55501df7f7c941c5ecb1b387e3a',
                                    'pydantic_ai.decision.answers': {
                                        'channel': {
                                            'type': 'choice',
                                            'choice': 'sms',
                                            'confidence': 0.94,
                                            'probabilities': {'email': 0.03, 'sms': 0.97},
                                        }
                                    },
                                    'pydantic_ai.decision.confidence': {'channel': 0.94},
                                },
                            },
                        ],
                    },
                    {'name': 'execute_tool send_sign_in_link'},
                    {
                        'name': 'chat jev-latest',
                        'attributes': {
                            'gen_ai.response.model': 'jev-1.13.0',
                            'gen_ai.usage.input_tokens': 428,
                            'gen_ai.usage.output_tokens': 38,
                        },
                        'children': [
                            {
                                'name': 'decide jev-latest',
                                'attributes': {
                                    'gen_ai.operation.name': 'decide',
                                    'gen_ai.provider.name': 'typesafe',
                                    'gen_ai.system': 'typesafe',
                                    'server.address': 'api.typesafe.ai',
                                    'gen_ai.request.model': 'jev-latest',
                                    'pydantic_ai.decision.questions': {
                                        'priority': {
                                            'type': 'choice',
                                            'criteria': {'low': None, 'normal': None, 'urgent': None},
                                            'instructions': {
                                                'field': 'priority',
                                                'question': 'How quickly does this ticket need a response?',
                                                'goal': 'Triage an incoming support ticket.',
                                            },
                                        }
                                    },
                                    'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
                                    'pydantic_ai.decision.route': 'final_result',
                                    'pydantic_ai.decision.state': {
                                        'history': [
                                            {
                                                'user': "I can't sign in. Can you text me a sign-in link? My email is not working."
                                            },
                                            {'tool_call': {'name': 'send_sign_in_link', 'args': {'channel': 'sms'}}},
                                            {
                                                'tool_return': {
                                                    'name': 'send_sign_in_link',
                                                    'content': 'Sign-in link sent by sms.',
                                                }
                                            },
                                        ]
                                    },
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
                                            'pydantic_ai.decision.confidence': {'type': 'object'},
                                            'pydantic_ai.decision.state': {'type': 'object'},
                                            'pydantic_ai.decision.answers': {'type': 'object'},
                                        },
                                    },
                                    'gen_ai.agent.name': 'support',
                                    'gen_ai.response.model': 'jev-1.13.0',
                                    'pydantic_ai.decision.usage.input_tokens': 428,
                                    'pydantic_ai.decision.usage.output_tokens': 38,
                                    'gen_ai.response.id': 'req_01a0d55502617893b15f0c13201c0f71',
                                    'pydantic_ai.decision.answers': {
                                        'priority': {
                                            'type': 'choice',
                                            'choice': 'urgent',
                                            'confidence': 0.44,
                                            'probabilities': {'low': 0.01, 'normal': 0.36, 'urgent': 0.63},
                                        }
                                    },
                                    'pydantic_ai.decision.confidence': {'priority': 0.44},
                                },
                            }
                        ],
                    },
                ],
            }
        ]
    )


class Reply(BaseModel):
    """Write back to the customer."""

    message: str = Field(description='What should the reply say?')


def write_reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """The language model behind Jev, which gets the whole step Jev hands off."""
    return ModelResponse(
        parts=[ToolCallPart('final_result_Reply', {'message': 'Sorry for the wait, we are on it.'})],
        model_name='writer',
    )


async def test_a_route_jev_cannot_fill_is_handed_off(
    allow_model_requests: None, jev: TypeSafeModel, capfire: CaptureLogfire
):
    """A picked route whose fields Jev cannot express is handed off before any fill request is sent.

    The route span says so with `route_reason`, and the step goes to the model behind Jev.
    """
    agent = Agent(
        FallbackModel(jev, FunctionModel(write_reply, model_name='writer')),
        output_type=[Escalation, Reply],
        capabilities=[Instrumentation()],
        name='support',
    )
    result = await agent.run('Please just write back to the customer and apologise that we are running late.')
    assert result.output == Reply(message='Sorry for the wait, we are on it.')
    assert span_tree(capfire) == snapshot(
        [
            {
                'name': 'invoke_agent support',
                'children': [
                    {
                        'name': 'chat writer',
                        'attributes': {
                            'gen_ai.response.model': 'writer',
                            'gen_ai.usage.input_tokens': 65,
                            'gen_ai.usage.output_tokens': 12,
                        },
                        'children': [
                            {
                                'name': 'decide jev-latest',
                                'attributes': {
                                    'gen_ai.operation.name': 'decide',
                                    'gen_ai.provider.name': 'typesafe',
                                    'gen_ai.system': 'typesafe',
                                    'server.address': 'api.typesafe.ai',
                                    'gen_ai.request.model': 'jev-latest',
                                    'pydantic_ai.decision.questions': {
                                        'tool': {
                                            'type': 'choice',
                                            'criteria': {
                                                'final_result_Escalation': 'Hand the ticket to a specialist team.',
                                                'final_result_Reply': 'Write back to the customer.',
                                            },
                                            'instructions': 'Which of these does this call for?',
                                        }
                                    },
                                    'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
                                    'pydantic_ai.decision.route_question': 'tool',
                                    'pydantic_ai.decision.route_options': [
                                        'final_result_Escalation',
                                        'final_result_Reply',
                                    ],
                                    'pydantic_ai.decision.state': 'Please just write back to the customer and apologise that we are running late.',
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
                                            'pydantic_ai.decision.route_options': {'type': 'array'},
                                            'pydantic_ai.decision.answers': {'type': 'object'},
                                        },
                                    },
                                    'gen_ai.agent.name': 'support',
                                    'gen_ai.response.model': 'jev-1.13.0',
                                    'pydantic_ai.decision.usage.input_tokens': 343,
                                    'pydantic_ai.decision.usage.output_tokens': 42,
                                    'gen_ai.response.id': 'req_01a0d552422b7f139dc7dc30bb37c3c7',
                                    'pydantic_ai.decision.answers': {
                                        'tool': {
                                            'type': 'choice',
                                            'choice': 'final_result_Reply',
                                            'confidence': 1.0,
                                            'probabilities': {
                                                'final_result_Reply': 1.0,
                                                'final_result_Escalation': 0.0,
                                            },
                                        }
                                    },
                                    'pydantic_ai.decision.route_taken': 'final_result_Reply',
                                    'pydantic_ai.decision.route_reason': 'handed_off',
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    )


async def test_the_last_route_left_is_filled_without_a_route_question(
    allow_model_requests: None, jev: TypeSafeModel, capfire: CaptureLogfire
):
    """With one route left, it is taken without a route question, and its fill span says it was `forced`."""
    lookup_order = ToolDefinition(name='lookup_order', description='Look up the order the customer means.')
    issue_refund = ToolDefinition(
        name='issue_refund',
        description='Refund the customer for the order.',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'reason': {
                    'type': 'string',
                    'enum': ['duplicate_charge', 'service_outage', 'cancelled_order'],
                    'description': 'Why is the refund due?',
                }
            },
            'required': ['reason'],
        },
    )
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('I was charged twice for order 1042. Please refund the second charge.')]),
        ModelResponse(parts=[ToolCallPart('lookup_order', {}, 'call_1')]),
        ModelRequest(parts=[ToolReturnPart('lookup_order', 'Order 1042: charged twice on March 3.', 'call_1')]),
    ]
    response = await model_request(
        jev,
        messages,
        model_request_parameters=ModelRequestParameters(
            function_tools=[lookup_order, issue_refund], allow_text_output=False
        ),
        instrument=True,
    )
    assert response.parts == [ToolCallPart('issue_refund', {'reason': 'duplicate_charge'}, tool_call_id=IsStr())]
    assert span_tree(capfire) == snapshot(
        [
            {
                'name': 'chat jev-latest',
                'attributes': {
                    'gen_ai.response.model': 'jev-1.13.0',
                    'gen_ai.usage.input_tokens': 436,
                    'gen_ai.usage.output_tokens': 46,
                },
                'children': [
                    {
                        'name': 'decide jev-latest',
                        'attributes': {
                            'gen_ai.operation.name': 'decide',
                            'gen_ai.provider.name': 'typesafe',
                            'gen_ai.system': 'typesafe',
                            'server.address': 'api.typesafe.ai',
                            'gen_ai.request.model': 'jev-latest',
                            'pydantic_ai.decision.questions': {
                                'reason': {
                                    'type': 'choice',
                                    'criteria': {
                                        'duplicate_charge': None,
                                        'service_outage': None,
                                        'cancelled_order': None,
                                    },
                                    'instructions': {
                                        'field': 'reason',
                                        'question': 'Why is the refund due?',
                                        'chosen': 'issue_refund',
                                        'goal': 'Refund the customer for the order.',
                                    },
                                }
                            },
                            'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
                            'pydantic_ai.decision.route': 'issue_refund',
                            'pydantic_ai.decision.route_reason': 'forced',
                            'pydantic_ai.decision.state': {
                                'history': [
                                    {'user': 'I was charged twice for order 1042. Please refund the second charge.'},
                                    {'tool_call': {'name': 'lookup_order', 'args': {}}},
                                    {
                                        'tool_return': {
                                            'name': 'lookup_order',
                                            'content': 'Order 1042: charged twice on March 3.',
                                        }
                                    },
                                ]
                            },
                            'logfire.json_schema': {
                                'type': 'object',
                                'properties': {
                                    'pydantic_ai.decision.questions': {'type': 'object'},
                                    'pydantic_ai.decision.thresholds': {'type': 'object'},
                                    'pydantic_ai.decision.confidence': {'type': 'object'},
                                    'pydantic_ai.decision.state': {'type': 'object'},
                                    'pydantic_ai.decision.answers': {'type': 'object'},
                                },
                            },
                            'gen_ai.response.model': 'jev-1.13.0',
                            'pydantic_ai.decision.usage.input_tokens': 436,
                            'pydantic_ai.decision.usage.output_tokens': 46,
                            'gen_ai.response.id': 'req_01a0d552434c77ceb6711d6cb7b7a16f',
                            'pydantic_ai.decision.answers': {
                                'reason': {
                                    'type': 'choice',
                                    'choice': 'duplicate_charge',
                                    'confidence': 1.0,
                                    'probabilities': {
                                        'service_outage': 0.0,
                                        'duplicate_charge': 1.0,
                                        'cancelled_order': 0.0,
                                    },
                                }
                            },
                            'pydantic_ai.decision.confidence': {'reason': 1.0},
                        },
                    }
                ],
            }
        ]
    )
