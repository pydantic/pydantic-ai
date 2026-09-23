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
from pydantic_ai.models.instrumented import InstrumentationSettings

from .._inline_snapshot import snapshot
from ..conftest import try_import

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
                                    'pydantic_ai.decision.state': 'Our whole team has been locked out of the dashboard since this morning, and we have a client demo in an hour.',
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
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
                                    'pydantic_ai.decision.state': 'I was charged twice for my March subscription. Please put the second charge back.',
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
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
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
                                        },
                                    },
                                    'gen_ai.agent.name': 'moderation',
                                    'gen_ai.response.model': 'jev-1.13.0',
                                    'pydantic_ai.decision.usage.input_tokens': 389,
                                    'pydantic_ai.decision.usage.output_tokens': 55,
                                    'gen_ai.response.id': 'req_01a0d06d1e447e9c9680f758b1884a95',
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
                                    'pydantic_ai.decision.state': 'Could you update the billing address on my account when you get a chance?',
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
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
                                    'pydantic_ai.decision.state': 'The export button does nothing when I click it.',
                                    'logfire.json_schema': {
                                        'type': 'object',
                                        'properties': {
                                            'pydantic_ai.decision.questions': {'type': 'object'},
                                            'pydantic_ai.decision.thresholds': {'type': 'object'},
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
