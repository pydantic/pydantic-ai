from typing import Annotated

from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


class Ticket(BaseModel):
    """Triage a ticket."""

    urgent: bool


class Escalation(BaseModel):
    to: str


def test_regular_types_preserve_tool_names_and_descriptions():
    agent = Agent('test', output_type=[Ticket, Escalation])
    model = TestModel()
    agent.run_sync('Triage this ticket.', model=model)

    parameters = model.last_model_request_parameters
    assert parameters is not None
    assert parameters.output_tools[0].name == 'final_result_Ticket'
    assert parameters.output_tools[0].description == 'Triage a ticket.'
    assert parameters.output_tools[1].name == 'final_result_Escalation'
    assert parameters.output_tools[1].description == 'Escalation: The final response which ends this conversation'


def test_tool_name_extracted_from_annotated_type():
    agent = Agent('test', output_type=[Annotated[Ticket, Field()], Escalation])  # pyright: ignore[reportCallIssue, reportArgumentType]
    model = TestModel()
    agent.run_sync('Triage this ticket.', model=model)

    parameters = model.last_model_request_parameters
    assert parameters is not None
    assert parameters.output_tools[0].name == 'final_result_Ticket'


def test_tool_description_extracted_from_annotated_type():
    agent = Agent('test', output_type=[Annotated[Ticket, Field(description='annotated desc')], Escalation])  # pyright: ignore[reportCallIssue, reportArgumentType]
    model = TestModel()
    agent.run_sync('Triage this ticket.', model=model)

    parameters = model.last_model_request_parameters
    assert parameters is not None
    assert parameters.output_tools[0].description == 'annotated desc'


def test_original_tool_description_used_if_none_provided_in_annotated_type():
    agent = Agent('test', output_type=[Annotated[Ticket, Field()], Escalation])  # pyright: ignore[reportCallIssue, reportArgumentType]
    model = TestModel()
    agent.run_sync('Triage this ticket.', model=model)

    parameters = model.last_model_request_parameters
    assert parameters is not None
    assert parameters.output_tools[0].description == 'Triage a ticket.'


def test_mixed_annotated_and_regular_types():
    agent = Agent('test', output_type=[Annotated[Ticket, Field(description='annotated desc')], Escalation])  # pyright: ignore[reportCallIssue, reportArgumentType]
    model = TestModel()
    agent.run_sync('Triage this ticket.', model=model)

    parameters = model.last_model_request_parameters
    assert parameters is not None
    assert parameters.output_tools[0].name == 'final_result_Ticket'
    assert parameters.output_tools[0].description == 'annotated desc'
    assert parameters.output_tools[1].name == 'final_result_Escalation'
    assert parameters.output_tools[1].description == 'Escalation: The final response which ends this conversation'


def test_two_annotated_types():
    agent = Agent(  # pyright: ignore[reportCallIssue]
        'test',
        output_type=[  # pyright: ignore[reportArgumentType]
            Annotated[Ticket, Field(description='annotated desc')],
            Annotated[Escalation, Field(description='annotated escalation desc')],
        ],
    )
    model = TestModel()
    agent.run_sync('Triage this ticket.', model=model)

    parameters = model.last_model_request_parameters
    assert parameters is not None
    assert parameters.output_tools[0].name == 'final_result_Ticket'
    assert parameters.output_tools[0].description == 'annotated desc'
    assert parameters.output_tools[1].name == 'final_result_Escalation'
    assert parameters.output_tools[1].description == 'annotated escalation desc'
