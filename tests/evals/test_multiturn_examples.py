from __future__ import annotations

import pytest
from pydantic_ai_examples.evals import example_05_multiturn, example_06_multiturn_veterinary

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import HasMatchingSpan

from ..conftest import try_import

with try_import() as logfire_import_successful:
    from logfire.testing import CaptureLogfire

pytestmark = pytest.mark.anyio
needs_logfire = pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed')


def _structured_response(info: AgentInfo, value: dict[str, object]) -> ModelResponse:
    assert info.output_tools is not None
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, value)])


async def test_minimal_example_uses_public_conversation_task() -> None:
    def target_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse(parts=[TextPart(content='The animal makes its characteristic sound.')])

    def simulator_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        return _structured_response(info, {'message': '', 'done': True})

    conversation = example_05_multiturn.build_conversation_task(
        target_model=FunctionModel(target_model),
        simulator_model=FunctionModel(simulator_model),
    )
    report = await example_05_multiturn.build_dataset(include_llm_judge=False).evaluate(
        conversation.run, progress=False
    )

    assert len(report.cases) == 2
    assert all(case.output.stop_reason == 'simulator_finished' for case in report.cases)
    assert all(case.output.turn_count == 1 for case in report.cases)


async def test_veterinary_regex_target_reuses_patch_evaluator() -> None:
    def simulator_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        return _structured_response(info, {'message': '', 'done': True})

    with example_06_multiturn_veterinary.temporary_record_files() as records:
        record_paths = tuple(records.values())
        assert all(path.exists() for path in record_paths)
        conversation = example_06_multiturn_veterinary.build_regex_conversation_task(
            simulator_model=FunctionModel(simulator_model)
        )
        report = await example_06_multiturn_veterinary.build_dataset(
            records=records, include_tool_evaluators=False
        ).evaluate(
            conversation.run,
            progress=False,
        )

    assert all(not path.exists() for path in record_paths)
    assert len(report.cases) == 2
    for case in report.cases:
        assert case.assertions['species_patch'].value is True
        assert case.assertions['urgency_patch'].value is True
        assert 'Patient record:' not in case.output.turns[0].user_message


@needs_logfire
async def test_veterinary_agent_evaluates_patch_and_target_tools(capfire: CaptureLogfire) -> None:
    assert capfire
    Agent.instrument_all()

    def target_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            prompt = str(messages)
            return ModelResponse(
                parts=[
                    ToolCallPart('classify_species', {'description': prompt}, tool_call_id='species'),
                    ToolCallPart('classify_urgency', {'description': prompt}, tool_call_id='urgency'),
                ]
            )

        values = {part.tool_name: part.content for part in tool_returns}
        return _structured_response(
            info,
            {
                'message': 'The patient has been routed.',
                'patch': {
                    'species': values['classify_species'],
                    'urgency': values['classify_urgency'],
                },
            },
        )

    def simulator_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        return _structured_response(info, {'message': '', 'done': True})

    conversation = example_06_multiturn_veterinary.build_agent_conversation_task(
        target_model=FunctionModel(target_model), simulator_model=FunctionModel(simulator_model)
    )
    with example_06_multiturn_veterinary.temporary_record_files() as records:
        dataset = example_06_multiturn_veterinary.build_dataset(records=records, include_tool_evaluators=True)

        assert sum(isinstance(evaluator, HasMatchingSpan) for evaluator in dataset.evaluators) == 2

        report = await dataset.evaluate(conversation.run, max_concurrency=1, progress=False)

    for case in report.cases:
        assert case.assertions['species_patch'].value is True
        assert case.assertions['urgency_patch'].value is True
        assert case.assertions['used_classify_species'].value is True
        assert case.assertions['used_classify_urgency'].value is True
