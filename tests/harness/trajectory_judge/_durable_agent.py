"""A judged agent shared by the durable-engine tests: the judge steers once, then approves."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo
from pydantic_ai_harness.trajectory_judge import JudgeSteeringPart, TrajectoryJudge

judge_calls: list[str] = []


def judge_respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    prompt = str(messages[-1])
    judge_calls.append(prompt)
    if 'steered' in prompt:
        return ModelResponse(parts=[ToolCallPart('final_result_AllGood', {})])
    return ModelResponse(parts=[ToolCallPart('final_result_Steer', {'message': 'Cite the file you read.'})])


def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) > 1:
        return ModelResponse(parts=[TextPart('notes.txt says hi.')])
    return ModelResponse(parts=[ToolCallPart('work', {})])


def work() -> str:
    return 'hi'


def trajectory_judge(model: Model) -> TrajectoryJudge[None]:
    return TrajectoryJudge(model=model, id='judge', every=1)


def steering(messages: list[ModelMessage]) -> list[str]:
    """The steering notes delivered to the run, each naming its frame's random `ref`."""
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, JudgeSteeringPart)
    ]
