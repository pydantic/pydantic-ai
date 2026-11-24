"""What the D1 suites share: the model that makes a case's calls, how a case is built, and how a journal is read.

`tests/harness/test_durable_tool_io.py` runs every case with `calls` in a DBOS workflow, and
`tests/harness/test_durable_temporal_replay.py` in a Temporal one; each checks what the run journaled against the
case's `durable_steps`.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ..append_only import ToolCall
from ._contracts import CASES
from ._counting_clients import REQUESTS

PROMPT = 'go'
"""The prompt a case's run starts with, which tells its requests apart from a nested run's."""

UNPROMPTED_REQUEST = 'model.unprompted_request'
"""What `REQUESTS` counts a request without `PROMPT` under: a nested run's, or one after compaction dropped it."""


def one_call_per_response(made: tuple[ToolCall, ...]) -> FunctionModel:
    """A model that makes the calls in order, one per response, then answers; call `i` gets the ID `call_{i}`.

    It doesn't stream, so a run makes one model request per call and one to answer. A request whose first message
    isn't the run's `PROMPT` gets the answer, and is counted in `REQUESTS`: it is a capability's own run on the
    run's model, such as `BrowserUse`'s browser agent, whose requests recovery must not make again.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if not any(isinstance(part, UserPromptPart) and part.content == PROMPT for part in messages[0].parts):
            REQUESTS[UNPROMPTED_REQUEST] += 1
            return ModelResponse(parts=[TextPart('done')])
        if step >= len(made):
            return ModelResponse(parts=[TextPart('done')])
        name, args = made[step]
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=f'call_{step}')])

    return FunctionModel(respond)


def installed(name: str) -> bool:
    """Whether the optional dependency the case of `name` needs, if any, is installed: the test skips it if not."""
    requires = CASES[name].requires
    return requires is None or importlib.util.find_spec(requires) is not None


def build_case(name: str, directory: Path) -> list[AbstractCapability[None]]:
    """What the calls of the case of `name` run against, built in `directory` with the credentials it checks."""
    case = CASES[name]
    assert case.calls is not None
    with patch.dict(os.environ, case.env):
        built = case.calls.build(directory)
    return [built] if isinstance(built, AbstractCapability) else list(built)


def journal_breaks(
    recorded: Sequence[str], *, prefix: str, model_request: str, calls: int, expected: tuple[str, ...], engine: str
) -> list[str]:
    """How the durable units a run recorded differ from what its case's `durable_steps` lists for `engine`.

    `recorded` holds the units' names, each starting with `prefix`. The run's own model requests are left out of
    the comparison, but there must be one per call and one to answer.
    """
    units = [unit.removeprefix(prefix) for unit in recorded]
    requests = units.count(model_request)
    journaled = tuple(unit for unit in units if unit != model_request)
    counted = (
        []
        if requests == calls + 1
        else [f'the run recorded {requests} `{model_request}` units, not one per call and one to answer']
    )
    drifted = (
        []
        if journaled == expected
        else [
            f'the calls record `{engine}={journaled!r}`, not {expected!r}. A recorded name is what a workflow from an '
            'earlier release replays by: change the case only with a compatibility note '
            '(agent_docs/capability-conformance.md, "Durable execution")'
        ]
    )
    return counted + drifted
