from pathlib import Path
from types import NoneType

import logfire
from pydantic_evals import Dataset

from pydantic_ai_examples.evals import infer_time_range
from pydantic_ai_examples.evals.agent import time_range_agent
from pydantic_ai_examples.evals.custom_evaluators import (
    CUSTOM_EVALUATOR_TYPES,
)
from pydantic_ai_examples.evals.models import (
    TimeRangeInputs,
    TimeRangeResponse,
)

logfire.configure(
    send_to_logfire='if-token-present',
    environment='development',
    service_name='evals',
)
logfire.instrument_pydantic_ai()


def compare_models():
    dataset_path = Path(__file__).parent / 'datasets' / 'time_range_v2.yaml'
    dataset = Dataset[TimeRangeInputs, TimeRangeResponse, NoneType].from_file(
        dataset_path, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES
    )
    with logfire.span('Comparing different models for time_range_agent'):
        with time_range_agent.override(model='openai:gpt-4o'):
            dataset.evaluate_sync(infer_time_range, name='openai:gpt-4o')
        with time_range_agent.override(model='openai:o1'):
            dataset.evaluate_sync(infer_time_range, name='openai:o1')


if __name__ == '__main__':
    compare_models()
