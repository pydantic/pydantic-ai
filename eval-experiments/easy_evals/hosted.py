"""Logfire-hosted datasets & managed-variable rubrics for easy_evals.

Grounded in the real Logfire API (logfire >= 4.35):

- Hosted datasets (experimental):
    from logfire.experimental.api_client import LogfireAPIClient
    client.push_dataset(pyd_dataset, name=..., description=...)
    client.get_dataset(name, input_type=..., output_type=...) -> pydantic_evals.Dataset
  https://pydantic.dev/docs/logfire/evaluate/datasets/sdk/

- Managed variables ("managed prompts") -- a judge rubric your team edits in the
  Logfire UI, with immutable versions and movable labels (production/canary):
    logfire.var(name, default=...).get(label='production').value
  https://pydantic.dev/docs/logfire/manage/managed-variables/

For real use:  pip install 'logfire[datasets]'  and set LOGFIRE_TOKEN, then call
`logfire.configure()` once so experiment results stream to the Evals UI.

If the client / token isn't available this module transparently falls back to an
in-process simulation with IDENTICAL call sites, so demos run offline.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from typing_extensions import TypeVar

from pydantic_evals import Dataset

from .core import EvalSuite

OutputT = TypeVar('OutputT', default=str)


def live() -> bool:
    """True if we can talk to real Logfire (token set + experimental client available)."""
    if not os.environ.get('LOGFIRE_TOKEN'):
        return False
    return importlib.util.find_spec('logfire.experimental.api_client') is not None


_MODE = 'LIVE Logfire' if live() else 'offline simulation'

# In-process fallback store. Datasets here are dynamically typed, mirroring the real
# `LogfireAPIClient.get_dataset(...) -> Dataset[Any, Any, Any] | dict`.
_DATASETS: dict[str, Dataset[Any, Any, Any]] = {}
_RUBRICS: dict[str, str] = {
    'prompt__haiku_rubric': 'is a haiku: three short lines, evoking a single image',
    'prompt__concise_and_correct': 'is concise (one or two sentences) and factually correct',
}


def push(
    suite_or_dataset: EvalSuite[OutputT] | Dataset[str, OutputT, None], name: str, *, description: str | None = None
) -> str:
    """Push a suite's cases to Logfire as a hosted dataset. Returns the dataset URL."""
    dataset = suite_or_dataset.to_dataset() if isinstance(suite_or_dataset, EvalSuite) else suite_or_dataset
    if live():
        from logfire.experimental.api_client import LogfireAPIClient

        with LogfireAPIClient() as client:
            client.push_dataset(dataset, name=name, description=description)
    else:
        _DATASETS[name] = dataset
        print(f'[{_MODE}] would push {len(dataset.cases)} cases to Logfire dataset {name!r}')
    return dataset_url(name)


def pull(name: str, *, output_type: type[OutputT] = str) -> Dataset[str, OutputT, None]:
    """Fetch a hosted dataset by name as a real pydantic_evals.Dataset (ready to evaluate)."""
    if live():
        from logfire.experimental.api_client import LogfireAPIClient

        with LogfireAPIClient() as client:
            result = client.get_dataset(name, input_type=str, output_type=output_type, metadata_type=type(None))
        if not isinstance(result, Dataset):
            raise TypeError(f'Logfire returned a non-dataset for {name!r}')
        return result
    if name not in _DATASETS:
        raise KeyError(f'no hosted dataset {name!r} (would 404 from Logfire)')
    print(f'[{_MODE}] fetched dataset {name!r} ({len(_DATASETS[name].cases)} cases)')
    return _DATASETS[name]


def hosted_rubric(name: str, *, label: str | None = None, default: str = '') -> str:
    """A judge rubric stored as a Logfire managed variable -- editable in the UI, no deploy.

    Use as: suite.case(..., judge=hosted_rubric('prompt__haiku_rubric', label='production')).
    """
    if live():
        import logfire

        resolved = logfire.var(name, default=default).get(label=label).value
        return resolved if isinstance(resolved, str) else default
    rubric = _RUBRICS.get(name, default)
    suffix = f' @ {label}' if label else ''
    print(f'[{_MODE}] resolved managed rubric {name!r}{suffix}: {rubric!r}')
    return rubric


def dataset_url(name: str) -> str:
    project = os.environ.get('LOGFIRE_PROJECT', '_')
    return f'https://logfire.pydantic.dev/{project}/evals/datasets/{name}'


def experiment_url(experiment: str) -> str:
    project = os.environ.get('LOGFIRE_PROJECT', '_')
    return f'https://logfire.pydantic.dev/{project}/evals/experiments/{experiment}'
