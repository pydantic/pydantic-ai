from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class Case:
    id: str
    target_module: str
    blocked_module: str
    error_expression: str
    expected_message: str
    expected_extra: str | None = None


CASES = [
    Case(
        id='mistral-model-missing-sdk',
        target_module='pydantic_ai.models.mistral',
        blocked_module='mistralai',
        error_expression="ModuleNotFoundError('blocked dependency: mistralai', name='mistralai')",
        expected_message='Please install the `mistral` optional group to use the Mistral model',
        expected_extra='mistral',
    ),
    Case(
        id='mistral-provider-missing-sdk',
        target_module='pydantic_ai.providers.mistral',
        blocked_module='mistralai',
        error_expression="ModuleNotFoundError('blocked dependency: mistralai', name='mistralai')",
        expected_message='Please install the `mistral` optional group to use the Mistral provider',
        expected_extra='mistral',
    ),
    Case(
        id='mistral-name-import',
        target_module='pydantic_ai.providers.mistral',
        blocked_module='mistralai',
        error_expression='ImportError("cannot import name \'Mistral\'")',
        expected_message="cannot import name 'Mistral'",
    ),
    Case(
        id='openai-provider-missing-sdk',
        target_module='pydantic_ai.providers.alibaba',
        blocked_module='openai',
        error_expression="ModuleNotFoundError('blocked dependency: openai', name='openai')",
        expected_message='Please install the `openai` optional group to use the Alibaba provider',
        expected_extra='openai',
    ),
    Case(
        id='moonshotai-provider-missing-sdk',
        target_module='pydantic_ai.providers.moonshotai',
        blocked_module='openai',
        error_expression="ModuleNotFoundError('blocked dependency: openai', name='openai')",
        expected_message='Please install the `openai` optional group to use the MoonshotAI provider',
        expected_extra='openai',
    ),
    Case(
        id='anthropic-model-before-provider',
        target_module='pydantic_ai.models.anthropic',
        blocked_module='anthropic',
        error_expression="ModuleNotFoundError('blocked dependency: anthropic', name='anthropic')",
        expected_message='Please install the `anthropic` optional group to use the Anthropic model',
        expected_extra='anthropic',
    ),
    Case(
        id='openrouter-model-before-provider',
        target_module='pydantic_ai.models.openrouter',
        blocked_module='openai',
        error_expression="ModuleNotFoundError('blocked dependency: openai', name='openai')",
        expected_message='Please install the `openrouter` optional group to use the OpenRouter model',
        expected_extra='openrouter',
    ),
    Case(
        id='google-cloud-missing-auth',
        target_module='pydantic_ai.providers.google_cloud',
        blocked_module='google.auth',
        error_expression="ModuleNotFoundError('blocked dependency: google.auth', name='google.auth')",
        expected_message='Please install the `google` optional group to use the Google Cloud provider',
        expected_extra='google',
    ),
    Case(
        id='xai-missing-grpc',
        target_module='pydantic_ai.models.xai',
        blocked_module='grpc',
        error_expression="ModuleNotFoundError('blocked dependency: grpc', name='grpc')",
        expected_message='Please install the `xai` optional group to use the xAI model',
        expected_extra='xai',
    ),
    Case(
        id='bedrock-missing-botocore',
        target_module='pydantic_ai.providers.bedrock',
        blocked_module='botocore',
        error_expression="ModuleNotFoundError('blocked dependency: botocore', name='botocore')",
        expected_message='Please install the `bedrock` optional group to use the Bedrock provider',
        expected_extra='bedrock',
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
def test_optional_import_error(case: Case) -> None:
    """Optional import behavior runs before requests, so a subprocess isolates each module import."""
    code = f"""
import importlib.abc
import importlib.util
import sys

class BlockImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {case.blocked_module!r}:
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise {case.error_expression}

sys.meta_path.insert(0, BlockImport())
import {case.target_module}
"""

    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)

    assert result.returncode == 1
    assert case.expected_message in result.stderr
    if case.expected_extra is None:
        assert 'pip install "pydantic-ai-slim[' not in result.stderr
    else:
        assert f'pip install "pydantic-ai-slim[{case.expected_extra}]"' in result.stderr
