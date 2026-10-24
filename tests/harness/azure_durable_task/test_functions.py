from __future__ import annotations

import asyncio
import inspect
import re
import runpy
import sys
from collections.abc import Awaitable, Callable, Coroutine, Generator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import InstanceOf, TypeAdapter

from pydantic_ai import Agent, UserError
from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    import azure.durable_functions as df
    import azure.functions as func
    from azure.functions.decorators.function_app import FunctionBuilder
    from durabletask import task

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability
    from pydantic_ai_harness.azure_durable_task.functions import register_activities

    from .examples import python_examples
    from .worker import InMemoryWorker, completed

pytestmark = pytest.mark.skipif(not imports_successful(), reason='Azure Functions preview SDK not installed')

_SAMPLE = Path(__file__).parents[3] / 'src' / 'pydantic_ai_harness' / 'examples' / 'azure_functions' / 'function_app.py'


@dataclass
class _Example:
    app: InstanceOf[df.DFApp]
    support: Callable[[task.OrchestrationContext, str], Generator[task.Task[object], object, str]]
    start: Callable[[func.TimerRequest, df.DurableFunctionsClient], Coroutine[object, object, None]]


@cache
def _sample() -> _Example:
    namespace = runpy.run_path(str(_SAMPLE))
    builder = namespace['start']
    assert isinstance(builder, FunctionBuilder)
    namespace['start'] = inspect.unwrap(builder.build().get_user_function())
    return TypeAdapter(_Example).validate_python(namespace)


@dataclass
class _Binding:
    type: str
    activity: str | None = None


@dataclass
class _Bindings:
    bindings: list[_Binding]


@cache
def _functions(app: df.DFApp) -> list[func.Function]:
    return app.get_functions()


def _activities(app: df.DFApp) -> dict[str, Callable[..., Awaitable[object]]]:
    return {
        binding.activity: function.get_user_function()
        for function in _functions(app)
        for binding in TypeAdapter(_Bindings).validate_json(function.get_function_json()).bindings
        if binding.activity is not None and (function.get_function_name() or '').startswith('pydantic_ai_')
    }


def _install_functions(worker: InMemoryWorker, app: df.DFApp) -> None:
    def install(name: str, execute: Callable[..., Awaitable[object]]) -> None:
        def activity(context: task.ActivityContext, payload: object) -> object:
            async def invoke() -> object:
                return await execute(payload=payload)

            return asyncio.run(invoke())

        activity.__name__ = name
        worker.add_activity(activity)

    for name, execute in _activities(app).items():
        install(name, execute)


class TestAzureDurableTaskFunctions:
    def test_missing_optional_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, 'azure.durable_functions', None)
        with pytest.raises(ImportError, match=r'pydantic-ai-harness\[azure-functions\]'):
            runpy.run_path(register_activities.__code__.co_filename)

    def test_documented_functions_app(self, durable_worker: InMemoryWorker) -> None:
        source = next(source for source in python_examples() if 'azure_durable_task.functions import' in source)
        namespace: dict[str, object] = {'__name__': __name__}
        exec(compile(source, '<documented-function-app>', 'exec', dont_inherit=True), namespace)
        app = namespace['app']
        support = namespace['support']
        assert isinstance(app, df.DFApp)
        assert callable(support)
        _install_functions(durable_worker, app)
        durable_worker.add_orchestrator(support)
        instance_id = durable_worker.schedule('support', input='Hello')
        assert completed(durable_worker, instance_id) == 'Hosted durably'

    def test_registration_preserves_scheduler_names(self) -> None:
        app = df.DFApp()
        for name in ('first/<agent>', 'second' * 25):
            agent = Agent(TestModel(), name=name, capabilities=[AzureDurableTaskDurability()])
            durability = AzureDurableTaskDurability.from_agent(agent)
            assert durability is not None
            register_activities(app, durability)
        functions = [
            function for function in _functions(app) if (function.get_function_name() or '').startswith('pydantic_ai_')
        ]
        assert len(functions) == 12
        names = _activities(app)
        assert len(names) == 12
        assert all(re.fullmatch(r'[a-zA-Z0-9_.`+:-]{1,100}', name) for name in names)
        assert 'first_x2F__x3C_agent_x3E_____model.request' in names
        assert all(re.fullmatch(r'pydantic_ai_[a-f0-9]{64}', f.get_function_name() or '') for f in functions)

    def test_unbound_capability(self) -> None:
        with pytest.raises(UserError, match='Bind `AzureDurableTaskDurability`'):
            register_activities(df.DFApp(), AzureDurableTaskDurability())

    def test_duplicate_registration_is_rejected_at_indexing(self) -> None:
        app = df.DFApp()
        agent = Agent(TestModel(), name='duplicate', capabilities=[AzureDurableTaskDurability()])
        durability = AzureDurableTaskDurability.from_agent(agent)
        assert durability is not None
        register_activities(app, durability)
        register_activities(app, durability)
        with pytest.raises(ValueError, match='unique function name'):
            app.get_functions()

    def test_blueprint(self) -> None:
        blueprint = df.Blueprint()
        agent = Agent(TestModel(), name='blueprint', capabilities=[AzureDurableTaskDurability()])
        durability = AzureDurableTaskDurability.from_agent(agent)
        assert durability is not None
        register_activities(blueprint, durability)
        app = df.DFApp()
        app.register_functions(blueprint)
        assert len(_activities(app)) == 6

    async def test_activity_uses_existing_codec_errors(self) -> None:
        functions = _activities(_sample().app)
        result = await functions['support____model.request'](payload='{}')
        assert isinstance(result, dict)
        assert 'serialization_error' in result

    def test_sample_native_orchestration(self, durable_worker: InMemoryWorker) -> None:
        example = _sample()
        _install_functions(durable_worker, example.app)
        durable_worker.add_orchestrator(example.support)
        instance_id = durable_worker.schedule('support', input='Hello')
        assert completed(durable_worker, instance_id) == 'Hosted durably'
        assert durable_worker.activity_calls.count('support____model.request') == 1
        assert all(
            binding.type not in {'httpTrigger', 'http'}
            for function in _functions(example.app)
            for binding in TypeAdapter(_Bindings).validate_json(function.get_function_json()).bindings
        )

    @pytest.mark.parametrize('completed_run', [False, True])
    async def test_sample_sdk_client(self, completed_run: bool, caplog: pytest.LogCaptureFixture) -> None:
        """Check the application-owned timer pattern without a host or a real wait."""
        client = MagicMock(spec=df.DurableFunctionsClient)
        client.schedule_new_orchestration.return_value = 'native-run'
        state = MagicMock(serialized_output='"Hosted durably"')
        client.wait_for_orchestration_completion.return_value = state if completed_run else None
        with caplog.at_level('INFO'):
            if completed_run:
                await _sample().start(MagicMock(spec=func.TimerRequest), client)
                state.raise_if_failed.assert_called_once_with()
                assert 'Agent output: "Hosted durably"' in caplog.text
            else:
                with pytest.raises(TimeoutError, match='native-run'):
                    await _sample().start(MagicMock(spec=func.TimerRequest), client)
        client.schedule_new_orchestration.assert_awaited_once_with('support', input='Hello')
        client.wait_for_orchestration_completion.assert_awaited_once_with('native-run', timeout=60)
