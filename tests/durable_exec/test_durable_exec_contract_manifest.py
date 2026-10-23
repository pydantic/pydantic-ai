"""The durable-exec contract manifest stays in sync with the engines it is generated from.

`scripts/durable_exec_contract.py` walks the durable engines and records the core definitions they
depend on. The committed manifest is what a review of a core change is matched against, so a stale
one silently stops routing a change to the engines it can break.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest
from inline_snapshot import snapshot

REPO_ROOT = Path(__file__).parents[2]

Manifest = dict[str, dict[str, dict[str, list[str]]]]


@runtime_checkable
class _ContractGenerator(Protocol):
    MANIFEST_PATH: str
    WRITE_COMMAND: str
    build_manifest: Callable[[Path], Manifest]
    render_manifest: Callable[[Manifest], str]
    main: Callable[..., int]


def _load_generator() -> _ContractGenerator:
    script = REPO_ROOT / 'scripts' / 'durable_exec_contract.py'
    spec = importlib.util.spec_from_file_location('durable_exec_contract', script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `dataclasses` looks the defining module up in `sys.modules` while it builds the classes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert isinstance(module, _ContractGenerator)
    return module


generator = _load_generator()


def test_manifest_is_current():
    """Regenerated in-process, the manifest matches the committed one byte for byte."""
    committed = (REPO_ROOT / generator.MANIFEST_PATH).read_text(encoding='utf-8')
    assert generator.render_manifest(generator.build_manifest(REPO_ROOT)) == committed, (
        f'{generator.MANIFEST_PATH} is stale; run `{generator.WRITE_COMMAND}` and commit the result'
    )


def test_manifest_records_the_member_level_contracts():
    """The classes nearly every core change touches are recorded member by member, never as a whole."""
    manifest = generator.build_manifest(REPO_ROOT)
    for core_file, class_name in [
        ('pydantic_ai_slim/pydantic_ai/_run_context.py', 'RunContext'),
        ('pydantic_ai_slim/pydantic_ai/models/__init__.py', 'ModelRequestContext'),
        ('pydantic_ai_slim/pydantic_ai/models/__init__.py', 'ModelRequestParameters'),
        ('pydantic_ai_slim/pydantic_ai/agent/__init__.py', 'Agent'),
    ]:
        qualnames = manifest[core_file]
        assert class_name not in qualnames
        assert any(qualname.startswith(f'{class_name}.') for qualname in qualnames), class_name
    run_context = manifest['pydantic_ai_slim/pydantic_ai/_run_context.py']
    assert run_context['RunContext.usage']['kinds'] == snapshot(['attribute_read'])
    assert run_context['RunContext.available_tool_names']['kinds'] == snapshot(['attribute_read', 'override'])
    projection = manifest['pydantic_ai_slim/pydantic_ai/models/__init__.py']['ModelRequestContext.streaming']
    assert projection['kinds'] == snapshot(['attribute_read', 'projected_field'])


# A miniature repository: a core package, an engine under `durable_exec/` and a Harness engine, which
# between them use every form of dependency the walk resolves.
_FIXTURE: dict[str, str] = {
    'pydantic_ai_slim/pydantic_ai/__init__.py': """
        from . import tools
        from ._run_context import RunContext as RunContext
        from .models import Model
        """,
    'pydantic_ai_slim/pydantic_ai/_run_context.py': """
        from dataclasses import KW_ONLY, dataclass
        from typing import ClassVar, Generic, TypeVar

        T = TypeVar('T')


        @dataclass
        class RunContext(Generic[T]):
            deps: T
            _: KW_ONLY
            usage: int = 0
            retries: int = 0
            kind: ClassVar[str] = 'run'
        """,
    'pydantic_ai_slim/pydantic_ai/tools.py': """
        from typing import TypeGuard

        from temporalio import workflow
        from ._run_context import RunContext
        from .cycle_a import Looped


        def is_run_context(value: object) -> TypeGuard[RunContext[object]]:
            return isinstance(value, RunContext)


        def make_context() -> 'RunContext[None]':
            return RunContext(None)
        """,
    'pydantic_ai_slim/pydantic_ai/cycle_a.py': 'from .cycle_b import Looped\n',
    'pydantic_ai_slim/pydantic_ai/cycle_b.py': 'from .cycle_a import Looped\n',
    'pydantic_ai_slim/pydantic_ai/profiles.py': """
        class Profile:
            supports_tools: bool = True


        class Response:
            text: str = ''
        """,
    'pydantic_ai_slim/pydantic_ai/models/__init__.py': """
        from __future__ import annotations

        from abc import ABC
        from typing import TYPE_CHECKING, Annotated, Optional

        from pydantic import BaseModel

        from ..profiles import Profile, Response

        if TYPE_CHECKING:
            from .. import tools
        else:
            tools = None

        try:
            import json as _json
        except ImportError:
            _json = None
        else:
            pass
        finally:
            pass

        MODEL_NAMES = ('a', 'b')


        class Model(ABC):
            default = Profile()
            first, second = 1, 2
            label = 'model'

            def __init__(self, profile: Profile, response: 'Response') -> None:
                self._profile = profile
                self.last: Optional[Response] = None
                self.fresh = Profile()
                self.computed = 1 + 1
                self._profile = profile
                response.text = ''

            @property
            def profile(self) -> Profile:
                return self._profile

            def request(self) -> Response:
                return Response()

            def request(self) -> Response:
                return Response()

            @staticmethod
            def helper():
                pass

            def setter(self) -> None:
                other = self
                other.label = 'other'


        class WrapperModel(Model):
            wrapped: Annotated[Model, 'the wrapped model']

            def request(self) -> Response:
                return self.wrapped.request()


        class Settings(BaseModel):
            timeout: 'float | None' = None
            broken: 'not valid python(' = None


        class Left(Model):
            pass


        class Right(Model):
            pass


        class Diamond(Left, Right):
            pass
        """,
    'pydantic_ai_slim/pydantic_ai/durable_exec/__init__.py': '',
    'pydantic_ai_slim/pydantic_ai/durable_exec/temporal/__init__.py': """
        from ._model import TemporalModel as TemporalModel
        """,
    'pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_model.py': """
        from __future__ import annotations

        import dataclasses
        import pydantic_ai.tools as tool_module
        from dataclasses import dataclass, fields, replace
        from typing import Any, cast

        import temporalio
        from temporalio import workflow

        from pydantic_ai import RunContext, tools
        from pydantic_ai.cycle_a import Looped
        from pydantic_ai.models import MODEL_NAMES, Diamond, Model, Settings, WrapperModel
        from pydantic_ai.profiles import Profile, Response
        from ...tools import make_context


        class TemporalModel(WrapperModel):
            name = 'temporal'
            wrapped: Model

            def __init__(self, wrapped: Model) -> None:
                self.wrapped = wrapped

            def request(self) -> Response:
                response = super().request()
                profile = self.wrapped.profile
                return cast(Response, profile.supports_tools and response)

            async def stream(self, ctx: RunContext[Any], *values: object, **options: object) -> None:
                ctx.usage
                getattr(ctx, 'retries')
                getattr(ctx, ctx.deps)
                ctx.__dict__['deps']
                ctx.__dict__.get('usage')
                ctx.__dataclass_fields__
                ctx.__class__
                ctx.missing
                fields(ctx)
                replace(ctx, retries=1, **options)
                RunContext(None, usage=1, **options)
                (await self.fetch()).text
                (self.wrapped or Model()).label
                (self.wrapped if values else Model()).label

            async def fetch(self) -> Response:
                return Response()

            @staticmethod
            def build(model: Model) -> Profile:
                return model.profile


        @dataclass
        class ContextProjection:
            usage: int
            retries: int
            elsewhere: str

            @classmethod
            def from_context(cls, ctx: RunContext[Any]) -> ContextProjection:
                return cls(usage=ctx.usage, retries=ctx.retries, elsewhere='')

            @staticmethod
            def from_model(model: Model, *, settings: Settings) -> None:
                pass


        def module_level(
            model: Model | None, value: object, other: object, diamond: Diamond, wrapper: WrapperModel
        ) -> None:
            super().request()
            tool_module.make_context().usage
            tools.make_context().deps
            tools.MODEL_NAMES
            make_context().retries
            dataclasses.fields(model)
            temporalio.workflow
            workflow.now()
            if isinstance(value, (Model, Profile)):
                value.label
            if tools.is_run_context(other):
                other.deps
            if callable(value):
                value()
            isinstance()
            MODEL_NAMES[0]
            diamond.missing
            Settings().missing
            Settings().timeout
            Settings().broken.missing
            wrapper.wrapped.label
            tools.RunContext.retries
            tools.make_context.__name__

            def nested(inner: Model) -> None:
                inner.label
                getattr(Settings(), value)

            nested(Model())
            lambda: model
        """,
    'src/pydantic_ai_harness/pydantic_ai_harness/__init__.py': '',
    'src/pydantic_ai_harness/pydantic_ai_harness/_shared.py': 'class Shared:\n    value: int = 0\n',
    'src/pydantic_ai_harness/pydantic_ai_harness/render/__init__.py': '',
    'src/pydantic_ai_harness/pydantic_ai_harness/render/_capability.py': """
        from pydantic_ai.models import Model

        from .._shared import Shared


        class RenderModel(Model, Shared):
            value = 1

            def request(self):
                return self.value
        """,
}


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    for relative, source in _FIXTURE.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip(), encoding='utf-8')
    return tmp_path


def test_build_manifest_resolves_every_dependency_form(fixture_repo: Path):
    """Imports, overrides, annotation-resolved reads and projections, on a repository small enough to read."""
    manifest = generator.build_manifest(fixture_repo)

    flattened = {
        f'{core_file.removeprefix("pydantic_ai_slim/pydantic_ai/")}::{qualname}': (
            entry['kinds'],
            entry['engines'],
            [site.rsplit('::', 1)[1] for site in entry['sites']],
        )
        for core_file, definitions in manifest.items()
        for qualname, entry in definitions.items()
    }
    assert flattened == snapshot(
        {
            '_run_context.py::RunContext.deps': (
                ['attribute_read'],
                ['temporal'],
                ['TemporalModel.stream', 'module_level'],
            ),
            '_run_context.py::RunContext.retries': (
                ['attribute_read', 'projected_field'],
                ['temporal'],
                ['ContextProjection.from_context', 'ContextProjection.retries', 'TemporalModel.stream', 'module_level'],
            ),
            '_run_context.py::RunContext.usage': (
                ['attribute_read', 'projected_field'],
                ['temporal'],
                ['ContextProjection.from_context', 'ContextProjection.usage', 'TemporalModel.stream', 'module_level'],
            ),
            'models/__init__.py::Diamond': (['attribute_read', 'import'], ['temporal'], ['<module>', 'module_level']),
            'models/__init__.py::MODEL_NAMES': (['import'], ['temporal'], ['<module>']),
            'models/__init__.py::Model': (['import'], ['render', 'temporal'], ['<module>', '<module>']),
            'models/__init__.py::Model.__init__': (['override'], ['temporal'], ['TemporalModel.__init__']),
            'models/__init__.py::Model.label': (
                ['attribute_read'],
                ['temporal'],
                ['TemporalModel.stream', 'module_level', 'module_level.nested'],
            ),
            'models/__init__.py::Model.profile': (
                ['attribute_read'],
                ['temporal'],
                ['TemporalModel.build', 'TemporalModel.request'],
            ),
            'models/__init__.py::Model.request': (
                ['override'],
                ['render', 'temporal'],
                ['TemporalModel.request', 'RenderModel.request'],
            ),
            'models/__init__.py::Settings': (['import'], ['temporal'], ['<module>']),
            'models/__init__.py::Settings.broken': (['attribute_read'], ['temporal'], ['module_level']),
            'models/__init__.py::Settings.timeout': (['attribute_read'], ['temporal'], ['module_level']),
            'models/__init__.py::WrapperModel': (['import'], ['temporal'], ['<module>']),
            'models/__init__.py::WrapperModel.request': (
                ['attribute_read', 'override'],
                ['temporal'],
                ['TemporalModel.request'],
            ),
            'models/__init__.py::WrapperModel.wrapped': (
                ['attribute_read', 'override'],
                ['temporal'],
                ['TemporalModel.wrapped', 'module_level'],
            ),
            'profiles.py::Profile': (['import'], ['temporal'], ['<module>']),
            'profiles.py::Profile.supports_tools': (['attribute_read'], ['temporal'], ['TemporalModel.request']),
            'profiles.py::Response': (['import'], ['temporal'], ['<module>']),
            'profiles.py::Response.text': (['attribute_read'], ['temporal'], ['TemporalModel.stream']),
            'tools.py::is_run_context': (['import'], ['temporal'], ['module_level']),
            'tools.py::make_context': (['import'], ['temporal'], ['<module>', 'module_level']),
        }
    )


def test_main_writes_then_checks_the_manifest(fixture_repo: Path, capsys: pytest.CaptureFixture[str]):
    manifest_path = fixture_repo / generator.MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True)

    assert generator.main([], root=fixture_repo) == 1
    assert generator.main(['--write'], root=fixture_repo) == 0
    assert generator.main([], root=fixture_repo) == 0

    manifest_path.write_text('{}\n', encoding='utf-8')
    assert generator.main([], root=fixture_repo) == 1
    assert capsys.readouterr().err == snapshot("""\
scripts/durable_exec_contract.json is stale; run `uv run python scripts/durable_exec_contract.py --write`
scripts/durable_exec_contract.json is stale; run `uv run python scripts/durable_exec_contract.py --write`
""")
