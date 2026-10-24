"""The durable-exec contract manifest stays in sync with the engines it is generated from.

`scripts/durable_exec_contract.py` walks the durable engines and records the core definitions they
depend on. The committed manifest is what a review of a core change is matched against, so a stale
one silently stops routing a change to the engines it can break. These tests read source only, so
they are not VCR tests.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
from inline_snapshot import snapshot

sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts'))

import durable_exec_contract as generator
from durable_exec_contract import REPO_ROOT, Manifest

CORE = 'pydantic_ai_slim/pydantic_ai'
_DURABILITY_BASE = re.compile(r'(?:\w+\.)*BaseDurabilityCapability(?:\[.*\])?')


def test_manifest_is_current(capsys: pytest.CaptureFixture[str]):
    """Regenerated in-process, the manifest matches the committed one byte for byte."""
    assert generator.main([]) == 0, capsys.readouterr().err


def test_engines_cover_every_durability_capability():
    """Every `durable_exec` subpackage and every module with a class based on `BaseDurabilityCapability` is walked."""
    walked = {path for pattern in generator.ENGINES.values() for path in REPO_ROOT.glob(pattern)}
    durable_exec = REPO_ROOT / CORE / 'durable_exec'
    subpackages = {
        path
        for package in durable_exec.iterdir()
        if (package / '__init__.py').is_file()
        for path in package.rglob('*.py')
    }
    capabilities = {
        path
        for root in (CORE, 'src/pydantic_ai_harness/pydantic_ai_harness')
        for path in (REPO_ROOT / root).rglob('*.py')
        if 'BaseDurabilityCapability' in (source := path.read_text(encoding='utf-8'))
        and any(
            isinstance(node, ast.ClassDef) and any(_DURABILITY_BASE.fullmatch(ast.unparse(base)) for base in node.bases)
            for node in ast.walk(ast.parse(source))
        )
    }
    assert capabilities
    assert sorted(str(path.relative_to(REPO_ROOT)) for path in (subpackages | capabilities) - walked) == []


def test_manifest_records_the_member_level_contracts():
    """In the committed manifest, the classes nearly every core change touches are recorded member by member only."""
    manifest: Manifest = json.loads((REPO_ROOT / generator.MANIFEST_PATH).read_text(encoding='utf-8'))
    for core_file, class_name in [
        (f'{CORE}/_run_context.py', 'RunContext'),
        (f'{CORE}/models/__init__.py', 'ModelRequestContext'),
        (f'{CORE}/models/__init__.py', 'ModelRequestParameters'),
        (f'{CORE}/agent/__init__.py', 'Agent'),
    ]:
        qualnames = manifest[core_file]
        assert class_name not in qualnames
        assert any(qualname.startswith(f'{class_name}.') for qualname in qualnames), class_name
    run_context = manifest[f'{CORE}/_run_context.py']
    assert run_context['RunContext.usage']['kinds'] == snapshot(['attribute_read'])
    assert run_context['RunContext.available_tool_names']['kinds'] == snapshot(['attribute_read', 'override'])
    projection = manifest[f'{CORE}/models/__init__.py']['ModelRequestContext.streaming']
    assert projection['kinds'] == snapshot(['attribute_read', 'projected_field'])


def _keyed_entries(manifest: Manifest) -> Iterator[tuple[str, dict[str, list[str]]]]:
    """Each manifest entry, keyed `<core file under pydantic_ai>::<qualname>`."""
    for core_file, definitions in manifest.items():
        for qualname, entry in definitions.items():
            yield core_file.removeprefix(CORE + '/') + f'::{qualname}', entry


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    # A miniature repository: a core package, an engine under `durable_exec/` and a Harness engine, which
    # between them use every form of dependency the walk resolves.
    files: dict[str, str] = {
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
            from typing import TypeGuard, TypeIs

            from temporalio import workflow
            from ._run_context import RunContext
            from .cycle_a import Looped


            def is_run_context(value: object) -> TypeGuard[RunContext[object]]:
                return isinstance(value, RunContext)


            def is_context(value: object) -> TypeIs[RunContext[object]]:
                return isinstance(value, RunContext)


            def make_context() -> 'RunContext[None]':
                return RunContext(None)


            def make_contexts() -> list[RunContext[None]]:
                return [RunContext(None)]
            """,
        'pydantic_ai_slim/pydantic_ai/cycle_a.py': 'from .cycle_b import Looped\n',
        'pydantic_ai_slim/pydantic_ai/cycle_b.py': 'from .cycle_a import Looped\n',
        'pydantic_ai_slim/pydantic_ai/profiles.py': """
            class Profile:
                supports_tools: bool = True


            KNOWN: 'list[Profile]' = []


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
                responses: 'list[Response] | None' = None
                profiles: Optional[list[Profile]] = None
                tagged: Annotated[list[Profile], 'tagged'] = []
                unparsable: 'list[(' = []

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

                def history(self) -> list[Response]:
                    return []

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


            ModelAlias = Model
            Sides = Annotated[Left | Right, 'sides']
            Pending: Model
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
            from typing import TYPE_CHECKING, Any, cast

            import temporalio
            from temporalio import workflow

            from pydantic_ai import RunContext, profiles as profile_module, tools
            from pydantic_ai.cycle_a import Looped
            from pydantic_ai.models import MODEL_NAMES, Diamond, Left, Model, ModelAlias, Pending, Settings, Sides, WrapperModel
            from pydantic_ai.profiles import KNOWN, Profile, Response
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


            @dataclass
            class StaticProjection:
                deps: object

                @staticmethod
                def from_context(ctx: RunContext[Any]) -> None:
                    ctx.kind


            def guarded(candidate: object) -> None:
                if tools.is_context(candidate):
                    candidate.kind


            def reflective(model: Model, name: str) -> None:
                setattr(model, 'label', '')
                hasattr(model, 'profile')
                delattr(model, 'default')
                model.__dict__[name]
                model.__dict__.get(name)


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


            async def iterate(
                models: list[ModelAlias], model: Model, stream: 'AsyncIterator[Sides]', pending: list[Pending], count: int
            ) -> None:
                for each in models:
                    each.request
                async for side in stream:
                    side.missing
                for response in model.responses:
                    response.text
                for profile in model.profiles:
                    profile.supports_tools
                for tagged in model.tagged:
                    tagged.supports_tools
                for unparsed in model.unparsable:
                    unparsed.missing
                for character in model.label:
                    character.upper
                for number in range(count):
                    number.real
                for left, right in pending:
                    left.label
                annotated: list[Profile] = []
                for local in annotated:
                    local.supports_tools
                for absent in model.absent:
                    absent.missing
                for absent_call in model.absent():
                    absent_call.missing
                for made in tools.make_contexts():
                    made.usage
                for built in Model():
                    built.missing
                for from_module in profile_module.unknown():
                    from_module.missing


            def spread(model: Model, options: dict[str, object]) -> None:
                replace(model, **options)
                Profile(**options)
                object.__getattribute__(model, 'label')


            def through_an_imported_variable() -> None:
                for listed in KNOWN:
                    listed.supports_tools


            def through_a_module_variable() -> None:
                for known in profile_module.KNOWN:
                    known.supports_tools


            def through_varargs(*profiles: Profile) -> None:
                for each in profiles:
                    each.supports_tools


            def through_a_method(model: Model) -> None:
                for called in model.history():
                    called.text


            def through_a_comprehension(model: Model) -> None:
                [entry.text for entry in model.responses]


            def local_import() -> None:
                from pydantic_ai._run_context import RunContext as LocalContext

                LocalContext.kind


            if TYPE_CHECKING:
                from pydantic_ai.models import Right

                RightAlias = Right


            def typed(
                model: Left, response: profile_module.Response, *, label: str = Model.label, retries: int
            ) -> 'Left':
                value: Left = model
                return value
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
    for relative, source in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip(), encoding='utf-8')
    return tmp_path


def test_build_manifest_resolves_every_dependency_form(fixture_repo: Path):
    """Imports, overrides, annotation-resolved reads and projections, on a repository small enough to read."""
    manifest = generator.build_manifest(fixture_repo)

    flattened = {
        key: (
            entry['kinds'],
            entry['engines'],
            [site.rsplit('::', 1)[1] for site in entry['sites']],
        )
        for key, entry in _keyed_entries(manifest)
    }
    assert flattened == snapshot(
        {
            '_run_context.py::RunContext.deps': (
                ['attribute_read', 'projected_field'],
                ['temporal'],
                ['StaticProjection.deps', 'TemporalModel.stream', 'module_level'],
            ),
            '_run_context.py::RunContext.kind': (
                ['attribute_read'],
                ['temporal'],
                ['StaticProjection.from_context', 'guarded', 'local_import'],
            ),
            '_run_context.py::RunContext.retries': (
                ['attribute_read', 'projected_field'],
                ['temporal'],
                ['ContextProjection.from_context', 'ContextProjection.retries', 'TemporalModel.stream', 'module_level'],
            ),
            '_run_context.py::RunContext.usage': (
                ['attribute_read', 'projected_field'],
                ['temporal'],
                [
                    'ContextProjection.from_context',
                    'ContextProjection.usage',
                    'TemporalModel.stream',
                    'iterate',
                    'module_level',
                ],
            ),
            'models/__init__.py::Diamond': (
                ['annotation_only', 'attribute_read'],
                ['temporal'],
                ['<module>', 'module_level'],
            ),
            'models/__init__.py::Left': (['annotation_only', 'attribute_read'], ['temporal'], ['<module>', 'iterate']),
            'models/__init__.py::MODEL_NAMES': (['import'], ['temporal'], ['<module>']),
            'models/__init__.py::Model': (
                ['attribute_read', 'import'],
                ['render', 'temporal'],
                ['<module>', 'iterate', 'reflective', 'spread', '<module>'],
            ),
            'models/__init__.py::Model.__init__': (['override'], ['temporal'], ['TemporalModel.__init__']),
            'models/__init__.py::Model.default': (['attribute_read'], ['temporal'], ['reflective']),
            'models/__init__.py::Model.history': (['attribute_read'], ['temporal'], ['through_a_method']),
            'models/__init__.py::Model.label': (
                ['attribute_read'],
                ['temporal'],
                [
                    '<module>',
                    'TemporalModel.stream',
                    'iterate',
                    'module_level',
                    'module_level.nested',
                    'reflective',
                    'spread',
                ],
            ),
            'models/__init__.py::Model.profile': (
                ['attribute_read'],
                ['temporal'],
                ['TemporalModel.build', 'TemporalModel.request', 'reflective'],
            ),
            'models/__init__.py::Model.profiles': (['attribute_read'], ['temporal'], ['iterate', 'module_level']),
            'models/__init__.py::Model.request': (
                ['attribute_read', 'override'],
                ['render', 'temporal'],
                ['TemporalModel.request', 'iterate', 'RenderModel.request'],
            ),
            'models/__init__.py::Model.responses': (
                ['attribute_read'],
                ['temporal'],
                ['iterate', 'module_level', 'through_a_comprehension'],
            ),
            'models/__init__.py::Model.tagged': (['attribute_read'], ['temporal'], ['iterate', 'module_level']),
            'models/__init__.py::Model.unparsable': (['attribute_read'], ['temporal'], ['iterate', 'module_level']),
            'models/__init__.py::ModelAlias': (['annotation_only'], ['temporal'], ['<module>']),
            'models/__init__.py::Pending': (['annotation_only'], ['temporal'], ['<module>']),
            'models/__init__.py::Right': (['annotation_only', 'attribute_read'], ['temporal'], ['<module>', 'iterate']),
            'models/__init__.py::Settings': (['import'], ['temporal'], ['<module>']),
            'models/__init__.py::Settings.broken': (['attribute_read'], ['temporal'], ['module_level']),
            'models/__init__.py::Settings.timeout': (['attribute_read'], ['temporal'], ['module_level']),
            'models/__init__.py::Sides': (['annotation_only'], ['temporal'], ['<module>']),
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
            'profiles.py::KNOWN': (['import'], ['temporal'], ['<module>', 'through_a_module_variable']),
            'profiles.py::Profile': (['attribute_read', 'import'], ['temporal'], ['<module>', 'spread']),
            'profiles.py::Profile.supports_tools': (
                ['attribute_read'],
                ['temporal'],
                [
                    'TemporalModel.request',
                    'iterate',
                    'through_a_module_variable',
                    'through_an_imported_variable',
                    'through_varargs',
                ],
            ),
            'profiles.py::Response': (['annotation_only', 'import'], ['temporal'], ['<module>', 'typed']),
            'profiles.py::Response.text': (
                ['attribute_read'],
                ['temporal'],
                ['TemporalModel.stream', 'iterate', 'through_a_comprehension', 'through_a_method'],
            ),
            'tools.py::is_context': (['import'], ['temporal'], ['guarded']),
            'tools.py::is_run_context': (['import'], ['temporal'], ['module_level']),
            'tools.py::make_context': (['import'], ['temporal'], ['<module>', 'module_level']),
            'tools.py::make_contexts': (['import'], ['temporal'], ['iterate']),
        }
    )


def test_gates_ignores_annotation_only_entries(fixture_repo: Path):
    """A definition the engines name only in annotations stays in the manifest but does not gate review."""
    manifest = generator.build_manifest(fixture_repo)

    kinds = {
        key: (
            entry['kinds'],
            generator.gates(entry),
        )
        for key, entry in _keyed_entries(manifest)
        if 'annotation_only' in entry['kinds']
    }
    assert kinds == snapshot(
        {
            'models/__init__.py::Diamond': (['annotation_only', 'attribute_read'], True),
            'models/__init__.py::Left': (['annotation_only', 'attribute_read'], True),
            'models/__init__.py::ModelAlias': (['annotation_only'], False),
            'models/__init__.py::Pending': (['annotation_only'], False),
            'models/__init__.py::Right': (['annotation_only', 'attribute_read'], True),
            'models/__init__.py::Sides': (['annotation_only'], False),
            'profiles.py::Response': (['annotation_only', 'import'], True),
        }
    )


def test_main_writes_then_checks_the_manifest(fixture_repo: Path, capsys: pytest.CaptureFixture[str]):
    manifest_path = fixture_repo / generator.MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True)

    assert generator.main([], root=fixture_repo) == 1
    assert capsys.readouterr().err == snapshot(
        '`scripts/durable_exec_contract.json` is stale; run `uv run python scripts/durable_exec_contract.py --write`\n'
    )
    assert generator.main(['--write'], root=fixture_repo) == 0
    assert generator.main([], root=fixture_repo) == 0
    assert capsys.readouterr().err == ''

    manifest_path.write_text('{}\n', encoding='utf-8')
    assert generator.main([], root=fixture_repo) == 1
    assert capsys.readouterr().err == snapshot(
        '`scripts/durable_exec_contract.json` is stale; run `uv run python scripts/durable_exec_contract.py --write`\n'
    )
