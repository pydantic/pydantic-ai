# TODO: Add assertions to DatasetRow and shared_assertions to Dataset
from __future__ import annotations as _annotations

import asyncio
import functools
import sys
from inspect import iscoroutinefunction
from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar, Generic, Self

import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic_core import to_json, to_jsonable_python
from typing_extensions import TypeVar

from pydantic_ai import models
from pydantic_evals.assertions import Assertion, AssertionFunction, SerializedAssertion
from pydantic_evals.scoring import ScoringContext

InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])

DEFAULT_DATASET_PATH = './test_cases.yaml'


class _BaseDatasetRow(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid'):
    """A single row of a "dataset", consisting of input, expected output, and metadata."""

    name: str
    inputs: InputsT
    metadata: MetadataT
    expected_output: OutputT | None = None


class SerializedDatasetRow(_BaseDatasetRow[InputsT, OutputT, MetadataT]):
    """A single row of a "dataset", consisting of input, expected output, and metadata."""

    __pydantic_config__ = {'extra': 'forbid'}

    assertions: list[SerializedAssertion] = Field(default_factory=list)


class DatasetRow(_BaseDatasetRow[InputsT, OutputT, MetadataT]):
    """A single row of a "dataset", consisting of input, expected output, and metadata."""

    __pydantic_config__ = {'extra': 'forbid'}

    assertions: list[Assertion[InputsT, OutputT, MetadataT]] = Field(default_factory=list)


class Dataset(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid'):
    """A dataset of test cases, each consisting of input, expected output, and metadata."""

    rows: list[SerializedDatasetRow[InputsT, OutputT, MetadataT]]

    assertion_registry: ClassVar[dict[str, AssertionFunction[Any, Any, Any]]] = {}

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls.assertion_registry = cls.assertion_registry.copy()

    @classmethod
    def register_assertion(cls, function: AssertionFunction[InputsT, OutputT, MetadataT], name: str | None = None):
        name = name or function.__name__
        if name in cls.assertion_registry:
            raise ValueError(f'Assertion {name!r} is already registered')
        cls.assertion_registry[name] = function

        if iscoroutinefunction(function):

            async def not_function(*args: Any, **kwargs: Any) -> bool:  # pyright: ignore[reportRedeclaration]
                return not await function(*args, **kwargs)
        else:

            def not_function(*args: Any, **kwargs: Any) -> bool:
                return not function(*args, **kwargs)

        not_function = functools.wraps(function)(not_function)  # pyright: ignore[reportAssignmentType]
        not_name = f'not_{name}'
        not_function.__name__ = not_name
        not_function.__qualname__ = f'{".".join(not_function.__qualname__.split(".")[:-1])}.{not_name}'
        cls.assertion_registry[not_name] = not_function

    @classmethod
    def inherited_assertion_registry(cls) -> dict[str, AssertionFunction[Any, Any, Any]]:
        full_registry: dict[str, AssertionFunction[Any, Any, Any]] = {}
        for base in cls.__mro__[::-1]:
            if issubclass(base, Dataset):
                full_registry.update(base.assertion_registry)
        return full_registry

    @classmethod
    @functools.cache
    def type_adapter(cls) -> TypeAdapter[Self]:
        return TypeAdapter(cls)

    @classmethod
    @functools.cache
    def serialized_row_type(cls) -> type[SerializedDatasetRow[InputsT, OutputT, MetadataT]]:
        for c in cls.__mro__:
            metadata = getattr(c, '__pydantic_generic_metadata__')
            if len(args := metadata.get('args')) == 3:
                return SerializedDatasetRow[args]  # type: ignore
            elif len(args := getattr(c, '__args__', ())) == 3:
                return SerializedDatasetRow[args]  # type: ignore
        raise ValueError(f'Could not find the DatasetRow type for {cls}')

    @classmethod
    @functools.cache
    def row_type(cls) -> type[DatasetRow[InputsT, OutputT, MetadataT]]:
        for c in cls.__mro__:
            metadata = getattr(c, '__pydantic_generic_metadata__')
            if len(args := metadata.get('args')) == 3:
                return DatasetRow[args]  # type: ignore
            elif len(args := getattr(c, '__args__', ())) == 3:
                return DatasetRow[args]  # type: ignore
        raise ValueError(f'Could not find the DatasetRow type for {cls}')

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_DATASET_PATH) -> Self:
        path = cls._get_relative_path(path)
        if not path.exists():
            raise FileNotFoundError(f'{cls.__name__} dataset file {path} does not exist')

        raw = path.read_text()
        loaded = yaml.safe_load(raw)
        try:
            result = cls.model_validate(loaded)
            result.rows = [cls.serialized_row_type().model_validate(row.model_dump()) for row in result.rows]
        except ValidationError as e:
            raise ValueError(
                f'{cls.__name__} dataset file {path} contains data that does not match the schema:\n{e}.'
            ) from e
        try:
            result.validate_assertions()
        except ValueError as e:
            raise ValueError(f'{cls.__name__} dataset file {path} contains invalid assertions:\n{e}.') from e
        return result

    def deserialized_rows(self) -> list[DatasetRow[InputsT, OutputT, MetadataT]]:
        registry = self.inherited_assertion_registry()

        rows: list[DatasetRow[InputsT, OutputT, MetadataT]] = []
        for row in self.rows:
            assertions = [a.bind(registry) for a in row.assertions]
            row = self.row_type()(
                name=row.name,
                inputs=row.inputs,
                metadata=row.metadata,
                expected_output=row.expected_output,
                assertions=assertions,
            )
            rows.append(row)

        return rows

    def validate_assertions(self) -> None:
        registry = self.inherited_assertion_registry()
        for row in self.rows:
            for assertion in row.assertions:
                assertion.validate_against_registry(registry, row.name)

    # @classmethod
    # def model_json_schema_with_assertions(cls) -> dict[str, Any]:
    #     # TODO: Implement this to get type-safe JSON schema for registered assertions
    #     registry = cls.inherited_assertion_registry()
    #     tds: list[Any] = []
    #     for name, function in registry.items():
    #         signature = inspect.signature(function)
    #         first_param = list(signature.parameters.values())[0]
    #         type_hints = _typing_extra.get_function_type_hints(function)
    #         type_hints.pop(first_param.name)
    #         type_hints.pop('return', None)
    #         for p in signature.parameters.values():
    #             if p.default is not p.empty:
    #                 type_hints[p.name] = NotRequired[type_hints[p.name]]
    #         type_hints['call'] = Literal[name]
    #         td = TypedDict(f'{function.__name__.title()}Assertion', type_hints)  # pyright : ignore
    #         td.__pydantic_config__ = {'extra': 'forbid'}  # pyright : ignore
    #         tds.append(td)
    #
    #     params: tuple[Any, ...] | None = None
    #     for cls_ in cls.__mro__:
    #         if issubclass(cls_, BaseModel) and cls_.__pydantic_generic_metadata__['args']:
    #             params = cls_.__pydantic_generic_metadata__['args']
    #             break
    #     assert params is not None
    #
    #     class ClsDatasetRow(DatasetRow[params[0], params[1], params[2]]):
    #         assertions: list[Union[tuple(tds)]]  # pyright : ignore
    #     ClsDatasetRow.__name__ = cls.__name__ + 'Row'
    #
    #     class ClsDataset(cls):
    #         rows: list[ClsDatasetRow[params[0], params[1], params[2]]]  # pyright : ignore
    #
    #     ClsDataset.__name__ = cls.__name__
    #
    #     return ClsDataset.model_json_schema()

    @classmethod
    def generate_dataset_files(
        cls,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
        schema_path: Path | str | None = None,
    ) -> str:
        dataset_path = cls._get_relative_path(dataset_path)

        if schema_path is None:
            if dataset_path.exists():
                # Try to infer the schema path from the first line of the existing dataset file
                first_line = dataset_path.read_text().split('\n', 1)[0]
                if first_line.startswith('# yaml-language-server: $schema='):
                    schema_path = (dataset_path.parent / first_line.split('=', 1)[1]).resolve()
            if schema_path is None:
                schema_path = dataset_path.parent / f'./{dataset_path.stem}_schema.json'
        else:
            schema_path = cls._get_relative_path(schema_path)

        schema_content = to_json(cls.type_adapter().json_schema(), indent=2).decode() + '\n'
        if not schema_path.exists() or schema_path.read_text() != schema_content:
            schema_path.write_text(schema_content)

        schema_ref = str(_get_relative_path_reference(schema_path, dataset_path.parent))
        yaml_language_server_line = f'# yaml-language-server: $schema={schema_ref}'
        if dataset_path.exists():
            try:
                cls.from_yaml(dataset_path)
            except ValueError as e:
                if isinstance(e.__cause__, ValidationError):
                    raise ValueError(
                        f'{cls.__name__} dataset file {dataset_path} already exists, but does not contain compatible data.'
                        f' Fix or delete the file before calling this function:\n{e.__cause__}'
                    ) from e.__cause__
                else:
                    raise
            dataset_text = dataset_path.read_text()
            cases_text_with_schema = _ensure_yaml_language_server_line(dataset_text, schema_ref)
            if cases_text_with_schema != dataset_text:
                dataset_path.write_text(cases_text_with_schema)
        else:
            content = yaml.dump(to_jsonable_python(cls(rows=[])), sort_keys=False)
            dataset_path.write_text(f'{yaml_language_server_line}\n{content}')
        return schema_ref

    @classmethod
    def generate_dataset_examples(
        cls,
        model: models.Model | models.KnownModelName = 'gpt-4o',
        min_count: int = 3,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
    ):
        dataset_path = cls._get_relative_path(dataset_path)
        schema_ref = cls.generate_dataset_files(dataset_path=dataset_path, schema_path=None)

        existing_content: str | None = None

        try:
            existing_rows = cls.from_yaml(dataset_path).rows
            min_count = max(0, min_count - len(existing_rows))
            if min_count == 0:
                return  # nothing to do, already have enough examples
            if existing_rows:
                existing_content = dataset_path.read_text()
        except FileNotFoundError:
            pass  # in this case, we'll generate a new file, so we ignore the error

        examples = asyncio.run(_generate_examples(cls, dataset_path, model, min_count))

        if existing_content is None:
            content = yaml.dump(to_jsonable_python(cls(rows=examples)), sort_keys=False)
            content = _ensure_yaml_language_server_line(content, schema_ref)
            dataset_path.write_text(content)
        else:
            new_lines = yaml.dump(to_jsonable_python(cls(rows=examples)), sort_keys=False).splitlines()
            new_lines = new_lines[1:]  # drop the first line, which is the document start
            new_content = _ensure_yaml_language_server_line(existing_content, schema_ref)
            if not new_content.endswith('\n'):
                new_content += '\n'
            new_content += '\n'.join(new_lines)
            dataset_path.write_text(new_content)

    def save(self, path: Path | str = DEFAULT_DATASET_PATH, schema_ref: str | None = None) -> None:
        path = self._get_relative_path(path)
        content = yaml.dump(to_jsonable_python(self), sort_keys=False)
        if schema_ref is not None:
            content = _ensure_yaml_language_server_line(content, schema_ref)
        path.write_text(content)

    @classmethod
    def _get_schema_path(cls, dataset_path: Path) -> Path:
        return dataset_path.parent / f'./{dataset_path.stem}_schema.json'

    @classmethod
    def _get_relative_path(cls, path: Path | str) -> Path:
        """Resolve relative paths as relative to the module in which the (sub)class is defined."""
        path = Path(path)

        if path.is_absolute():
            return path

        module_path = sys.modules[cls.__module__].__file__
        if module_path == __file__:
            raise ValueError(f'You should only call this method from a proper subclass of `{cls.__name__}`')

        assert module_path is not None, 'Module must be a file-based module'
        root = Path(module_path).parent
        return root / path


async def llm_rubric(
    ctx: ScoringContext[Any, Any, Any],
    rubric: str,
    model: models.KnownModelName = 'gpt-4o',
    include_input: bool = False,
) -> tuple[bool, str]:
    """Judge whether the output of a language model meets the criteria of a provided rubric."""
    if include_input:
        from pydantic_evals.llm_as_a_judge import judge_input_output

        grading_output = await judge_input_output(ctx.inputs, ctx.output, rubric, model)
    else:
        from pydantic_evals.llm_as_a_judge import judge_output

        grading_output = await judge_output(ctx.output, rubric, model)
    return grading_output.pass_, grading_output.reason


async def is_instance(ctx: ScoringContext[Any, Any, Any], type_name: str) -> bool | tuple[bool, str]:
    """Check if the output is an instance of a type with the given name."""
    output = ctx.output
    for cls in type(output).__mro__:
        if cls.__name__ == type_name:
            return True
    return False, f'output is of type {type(output).__qualname__}'


Dataset.register_assertion(llm_rubric, 'llm_rubric')
Dataset.register_assertion(is_instance, 'is_instance')
# TODO: Other common assertions could be added here, such as "is_equal", "is_greater_than", etc.


def _ensure_yaml_language_server_line(content: str, schema_ref: str) -> str:
    first_line = content.split('\n', 1)[0]
    yaml_language_server_line = f'# yaml-language-server: $schema={schema_ref}'
    if first_line == yaml_language_server_line:
        return content
    elif first_line.startswith('# yaml-language-server: $schema='):
        return '\n'.join([yaml_language_server_line] + content.split('\n')[1:])
    else:
        return f'{yaml_language_server_line}\n{content}'


async def _generate_examples(
    dataset_type: type[Dataset[Any, Any, Any]],
    path: Path,
    model: models.Model | models.KnownModelName = 'gpt-4o',
    n_examples: int = 3,
) -> list[SerializedDatasetRow[Any, Any, Any]]:
    if path.exists():
        cases_text = path.read_text()
        try:
            loaded = yaml.safe_load(cases_text)
        except yaml.YAMLError:
            raise ValueError(f'Cases file {path} is not valid YAML')

        try:
            existing_cases = dataset_type.type_adapter().validate_python(loaded).rows
        except ValidationError as e:
            raise ValueError(
                f'Cases file {path} contains data that does not match the schema. Delete the file before calling this function.'
            ) from e
    else:
        existing_cases = []

    n_examples = max(0, n_examples - len(existing_cases))
    if n_examples == 0:
        return []

    from pydantic_ai import Agent  # import locally to prevent circular dependencies

    agent = Agent(
        model,
        system_prompt=dedent('Generate concise example test cases that comply with the provided JSON schema.'),
        result_type=dataset_type,
        retries=1,
    )
    return (await agent.run(f'Generate {n_examples} examples')).data.rows


def _get_relative_path_reference(target: Path, source: Path, _prefix: str = '') -> Path:
    # Recursively resolve a relative path to target from source, adding '..' as needed.
    # This is useful for creating a relative path reference from a source file to a target file.
    # For example, if source is '/a/b/c.py' and target is '/a/d/e.py', the relative path reference
    # would be '../../d/e.py'.
    if not target.is_absolute():
        target = target.resolve()
    try:
        return Path(f'{_prefix}{Path(target).relative_to(source)}')
    except ValueError:
        return _get_relative_path_reference(target, source.parent, _prefix=f'{_prefix}../')


# TODO: Use this elsewhere in the library to make it easier to reuse
# def infer_name(value: Any, function_frame: types.FrameType | None) -> str | None:
#     """Infer the variable name from the call frame.
#
#     Usage should be `infer_name(value, inspect.currentframe())`.
#     """
#     if function_frame is not None and (parent_frame := function_frame.f_back):  # pragma: no branch
#         for name, item in parent_frame.f_locals.items():
#             if item is value:
#                 return name
#         if parent_frame.f_locals != parent_frame.f_globals:
#             # if we couldn't find the agent in locals and globals are a different dict, try globals
#             for name, item in parent_frame.f_globals.items():
#                 if item is value:
#                     return name
#     return None
