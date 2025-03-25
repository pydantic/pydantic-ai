"""Dataset management for pydantic evals.

This module provides functionality for creating, loading, saving, and evaluating datasets of test cases.
Each case must have inputs, and can optionally have a name, expected output, metadata, and case-specific evaluators.

Datasets can be loaded from and saved to YAML or JSON files, and can be evaluated against
a task function to produce an evaluation report.
"""

from __future__ import annotations as _annotations

import asyncio
import functools
import inspect
import sys
import time
import warnings
from collections.abc import Awaitable, Mapping, Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Union

import logfire_api
import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic._internal import _typing_extra
from pydantic_core import to_json, to_jsonable_python
from typing_extensions import NotRequired, Self, TypedDict, TypeVar

from pydantic_graph._utils import run_until_complete

from ._utils import get_unwrapped_function_name
from .evaluators import EvaluationResult, Evaluator, run_evaluator
from .evaluators._spec import EvaluatorSpec
from .evaluators.common import DEFAULT_EVALUATORS
from .evaluators.context import EvaluatorContext
from .otel import SpanTree
from .otel._context_in_memory_span_exporter import context_subtree
from .reporting import EvaluationReport, ReportCase, ReportCaseAggregate

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup
else:
    ExceptionGroup = ExceptionGroup

# while waiting for https://github.com/pydantic/logfire/issues/745
try:
    import logfire._internal.stack_info
except ImportError:
    pass
else:
    from pathlib import Path

    logfire._internal.stack_info.NON_USER_CODE_PREFIXES += (str(Path(__file__).parent.absolute()),)

_logfire = logfire_api.Logfire(otel_scope='pydantic-evals')

InputsT = TypeVar('InputsT', default=Any)
"""Generic type for the inputs to the task being evaluated."""
OutputT = TypeVar('OutputT', default=Any)
"""Generic type for the expected output of the task being evaluated."""
MetadataT = TypeVar('MetadataT', default=Any)
"""Generic type for the metadata associated with the task being evaluated."""

DEFAULT_DATASET_PATH = './test_cases.yaml'
"""Default path for saving/loading datasets."""
DEFAULT_SCHEMA_PATH_TEMPLATE = './{stem}_schema.json'
"""Default template for schema file paths, where {stem} is replaced with the dataset filename stem."""


class _CaseModel(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid'):
    """Internal model for a case, used for serialization/deserialization."""

    name: str | None = None
    inputs: InputsT
    metadata: MetadataT | None = None
    expected_output: OutputT | None = None
    evaluators: list[EvaluatorSpec] = Field(default_factory=list)


class _DatasetModel(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid'):
    """Internal model for a dataset, used for serialization/deserialization."""

    cases: list[_CaseModel[InputsT, OutputT, MetadataT]]
    evaluators: list[EvaluatorSpec] = Field(default_factory=list)


@dataclass(init=False)
class Case(Generic[InputsT, OutputT, MetadataT]):
    """A single row of a [`Dataset`][pydantic_evals.Dataset].

    Each case represents a single test scenario with inputs to test. A case may optionally specify a name, expected
    outputs to compare against, and arbitrary metadata.

    Cases can also have their own specific evaluators which are run in addition to dataset-level evaluators.

    Example:
    ```python
    case = Case(
        name="Simple addition",
        inputs={"a": 1, "b": 2},
        expected_output=3,
        metadata={"description": "Tests basic addition"}
    )
    ```
    """

    name: str | None
    """Name of the case. This is used to identify the case in the report and can be used to filter cases."""
    inputs: InputsT
    """Inputs to the task. This is the input to the task that will be evaluated."""
    metadata: MetadataT | None
    """Metadata to be used in the evaluation.

    This can be used to provide additional information about the case to the evaluators.
    """
    expected_output: OutputT | None
    """Expected output of the task. This is the expected output of the task that will be evaluated."""
    evaluators: list[Evaluator[InputsT, OutputT, MetadataT]]
    """Evaluators to be used just on this case."""

    def __init__(
        self,
        *,
        name: str | None = None,
        inputs: InputsT,
        metadata: MetadataT | None = None,
        expected_output: OutputT | None = None,
        evaluators: tuple[Evaluator[InputsT, OutputT, MetadataT], ...] = (),
    ):
        """Initialize a new test case.

        Args:
            name: Optional name for the case. If not provided, a generic name will be assigned when added to a dataset.
            inputs: The inputs to the task being evaluated.
            metadata: Optional metadata for the case, which can be used by evaluators.
            expected_output: Optional expected output of the task, used for comparison in evaluators.
            evaluators: Tuple of evaluators specific to this case. These are in addition to any
                dataset-level evaluators.

        """
        # Note: `evaluators` must be a tuple instead of Sequence due to misbehavior with pyright's generic parameter
        # inference if it has type `Sequence`
        self.name = name
        self.inputs = inputs
        self.metadata = metadata
        self.expected_output = expected_output
        self.evaluators = list(evaluators)


# TODO: Consider making one or more of the following changes to this type:
#  * Add `task: Callable[[InputsT], Awaitable[OutputT]` as a field
#  * Add `inputs_type`, `output_type`, etc. as kwargs on `__init__`
#  * Rename to `Evaluation`
# TODO: Allow `task` to be sync _or_ async
class Dataset(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid', arbitrary_types_allowed=True):
    """A dataset of test [cases][pydantic_evals.Case].

    Datasets allow you to organize a collection of test cases and evaluate them against a task function.
    They can be loaded from and saved to YAML or JSON files, and can have dataset-level evaluators that
    apply to all cases.

    Example:
    ```python
    # Create a dataset with two test cases
    dataset = Dataset(
        cases=[
            Case(name="test1", inputs={"text": "Hello"}, expected_output="HELLO"),
            Case(name="test2", inputs={"text": "World"}, expected_output="WORLD"),
        ],
        evaluators=[ExactMatch()]
    )

    # Evaluate the dataset against a task function
    async def uppercase(inputs: dict) -> str:
        return inputs["text"].upper()

    report = await dataset.evaluate(uppercase)
    ```
    """

    cases: list[Case[InputsT, OutputT, MetadataT]]
    """List of test cases in the dataset."""
    evaluators: list[Evaluator[InputsT, OutputT, MetadataT]] = []
    """List of evaluators to be used on all cases in the dataset."""

    def __init__(
        self,
        *,
        cases: Sequence[Case[InputsT, OutputT, MetadataT]],
        evaluators: Sequence[Evaluator[InputsT, OutputT, MetadataT]] = (),
    ):
        """Initialize a new dataset with test cases and optional evaluators.

        Args:
            cases: Sequence of test cases to include in the dataset.
            evaluators: Optional sequence of evaluators to apply to all cases in the dataset.
        """
        case_names = set[str]()
        for case in cases:
            if case.name is None:
                continue
            if case.name in case_names:
                raise ValueError(f'Duplicate case name: {case.name}')
            case_names.add(case.name)

        super().__init__(
            cases=cases,
            evaluators=list(evaluators),
        )

    async def evaluate(
        self, task: Callable[[InputsT], Awaitable[OutputT]], name: str | None = None, max_concurrency: int | None = None
    ) -> EvaluationReport:
        """Evaluates the test cases in the dataset using the given task.

        This method runs the task on each case in the dataset, applies evaluators,
        and collects results into a report. Cases are run concurrently, limited by `max_concurrency` if specified.

        Args:
            task: The task to evaluate. This should be a callable that takes the inputs of the case
                and returns the output.
            name: The name of the task being evaluated, this is used to identify the task in the report.
                If omitted, the name of the task function will be used.
            max_concurrency: The maximum number of concurrent evaluations of the task to allow.
                If None, all cases will be evaluated concurrently.

        Returns:
            A report containing the results of the evaluation.
        """
        name = name or get_unwrapped_function_name(task)

        limiter = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else nullcontext()
        with _logfire.span('evaluate {name}', name=name) as eval_span:

            async def _handle_case(case: Case[InputsT, OutputT, MetadataT], report_case_name: str) -> ReportCase:
                async with limiter:
                    return await _run_task_and_evaluators(task, case, report_case_name, self.evaluators)

            async_tasks: list[asyncio.Task[ReportCase]] = []
            async with asyncio.TaskGroup() as group:
                for i, case in enumerate(self.cases, 1):
                    async_tasks.append(group.create_task(_handle_case(case, case.name or f'Case {i}')))

            report = EvaluationReport(name=name, cases=[x.result() for x in async_tasks])
            # TODO(DavidM): This attribute will be too big in general; remove it once we can use child spans in details panel:
            eval_span.set_attribute('cases', report.cases)
            # TODO(DavidM): Remove this 'averages' attribute once we compute it in the details panel
            eval_span.set_attribute('averages', ReportCaseAggregate.average(report.cases))
        return report

    def evaluate_sync(
        self, task: Callable[[InputsT], Awaitable[OutputT]], name: str | None = None, max_concurrency: int | None = None
    ) -> EvaluationReport:
        """Evaluates the test cases in the dataset using the given task.

        This is a synchronous wrapper around [`evaluate`][pydantic_evals.Dataset.evaluate] provided for convenience.

        Args:
            task: The task to evaluate. This should be a callable that takes the inputs of the case
                and returns the output.
            name: The name of the task being evaluated, this is used to identify the task in the report.
                If omitted, the name of the task function will be used.
            max_concurrency: The maximum number of concurrent evaluations of the task to allow.
                If None, all cases will be evaluated concurrently.

        Returns:
            A report containing the results of the evaluation.
        """
        return run_until_complete(self.evaluate(task, name=name, max_concurrency=max_concurrency))

    def add_case(
        self,
        *,
        name: str | None = None,
        inputs: InputsT,
        metadata: MetadataT | None = None,
        expected_output: OutputT | None = None,
        evaluators: tuple[Evaluator[InputsT, OutputT, MetadataT], ...] = (),
    ) -> None:
        """Adds a case to the dataset.

        This is a convenience method for creating a [`Case`][pydantic_evals.Case] and adding it to the dataset.

        Args:
            name: Optional name for the case. If not provided, a generic name will be assigned.
            inputs: The inputs to the task being evaluated.
            metadata: Optional metadata for the case, which can be used by evaluators.
            expected_output: The expected output of the task, used for comparison in evaluators.
            evaluators: Tuple of evaluators specific to this case, in addition to dataset-level evaluators.
        """
        case = Case[InputsT, OutputT, MetadataT](
            name=name,
            inputs=inputs,
            metadata=metadata,
            expected_output=expected_output,
            evaluators=evaluators,
        )
        self.cases.append(case)

    def add_evaluator(
        self,
        evaluator: Evaluator[InputsT, OutputT, MetadataT],
        specific_case: str | None = None,
    ) -> None:
        """Adds an evaluator to the dataset or a specific case.

        Args:
            evaluator: The evaluator to add.
            specific_case: If provided, the evaluator will only be added to the case with this name.
                If None, the evaluator will be added to all cases in the dataset.

        Raises:
            ValueError: If `specific_case` is provided but no case with that name exists in the dataset.
        """
        if specific_case is None:
            self.evaluators.append(evaluator)
        else:
            # If this is too slow, we could try to add a case lookup dict.
            # Note that if we do that, we'd need to make the cases list private to prevent modification.
            added = False
            for case in self.cases:
                if case.name == specific_case:
                    case.evaluators.append(evaluator)
                    added = True
            if not added:
                raise ValueError(f'Case {specific_case!r} not found in the dataset')

    @classmethod
    @functools.cache
    def _params(cls) -> tuple[type[InputsT], type[OutputT], type[MetadataT]]:
        """Get the type parameters for the Dataset class.

        Returns:
            A tuple of (InputsT, OutputT, MetadataT) types.
        """
        for c in cls.__mro__:
            metadata = getattr(c, '__pydantic_generic_metadata__', {})
            if len(args := (metadata.get('args', ()) or getattr(c, '__args__', ()))) == 3:
                return args
        warnings.warn(
            f'Could not determine the generic parameters for {cls}; using `Any` for each. '
            f'You should explicitly set the generic parameters via `Dataset[MyInputs, MyOutput, MyMetadata]`'
            f'when serializing or deserializing.',
            UserWarning,
        )
        return Any, Any, Any  # type: ignore

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        fmt: Literal['yaml', 'json'] | None = None,
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    ) -> Self:
        """Load a dataset from a file.

        Args:
            path: Path to the file to load.
            fmt: Format of the file. If None, the format will be inferred from the file extension.
                Must be either 'yaml' or 'json'.
            custom_evaluator_types: Custom evaluator classes to use when deserializing the dataset.
                These are additional evaluators beyond the default ones.

        Returns:
            A new Dataset instance loaded from the file.

        Raises:
            ValidationError: If the file cannot be parsed as a valid dataset.
            ValueError: If the format cannot be inferred from the file extension.
        """
        path = Path(path)
        fmt = cls._infer_fmt(path, fmt)

        raw = Path(path).read_text()
        try:
            return cls.from_text(raw, fmt=fmt, custom_evaluator_types=custom_evaluator_types)
        except ValidationError as e:
            raise ValueError(f'{path} contains data that does not match the schema for {cls.__name__}:\n{e}.') from e

    @classmethod
    def from_text(
        cls,
        contents: str,
        fmt: Literal['yaml', 'json'] = 'yaml',
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    ) -> Self:
        """Load a dataset from a string.

        Args:
            contents: The string content to parse.
            fmt: Format of the content. Must be either 'yaml' or 'json'.
            custom_evaluator_types: Custom evaluator classes to use when deserializing the dataset.
                These are additional evaluators beyond the default ones.

        Returns:
            A new Dataset instance parsed from the string.

        Raises:
            ValidationError: If the content cannot be parsed as a valid dataset.
        """
        if fmt == 'yaml':
            loaded = yaml.safe_load(contents)
            return cls.from_dict(loaded, custom_evaluator_types)
        else:
            dataset_model_type = cls._serialization_type()
            dataset_model = dataset_model_type.model_validate_json(contents)
            return cls._from_dataset_model(dataset_model, custom_evaluator_types)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    ) -> Self:
        """Load a dataset from a dictionary.

        Args:
            data: Dictionary representation of the dataset.
            custom_evaluator_types: Custom evaluator classes to use when deserializing the dataset.
                These are additional evaluators beyond the default ones.

        Returns:
            A new Dataset instance created from the dictionary.

        Raises:
            ValidationError: If the dictionary cannot be converted to a valid dataset.
        """
        dataset_model_type = cls._serialization_type()
        dataset_model = dataset_model_type.model_validate(data)
        return cls._from_dataset_model(dataset_model, custom_evaluator_types)

    @classmethod
    def _from_dataset_model(
        cls,
        dataset_model: _DatasetModel[InputsT, OutputT, MetadataT],
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    ) -> Self:
        """Create a Dataset from a _DatasetModel.

        Args:
            dataset_model: The _DatasetModel to convert.
            custom_evaluator_types: Custom evaluator classes to register for deserialization.

        Returns:
            A new Dataset instance created from the _DatasetModel.
        """
        registry = _get_registry(custom_evaluator_types)

        cases: list[Case[InputsT, OutputT, MetadataT]] = []
        errors: list[ValueError] = []
        dataset_evaluators: list[Evaluator[Any, Any, Any]] = []
        for spec in dataset_model.evaluators:
            try:
                dataset_evaluator = _load_evaluator_from_registry(registry, None, spec)
            except ValueError as e:
                errors.append(e)
                continue
            dataset_evaluators.append(dataset_evaluator)

        for row in dataset_model.cases:
            evaluators: list[Evaluator[Any, Any, Any]] = []
            for spec in row.evaluators:
                try:
                    evaluator = _load_evaluator_from_registry(registry, row.name, spec)
                except ValueError as e:
                    errors.append(e)
                    continue
                evaluators.append(evaluator)
            row = Case[InputsT, OutputT, MetadataT](
                name=row.name,
                inputs=row.inputs,
                metadata=row.metadata,
                expected_output=row.expected_output,
            )
            row.evaluators = evaluators
            cases.append(row)
        if errors:
            raise ExceptionGroup(f'{len(errors)} error(s) loading evaluators from registry', errors[:3])
        result = cls(cases=cases)
        result.evaluators = dataset_evaluators
        return result

    def to_file(
        self,
        path: Path | str,
        fmt: Literal['yaml', 'json'] | None = None,
        schema_path: Path | str | None = DEFAULT_SCHEMA_PATH_TEMPLATE,
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    ):
        """Save the dataset to a file.

        Args:
            path: Path to save the dataset to.
            fmt: Format to use. If None, the format will be inferred from the file extension.
                Must be either 'yaml' or 'json'.
            schema_path: Path to save the JSON schema to. If None, no schema will be saved.
                Can be a string template with {stem} which will be replaced with the dataset filename stem.
            custom_evaluator_types: Custom evaluator classes to include in the schema.
        """
        path = Path(path)
        fmt = self._infer_fmt(path, fmt)

        schema_ref: str | None = None
        if schema_path is not None:
            if isinstance(schema_path, str):
                schema_path = Path(schema_path.format(stem=path.stem))

            if not schema_path.is_absolute():
                schema_ref = str(schema_path)
                schema_path = path.parent / schema_path
            elif schema_path.is_relative_to(path):
                schema_ref = str(_get_relative_path_reference(schema_path, path))
            else:
                schema_ref = str(schema_path)
            self._save_schema(schema_path, custom_evaluator_types)

        if fmt == 'yaml':
            dumped_data = self.model_dump(mode='json', exclude_defaults=True, context={'use_short_form': True})
            content = yaml.dump(dumped_data, sort_keys=False)
            if schema_ref:
                yaml_language_server_line = f'# yaml-language-server: $schema={schema_ref}'
                content = f'{yaml_language_server_line}\n{content}'
            path.write_text(content)
        else:
            json_data = self.model_dump_json(indent=2, exclude_defaults=True, context={'use_short_form': True})
            path.write_text(json_data + '\n')

    @classmethod
    def model_json_schema_with_evaluators(
        cls,
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    ) -> dict[str, Any]:
        """Generate a JSON schema for this dataset type, including evaluator details.

        This is useful for generating a schema that can be used to validate YAML-format dataset files.

        Args:
            custom_evaluator_types: Custom evaluator classes to include in the schema.

        Returns:
            A dictionary representing the JSON schema.
        """
        # Note: this function could maybe be simplified now that Evaluators are always dataclasses
        registry = _get_registry(custom_evaluator_types)

        evaluator_schema_types: list[Any] = []
        for name, evaluator_class in registry.items():
            type_hints = _typing_extra.get_function_type_hints(evaluator_class)
            type_hints.pop('return', None)
            required_type_hints: dict[str, Any] = {}

            for p in inspect.signature(evaluator_class).parameters.values():
                type_hints.setdefault(p.name, Any)
                if p.default is not p.empty:
                    type_hints[p.name] = NotRequired[type_hints[p.name]]
                else:
                    required_type_hints[p.name] = type_hints[p.name]

            if len(type_hints) == 0 or not required_type_hints:
                # Shortest option: just the call name
                evaluator_schema_types.append(Literal[name])
            if len(type_hints) == 1:
                # Short option: only have one parameter, so we can drop the nesting
                [type_hint_type] = type_hints.values()  # pyright: ignore
                td = TypedDict(f'short_evaluator_{name}', {name: type_hint_type})  # pyright: ignore
                td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                evaluator_schema_types.append(td)
            if len(type_hints) > 1:
                if len(required_type_hints) == 1:
                    # Short option: only have one required parameter, so we can drop the nesting
                    type_hint_type = next(iter(required_type_hints.values()))  # pyright: ignore
                    td = TypedDict(f'short_evaluator_{name}', {name: type_hint_type})  # pyright: ignore
                    td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                    evaluator_schema_types.append(td)

                # Long form: multiple parameters, or multiple required parameters
                params_td = TypedDict(f'evaluator_params_{name}', type_hints)  # pyright: ignore
                params_td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                td = TypedDict(f'evaluator_{name}', {name: params_td})  # pyright: ignore
                td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                evaluator_schema_types.append(td)
            # Note: We might want to also generate the JSON schema for the format `call: '...', args: [...], kwargs: {...}`.
            #   It would be a bit complex to implement but not impossible.

        in_type, out_type, meta_type = cls._params()

        class ClsDatasetRow(BaseModel, extra='forbid'):
            name: str
            inputs: in_type
            metadata: meta_type
            expected_output: out_type | None = None
            if evaluator_schema_types:
                evaluators: list[Union[tuple(evaluator_schema_types)]] = []  # pyright: ignore  # noqa UP007

        ClsDatasetRow.__name__ = cls.__name__ + 'Row'

        class ClsDataset(BaseModel, extra='forbid'):
            cases: list[ClsDatasetRow]
            if evaluator_schema_types:
                evaluators: list[Union[tuple(evaluator_schema_types)]] = []  # pyright: ignore  # noqa UP007

        ClsDataset.__name__ = cls.__name__

        return ClsDataset.model_json_schema()

    @classmethod
    def _save_schema(
        cls, path: Path | str, custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = ()
    ):
        """Save the JSON schema for this dataset type to a file.

        Args:
            path: Path to save the schema to.
            custom_evaluator_types: Custom evaluator classes to include in the schema.
        """
        path = Path(path)
        json_schema = cls.model_json_schema_with_evaluators(custom_evaluator_types)
        schema_content = to_json(json_schema, indent=2).decode() + '\n'
        if not path.exists() or path.read_text() != schema_content:
            path.write_text(schema_content)

    @classmethod
    @functools.cache
    def _serialization_type(cls) -> type[_DatasetModel[InputsT, OutputT, MetadataT]]:
        """Get the serialization type for this dataset class.

        Returns:
            A _DatasetModel type with the same generic parameters as this Dataset class.
        """
        input_type, output_type, metadata_type = cls._params()
        return _DatasetModel[input_type, output_type, metadata_type]

    @classmethod
    def _infer_fmt(cls, path: Path, fmt: Literal['yaml', 'json'] | None) -> Literal['yaml', 'json']:
        """Infer the format to use for a file based on its extension.

        Args:
            path: The path to infer the format for.
            fmt: The explicitly provided format, if any.

        Returns:
            The inferred format ('yaml' or 'json').

        Raises:
            ValueError: If the format cannot be inferred from the file extension.
        """
        if fmt is not None:
            return fmt
        suffix = path.suffix.lower()
        if suffix in {'.yaml', '.yml'}:
            return 'yaml'
        elif suffix == '.json':
            return 'json'
        raise ValueError(f'Could not infer format from path {path}. Use the `fmt` argument to specify the format.')


def _get_relative_path_reference(target: Path, source: Path, _prefix: str = '') -> Path:
    """Get a relative path reference from source to target.

    Recursively resolve a relative path to target from source, adding '..' as needed.
    This is useful for creating a relative path reference from a source file to a target file.

    Args:
        target: The target path to reference.
        source: The source path to reference from.
        _prefix: Internal prefix used during recursion.

    Returns:
        A Path object representing the relative path from source to target.

    Example:
        If source is '/a/b/c.py' and target is '/a/d/e.py', the relative path reference
        would be '../../d/e.py'.
    """
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


@dataclass
class _TaskRun:
    """Internal class to track metrics and attributes for a task run."""

    attributes: dict[str, Any] = field(init=False, default_factory=dict)
    metrics: dict[str, int | float] = field(init=False, default_factory=dict)

    def record_metric(self, name: str, value: int | float) -> None:
        """Record a metric value.

        Args:
            name: The name of the metric.
            value: The value of the metric.
        """
        self.metrics[name] = value

    def increment_metric(self, name: str, amount: int | float) -> None:
        """Increment a metric value.

        Args:
            name: The name of the metric.
            amount: The amount to increment by.

        Note:
            If the current value is 0 and the increment amount is 0, no metric will be recorded.
        """
        current_value = self.metrics.get(name, 0)
        incremented_value = current_value + amount
        if current_value == 0 and incremented_value == 0:
            return  # Avoid recording a metric that is always zero
        self.record_metric(name, incremented_value)

    def record_attribute(self, name: str, value: Any) -> None:
        """Record an attribute value.

        Args:
            name: The name of the attribute.
            value: The value of the attribute.
        """
        self.attributes[name] = value


async def _run_task(
    task: Callable[[InputsT], Awaitable[OutputT]], case: Case[InputsT, OutputT, MetadataT]
) -> EvaluatorContext[InputsT, OutputT, MetadataT]:
    """Run a task on a case and return the context for evaluators.

    Args:
        task: The task to run.
        case: The case to run the task on.

    Returns:
        An EvaluatorContext containing the inputs, actual output, expected output, and metadata.

    Raises:
        Exception: Any exception raised by the task.
    """
    task_run = _TaskRun()
    if _CURRENT_TASK_RUN.get() is not None:
        raise RuntimeError('A task run has already been entered. Task runs should not be nested')

    # TODO: Should we handle task execution errors in some way? Right now they bubble up immediately
    token = _CURRENT_TASK_RUN.set(task_run)
    try:
        with _logfire.span('execute {task}', task=get_unwrapped_function_name(task)) as task_span:
            with context_subtree() as span_tree:
                t0 = time.time()
                task_output = await task(case.inputs)
                fallback_duration = time.time() - t0
    finally:
        _CURRENT_TASK_RUN.reset(token)

    if isinstance(span_tree, SpanTree):
        # TODO: Question: Should we make this metric-attributes functionality more user-configurable in some way before merging?
        #   Note: the use of otel for collecting these metrics is the main reason why I think we should require at least otel as a dependency, if not logfire;
        #   otherwise, we don't have a great way to get usage data from arbitrary frameworks.
        #   Ideally we wouldn't need to hard-code the specific logic here, but I'm not sure a great way to expose it to
        #   users. Maybe via an argument of type Callable[[SpanTree], dict[str, int | float]] or similar?
        for node in span_tree.flattened():
            if node.attributes.get('gen_ai.operation.name') == 'chat':
                task_run.increment_metric('requests', 1)
            for k, v in node.attributes.items():
                if not isinstance(v, (int, float)):
                    continue
                if k.startswith('gen_ai.usage.details.'):
                    task_run.increment_metric(k[21:], v)
                if k.startswith('gen_ai.usage.'):
                    task_run.increment_metric(k[13:], v)

    return EvaluatorContext[InputsT, OutputT, MetadataT](
        name=case.name,
        inputs=case.inputs,
        metadata=case.metadata,
        expected_output=case.expected_output,
        output=task_output,
        duration=_get_span_duration(task_span, fallback_duration),
        _span_tree=span_tree,
        attributes=task_run.attributes,
        metrics=task_run.metrics,
    )


async def _run_task_and_evaluators(
    task: Callable[[InputsT], Awaitable[OutputT]],
    case: Case[InputsT, OutputT, MetadataT],
    report_case_name: str,
    dataset_evaluators: list[Evaluator[InputsT, OutputT, MetadataT]],
) -> ReportCase:
    """Run a task on a case and evaluate the results.

    Args:
        task: The task to run.
        case: The case to run the task on.
        report_case_name: The name to use for this case in the report.
        dataset_evaluators: Evaluators from the dataset to apply to this case.

    Returns:
        A ReportCase containing the evaluation results.
    """
    with _logfire.span(
        '{task_name}: {case_name}',
        task_name=get_unwrapped_function_name(task),
        case_name=case.name,
        inputs=case.inputs,
        metadata=case.metadata,
    ) as case_span:
        t0 = time.time()
        scoring_context = await _run_task(task, case)

        case_span.set_attribute('output', scoring_context.output)
        case_span.set_attribute('task_duration', scoring_context.duration)
        case_span.set_attribute('metrics', scoring_context.metrics)
        case_span.set_attribute('attributes', scoring_context.attributes)

        evaluators = case.evaluators + dataset_evaluators
        evaluator_outputs: list[EvaluationResult] = []
        if evaluators:
            async with asyncio.TaskGroup() as tg:
                tasks: list[asyncio.Task[list[EvaluationResult]]] = []
                for evaluator in evaluators:
                    tasks.append(tg.create_task(run_evaluator(evaluator, scoring_context)))
            for t in tasks:
                evaluator_outputs.extend(t.result())

        assertions, scores, labels = _group_evaluator_outputs_by_type(evaluator_outputs)
        case_span.set_attribute('assertions', _evaluation_results_adapter.dump_python(assertions))
        case_span.set_attribute('scores', _evaluation_results_adapter.dump_python(scores))
        case_span.set_attribute('labels', _evaluation_results_adapter.dump_python(labels))

        context = case_span.context
        if context is None:
            trace_id = ''
            span_id = ''
        else:
            trace_id = f'{context.trace_id:032x}'
            span_id = f'{context.span_id:016x}'
        fallback_duration = time.time() - t0

    report_inputs = to_jsonable_python(case.inputs)

    return ReportCase(
        name=report_case_name,
        inputs=report_inputs,
        metadata=case.metadata,
        expected_output=case.expected_output,
        output=scoring_context.output,
        metrics=scoring_context.metrics,
        attributes=scoring_context.attributes,
        scores=scores,
        labels=labels,
        assertions=assertions,
        task_duration=scoring_context.duration,
        total_duration=_get_span_duration(case_span, fallback_duration),
        trace_id=trace_id,
        span_id=span_id,
    )


_evaluation_results_adapter = TypeAdapter(Mapping[str, EvaluationResult])


def _group_evaluator_outputs_by_type(
    evaluation_results: Sequence[EvaluationResult],
) -> tuple[
    dict[str, EvaluationResult[bool]],
    dict[str, EvaluationResult[int | float]],
    dict[str, EvaluationResult[str]],
]:
    """Group evaluator outputs by their result type.

    Args:
        evaluation_results: Sequence of evaluation results to group.

    Returns:
        A tuple of dictionaries mapping evaluator names to their results, grouped by result type:
        (success_evaluations, metric_evaluations, string_evaluations)
    """
    assertions: dict[str, EvaluationResult[bool]] = {}
    scores: dict[str, EvaluationResult[int | float]] = {}
    labels: dict[str, EvaluationResult[str]] = {}
    seen_names = set[str]()
    for er in evaluation_results:
        name = er.name
        # Dedupe repeated names by adding a numeric suffix
        if name in seen_names:
            suffix = 2
            while f'{name}_{suffix}' in seen_names:
                suffix += 1
            name = f'{name}_{suffix}'
        seen_names.add(name)
        if assertion := er.downcast(bool):
            assertions[name] = assertion
        elif score := er.downcast(int, float):
            scores[name] = score
        elif label := er.downcast(str):
            labels[name] = label
    return assertions, scores, labels


_CURRENT_TASK_RUN = ContextVar['_TaskRun | None']('_CURRENT_TASK_RUN', default=None)


def set_eval_attribute(name: str, value: Any) -> None:
    """Set an attribute on the current task run.

    Args:
        name: The name of the attribute.
        value: The value of the attribute.
    """
    current_case = _CURRENT_TASK_RUN.get()
    if current_case is not None:
        current_case.record_attribute(name, value)


def increment_eval_metric(name: str, amount: int | float) -> None:
    """Increment a metric on the current task run.

    Args:
        name: The name of the metric.
        amount: The amount to increment by.
    """
    current_case = _CURRENT_TASK_RUN.get()
    if current_case is not None:
        current_case.increment_metric(name, amount)


def _get_span_duration(span: logfire_api.LogfireSpan, fallback: float) -> float:
    """Calculate the duration of a span in seconds.

    We prefer to obtain the duration from a span for the sake of consistency with observability and to make
    the values more reliable during testing. However, if the span is not available (e.g. when using logfire_api
    without logfire installed), we fall back to the provided duration.

    Args:
        span: The span to calculate the duration for.
        fallback: The fallback duration to use if unable to obtain the duration from the span.

    Returns:
        The duration of the span in seconds.
    """
    try:
        return (span.end_time - span.start_time) / 1_000_000_000  # type: ignore
    except (AttributeError, TypeError):  #
        return fallback


def _get_registry(
    custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]],
) -> Mapping[str, type[Evaluator[InputsT, OutputT, MetadataT]]]:
    """Create a registry of evaluator types from default and custom evaluators.

    Args:
        custom_evaluator_types: Additional evaluator classes to include in the registry.

    Returns:
        A mapping from evaluator names to evaluator classes.
    """
    registry: dict[str, type[Evaluator[InputsT, OutputT, MetadataT]]] = {}

    for evaluator_class in custom_evaluator_types:
        if not issubclass(evaluator_class, Evaluator):
            raise ValueError(
                f'All custom evaluator classes must be subclasses of Evaluator, but {evaluator_class} is not'
            )
        if '__dataclass_fields__' not in evaluator_class.__dict__:
            raise ValueError(f'All custom evaluator must be decorated with `@dataclass`, but {evaluator_class} is not')
        name = evaluator_class.name()
        if name in registry:
            raise ValueError(f'Duplicate evaluator class name: {name}')
        registry[name] = evaluator_class

    for evaluator_class in DEFAULT_EVALUATORS:
        # Allow overriding the default evaluators with custom evaluators raising an error
        registry.setdefault(evaluator_class.name(), evaluator_class)

    return registry


def _load_evaluator_from_registry(
    registry: Mapping[str, type[Evaluator[InputsT, OutputT, MetadataT]]],
    case_name: str | None,
    spec: EvaluatorSpec,
) -> Evaluator[InputsT, OutputT, MetadataT]:
    """Load an evaluator from the registry based on a specification.

    Args:
        registry: Mapping from evaluator names to evaluator classes.
        case_name: Name of the case this evaluator will be used for, or None for dataset-level evaluators.
        spec: Specification of the evaluator to load.

    Returns:
        An initialized evaluator instance.

    Raises:
        ValueError: If the evaluator name is not found in the registry.
    """
    evaluator_class = registry.get(spec.name)
    if evaluator_class is None:
        raise ValueError(
            f'Evaluator {spec.name!r} is not in the provided registry. Registered choices: {list(registry.keys())}'
        )
    try:
        return evaluator_class(*spec.args, **spec.kwargs)
    except Exception as e:
        case_detail = f'case {case_name!r}' if case_name is not None else 'dataset'
        raise ValueError(f'Failed to instantiate evaluator {spec.name!r} for {case_detail}: {e}') from e
