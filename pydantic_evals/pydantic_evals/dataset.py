from __future__ import annotations as _annotations

import asyncio
import functools
import inspect
import sys
import warnings
from collections.abc import Awaitable, Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, NotRequired, Self, Union

import logfire
import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic._internal import _typing_extra
from pydantic_core import to_json, to_jsonable_python
from typing_extensions import TypedDict, TypeVar

from pydantic_graph._utils import run_until_complete

from ._utils import get_unwrapped_function_name
from .evaluators.common import DEFAULT_EVALUATORS
from .evaluators.context import EvaluatorContext
from .evaluators.spec import Evaluator, EvaluatorDetails, EvaluatorFunction, EvaluatorSpec, SourcedEvaluatorOutput
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

_logfire = logfire.Logfire(otel_scope='pydantic-evals')

InputsT = TypeVar('InputsT', default=Any)
"""Generic type for the inputs to the task being evaluated."""
OutputT = TypeVar('OutputT', default=Any)
"""Generic type for the expected output of the task being evaluated."""
MetadataT = TypeVar('MetadataT', default=Any)
"""Generic type for the metadata associated with the task being evaluated."""

DEFAULT_DATASET_PATH = './test_cases.yaml'
DEFAULT_SCHEMA_PATH_TEMPLATE = './{stem}_schema.json'


class _CaseModel(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid'):
    name: str | None = None
    inputs: InputsT
    metadata: MetadataT | None = None
    expected_output: OutputT | None = None
    evaluators: list[EvaluatorSpec] = Field(default_factory=list)


class _DatasetModel(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid'):
    cases: list[_CaseModel[InputsT, OutputT, MetadataT]]
    evaluators: list[EvaluatorSpec] = Field(default_factory=list)


@dataclass(init=False)
class Case(Generic[InputsT, OutputT, MetadataT]):
    """A single row of a [`Dataset`][pydantic_evals.Dataset], consisting of input, expected output, and metadata."""

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
    evaluators: list[EvaluatorDetails[InputsT, OutputT, MetadataT]]
    """Evaluators to be used just on this case."""

    def __init__(
        self,
        *,
        name: str | None = None,
        inputs: InputsT,
        metadata: MetadataT | None = None,
        expected_output: OutputT | None = None,
        evaluators: tuple[
            Evaluator[InputsT, OutputT, MetadataT] | EvaluatorDetails[InputsT, OutputT, MetadataT], ...
        ] = (),
    ):
        # Note: `evaluators` must be a tuple instead of Sequence due to misbehavior with pyright's generic parameter
        # inference if it has type `Sequence`
        self.name = name
        self.inputs = inputs
        self.metadata = metadata
        self.expected_output = expected_output
        self.evaluators = [
            e if isinstance(e, EvaluatorDetails) else EvaluatorDetails[InputsT, OutputT, MetadataT].from_function(e)
            for e in evaluators
        ]


# TODO: Consider making the following changes to this type:
#  * Add `task: Callable[[InputsT], Awaitable[OutputT]` as a field
#  * Add `inputs_type`, `output_type`, etc. as kwargs on `__init__`
#  * Rename to `Evaluation`
#  * Allow `task` to be sync _or_ async
class Dataset(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid', arbitrary_types_allowed=True):
    """A dataset of test [cases][pydantic_evals.Case]."""

    cases: list[Case[InputsT, OutputT, MetadataT]]
    """List of test cases in the dataset."""
    evaluators: list[EvaluatorDetails[InputsT, OutputT, MetadataT]] = []
    """List of evaluators to be used on all cases in the dataset."""

    def __init__(
        self,
        *,
        cases: Sequence[Case[InputsT, OutputT, MetadataT]],
        evaluators: Sequence[
            Evaluator[InputsT, OutputT, MetadataT] | EvaluatorDetails[InputsT, OutputT, MetadataT]
        ] = (),
    ):
        super().__init__(
            cases=cases,
            evaluators=[
                a if isinstance(a, EvaluatorDetails) else EvaluatorDetails[InputsT, OutputT, MetadataT].from_function(a)
                for a in evaluators
            ],
        )

    async def evaluate(
        self, task: Callable[[InputsT], Awaitable[OutputT]], name: str | None = None, max_concurrency: int | None = None
    ) -> EvaluationReport:
        """Evaluates the test cases in the dataset using the given task.

        Args:
            task: The task to evaluate. This should be a callable that takes the inputs of the case
                and returns the output.
            name: The name of the task being evaluated, this is used to identify the task in the report.
                If omitted, the name of the task function will be used.
            max_concurrency: The maximum number of concurrent evaluations of the task to allow.

        Returns: A report containing the results of the evaluation.
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

        This is just a synchronous wrapper around `evaluate` provided for convenience.
        """
        return run_until_complete(self.evaluate(task, name=name, max_concurrency=max_concurrency))

    def add_case(
        self,
        *,
        name: str | None = None,
        inputs: InputsT,
        metadata: MetadataT | None = None,
        expected_output: OutputT | None = None,
        evaluators: tuple[
            Evaluator[InputsT, OutputT, MetadataT] | EvaluatorDetails[InputsT, OutputT, MetadataT], ...
        ] = (),
    ) -> None:
        """Adds a case to the evaluation."""
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
        evaluator: Evaluator[InputsT, OutputT, MetadataT] | EvaluatorDetails[InputsT, OutputT, MetadataT],
        specific_case: str | None = None,
    ) -> None:
        """Adds an evaluator to the dataset or to a specific case.

        Args:
            evaluator: The evaluator to add, TODO.
            specific_case: Specific case to add the evaluator to. If `None`, the evaluator is added to all cases.
        """
        evaluator = (
            evaluator
            if isinstance(evaluator, EvaluatorDetails)
            else EvaluatorDetails[InputsT, OutputT, MetadataT].from_function(evaluator)
        )
        if specific_case is None:
            # Add the evaluator to the dataset itself
            self.evaluators.append(evaluator)
            return

        # Add the evaluator to rows with the given name (generally there should only be one), but error if none found
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
        custom_evaluators: Sequence[EvaluatorFunction[InputsT, OutputT, MetadataT]] = (),
    ) -> Self:
        path = Path(path)
        fmt = cls._infer_fmt(path, fmt)

        raw = Path(path).read_text()
        try:
            return cls.from_text(raw, fmt=fmt, custom_evaluators=custom_evaluators)
        except ValidationError as e:
            raise ValueError(f'{path} contains data that does not match the schema for {cls.__name__}:\n{e}.') from e

    @classmethod
    def from_text(
        cls,
        contents: str,
        fmt: Literal['yaml', 'json'] = 'yaml',
        custom_evaluators: Sequence[EvaluatorFunction[InputsT, OutputT, MetadataT]] = (),
    ) -> Self:
        if fmt == 'yaml':
            loaded = yaml.safe_load(contents)
            return cls.from_dict(loaded, custom_evaluators)
        else:
            dataset_model_type = cls._serialization_type()
            dataset_model = dataset_model_type.model_validate_json(contents)
            return cls._from_dataset_model(dataset_model, custom_evaluators)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        custom_evaluators: Sequence[EvaluatorFunction[InputsT, OutputT, MetadataT]] = (),
    ) -> Self:
        dataset_model_type = cls._serialization_type()
        dataset_model = dataset_model_type.model_validate(data)
        return cls._from_dataset_model(dataset_model, custom_evaluators)

    @classmethod
    def _from_dataset_model(
        cls,
        dataset_model: _DatasetModel[InputsT, OutputT, MetadataT],
        custom_evaluators: Sequence[EvaluatorFunction[InputsT, OutputT, MetadataT]] = (),
    ) -> Self:
        registry = _get_registry(custom_evaluators)

        cases: list[Case[InputsT, OutputT, MetadataT]] = []
        errors: list[ValueError] = []
        dataset_evaluators: list[EvaluatorDetails[Any, Any, Any]] = []
        for evaluator in dataset_model.evaluators:
            try:
                dataset_evaluator = EvaluatorDetails[InputsT, OutputT, MetadataT].from_registry(
                    registry, None, evaluator
                )
            except ValueError as e:
                errors.append(e)
                continue
            dataset_evaluators.append(dataset_evaluator)

        for row in dataset_model.cases:
            evaluators: list[EvaluatorDetails[Any, Any, Any]] = []
            for spec in row.evaluators:
                try:
                    evaluator = EvaluatorDetails[InputsT, OutputT, MetadataT].from_registry(registry, row.name, spec)
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
        custom_evaluators: Sequence[EvaluatorFunction[InputsT, OutputT, MetadataT]] = (),
    ):
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
            self._save_schema(schema_path, custom_evaluators)

        if fmt == 'yaml':
            dumped_data = self.model_dump(context={'use_short_forms': True}, mode='json', exclude_defaults=True)
            content = yaml.dump(dumped_data, sort_keys=False)
            if schema_ref:
                yaml_language_server_line = f'# yaml-language-server: $schema={schema_ref}'
                content = f'{yaml_language_server_line}\n{content}'
            path.write_text(content)
        else:
            json_data = self.model_dump_json(context={'use_short_forms': True}, indent=2, exclude_defaults=True)
            path.write_text(json_data + '\n')

    @classmethod
    def model_json_schema_with_evaluators(
        cls,
        custom_evaluators: Sequence[EvaluatorFunction[InputsT, OutputT, MetadataT]] = (),
    ) -> dict[str, Any]:
        registry = _get_registry(custom_evaluators)

        evaluator_types: list[Any] = []
        for name, function in registry.items():
            signature = inspect.signature(function)

            scoring_context_param, *other_params = signature.parameters.values()
            type_hints = _typing_extra.get_function_type_hints(function)
            type_hints.pop(scoring_context_param.name, None)
            type_hints.pop('return', None)
            required_type_hints: dict[str, Any] = {}

            for p in other_params:
                type_hints.setdefault(p.name, Any)
                if p.default is not p.empty:
                    type_hints[p.name] = NotRequired[type_hints[p.name]]
                else:
                    required_type_hints[p.name] = type_hints[p.name]

            if len(type_hints) == 0 or not required_type_hints:
                # Shortest option: just the call name
                evaluator_types.append(Literal[name])
            if len(type_hints) == 1:
                # Short option: only have one parameter, so we can drop the nesting
                [type_hint_type] = type_hints.values()  # pyright: ignore
                td = TypedDict(f'short_evaluator_{name}', {name: type_hint_type})  # pyright: ignore
                td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                evaluator_types.append(td)
            if len(type_hints) > 1:
                if len(required_type_hints) == 1:
                    # Short option: only have one required parameter, so we can drop the nesting
                    type_hint_type = next(iter(required_type_hints.values()))  # pyright: ignore
                    td = TypedDict(f'short_evaluator_{name}', {name: type_hint_type})  # pyright: ignore
                    td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                    evaluator_types.append(td)

                # Long form: multiple parameters, or multiple required parameters
                params_td = TypedDict(f'evaluator_params_{name}', type_hints)  # pyright: ignore
                params_td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                td = TypedDict(f'evaluator_{name}', {name: params_td})  # pyright: ignore
                td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                evaluator_types.append(td)
            # Note: We might want to also generate the JSON schema for the format `call: '...', args: [...], kwargs: {...}`.
            #   It would be a bit complex to implement but not impossible.

        in_type, out_type, meta_type = cls._params()

        class ClsDatasetRow(BaseModel, extra='forbid'):
            name: str
            inputs: in_type
            metadata: meta_type
            expected_output: out_type | None = None
            if evaluator_types:
                # TODO: Add some default evaluator_types that are always included and remove this conditional
                evaluators: list[Union[tuple(evaluator_types)]] = []  # pyright: ignore  # noqa UP007

        ClsDatasetRow.__name__ = cls.__name__ + 'Row'

        class ClsDataset(BaseModel, extra='forbid'):
            cases: list[ClsDatasetRow]
            if evaluator_types:
                # TODO: Add some default evaluator_types that are always included and remove this conditional
                evaluators: list[Union[tuple(evaluator_types)]] = []  # pyright: ignore  # noqa UP007

        ClsDataset.__name__ = cls.__name__

        return ClsDataset.model_json_schema()

    @classmethod
    def _save_schema(
        cls, path: Path | str, custom_evaluators: Sequence[EvaluatorFunction[InputsT, OutputT, MetadataT]] = ()
    ):
        path = Path(path)
        json_schema = cls.model_json_schema_with_evaluators(custom_evaluators)
        schema_content = to_json(json_schema, indent=2).decode() + '\n'
        if not path.exists() or path.read_text() != schema_content:
            path.write_text(schema_content)

    @classmethod
    @functools.cache
    def _serialization_type(cls) -> type[_DatasetModel[InputsT, OutputT, MetadataT]]:
        return _DatasetModel[cls._params()]  # type: ignore

    @classmethod
    def _get_relative_path(cls, path: Path | str) -> Path:
        """Resolve relative paths as relative to the module in which the subclass is defined."""
        path = Path(path)

        if path.is_absolute():
            return path

        # TODO: Should we use the cwd instead of the module path? Then it would work for non-proper-subclasses..
        module_path = sys.modules[cls.__module__].__file__
        if module_path == __file__:
            raise ValueError(f'You should only call this method from a proper subclass of `{cls.__name__}`')

        assert module_path is not None, 'Module must be a file-based module'
        root = Path(module_path).parent
        return root / path

    @classmethod
    def _infer_fmt(cls, path: Path, fmt: Literal['yaml', 'json'] | None) -> Literal['yaml', 'json']:
        if fmt is not None:
            return fmt
        if path.suffix in {'.yaml', '.yml'}:
            return 'yaml'
        if path.suffix == '.json':
            return 'json'
        raise ValueError(f'Unrecognized file format: {path.suffix}. Use the `fmt` argument to specify the format.')


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


@dataclass
class _TaskRun:
    attributes: dict[str, Any] = field(init=False, default_factory=dict)
    metrics: dict[str, int | float] = field(init=False, default_factory=dict)

    def record_metric(self, name: str, value: int | float) -> None:
        self.metrics[name] = value

    def increment_metric(self, name: str, amount: int | float) -> None:
        current_value = self.metrics.get(name, 0)
        incremented_value = current_value + amount
        if current_value == 0 and incremented_value == 0:
            return  # Avoid recording a metric that is always zero
        self.record_metric(name, incremented_value)

    def record_attribute(self, name: str, value: Any) -> None:
        self.attributes[name] = value


async def _run_task(
    task: Callable[[InputsT], Awaitable[OutputT]], case: Case[InputsT, OutputT, MetadataT]
) -> EvaluatorContext[InputsT, OutputT, MetadataT]:
    task_run = _TaskRun()
    if _CURRENT_TASK_RUN.get() is not None:
        raise RuntimeError('A task run has already been entered. Task runs should not be nested')

    # TODO: Should we handle task execution errors in some way? Right now they bubble up immediately
    token = _CURRENT_TASK_RUN.set(task_run)
    try:
        with _logfire.span('execute {task}', task=get_unwrapped_function_name(task)) as task_span:
            with context_subtree() as span_tree:
                task_output = await task(case.inputs)
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
        duration=_get_span_duration(task_span),
        _span_tree=span_tree,
        attributes=task_run.attributes,
        metrics=task_run.metrics,
    )


async def _run_task_and_evaluators(
    task: Callable[[InputsT], Awaitable[OutputT]],
    case: Case[InputsT, OutputT, MetadataT],
    report_case_name: str,
    dataset_evaluators: list[EvaluatorDetails[InputsT, OutputT, MetadataT]],
) -> ReportCase:
    with _logfire.span(
        '{task_name}: {case_name}',
        task_name=get_unwrapped_function_name(task),
        case_name=case.name,
        inputs=case.inputs,
        metadata=case.metadata,
    ) as case_span:
        scoring_context = await _run_task(task, case)

        case_span.set_attribute('output', scoring_context.output)
        case_span.set_attribute('task_duration', scoring_context.duration)
        case_span.set_attribute('metrics', scoring_context.metrics)
        case_span.set_attribute('attributes', scoring_context.attributes)

        evaluators = case.evaluators + dataset_evaluators
        evaluator_outputs: list[SourcedEvaluatorOutput] = []
        if evaluators:
            async with asyncio.TaskGroup() as tg:
                tasks: list[asyncio.Task[list[SourcedEvaluatorOutput]]] = []
                for evaluator in evaluators:
                    tasks.append(tg.create_task(evaluator.execute(scoring_context)))
            for t in tasks:
                evaluator_outputs.extend(t.result())

        assertions, scores, labels = _group_evaluator_outputs_by_type(evaluator_outputs)
        case_span.set_attribute('assertions', assertions)
        case_span.set_attribute('scores', scores)
        case_span.set_attribute('labels', labels)

        context = case_span.context
        assert context is not None
        trace_id = f'{context.trace_id:032x}'
        span_id = f'{context.span_id:016x}'

    report_inputs = to_jsonable_python(case.inputs)

    scores, labels, assertions = ReportCase.group_evaluator_outputs(evaluator_outputs)
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
        total_duration=_get_span_duration(case_span),
        trace_id=trace_id,
        span_id=span_id,
    )


def _group_evaluator_outputs_by_type(
    evaluators: Sequence[SourcedEvaluatorOutput],
) -> tuple[
    dict[str, SourcedEvaluatorOutput[bool]],
    dict[str, SourcedEvaluatorOutput[int | float]],
    dict[str, SourcedEvaluatorOutput[str]],
]:
    """Groups evaluators by value type."""
    assertions: dict[str, SourcedEvaluatorOutput[bool]] = {}
    scores: dict[str, SourcedEvaluatorOutput[int | float]] = {}
    labels: dict[str, SourcedEvaluatorOutput[str]] = {}
    seen_names = set[str]()
    for a in evaluators:
        name = a.name
        # Dedupe repeated names by adding a numeric suffix
        if name in seen_names:
            suffix = 2
            while f'{name}_{suffix}' in seen_names:
                suffix += 1
            name = f'{name}_{suffix}'
        seen_names.add(name)
        if assertion := a.downcast(bool):
            assertions[name] = assertion
        elif score := a.downcast(int, float):
            scores[name] = score
        elif label := a.downcast(str):
            labels[name] = label
    return assertions, scores, labels


_CURRENT_TASK_RUN = ContextVar[_TaskRun | None]('_CURRENT_TASK_RUN', default=None)


def set_eval_attribute(name: str, value: Any) -> None:
    """Set the named attribute for the current eval task run. Do nothing if not in an eval task run."""
    current_case = _CURRENT_TASK_RUN.get()
    if current_case is not None:
        current_case.record_attribute(name, value)


def increment_eval_metric(name: str, amount: int | float) -> None:
    """Increment the named metric for the current eval task run. Do nothing if not in an eval task run."""
    current_case = _CURRENT_TASK_RUN.get()
    if current_case is not None:
        current_case.increment_metric(name, amount)


def _get_span_duration(span: logfire.LogfireSpan) -> float:
    end_time = span.end_time
    start_time = span.start_time
    assert isinstance(start_time, int), 'span is not started'
    assert isinstance(end_time, int), 'span is not finished'
    return (end_time - start_time) / 1_000_000_000


def _get_registry(
    custom_evaluators: Sequence[EvaluatorFunction[InputsT, OutputT, MetadataT]],
) -> dict[str, EvaluatorFunction[InputsT, OutputT, MetadataT]]:
    return {get_unwrapped_function_name(f): f for f in tuple(custom_evaluators) + DEFAULT_EVALUATORS}
