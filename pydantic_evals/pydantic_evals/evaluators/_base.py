"""Shared serialization base for Evaluator and ReportEvaluator."""

from __future__ import annotations

from abc import ABCMeta
from dataclasses import MISSING, dataclass, fields
from typing import Any

from pydantic import ConfigDict, model_serializer
from pydantic_core import to_jsonable_python
from pydantic_core.core_schema import SerializationInfo

from pydantic_ai import _utils

from .spec import EvaluatorSpec


class _StrictABCMeta(ABCMeta):
    """An ABC-like metaclass that disallows defining subclasses with unimplemented inherited abstract methods.

    Unlike standard ABCMeta (which allows abstract subclasses and only errors at instantiation),
    this metaclass raises TypeError at class definition time if a subclass inherits abstract methods
    without implementing them. Classes that define their own new abstract methods are allowed, since
    they are intentionally creating a new abstract layer (e.g. Evaluator and ReportEvaluator).
    """

    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any):
        result = super().__new__(mcls, name, bases, namespace, **kwargs)
        is_proper_subclass = any(isinstance(c, _StrictABCMeta) for c in result.__mro__[1:])
        if is_proper_subclass and result.__abstractmethods__:
            # Only error on abstract methods inherited from a parent but not implemented.
            # Methods defined in this class's own namespace are intentionally abstract (new abstract layer).
            own_abstracts = frozenset(m for m in result.__abstractmethods__ if m in namespace)
            inherited_unimplemented = result.__abstractmethods__ - own_abstracts
            if inherited_unimplemented:
                abstractmethods = ', '.join(f'{m!r}' for m in inherited_unimplemented)
                raise TypeError(f'{name} must implement all abstract methods: {abstractmethods}')
        return result


@dataclass(repr=False)
class BaseEvaluator(metaclass=_StrictABCMeta):
    """Shared serialization, spec-building, and repr logic for Evaluator and ReportEvaluator."""

    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the 'name' of this evaluator to use during serialization.

        Returns:
            The name of the evaluator, which is typically the class name.
        """
        return cls.__name__

    @model_serializer(mode='plain')
    def serialize(self, info: SerializationInfo) -> Any:
        """Serialize this evaluator to a JSON-serializable form.

        Returns:
            A JSON-serializable representation of this evaluator as an EvaluatorSpec.
        """
        return to_jsonable_python(
            self.as_spec(),
            context=info.context,
            serialize_unknown=True,
        )

    def as_spec(self) -> EvaluatorSpec:
        raw_arguments = self.build_serialization_arguments()

        arguments: None | tuple[Any,] | dict[str, Any]
        if len(raw_arguments) == 0:
            arguments = None
        elif len(raw_arguments) == 1:
            # Only use the compact tuple form if the single non-default field is the first
            # dataclass field, since the tuple form passes the value as the first positional arg.
            first_field_name = fields(self)[0].name
            key = next(iter(raw_arguments))
            if key == first_field_name:
                arguments = (raw_arguments[key],)
            else:
                arguments = raw_arguments
        else:
            arguments = raw_arguments

        return EvaluatorSpec(name=self.get_serialization_name(), arguments=arguments)

    def build_serialization_arguments(self) -> dict[str, Any]:
        """Build the arguments for serialization.

        If you want to modify how the evaluator is serialized, you can override this method.

        Returns:
            A dictionary of arguments to be used during serialization.
        """
        raw_arguments: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            # always exclude defaults:
            if field.default is not MISSING:
                if value == field.default:
                    continue
            if field.default_factory is not MISSING:
                if value == field.default_factory():  # pragma: no branch
                    continue
            raw_arguments[field.name] = value
        return raw_arguments

    __repr__ = _utils.dataclasses_no_defaults_repr
