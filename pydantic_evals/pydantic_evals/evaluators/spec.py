"""This module provides the EvaluatorSpec class for specifying evaluators in a serializable format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import (
    BaseModel,
    RootModel,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core.core_schema import SerializationInfo, SerializerFunctionWrapHandler

if TYPE_CHECKING:
    # This import seems to fail on Pydantic 2.10.1 in CI
    from pydantic import ModelWrapValidatorHandler
    # TODO: Remove this once pydantic 2.11 is the min supported version


class EvaluatorSpec(BaseModel):
    """The specification of an evaluator to be run.

    This class is used to represent evaluators in a serializable format, supporting various
    short forms for convenience when defining evaluators in YAML or JSON dataset files.

    In particular, each of the following forms is supported for specifying an evaluator with name `MyEvaluator`:
    * `'MyEvaluator'` - Just the (string) name of the Evaluator subclass is used if its `__init__` takes no arguments
    * `{'MyEvaluator': first_arg}` - A single argument is passed as the first positional argument to `MyEvaluator.__init__`
    * `{'MyEvaluator': {k1: v1, k2: v2}}` - Multiple kwargs are passed to `MyEvaluator.__init__`
    """

    name: str
    """The name of the evaluator class; should be the value returned by `EvaluatorClass.get_serialization_name()`"""

    arguments: None | tuple[Any] | dict[str, Any]
    """The arguments to pass to the evaluator's constructor.

    Can be None (no arguments), a tuple (a single positional argument), or a dict (keyword arguments).
    """

    @property
    def args(self) -> tuple[Any, ...]:
        """Get the positional arguments for the evaluator.

        Returns:
            A tuple of positional arguments if arguments is a tuple, otherwise an empty tuple.
        """
        if isinstance(self.arguments, tuple):
            return self.arguments
        return ()

    @property
    def kwargs(self) -> dict[str, Any]:
        """Get the keyword arguments for the evaluator.

        Returns:
            A dictionary of keyword arguments if arguments is a dict, otherwise an empty dict.
        """
        if isinstance(self.arguments, dict):
            return self.arguments
        return {}

    @model_validator(mode='wrap')
    @classmethod
    def deserialize(cls, value: Any, handler: ModelWrapValidatorHandler[EvaluatorSpec]) -> EvaluatorSpec:
        """Deserialize an EvaluatorSpec from various formats.

        This validator handles the various short forms of evaluator specifications,
        converting them to a consistent EvaluatorSpec instance.

        Args:
            value: The value to deserialize.
            handler: The validator handler.

        Returns:
            The deserialized EvaluatorSpec.

        Raises:
            ValidationError: If the value cannot be deserialized.
        """
        try:
            result = handler(value)
            return result
        except ValidationError as exc:
            try:
                deserialized = _SerializedEvaluatorSpec.model_validate(value)
            except ValidationError:
                raise exc  # raise the original error
            return deserialized.to_evaluator_spec()

    @model_serializer(mode='wrap')
    def serialize(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> Any:
        """Serialize using the appropriate short-form if possible.

        Returns:
            The serialized evaluator specification, using the shortest form possible:
            - Just the name if there are no arguments
            - {name: first_arg} if there's a single positional argument
            - {name: {kwargs}} if there are multiple (keyword) arguments
        """
        if isinstance(info.context, dict) and info.context.get('use_short_form'):  # pyright: ignore[reportUnknownMemberType]
            if self.arguments is None:
                return self.name
            elif isinstance(self.arguments, tuple):
                return {self.name: self.arguments[0]}
            else:
                return {self.name: self.arguments}
        else:
            return handler(self)


class _SerializedEvaluatorSpec(RootModel[str | dict[str, Any]]):
    """Internal class for handling the serialized form of an EvaluatorSpec.

    This is an auxiliary class used to serialize/deserialize instances of EvaluatorSpec
    from their serialized representation in YAML or JSON.
    """

    @field_validator('root')
    @classmethod
    def enforce_one_key(cls, value: str | dict[str, Any]) -> Any:
        """Enforce that the root value has exactly one key (the evaluator name) when it is a dict.

        Args:
            value: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValueError: If the value is a dict with multiple keys.
        """
        if isinstance(value, str):
            return value
        if len(value) != 1:
            raise ValueError(
                f'Expected a single key containing the Evaluator class name, found keys {list(value.keys())}'
            )
        return value

    @property
    def _name(self) -> str:
        """Get the name of the evaluator from the serialized form.

        Returns:
            The name of the evaluator.
        """
        if isinstance(self.root, str):
            return self.root
        return next(iter(self.root.keys()))

    @property
    def _args(self) -> None | tuple[Any] | dict[str, Any]:
        """Get the arguments for the evaluator from the serialized form.

        Returns:
            The arguments for the evaluator, which can be None (no arguments),
            a tuple (one positional arguments), or a dict (multiple keyword arguments).
        """
        if isinstance(self.root, str):
            return None  # no init args or kwargs

        value = next(iter(self.root.values()))

        if isinstance(value, dict):
            keys: list[Any] = list(value.keys())  # pyright: ignore[reportUnknownArgumentType]
            if all(isinstance(k, str) for k in keys):
                # dict[str, Any]s are treated as init kwargs
                return cast(dict[str, Any], value)

        # Anything else is passed as a single positional argument to __init__
        return (cast(Any, value),)

    def to_evaluator_spec(self) -> EvaluatorSpec:
        """Convert this serialized form to an EvaluatorSpec.

        Returns:
            An EvaluatorSpec instance.
        """
        return EvaluatorSpec(name=self._name, arguments=self._args)
