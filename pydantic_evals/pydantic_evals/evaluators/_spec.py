from __future__ import annotations

from typing import Any, cast

from pydantic import (
    BaseModel,
    ModelWrapValidatorHandler,
    RootModel,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)


class EvaluatorSpec(BaseModel):
    """The specification of an evaluator to be run.

    Note that special "short forms" are supported during serialization/deserialization.

    In particular, each of the following forms is supported for specifying an evaluator with name `MyEvaluator`:
    * `'MyEvaluator'` Just the (string) name of the Evaluator subclass is used if its `__init__` takes no arguments
    * `{'MyEvaluator': first_arg}` A single argument is passed as the first argument to the `MyEvaluator.__init__`
    * `{'MyEvaluator': {k1: v1, k2: v2}}` Multiple kwargs are passed to `MyEvaluator.__init__`
    """

    name: str
    arguments: None | tuple[Any] | dict[str, Any]

    @property
    def args(self) -> tuple[Any, ...]:
        if isinstance(self.arguments, tuple):
            return self.arguments
        return ()

    @property
    def kwargs(self) -> dict[str, Any]:
        if isinstance(self.arguments, dict):
            return self.arguments
        return {}

    @model_validator(mode='wrap')
    @classmethod
    def deserialize(cls, value: Any, handler: ModelWrapValidatorHandler[EvaluatorSpec]) -> EvaluatorSpec:
        try:
            result = handler(value)
            return result
        except ValidationError as exc:
            try:
                deserialized = _SerializedEvaluatorSpec.model_validate(value)
            except ValidationError:
                raise exc  # raise the original error; TODO: We should somehow combine the errors
            return deserialized.to_evaluator_spec()

    @model_serializer(mode='plain')
    def serialize(self) -> Any:
        """Serialize using the appropriate short-form if possible."""
        if self.arguments is None:
            return self.name
        elif isinstance(self.arguments, tuple):
            return {self.name: self.arguments[0]}
        else:
            return {self.name: self.arguments}


class _SerializedEvaluatorSpec(RootModel[str | dict[str, Any]]):
    """The specification of an evaluator to be run.

    Corresponds to the serialized form of an Evaluator (e.g., in a yaml file).

    This is just an auxiliary class used to serialize/deserialize instances of EvaluatorSpec.
    """

    @field_validator('root')
    @classmethod
    def enforce_one_key(cls, value: str | dict[str, Any]) -> Any:
        if isinstance(value, str):
            return value
        if len(value) != 1:
            raise ValueError(
                f'Expected a single key containing the Evaluator class name, found keys {list(value.keys())}'
            )
        return value

    @property
    def _name(self) -> str:
        if isinstance(self.root, str):
            return self.root
        return next(iter(self.root.keys()))

    @property
    def _args(self) -> None | tuple[Any] | dict[str, Any]:
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
        return EvaluatorSpec(name=self._name, arguments=self._args)
