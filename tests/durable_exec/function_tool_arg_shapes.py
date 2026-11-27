"""Function tool argument shapes a durable engine's JSON boundary can lose, for each engine's suite to run.

Each engine's suite runs `SHAPES` in one agent run: the model calls every shape's tool at once, and each tool
records what it received. `check_shapes` then compares that with what the same tools receive without a durable
engine, and with whether each call's durable unit input carried the model's arguments (`raw_tool_args`).

A shape's call carries them only when its dump loses values, the tool's argument validation runs no user code,
and no hook changed the arguments. Every other call's unit input has no `raw_tool_args` key, so it is the same
input the engine sent before the field existed.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Json,
    PrivateAttr,
    SecretStr,
    Tag,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import TypedDict

from pydantic_ai import RunContext
from pydantic_ai._utils import is_list_tuple_or_set, is_str_dict
from pydantic_ai.capabilities import Hooks, ValidatedToolArgs
from pydantic_ai.messages import BinaryContent, ImageUrl, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

RECEIVED: dict[str, str] = {}
"""What each shape's tool received, by tool name."""

Engine = Literal['temporal', 'render', 'azure']


def describe(value: Any) -> str:
    """`value` with every attribute, its exact type and any secret, so a lost or changed value shows."""
    if isinstance(value, SecretStr):
        return f'SecretStr({value.get_secret_value()!r})'
    if isinstance(value, BaseModel):
        private = describe(value.__pydantic_private__) if value.__pydantic_private__ else ''
        return f'{type(value).__name__}({describe(value.__dict__)}{private})'
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return f'{type(value).__name__}({describe(vars(value))})'
    if is_str_dict(value):
        return '{' + ', '.join(f'{key!r}: {describe(item)}' for key, item in value.items()) + '}'
    if is_list_tuple_or_set(value):
        return f'{type(value).__name__}[' + ', '.join(describe(item) for item in value) + ']'
    return f'{type(value).__name__}:{value!r}'


def _record(name: str, **kwargs: Any) -> str:
    RECEIVED[name] = describe(kwargs)
    return 'ok'


# Shared parameter types.


class AliasedCity(TypedDict):
    city: Annotated[str, Field(alias='cityName')]


class OptionalPrice(TypedDict, total=False):
    max_price: Annotated[float, Field(alias='maxPrice')]


@dataclass
class AliasedCityDataclass:
    city: Annotated[str, Field(alias='cityName')]


@dataclass
class OptionalPriceDataclass:
    max_price: Annotated[float | None, Field(alias='maxPrice')] = None


class SerializedRoom(BaseModel):
    room: str | None = Field(None, serialization_alias='roomOut')


class SplitRoom(BaseModel):
    room: str = Field(validation_alias='roomIn', serialization_alias='roomOut')


class ChosenCity(BaseModel):
    city: str = Field(validation_alias=AliasChoices('cityName', 'town'))


class NameOnlyCity(BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=False)
    city: str = Field(alias='cityName')


class Town(BaseModel):
    model_config = ConfigDict(extra='forbid')
    city: str = Field(alias='town')


class Credentials(BaseModel):
    token: SecretStr


class JsonHolder(BaseModel):
    nums: Json[list[int]]


class Counted(BaseModel):
    model_config = ConfigDict(extra='forbid')
    qty: int

    @computed_field
    @property
    def doubled(self) -> int:
        return self.qty * 2


class Shouted(BaseModel):
    word: str

    @field_serializer('word')
    def shout(self, word: str) -> str:
        return word.upper()


class Hidden(BaseModel):
    keep: int = 0
    hidden: int = Field(5, exclude=True)


class Tagged(BaseModel):
    tags: list[str] = Field(default_factory=list)


class Cached(BaseModel):
    n: int
    _cache: int = PrivateAttr(default=0)


class Plain(BaseModel):
    city: str


class AliasedModel(BaseModel):
    city: str = Field(alias='cityName')


class Lowered(BaseModel):
    email: str

    @field_validator('email')
    @classmethod
    def lower(cls, email: str) -> str:
        return email.lower()


class PostInit(BaseModel):
    n: int

    def model_post_init(self, context: Any, /) -> None:
        pass


def _default_key() -> str:
    # Deterministic: Temporal's workflow sandbox rejects `uuid.uuid4`, which the workflow's own validation would call.
    return 'generated'


class Keyed(BaseModel):
    key: str = Field(default_factory=_default_key)
    n: int = 0


class Cat(BaseModel):
    kind: Literal['cat'] = 'cat'


class Dog(BaseModel):
    kind: Literal['dog'] = 'dog'


def _pet_kind(value: Cat | Dog | dict[str, str]) -> str:
    return value.get('kind', 'cat') if isinstance(value, dict) else value.kind


class Color(Enum):
    RED = 'red'

    @classmethod
    def _missing_(cls, value: object) -> Color:
        return cls.RED


class Folded(BaseModel):
    price: int
    qty: int = 1

    @model_validator(mode='before')
    @classmethod
    def fold_quantity(cls, data: dict[str, int]) -> dict[str, int]:
        # Mutates its input: run twice on one dict, the second run finds no `qty`.
        data['price'] = data.pop('price') * data.pop('qty')
        return data


class Tenant(BaseModel):
    name: str
    tenant_id: int = Field(0, exclude=True)


@dataclass
class Account:
    name: str
    tenant_id: int = dataclasses.field(compare=False)


# Tools. Each records the arguments it received.


def typed_dict_required(loc: AliasedCity, n: int = 0) -> str:
    return _record('typed_dict_required', loc=loc, n=n)


def typed_dict_optional(filters: OptionalPrice) -> str:
    return _record('typed_dict_optional', filters=filters)


def typed_dict_single(loc: AliasedCity) -> str:
    return _record('typed_dict_single', loc=loc)


def typed_dict_list(locs: list[AliasedCity]) -> str:
    return _record('typed_dict_list', locs=locs)


def dataclass_required(loc: AliasedCityDataclass, n: int = 0) -> str:
    return _record('dataclass_required', loc=loc, n=n)


def dataclass_optional(filters: OptionalPriceDataclass, n: int = 0) -> str:
    return _record('dataclass_optional', filters=filters, n=n)


def serialization_alias_only(m: SerializedRoom, n: int = 0) -> str:
    return _record('serialization_alias_only', m=m, n=n)


def split_alias(m: SplitRoom, n: int = 0) -> str:
    return _record('split_alias', m=m, n=n)


def alias_choices(loc: ChosenCity, n: int = 0) -> str:
    return _record('alias_choices', loc=loc, n=n)


def validate_by_name_only(loc: NameOnlyCity, n: int = 0) -> str:
    return _record('validate_by_name_only', loc=loc, n=n)


def union_member(loc: Town | AliasedCity, n: int = 0) -> str:
    return _record('union_member', loc=loc, n=n)


def secrets(token: SecretStr, creds: Credentials) -> str:
    return _record('secrets', token=token, creds=creds)


def json_top_level(nums: Json[list[int]], n: int = 0) -> str:
    return _record('json_top_level', nums=nums, n=n)


def json_nested(holder: JsonHolder, n: int = 0) -> str:
    return _record('json_nested', holder=holder, n=n)


def computed_field_forbidden(counted: Counted, n: int = 0) -> str:
    return _record('computed_field_forbidden', counted=counted, n=n)


def field_serializer_changes_value(shouted: Shouted, n: int = 0) -> str:
    return _record('field_serializer_changes_value', shouted=shouted, n=n)


def excluded_field(h: Hidden, n: int = 0) -> str:
    return _record('excluded_field', h=h, n=n)


def media(image: ImageUrl, content: BinaryContent) -> str:
    return _record('media', image=image, content=content)


def builtin_default_factory(t: Tagged, loc: AliasedCity) -> str:
    return _record('builtin_default_factory', t=t, loc=loc)


def private_attribute(c: Cached, loc: AliasedCity) -> str:
    return _record('private_attribute', c=c, loc=loc)


def any_dict(data: dict[str, Any], loc: AliasedCity) -> str:
    return _record('any_dict', data=data, loc=loc)


def plain_model(loc: Plain, n: int = 0) -> str:
    return _record('plain_model', loc=loc, n=n)


def aliased_model(loc: AliasedModel, n: int = 0) -> str:
    return _record('aliased_model', loc=loc, n=n)


def top_level_alias(guest: Annotated[str, Field(alias='guestName')], n: int = 0) -> str:
    return _record('top_level_alias', guest=guest, n=n)


def no_arguments() -> str:
    return _record('no_arguments')


def field_validator_runs(user: Lowered, token: SecretStr) -> str:
    return _record('field_validator_runs', user=user, token=token)


def model_post_init_runs(p: PostInit, token: SecretStr) -> str:
    return _record('model_post_init_runs', p=p, token=token)


def custom_default_factory(k: Keyed, token: SecretStr) -> str:
    return _record('custom_default_factory', k=k, token=token)


def callable_discriminator(
    pet: Annotated[Annotated[Cat, Tag('cat')] | Annotated[Dog, Tag('dog')], Discriminator(_pet_kind)],
    token: SecretStr,
) -> str:
    return _record('callable_discriminator', pet=pet, token=token)


def enum_missing(color: Color, token: SecretStr) -> str:
    return _record('enum_missing', color=color, token=token)


def mutating_validator(filters: Folded) -> str:
    return _record('mutating_validator', filters=filters)


def hook_pins_excluded_field(tenant: Tenant, token: SecretStr) -> str:
    return _record('hook_pins_excluded_field', tenant=tenant, token=token)


def hook_resets_field_to_default(tenant: Tenant, token: SecretStr) -> str:
    return _record('hook_resets_field_to_default', tenant=tenant, token=token)


def hook_mutates_in_place(account: Account, token: SecretStr) -> str:
    return _record('hook_mutates_in_place', account=account, token=token)


def hook_sets_negative_zero(x: float, token: SecretStr) -> str:
    return _record('hook_sets_negative_zero', x=x, token=token)


def hook_widens_decimal(d: Decimal, token: SecretStr) -> str:
    return _record('hook_widens_decimal', d=d, token=token)


# Hooks that change validated arguments, each in a way `==` or the arguments' JSON form misses.


def _pin_excluded(args: ValidatedToolArgs) -> ValidatedToolArgs:
    return {**args, 'tenant': args['tenant'].model_copy(update={'tenant_id': 1})}


def _reset_to_default(args: ValidatedToolArgs) -> ValidatedToolArgs:
    args['tenant'].tenant_id = 0
    return args


def _mutate_in_place(args: ValidatedToolArgs) -> ValidatedToolArgs:
    args['account'].tenant_id = 1
    return args


def _negative_zero(args: ValidatedToolArgs) -> ValidatedToolArgs:
    return {**args, 'x': -0.0}


def _widen_decimal(args: ValidatedToolArgs) -> ValidatedToolArgs:
    return {**args, 'd': Decimal('1.00')}


@dataclass(frozen=True)
class ArgShape:
    tool: Callable[..., str]
    args: dict[str, Any] | None
    raw: bool
    """Whether the call's durable unit input carries the model's arguments."""
    in_process: bool = True
    """Whether the tool receives what it does without a durable engine; otherwise the dump's losses remain."""
    raw_on_azure: bool | None = None
    """`raw` on Azure Durable Task, whose dump keeps computed fields and `Json` values in models, when it differs."""
    hook: Callable[[ValidatedToolArgs], ValidatedToolArgs] | None = None
    """An `after_tool_validate` hook for the call."""
    hook_value: str | None = None
    """What the hook set, which the tool must receive."""

    def carries_raw(self, engine: Engine) -> bool:
        return self.raw_on_azure if engine == 'azure' and self.raw_on_azure is not None else self.raw


CITY = {'cityName': 'Paris'}
SECRET = 'hunter2'

SHAPES = [
    # The dump loses values the tool's schema reads, and validation runs no user code: the unit validates the
    # model's arguments.
    ArgShape(typed_dict_required, {'loc': CITY, 'n': 1}, raw=True),
    ArgShape(typed_dict_optional, {'filters': {'maxPrice': 10}}, raw=True),
    ArgShape(typed_dict_single, CITY, raw=True),
    ArgShape(typed_dict_list, {'locs': [CITY]}, raw=True),
    ArgShape(dataclass_required, {'loc': CITY, 'n': 1}, raw=True),
    ArgShape(dataclass_optional, {'filters': {'maxPrice': 10}}, raw=True),
    ArgShape(serialization_alias_only, {'m': {'room': 'A1'}}, raw=True),
    ArgShape(split_alias, {'m': {'roomIn': 'A1'}}, raw=True),
    ArgShape(alias_choices, {'loc': {'town': 'Paris'}}, raw=True),
    ArgShape(validate_by_name_only, {'loc': {'city': 'Paris'}}, raw=True),
    ArgShape(union_member, {'loc': CITY}, raw=True),
    ArgShape(secrets, {'token': SECRET, 'creds': {'token': SECRET}}, raw=True),
    ArgShape(json_top_level, {'nums': '[1, 2]'}, raw=True),
    ArgShape(json_nested, {'holder': {'nums': '[1, 2]'}}, raw=True, raw_on_azure=False),
    ArgShape(computed_field_forbidden, {'counted': {'qty': 2}}, raw=True, raw_on_azure=False),
    ArgShape(field_serializer_changes_value, {'shouted': {'word': 'hi'}}, raw=True),
    ArgShape(excluded_field, {'h': {'keep': 1, 'hidden': 9}}, raw=True),
    ArgShape(
        media,
        {
            'image': {'url': 'https://example.com/a', 'media_type': 'image/png'},
            'content': {'data': 'aGk=', 'media_type': 'text/plain', 'identifier': 'note-1'},
        },
        raw=True,
    ),
    ArgShape(builtin_default_factory, {'t': {}, 'loc': CITY}, raw=True),
    ArgShape(private_attribute, {'c': {'n': 1}, 'loc': CITY}, raw=True),
    ArgShape(any_dict, {'data': {'a': [1, {'b': 2}]}, 'loc': CITY}, raw=True),
    # The dump already validates back: the call crosses as before.
    ArgShape(plain_model, {'loc': {'city': 'Paris'}, 'n': 1}, raw=False),
    ArgShape(aliased_model, {'loc': CITY, 'n': 1}, raw=False),
    ArgShape(top_level_alias, {'guestName': 'Ada', 'n': 1}, raw=False),
    ArgShape(no_arguments, None, raw=False),
    # Validation runs user code, so the call crosses as before: the tool gets the masked secret.
    ArgShape(field_validator_runs, {'user': {'email': 'A@B.C'}, 'token': SECRET}, raw=False, in_process=False),
    ArgShape(model_post_init_runs, {'p': {'n': 1}, 'token': SECRET}, raw=False, in_process=False),
    ArgShape(custom_default_factory, {'k': {'n': 1}, 'token': SECRET}, raw=False, in_process=False),
    ArgShape(callable_discriminator, {'pet': {'kind': 'dog'}, 'token': SECRET}, raw=False, in_process=False),
    ArgShape(enum_missing, {'color': 'crimson', 'token': SECRET}, raw=False, in_process=False),
    ArgShape(mutating_validator, {'filters': {'price': 3, 'qty': 2}}, raw=False),
    # A hook changed the arguments, so the call crosses as before: the tool gets the hook's value wherever the
    # dump keeps it, and the masked secret either way.
    ArgShape(
        hook_pins_excluded_field,
        {'tenant': {'name': 'a', 'tenant_id': 2}, 'token': SECRET},
        raw=False,
        in_process=False,
        hook=_pin_excluded,
    ),
    ArgShape(
        hook_resets_field_to_default,
        {'tenant': {'name': 'a', 'tenant_id': 2}, 'token': SECRET},
        raw=False,
        in_process=False,
        hook=_reset_to_default,
        hook_value="'tenant_id': int:0",
    ),
    ArgShape(
        hook_mutates_in_place,
        {'account': {'name': 'a', 'tenant_id': 2}, 'token': SECRET},
        raw=False,
        in_process=False,
        hook=_mutate_in_place,
        hook_value="'tenant_id': int:1",
    ),
    ArgShape(
        hook_sets_negative_zero,
        {'x': 0.0, 'token': SECRET},
        raw=False,
        in_process=False,
        hook=_negative_zero,
        hook_value="'x': float:-0.0",
    ),
    ArgShape(
        hook_widens_decimal,
        {'d': '1.0', 'token': SECRET},
        raw=False,
        in_process=False,
        hook=_widen_decimal,
        hook_value="'d': Decimal:Decimal('1.00')",
    ),
]

TOOLS = [shape.tool for shape in SHAPES]


def _after_validate(change: Callable[[ValidatedToolArgs], ValidatedToolArgs]):
    async def apply(
        ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        return change(args)

    return apply


def shapes_hooks() -> Hooks[Any]:
    """An `after_tool_validate` hook on each shape that has one."""
    hooks = Hooks[Any]()
    for shape in SHAPES:
        if shape.hook is not None:
            hooks.on.after_tool_validate(tools=[shape.tool.__name__])(_after_validate(shape.hook))
    return hooks


def shapes_model() -> FunctionModel:
    """A model that calls every shape's tool at once, with the tool's name as the call ID, then finishes."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(
            parts=[
                ToolCallPart(shape.tool.__name__, copy.deepcopy(shape.args), tool_call_id=shape.tool.__name__)
                for shape in SHAPES
            ]
        )

    return FunctionModel(respond)


def check_shapes(
    engine: Engine, received: Mapping[str, str], in_process: Mapping[str, str], carried_raw: Mapping[str, bool]
) -> None:
    """Compare a durable run's results with a run without an engine, shape by shape."""
    wrong = {
        name: {'received': received.get(name), 'in process': in_process[name], 'carried raw': carried_raw.get(name)}
        for shape in SHAPES
        if (name := shape.tool.__name__)
        and (
            name not in received
            or carried_raw.get(name) is not shape.carries_raw(engine)
            or (received.get(name) == in_process[name]) is not shape.in_process
            or (shape.hook_value is not None and shape.hook_value not in (received.get(name) or ''))
        )
    }
    assert wrong == {}
