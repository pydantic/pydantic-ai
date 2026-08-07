from __future__ import annotations as _annotations

from .._json_schema import JsonSchema, JsonSchemaTransformer
from ..exceptions import UserError
from . import ModelProfile

# MIME types supported in native FunctionResponseDict.parts for Gemini 3+.
# See https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#multimodal
_GOOGLE_NATIVE_TOOL_RETURN_MIME_TYPES: tuple[str, ...] = (
    'image/png',
    'image/jpeg',
    'image/webp',
    'application/pdf',
    'text/plain',
)


class GoogleModelProfile(ModelProfile, total=False):
    """Profile for models used with `GoogleModel`.

    ALL FIELDS MUST BE `google_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    google_supports_tool_combination: bool
    """Whether the model supports combining function declarations with native tools and response_schema. Default: `False`.

    Gemini 3+ supports all tool combinations:
    - function_declarations + native_tools
    - output_tools (function declarations) + native_tools
    - response_schema (NativeOutput) + function_declarations
    See https://ai.google.dev/gemini-api/docs/tool-combination
    """

    google_supports_server_side_tool_invocations: bool
    """Whether the model accepts the `include_server_side_tool_invocations` tool-config field. Default: `False`.

    When enabled, Gemini emits explicit `tool_call`/`tool_response` parts for server-side
    native tools (Google Search, URL Context, File Search) that we round-trip through
    [`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart] /
    [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart]. Pre-Gemini-3 models
    reject the field with `'Tool call context circulation is not enabled'`.

    This is a Gemini Developer API (ML Dev) only parameter: the google-genai SDK's Vertex
    converter raises `ValueError` when the field is set, so `GoogleModel` skips it for
    Google Cloud (Vertex) even on Gemini 3+ models.

    Distinct from [`google_supports_tool_combination`][pydantic_ai.profiles.google.GoogleModelProfile.google_supports_tool_combination]
    even though both currently flip on for Gemini 3+ — the former gates the SDK request
    field, the latter gates which combinations of native / function / output tools are
    allowed in the same request.
    """

    google_supported_mime_types_in_tool_returns: tuple[str, ...]
    """MIME types supported in native FunctionResponseDict.parts. Default: `()`.
    See https://ai.google.dev/gemini-api/docs/function-calling#multimodal-function-responses"""

    google_supports_thinking_level: bool
    """Whether the model uses `thinking_level` (enum: LOW/MEDIUM/HIGH) instead of `thinking_budget` (int). Default: `False`.

    Gemini 3+ models use `thinking_level`; Gemini 2.5 uses `thinking_budget`.
    """

    google_supports_strict_tool_definition: bool
    """Whether the model supports Gemini's `VALIDATED` function-calling mode. Default: `False`.

    `VALIDATED` is Gemini's equivalent of the cross-provider `strict` tool flag (like OpenAI/Anthropic
    strict tool calling): it behaves like `AUTO` but the API enforces that the model adheres to the
    declared function schema. Issue reports also observe that it mitigates the function-name hallucination
    some Gemini models exhibit (an observed effect, not a documented guarantee). When the flag is set,
    `GoogleModel` upgrades `AUTO` to `VALIDATED` by default (every schema is VALIDATED-compatible — no
    rewrites), and a caller opts out per tool with `ToolDefinition.strict=False`. Because Gemini's mode is
    request-wide, any function or output tool with `strict=False` keeps the whole request on `AUTO`.

    See <https://ai.google.dev/gemini-api/docs/function-calling#function_calling_config>.
    """


def google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model."""
    is_image_model = 'image' in model_name
    is_3_or_newer = 'gemini-3' in model_name
    is_thinking_model = 'gemini-2.5' in model_name or is_3_or_newer
    # `VALIDATED` function-calling mode is available on Gemini 2.5 and newer (the models targeted by
    # https://github.com/pydantic/pydantic-ai/issues/5366); image models don't support function tools,
    # so leave it off there.
    supports_strict_tool_definition = is_thinking_model and not is_image_model
    # Pro models have always-on thinking: Gemini 2.5 Pro rejects budget=0, Gemini 3+ Pro rejects MINIMAL
    is_pro = 'pro' in model_name and 'flash' not in model_name
    thinking_always_enabled = is_thinking_model and is_pro
    return GoogleModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_image_output=is_image_model,
        supports_json_schema_output=is_3_or_newer or not is_image_model,
        supports_json_object_output=is_3_or_newer or not is_image_model,
        supports_tools=not is_image_model,
        supports_tool_return_schema=not is_image_model,
        supports_thinking=is_thinking_model,
        thinking_always_enabled=thinking_always_enabled,
        google_supports_tool_combination=is_3_or_newer,
        google_supports_server_side_tool_invocations=is_3_or_newer,
        google_supported_mime_types_in_tool_returns=_GOOGLE_NATIVE_TOOL_RETURN_MIME_TYPES if is_3_or_newer else (),
        google_supports_thinking_level=is_3_or_newer,
        google_supports_strict_tool_definition=supports_strict_tool_definition,
    )


class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Gemini.

    Gemini supports [a subset of OpenAPI v3.0.3](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations).
    """

    def walk(self) -> JsonSchema:
        schema = super().walk()
        # Gemini's `VALIDATED` mode enforces the declared schema with no rewrites (unlike OpenAI/Anthropic
        # strict), so every schema is compatible. Keeping `is_strict_compatible` at `True` lets a `strict=None`
        # tool resolve as VALIDATED-eligible: `GoogleModel._get_tool_config` defaults supported models to
        # `VALIDATED` and a caller opts out per tool with `strict=False`.
        self.is_strict_compatible = True
        return schema

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Remove properties not supported by Gemini
        schema.pop('$schema', None)
        if (const := schema.pop('const', None)) is not None:
            # Gemini doesn't support const, but it does support enum with a single value
            schema['enum'] = [const]
            # If type is not present, infer it from the const value for Gemini API compatibility
            if 'type' not in schema:
                if isinstance(const, str):
                    schema['type'] = 'string'
                elif isinstance(const, bool):
                    # bool must be checked before int since bool is a subclass of int in Python
                    schema['type'] = 'boolean'
                elif isinstance(const, int):
                    schema['type'] = 'integer'
                elif isinstance(const, float):
                    schema['type'] = 'number'
        schema.pop('discriminator', None)
        schema.pop('examples', None)

        # Remove 'title' due to https://github.com/googleapis/python-genai/issues/1732
        schema.pop('title', None)

        type_ = schema.get('type')
        if type_ == 'string' and (fmt := schema.pop('format', None)):
            description = schema.get('description')
            if description:
                schema['description'] = f'{description} (format: {fmt})'
            else:
                schema['description'] = f'Format: {fmt}'

        # Note: exclusiveMinimum/exclusiveMaximum are NOT yet supported
        schema.pop('exclusiveMinimum', None)
        schema.pop('exclusiveMaximum', None)

        return schema


class GoogleOpenAPISchemaTransformer(GoogleJsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic into the OpenAPI v3.0.3 subset Gemini's `Schema` accepts.

    A function declaration carries its parameters as *either* `parametersJsonSchema` (full JSON Schema,
    which [`GoogleModel`][pydantic_ai.models.google.GoogleModel] sends) *or* `parameters` (an
    [OpenAPI v3.0.3 subset](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations)) —
    the two are mutually exclusive. The Live API only implements `parameters`, so
    [`GoogleRealtimeModel`][pydantic_ai.realtime.google.GoogleRealtimeModel] needs this narrower form,
    where a union is `anyOf`, an enum is a list of strings, and there are no `$ref`s to resolve.
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        # `$defs`/`$ref` and `anyOf [X, null]` have no OpenAPI-subset equivalent, so definitions are
        # inlined and a nullable union becomes the plain type plus `nullable: true`.
        super().__init__(schema, strict=strict, prefer_inlined_defs=True, simplify_nullable_unions=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # `additionalProperties` is mishandled by Gemini, so a `dict[str, MyType]` field always arrives
        # empty. Dropping it is what makes the rest of the schema usable; the alternative is refusing
        # the whole tool.
        schema.pop('additionalProperties', None)

        schema = super().transform(schema)

        # Gemini only accepts string enums here, and `Schema.enum` is typed `list[str]`.
        if enum := schema.get('enum'):
            if all(isinstance(value, str) for value in enum):
                schema['type'] = 'string'
            else:
                # Stringifying the values would be a trap: the model would answer `"1"` for a
                # `Literal[1, 2]`, exactly as asked, and validation would then reject its own
                # schema's answer, since Pydantic won't coerce a string into an int literal. Keep the
                # declared type so the answer validates, and move the choices into the description so
                # they're still stated — an unenforced hint beats an unanswerable argument.
                del schema['enum']
                allowed = ', '.join(repr(value) for value in enum)
                description = schema.get('description')
                schema['description'] = (
                    f'{description} (allowed values: {allowed})' if description else f'Allowed values: {allowed}'
                )

        if 'oneOf' in schema and 'type' not in schema:
            # A discriminated union. Gemini rejects `oneOf` outright (despite what its own error message
            # says), and `anyOf` is functionally equivalent for a schema whose members are disjoint.
            schema['anyOf'] = schema.pop('oneOf')

        if '$ref' in schema:
            # `prefer_inlined_defs` resolved every reference it could; one left over is a recursive
            # schema, which the OpenAPI subset cannot express at all.
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Gemini: {schema["$ref"]}')

        if 'prefixItems' in schema:
            # A tuple. The subset has no positional item types, so the element type widens to the union
            # of the positions, and the length is pinned instead.
            prefix_items = schema.pop('prefixItems')
            items = schema.get('items')
            unique_items = [items] if items is not None else []
            for item in prefix_items:
                if item not in unique_items:
                    unique_items.append(item)
            if len(unique_items) > 1:
                schema['items'] = {'anyOf': unique_items}
            elif len(unique_items) == 1:  # pragma: no branch
                schema['items'] = unique_items[0]
            schema.setdefault('minItems', len(prefix_items))
            if items is None:  # pragma: no branch
                schema.setdefault('maxItems', len(prefix_items))

        return schema
