import type { JsonValue, Schema } from './protocol'

/** How one setting is edited, chosen from its JSON Schema. */
export type Control =
  | { kind: 'switch' }
  | { kind: 'choice'; options: JsonValue[] }
  | { kind: 'text' }
  | { kind: 'number'; integer: boolean; min?: number; max?: number }
  | { kind: 'lines' }
  | { kind: 'group'; fields: Field[] }
  | { kind: 'json' }

export interface Field {
  key: string
  label: string
  description: string
  control: Control
  /** Whether `null` is a value, meaning "not set": an empty text, number, or list saves it. */
  nullable: boolean
  default?: JsonValue
}

/** The fields of a settings model's JSON Schema, in its order; `$ref`s resolve against its `$defs`. */
export function fields(schema: Schema, root: Schema = schema): Field[] {
  return Object.entries(schema.properties ?? {}).map(([key, property]) => field(key, property, root))
}

function field(key: string, property: Schema, root: Schema): Field {
  const resolved = resolve(property, root)
  const { control, nullable } = controlFor(resolved, root)
  return {
    key,
    label: resolved.title ?? key,
    description: resolved.description ?? '',
    control,
    nullable,
    ...('default' in resolved ? { default: resolved.default } : {}),
  }
}

/** The schema a `$ref` (or a lone `allOf`) points at, keeping what sits beside it, such as a description. */
function resolve(schema: Schema, root: Schema): Schema {
  const { $ref, allOf, ...rest } = schema
  if ($ref) {
    const target = root.$defs?.[$ref.replace('#/$defs/', '')] ?? {}
    return { ...resolve(target, root), ...rest }
  }
  if (allOf?.length === 1 && allOf[0]) return { ...resolve(allOf[0], root), ...rest }
  return schema
}

function controlFor(schema: Schema, root: Schema): { control: Control; nullable: boolean } {
  const variants = schema.anyOf ?? schema.oneOf
  if (variants) {
    const present = variants.filter((variant) => variant.type !== 'null').map((variant) => resolve(variant, root))
    const nullable = present.length < variants.length
    const [only] = present
    if (present.length === 1 && only) {
      const inner = controlFor(only, root)
      // A nullable switch or group has no empty state to save `null` with, so it needs another control.
      if (inner.control.kind === 'switch') return { control: { kind: 'choice', options: [true, false, null] }, nullable }
      if (inner.control.kind === 'group') return { control: { kind: 'json' }, nullable }
      if (inner.control.kind === 'choice') return { control: { kind: 'choice', options: [...inner.control.options, null] }, nullable }
      return { control: inner.control, nullable }
    }
    if (present.every((variant) => variant.enum || 'const' in variant)) {
      const options = present.flatMap((variant) => variant.enum ?? [variant.const ?? null])
      return { control: { kind: 'choice', options: nullable ? [...options, null] : options }, nullable }
    }
    return { control: { kind: 'json' }, nullable }
  }
  if (schema.enum) return { control: { kind: 'choice', options: schema.enum }, nullable: false }
  if ('const' in schema) return { control: { kind: 'choice', options: [schema.const ?? null] }, nullable: false }
  switch (schema.type) {
    case 'boolean':
      return { control: { kind: 'switch' }, nullable: false }
    case 'string':
      return { control: { kind: 'text' }, nullable: false }
    case 'integer':
    case 'number':
      return {
        control: { kind: 'number', integer: schema.type === 'integer', min: schema.minimum, max: schema.maximum },
        nullable: false,
      }
    case 'array': {
      const items = schema.items ? resolve(schema.items, root) : {}
      const strings = items.type === 'string' || Boolean(items.enum?.every((value) => typeof value === 'string'))
      return { control: strings ? { kind: 'lines' } : { kind: 'json' }, nullable: false }
    }
    case 'object':
      if (schema.properties) return { control: { kind: 'group', fields: fields(schema, root) }, nullable: false }
  }
  return { control: { kind: 'json' }, nullable: false }
}

/** A value as a choice or a hint shows it. */
export function shown(value: JsonValue | undefined): string {
  if (value === undefined) return ''
  if (value === null) return 'Not set'
  return typeof value === 'string' ? value : JSON.stringify(value)
}

export function same(left: JsonValue | undefined, right: JsonValue | undefined): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}
