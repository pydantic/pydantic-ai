import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { ApiError, api } from '../api'
import type { JsonValue, PluginForm } from '../protocol'
import { type Field, fields, same, shown } from '../settings'
import { CheckIcon } from './icons'

/** Save one setting, or reset it to its default with `undefined`. */
type Save = (value: JsonValue | undefined) => void
type Invalid = (message: string) => void

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** A plugin's settings, one row per field of its settings model; each change saves at once, as its terminal menu does. */
export function SettingsForm({
  plugin,
  form,
  readOnly,
  onError,
}: {
  plugin: string
  form: PluginForm
  readOnly: boolean
  onError: (message: string) => void
}) {
  const rows = useMemo(() => fields(form.json_schema), [form.json_schema])
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState<string | null>(null)
  const [saved, setSaved] = useState<string | null>(null)

  const invalid = (key: string) => (text: string) => setErrors((current) => ({ ...current, [key]: text }))

  const save = (key: string) => async (value: JsonValue | undefined) => {
    setSaving(key)
    try {
      await (value === undefined ? api.pluginSettings(plugin, {}, [key]) : api.pluginSettings(plugin, { [key]: value }))
      setErrors((current) => Object.fromEntries(Object.entries(current).filter(([name]) => name !== key)))
      setSaved(key)
      window.setTimeout(() => setSaved((current) => (current === key ? null : current)), 1600)
    } catch (error) {
      const rejected = error instanceof ApiError ? Object.entries(error.fields) : []
      if (rejected.length === 0) {
        onError(message(error))
        return
      }
      // A path such as `token.name` belongs to the `token` row.
      const byRow = rejected.map(([path, text]): [string, string] => {
        const [row = path, ...rest] = path.split('.')
        return [row, rest.length ? `${rest.join('.')}: ${text}` : text]
      })
      // A model-level validator rejects with no field path, which no row shows.
      const known = new Set(rows.map((row) => row.key))
      const matched = byRow.filter(([row]) => known.has(row))
      if (matched.length === 0) {
        onError(rejected.map(([, text]) => text).join(' '))
        return
      }
      setErrors((current) => ({ ...current, ...Object.fromEntries(matched) }))
    } finally {
      setSaving(null)
    }
  }

  if (rows.length === 0) return <p className="plugin-empty">It takes no settings.</p>
  return (
    <div className="settings-form">
      {rows.map((row) => (
        <SettingRow
          key={row.key}
          field={row}
          // A model may save only what was chosen, as `compaction`'s does: a missing setting has its default.
          value={row.key in form.values ? form.values[row.key] : row.default}
          error={errors[row.key]}
          state={saving === row.key ? 'saving' : saved === row.key ? 'saved' : null}
          readOnly={readOnly || saving !== null}
          onSave={(value) => void save(row.key)(value)}
          onInvalid={invalid(row.key)}
        />
      ))}
    </div>
  )
}

const INLINE = new Set(['switch', 'choice', 'number'])

function SettingRow({
  field,
  value,
  error,
  state,
  readOnly,
  onSave,
  onInvalid,
}: {
  field: Field
  value: JsonValue | undefined
  error: string | undefined
  state: 'saving' | 'saved' | null
  readOnly: boolean
  onSave: Save
  onInvalid: Invalid
}) {
  const id = useId()
  const resettable = field.default !== undefined && !same(value, field.default)
  const inline = INLINE.has(field.control.kind)
  const editor = <Editor id={id} field={field} value={value} readOnly={readOnly} onSave={onSave} onInvalid={onInvalid} />
  return (
    <div className={`setting ${inline ? 'inline' : ''} ${error ? 'invalid' : ''}`}>
      <div className="setting-info">
        <div className="setting-head">
          <label className="setting-label" htmlFor={id}>
            {field.label}
          </label>
          <code className="setting-key">{field.key}</code>
          {state === 'saving' && <span className="setting-state">Saving…</span>}
          {state === 'saved' && (
            <span className="setting-state saved">
              <CheckIcon size={12} /> Saved
            </span>
          )}
          {resettable && !readOnly && (
            <button className="setting-reset" onClick={() => onSave(undefined)} title={`Reset to ${shown(field.default)}`}>
              Reset
            </button>
          )}
        </div>
        {field.description && <p className="setting-description">{field.description}</p>}
      </div>
      <div className="setting-control">{editor}</div>
      {error && (
        <p className="setting-error" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}

function Editor({
  id,
  field,
  value,
  readOnly,
  onSave,
  onInvalid,
}: {
  id: string
  field: Field
  value: JsonValue | undefined
  readOnly: boolean
  onSave: Save
  onInvalid: Invalid
}) {
  const control = field.control
  switch (control.kind) {
    case 'switch':
      return (
        <button
          id={id}
          className={`toggle ${value === true ? 'on' : ''}`}
          role="switch"
          aria-checked={value === true}
          disabled={readOnly}
          onClick={() => onSave(value !== true)}
        >
          <span />
        </button>
      )
    case 'choice': {
      const index = control.options.findIndex((option) => same(option, value))
      return (
        <select
          id={id}
          className="setting-select"
          value={index}
          disabled={readOnly}
          onChange={(event) => onSave(control.options[Number(event.target.value)] ?? null)}
        >
          {index < 0 && <option value={-1}>{shown(value)}</option>}
          {control.options.map((option, position) => (
            <option key={position} value={position}>
              {shown(option)}
            </option>
          ))}
        </select>
      )
    }
    case 'text':
      return (
        <Draft
          id={id}
          current={typeof value === 'string' ? value : ''}
          placeholder={field.nullable ? 'Not set' : ''}
          readOnly={readOnly}
          onCommit={(text) => onSave(text === '' && field.nullable ? null : text)}
          unchanged={(text) => same(text === '' && field.nullable ? null : text, value)}
        />
      )
    case 'number':
      return (
        <Draft
          id={id}
          type="number"
          current={typeof value === 'number' ? String(value) : ''}
          placeholder={field.nullable ? 'Not set' : ''}
          readOnly={readOnly}
          min={control.min}
          max={control.max}
          unchanged={(text) => (text === '' ? value === null || value === undefined : Number(text) === value)}
          onCommit={(text) => {
            if (text === '') {
              if (field.nullable) onSave(null)
              else onInvalid('Enter a number.')
              return
            }
            const number = Number(text)
            if (!Number.isFinite(number) || (control.integer && !Number.isInteger(number))) {
              onInvalid(control.integer ? 'Enter a whole number.' : 'Enter a number.')
              return
            }
            onSave(number)
          }}
        />
      )
    case 'lines': {
      const current = Array.isArray(value) ? value.map((line) => shown(line)).join('\n') : ''
      const parse = (text: string): JsonValue => {
        const lines = text
          .split('\n')
          .map((line) => line.trim())
          .filter(Boolean)
        return lines.length === 0 && field.nullable ? null : lines
      }
      return (
        <Draft
          id={id}
          multiline
          current={current}
          placeholder={field.nullable ? 'Not set: one per line' : 'One per line'}
          readOnly={readOnly}
          unchanged={(text) => same(parse(text), value ?? null)}
          onCommit={(text) => onSave(parse(text))}
        />
      )
    }
    case 'group': {
      const object = value && typeof value === 'object' && !Array.isArray(value) ? value : {}
      // A model with one field, such as a `/keys` reference by name, reads as that field.
      const [only] = control.fields
      if (control.fields.length === 1 && only) {
        return (
          <Editor
            id={id}
            field={only}
            value={object[only.key]}
            readOnly={readOnly}
            onSave={(inner) => onSave({ ...object, [only.key]: inner ?? null })}
            onInvalid={onInvalid}
          />
        )
      }
      return (
        <div className="setting-group" id={id}>
          {control.fields.map((inner) => (
            <label key={inner.key} className="setting-group-field">
              <span>{inner.label}</span>
              <Editor
                id={`${id}-${inner.key}`}
                field={inner}
                value={object[inner.key]}
                readOnly={readOnly}
                onSave={(next) => onSave({ ...object, [inner.key]: next ?? null })}
                onInvalid={onInvalid}
              />
            </label>
          ))}
        </div>
      )
    }
    case 'json':
      return (
        <Draft
          id={id}
          multiline
          code
          current={value === undefined ? '' : JSON.stringify(value, null, 2)}
          placeholder="JSON"
          readOnly={readOnly}
          unchanged={(text) => {
            try {
              return same(JSON.parse(text) as JsonValue, value)
            } catch {
              return false
            }
          }}
          onCommit={(text) => {
            try {
              onSave(JSON.parse(text) as JsonValue)
            } catch {
              onInvalid('Not valid JSON.')
            }
          }}
        />
      )
  }
}

/** A text field that saves on Enter (Mod+Enter when multiline) or when it loses focus; Esc puts the saved value back. */
export function Draft({
  id,
  current,
  placeholder,
  readOnly,
  unchanged,
  onCommit,
  multiline = false,
  code = false,
  type = 'text',
  min,
  max,
}: {
  id: string
  current: string
  placeholder: string
  readOnly: boolean
  unchanged: (text: string) => boolean
  onCommit: (text: string) => void
  multiline?: boolean
  code?: boolean
  type?: 'text' | 'number' | 'password'
  min?: number
  max?: number
}) {
  const [draft, setDraft] = useState(current)
  const [editing, setEditing] = useState(false)
  // Esc blurs the field, which would otherwise save the edit it just undid.
  const undone = useRef(false)
  // A save, or a change from the terminal, shows unless the user is mid-edit.
  useEffect(() => {
    if (!editing) setDraft(current)
  }, [current, editing])

  const commit = () => {
    setEditing(false)
    if (undone.current) {
      undone.current = false
      return
    }
    if (!unchanged(draft.trim())) onCommit(draft.trim())
  }
  const props = {
    id,
    value: draft,
    placeholder,
    disabled: readOnly,
    className: `setting-input ${code ? 'code' : ''}`,
    onFocus: () => setEditing(true),
    onBlur: commit,
    onChange: (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => setDraft(event.target.value),
    onKeyDown: (event: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      if (event.key === 'Escape') {
        // The panel leaves Esc in a text field to it: the first one undoes the edit, the next one closes the panel.
        undone.current = true
        setDraft(current)
        event.currentTarget.blur()
      } else if (event.key === 'Enter' && (!multiline || event.metaKey || event.ctrlKey)) {
        event.preventDefault()
        event.currentTarget.blur()
      }
    },
  }
  if (multiline) return <textarea {...props} rows={Math.min(8, Math.max(2, draft.split('\n').length))} spellCheck={false} />
  return <input {...props} type={type} min={min} max={max} spellCheck={false} />
}
