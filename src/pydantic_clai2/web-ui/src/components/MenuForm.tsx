import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { api } from '../api'
import type { MenuEdited, MenuField, MenuInfo } from '../protocol'
import { Draft } from './SettingsForm'
import { CheckIcon } from './icons'

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

const NOT_SET = '(not set)'
const OTHER = '\u0000other'

/** A plugin's declared settings menu: the rows its terminal menu lists, each saved as it changes. */
export function MenuForm({
  plugin,
  version,
  onError,
  onNotice,
}: {
  plugin: string
  /** Bumped when a command may have changed the plugin; the rows are read again. */
  version: number
  onError: (message: string) => void
  onNotice: (text: string) => void
}) {
  const [menu, setMenu] = useState<MenuInfo | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [saved, setSaved] = useState<string | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const request = useRef(0)

  const load = useCallback(async () => {
    const id = ++request.current
    try {
      const found = await api.menu(plugin)
      if (id === request.current) setMenu(found)
    } catch (error) {
      onError(message(error))
    }
  }, [plugin, onError])

  useEffect(() => {
    void load()
  }, [load, version])

  const edited = (key: string, result: MenuEdited) => {
    ++request.current
    setMenu(result.menu)
    setErrors((current) => Object.fromEntries(Object.entries(current).filter(([name]) => name !== key)))
    setSaved(key)
    window.setTimeout(() => setSaved((current) => (current === key ? null : current)), 1600)
    const text = result.messages.filter(Boolean).join('\n')
    if (text) onNotice(text)
  }

  const run = async (key: string, edit: () => Promise<MenuEdited>, inline: boolean) => {
    setBusy(key)
    try {
      edited(key, await edit())
    } catch (error) {
      // A value the menu rejects shows by its row; anything else, such as a failed sign-in, as a toast.
      if (inline) setErrors((current) => ({ ...current, [key]: message(error) }))
      else onError(message(error))
    } finally {
      setBusy(null)
    }
  }

  if (menu === null) return <p className="plugin-empty">Loading its settings…</p>
  if (menu.fields.length === 0) return <p className="plugin-empty">It takes no settings.</p>
  return (
    <div className="settings-form">
      {menu.fields.map((field) => (
        <MenuRow
          key={field.key}
          field={field}
          error={errors[field.key]}
          state={busy === field.key ? 'saving' : saved === field.key ? 'saved' : null}
          disabled={busy !== null}
          onSet={(value) => void run(field.key, () => api.setField(plugin, field.key, value), true)}
          onAction={() => void run(field.key, () => api.runAction(plugin, field.key), false)}
        />
      ))}
    </div>
  )
}

/** Whether a row's choices are a yes/no switch, as a boolean setting's are. */
const isSwitch = (field: MenuField) =>
  !field.allow_custom &&
  field.choices.length === 2 &&
  field.choices.every((choice) => choice.value === 'true' || choice.value === 'false')

function MenuRow({
  field,
  error,
  state,
  disabled,
  onSet,
  onAction,
}: {
  field: MenuField
  error: string | undefined
  state: 'saving' | 'saved' | null
  disabled: boolean
  /** Save a value; `null` resets the row to its default. */
  onSet: (value: string | null) => void
  onAction: () => void
}) {
  const id = useId()
  const label = (value: string) => field.choices.find((choice) => choice.value === value)?.label ?? value
  const resettable = field.value !== field.default
  const inline = isSwitch(field) || field.action || (field.choices.length > 0 && !field.allow_custom)
  return (
    <div className={`setting ${inline ? 'inline' : ''} ${error ? 'invalid' : ''}`}>
      <div className="setting-info">
        <div className="setting-head">
          {/* An action's button says what it does, rather than taking the row's label as its name. */}
          <label className="setting-label" htmlFor={field.action ? undefined : id}>
            {field.label}
          </label>
          {state === 'saving' && <span className="setting-state">{field.action ? 'In progress…' : 'Saving…'}</span>}
          {state === 'saved' && (
            <span className="setting-state saved">
              <CheckIcon size={12} /> Saved
            </span>
          )}
          {resettable && (
            <button
              className="setting-reset"
              disabled={disabled}
              onClick={() => onSet(null)}
              title={`Reset to ${label(field.default)}`}
            >
              Reset
            </button>
          )}
        </div>
        {field.description && <p className="setting-description">{field.description}</p>}
        {field.note && <p className="setting-note">{field.note}</p>}
      </div>
      <div className="setting-control">
        <Control id={id} field={field} label={label} disabled={disabled} onSet={onSet} onAction={onAction} />
      </div>
      {error && (
        <p className="setting-error" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}

function Control({
  id,
  field,
  label,
  disabled,
  onSet,
  onAction,
}: {
  id: string
  field: MenuField
  label: (value: string) => string
  disabled: boolean
  onSet: (value: string | null) => void
  onAction: () => void
}) {
  const [other, setOther] = useState(false)
  if (field.action) {
    return (
      <div className="setting-action">
        <span className="setting-current">{field.secret ? '' : label(field.value)}</span>
        <button id={id} className="ghost small" disabled={disabled} onClick={onAction} aria-label={`Change ${field.label}`}>
          Change…
        </button>
      </div>
    )
  }
  if (isSwitch(field)) {
    const on = field.value === 'true'
    return (
      <button
        id={id}
        className={`toggle ${on ? 'on' : ''}`}
        role="switch"
        aria-checked={on}
        disabled={disabled}
        onClick={() => onSet(on ? 'false' : 'true')}
      >
        <span />
      </button>
    )
  }
  const listed = field.choices.some((choice) => choice.value === field.value)
  if (field.choices.length > 0 && !other) {
    return (
      <select
        id={id}
        className="setting-select"
        value={listed ? field.value : field.allow_custom ? OTHER : field.value}
        disabled={disabled}
        onChange={(event) => (event.target.value === OTHER ? setOther(true) : onSet(event.target.value))}
      >
        {!listed && !field.allow_custom && <option value={field.value}>{label(field.value)}</option>}
        {field.choices.map((choice) => (
          <option key={choice.value} value={choice.value}>
            {choice.label}
          </option>
        ))}
        {field.allow_custom && (
          <option value={OTHER}>{listed || field.value === NOT_SET ? 'Other…' : `${field.value} (typed)`}</option>
        )}
      </select>
    )
  }
  const current = field.secret || field.value === NOT_SET ? '' : field.value
  return (
    <Draft
      id={id}
      type={field.secret ? 'password' : 'text'}
      current={current}
      placeholder={field.secret ? 'Enter a new secret' : field.value === NOT_SET ? 'Not set' : ''}
      readOnly={disabled}
      unchanged={(text) => (field.secret ? text === '' : text === current)}
      onCommit={(text) => {
        setOther(false)
        // Empty text resets, as an empty edit in the terminal menu does.
        onSet(text === '' ? null : text)
      }}
    />
  )
}
