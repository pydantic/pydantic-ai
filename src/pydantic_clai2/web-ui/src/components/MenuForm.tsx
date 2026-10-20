import { useCallback, useContext, useEffect, useId, useRef, useState } from 'react'
import { SessionContext, type MenuTarget, type SessionApi } from '../api'
import type { MenuEdited, MenuField, MenuInfo } from '../protocol'
import { Draft } from './SettingsForm'
import { CheckIcon, ChevronIcon } from './icons'

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

const NOT_SET = '(not set)'
const OTHER = '\u0000other'

/** `/set`'s keys read `group.name`; a key without a dot is a group of its own. */
const keyGroup = (field: MenuField) => field.key.split('.')[0] ?? field.key

function groups(fields: MenuField[], group: (field: MenuField) => string): [string, MenuField[]][] {
  const found = new Map<string, MenuField[]>()
  for (const field of fields) found.set(group(field), [...(found.get(group(field)) ?? []), field])
  return [...found]
}

const sentence = (text: string) => {
  const words = text.replace(/_/g, ' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

/** A declared settings menu, a plugin's or a command's: the rows its terminal menu lists, each saved as it changes. */
export function MenuForm({
  target,
  version,
  grouped = false,
  group,
  titled = false,
  opened,
  onError,
  onNotice,
}: {
  target: MenuTarget
  /** Bumped when a command may have changed what the menu edits; the rows are read again. */
  version: number
  /** Rows keyed `group.name` show under a heading per group, labelled by name, as cards. */
  grouped?: boolean
  /** With `grouped`, the heading each row shows under instead, the rows keeping their own labels. */
  group?: (field: MenuField) => string
  /** Show the menu's title above its rows, and label each row by the last part of its key. */
  titled?: boolean
  /** The key of the row whose menu shows open at first, such as the current model's. */
  opened?: (menu: MenuInfo) => string | undefined
  onError: (message: string) => void
  onNotice: (text: string) => void
}) {
  const [menu, setMenu] = useState<MenuInfo | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [saved, setSaved] = useState<string | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})
  // `null` until the rows first load, when `opened` picks the row to show open.
  const [open, setOpen] = useState<string | null>(null)
  const request = useRef(0)
  const path = target.path?.join('\u0000')

  // A command's menu acts on the shown session, so `/model` picks its model, not another session's.
  const session = useContext(SessionContext)
  // The session shown now: an edit answered after a switch must not show the last session's rows.
  const shown = useRef(session)
  shown.current = session

  const load = useCallback(async () => {
    const id = ++request.current
    try {
      const found = await session.menu(target)
      if (id !== request.current) return
      setMenu(found)
      const first = opened?.(found)
      setOpen((current) => current ?? found.fields.find((field) => field.menu && field.key === first)?.key ?? '')
    } catch (error) {
      onError(message(error))
    }
    // A fresh target each render names the same menu.
  }, [session, target.kind, target.name, path, onError])

  useEffect(() => {
    void load()
  }, [load, version])

  const edited = (key: string, result: MenuEdited, owner: SessionApi) => {
    if (owner === shown.current) {
      ++request.current
      setMenu(result.menu)
    }
    setErrors((current) => Object.fromEntries(Object.entries(current).filter(([name]) => name !== key)))
    const text = result.messages.filter(Boolean).join('\n')
    // An action the user backed out of says nothing, and saved nothing.
    if (!text) return
    setSaved(key)
    window.setTimeout(() => setSaved((current) => (current === key ? null : current)), 1600)
    onNotice(text)
  }

  const run = async (key: string, edit: () => Promise<MenuEdited>, inline: boolean) => {
    setBusy(key)
    const owner = session
    try {
      edited(key, await edit(), owner)
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
  const row = (field: MenuField, named = false) => (
    <MenuRow
      key={field.key}
      field={field}
      named={named}
      error={errors[field.key]}
      state={busy === field.key ? 'saving' : saved === field.key ? 'saved' : null}
      disabled={busy !== null}
      open={open === field.key}
      onSet={(value) => void run(field.key, () => session.setField(target, field.key, value), true)}
      onAction={() => void run(field.key, () => session.runAction(target, field.key), false)}
      onOpen={() => setOpen((current) => (current === field.key ? '' : field.key))}
    >
      {field.menu && open === field.key && (
        <div className="setting-submenu">
          <MenuForm
            target={{ ...target, path: [...(target.path ?? []), field.key] }}
            version={version}
            onError={onError}
            onNotice={onNotice}
          />
        </div>
      )}
    </MenuRow>
  )
  if (!grouped)
    return (
      <div className="settings-form">
        {titled && <h2 className="menu-title">{menu.title}</h2>}
        {menu.fields.map((field) => row(field, titled))}
      </div>
    )
  return (
    <div className="settings-groups">
      {groups(menu.fields, group ?? keyGroup).map(([title, fields]) => (
        <section key={title} className="settings-group">
          <h3 className="settings-group-title">{sentence(title)}</h3>
          <div className="settings-card">{fields.map((field) => row(field, !group))}</div>
        </section>
      ))}
    </div>
  )
}

/** Whether a row's choices are a yes/no switch, as a boolean setting's are. A `named` row, one of `/set`'s, takes no other
text even where its menu offers typing one; a plugin's row with typed values allowed keeps its list. */
const isSwitch = (field: MenuField, named: boolean) =>
  (named || !field.allow_custom) &&
  field.choices.length === 2 &&
  field.choices.every((choice) => choice.value === 'true' || choice.value === 'false')

function MenuRow({
  field,
  named,
  error,
  state,
  disabled,
  open,
  onSet,
  onAction,
  onOpen,
  children,
}: {
  field: MenuField
  /** Label the row by the last part of its key, with the whole key beside it, as a grouped menu does. */
  named: boolean
  error: string | undefined
  state: 'saving' | 'saved' | null
  disabled: boolean
  /** Whether the menu the row opens shows, as `children`, below it. */
  open: boolean
  /** Save a value; `null` resets the row to its default. */
  onSet: (value: string | null) => void
  onAction: () => void
  onOpen: () => void
  children?: React.ReactNode
}) {
  const id = useId()
  const label = (value: string) => field.choices.find((choice) => choice.value === value)?.label ?? value
  const resettable = !field.menu && field.value !== field.default
  const inline =
    named || isSwitch(field, named) || field.action || field.menu || (field.choices.length > 0 && !field.allow_custom)
  const titled = named && field.label === field.key
  return (
    <div className={`setting ${inline ? 'inline' : ''} ${error ? 'invalid' : ''} ${open ? 'open' : ''}`}>
      <div className="setting-info">
        <div className="setting-head">
          {/* An action's button says what it does, rather than taking the row's label as its name. */}
          <label className="setting-label" htmlFor={field.action ? undefined : id}>
            {titled ? sentence(field.key.split('.').pop() ?? field.key) : field.label}
          </label>
          {titled && <code className="setting-key">{field.key}</code>}
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
        {field.menu ? (
          <div className="setting-action">
            <span className="setting-current" title={field.value}>
              {field.value}
            </span>
            <button
              id={id}
              className="ghost small setting-open"
              aria-expanded={open}
              aria-label={`${open ? 'Hide' : 'Show'} ${field.label}`}
              onClick={onOpen}
            >
              Settings <ChevronIcon size={13} />
            </button>
          </div>
        ) : (
          <Control
            id={id}
            field={field}
            named={named}
            label={label}
            disabled={disabled}
            onSet={onSet}
            onAction={onAction}
          />
        )}
      </div>
      {error && (
        <p className="setting-error" role="alert">
          {error}
        </p>
      )}
      {children}
    </div>
  )
}

function Control({
  id,
  field,
  named,
  label,
  disabled,
  onSet,
  onAction,
}: {
  id: string
  field: MenuField
  named: boolean
  label: (value: string) => string
  disabled: boolean
  onSet: (value: string | null) => void
  onAction: () => void
}) {
  const [other, setOther] = useState(false)
  if (field.action) {
    // A row says what its button does, or else one with nothing to show, such as one that adds something, opens.
    const verb = field.button ? field.button.replace(/…$/, '') : field.value || field.secret ? 'Change' : 'Open'
    return (
      <div className="setting-action">
        <span className="setting-current">{field.secret ? '' : label(field.value)}</span>
        <button id={id} className="ghost small" disabled={disabled} onClick={onAction} aria-label={`${verb} ${field.label}`}>
          {verb}…
        </button>
      </div>
    )
  }
  if (isSwitch(field, named)) {
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
