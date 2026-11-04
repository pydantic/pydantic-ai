import { useContext } from 'react'
import { SessionContext } from '../api'
import type { PermissionMode, StatusInfo } from '../protocol'
import { ShieldIcon } from './icons'

const ORDER: readonly PermissionMode[] = ['full', 'auto', 'read-only']
const MODES: Record<PermissionMode, { label: string; title: string }> = {
  full: { label: 'Full access', title: 'Everything runs without asking.' },
  auto: {
    label: 'Auto',
    title: 'File edits and commands run; commands that delete, push, install, or reach the network ask first.',
  },
  'read-only': { label: 'Read only', title: 'Every file write and command asks first.' },
}

/** The session's permission mode, as `/permissions` sets it; changing it applies from the next tool call. */
export function PermissionsChip({ status, onError }: { status: StatusInfo | null; onError: (message: string) => void }) {
  const session = useContext(SessionContext)
  const mode = status?.permissions
  if (mode == null) return null
  const choose = (value: string) =>
    void session
      .setField({ kind: 'commands', name: 'permissions' }, 'run.permissions', value)
      .catch((error: unknown) => onError(error instanceof Error ? error.message : String(error)))
  return (
    <label className={`switch permissions ${mode === 'full' ? '' : 'on'}`} title={`Permissions: ${MODES[mode].title}`}>
      <ShieldIcon size={13} />
      <select aria-label="Permissions" value={mode} onChange={(event) => choose(event.target.value)}>
        {ORDER.map((value) => (
          <option key={value} value={value}>
            {MODES[value].label}
          </option>
        ))}
      </select>
    </label>
  )
}
