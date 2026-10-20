import { useContext } from 'react'
import { SessionContext } from '../api'
import type { StatusInfo, Switch } from '../protocol'
import { CodeIcon, BoltIcon } from './icons'

/** Code mode and speculative execution, as `/set` and Ctrl+X Ctrl+S change them in the terminal. */
export function Switches({
  status,
  running,
  onError,
}: {
  status: StatusInfo | null
  running: boolean
  onError: (message: string) => void
}) {
  const session = useContext(SessionContext)
  if (status?.code_mode == null || status.speculation == null) return null
  const flip = (key: Switch, on: boolean) =>
    void session
      .setSwitch(key, on)
      .catch((error: unknown) => onError(error instanceof Error ? error.message : String(error)))
  const later = running ? ' The running turn keeps its tools; this applies from the next prompt.' : ''
  const codeOn = status.code_mode || status.speculation
  return (
    <div className="switches" role="group" aria-label="Run switches">
      <button
        className={`switch ${codeOn ? 'on' : ''}`}
        aria-pressed={codeOn}
        aria-disabled={status.speculation}
        onClick={() => {
          if (!status.speculation) flip('run.code_mode', !status.code_mode)
        }}
        title={
          status.speculation
            ? 'Code mode is on while speculative execution is.'
            : `Code mode ${status.code_mode ? 'on' : 'off'}: tools run inside one sandboxed run_code script.${later}`
        }
      >
        <CodeIcon size={13} />
        <span>Code</span>
      </button>
      <button
        className={`switch ${status.speculation ? 'on' : ''}`}
        aria-pressed={status.speculation}
        onClick={() => flip('run.speculative_code_mode', !status.speculation)}
        title={`Speculative execution ${status.speculation ? 'on' : 'off'}: read-only calls start before the model reaches them.${later}`}
      >
        <BoltIcon size={13} />
        <span>Speculative</span>
      </button>
    </div>
  )
}
