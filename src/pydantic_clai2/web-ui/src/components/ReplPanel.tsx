import { memo, useLayoutEffect, useRef } from 'react'
import { cell } from '../code'
import { highlight } from '../markdown'
import type { Item, StatusInfo, SubCall } from '../protocol'
import type { Entry } from '../state'
import { codeCalls, codeResult } from './ToolCard'
import { CloseIcon, TerminalIcon } from './icons'

interface Cell {
  item: Item
  number: number
  /** The turn this cell starts, shown as a divider above it. */
  turn: number | null
  /** Whether this cell's `restart: true` cleared the REPL before it ran. */
  restart: boolean
}

/** The conversation's `run_code` cells, numbered as one REPL kept with the session until a restart clears it. */
function cells(entries: Entry[]): Cell[] {
  const found: Cell[] = []
  let turn = 0
  let number = 0
  let shownTurn = 0
  for (const entry of entries) {
    if (entry.kind === 'user') {
      turn += 1
    } else if (entry.kind === 'tool' && entry.name === 'run_code') {
      const restart = cell(entry.args).restart
      number = restart ? 1 : number + 1
      found.push({ item: entry, number, turn: turn !== shownTurn ? Math.max(turn, 1) : null, restart })
      shownTurn = turn
    }
  }
  return found
}

function callSubject(call: SubCall): string {
  try {
    const args = JSON.parse(call.args) as Record<string, unknown>
    const subject = args.command ?? args.pattern ?? args.path ?? args.query ?? ''
    return typeof subject === 'string' ? subject : JSON.stringify(subject)
  } catch {
    return ''
  }
}

const CellView = memo(function CellView({ item, number, turn, restart }: Cell) {
  const { code } = cell(item.args)
  const running = item.state === 'running'
  const result = item.text ? codeResult(item.text) : ''
  // History keeps no sandbox calls, so a saved cell lists the calls its code makes instead.
  const seen = item.calls.length > 0 || running
  const calls: SubCall[] = seen
    ? item.calls
    : codeCalls(code).map((call, index) => ({
        id: `${item.id}-${index}`,
        name: call.name,
        args: JSON.stringify({ command: call.argument }),
        state: 'done',
        text: '',
        output: '',
      }))
  return (
    <>
      {turn !== null && <div className="repl-divider">Turn {turn}</div>}
      {restart && <div className="repl-divider restart">REPL restarted</div>}
      <section className={`repl-cell ${item.state}`}>
        <div className="repl-prompt">
          <span className="repl-label">In [{number}]</span>
          {running && <span className="shimmer repl-running">running</span>}
          {item.state === 'error' && <span className="badge error">error</span>}
          {item.state === 'interrupted' && <span className="badge">interrupted</span>}
        </div>
        <pre className="repl-code">
          <code className="hljs" dangerouslySetInnerHTML={{ __html: highlight(code, 'python') }} />
          {running && <span className="repl-caret" />}
        </pre>
        {calls.length > 0 && (
          <ul className={`repl-calls ${seen ? '' : 'derived'}`} title={seen ? undefined : 'From the code; saved sessions keep no call log'}>
            {calls.map((call) => (
              <li key={call.id} className={call.state}>
                <span className={`repl-dot ${call.state}`} />
                <span className="repl-call-name">{call.name}</span>
                <span className="repl-call-subject">{callSubject(call)}</span>
                {call.id.includes('__spec_') && <span className="badge" title="Started speculatively">spec</span>}
              </li>
            ))}
          </ul>
        )}
        {result && (
          <>
            <div className="repl-label repl-out">Out [{number}]</div>
            <pre className={`repl-result ${item.state === 'error' ? 'error' : ''}`}>{result}</pre>
          </>
        )}
      </section>
    </>
  )
})

/** The `run_code` REPL: every cell the model ran in code mode, its sandbox calls, and its output. */
export function ReplPanel({
  entries,
  status,
  onClose,
}: {
  entries: Entry[]
  status: StatusInfo | null
  onClose: () => void
}) {
  const found = cells(entries)
  const scroller = useRef<HTMLDivElement>(null)
  const last = found[found.length - 1]?.item
  useLayoutEffect(() => {
    const element = scroller.current
    if (element) element.scrollTop = element.scrollHeight
  }, [last])
  const codeMode = status?.code_mode || status?.speculation
  return (
    <aside className="repl" aria-label="run_code REPL">
      <header className="repl-head">
        <TerminalIcon size={15} />
        <span className="repl-title">REPL</span>
        <span className={`badge ${codeMode ? 'on' : ''}`}>
          {status?.speculation ? 'speculative' : codeMode ? 'code mode' : 'code mode off'}
        </span>
        <button className="icon-button" onClick={onClose} title="Hide the REPL (⌘J)" aria-label="Hide the REPL">
          <CloseIcon size={14} />
        </button>
      </header>
      <div className="repl-body" ref={scroller}>
        {found.length === 0 ? (
          <div className="repl-empty">
            {codeMode
              ? 'No run_code cells yet. They appear here as the model writes them.'
              : 'Code mode is off, so tools run natively. Turn it on in the composer to use the REPL.'}
          </div>
        ) : (
          found.map((entry) => <CellView key={entry.item.id} {...entry} />)
        )}
      </div>
      <footer className="repl-foot">
        Read-only. The REPL is kept with the session between turns and on resume; <code>restart</code> clears it.
      </footer>
    </aside>
  )
}
