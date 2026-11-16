import { memo, useContext, useLayoutEffect, useRef, useState } from 'react'
import { SessionContext } from '../api'
import { cell } from '../code'
import { highlight } from '../markdown'
import type { Item, StatusInfo, SubCall } from '../protocol'
import type { Entry } from '../state'
import { JumpToLatest, useStickToBottom } from '../stickToBottom'
import { codeCalls, codeResult } from './ToolCard'
import { CloseIcon, TerminalIcon } from './icons'

interface Cell {
  item: Item
  number: number
  /** The turn this cell starts, shown as a divider above it. */
  turn: number | null
  /** Whether this cell's `restart: true` cleared Monty's state before it ran. */
  restart: boolean
}

/** The conversation's `run_code` cells, numbered as one Monty state kept with the session until a restart clears it. */
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
      {turn !== null && <div className="monty-divider">Turn {turn}</div>}
      {restart && <div className="monty-divider restart">Monty restarted</div>}
      <section className={`monty-cell ${item.state}`}>
        <div className="monty-prompt">
          <span className="monty-label">In [{number}]</span>
          {item.by_user && <span className="badge on">you</span>}
          {running && <span className="shimmer monty-running">running</span>}
          {item.state === 'error' && <span className="badge error">error</span>}
          {item.state === 'interrupted' && <span className="badge">interrupted</span>}
        </div>
        <pre className="monty-code">
          <code className="hljs" dangerouslySetInnerHTML={{ __html: highlight(code, 'python') }} />
          {running && <span className="monty-caret" />}
        </pre>
        {calls.length > 0 && (
          <ul className={`monty-calls ${seen ? '' : 'derived'}`} title={seen ? undefined : 'From the code; saved sessions keep no call log'}>
            {calls.map((call) => (
              <li key={call.id} className={call.state}>
                <span className={`monty-dot ${call.state}`} />
                <span className="monty-call-name">{call.name}</span>
                <span className="monty-call-subject">{callSubject(call)}</span>
                {call.id.includes('__spec_') && <span className="badge" title="Started speculatively">spec</span>}
              </li>
            ))}
          </ul>
        )}
        {result && (
          <>
            <div className="monty-label monty-out">Out [{number}]</div>
            <pre className={`monty-result ${item.state === 'error' ? 'error' : ''}`}>{result}</pre>
          </>
        )}
      </section>
    </>
  )
})

/** Run your own cell in Monty with `/monty`: Enter runs it, Shift+Enter adds a line. */
function CellInput({ disabled, reason, onError }: { disabled: boolean; reason: string; onError: (message: string) => void }) {
  const [code, setCode] = useState('')
  const area = useRef<HTMLTextAreaElement>(null)
  const session = useContext(SessionContext)
  useLayoutEffect(() => {
    const element = area.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`
  }, [code])
  const run = async () => {
    const text = code.trim()
    if (!text || disabled) return
    setCode('')
    try {
      await session.send(`/monty ${text}`)
    } catch (error) {
      setCode(text)
      onError(error instanceof Error ? error.message : String(error))
    }
  }
  return (
    <div className={`monty-input ${disabled ? 'disabled' : ''}`} title={disabled ? reason : undefined}>
      <span className="monty-label">&gt;&gt;&gt;</span>
      <textarea
        ref={area}
        rows={1}
        value={code}
        disabled={disabled}
        placeholder={disabled ? reason : 'Run Python in Monty. The model hears about it on your next prompt.'}
        aria-label="Run a Monty cell"
        spellCheck={false}
        onChange={(event) => setCode(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault()
            void run()
          }
        }}
      />
    </div>
  )
}

/** Monty, code mode's `run_code` sandbox: every cell the model ran, its sandbox calls, and its output. */
export function MontyPanel({
  entries,
  status,
  running,
  onClose,
  onError,
}: {
  entries: Entry[]
  status: StatusInfo | null
  running: boolean
  onClose: () => void
  onError: (message: string) => void
}) {
  const found = cells(entries)
  const { scroller, behind, jump } = useStickToBottom<HTMLDivElement>(
    found[found.length - 1]?.item,
    entries.findLast((entry) => entry.kind === 'user'),
  )
  const codeMode = status?.code_mode || status?.speculation
  return (
    <aside className="monty" aria-label="Monty, the run_code sandbox">
      <header className="monty-head">
        <TerminalIcon size={15} />
        <span className="monty-title">Monty</span>
        <span className={`badge ${codeMode ? 'on' : ''}`}>
          {status?.speculation ? 'speculative' : codeMode ? 'code mode' : 'code mode off'}
        </span>
        <button className="icon-button" onClick={onClose} title="Hide the Monty panel (⌘J)" aria-label="Hide the Monty panel">
          <CloseIcon size={14} />
        </button>
      </header>
      <div className="monty-body" ref={scroller}>
        {found.length === 0 ? (
          <div className="monty-empty">
            {codeMode
              ? 'No run_code cells yet. They appear here as the model writes them.'
              : 'Code mode is off, so tools run natively. Turn it on in the composer to use Monty.'}
          </div>
        ) : (
          found.map((entry) => <CellView key={entry.item.id} {...entry} />)
        )}
        <JumpToLatest shown={behind} onJump={jump} />
      </div>
      <CellInput
        disabled={!codeMode || running}
        reason={!codeMode ? 'Turn code mode on to use Monty' : 'Cells run between turns'}
        onError={onError}
      />
      <footer className="monty-foot">
        Kept with the session between turns and on resume; <code>restart</code> clears it.
      </footer>
    </aside>
  )
}
