import { memo, useState } from 'react'
import { highlight } from '../markdown'
import type { Diff, Item } from '../protocol'
import { DiffView, diffStats } from './DiffView'
import { ChevronIcon, FileIcon, PencilIcon, QuestionIcon, SearchIcon, TerminalIcon, ToolIcon } from './icons'

type Args = Record<string, unknown>

function parseArgs(args: string): Args {
  try {
    const value: unknown = JSON.parse(args)
    return value && typeof value === 'object' && !Array.isArray(value) ? (value as Args) : {}
  } catch {
    return {}
  }
}

const str = (value: unknown): string => (typeof value === 'string' ? value : value == null ? '' : JSON.stringify(value))
const base = (path: string) => path.split('/').filter(Boolean).pop() ?? path

export interface Call {
  name: string
  argument: string
}

const CALL = /\b(read_file|list_files|grep|glob|shell|write_file|edit_file)\(\s*(?:\w+\s*=\s*)?(?:[rbf]?(["'])((?:\\.|(?!\2).)*)\2)?/g

/** The tool calls inside a `run_code` script, with their first literal argument, as code mode makes them. */
export function codeCalls(code: string): Call[] {
  return [...code.matchAll(CALL)].map((match) => ({ name: match[1] ?? '', argument: (match[3] ?? '').replace(/\\(.)/g, '$1') }))
}

interface Summary {
  icon: React.ReactNode
  verb: string
  subject: string
  mono: boolean
}

function summarize(item: Item, args: Args): Summary {
  const running = item.state === 'running'
  switch (item.name) {
    case 'shell':
      return { icon: <TerminalIcon size={14} />, verb: running ? 'Running' : 'Ran', subject: str(args.command), mono: true }
    case 'read_file':
      return { icon: <FileIcon size={14} />, verb: running ? 'Reading' : 'Read', subject: base(str(args.path)), mono: false }
    case 'write_file':
    case 'edit_file':
      return {
        icon: <PencilIcon size={14} />,
        verb: running ? 'Editing' : item.name === 'write_file' ? 'Wrote' : 'Edited',
        subject: base(str(args.path)),
        mono: false,
      }
    case 'grep':
      return { icon: <SearchIcon size={14} />, verb: running ? 'Searching for' : 'Searched for', subject: str(args.pattern), mono: true }
    case 'list_files':
      return { icon: <FileIcon size={14} />, verb: running ? 'Listing' : 'Listed', subject: str(args.path) || '.', mono: false }
    case 'run_code': {
      const calls = codeCalls(str(args.code))
      const [only] = calls
      if (calls.length === 1 && only) {
        // A script around one call reads as that call.
        const key = only.name === 'shell' ? 'command' : only.name === 'grep' ? 'pattern' : 'path'
        return summarize({ ...item, name: only.name }, { [key]: only.argument })
      }
      const first = str(args.code).split('\n').find((line) => line.trim() && !line.trim().startsWith('#'))
      return { icon: <TerminalIcon size={14} />, verb: running ? 'Running' : 'Ran', subject: first?.trim() ?? 'code', mono: true }
    }
    case 'ask_user_question':
      return { icon: <QuestionIcon size={14} />, verb: running ? 'Asking' : 'Asked', subject: 'you a question', mono: false }
    default:
      return { icon: <ToolIcon size={14} />, verb: running ? 'Calling' : 'Called', subject: item.name, mono: true }
  }
}

const lines = (text: string, sign: string) =>
  text
    .replace(/\n$/, '')
    .split('\n')
    .map((line) => sign + line)
    .join('\n')

/** The diffs a replayed edit or write made, rebuilt from its arguments; live runs send the real ones. */
function argumentDiffs(item: Item, args: Args): Diff[] {
  if (item.diffs.length > 0 || item.state === 'error') return item.diffs
  const path = str(args.path)
  if (item.name === 'write_file' && typeof args.content === 'string') {
    const content = args.content.split('\n').slice(0, 400).join('\n')
    return [{ path, diff: `@@ -0,0 +1 @@\n${lines(content, '+')}`, truncated: content.length < args.content.length }]
  }
  if (item.name !== 'edit_file') return []
  const replacements = Array.isArray(args.replacements)
    ? (args.replacements as unknown[])
    : [{ old_text: args.old_text, new_text: args.new_text }]
  return replacements
    .filter((r): r is { old_text: string; new_text: string } => {
      const value = r as Record<string, unknown>
      return typeof value.old_text === 'string' && typeof value.new_text === 'string'
    })
    .map((r) => ({ path, diff: `${lines(r.old_text, '-')}\n${lines(r.new_text, '+')}`, truncated: false }))
}

/** What `run_code` returned, as its printed output, rather than the JSON envelope. */
function codeResult(text: string): string {
  try {
    const value: unknown = JSON.parse(text)
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      const { output, result } = value as Record<string, unknown>
      return [str(output), result === undefined ? '' : `→ ${str(result)}`].filter(Boolean).join('\n')
    }
  } catch {
    // Not JSON: show it as it is.
  }
  return text
}

/** One tool call as a quiet row; open it for the arguments, output, diff, and result. */
export const ToolCard = memo(function ToolCard({ item: raw }: { item: Item }) {
  const [open, setOpen] = useState(false)
  const args = parseArgs(raw.args)
  const item = { ...raw, diffs: argumentDiffs(raw, args) }
  const summary = summarize(item, args)
  const stats = item.diffs.reduce(
    (total, diff) => {
      const { added, removed } = diffStats(diff.diff)
      return { added: total.added + added, removed: total.removed + removed }
    },
    { added: 0, removed: 0 },
  )
  const failed = item.state === 'error' || (item.exit_code !== null && item.exit_code !== 0)
  const firstLine = summary.subject.split('\n')[0] ?? ''
  return (
    <div className={`tool ${open ? 'open' : ''} ${item.state}`}>
      <button className="row-toggle tool-row" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="tool-icon">{summary.icon}</span>
        <span className={item.state === 'running' ? 'shimmer verb' : 'verb'}>{summary.verb}</span>
        <span className={summary.mono ? 'subject mono' : 'subject'} title={summary.subject}>
          {firstLine}
          {summary.subject.includes('\n') ? ' …' : ''}
        </span>
        {item.diffs.length > 0 && (
          <span className="stats">
            <span className="added">+{stats.added}</span> <span className="removed">−{stats.removed}</span>
          </span>
        )}
        {failed && <span className="badge error">{item.exit_code !== null && item.exit_code !== 0 ? `exit ${item.exit_code}` : 'failed'}</span>}
        {item.state === 'interrupted' && <span className="badge">interrupted</span>}
        <ChevronIcon size={12} className="chevron" />
      </button>
      {item.diffs.length > 0 && !open && (
        <div className="tool-diffs">
          {item.diffs.map((diff, index) => (
            <DiffView key={index} diff={diff} />
          ))}
        </div>
      )}
      {open && <ToolDetails item={item} args={args} />}
    </div>
  )
})

function ToolDetails({ item, args }: { item: Item; args: Args }) {
  if (item.name === 'run_code') {
    return (
      <div className="tool-body">
        <pre className="code-small">
          <code className="hljs" dangerouslySetInnerHTML={{ __html: highlight(str(args.code).trim(), 'python') }} />
        </pre>
        {item.text && <pre className={`result ${item.state === 'error' ? 'error' : ''}`}>{codeResult(item.text)}</pre>}
      </div>
    )
  }
  if (item.name === 'shell') {
    const output = item.output || item.text
    return (
      <div className="tool-body">
        <pre className="terminal">
          <span className="prompt">$ </span>
          {str(args.command)}
          {output && '\n'}
          {output}
        </pre>
      </div>
    )
  }
  return (
    <div className="tool-body">
      {item.diffs.map((diff, index) => (
        <DiffView key={index} diff={diff} />
      ))}
      {Object.keys(args).length > 0 && item.diffs.length === 0 && (
        <pre className="code-small">
          <code
            className="hljs"
            dangerouslySetInnerHTML={{ __html: highlight(JSON.stringify(args, null, 2), 'json') }}
          />
        </pre>
      )}
      {item.text && <pre className={`result ${item.state === 'error' ? 'error' : ''}`}>{item.text}</pre>}
    </div>
  )
}
