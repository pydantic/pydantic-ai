import { memo, useContext } from 'react'
import { cell } from '../code'
import { OpenCardsContext } from '../openCards'
import { WorkspaceLinkContext } from '../workspaceLink'
import { highlight } from '../markdown'
import type { Diff, Item } from '../protocol'
import { DiffView, diffStats } from './DiffView'
import {
  BranchIcon,
  ChevronIcon,
  FileIcon,
  FolderIcon,
  OrchestratorIcon,
  PencilIcon,
  QuestionIcon,
  SearchIcon,
  TerminalIcon,
  ToolIcon,
} from './icons'
import { WorkerChips } from './WorkersTree'

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

// The tools code mode puts in the sandbox (`write_file` and `edit_file` stay native). The two alternatives
// inside the string never match the same text, so a run of backslashes cannot backtrack exponentially.
const CALL = /\b(read_file|list_files|grep|shell)\(\s*(?:\w+\s*=\s*)?(?:([rRbBfFuU]{0,2})(["'])((?:\\.|(?!\3|\\).)*)\3)?/g

/** The tool calls inside a `run_code` script, with their first literal argument, as code mode makes them.
 * Comment lines are skipped; a raw string keeps its backslashes, as a regex pattern needs them. */
export function codeCalls(code: string): Call[] {
  const lines = code.split('\n').filter((line) => !line.trim().startsWith('#'))
  return [...lines.join('\n').matchAll(CALL)].map((match) => {
    const raw = /r/i.test(match[2] ?? '')
    const body = match[4] ?? ''
    return { name: match[1] ?? '', argument: raw ? body : body.replace(/\\(["'\\])/g, '$1') }
  })
}

/** The orchestrator's tools, as `[running, done]` verbs; each names the workspace it works on. */
const WORKSPACE_TOOLS: Record<string, [string, string]> = {
  workspace_create: ['Creating workspace', 'Created workspace'],
  workspace_destroy: ['Destroying workspace', 'Destroyed workspace'],
  session_start: ['Starting', 'Started'],
  session_send: ['Sending to', 'Sent to'],
  session_read: ['Reading', 'Read'],
  session_wait: ['Waiting for', 'Waited for'],
  session_interrupt: ['Interrupting', 'Interrupted'],
  session_answer: ['Answering', 'Answered'],
}

/** The workspace an orchestrator tool call works on: its `workspace` argument, or the one `workspace_create` made. */
export function workspaceOf(item: Item, args: Args): string | null {
  if (!(item.name in WORKSPACE_TOOLS)) return null
  if (typeof args.workspace === 'string' && args.workspace) return args.workspace
  return /^Created workspace (\S+) /.exec(item.text)?.[1] ?? null
}

/** Whether `name` is one of the orchestrator's tools, each shown on its own rather than folded into a group. */
export const isWorkspaceTool = (name: string) => name in WORKSPACE_TOOLS || name === 'workspace_list'

/** An orchestrator tool's problem comes back as its result, an `Error: ...` line, rather than a failed call. */
const reportedError = (item: Item) => item.text.startsWith('Error: ')

interface Summary {
  icon: React.ReactNode
  verb: string
  subject: string
  mono: boolean
}

/** An orchestrator tool call: the verb, and the workspace it works on. */
function workspaceSummary(item: Item, args: Args): Summary {
  const running = item.state === 'running'
  if (item.name === 'workspace_list') {
    return { icon: <FolderIcon size={14} />, verb: running ? 'Listing' : 'Listed', subject: 'workspaces', mono: false }
  }
  const [doing, done] = WORKSPACE_TOOLS[item.name] ?? ['', '']
  const subject = workspaceOf(item, args) ?? str(args.kind)
  if (item.name !== 'workspace_create' && item.name !== 'workspace_destroy') {
    return { icon: <OrchestratorIcon size={14} />, verb: running ? doing : done, subject, mono: true }
  }
  const kept = item.name === 'workspace_destroy' && item.text.startsWith('Not destroyed')
  return { icon: <BranchIcon size={14} />, verb: running ? doing : kept ? 'Kept workspace' : done, subject, mono: true }
}

function summarize(item: Item, args: Args, inOrchestrator: boolean): Summary {
  const running = item.state === 'running'
  if (inOrchestrator && isWorkspaceTool(item.name)) return workspaceSummary(item, args)
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
      const calls = codeCalls(cell(item.args).code)
      const [only] = calls
      if (calls.length === 1 && only?.argument) {
        // A script around one call reads as that call.
        const key = only.name === 'shell' ? 'command' : only.name === 'grep' ? 'pattern' : 'path'
        return summarize({ ...item, name: only.name }, { [key]: only.argument }, inOrchestrator)
      }
      const names = item.calls.length ? item.calls.map((call) => call.name) : calls.map((call) => call.name)
      if (names.length > 1) {
        // A script around several calls reads as the calls it made.
        return { icon: <TerminalIcon size={14} />, verb: running ? 'Running' : 'Ran', subject: names.join(' · '), mono: true }
      }
      const first = cell(item.args).code.split('\n').find((line) => line.trim() && !line.trim().startsWith('#'))
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

/** The diffs an edit or write made, or tried to make, rebuilt from its arguments; live runs send the real ones. */
function argumentDiffs(item: Item, args: Args): Diff[] {
  if (item.diffs.length > 0) return item.diffs
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
export function codeResult(text: string): string {
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

const TRACEBACK = 'Traceback (most recent call last):'
/** A line of only `~` and `^` under a traceback's echoed source, marking the failing expression. */
const UNDERLINE = /^\s*[~^]+\s*$/
/** What a `run_code` cell printed before it failed, which code mode puts before the error. */
const PRINTED = /\[stdout before error\]\n[\s\S]*?\n\[\/stdout before error\]\n?/

interface Failure {
  /** The line that says why: a traceback's exception, else the message's first line. */
  summary: string
  /** The exception and the lines that continue it; the whole message when it has no traceback. */
  exception: string
  /** The traceback's frames without their source underlines; `null` when there is no traceback. */
  frames: string | null
  /** The rest of a message with a traceback: what the cell printed, and anything after the exception. */
  rest: string
}

/** Why a tool failed, from its error text: code mode's `run_code` ends a traceback with the exception,
 * and other messages lead with it, after a heading such as `Type error in code:`. */
function failure(text: string): Failure {
  // What the cell printed may hold a traceback of its own, so look for the error's after it.
  const printed = PRINTED.exec(text)?.[0] ?? ''
  const lines = text.replace(printed, '').split('\n')
  const start = lines.map((line) => line.trim()).lastIndexOf(TRACEBACK)
  // The exception is the first line after the traceback's indented frames.
  const at = start < 0 ? -1 : lines.findIndex((line, index) => index > start && line.trim() !== '' && !/^\s/.test(line))
  if (at < 0) {
    const [first = '', next] = lines.map((line) => line.trim()).filter(Boolean)
    return { summary: first.endsWith(':') && next ? next : first, exception: text, frames: null, rest: '' }
  }
  const blank = lines.findIndex((line, index) => index > at && line.trim() === '')
  const end = blank < 0 ? lines.length : blank
  const before = lines.slice(0, start).join('\n').trim()
  // A heading before the traceback, such as `Runtime error:`, says no more than the exception shown first.
  const heading = !before.includes('\n') && before.endsWith(':')
  const rest = [heading ? '' : before, printed.trim(), lines.slice(end).join('\n').trim()].filter(Boolean).join('\n\n')
  return {
    summary: lines[at] ?? '',
    exception: lines.slice(at, end).join('\n'),
    frames: lines.slice(start, at).filter((line) => !UNDERLINE.test(line)).join('\n'),
    rest,
  }
}

/** A failed call's result: the exception first, its traceback folded, then the rest of the message. */
function FailureResult({ text }: { text: string }) {
  const { exception, frames, rest } = failure(text)
  return (
    <>
      <pre className="result error">{exception}</pre>
      {frames !== null && (
        <details className="traceback">
          <summary>Traceback</summary>
          <pre className="result">{frames}</pre>
        </details>
      )}
      {rest && <pre className="result">{rest}</pre>}
    </>
  )
}

/** The lines each file-change diff shows until the user asks for the rest, so long diffs don't mount whole. */
const PREVIEW_LINES = 8

/** One tool call as a quiet row; open it for the arguments, output, diff, and result. A file change starts open. */
export const ToolCard = memo(function ToolCard({ item: raw }: { item: Item }) {
  const cards = useContext(OpenCardsContext)
  const args = parseArgs(raw.args)
  const attempted = argumentDiffs(raw, args)
  // A failed call changed nothing: its row shows no stats and starts closed; open, it shows what it tried.
  const item = { ...raw, diffs: raw.state === 'error' ? raw.diffs : attempted }
  const changed = item.diffs.length > 0
  const open = cards.open.get(raw.id) ?? changed
  const openWorkspace = useContext(WorkspaceLinkContext)
  // The orchestrator's tools read as such only in its chat; a session's tool of the same name is another tool.
  const orchestrating = openWorkspace !== null && isWorkspaceTool(item.name)
  const summary = summarize(item, args, openWorkspace !== null)
  const stats = item.diffs.reduce(
    (total, diff) => {
      const { added, removed } = diffStats(diff.diff)
      return { added: total.added + added, removed: total.removed + removed }
    },
    { added: 0, removed: 0 },
  )
  const refused = orchestrating && reportedError(item)
  const failed = item.state === 'error' || (item.exit_code !== null && item.exit_code !== 0) || refused
  const reason = item.state === 'error' || refused ? failure(item.text).summary : ''
  const firstLine = summary.subject.split('\n')[0] ?? ''
  const workspace = orchestrating && item.state !== 'running' && !refused ? workspaceOf(item, args) : null
  const row = (
    <button className="row-toggle tool-row" onClick={() => cards.toggle(raw.id, open)} aria-expanded={open}>
      <span className="tool-icon">{summary.icon}</span>
      <span className={item.state === 'running' ? 'shimmer verb' : 'verb'}>{summary.verb}</span>
      <span className={summary.mono ? 'subject mono' : 'subject'} title={summary.subject}>
        {firstLine}
        {summary.subject.includes('\n') ? ' …' : ''}
      </span>
      {changed && (
        <span className="stats">
          <span className="added">+{stats.added}</span> <span className="removed">−{stats.removed}</span>
        </span>
      )}
      {failed && <span className="badge error">{item.exit_code !== null && item.exit_code !== 0 ? `exit ${item.exit_code}` : 'failed'}</span>}
      {reason && (
        <span className="error-summary" title={reason}>
          {reason}
        </span>
      )}
      {item.state === 'interrupted' && <span className="badge">interrupted</span>}
      <ChevronIcon size={12} className="chevron" />
    </button>
  )
  return (
    <div className={`tool ${open ? 'open' : ''} ${item.state}`}>
      {workspace && openWorkspace && item.name !== 'workspace_destroy' ? (
        <div className="tool-head">
          {row}
          <button className="tool-link" onClick={() => openWorkspace(workspace)} title={`Open ${workspace}'s session`}>
            Open
          </button>
        </div>
      ) : (
        row
      )}
      <WorkerChips item={raw} />
      {open && <ToolDetails item={{ ...item, diffs: attempted }} args={args} />}
    </div>
  )
})

function ToolDetails({ item, args }: { item: Item; args: Args }) {
  if (item.name === 'run_code') {
    const failed = item.state === 'error' && item.text !== ''
    return (
      <div className="tool-body">
        {failed && <FailureResult text={item.text} />}
        <pre className="code-small">
          <code className="hljs" dangerouslySetInnerHTML={{ __html: highlight(cell(item.args).code.trim(), 'python') }} />
        </pre>
        {item.text && !failed && <pre className="result">{codeResult(item.text)}</pre>}
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
  // A failed edit or write leads with why it failed, then what it tried to change.
  const errorFirst = item.state === 'error' && item.diffs.length > 0
  const result =
    item.text && (item.state === 'error' ? <FailureResult text={item.text} /> : <pre className="result">{item.text}</pre>)
  return (
    <div className="tool-body">
      {errorFirst && result}
      {item.diffs.map((diff, index) => (
        <DiffView key={index} diff={diff} preview={PREVIEW_LINES} />
      ))}
      {Object.keys(args).length > 0 && item.diffs.length === 0 && (
        <pre className="code-small">
          <code
            className="hljs"
            dangerouslySetInnerHTML={{ __html: highlight(JSON.stringify(args, null, 2), 'json') }}
          />
        </pre>
      )}
      {!errorFirst && result}
    </div>
  )
}
