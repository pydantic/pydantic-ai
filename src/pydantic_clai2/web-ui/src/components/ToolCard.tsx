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

/** One tool call as a quiet row; open it for the arguments, output, diff, and result. */
export const ToolCard = memo(function ToolCard({ item: raw }: { item: Item }) {
  const cards = useContext(OpenCardsContext)
  const open = cards.open.has(raw.id)
  const args = parseArgs(raw.args)
  const item = { ...raw, diffs: argumentDiffs(raw, args) }
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
  const firstLine = summary.subject.split('\n')[0] ?? ''
  const workspace = orchestrating && item.state !== 'running' && !refused ? workspaceOf(item, args) : null
  const row = (
    <button className="row-toggle tool-row" onClick={() => cards.toggle(raw.id)} aria-expanded={open}>
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
          <code className="hljs" dangerouslySetInnerHTML={{ __html: highlight(cell(item.args).code.trim(), 'python') }} />
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
