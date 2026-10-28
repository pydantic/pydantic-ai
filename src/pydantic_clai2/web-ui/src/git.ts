import { call } from './api'

/** A file with uncommitted changes; `status` is one letter, as `git status --short` shows it. */
export interface ChangedFile {
  path: string
  status: string
  original: string | null
}

/** The live session's repository, as `GitStatus` in `runtime/git_actions.py` has it. */
export interface GitStatus {
  branch: string | null
  upstream: string | null
  ahead: number
  behind: number
  files: ChangedFile[]
  default_branch: boolean
  /** Why it cannot be pushed; `null` when it can. */
  push: string | null
  /** Why no pull request can be opened from it; `null` when one can. */
  pull_request: string | null
}

export interface GitDone {
  message: string
  status: GitStatus | null
}

export interface PullRequestDraft {
  title: string
  body: string
}

/** The git routes of one live session, by conversation ID; each acts in that session's directory. */
export function gitApi(id: string) {
  const base = `/api/sessions/${encodeURIComponent(id)}/git`
  return {
    /** `null` outside a Git repository. */
    status: () => call<GitStatus | null>('GET', base),
    /** A commit message the session's model writes for `files`, every changed one when `null`. */
    draftCommit: (files: string[] | null) => call<{ message: string }>('POST', `${base}/commit/draft`, { files }),
    commit: (message: string, files: string[] | null) => call<GitDone>('POST', `${base}/commit`, { message, files }),
    /** `confirmed` allows pushing the default branch. */
    push: (confirmed: boolean) => call<GitDone>('POST', `${base}/push`, { confirmed }),
    draftPullRequest: () => call<PullRequestDraft>('POST', `${base}/pr/draft`),
    /** Push the branch, then open the pull request; it answers its URL. */
    openPullRequest: (title: string, body: string) => call<{ url: string }>('POST', `${base}/pr`, { title, body }),
  }
}

export type GitApi = ReturnType<typeof gitApi>

/** The chip's counts: `3 changed, 1 ahead`, and how far behind when it is. */
export function gitCounts(status: GitStatus): string {
  const counts = `${status.files.length} changed, ${status.ahead} ahead`
  return status.behind ? `${counts}, ${status.behind} behind` : counts
}
