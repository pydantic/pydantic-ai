import { createContext } from 'react'

/** Opens a workspace's own page; given in the orchestrator chat, whose tool cards then link the workspaces they name. */
export const WorkspaceLinkContext = createContext<((id: string) => void) | null>(null)
