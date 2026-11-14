import { createContext } from 'react'

/** Opens a workspace's session in this page; given in the orchestrator chat, whose tool cards then link the workspaces they name. */
export const WorkspaceLinkContext = createContext<((id: string) => void) | null>(null)
