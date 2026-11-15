import { createContext } from 'react'

/** Whether the user opened or closed each tool card they clicked, kept above the transcript so regrouping or
 * remounting a card keeps it as they left it. */
export interface OpenCards {
  open: ReadonlyMap<string, boolean>
  toggle: (id: string, open: boolean) => void
}

export const OpenCardsContext = createContext<OpenCards>({ open: new Map(), toggle: () => {} })
