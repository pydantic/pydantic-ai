import { createContext } from 'react'

/** The tool cards the user opened, kept above the transcript so regrouping or remounting a card keeps it open. */
export interface OpenCards {
  open: ReadonlySet<string>
  toggle: (id: string) => void
}

export const OpenCardsContext = createContext<OpenCards>({ open: new Set(), toggle: () => {} })
