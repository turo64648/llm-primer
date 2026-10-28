// Plain-language definitions for terms used across chapters.
// Used by <Term id="..."> popovers and the Glossary page.
// Entries live in glossary/<chapter-slug>.ts (each exports `terms`) and are merged here,
// so chapters written in parallel never edit the same file.

export interface GlossaryEntry {
  term: string
  def: string
  chapter?: string // link to the chapter that explains it in full
}

const files = import.meta.glob('./glossary/*.ts', { eager: true }) as Record<string, { terms: Record<string, GlossaryEntry> }>

export const glossary: Record<string, GlossaryEntry> = Object.assign({}, ...Object.values(files).map((m) => m.terms))
