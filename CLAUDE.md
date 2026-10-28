# LLM Primer

A VitePress site: how LLMs work end to end, for an infra engineer with no ML background preparing for AI lab
interviews. Sibling of the OS Primer. Deployed to GitHub Pages by `.github/workflows/deploy.yml` on every push
to `main`.

- **Before writing or editing any chapter, read `STYLE.md` (including "LLM Primer additions") and follow it.**
- Table of contents and each chapter's scope: `docs/.vitepress/chapters.ts`. Set `ready: true` when complete.
- Glossary: one file per chapter, `docs/.vitepress/glossary/<chapter-id>.ts`, exporting
  `export const terms: Record<string, GlossaryEntry> = {...}` (import the type from `../glossary`). They are
  merged automatically. Use `<Term id="...">` for the first use of a glossary term in a chapter. Before
  adding a term, check the other glossary files; reuse existing ids.
- Components: any `.vue` file in `docs/.vitepress/theme/components/` (or `diagrams/`) is registered
  globally by file name. No registration step.
- Flashcards live next to the chapter (`<slug>-review.ts`, exporting `cards`). See the OS Primer's
  `docs/memory/virtual-memory.md` and `vm-review.ts` for the reference chapter format.
- Exercises: `exercises/chNN_<name>.py`. Python venv with numpy and CPU torch: `~/.venvs/llm/bin/python`.
- Build: `PATH=$HOME/.local/node/bin:$PATH npm run build` (fails on dead links).

## Writing a chapter

1. Read `STYLE.md`, the chapter's entry in `chapters.ts`, and the finished chapters before it (skim) so you
   do not re-explain what they cover in full; recap in a sentence and link instead.
2. Write the chapter at the path given by its `link`, replacing the stub. Flashcards: 12–20 cards.
3. Diagrams: Vue SVG components in `theme/components/diagrams/`, `viewBox` 640 wide, the shared classes in
   `custom.css` (`box`, `box-a`…`box-d`, `t`, `tb`, `h`, `m`, `ln`, …), plain labels, a `<figcaption>`.
   1–3 diagrams where they help understanding; none for decoration.
4. Run every exercise and paste the real output.
5. Re-read the whole chapter twice: once for technical correctness, once as a reader with no ML background.
   Fix anything used before it is explained.
6. Length follows scope. No word target.

### When several chapters are written in parallel

- Only create or edit your own chapter's files: the `.md`, `-review.ts`, `glossary/<id>.ts`, your exercises,
  and new diagram components (prefix diagram names to avoid collisions if unsure).
- Do not edit shared files: `chapters.ts`, `config.mts`, `theme/index.ts`, `custom.css`, `STYLE.md`.
- Do not commit, push, or run `npm run build`; the integrator does.
