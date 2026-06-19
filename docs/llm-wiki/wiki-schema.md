# LLM Wiki Schema

Last updated: 2026-05-21

This file defines the local convention for `docs/llm-wiki/`.

## Layers

- Raw sources: existing files under `docs/`, repo code, commits, trace JSON, and
  operational logs. Do not move or rewrite raw sources as part of wiki
  maintenance.
- Source summaries: `source-docs/*.md`, one page per raw documentation source.
  These pages preserve provenance and make long documents skimmable.
- Code summaries: `source-code/*.md`, module-level maps for source code,
  tests, scripts, settings, and repo-operational files. These are not one page
  per Python file; the complete file list belongs in `source-code/file-inventory.md`.
- Synthesis: `synthesis/*.md`, cross-source explanations of the current system.
- Navigation: `index.md`, the content catalog and recommended reading order.
- History: `log.md`, append-only maintenance entries.

## Source Summary Template

Each `source-docs` page should include:

- `Source`: relative link to the raw document.
- `Status`: `current`, `current with caveats`, or `historical`.
- `Use this when`: one-line routing guidance.
- `Key facts`: concise bullets.
- `Related code`: important code areas, not exhaustive file lists.
- `Caveats`: stale claims, conflicts, encoding issues, or validation limits.
- `Related wiki pages`: links to synthesis pages.

## Synthesis Template

Each synthesis page should include:

- `Current thesis`: what a new LLM should believe first.
- `Flow` or `Concept map`: the system-level explanation.
- `Operational rules`: things that should constrain future work.
- `Evidence`: links to source summaries and code/docs.
- `Open questions`: only where uncertainty remains.

## Code Map Template

Each `source-code` page should include:

- `Source`: relative links to the raw code/config/test/script files.
- `Status`: currentness and scope.
- `Current thesis`: what another LLM should believe first.
- `Main responsibilities`: module or subsystem responsibilities.
- `Operational rules`: constraints that matter for future edits.
- `Related wiki pages`: links to synthesis and adjacent code maps.

## Link Style

Use ordinary relative Markdown links. The CRK-model repo is not an Obsidian
vault, and GitHub-style Markdown rendering is the lowest-friction default.

## Freshness Rules

- Prefer `docs/agent-guides/*` and recently committed code/config for current
  operational defaults.
- Treat `IMPLEMENTATION_ROADMAP.md`, `MODEL_PROCESS_RISK_REVIEW_2026-03-04.md`,
  and `FIXES_APPLIED_2026-03-05.md` as historical unless another current source
  confirms the claim.
- When old docs conflict with current runtime defaults, keep both facts visible
  and mark the old one as historical.
- Do not claim local PC backend startup proves Jetson runtime readiness.
- Do not delete or rewrite untracked trace JSON files while maintaining this
  wiki unless the user explicitly asks for that cleanup.
- Untracked trace JSON files are raw operational evidence. Do not ingest them
  into durable wiki pages by default; summarize them only when the user asks.
- Current README-level repo facts override older docs when they conflict. In
  particular, this Python repo is the legacy/reference TensorRT path, while
  README points fresh clone-based operation toward `CRK-model-go`; and current
  README/tests describe zone loadcell channels as summed, not averaged.

## Update Workflow

1. Read `index.md`.
2. Add or update the relevant `source-docs` or `source-code` page.
3. Update any affected `synthesis` page.
4. Add a dated `log.md` entry.
5. Run link/search checks before committing.
