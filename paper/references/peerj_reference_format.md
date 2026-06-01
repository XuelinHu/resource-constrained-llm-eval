# PeerJ Reference Formatting Plan

This note normalizes the manuscript bibliography workflow for the current
PeerJ Computer Science draft.

## Format Policy

- Maintain all references in `paper/bib/references.bib`.
- Use BibTeX entries rather than hand-written reference paragraphs.
- Keep DOI fields whenever a DOI is available.
- Keep stable URLs for arXiv preprints, model cards, theses, and online-first
  articles.
- Use `@article` for journal articles and arXiv preprints.
- Use `@inproceedings` or `@incollection` for conference and LNCS-style
  proceedings chapters.
- Use `@phdthesis` for dissertations.
- Use `@misc` only for model cards, blog/model documentation, or items without
  a clear archival venue.

## Required Fields

Each finalized entry should include, when available:

- `author`
- `title`
- `year`
- `journal` or `booktitle`
- `volume`, `number`, and `pages`
- `doi`
- `url`
- `note` for preprints, model cards, online-first status, or access limitations

## Current Draft Convention

The current `main.tex` uses `\nocite{*}` so that every curated reference appears
in the draft bibliography. This is useful while the paper is still being
assembled from a literature matrix. Before final submission, unused references
should be cited in the appropriate section or removed.

## Section Mapping

- Open LLM families and model cards: Introduction, Experimental Setup.
- LLM evaluation surveys and benchmarks: Related Work, Methodology.
- Resource-constrained LLM deployment: Related Work, Discussion.
- PEFT, LoRA, QLoRA, pruning, and federated adaptation: Related Work,
  Methodology.
- Domain and low-resource application studies: Problem Setting, Discussion.

## PeerJ Notes

PeerJ Computer Science accepts conventional BibTeX-managed references. The
final manuscript should keep the bibliography consistent with the submitted
`.tex` and `.bib` files, and all cited code/data/model resources should have
stable public links where possible.
