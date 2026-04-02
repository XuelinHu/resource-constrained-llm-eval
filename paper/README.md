# Paper Workspace

This directory is isolated from the experiment code and is intended for your SCI manuscript.

## Files

- `main.tex`: manuscript entry point
- `sections/`: section-level LaTeX sources
- `bib/references.bib`: bibliography database
- `tables/`: generated or manually maintained tables
- `figures/`: charts and exported plots

## Suggested Compile Command

```powershell
cd paper
latexmk -pdf main.tex
```

## Writing Rule

Do not write this paper like a benchmark note.

The manuscript should argue:

1. Why single-GPU evaluation is a real deployment problem.
2. Which models offer the best performance-efficiency tradeoff.
3. Whether QLoRA meaningfully improves domain performance under 24 GB VRAM.
4. How the cost of reasoning-oriented models compares with their gains.
