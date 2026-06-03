# Writing TODO

## Completed in Current Draft

1. Reorganized the manuscript around the latest single-RTX-3090 work plan.
2. Added a dedicated dataset section for Domain-QA and Domain-RegQA.
3. Updated the model pool to the ten-model plan.
4. Added exported-result tables for public benchmarks, domain tasks, efficiency, QLoRA training, and adaptation evaluation.
5. Marked unfinished experiments with explicit placeholders instead of leaving implicit gaps.
6. Recompiled `paper/out/main.pdf` successfully.

## Experiments Still Needed

1. Run the final ten-model int4 public benchmark sweep.
2. Run the final ten-model Domain-QA and Domain-RegQA baseline evaluation.
3. Complete matched baseline-versus-adapter evaluation for Qwen2.5-7B-Instruct.
4. Evaluate Qwen3-8B Adapter-A on Domain-QA and Domain-RegQA.
5. Train and evaluate Adapter-B on Domain-RegQA.
6. Train and evaluate Adapter-C on mixed Domain-QA + Domain-RegQA.
7. Train and evaluate Adapter-D with completion-only loss masking.
8. Add qualitative error examples for terminology, translation, and regulatory QA.

## Figures to Produce

1. Public benchmark score comparison across the ten-model pool.
2. Domain-QA and Domain-RegQA score comparison.
3. Peak VRAM comparison under int4 inference.
4. Throughput-versus-quality trade-off plot.
5. QLoRA before/after and data-composition ablation plot.

## Tables to Regenerate After Final Runs

1. Table 3: public benchmark results.
2. Table 4: domain-task results.
3. Table 5: efficiency results.
4. Table 7: before/after QLoRA comparison.
5. Table 8: data-composition ablation.
