# Figure captions

**Figure 1.** Detailed neural retrieval and QLoRA workflow for bilingual railway question answering. The upper path separates governed bilingual inputs, shared BGE-M3 encoding, parallel dense/BM25 retrieval, reciprocal-rank fusion, evidence-conditioned generation and independent evaluation. The lower path isolates completion-only adapter training with a frozen quantised base model; retrieval and evaluation remain outside the gradient path. The schematic clarifies information flow and trainable boundaries rather than claiming identical internals for the two generators. Source: Authors' own work.

**Figure 2.** Hybrid evidence-equivalent retrieval quality and latency across top-k settings. Left: Evidence Recall@k; right: mean retrieval latency. Source: Authors' own work.

**Figure 3.** Completion-only QLoRA optimisation for Qwen2.5-7B and GLM-4-9B. Lines show logged training loss; diamonds mark the single end-of-epoch validation measurement for each model. Source: Authors' own work.

**Figure 4.** Direction- and task-separated COMET before and after QLoRA. Left: Qwen2.5-7B; right: GLM-4-9B. Source: Authors' own work.

**Figure 5.** Mean bilingual standalone character-level F1 against generation latency and peak reserved GPU memory (GiB) for the four Qwen2.5/GLM original and QLoRA conditions. Left: mean generation latency; right: PyTorch reserved GPU memory. Quality and resources are measured on separate workloads. Source: Authors' own work.

**Figure 6.** Standalone bilingual-QA and domain/translation output flags. Bars show mean condition/task-level proportions among groups in which each flag occurred, not pooled sample-level rates. Flags are non-exclusive; RAG retrieval misses and citation omissions are reported separately in the text. Source: Authors' own work.

**Figure 7.** Three complementary information-system checks. Panel A tests bilingual index fields, Panel B measures automated evidence support, and Panel C audits governance history; the panels are arranged by validation layer rather than merged into one numerical score. Source: Authors' own work.
