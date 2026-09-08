# Figure captions

**Figure 1.** Hybrid retrieval and QLoRA architecture for bilingual railway question answering. Stages 1-5 connect governed inputs, frozen BGE-M3 encoding, BM25/dense rank fusion, evidence-conditioned generation and independent evaluation. The lower branch shows offline answer-only adapter training with frozen base weights; retrieval and evaluation are outside the gradient path. Trade-off annotations summarise task-dependent empirical results, not guaranteed gains. Source: Authors' own work.

**Figure 2.** Four-stage expert-governed lifecycle covering source acquisition and bilingual editing, review-state decisions, approved production indexing, and evaluation splits with exact held-out records excluded from indexing and training. Source: Authors' own work.

**Figure 3.** Hybrid evidence-equivalent retrieval quality and latency across top-k settings. Left: Evidence Recall@k; right: mean retrieval latency. Source: Authors' own work.

**Figure 4.** Completion-only QLoRA optimisation for Qwen2.5-7B and GLM-4-9B. Lines show logged training loss; diamonds mark the single end-of-epoch validation measurement for each model. Source: Authors' own work.

**Figure 5.** Direction- and task-separated COMET before and after QLoRA. Left: Qwen2.5-7B; right: GLM-4-9B. Source: Authors' own work.

**Figure 6.** Mean bilingual standalone character-level F1 against generation latency and peak reserved GPU memory (GiB) for the four Qwen2.5/GLM original and QLoRA conditions. Left: mean generation latency; right: PyTorch reserved GPU memory. Quality and resources are measured on separate workloads. Source: Authors' own work.

**Figure 7.** Standalone bilingual-QA and domain/translation output flags. Bars show mean condition/task-level proportions among groups in which each flag occurred, not pooled sample-level rates. Flags are non-exclusive; RAG retrieval misses and citation omissions are reported separately in the text. Source: Authors' own work.

**Figure 8.** Bilingual index, automated evidence support and governance-history validation. Panels A-C report field ablation, evidence support and governance audit results, respectively. Source: Authors' own work.
