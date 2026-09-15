# Figure captions

**Figure 1.** Bilingual railway QA workflow from left to right: governed bilingual input, shared encoding, lexical and dense retrieval, fusion, evidence-conditioned generation and evaluation. The rightmost module shows the QLoRA adapter path alongside the frozen base model.

**Figure 2.** Hybrid evidence-equivalent retrieval quality and latency across top-k settings. Left: Evidence Recall@k; right: mean retrieval latency. Source: Authors' own work.

**Figure 3.** Completion-only QLoRA optimisation for Qwen2.5-7B and GLM-4-9B. Lines show logged training loss; diamonds mark the end-of-epoch validation measurement for each model. Source: Authors' own work.

**Figure 4.** Mean bilingual standalone character-level F1 against generation latency and peak reserved GPU memory (GiB) for the four Qwen2.5/GLM original and QLoRA conditions. Left: mean generation latency; right: PyTorch reserved GPU memory. Quality and resources are measured on separate workloads. Source: Authors' own work.

**Figure 5.** Three validation layers. Panel A compares source-only, Chinese-field, English-field and bilingual indexes; Panel B compares semantic support against retrieved and explicitly cited evidence; Panel C audits immutable review events and before-state snapshots. Together the panels show retrieval balance, evidence support and governance traceability without reducing them to one score. An em dash indicates that a measure is not language-specific.
