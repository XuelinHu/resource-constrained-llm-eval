# Knowledge-Enhanced Large Language Models for Bilingual Railway Vocational Education under Resource Constraints

## Structured abstract

**Background.** The international adoption of Chinese high-speed railway technology has increased demand for bilingual knowledge transfer and vocational training. Existing railway large language model studies often examine domain adaptation or retrieval-augmented generation separately, leaving limited evidence on how bilingual model adaptation, governed retrieval, evidence provenance and resource-constrained deployment interact. This study therefore investigates knowledge-enhanced large language models and implements a bounded artificial intelligence agent system as a controlled experimental and application environment for international learners and vocational trainees, railway technical and service professionals, and personnel engaged in training, consulting and management.

**Methods.** An expert-governed bilingual railway corpus was organised with pair-grouped controls to separate training, validation and held-out evaluation records. Qwen2.5-7B and GLM-4-9B were evaluated before and after bilingual QLoRA adaptation, with Qwen3-14B retained as a reference. BM25, BGE-M3 dense retrieval and reciprocal-rank-fusion hybrid retrieval were compared on held-out bilingual queries. Model adaptation was assessed using a separate bilingual QA evaluation, complemented by a controlled no-retrieval and retrieval-augmented generation comparison. The evaluation covered Evidence Recall@k, mean reciprocal rank, Answer F1, paired significance, evidence support and resource use under single-machine deployment.

**Results.** Hybrid retrieval achieved Evidence Recall@5 of 0.715 in Chinese and 0.708 in English, exceeding the strongest single-retriever results of 0.690 and 0.685, respectively; the bilingual-field index attained the highest balanced mean of 0.696. Qwen QLoRA increased held-out character-level F1 from 0.189 to 0.398 in Chinese and from 0.437 to 0.648 in English, with both gains remaining significant after Holm correction. Under hybrid retrieval-augmented generation, it achieved the highest Answer F1 of 0.660 and 0.651, exceeding the Qwen3-14B reference values of 0.442 and 0.454. All evaluated generator conditions completed within the defined single-machine resource boundary.

**Conclusion.** Combining bilingual parameter-efficient adaptation with governed hybrid retrieval improves cross-language evidence access and domain question answering while remaining feasible under resource-constrained single-machine deployment. The model-centred framework provides a transferable approach for jointly evaluating adaptation, retrieval and evidence governance, and offers a locally controlled foundation for bilingual railway education and international workforce development.

**Keywords:** bilingual railway education; retrieval-augmented generation; QLoRA adaptation; Web information systems

## 1. Introduction

China's high-speed railway development has attracted sustained international attention, creating a growing need for bilingual technical exchange and vocational training (Lawrence *et al.*, 2019; Xu, 2018). Yet such exchange is hindered by differences in language, terminology, source access and review practice. Railway vocational education combines safety-critical regulations, specialised terminology and procedural textbook knowledge, so international learners, vocational trainees and railway personnel often move between Chinese source material and English learning support. Generic large language models can produce fluent answers, but their parametric knowledge does not clearly distinguish verified railway knowledge from unsupported generation. A useful Web information system must therefore acquire, govern, retrieve, present and trace bilingual knowledge within a controlled local workflow.

The intended users extend beyond enrolled students to locally employed staff, technical specialists, service personnel, trainers, consultants and managers. The system must connect a bilingual question to an admissible regulation or teaching passage, preserve provenance and expose review state rather than merely translate isolated terms. Railway studies have explored specialised language models for safety, rail design, track management, prognostics, driver advice and interactive railway training (Zheng *et al.*, 2023; Lahnalampi, 2024; Wang and Li, 2024; Song *et al.*, 2025; Luo *et al.*, 2025; Li *et al.*, 2025), while broader reviews identify reliability and deployment as continuing challenges (Wandelt *et al.*, 2024). Bilingual access, governed evidence and reproducible local deployment nevertheless remain under-examined as a combined information-system problem.

Existing studies commonly evaluate either general-purpose language models or retrieval-augmented generation in isolation. Four gaps motivate this work. First, bilingual educational retrieval is rarely evaluated separately by query language. Second, expert review is often described as preprocessing rather than implemented as a queryable governance state. Third, domain adaptation is frequently evaluated only on the target task, leaving translation regressions and citation-following failures hidden. Fourth, the term agent is often used without defining its action boundary, while deployment reports may omit the interaction between retrieval quality, answer traceability and measured single-workstation cost.

This study addresses the following research questions:

- **RQ1:** How accurately do BM25, pgvector dense retrieval and hybrid fusion retrieve evidence-equivalent indexed records for held-out Chinese and English railway question-answer pairs?
- **RQ2:** How do retrieval strategies affect answer overlap and citation-format compliance, and can approved-only record eligibility be enforced and audited at retrieval time?
- **RQ3:** What bilingual translation quality is achieved in each direction, and how does completion-only QLoRA affect domain QA under pair-grouped held-out evaluation?

The study makes four contributions. First, it operationalises a bilingual railway knowledge base in which review state, source metadata, language variants and revision history remain queryable rather than being flattened into an anonymous vector collection. Second, it develops a bounded PostgreSQL/pgvector agent workflow that combines lexical and multilingual dense retrieval with evidence-labelled local generation while keeping retrieval policy operator-configurable. Third, it introduces a pair-grouped protocol that separates Chinese and English retrieval, QA and translation outcomes with record-level split controls. Fourth, it evaluates adaptation, retrieval and inference jointly on the defined single workstation, exposing both quality gains and failures in citation following, directional translation and runtime resource use.

## 2. Related work

### 2.1 Knowledge-enhanced Web information systems

Web information systems increasingly combine structured data management, semantic access and generative interfaces. Work in *International Journal of Web Information Systems* shows the value of integrating knowledge representation, process state and user-facing tasks rather than treating them as isolated technical components (Hubscher *et al.*, 2021). Retrieval-augmented generation (RAG) provides a practical separation between parametric generation and an externally maintained evidence collection (Lewis *et al.*, 2020). This separation is relevant to institutional knowledge systems because source records can be revised without retraining the generator. It does not, however, guarantee faithful generation: retrieval can miss relevant evidence, and a generator can still misrepresent correctly retrieved material. The present study therefore evaluates retrieval, answer overlap, evidence support and citation-format behaviour as distinct outcomes.

The retrieval layer combines complementary lexical and semantic signals. BM25 remains an efficient lexical baseline for exact domain terms (Robertson and Zaragoza, 2009), while reciprocal-rank fusion can combine rankings without requiring directly comparable score scales (Cormack *et al.*, 2009). PostgreSQL and pgvector place vector search beside governed relational metadata, allowing approval state and held-out split boundaries to be applied in the query rather than after generation. This makes knowledge governance an executable part of the information-system design.

This distinction separates the proposed system from a document-only RAG demonstration. A Web information system must maintain record identity and lifecycle state across ingestion, review, retrieval and presentation. In this study the vector representation is derived data attached to a governed record, not the primary source of truth. Deleting, revising or withholding a record can therefore change retrieval eligibility without rebuilding the entire application or relying on prompt instructions to enforce governance.

### 2.2 Retrieval-augmented generation in education

Generative AI can support explanation, feedback and access to learning resources, but education research has also identified risks involving factual reliability, learner over-reliance and opaque provenance (Kasneci *et al.*, 2023; UNESCO, 2023). Survey evidence from legal-document question answering similarly shows that specialised LLM services require coordinated data processing, model adjustment and application frameworks rather than model substitution alone (Yang *et al.*, 2024). Educational RAG addresses part of this problem by conditioning responses on course or institutional material. Many demonstrations nevertheless report a pooled answer score without isolating retrieval failure, language effects or deployment cost. Such aggregation is particularly problematic for international vocational education, where an English query may need evidence originating in Chinese regulations or textbooks. This study consequently reports Chinese and English retrieval and generation separately and preserves document-level provenance in every retrieved result.

Railway applications illustrate why this separation matters. RSChat addresses railway safety question answering (Li *et al.*, 2024), and an LLM-based human-computer training assistant has been evaluated for railway dispatching, education and professional training (Li *et al.*, 2025). Domain-specialised systems have also been proposed for train-driver advice and emergency-response generation (Luo *et al.*, 2025; Huang *et al.*, 2026). These studies establish the value of domain knowledge and training support, yet the present operational target differs: a bilingual user may formulate a concept in English, require Chinese-source evidence and need visible provenance and review state. The present work therefore evaluates cross-language access, governance and a defined single-workstation deployment together.

### 2.3 Bounded agents and resource-aware deployment

Classical agent research associates agency with situated action and a degree of autonomy (Wooldridge and Jennings, 1995), while recent LLM-agent surveys describe planning, memory, tool use and environmental action as common but variable components (Xi *et al.*, 2025). The present system uses the term **bounded workflow agent** in a narrower, operational sense. Its fixed action space is to route Chinese or English fields, invoke an operator-selected BM25, vector or hybrid retriever, enforce record eligibility, assemble numbered evidence, call a local generator and return a deterministic no-evidence or retrieval fallback. It does not set open-ended goals, browse the external Web, select arbitrary tools or change the knowledge base without human approval. Retrieval mode and top-*k* remain user- or operator-configured. This boundary distinguishes the implemented service from an autonomous general-purpose agent and makes each action auditable.

Resource-aware evaluation is part of that boundary. Efficient-AI research argues that computational cost and efficiency should be reported alongside predictive quality (Schwartz *et al.*, 2020). For a local agent, a small adapter file or a low trainable-parameter percentage does not establish low inference memory or latency. The present study therefore treats the defined workstation, peak reserved GPU memory, generation speed and latency as reproducibility conditions rather than inferring affordability from parameter count alone.

### 2.4 Bilingual domain adaptation and evaluation

Bilingual domain systems require both cross-lingual retrieval and controlled generation. BGE-M3 supports multilingual dense retrieval and was selected to provide a shared embedding space for the Chinese and English knowledge fields (Chen *et al.*, 2024). Dense similarity is not assumed to dominate lexical matching; the two methods are compared independently before fusion.

Parameter-efficient adaptation provides a second mechanism for domain specialisation. QLoRA trains low-rank adapters through a frozen 4-bit quantised base model, reducing memory requirements while retaining task adaptation capacity (Dettmers *et al.*, 2023). In this study, completion-only masking prevents the prompt from contributing to the training loss, and bilingual variants sharing a knowledge-pair identifier remain in the same split. Translation is evaluated directionally because Chinese-to-English and English-to-Chinese scores are not interchangeable. Terminology and complete-sentence translation are also kept separate.

PEFT methods reduce trainable parameters but do not guarantee a better deployed system (Ding *et al.*, 2023; Wang *et al.*, 2025). Limited-resource training remains sensitive to quantisation, sequence length, optimiser state and model-specific target modules (Lv *et al.*, 2024). Moreover, task adaptation can improve answer overlap while degrading factuality, instruction following or unrelated capabilities (Tian *et al.*, 2024; Barnett *et al.*, 2024). This motivates the present before-and-after design, which reports not only domain QA but also general-capability checks, citation behaviour and memory use.

The frontier is consequently shifting from demonstrating that a domain large language model can generate an answer to establishing whether a knowledge-enhanced workflow can be adapted, governed and served locally within bounded compute and memory. This systems perspective requires joint reporting of model quality, retrieval quality, traceability, latency and peak memory; parameter efficiency alone is not a deployment result (Ding *et al.*, 2023; Lv *et al.*, 2024; Wang *et al.*, 2025).

Domain translation creates a further risk of pooled evaluation. Prior work shows that fine-tuning effects depend on training direction, domain composition and data volume (Hu *et al.*, 2024; Vieira *et al.*, 2024; Zheng *et al.*, 2024). Consequently, this study does not interpret a single bilingual mean as evidence of robust translation. It reports terminology and sentence tasks independently in both directions and retains empty or wrong-language outputs as failures rather than silently discarding them.

## 3. System design and research method

### 3.1 Application context and requirements

The target users comprise four role groups. First, learners include international students, vocational trainees and Chinese-speaking learners requiring bilingual support. Second, railway practitioners include service personnel, locally employed staff and technical specialists in cross-border railway settings. Third, education and advisory users include vocational teachers, workplace trainers and technical consultants. Fourth, management and system-administration users include training managers and platform administrators. These roles map to three access clusters: learners and practitioners use the bilingual portal; teachers, trainers and consultants use the review console; managers and administrators use the operations console. The system must return concise answers, expose numbered evidence, preserve document metadata and run on the defined workstation without required cloud inference.

Figure 1 presents a detailed neural retrieval and QLoRA workflow for bilingual railway question answering. The upper path separates governed bilingual inputs, shared BGE-M3 encoding, parallel dense/BM25 retrieval, reciprocal-rank fusion, evidence-conditioned generation and independent evaluation. The lower path isolates completion-only adapter training with a frozen quantised base model. The figure makes the information flow, evidence serialisation and trainable boundaries explicit; its encoder and decoder blocks are schematic and do not claim identical internals for the two generators or a new model architecture.

The lower branch of Figure 1 separates offline completion-only QLoRA training from online retrieval and evaluation. Approved bilingual QA pairs are partitioned by knowledge-pair identifier; prompt tokens are masked and only answer tokens contribute to the training loss. The quantised base weights remain frozen while gradients update the low-rank adapters. The LoRA rank of 64 is independent of the RRF rank constant of 60. Retrieval contexts and downstream evaluation scores do not backpropagate through the encoder or generator. The right-hand evaluation panel distinguishes answer overlap, citation-format compliance, translation quality, an automated evidence-support proxy and resource use. Its trade-off annotations summarise the task-dependent observations in Section 4 rather than guarantee that adaptation improves every outcome; citation syntax is not evidence of factual correctness.

### 3.2 Expert-governed knowledge base

Knowledge records include Chinese and English questions and answers, evidence, original text, source document, chapter, page, task type, quality flags, review status and revision history. PostgreSQL stores governed records; pgvector stores 1,024-dimensional BGE-M3 embeddings. The production index excludes the exact records assigned to the held-out test split.

The knowledge sources comprise railway terminology, operating and safety regulations, and vocational textbook or concept material. Ingestion preserves source-document, chapter and page fields whenever available. Chinese and English questions and answers are stored in separate fields on the same logical record so that a language variant does not lose its source identity. Embeddings are linked by record identifier and model revision; they can be rebuilt without changing the reviewed text. A record marked `rejected`, `needs_revision` or as part of the frozen RAG test is excluded before ranking. This database-level decision prevents the exact test record from leaking through BM25 or vector retrieval. The retrieval outcome is defined as evidence-equivalent recall rather than exact-record recall because another admissible record may contain relevant evidence.

The three authors jointly constructed and checked the knowledge base and the training and evaluation datasets; their combined expertise covers railway vocational education, international vocational education, environmental and data analysis, and software and multimodal language-model engineering. The released governance audit, however, contains only two distinct reviewer identifiers. The paper therefore reports two auditable reviewer accounts and does not reinterpret author-level checking outside those identifiers as a third independent review pass. Because this was an author-led governance procedure rather than an independent inter-rater study, no inter-rater agreement statistic or external expert consensus is claimed. `Approved` is analysed only as an executable eligibility state.

### 3.3 Retrieval and answer generation

BM25 indexes Chinese and English question-answer fields and evidence. Dense search embeds a query with BGE-M3 and ranks chunks by vector similarity. Hybrid retrieval fuses keyword and dense rankings using reciprocal-rank fusion. The Approved-Only condition applies review status as a database-level eligibility filter. Retrieval mode and top-*k* are configured by the user or operator rather than selected autonomously by the language model. Retrieved evidence is numbered and supplied to the local generator with an instruction to answer in the query language and cite evidence labels.

Following Cormack *et al.* (2009), the reciprocal-rank-fusion score for a candidate document *d* is defined in Equation (1):

<div class="equation">RRF(d) = sum<sub>r in {BM25, vector}</sub> 1 / (k<sub>0</sub> + rank<sub>r</sub>(d)).</div>

In Equation (1), a document absent from a retriever's candidate list contributes zero for that retriever. The candidate pool is fixed at 50 per retriever and the conventional rank constant is `k0 = 60`; Equation (1) is applied before final top-*k* is varied. This controlled top-*k* comparison changes the amount of evidence delivered to the generator without changing the upstream search pool. Dense ranking uses BGE-M3 query embeddings and pgvector distance, while lexical ranking retains exact matches that are valuable for abbreviations, equipment names and regulation numbers. Every returned evidence object contains record identifier, source document, task type and retrieval score.

For query and candidate embeddings, dense retrieval uses cosine similarity:

```{=latex}
\begin{equation}
s_{\mathrm{dense}}(q,d)=\frac{\mathbf{e}_q^{\top}\mathbf{e}_d}{\lVert\mathbf{e}_q\rVert_2\lVert\mathbf{e}_d\rVert_2}.
\end{equation}
```

The completion-only adaptation objective is computed on answer positions only. If $m_t$ is one for an answer token and zero for a masked prompt token, the optimisation target is:

```{=latex}
\begin{equation}
\mathcal{L}_{\mathrm{answer}}(\theta_A)=-\frac{1}{\sum_t m_t}\sum_t m_t\log p_{\theta_0,\theta_A}(y_t\mid y_{<t},x).
\end{equation}
```

Here $\theta_0$ denotes frozen NF4 base weights and $\theta_A$ denotes trainable LoRA parameters. For an answer prediction $\hat{y}$ and reference $y$, the character-level overlap used for standalone bilingual QA is:

```{=latex}
\begin{equation}
P=\frac{|\hat{y}\cap y|}{|\hat{y}|},\qquad R=\frac{|\hat{y}\cap y|}{|y|},\qquad F1=\frac{2PR}{P+R}.
\end{equation}
```

A companion Web system provides knowledge review, search, conversational access and source-evidence display through a Vite client and FastAPI backend. It offers an application interface to the locally maintained knowledge base and generator. The experimental comparisons in this paper focus on retrieval, model outputs and resource use; the Web interface is not itself an educational-effectiveness evaluation. Generated answers are described as evidence-conditioned because citation-format compliance and semantic support are evaluated separately.

### 3.4 Bilingual QLoRA adaptation

The adaptation dataset is generated only from approved records and split by knowledge-pair identifier, task type and source document. Chinese and English variants of the same knowledge item are assigned to the same partition. The current frozen 80/10/10 corpus contains 12,178 training, 1,524 validation and 1,526 test examples, balanced equally by language, with zero knowledge-pair identifier overlap among partitions. The 120 regulation items reserved for RAG evaluation are forced into the test partition and never used for training. Loss is computed only on answer tokens; all prompt labels are masked with Minus 100.

Qwen2.5-7B-Instruct and GLM-4-9B-Chat-HF form the adaptation matrix; their architectures and post-training lineage are documented in the corresponding official technical reports (Qwen Team, 2024; GLM Team, 2024). Both were loaded with NF4 quantisation and PEFT adapters on the target GPU. Qwen exposes separate Gate Projection and Up Projection modules, whereas GLM uses a fused Gate-Up Projection; model-specific target lists are therefore required. The formal Rank 64 configuration trained 161,480,704 Qwen parameters and 190,382,080 GLM parameters, representing 3.58 and 3.45 per cent of the parameters visible to the quantised training process, respectively. Qwen3-14B is retained only as an unadapted reference generator and is associated with the Qwen3 technical report (Qwen Team, 2025).

Both runs use LoRA Rank 64, Alpha 16, Dropout 0.05, Learning Rate 2e-4, Maximum Sequence Length 1,024 and one epoch. The Per-Device Batch Size is one with 16 Gradient Accumulation Steps and a 0.03 Warm-Up Ratio. Random Seed 42 is fixed in the experiment configuration. Generation uses greedy decoding with Temperature 0 and Top-P 1, plus a task-dependent output limit. Completion-only supervision masks every prompt token with label Minus 100; therefore, training optimises the answer sequence rather than teaching the model to reproduce instructions. This choice is tested rather than assumed to improve every downstream behaviour.

### 3.5 Evaluation design

Retrieval conditions are BM25, vector, hybrid and approved hybrid. Metrics are evidence-equivalent Recall@1/3/5, MRR and mean latency. The formal generator matrix fixes no retrieval, BM25-RAG and approved hybrid-RAG so that five generators receive identical cached contexts. Automatic metrics include Answer F1, reference containment, citation-format coverage, citation omission and end-to-end latency. Citation measures are treated as instruction-following indicators, not evidence entailment or factual hallucination measures. Translation is reported independently for Chinese-to-English and English-to-Chinese terminology and sentences using SacreBLEU, chrF++ and COMET (`Unbabel/wmt22-comet-da`). This multi-dimensional protocol follows the broader principle that LLM evaluation should separate task capability, robustness and deployment behaviour rather than rely on one aggregate score (Chang *et al.*, 2024).

Evidence-equivalent Recall@*k* is one when at least one of the first *k* candidates satisfies the frozen matching rule: the same record identifier, normalised gold and candidate evidence containment in either direction, or the same source document with the gold answer contained in candidate evidence. Because exact RAG-test records are excluded from the production index, observed hits normally arise from the latter evidence-equivalence criteria rather than identity. MRR averages the reciprocal rank of the first such match. The standalone bilingual QA evaluation uses character-level F1 in both languages: precision and recall are computed from multiset character overlap after answer normalisation and whitespace removal. The paired QA tests and the bilingual mean in Figure 5 use this same metric. The RAG evaluation instead computes overlap F1 over individual Chinese characters and lower-cased Latin/alphanumeric word tokens. These two F1 definitions are reported within their respective evaluation settings, not treated as interchangeable absolute scores. In standalone QA, reference containment is a binary indicator of whether the normalised reference occurs in the prediction. In RAG, the stored reference-containment score is one when the whitespace-normalised reference is contained in the prediction, otherwise it falls back to RAG F1; empty predictions or references score zero. It is therefore a graded containment score, not a binary containment rate. Citation-format coverage records only whether the requested evidence-label syntax appears; its complement is reported as citation omission. Neither measure tests claim-evidence entailment or factual hallucination.

All core comparisons use paired samples. Mean metrics are accompanied by 2,000-resample bootstrap 95 per cent confidence intervals in the released analysis tables. Original-versus-QLoRA comparisons use two-sided paired Wilcoxon tests, Cohen's *d*<sub>z</sub> for paired effects and Holm correction across the pre-specified family recorded in the analysis configuration. Exploratory error categories are reported separately and are not treated as confirmatory hypothesis tests. Efficiency measurements use 30 Chinese short questions, 30 English short questions and 30 regulation-oriented long-context questions, with one warm-up and three measured repetitions per model condition.

Three automated system checks extend the primary protocol without introducing new human labels. First, source-only, Chinese-field, English-field and bilingual-field indexes are compared on the same 800 queries with a 512-token BGE-M3 window. Second, each of 8,000 RAG answers is segmented into claims; BGE-M3 cosine similarity estimates semantic support against all retrieved evidence and explicitly cited evidence at a pre-specified 0.45 threshold. This is a support proxy, not textual entailment or expert correctness. Third, immutable review events and before-state snapshots are audited for action and changed-field coverage. Generator resource behaviour is reported from the separate 270-measurement inference experiment for each condition.

### 3.6 Implementation and reproducibility

The application uses a Vite-based Web client, a FastAPI backend and PostgreSQL 16.14 with pgvector 0.8.4. Experiments ran on Linux with one NVIDIA GeForce RTX 3090 (24,576 MiB), 32 GB host memory, Python 3.11.15, PyTorch 2.11.0, Transformers 5.12.1, PEFT 0.18.1 and bitsandbytes 0.49.2. No multi-GPU or cloud inference resource is required by the evaluated path. This is the paper's reproducible single-workstation constraint and its comparative low-cost system profile; no purchase-price or total-cost claim is made. Qwen2.5-7B-Instruct, GLM-4-9B-Chat-HF and BGE-M3 are pinned by model revision in the frozen manifest. COMET uses the pinned `Unbabel/wmt22-comet-da` checkpoint. Input datasets, configurations, generated tables and final analysis assets are recorded with SHA-256 hashes. Table I summarises the formal implementation and experimental environment.

**Table I. Formal implementation and experimental environment.**

| Component | Formal setting |
|---|---|
| Database | PostgreSQL 16.14; pgvector 0.8.4 |
| Embedding | BAAI/bge-m3; 1,024 dimensions |
| Adapted generators | Qwen2.5-7B-Instruct; GLM-4-9B-Chat-HF |
| Reference generator | Qwen3-14B through Ollama 0.19.0 |
| Training | NF4 QLoRA; rank 64; one epoch; seed 42 |
| Hardware boundary | One RTX 3090 24 GB; 32 GB RAM; no multi-GPU or required cloud inference |
| Statistical analysis | 2,000 bootstrap resamples; paired Wilcoxon; Cohen's *d*<sub>z</sub>; Holm correction |

The code separates test-set construction, embedding, retrieval evaluation, generator evaluation, adaptation and report export. This prevents an analysis script from silently changing the production index or training split. Cached retrieval contexts are reused across generators in the formal RAG matrix, so model comparisons receive identical evidence. The manifest records the Git state and every input hash at freeze time, supporting result reconstruction and consistent comparison across runs.

Production embedding loading is local-files-only by default, preventing an unavailable model registry from becoming a hidden runtime dependency. The real review and question-answering views were rendered through the running Vite and FastAPI services rather than reconstructed as mock-ups. The fixed environment, model revisions, split statistics and asset hashes allow reconstruction of the reported single-workstation comparisons.

## 4. Results

### 4.1 Knowledge-base composition

The database contains 37,661 approved records and 37,109 complete bilingual records. After freezing the formal RAG test, 37,261 approved non-test knowledge chunks have BGE-M3 embeddings. The QLoRA test partition contains 763 knowledge pairs across four source documents. From this partition, the formal cross-source RAG set contains 400 knowledge pairs and 800 language-specific queries: 150 regulation, 127 terminology and 123 textbook/concept pairs across 17 task types. The earlier 120-pair regulation pilot is a subset of this formal set and remains available in the reproducibility artifacts; it is not reported as an additional sample in the main results.

### 4.2 Formal cross-source bilingual retrieval

**Table II. Formal cross-source evidence-equivalent retrieval on 400 knowledge pairs.**

| Retrieval | Language | Evidence R@1 | Evidence R@3 | Evidence R@5 | MRR | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | Chinese | 0.520 | 0.595 | 0.620 | 0.560 | **69.5** |
| Vector | Chinese | 0.518 | 0.653 | 0.690 | 0.588 | 134.4 |
| Hybrid approved | Chinese | **0.570** | **0.675** | **0.715** | **0.628** | 217.6 |
| BM25 | English | **0.570** | **0.670** | 0.685 | 0.620 | **59.0** |
| Vector | English | 0.463 | 0.553 | 0.580 | 0.509 | 134.4 |
| Hybrid approved | English | **0.570** | 0.668 | **0.708** | **0.620** | 209.2 |

Table II reports the formal evidence-equivalent retrieval comparison. BM25 is particularly competitive for English terminology queries, while dense retrieval records higher Chinese evidence recall in this evaluation. With the RRF candidate pool fixed at 50 for every final cutoff, hybrid fusion obtains the highest Evidence Recall@5 in both languages, although its latency is approximately three times that of BM25. Increasing the final cutoff from three to eight raises hybrid evidence recall from 0.675 to 0.743 in Chinese and from 0.668 to 0.723 in English, while mean retrieval latency remains nearly constant in this controlled evaluation (Figure 2). Approved-only and unfiltered hybrid results are identical because all admissible indexed records in this frozen experiment satisfy the approval condition. These values concern the matching rule in Section 3.5; they must not be interpreted as retrieval of the excluded test record itself.

### 4.3 QLoRA adaptation and held-out QA

Both one-epoch training runs completed without interruption. Qwen trained for 3,546 s (59.1 min), with mean training loss of 0.950, end-of-epoch validation loss of 0.720 and peak reserved GPU memory of 13.0 GiB. GLM trained for 4,571 s (76.2 min), with mean training loss of 1.146, end-of-epoch validation loss of 0.794 and peak reserved GPU memory of 15.0 GiB. The respective validation perplexities were 2.05 and 2.21.

Figure 3 presents the logged training-loss curves and one end-of-epoch validation point per model. The final logged training losses were approximately 0.700 for Qwen and 0.791 for GLM; these are distinct from the whole-run mean training losses above. The one-epoch curves describe optimisation behaviour, not demonstrated convergence across extended or repeated training schedules. The absence of an extended hyperparameter search is consistent with the resource-constrained design but limits claims about globally optimal adapter settings.

On the 1,526-example held-out bilingual QA set, Qwen QLoRA increased character-level F1 from 0.189 (95 per cent CI 0.175-0.204) to 0.398 (0.376-0.422) in Chinese and from 0.437 (0.419-0.455) to 0.648 (0.632-0.664) in English. GLM increased from 0.192 to 0.410 in Chinese and from 0.498 to 0.552 in English. All four paired gains remained significant after Holm correction (all adjusted p-values below 0.01); Cohen's *d*<sub>z</sub> was 0.569 and 0.634 for Qwen Chinese and English, and 0.614 and 0.146 for GLM. Qwen therefore provides the strongest balanced adapted QA result under this character-level metric, while the small GLM English effect cautions against pooling languages. These sample-level tests do not estimate variation across repeated training seeds.

A limited general-capability check (maximum 200 examples per subtask) found that Qwen C-Eval/MMLU accuracy changed from 0.788/0.739 to 0.775/0.728, whereas GLM changed from 0.675/0.673 to 0.683/0.679. These small changes do not indicate broad catastrophic forgetting, but the limited protocol is a regression check rather than a comprehensive general benchmark.

### 4.4 Multi-generator RAG comparisons

Approved-hybrid RAG significantly improved Answer F1 over no retrieval for every generator and language after Holm correction. The adjusted p-values for Chinese and English, respectively, were 1.89e-39 and 2.28e-42 for original Qwen, 1.13e-22 and 9.56e-31 for original GLM, 1.34e-06 and 1.41e-11 for Qwen QLoRA, 3.16e-09 and 9.37e-03 for GLM QLoRA, and 6.66e-32 and 7.58e-29 for Qwen3-14B. For original Qwen, F1 rose from 0.262 to 0.402 in Chinese and from 0.306 to 0.440 in English (paired effect sizes 0.796 and 0.798). Qwen QLoRA achieved the highest absolute RAG F1, 0.660 in Chinese and 0.651 in English, exceeding the Qwen3-14B reference values of 0.442 and 0.454. However, its incremental hybrid-RAG gains over no retrieval were only 0.030 and 0.078, compared with 0.141 and 0.133 for original Qwen. BM25 and hybrid did not have a uniform answer-quality ordering: original-Qwen BM25/hybrid F1 was 0.406/0.402 in Chinese and 0.433/0.439 in English, while Qwen-QLoRA values were 0.674/0.660 and 0.644/0.651. Similar rank reversals occurred for GLM and Qwen3. The observed answer-overlap gains therefore differed by adaptation, retrieval strategy and language.

The traceability result was different. For original Qwen, citation-format coverage moved from 0.543/0.638 under BM25 to 0.595/0.643 under hybrid in Chinese/English; Qwen3 moved from 0.715/0.900 to 0.788/0.905. By contrast, Qwen QLoRA achieved only 0.000/0.003 under hybrid, and GLM showed the same, less extreme adaptation-related pattern. In the tested conditions, QLoRA produced stronger answer overlap but weaker compliance with the citation instruction used in the evaluation targets; citation omission consequently approached one for the adapted models. This is an instruction-following failure, not a measured factual hallucination rate. For an operational workflow, Qwen2.5 QLoRA is the primary answer-quality model, while original Qwen or Qwen3 remains the safer traceability control until citation-aware adaptation is added.

### 4.5 Directional translation

Translation effects were direction- and task-dependent. Qwen COMET moved from 0.501 to 0.509 for Chinese-to-English terminology and from 0.610 to 0.614 for English-to-Chinese sentences, but fell from 0.656 to 0.599 for Chinese-to-English sentences and from 0.596 to 0.485 for English-to-Chinese terminology. Its lexical metrics show the same lack of uniform benefit: Chinese-to-English terminology chrF++ increased from 10.41 to 16.69, whereas sentence chrF++ fell from 47.43 to 31.28.

Figure 4 visualises the before-and-after COMET changes by direction and task. The paired bars show increases in some conditions and decreases in others, so the result cannot be summarised as a uniform improvement or regression.

GLM QLoRA is a clear failure case. Its sentence COMET dropped from 0.667 to 0.348 for Chinese-to-English and from 0.742 to 0.343 for English-to-Chinese; corpus BLEU was zero in both directions. Automated inspection found 2,168 empty outputs among 2,634 translation examples (82.3 per cent). This result does not support a general claim that QA-oriented completion-only adaptation improves translation. Terminology and sentence translation are therefore reported separately from QA evaluation.

### 4.6 Resource use and automated error analysis

All five deployment conditions completed 270 measurements without execution failure or OOM. GPU memory measurements are expressed in GiB (2^30 bytes). Original Qwen and GLM reached 5.49 and 6.76 GiB peak PyTorch reserved memory and generated at 31.9 and 22.7 tokens/s. Their QLoRA variants reached 16.11 and 19.49 GiB and generated at 17.5 and 11.9 tokens/s. Peak allocated memory was 5.35 and 6.48 GiB for the original models, compared with 15.40 and 7.20 GiB for the QLoRA variants. Reserved memory includes caching-allocator reservations and should not be interpreted as memory occupied by live tensors or as the intrinsic adapter overhead. Qwen3-14B through Ollama reached 14.26 GiB of GPU-process memory and 78.5 tokens/s; this memory statistic is not PyTorch reserved memory, and cross-backend memory and timing comparisons are descriptive. The Qwen and GLM adapters occupy approximately 627 and 746 MB. The low 2.63 s GLM QLoRA mean latency reflects abnormally short or empty outputs and is not an efficiency advantage. All configurations fit the target 24 GB workstation, but original Qwen provides the lowest resource cost and Qwen QLoRA the strongest QA quality within the limit.

Figure 5 relates the bilingual mean standalone character-level F1 to generation latency and peak reserved GPU memory for the four Qwen2.5/GLM original and QLoRA conditions. The quality and resource measurements come from their respective evaluation workloads. The plot is a descriptive quality-resource comparison, not a controlled causal estimate of adapter overhead; Qwen3 through Ollama is not included in this figure.

Automated failure flags provide additional descriptive comparisons for the aggregate results. At top three, hybrid retrieval failed to return evidence satisfying the frozen equivalence rule for 32.5 per cent of Chinese and 33.3 per cent of English queries. The joint event of an evidence hit and a low-overlap answer occurred in 18.0 per cent of all Chinese queries and 4.3 per cent of all English queries for original Qwen; these denominators include all queries, not only retrieval hits. Qwen QLoRA reduced these rates to 4.0 and 1.5 per cent, but omitted citation labels in 100.0 and 99.8 per cent. Figure 6 reports a separate summary of the standalone bilingual-QA and domain/translation outputs: terminology mismatch, low answer overlap, empty answer, wrong-language flag and overlong answer. Each bar averages the condition/task-level proportions among groups in which that flag occurred; it is not a pooled sample-level rate or a plot of the RAG retrieval/citation statistics above. Flags are non-exclusive, and terminology mismatch denotes failure of the reference-containment check rather than an expert judgement of terminology correctness.

### 4.7 Index, evidence-support and governance validation

**Table III. Supplementary information-system validation results.**

| Validation | Chinese | English | Operational result |
|---|---:|---:|---|
| Bilingual-field hybrid index, Evidence Recall@5 | 0.718 | 0.675 | Highest balanced mean (0.696) |
| Original Qwen hybrid, supported-claim proxy | 0.878 | 0.836 | Citation precision 0.955/0.970 |
| Qwen QLoRA hybrid, supported-claim proxy | 0.964 | 0.905 | Citation recall 0.000/0.002 |
| Governance history | - | - | 1,337 events; 82 edits; two recorded reviewers |

Table III reports three supplementary information-system validations. The field comparison reports different language-specific outcomes: English-only hybrid is strongest for English Evidence Recall@5 (0.713) but falls to 0.560 in Chinese, whereas the bilingual index gives the highest balanced mean. Across 27,668 segmented claims, hybrid retrieval generally records higher automated semantic support, but high support does not repair absent citations: Qwen QLoRA retains strong support scores while almost never attaching a valid label. The governance database contains 37,664 records and 1,337 review events, including before-state snapshots for edits; only three current records are rejected, so the experiment does not provide an approved-versus-rejected quality comparison. The audit exposes two distinct reviewer identifiers, consistent with the reporting boundary in Section 3.2. Figure 7 separates the index-field, evidence-support and governance-audit checks into panels A-C rather than combining them into a single score.

## 5. Discussion

### 5.1 Answers to the research questions

**RQ1 concerned bilingual retrieval.** The evidence-equivalent comparison produced language-specific outcomes. Dense retrieval had higher recall in Chinese in this evaluation, whereas exact terminology made BM25 competitive in English. Hybrid fusion reached the highest Evidence Recall@5 in both languages, and the field comparison reported the strongest balanced result for the bilingual index even though an English-only index was best for English alone. Hybrid had higher latency, while lexical retrieval remained a credible lower-cost configuration. These conclusions apply to the pre-specified equivalence rule, not exact retrieval of records excluded from the index (Robertson and Zaragoza, 2009; Cormack *et al.*, 2009).

**RQ2 concerned retrieval strategy, enforceable eligibility and answer behaviour.** RAG produced higher answer overlap than no retrieval for every evaluated generator, but retrieval and answer-quality rankings were not identical. The adaptation and RAG conditions also showed different answer-overlap and citation-format patterns. Automated semantic support remained high for adapted answers while citation recall approached zero, so relevance to retrieved evidence and visible attribution should be treated as separate requirements. This qualifies railway systems that report aggregate gains from fine-tuning or RAG (Li *et al.*, 2024; Li *et al.*, 2025; Luo *et al.*, 2025; Huang *et al.*, 2026). Citation-aware targets, constrained citation insertion and human-validated entailment checking are required before operational use.

The formal generator matrix confirms that approved-hybrid RAG improves Answer F1 over no retrieval for every evaluated generator in both Chinese and English, while the magnitude of the gain remains model- and language-dependent.

The approved-only and unfiltered hybrid conditions were identical because every admissible record in the frozen production index was already approved. The implementation and audit verify that eligibility can be enforced and traced, thereby answering the governance part of RQ2. They do not provide an approved-versus-unreviewed quality comparison. The contribution is the executable review-state mechanism, not a claim that approval alone raises F1.

**RQ3 concerned bilingual adaptation and translation.** Completion-only QLoRA substantially improves held-out QA, particularly for Qwen, but adaptation quality is task-specific. QA gains coexist with asymmetric Qwen translation regressions and severe GLM sentence-translation failure. This mirrors concerns that domain-limited fine-tuning can overfit task form or translation direction (Hu *et al.*, 2024; Vieira *et al.*, 2024). The practical implication is that QA, terminology translation, sentence translation and citation compliance need independent acceptance thresholds; no pooled bilingual score can justify deployment.

The existing retrieval, index-field, adaptation and RAG comparisons provide component-level evidence for the reported performance differences. They show, for example, that bilingual fields balance the two query languages and that retrieval improves answer overlap across generators. However, they do not isolate a single causal mechanism for every gain: multilingual representation, lexical matching, training adaptation and prompt behaviour remain partly coupled. Further controlled ablations and expert-rated error analysis are therefore needed before attributing the improvements to one component.

### 5.2 System and deployment implications

The results support evaluating domain adaptation as a Web information-system workflow, not as a single-model treatment. Governance determines admissible evidence, retrieval determines its availability, generation determines its use, and the interface determines whether provenance is visible. PostgreSQL remains the authoritative store while pgvector adds semantic access. On the defined workstation, original Qwen is the lower-resource option and Qwen QLoRA the stronger QA option, but citation and translation require separate acceptance gates. These findings support local, staged deployment with reviewed retrieval, visible evidence and explicit no-evidence behaviour; they do not establish educational effectiveness.

## 6. Practical implications and conclusion

This study demonstrates a bounded AI workflow for bilingual railway education and workforce development. Hybrid retrieval achieved the strongest Evidence Recall@5, while bilingual fields produced the strongest balanced index result. Qwen2.5 QLoRA achieved the strongest held-out QA and RAG answer-overlap scores within the workstation boundary, but used more reserved memory, generated more slowly and showed weaker citation compliance and inconsistent translation. The main contribution is an executable workflow that keeps review state, held-out exclusions, retrieval, provenance and resource limits visible. The system should support learning and decisions, not replace current safety-critical authority; teachers and operators should inspect the displayed evidence before use.

## Declarations

**Data availability:** Data are available from the corresponding author on reasonable request, subject to rights and access conditions.

**Ethics and consent:** No learner personal data were used.

**Conflict of interest:** None.

**Funding:** None.

**Author contributions:** All authors contributed to the study and approved the manuscript.

**Acknowledgements:** None.

**Use of generative AI:** AI was used only to polish English grammar. The authors take full responsibility for the manuscript.

## References

Barnett, S., Brannelly, Z., Kurniawan, S. and Wong, S. (2024), “Fine-tuning or fine-failing? Debunking performance myths in large language models”, arXiv:2406.11201, doi: 10.48550/arXiv.2406.11201, available at: <https://arxiv.org/abs/2406.11201>.

Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L., Zhu, K., Chen, H., Yi, X., Wang, C., Wang, Y., Ye, W., Zhang, Y., Chang, Y., Yu, P.S., Yang, Q. and Xie, X. (2024), “A survey on evaluation of large language models”, *ACM Transactions on Intelligent Systems and Technology*, Vol. 15 No. 3, pp. 1-45, doi: 10.1145/3641289.

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D. and Liu, Z. (2024), “BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation”, arXiv:2402.03216, doi: 10.48550/arXiv.2402.03216, available at: <https://arxiv.org/abs/2402.03216>.

Cormack, G.V., Clarke, C.L.A. and Buettcher, S. (2009), “Reciprocal rank fusion outperforms Condorcet and individual rank learning methods”, *Proceedings of the 32nd International ACM SIGIR Conference*, pp. 758-759, doi: 10.1145/1571941.1572114.

Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L. (2023), “QLoRA: Efficient finetuning of quantized LLMs”, *Advances in Neural Information Processing Systems*, Vol. 36, arXiv:2305.14314, doi: 10.48550/arXiv.2305.14314, available at: <https://arxiv.org/abs/2305.14314>.

Ding, N., Qin, Y., Yang, G., Wei, F., Yang, Z., Su, Y., Hu, S., Chen, Y., Chan, C.-M. and Chen, W. (2023), “Parameter-efficient fine-tuning of large-scale pre-trained language models”, *Nature Machine Intelligence*, Vol. 5 No. 3, pp. 220-235, doi: 10.1038/s42256-023-00626-4.

GLM Team (2024), “ChatGLM: a family of large language models from GLM-130B to GLM-4 all tools”, arXiv:2406.12793, doi: 10.48550/arXiv.2406.12793, available at: <https://arxiv.org/abs/2406.12793>.

Hu, T., Zhang, P., Yang, B., Xie, J., Wong, D.F. and Wang, R. (2024), “Large language model for multi-domain translation: benchmarking and domain CoT fine-tuning”, *Findings of the Association for Computational Linguistics: EMNLP 2024*, pp. 5726-5746, doi: 10.18653/v1/2024.findings-emnlp.328.

Huang, L., Liu, Z., Yu, C., Zhu, T. and Yan, B. (2026), “Emergency operation scheme generation for urban rail transit train door systems using retrieval-augmented large language models”, *Sensors*, Vol. 26 No. 6, 2006, doi: 10.3390/s26062006.

Hubscher, G., Geist, V., Auer, D., Hubscher, N. and Kung, J. (2021), “Representation and presentation of knowledge and processes - an integrated approach for a dynamic communication-intensive environment”, *International Journal of Web Information Systems*, Vol. 17 No. 6, pp. 669-697, doi: 10.1108/IJWIS-03-2021-0031.

Kasneci, E., Sessler, K., Küchemann, S., Bannert, M., Dementieva, D., Fischer, F., Gasser, U., Groh, G., Günnemann, S., Hüllermeier, E., Krusche, S., Kutyniok, G., Michaeli, T., Nerdel, C., Pfeffer, J., Poquet, O., Sailer, M., Schmidt, A., Seidel, T., Stadler, M., Weller, J., Kuhn, J. and Kasneci, G. (2023), “ChatGPT for good? On opportunities and challenges of large language models for education”, *Learning and Individual Differences*, Vol. 103, 102274, doi: 10.1016/j.lindif.2023.102274.

Lahnalampi, A. (2024), *Utilizing Large Language Models in Rail Design Projects*, Master's thesis, Aalto University, available at: <https://aaltodoc.aalto.fi/items/87b7ac34-756c-4db2-8bea-f9b9e6b453ed>.

Lawrence, M., Bullock, R. and Liu, Z. (2019), *China's High-Speed Rail Development*, World Bank, Washington, DC, doi: 10.1596/978-1-4648-1425-9.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S. and Kiela, D. (2020), “Retrieval-augmented generation for knowledge-intensive NLP tasks”, *Advances in Neural Information Processing Systems*, Vol. 33, pp. 9459-9474, available at: <https://arxiv.org/abs/2005.11401>.

Li, J., Li, C., Niu, S. and Dai, B. (2024), “RSChat: intelligent question answering model for railway safety knowledge”, *2024 IEEE/ACIS 24th International Conference on Computer and Information Science*, pp. 55-60, doi: 10.1109/ICIS61260.2024.10778327.

Li, Y., Chen, J., Luo, X. and Zheng, H. (2025), “Intelligent human-computer interactive training assistant system for rail systems”, *High-speed Railway*, Vol. 3 No. 1, pp. 64-77, doi: 10.1016/j.hspr.2025.02.001.

Luo, Y.C., Xun, J., Wang, W., Zhang, R.Z. and Zhao, Z.C. (2025), “A driver advisory system based on large language model for high-speed train”, arXiv:2501.07837, doi: 10.48550/arXiv.2501.07837.

Lv, K., Yang, Y., Liu, T., Guo, Q. and Qiu, X. (2024), “Full parameter fine-tuning for large language models with limited resources”, *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*, pp. 8187-8198, doi: 10.18653/v1/2024.acl-long.445.

Qwen Team (2024), “Qwen2.5 technical report”, arXiv:2412.15115, doi: 10.48550/arXiv.2412.15115, available at: <https://arxiv.org/abs/2412.15115>.

Qwen Team (2025), “Qwen3 technical report”, arXiv:2505.09388, doi: 10.48550/arXiv.2505.09388, available at: <https://arxiv.org/abs/2505.09388>.

Robertson, S. and Zaragoza, H. (2009), “The probabilistic relevance framework: BM25 and beyond”, *Foundations and Trends in Information Retrieval*, Vol. 3 No. 4, pp. 333-389, doi: 10.1561/1500000019.

Schwartz, R., Dodge, J., Smith, N.A. and Etzioni, O. (2020), “Green AI”, *Communications of the ACM*, Vol. 63 No. 12, pp. 54-63, doi: 10.1145/3381831.

Song, Y., Tang, Y. and Liu, R. (2025), “Large language model and application for railway track management based on domain specialization”, *The Proceedings of the 11th International Conference on Traffic and Transportation Studies*, Vol. 616, pp. 194-204, doi: 10.1007/978-981-97-9644-1_21.

Tian, K., Mitchell, E., Yao, H., Manning, C. and Finn, C. (2024), “Fine-tuning language models for factuality”, *International Conference on Learning Representations*, available at: <https://proceedings.iclr.cc/paper_files/paper/2024/hash/c361ae924c23cafca6033610d25dbc65-Abstract-Conference.html>.

UNESCO (2023), *Guidance for Generative AI in Education and Research*, UNESCO, Paris, available at: <https://unesdoc.unesco.org/ark:/48223/pf0000386693>.

Vieira, I., Allred, W., Lankford, S., Castilho, S. and Way, A. (2024), “How much data is enough data? Fine-tuning large language models for in-house translation”, *Proceedings of the 16th Conference of the Association for Machine Translation in the Americas*, pp. 236-249, available at: <https://aclanthology.org/2024.amta-research.20/>.

Wandelt, S., Zheng, C., Wang, S., Liu, Y. and Sun, X. (2024), “Large language models for intelligent transportation: a review of the state of the art and challenges”, *Applied Sciences*, Vol. 14 No. 17, 7455, doi: 10.3390/app14177455.

Wang, H. and Li, Y.-F. (2024), “Large-scale language models for PHM in railway systems: potential applications, limitations, and solutions”, *Proceedings of the 6th International Conference on Electrical Engineering and Information Technologies for Rail Transportation*, Vol. 1137, pp. 591-599, doi: 10.1007/978-981-99-9311-6_59.

Wang, L., Chen, S., Jiang, L., Pan, S., Cai, R., Yang, S. and Yang, F. (2025), “Parameter-efficient fine-tuning in large language models: a survey of methodologies”, *Artificial Intelligence Review*, Vol. 58 No. 8, 227, doi: 10.1007/s10462-025-11236-4.

Wooldridge, M. and Jennings, N.R. (1995), “Intelligent agents: theory and practice”, *The Knowledge Engineering Review*, Vol. 10 No. 2, pp. 115-152, doi: 10.1017/S0269888900008122.

Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., Hong, B., Zhang, M., Wang, J., Jin, S., Zhou, E., Zheng, R., Fan, X., Wang, X., Xiong, L., Zhou, Y., Wang, W., Jiang, C., Zou, Y., Liu, X., Yin, Z., Dou, S., Weng, R., Qin, W., Zheng, Y., Qiu, X., Huang, X., Zhang, Q. and Gui, T. (2025), “The rise and potential of large language model based agents: a survey”, *Science China Information Sciences*, Vol. 68 No. 2, 121101, doi: 10.1007/s11432-024-4222-0.

Xu, F. (2018), “Future of high-speed railway and university education”, in *The Belt and Road*, Springer, Singapore, pp. 167-183, doi: 10.1007/978-981-13-1105-5_7.

Yang, X., Wang, Z., Wang, Q., Wei, K., Zhang, K. and Shi, J. (2024), “Large language models for automated Q&A involving legal documents: a survey on algorithms, frameworks and applications”, *International Journal of Web Information Systems*, Vol. 20 No. 4, pp. 413-435, doi: 10.1108/IJWIS-12-2023-0256.

Zheng, J., Hong, H., Liu, F., Wang, X., Su, J., Liang, Y. and Wu, S. (2024), “Fine-tuning large language models for domain-specific machine translation”, arXiv:2402.15061, doi: 10.48550/arXiv.2402.15061.

Zheng, O., Abdel-Aty, M., Wang, D., Wang, C. and Ding, S. (2023), “TrafficSafetyGPT: tuning a pre-trained large language model to a domain-specific expert in transportation safety”, arXiv:2307.15311, doi: 10.48550/arXiv.2307.15311.
