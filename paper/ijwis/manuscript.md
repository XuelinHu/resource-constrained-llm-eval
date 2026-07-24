# A Knowledge-Enhanced Large Language Model Web Information System for Bilingual Railway Vocational Education

## Structured abstract

**Purpose.** This study develops and evaluates a knowledge-enhanced Web information system that supports Chinese and English question answering for railway vocational education. It addresses the need of international learners to access terminology, regulations and textbook knowledge while preserving source traceability under limited local computing resources.

**Design/methodology/approach.** The system integrates an approval-controlled bilingual knowledge base with PostgreSQL and pgvector, BM25 retrieval, BGE-M3 dense retrieval, reciprocal-rank hybrid fusion and locally deployed language models. A leakage-controlled test set of 400 knowledge pairs compares lexical, dense and hybrid retrieval. Five generators are evaluated under no-retrieval, BM25-RAG and approved-hybrid-RAG conditions. Directional translation, QLoRA adaptation and deployment efficiency are evaluated separately on a single 24 GB GPU workstation.

**Findings.** The approved knowledge base contains 37,661 records, of which 37,109 have complete bilingual question-answer fields. Hybrid Recall@5 reached 0.715 in Chinese and 0.708 in English; dense retrieval contributed more in Chinese and BM25 more in English. Qwen2.5 QLoRA increased held-out QA Char F1 from 0.189 to 0.398 in Chinese and from 0.437 to 0.648 in English. RAG produced significant gains for all five generators, but adaptation reduced the incremental retrieval gain and QLoRA models frequently omitted evidence labels. Translation gains were not consistent across directions, and GLM-4 QLoRA suffered severe sentence-translation failure. Every deployment condition remained below 20 GB peak memory with no failures.

**Originality/value.** The study connects expert-governed bilingual knowledge, evidence-traceable RAG and resource-constrained deployment in a domain-specific Web information system. It reports language-disaggregated results and treats expert approval as a controllable retrieval condition rather than an informal data-cleaning step.

**Keywords:** knowledge-enhanced large language models; Web information systems; retrieval-augmented generation; bilingual education; railway vocational education; pgvector

## 1. Introduction

Railway vocational education combines safety-critical regulations, specialised terminology and procedural textbook knowledge. International students must often move between Chinese source material and English learning support. Generic large language models can produce fluent answers, but their parametric knowledge does not expose a stable boundary between verified railway knowledge and unsupported generation. This is a Web information system problem as much as a model problem: knowledge must be acquired, governed, retrieved, presented and traced to its source.

Existing studies commonly evaluate either general-purpose language models or retrieval-augmented generation in isolation. Three gaps motivate this work. First, bilingual educational retrieval is rarely evaluated separately by query language. Second, expert review is often described as preprocessing rather than implemented as a queryable governance state. Third, deployment studies frequently omit the interaction between retrieval quality, answer faithfulness and resource constraints on ordinary institutional hardware.

This study addresses the following research questions:

- **RQ1:** How accurately do BM25, pgvector dense retrieval and hybrid fusion retrieve held-out evidence for Chinese and English railway questions?
- **RQ2:** How do retrieval strategy and expert-approved filtering affect answer quality, citation coverage and hallucination risk?
- **RQ3:** What bilingual translation quality is achieved in each direction, and how does completion-only QLoRA affect domain QA without contaminating the held-out test set?
- **RQ4:** Can the resulting system operate within the latency and memory constraints of a single 24 GB GPU workstation?

The contributions are: an expert-governed bilingual railway knowledge base; a PostgreSQL/pgvector Web information system with keyword, vector and hybrid retrieval; a leakage-controlled bilingual evaluation protocol; and a deployment analysis that jointly reports quality and efficiency.

## 2. Related work

### 2.1 Knowledge-enhanced Web information systems

Web information systems increasingly combine structured data management, semantic access and generative interfaces. Retrieval-augmented generation (RAG) provides a practical separation between parametric generation and an externally maintained evidence collection (Lewis et al., 2020). This separation is relevant to institutional knowledge systems because source records can be revised without retraining the generator. It does not, however, guarantee faithful generation: retrieval can miss the relevant record, and a generator can still misrepresent correctly retrieved evidence. The present study therefore evaluates retrieval, answer overlap, evidence hit and citation behaviour as distinct outcomes.

The retrieval layer combines complementary lexical and semantic signals. BM25 remains an efficient lexical baseline for exact domain terms (Robertson and Zaragoza, 2009), while reciprocal-rank fusion can combine rankings without requiring directly comparable score scales (Cormack et al., 2009). PostgreSQL and pgvector place vector search beside governed relational metadata, allowing approval state and held-out split boundaries to be applied in the query rather than after generation. This makes knowledge governance an executable part of the information-system design.

### 2.2 Retrieval-augmented generation in education

Generative AI can support explanation, feedback and access to learning resources, but education research has also identified risks involving factual reliability, learner over-reliance and opaque provenance (Kasneci et al., 2023; UNESCO, 2023). Educational RAG addresses part of this problem by grounding responses in course or institutional material. Many demonstrations nevertheless report a pooled answer score without isolating retrieval failure, language effects or deployment cost. Such aggregation is particularly problematic for international vocational education, where an English query may need evidence originating in Chinese regulations or textbooks. This study consequently reports Chinese and English retrieval and generation separately and preserves document-level provenance in every retrieved result.

### 2.3 Bilingual domain adaptation and evaluation

Bilingual domain systems require both cross-lingual retrieval and controlled generation. BGE-M3 supports multilingual dense retrieval and was selected to provide a shared embedding space for the Chinese and English knowledge fields (Chen et al., 2024). Dense similarity is not assumed to dominate lexical matching; the two methods are compared independently before fusion.

Parameter-efficient adaptation provides a second mechanism for domain specialisation. QLoRA trains low-rank adapters through a frozen 4-bit quantised base model, reducing memory requirements while retaining task adaptation capacity (Dettmers et al., 2023). In this study, completion-only masking prevents the prompt from contributing to the training loss, and bilingual variants sharing a knowledge-pair identifier remain in the same split. Translation is evaluated directionally because Chinese-to-English and English-to-Chinese scores are not interchangeable. Terminology and complete-sentence translation are also kept separate.

## 3. System design and research method

### 3.1 Application context and requirements

The target users are railway vocational teachers, Chinese-speaking learners and international students requiring English support. The system must return concise answers, expose numbered evidence, preserve document metadata and run locally where cloud use is restricted.

### 3.2 Expert-governed knowledge base

Knowledge records include Chinese and English questions and answers, evidence, original text, source document, chapter, page, task type, quality flags, review status and revision history. PostgreSQL stores governed records; pgvector stores 1,024-dimensional BGE-M3 embeddings. The production index excludes the held-out test split.

### 3.3 Retrieval and answer generation

BM25 indexes Chinese and English question-answer fields and evidence. Dense search embeds a query with BGE-M3 and ranks chunks by vector similarity. Hybrid retrieval fuses keyword and dense rankings using reciprocal-rank fusion. The `approved_only` condition applies expert status as a database-level filter. Retrieved evidence is numbered and supplied to the local generator with an instruction to answer in the query language and cite evidence labels.

### 3.4 Bilingual QLoRA adaptation

The adaptation dataset is generated only from approved records and split by knowledge-pair identifier, task type and source document. Chinese and English variants of the same knowledge item are assigned to the same partition. The current frozen 80/10/10 corpus contains 12,178 training, 1,524 validation and 1,526 test examples, balanced equally by language, with zero pair overlap among all partitions. The 120 regulation items reserved for RAG evaluation are forced into the test partition and never used for training. Loss is computed only on answer tokens; all prompt labels are masked with `-100`.

Qwen2.5-7B-Instruct and GLM-4-9B-Chat-HF form the adaptation matrix. Both were loaded with NF4 quantisation and PEFT adapters on the target GPU. Qwen exposes separate `gate_proj` and `up_proj` modules, whereas GLM uses a fused `gate_up_proj`; model-specific target lists are therefore required. The formal rank-64 configuration trained 161,480,704 Qwen parameters and 190,382,080 GLM parameters, representing 3.58 and 3.45 per cent of the parameters visible to the quantised training process, respectively.

### 3.5 Evaluation design

Retrieval conditions are BM25, vector, hybrid and approved hybrid. Metrics are Recall@1/3/5, MRR and mean latency. The regulation pilot includes six answer conditions; the formal generator matrix fixes no retrieval, BM25-RAG and approved hybrid-RAG so that five generators receive identical cached contexts. Automatic metrics include answer F1, reference containment, citation-format coverage, a citation-format-based hallucination proxy and end-to-end latency. Citation measures are treated as instruction-following proxies, not evidence entailment. Translation is reported independently for Chinese-to-English and English-to-Chinese terminology and sentences using SacreBLEU, chrF++ and COMET (`Unbabel/wmt22-comet-da`).

All core comparisons use paired samples. Mean metrics are accompanied by 2,000-resample bootstrap 95 per cent confidence intervals. Original-versus-QLoRA comparisons use two-sided paired Wilcoxon tests, paired standardised effect sizes and Holm correction across the pre-specified comparison family. Exploratory error categories are reported separately and are not treated as confirmatory hypothesis tests. Efficiency measurements use 30 Chinese short questions, 30 English short questions and 30 regulation-oriented long-context questions, with one warm-up and three measured repetitions per model condition.

## 4. Results

### 4.1 Knowledge-base composition

The database contains 37,661 approved records and 37,109 complete bilingual records. After freezing the formal RAG test, 37,261 approved non-test knowledge chunks have BGE-M3 embeddings. The QLoRA test partition contains 763 knowledge pairs across four source documents. From this partition, the formal cross-source RAG set contains 400 knowledge pairs and 800 language-specific queries: 150 regulation, 127 terminology and 123 textbook/concept pairs across 17 task types. The original 120 regulation pairs remain a separately reported specialised subset.

### 4.2 Regulation-only pilot retrieval

| Retrieval | Language | R@1 | R@3 | R@5 | MRR | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | Chinese | 0.625 | 0.708 | 0.742 | 0.672 | 48.7 |
| Vector | Chinese | 0.575 | 0.717 | 0.742 | 0.648 | 125.4 |
| Hybrid approved | Chinese | **0.667** | **0.717** | **0.758** | **0.701** | 185.0 |
| BM25 | English | 0.517 | 0.642 | 0.667 | 0.580 | 70.0 |
| Vector | English | 0.483 | 0.567 | 0.575 | 0.523 | 361.7 |
| Hybrid approved | English | **0.525** | **0.667** | **0.708** | **0.595** | 185.7 |

In this regulation-only pilot, hybrid retrieval gives the strongest Recall@5 in both languages. English queries remain consistently harder, which supports reporting language-disaggregated results rather than a pooled bilingual score. These values will not be treated as the final cross-source system results.

### 4.3 Formal cross-source bilingual retrieval

| Retrieval | Language | R@1 | R@3 | R@5 | MRR | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | Chinese | 0.520 | 0.595 | 0.620 | 0.560 | **69.5** |
| Vector | Chinese | 0.518 | 0.653 | 0.690 | 0.588 | 134.4 |
| Hybrid approved | Chinese | **0.570** | **0.675** | **0.715** | **0.628** | 217.6 |
| BM25 | English | **0.570** | **0.670** | 0.685 | 0.620 | **59.0** |
| Vector | English | 0.463 | 0.553 | 0.580 | 0.509 | 134.4 |
| Hybrid approved | English | **0.570** | 0.668 | **0.708** | **0.620** | 209.2 |

The larger cross-source evaluation changes the interpretation of the pilot. BM25 is particularly competitive for English terminology queries, while dense retrieval contributes more strongly to Chinese evidence recall. With the RRF candidate pool fixed at 50 for every final cutoff, hybrid fusion obtains the highest Recall@5 in both languages, although its latency is approximately three times that of BM25. Increasing the final cutoff from three to eight raises approved-hybrid recall from 0.675 to 0.743 in Chinese and from 0.668 to 0.723 in English, while mean retrieval latency remains nearly constant because the candidate pool is controlled. Approved-only and unfiltered hybrid results are identical because all admissible indexed records in this frozen experiment satisfy the approval condition.

### 4.4 Regulation-only pilot answer generation

| Condition | Language | Answer F1 | Citation coverage | Hallucination proxy | End-to-end (ms) |
|---|---:|---:|---:|---:|---:|
| No retrieval | Chinese | 0.243 | 0.000 | 1.000 | 3,243.1 |
| BM25-RAG | Chinese | **0.621** | 0.967 | 0.033 | **1,141.8** |
| Vector-RAG | Chinese | 0.623 | 0.967 | 0.033 | 1,149.4 |
| Hybrid-RAG approved | Chinese | 0.604 | **0.975** | **0.025** | 1,413.8 |
| No retrieval | English | 0.243 | 0.000 | 1.000 | 3,495.5 |
| BM25-RAG | English | **0.498** | **0.992** | **0.008** | **1,322.5** |
| Vector-RAG | English | 0.492 | 0.950 | 0.050 | 1,425.4 |
| Hybrid-RAG approved | English | 0.489 | **0.992** | **0.008** | 1,586.2 |

RAG more than doubled F1 relative to no retrieval in both languages while reducing mean response time because evidence-constrained answers were shorter. Hybrid retrieval obtained the strongest evidence recall, but this did not translate into the highest answer F1; BM25-RAG was the strongest English generation condition and vector-RAG was marginally strongest in Chinese. This distinction supports evaluating retrieval and generation separately. Retrieval-only returned source-language evidence and is therefore excluded from the generative comparison, particularly for English questions.

### 4.5 QLoRA adaptation and held-out QA

Both one-epoch training runs converged without interruption. Qwen trained for 3,546 s (59.1 min), with final training and validation losses of 0.950 and 0.720 and peak reserved memory of 13.0 GB. GLM trained for 4,571 s (76.2 min), with losses of 1.146 and 0.794 and peak reserved memory of 15.0 GB. The respective validation perplexities were 2.05 and 2.21.

On the 1,526-example held-out bilingual QA set, Qwen QLoRA increased Char F1 from 0.189 (95 per cent CI 0.175-0.204) to 0.398 (0.376-0.422) in Chinese and from 0.437 (0.419-0.455) to 0.648 (0.632-0.664) in English. GLM increased from 0.192 to 0.410 in Chinese and from 0.498 to 0.552 in English. All four paired gains remained significant after Holm correction (`p` < 0.000001); paired effect sizes were 0.569 and 0.634 for Qwen Chinese and English, and 0.614 and 0.146 for GLM. Qwen therefore provides the strongest balanced adapted QA result, while the small GLM English effect cautions against pooling languages.

A limited general-capability check (maximum 200 examples per subtask) found that Qwen C-Eval/MMLU accuracy changed from 0.788/0.739 to 0.775/0.728, whereas GLM changed from 0.675/0.673 to 0.683/0.679. These small changes do not indicate broad catastrophic forgetting, but the limited protocol is a regression check rather than a comprehensive general benchmark.

### 4.6 Multi-generator RAG and interaction effects

Approved hybrid RAG significantly improved Answer F1 over no retrieval for every generator and language after Holm correction. For original Qwen, F1 rose from 0.262 to 0.402 in Chinese and from 0.306 to 0.440 in English (paired effect sizes 0.796 and 0.798). Qwen QLoRA achieved the highest absolute RAG F1, 0.660 in Chinese and 0.651 in English, exceeding the Qwen3-14B reference values of 0.442 and 0.454. However, its incremental hybrid-RAG gains over no retrieval were only 0.030 and 0.078, compared with 0.141 and 0.133 for original Qwen. Adaptation and retrieval are therefore complementary in answer overlap, but with diminishing returns.

The traceability result points in the opposite direction. Approved hybrid citation-format coverage was 0.595/0.643 for original Qwen and 0.788/0.905 for Qwen3 in Chinese/English, but 0.000/0.003 for Qwen QLoRA. GLM showed the same, less extreme pattern. QLoRA specialised answer content while weakening compliance with a citation instruction absent from its training targets. The citation-based hallucination proxy consequently approaches one for adapted models and must not be read as a factual hallucination rate. For an operational system, Qwen2.5 QLoRA is the primary answer-quality model, while original Qwen or Qwen3 remains the safer traceability control until citation-aware adaptation is added.

### 4.7 Directional translation

Translation effects were direction- and task-dependent. Qwen COMET moved from 0.501 to 0.509 for Chinese-to-English terminology and from 0.610 to 0.614 for English-to-Chinese sentences, but fell from 0.656 to 0.599 for Chinese-to-English sentences and from 0.596 to 0.485 for English-to-Chinese terminology. Its lexical metrics show the same lack of uniform benefit: Chinese-to-English terminology chrF++ increased from 10.41 to 16.69, whereas sentence chrF++ fell from 47.43 to 31.28.

GLM QLoRA is a clear failure case. Its sentence COMET dropped from 0.667 to 0.348 for Chinese-to-English and from 0.742 to 0.343 for English-to-Chinese; corpus BLEU was zero in both directions. Automated inspection found 2,168 empty outputs among 2,634 translation examples (82.3 per cent). This result rules out a claim that QA-oriented completion-only adaptation generally improves translation and shows why terminology and sentence translation must remain separate from QA evaluation.

### 4.8 Resource use and automated error analysis

All five deployment conditions completed 270 measurements without failure or OOM. Original Qwen and GLM used 5.49 and 6.76 GB peak reserved memory and generated at 31.9 and 22.7 tokens/s. Their QLoRA variants used 16.11 and 19.49 GB and generated at 17.5 and 11.9 tokens/s. Qwen3-14B through Ollama used 14.26 GB and 78.5 tokens/s, although cross-backend timing should be interpreted cautiously. The Qwen and GLM adapters occupy approximately 627 and 746 MB. The low 2.63 s GLM QLoRA mean latency reflects abnormally short or empty outputs and is not an efficiency advantage. All configurations fit the target 24 GB workstation, but original Qwen provides the lowest resource cost and Qwen QLoRA the strongest QA quality within the limit.

Automated failure flags explain several aggregate differences. At top three, approved hybrid retrieval missed the held-out item for 32.5 per cent of Chinese and 33.3 per cent of English queries. Even with a hit, original Qwen produced low-overlap answers in 18.0 per cent of Chinese versus 4.3 per cent of English cases. Qwen QLoRA reduced these rates to 4.0 and 1.5 per cent, but omitted citation labels in 100.0 and 99.8 per cent. The translation analysis additionally exposed empty answers, wrong-language outputs and terminology substitutions. Similar-version retrieval and citation entailment require expert semantic judgement and were not inferred from string-based flags.

## 5. Discussion

The results support three conclusions. First, lexical and semantic retrieval are language-dependent complements: dense retrieval contributes most to Chinese recall, while exact terminology makes BM25 particularly competitive in English. Hybrid fusion improves final recall, but the additional latency means BM25 remains a credible low-cost configuration. Second, QLoRA and RAG are complementary for answer overlap but not automatically for traceability. Stronger parametric adaptation reduces the marginal F1 gain from retrieval and can overwrite citation-format behaviour. Citation-aware targets or constrained post-generation citation are therefore required before deployment. Third, adaptation quality is task-specific. QA gains coexist with asymmetric Qwen translation regressions and catastrophic GLM translation failure, so no single pooled bilingual score can justify deployment.

The approved-only and unfiltered hybrid conditions were identical because every admissible record in the frozen production index was already approved. This validates enforcement of the governance boundary but does not estimate the causal quality benefit of review. Such an estimate would require a deliberately retained, ethically usable unreviewed comparison corpus.

## 6. Practical implications

The architecture can support institutions that need local control of teaching resources and cannot rely on an external hosted model. PostgreSQL review status, source metadata and held-out flags make governance rules executable at retrieval time. On a 24 GB workstation, original Qwen is the practical low-cost option, Qwen QLoRA is preferred when answer overlap is the priority, and a citation-compliant original model should remain in the workflow when source display is mandatory. Teachers should inspect evidence and citations rather than treating either F1 or citation presence as correctness. International-student interfaces should retain both source language and translated fields because retrieval and translation errors are direction-specific.

## 7. Limitations

The held-out set covers four source documents and 17 task types, but it remains specific to one railway vocational knowledge base and cannot represent every course, regulation version or learner need. The database approval field is an operational governance control; the present experiment does not establish inter-rater agreement or claim that every approved answer reaches expert consensus. Because the frozen index contains only approved admissible records, the experiment cannot quantify an approved-versus-unreviewed quality effect. Automatic F1 penalises valid paraphrases, while citation presence and the hallucination proxy do not establish that a citation entails the generated claim. The error analysis is rule-based and no new human scoring was conducted. General-benchmark checks were capped per subtask. The experiment uses one workstation and results should not be generalised to production concurrency without load testing.

## 8. Conclusion

This study demonstrates a bilingual railway education Web information system that combines governed PostgreSQL records, pgvector/BGE-M3 retrieval and local generation. Hybrid retrieval achieved the strongest final recall in both languages, while Qwen2.5 QLoRA produced the strongest held-out QA and RAG answer-overlap scores within a 24 GB GPU limit. The same adaptation weakened citation compliance and did not consistently improve translation; GLM QLoRA failed substantially on sentence translation. The principal design implication is therefore not to choose between adaptation and RAG, but to govern them as separate, testable components with language-, task- and traceability-specific acceptance criteria.

## Declarations

**Data availability:** Evaluation scripts, experiment configurations, aggregate metrics and non-copyrighted derived test identifiers will be made available with the article. Full source passages from textbooks and regulations cannot be redistributed where third-party copyright or licensing restrictions apply; access to those materials is subject to the original rights holders.

**Ethics and consent:** No learner personal data are used in the current experiments.

**Conflict of interest:** The authors declare no conflict of interest.

**Use of generative AI:** Generative AI tools were used to assist code development, result organisation and language editing. The authors designed the study, executed the experiments, checked the source data and numerical results, and take responsibility for the final manuscript. This statement will be reconciled with Emerald's policy in force at the time of submission.

## References

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D. and Liu, Z. (2024), “BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation”, arXiv:2402.03216.

Cormack, G.V., Clarke, C.L.A. and Buettcher, S. (2009), “Reciprocal rank fusion outperforms Condorcet and individual rank learning methods”, *Proceedings of the 32nd International ACM SIGIR Conference*, pp. 758-759.

Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L. (2023), “QLoRA: Efficient finetuning of quantized LLMs”, *Advances in Neural Information Processing Systems*, Vol. 36.

Kasneci, E., Sessler, K., Küchemann, S., Bannert, M., Dementieva, D., Fischer, F., Gasser, U., Groh, G., Günnemann, S., Hüllermeier, E., Krusche, S., Kutyniok, G., Michaeli, T., Nerdel, C., Pfeffer, J., Poquet, O., Sailer, M., Schmidt, A., Seidel, T., Stadler, M., Weller, J., Kuhn, J. and Kasneci, G. (2023), “ChatGPT for good? On opportunities and challenges of large language models for education”, *Learning and Individual Differences*, Vol. 103, 102274.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S. and Kiela, D. (2020), “Retrieval-augmented generation for knowledge-intensive NLP tasks”, *Advances in Neural Information Processing Systems*, Vol. 33, pp. 9459-9474.

Robertson, S. and Zaragoza, H. (2009), “The probabilistic relevance framework: BM25 and beyond”, *Foundations and Trends in Information Retrieval*, Vol. 3 No. 4, pp. 333-389.

UNESCO (2023), *Guidance for Generative AI in Education and Research*, UNESCO, Paris.
