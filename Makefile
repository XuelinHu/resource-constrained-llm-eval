EXPERIMENT ?= configs/experiments/ijwis_single_gpu_3090.yaml
MODEL ?= qwen2_5_7b_instruct
DATASET ?= bilingual_approved_qa

.PHONY: env plan baseline baseline-all summarize export-paper-tables qlora qlora-all paper

env:
	bash scripts/setup_conda.sh

plan:
	python -m src.rc_llm_eval.cli print-plan --experiment $(EXPERIMENT)

baseline:
	python -m src.rc_llm_eval.cli run-eval --experiment $(EXPERIMENT) --model $(MODEL)

baseline-all:
	bash scripts/run_baseline_all.sh $(EXPERIMENT)

summarize:
	python -m src.rc_llm_eval.cli summarize-results --experiment $(EXPERIMENT)

export-paper-tables:
	python -m src.rc_llm_eval.cli export-paper-tables --experiment $(EXPERIMENT)

qlora:
	python -m src.rc_llm_eval.cli run-qlora --experiment $(EXPERIMENT) --model $(MODEL) --dataset $(DATASET)

qlora-all:
	bash scripts/run_qlora_all.sh $(EXPERIMENT) $(DATASET)

paper:
	conda run -n rc-llm-eval python scripts/export_ijwis_latex.py
	latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error IJWIS__Copy_/Main.tex
	mkdir -p output/pdf
	cp IJWIS__Copy_/Main.pdf output/pdf/ijwis_bilingual_railway_manuscript_latex.pdf
	conda run -n rc-llm-eval python scripts/prepare_ijwis_submission.py
