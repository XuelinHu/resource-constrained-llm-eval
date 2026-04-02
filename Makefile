EXPERIMENT ?= configs/experiments/single_gpu_3090.yaml
MODEL ?= qwen3_4b
DATASET ?= domain_qa

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
	cd paper && latexmk -pdf main.tex
