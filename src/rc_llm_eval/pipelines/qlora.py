from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ..utils.modeling import resolve_dtype
from ..utils.system import ensure_directory, write_json


def _tokenize_dataset(dataset, tokenizer, text_field: str, max_length: int):
    def encode(batch: dict) -> dict:
        tokens = tokenizer(
            batch[text_field],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
        return tokens

    return dataset.map(encode, batched=True, remove_columns=dataset.column_names)


def run_qlora(configs: dict, model_key: str, dataset_key: str) -> None:
    exp_cfg = configs["experiment"]["experiment"]
    qlora_cfg = configs["experiment"]["qlora"]
    model_cfg = configs["models"][model_key]
    dataset_cfg = configs["tasks"][dataset_key]

    output_dir = ensure_directory(configs["root"] / exp_cfg["output_root"] / "qlora" / model_key)
    write_json(
        output_dir / "run_config.json",
        {
            "model": model_key,
            "hf_id": model_cfg["hf_id"],
            "dataset": dataset_key,
            "dataset_cfg": dataset_cfg,
            "qlora": qlora_cfg,
        },
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_id"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=resolve_dtype(model_cfg.get("default_dtype", "bfloat16")),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=qlora_cfg["lora_r"],
        lora_alpha=qlora_cfg["lora_alpha"],
        lora_dropout=qlora_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=qlora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(configs["root"] / dataset_cfg["train_file"]),
            "validation": str(configs["root"] / dataset_cfg["valid_file"]),
        },
    )
    train_dataset = _tokenize_dataset(
        dataset["train"],
        tokenizer,
        text_field=dataset_cfg["text_field"],
        max_length=qlora_cfg["max_seq_length"],
    )
    eval_dataset = _tokenize_dataset(
        dataset["validation"],
        tokenizer,
        text_field=dataset_cfg["text_field"],
        max_length=qlora_cfg["max_seq_length"],
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoint"),
        learning_rate=qlora_cfg["learning_rate"],
        num_train_epochs=qlora_cfg["num_train_epochs"],
        per_device_train_batch_size=qlora_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=qlora_cfg["gradient_accumulation_steps"],
        warmup_ratio=qlora_cfg["warmup_ratio"],
        logging_steps=qlora_cfg["logging_steps"],
        save_strategy=qlora_cfg["save_strategy"],
        eval_strategy=qlora_cfg["evaluation_strategy"],
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    train_result = trainer.train()
    write_json(output_dir / "train_metrics.json", train_result.metrics)
    eval_metrics = trainer.evaluate()
    write_json(output_dir / "eval_metrics.json", eval_metrics)
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
