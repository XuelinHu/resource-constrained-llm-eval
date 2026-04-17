"""QLoRA 训练流水线。"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any

import numpy as np

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

from ..utils.notifications import build_markdown_message, build_run_name, format_duration, send_dingtalk_notification
from ..utils.modeling import resolve_dtype
from ..utils.system import ensure_directory, write_json
from ..utils.tb_logger import (
    TensorBoardLoggerCallback,
    build_hparams,
    build_tensorboard_log_dir,
    finalize_hparams,
)


def _build_success_message(
    *,
    project_name: str,
    run_name: str,
    model_key: str,
    dataset_key: str,
    qlora_cfg: dict[str, Any],
    seed: int | None,
    start_time: datetime,
    end_time: datetime,
    log_dir: str,
    adapter_dir: str,
    eval_metrics: dict[str, Any],
) -> str:
    core_metric_name = "eval_accuracy" if "eval_accuracy" in eval_metrics else "eval_loss"
    core_metric_value = eval_metrics.get(core_metric_name)
    return build_markdown_message(
        "实验完成",
        [
            ("项目名", project_name),
            ("实验名", run_name),
            ("模型名", model_key),
            ("数据集名", dataset_key),
            ("batch size", qlora_cfg["per_device_train_batch_size"]),
            ("learning rate", qlora_cfg["learning_rate"]),
            ("epochs", qlora_cfg["num_train_epochs"]),
            ("seed", seed),
            ("开始时间", start_time.strftime("%Y-%m-%d %H:%M:%S")),
            ("结束时间", end_time.strftime("%Y-%m-%d %H:%M:%S")),
            ("总耗时", format_duration((end_time - start_time).total_seconds())),
            ("核心指标", f"{core_metric_name}={core_metric_value}"),
            ("val/loss", eval_metrics.get("eval_loss")),
            ("val/acc", eval_metrics.get("eval_accuracy")),
            ("perplexity", eval_metrics.get("perplexity")),
            ("TensorBoard", log_dir),
            ("模型保存路径", adapter_dir),
            ("说明", "实验已完成"),
        ]
    )


def _build_failure_message(
    *,
    project_name: str,
    run_name: str,
    model_key: str,
    dataset_key: str,
    qlora_cfg: dict[str, Any],
    seed: int | None,
    start_time: datetime,
    failure_time: datetime,
    log_dir: str,
    exc: Exception,
) -> str:
    return build_markdown_message(
        "实验失败",
        [
            ("项目名", project_name),
            ("实验名", run_name),
            ("模型名", model_key),
            ("数据集名", dataset_key),
            ("batch size", qlora_cfg["per_device_train_batch_size"]),
            ("learning rate", qlora_cfg["learning_rate"]),
            ("epochs", qlora_cfg["num_train_epochs"]),
            ("seed", seed),
            ("开始时间", start_time.strftime("%Y-%m-%d %H:%M:%S")),
            ("失败时间", failure_time.strftime("%Y-%m-%d %H:%M:%S")),
            ("已运行时长", format_duration((failure_time - start_time).total_seconds())),
            ("错误摘要", f"{type(exc).__name__}: {exc}"),
            ("TensorBoard", log_dir),
            ("说明", "实验执行失败"),
        ]
    )


def _tokenize_dataset(dataset, tokenizer, text_field: str, max_length: int):
    """将原始文本数据集转成自回归训练所需的 token/label 结构。"""

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


def _preprocess_logits_for_metrics(logits, labels):
    """在评估阶段仅保留 argmax 结果，降低指标计算内存开销。"""
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def _compute_autoregressive_metrics(eval_preds):
    """计算忽略 padding 后的 token-level accuracy。"""
    predictions, labels = eval_preds
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    shifted_predictions = predictions[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    if not np.any(mask):
        return {"accuracy": 0.0}
    accuracy = float((shifted_predictions[mask] == shifted_labels[mask]).mean())
    return {"accuracy": accuracy}


def run_qlora(configs: dict, model_key: str, dataset_key: str) -> None:
    """执行单个模型在指定数据集上的 QLoRA 训练与评估。"""
    exp_cfg = configs["experiment"]["experiment"]
    qlora_cfg = configs["experiment"]["qlora"]
    model_cfg = configs["models"][model_key]
    dataset_cfg = configs["tasks"][dataset_key]
    start_time = datetime.now()
    run_name = build_run_name(
        prefix_parts=[model_key, dataset_key],
        batch_size=qlora_cfg["per_device_train_batch_size"],
        learning_rate=qlora_cfg["learning_rate"],
        seed=exp_cfg.get("seed"),
        timestamp=start_time,
    )

    output_dir = ensure_directory(configs["root"] / exp_cfg["output_root"] / "qlora" / model_key)
    project_name = configs["root"].name
    log_dir = build_tensorboard_log_dir(
        project_root=configs["root"],
        project_name=project_name,
        model_name=model_key,
        dataset_name=dataset_key,
        batch_size=qlora_cfg["per_device_train_batch_size"],
        learning_rate=qlora_cfg["learning_rate"],
        seed=exp_cfg.get("seed"),
    )
    print(f"TensorBoard log dir: {log_dir}")
    write_json(
        output_dir / "run_config.json",
        {
            "project": project_name,
            "run_name": run_name,
            "model": model_key,
            "hf_id": model_cfg["hf_id"],
            "dataset": dataset_key,
            "dataset_cfg": dataset_cfg,
            "qlora": qlora_cfg,
            "tensorboard_log_dir": str(log_dir),
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
        # 训练阶段同样保证 batch 内补齐行为稳定。
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
    # 4bit 训练前先做 k-bit 训练准备，再挂载 LoRA 适配器。
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
        logging_dir=str(log_dir),
        run_name=run_name,
        seed=exp_cfg.get("seed", 42),
        remove_unused_columns=False,
    )
    hparams = build_hparams(
        project_name=project_name,
        model_key=model_key,
        dataset_key=dataset_key,
        exp_cfg=exp_cfg,
        qlora_cfg=qlora_cfg,
        training_args=training_args,
    )
    writer = None
    adapter_dir = output_dir / "adapter"

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            compute_metrics=_compute_autoregressive_metrics,
            preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
            callbacks=[TensorBoardLoggerCallback(writer=writer, model=model, hparams=hparams)],
        )

        # 先训练再评估，并把关键指标和适配器权重都落盘。
        train_result = trainer.train()
        write_json(output_dir / "train_metrics.json", train_result.metrics)
        eval_metrics = trainer.evaluate()
        if "eval_loss" in eval_metrics:
            eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
        write_json(output_dir / "eval_metrics.json", eval_metrics)
        finalize_hparams(writer, hparams, {**train_result.metrics, **eval_metrics})
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        send_dingtalk_notification(
            _build_success_message(
                project_name=project_name,
                run_name=run_name,
                model_key=model_key,
                dataset_key=dataset_key,
                qlora_cfg=qlora_cfg,
                seed=exp_cfg.get("seed"),
                start_time=start_time,
                end_time=datetime.now(),
                log_dir=str(log_dir),
                adapter_dir=str(adapter_dir),
                eval_metrics=eval_metrics,
            ),
            err=False,
        )
    except Exception as exc:
        send_dingtalk_notification(
            _build_failure_message(
                project_name=project_name,
                run_name=run_name,
                model_key=model_key,
                dataset_key=dataset_key,
                qlora_cfg=qlora_cfg,
                seed=exp_cfg.get("seed"),
                start_time=start_time,
                failure_time=datetime.now(),
                log_dir=str(log_dir),
                exc=exc,
            ),
            err=True,
        )
        raise
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
