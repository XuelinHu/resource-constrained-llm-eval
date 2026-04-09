"""TensorBoard 日志辅助工具。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from .system import ensure_directory

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:  # pragma: no cover - 依赖缺失时给出更明确错误
    raise RuntimeError(
        "TensorBoard support requires the 'tensorboard' package. "
        "Install it with `pip install tensorboard`."
    ) from exc


TENSORBOARD_ROOT = Path("/ds1/runs")


def _slugify(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "na"


def build_tensorboard_log_dir(
    project_name: str,
    model_name: str,
    dataset_name: str,
    batch_size: int,
    learning_rate: float,
    seed: int | None = None,
) -> Path:
    """构建本次实验独立的 TensorBoard run 目录。"""
    root_dir = ensure_directory(TENSORBOARD_ROOT / _slugify(project_name))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fragments = [
        _slugify(model_name),
        _slugify(dataset_name),
        f"bs{batch_size}",
        f"lr{learning_rate:g}",
    ]
    if seed is not None:
        fragments.append(f"seed{seed}")
    fragments.append(timestamp)
    run_name = "_".join(fragments)
    log_dir = root_dir / run_name
    suffix = 1
    while log_dir.exists():
        log_dir = root_dir / f"{run_name}_{suffix}"
        suffix += 1
    return ensure_directory(log_dir)


def build_hparams(
    *,
    project_name: str,
    model_key: str,
    dataset_key: str,
    exp_cfg: dict[str, Any],
    qlora_cfg: dict[str, Any],
    training_args,
) -> dict[str, Any]:
    """整理实验关键超参数，便于写入 TensorBoard。"""
    return {
        "project": project_name,
        "model": model_key,
        "dataset": dataset_key,
        "batch_size": qlora_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": qlora_cfg["gradient_accumulation_steps"],
        "learning_rate": qlora_cfg["learning_rate"],
        "epochs": qlora_cfg["num_train_epochs"],
        "optimizer": str(training_args.optim),
        "scheduler": str(training_args.lr_scheduler_type),
        "seed": exp_cfg.get("seed"),
        "device": exp_cfg.get("device"),
        "max_seq_length": qlora_cfg["max_seq_length"],
        "warmup_ratio": qlora_cfg["warmup_ratio"],
        "lora_r": qlora_cfg["lora_r"],
        "lora_alpha": qlora_cfg["lora_alpha"],
        "lora_dropout": qlora_cfg["lora_dropout"],
    }


class TensorBoardLoggerCallback(TrainerCallback):
    """把 Trainer 日志按项目规范映射到 TensorBoard。"""

    def __init__(self, writer: SummaryWriter, model: torch.nn.Module, hparams: dict[str, Any]) -> None:
        self.writer = writer
        self.model = model
        self.hparams = hparams
        self._epoch_step_losses: list[float] = []
        self._histogram_names: list[str] = []

    def _epoch_index(self, state) -> int:
        epoch = state.epoch or 0.0
        return max(1, int(round(epoch)))

    def _write_histograms(self, epoch: int) -> None:
        if not self._histogram_names:
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self._histogram_names.append(name)
                if len(self._histogram_names) >= 4:
                    break
        for name, parameter in self.model.named_parameters():
            if name not in self._histogram_names or parameter.numel() > 1_000_000:
                continue
            values = parameter.detach().float().cpu()
            self.writer.add_histogram(f"params/{name}", values, epoch)

    def on_train_begin(self, args, state, control, **kwargs):
        self.writer.add_text("run/log_dir", self.writer.log_dir, 0)
        self.writer.add_text("run/hparams", json.dumps(self.hparams, indent=2, ensure_ascii=False), 0)
        self.writer.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs and "eval_loss" not in logs:
            loss = float(logs["loss"])
            self.writer.add_scalar("train/loss_step", loss, state.global_step)
            self._epoch_step_losses.append(loss)
        if "learning_rate" in logs:
            self.writer.add_scalar("train/lr", float(logs["learning_rate"]), state.global_step)
        self.writer.flush()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_step_losses:
            epoch = self._epoch_index(state)
            epoch_loss = sum(self._epoch_step_losses) / len(self._epoch_step_losses)
            self.writer.add_scalar("train/loss_epoch", epoch_loss, epoch)
            self._epoch_step_losses.clear()
            self._write_histograms(epoch)
            self.writer.flush()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        epoch = self._epoch_index(state)
        metric_mapping = {
            "eval_loss": "val/loss",
            "eval_accuracy": "val/acc",
            "eval_acc": "val/acc",
            "eval_precision": "val/precision",
            "eval_recall": "val/recall",
            "eval_f1": "val/f1",
            "eval_auc": "val/auc",
            "eval_map": "val/mAP",
            "eval_miou": "val/miou",
            "eval_dice": "val/dice",
        }
        for source_key, target_key in metric_mapping.items():
            if source_key in metrics:
                self.writer.add_scalar(target_key, float(metrics[source_key]), epoch)
        self.writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.flush()


def finalize_hparams(writer: SummaryWriter, hparams: dict[str, Any], metrics: dict[str, Any]) -> None:
    """在 TensorBoard 中记录最终实验超参数与结果摘要。"""
    tracked_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and key in {"train_loss", "eval_loss", "eval_accuracy", "eval_acc"}
    }
    if tracked_metrics:
        writer.add_hparams(hparams, tracked_metrics, run_name="hparams")
    writer.flush()
