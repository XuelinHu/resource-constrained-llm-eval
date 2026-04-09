"""基线评测、领域问答评测与效率测试流水线。"""

from __future__ import annotations

from datetime import datetime
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import torch

from ..utils.io import read_json, read_jsonl, write_csv
from ..utils.modeling import clear_cuda, get_inference_device, load_model_and_tokenizer
from ..utils.notifications import build_markdown_message, build_run_name, format_duration, send_dingtalk_notification
from ..utils.system import ensure_directory, write_json
from ..utils.text import normalize_answer


def _build_eval_success_message(
    *,
    project_name: str,
    run_name: str,
    model_key: str,
    output_group: str,
    precision: str,
    batch_size: str | int,
    start_time: datetime,
    end_time: datetime,
    output_dir: str,
    peft_path: str | None,
    summary_rows: list[dict[str, Any]],
    efficiency_row: dict[str, Any],
) -> str:
    core_metrics = [
        f"{row.get('task')}={row.get('score')}"
        for row in summary_rows
        if row.get("score") is not None
    ]
    core_metric_text = ", ".join(core_metrics[:5]) if core_metrics else "N/A"
    return build_markdown_message(
        "实验完成",
        [
            ("项目名", project_name),
            ("实验名", run_name),
            ("模型名", model_key),
            ("数据集名", output_group),
            ("precision", precision),
            ("batch size", batch_size),
            ("peft_path", peft_path),
            ("开始时间", start_time.strftime("%Y-%m-%d %H:%M:%S")),
            ("结束时间", end_time.strftime("%Y-%m-%d %H:%M:%S")),
            ("总耗时", format_duration((end_time - start_time).total_seconds())),
            ("核心指标", core_metric_text),
            ("mean_latency_s", efficiency_row.get("mean_latency_s")),
            ("mean_tokens_per_second", efficiency_row.get("mean_tokens_per_second")),
            ("TensorBoard", "N/A"),
            ("输出目录", output_dir),
            ("说明", "评测已完成"),
        ]
    )


def _build_eval_failure_message(
    *,
    project_name: str,
    run_name: str,
    model_key: str,
    output_group: str,
    precision: str,
    batch_size: str | int,
    start_time: datetime,
    failure_time: datetime,
    output_dir: str,
    peft_path: str | None,
    exc: Exception,
) -> str:
    return build_markdown_message(
        "实验失败",
        [
            ("项目名", project_name),
            ("实验名", run_name),
            ("模型名", model_key),
            ("数据集名", output_group),
            ("precision", precision),
            ("batch size", batch_size),
            ("peft_path", peft_path),
            ("开始时间", start_time.strftime("%Y-%m-%d %H:%M:%S")),
            ("失败时间", failure_time.strftime("%Y-%m-%d %H:%M:%S")),
            ("已运行时长", format_duration((failure_time - start_time).total_seconds())),
            ("错误摘要", f"{type(exc).__name__}: {exc}"),
            ("TensorBoard", "N/A"),
            ("输出目录", output_dir),
            ("说明", "评测执行失败"),
        ]
    )


def build_model_args(model_cfg: dict, precision: str, peft_path: str | None = None) -> str:
    """拼接 lm-eval 使用的 Hugging Face 模型参数字符串。"""
    args = [
        f"pretrained={model_cfg['hf_id']}",
        "trust_remote_code=True",
    ]
    if precision == "int8":
        args.append("load_in_8bit=True")
    elif precision == "int4":
        args.append("load_in_4bit=True")
        args.append("bnb_4bit_quant_type=nf4")
    else:
        args.append(f"dtype={model_cfg.get('default_dtype', 'bfloat16')}")
    if peft_path:
        args.append(f"peft={peft_path}")
    return ",".join(args)


def build_lm_eval_command(
    model_cfg: dict,
    task_names: list[str],
    num_fewshot: int,
    output_path: Path,
    precision: str,
    batch_size: str | int,
    peft_path: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """构造 lm-eval CLI 命令，但当前仓库更常用 Python API 直调。"""
    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        build_model_args(model_cfg, precision, peft_path),
        "--tasks",
        ",".join(task_names),
        "--num_fewshot",
        str(num_fewshot),
        "--device",
        "cuda:0",
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def _extract_text_from_outputs(tokenizer, generated_ids, input_length: int) -> str:
    """只截取新生成的 token，避免把原始 prompt 解码回输出。"""
    new_tokens = generated_ids[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _safe_memory_stats() -> tuple[float, float]:
    """读取 CUDA 峰值显存；无 CUDA 时返回零值。"""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated() / (1024**3)
    reserved = torch.cuda.max_memory_reserved() / (1024**3)
    return allocated, reserved


def run_efficiency_benchmark(
    configs: dict,
    model_key: str,
    precision: str,
    output_dir: Path,
    file_stem: str,
    peft_path: str | None = None,
) -> dict:
    """执行吞吐、时延和峰值显存的效率基准测试。"""
    baseline_cfg = configs["experiment"]["baseline"]
    model_cfg = configs["models"][model_key]
    prompt_file = configs["root"] / baseline_cfg["efficiency_prompt_file"]
    prompts = read_jsonl(prompt_file)[: baseline_cfg["efficiency_num_samples"]]

    clear_cuda()
    model, tokenizer = load_model_and_tokenizer(
        model_cfg=model_cfg,
        quantization_mode=precision,
        dtype_name=model_cfg.get("default_dtype", "bfloat16"),
        peft_path=peft_path,
    )
    device = get_inference_device(model)

    latencies: list[float] = []
    throughputs: list[float] = []
    outputs: list[dict] = []

    generation_kwargs = {
        "max_new_tokens": baseline_cfg["max_new_tokens"],
        "do_sample": baseline_cfg["do_sample"],
        "temperature": baseline_cfg["temperature"],
        "top_p": baseline_cfg["top_p"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if not baseline_cfg["do_sample"]:
        generation_kwargs.pop("temperature", None)
        generation_kwargs.pop("top_p", None)

    # 预热阶段让 CUDA 图和内核缓存先稳定下来，减少首样本抖动。
    warmup_count = min(baseline_cfg["warmup_prompts"], len(prompts))
    for record in prompts[:warmup_count]:
        encoded = tokenizer(record["prompt"], return_tensors="pt").to(device)
        with torch.inference_mode():
            _ = model.generate(**encoded, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for record in prompts:
        encoded = tokenizer(record["prompt"], return_tensors="pt").to(device)
        prompt_length = int(encoded["input_ids"].shape[1])
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        if torch.cuda.is_available():
            # 在计时结束前同步，确保 latency 覆盖真实 GPU 执行时间。
            torch.cuda.synchronize()
        latency = time.perf_counter() - start
        answer = _extract_text_from_outputs(tokenizer, generated, prompt_length)
        new_token_count = max(int(generated.shape[1] - prompt_length), 1)
        tokens_per_second = new_token_count / max(latency, 1e-6)

        latencies.append(latency)
        throughputs.append(tokens_per_second)
        outputs.append(
            {
                "prompt": record["prompt"],
                "output": answer,
                "latency_s": round(latency, 6),
                "new_tokens": new_token_count,
                "tokens_per_second": round(tokens_per_second, 4),
            }
        )

    peak_allocated_gb, peak_reserved_gb = _safe_memory_stats()
    payload = {
        "model": model_key,
        "precision": precision,
        "num_prompts": len(outputs),
        "mean_latency_s": round(statistics.mean(latencies), 6) if latencies else 0.0,
        "median_latency_s": round(statistics.median(latencies), 6) if latencies else 0.0,
        "mean_tokens_per_second": round(statistics.mean(throughputs), 4) if throughputs else 0.0,
        "peak_memory_allocated_gb": round(peak_allocated_gb, 4),
        "peak_memory_reserved_gb": round(peak_reserved_gb, 4),
    }
    write_json(output_dir / f"{file_stem}_efficiency.json", payload)
    write_json(output_dir / f"{file_stem}_efficiency_generations.json", {"samples": outputs})

    del model
    clear_cuda()
    return payload


def run_local_domain_eval(
    configs: dict,
    model_key: str,
    precision: str,
    output_dir: Path,
    file_stem: str,
    peft_path: str | None = None,
) -> dict:
    """在本地域问答集上执行精确匹配评估。"""
    dataset_cfg = configs["tasks"]["domain_qa"]
    model_cfg = configs["models"][model_key]
    records = read_jsonl(configs["root"] / dataset_cfg["test_file"])

    clear_cuda()
    model, tokenizer = load_model_and_tokenizer(
        model_cfg=model_cfg,
        quantization_mode=precision,
        dtype_name=model_cfg.get("default_dtype", "bfloat16"),
        peft_path=peft_path,
    )
    device = get_inference_device(model)

    prompt_field = dataset_cfg["prompt_field"]
    answer_field = dataset_cfg["answer_field"]
    generations: list[dict] = []
    hits = 0

    generation_kwargs = {
        "max_new_tokens": 128,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for row in records:
        encoded = tokenizer(row[prompt_field], return_tensors="pt").to(device)
        prompt_length = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        prediction = _extract_text_from_outputs(tokenizer, generated, prompt_length)
        ref = row[answer_field]
        is_correct = normalize_answer(prediction) == normalize_answer(ref)
        hits += int(is_correct)
        generations.append(
            {
                "prompt": row[prompt_field],
                "reference": ref,
                "prediction": prediction,
                "exact_match": is_correct,
            }
        )

    score = hits / len(records) if records else 0.0
    payload = {
        "model": model_key,
        "precision": precision,
        "task": "domain_qa",
        "metric": "exact_match",
        "score": round(score, 6),
        "num_examples": len(records),
    }
    write_json(output_dir / f"{file_stem}_domain_qa.json", payload)
    write_json(output_dir / f"{file_stem}_domain_generations.json", {"samples": generations})

    del model
    clear_cuda()
    return payload


def parse_lm_eval_metrics(path: Path, model_key: str, precision: str) -> list[dict]:
    """从 lm-eval 原始 JSON 中抽取统一格式的任务指标。"""
    payload = read_json(path)
    results = payload.get("results", {})
    rows: list[dict] = []
    for task_name, metrics in results.items():
        metric_name = None
        metric_value = None
        for key in ("acc,none", "exact_match,strict-match", "pass@1,create_test", "acc_norm,none"):
            if key in metrics:
                metric_name = key
                metric_value = metrics[key]
                break
        if metric_name is None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_name = key
                    metric_value = value
                    break
        rows.append(
            {
                "model": model_key,
                "precision": precision,
                "task": task_name,
                "metric": metric_name or "unknown",
                "score": metric_value,
            }
        )
    return rows


def resolve_lm_eval_result_path(expected_path: Path) -> Path | None:
    """兼容 lm-eval 自动追加时间戳时的结果文件命名差异。"""
    if expected_path.exists():
        return expected_path
    pattern = f"{expected_path.stem}_*.json"
    candidates = sorted(expected_path.parent.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_lm_eval(
    configs: dict,
    model_key: str,
    precision: str,
    output_path: Path,
    peft_path: str | None = None,
) -> int:
    """通过 lm-eval Python API 执行标准任务集评测。"""
    baseline_cfg = configs["experiment"]["baseline"]
    exp_cfg = configs["experiment"]["experiment"]
    model_cfg = configs["models"][model_key]
    tasks = configs["tasks"]
    task_names = [tasks[task]["task_name"] for task in baseline_cfg["tasks"] if tasks[task]["suite"] == "lm_eval"]

    model = None
    lm = None
    clear_cuda()

    gen_kwargs = {
        "max_gen_toks": baseline_cfg["max_new_tokens"],
        "do_sample": baseline_cfg["do_sample"],
    }
    if baseline_cfg["do_sample"]:
        gen_kwargs["temperature"] = baseline_cfg["temperature"]
        gen_kwargs["top_p"] = baseline_cfg["top_p"]

    # 某些代码生成类任务需要显式允许执行评测代码。
    previous_code_eval = os.environ.get("HF_ALLOW_CODE_EVAL")
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_cfg=model_cfg,
            quantization_mode=precision,
            dtype_name=model_cfg.get("default_dtype", "bfloat16"),
            peft_path=peft_path,
        )

        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            dtype=model_cfg.get("default_dtype", "bfloat16"),
            batch_size=baseline_cfg["batch_size"],
            device=exp_cfg["device"],
        )

        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task_names,
            num_fewshot=baseline_cfg["num_fewshot"],
            batch_size=baseline_cfg["batch_size"],
            device=exp_cfg["device"],
            limit=baseline_cfg.get("lm_eval_limit"),
            log_samples=False,
            gen_kwargs=gen_kwargs,
            confirm_run_unsafe_code=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return 0
    except Exception:
        return 1
    finally:
        if previous_code_eval is None:
            os.environ.pop("HF_ALLOW_CODE_EVAL", None)
        else:
            os.environ["HF_ALLOW_CODE_EVAL"] = previous_code_eval
        if lm is not None:
            del lm
        if model is not None:
            del model
        clear_cuda()


def run_eval(
    configs: dict,
    model_key: str,
    precision: str | None = None,
    peft_path: str | None = None,
    output_group: str = "baseline",
    label: str | None = None,
) -> int:
    """执行单个模型的完整评测流程并输出汇总文件。"""
    start_time = datetime.now()
    exp_cfg = configs["experiment"]["experiment"]
    baseline_cfg = configs["experiment"]["baseline"]
    models = configs["models"]
    tasks = configs["tasks"]

    if model_key not in models:
        raise KeyError(f"Unknown model key: {model_key}")

    model_cfg = models[model_key]
    precision = precision or baseline_cfg["precision"]
    run_name = build_run_name(
        prefix_parts=[model_key, output_group],
        batch_size=baseline_cfg["batch_size"],
        precision=precision,
        label=label,
        timestamp=start_time,
    )
    task_names = [tasks[task]["task_name"] for task in baseline_cfg["tasks"] if tasks[task]["suite"] == "lm_eval"]

    output_dir = ensure_directory(configs["root"] / exp_cfg["output_root"] / output_group / model_key)
    file_stem = f"{model_key}_{precision}" if not label else f"{model_key}_{precision}_{label}"
    try:
        lm_eval_output_path = output_dir / f"{file_stem}_lm_eval.json"
        write_json(
            output_dir / f"{file_stem}_plan.json",
            {
                "run_name": run_name,
                "model": model_key,
                "hf_id": model_cfg["hf_id"],
                "precision": precision,
                "tasks": task_names,
                "device": exp_cfg["device"],
                "batch_size": baseline_cfg["batch_size"],
                "output_group": output_group,
                "label": label,
                "peft_path": peft_path,
            },
        )

        exit_code = run_lm_eval(
            configs=configs,
            model_key=model_key,
            precision=precision,
            output_path=lm_eval_output_path,
            peft_path=peft_path,
        )

        summary_rows: list[dict] = []
        resolved_lm_eval_output_path = resolve_lm_eval_result_path(lm_eval_output_path)
        if resolved_lm_eval_output_path is not None:
            summary_rows.extend(parse_lm_eval_metrics(resolved_lm_eval_output_path, model_key, precision))

        # 无论 lm-eval 是否成功，都尝试补充本地域任务与效率指标，方便排查问题。
        domain_row = run_local_domain_eval(
            configs,
            model_key,
            precision,
            output_dir,
            file_stem=file_stem,
            peft_path=peft_path,
        )
        summary_rows.append(domain_row)

        efficiency_row = run_efficiency_benchmark(
            configs,
            model_key,
            precision,
            output_dir,
            file_stem=file_stem,
            peft_path=peft_path,
        )
        write_json(output_dir / f"{file_stem}_summary.json", {"metrics": summary_rows, "efficiency": efficiency_row})
        write_csv(output_dir / f"{file_stem}_summary.csv", summary_rows)
        send_dingtalk_notification(
            _build_eval_success_message(
                project_name=configs["root"].name,
                run_name=run_name,
                model_key=model_key,
                output_group=output_group,
                precision=precision,
                batch_size=baseline_cfg["batch_size"],
                start_time=start_time,
                end_time=datetime.now(),
                output_dir=str(output_dir),
                peft_path=peft_path,
                summary_rows=summary_rows,
                efficiency_row=efficiency_row,
            ),
            err=False,
        )
        return exit_code
    except Exception as exc:
        send_dingtalk_notification(
            _build_eval_failure_message(
                project_name=configs["root"].name,
                run_name=run_name,
                model_key=model_key,
                output_group=output_group,
                precision=precision,
                batch_size=baseline_cfg["batch_size"],
                start_time=start_time,
                failure_time=datetime.now(),
                output_dir=str(output_dir),
                peft_path=peft_path,
                exc=exc,
            ),
            err=True,
        )
        raise


def summarize_results(configs: dict, output_group: str = "baseline") -> None:
    """扫描汇总 JSON，生成聚合后的总表 CSV。"""
    baseline_dir = configs["root"] / configs["experiment"]["experiment"]["output_root"] / output_group
    metric_rows: list[dict] = []
    efficiency_rows: list[dict] = []

    for summary_file in baseline_dir.rglob("*_summary.json"):
        payload = read_json(summary_file)
        metric_rows.extend(payload.get("metrics", []))
        if "efficiency" in payload:
            efficiency_rows.append(payload["efficiency"])

    write_csv(baseline_dir / "all_metrics.csv", metric_rows)
    write_csv(baseline_dir / "all_efficiency.csv", efficiency_rows)
