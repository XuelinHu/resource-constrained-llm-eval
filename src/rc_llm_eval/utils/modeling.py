"""模型加载与量化配置工具。"""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """将配置文件中的字符串精度映射为 PyTorch dtype。"""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise KeyError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def build_quantization_config(mode: str, dtype_name: str) -> BitsAndBytesConfig | None:
    """根据精度模式构建 bitsandbytes 配置。

    `bf16` 直接返回 `None`，交给常规半精度路径处理。
    """
    compute_dtype = resolve_dtype(dtype_name)
    if mode == "bf16":
        return None
    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    raise KeyError(f"Unsupported quantization mode: {mode}")


def load_model_and_tokenizer(
    model_cfg: dict,
    quantization_mode: str,
    dtype_name: str,
    peft_path: str | None = None,
):
    """加载基础模型、分词器，并在需要时挂载 PEFT 适配器。"""
    quantization_config = build_quantization_config(quantization_mode, dtype_name)
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantization_config is None:
        model_kwargs["torch_dtype"] = resolve_dtype(dtype_name)
    else:
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        # 许多因果语言模型没有独立 pad token，这里回退到 eos token。
        tokenizer.pad_token = tokenizer.eos_token
    # 推理阶段使用左侧填充，便于不同长度样本共享末尾对齐。
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_id"], **model_kwargs)
    if peft_path:
        model = PeftModel.from_pretrained(model, peft_path)
    model.eval()
    return model, tokenizer


def get_inference_device(model) -> torch.device:
    """从模型参数中推断当前推理设备。"""
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters to infer device from.") from exc


def clear_cuda() -> None:
    """主动释放 CUDA 缓存，降低多轮实验之间的显存残留。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
