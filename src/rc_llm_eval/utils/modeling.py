from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def resolve_dtype(dtype_name: str) -> torch.dtype:
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
):
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
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_id"], **model_kwargs)
    model.eval()
    return model, tokenizer


def get_inference_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters to infer device from.") from exc


def clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
