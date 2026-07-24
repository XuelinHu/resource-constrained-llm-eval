from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import yaml
from accelerate import init_empty_weights
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local QLoRA model assets and target modules.")
    parser.add_argument("--models", nargs="+", default=["qwen2_5_7b_instruct", "glm_4_9b_chat_hf"])
    parser.add_argument("--load-4bit", action="store_true", help="Also perform a real GPU load and attach LoRA.")
    args = parser.parse_args()
    configs = yaml.safe_load((ROOT / "configs/models/models.yaml").read_text(encoding="utf-8"))["models"]
    rows = []
    for key in args.models:
        model_cfg = configs[key]
        kwargs = {
            "cache_dir": model_cfg.get("cache_dir"),
            "local_files_only": True,
            "trust_remote_code": model_cfg.get("trust_remote_code", True),
        }
        kwargs = {name: value for name, value in kwargs.items() if value is not None}
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"], **kwargs)
        config = AutoConfig.from_pretrained(model_cfg["hf_id"], **kwargs)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=kwargs["trust_remote_code"])
        linear_modules = {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if module.__class__.__name__ == "Linear"
        }
        targets = set(model_cfg["qlora_target_modules"])
        missing = sorted(targets - linear_modules)
        trainable_parameters = None
        if args.load_4bit and not missing:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            loaded = AutoModelForCausalLM.from_pretrained(
                model_cfg["hf_id"], quantization_config=quantization, device_map="auto", **kwargs
            )
            loaded = prepare_model_for_kbit_training(loaded)
            loaded = get_peft_model(
                loaded,
                LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=sorted(targets),
                ),
            )
            trainable_parameters = sum(parameter.numel() for parameter in loaded.parameters() if parameter.requires_grad)
            del loaded
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(
            {
                "model": key,
                "hf_id": model_cfg["hf_id"],
                "architecture": type(config).__name__,
                "layers": getattr(config, "num_hidden_layers", None),
                "vocabulary_size": len(tokenizer),
                "target_modules": sorted(targets),
                "missing_target_modules": missing,
                "four_bit_lora_trainable_parameters": trainable_parameters,
                "compatible": not missing,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if any(not row["compatible"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
