"""Export knowledge-base and model configuration tables from frozen sources."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import case, func, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from annotation_system.backend.app.database import SessionLocal
from annotation_system.backend.app.models import CorpusItem
from rc_llm_eval.utils.config import load_all_configs


TABLES = ROOT / "paper/ijwis/tables"


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.tex").write_text(frame.to_latex(index=False), encoding="utf-8")


def knowledge_table() -> pd.DataFrame:
    with SessionLocal() as db:
        rows = db.execute(
            select(
                CorpusItem.source_type,
                func.count().label("records"),
                func.sum(case((CorpusItem.review_status == "approved", 1), else_=0)).label("approved"),
                func.sum(
                    case(
                        (
                            (CorpusItem.question != "") & (CorpusItem.answer != "")
                            & (CorpusItem.question_en != "") & (CorpusItem.answer_en != ""),
                            1,
                        ),
                        else_=0,
                    )
                ).label("complete_bilingual"),
                func.count(func.distinct(CorpusItem.task_type)).label("task_types"),
            ).group_by(CorpusItem.source_type).order_by(func.count().desc())
        ).all()
    return pd.DataFrame(rows, columns=["source_type", "records", "approved", "complete_bilingual", "task_types"])


def model_table() -> pd.DataFrame:
    configs = load_all_configs("configs/experiments/ijwis_single_gpu_3090.yaml")
    qlora = configs["experiment"]["qlora"]
    rows = []
    for key in ("qwen2_5_7b_instruct", "glm_4_9b_chat_hf"):
        model = configs["models"][key]
        parameter_path = ROOT / "results/ijwis_single_gpu_3090/qlora" / key / "parameter_metrics.json"
        parameters = json.loads(parameter_path.read_text()) if parameter_path.exists() else {}
        rows.append(
            {
                "model": key,
                "checkpoint": model["hf_id"],
                "parameters_b": model["params_b"],
                "inference_quantization": "NF4 4-bit",
                "lora_rank": qlora["lora_r"],
                "lora_alpha": qlora["lora_alpha"],
                "lora_dropout": qlora["lora_dropout"],
                "target_modules": ", ".join(model["qlora_target_modules"]),
                "trainable_parameters": parameters.get("trainable_parameters"),
                "epochs": qlora["num_train_epochs"],
                "learning_rate": qlora["learning_rate"],
                "effective_batch_size": qlora["per_device_train_batch_size"] * qlora["gradient_accumulation_steps"],
                "max_sequence_length": qlora["max_seq_length"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    write(knowledge_table(), "table1_knowledge_base")
    write(model_table(), "table2_model_configuration")
    print(f"wrote={TABLES / 'table1_knowledge_base.csv'}")
    print(f"wrote={TABLES / 'table2_model_configuration.csv'}")


if __name__ == "__main__":
    main()
