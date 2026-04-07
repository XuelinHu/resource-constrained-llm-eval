from __future__ import annotations

import argparse
import json

from .pipelines.baseline import run_eval, summarize_results
from .pipelines.qlora import run_qlora
from .pipelines.reporting import export_paper_tables
from .utils.config import load_all_configs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-GPU LLM evaluation workspace")
    subparsers = parser.add_subparsers(dest="command", required=True)

    print_plan = subparsers.add_parser("print-plan", help="Print the experiment plan")
    print_plan.add_argument(
        "--experiment",
        default="configs/experiments/single_gpu_3090.yaml",
        help="Path to experiment config",
    )

    run_eval_parser = subparsers.add_parser("run-eval", help="Run baseline lm-eval for a model")
    run_eval_parser.add_argument("--experiment", required=True, help="Path to experiment config")
    run_eval_parser.add_argument("--model", required=True, help="Model key from configs/models/models.yaml")
    run_eval_parser.add_argument(
        "--precision",
        default=None,
        choices=["bf16", "int8", "int4"],
        help="Optional precision override",
    )
    run_eval_parser.add_argument(
        "--peft-adapter",
        default=None,
        help="Optional PEFT adapter directory for post-QLoRA evaluation",
    )
    run_eval_parser.add_argument(
        "--output-group",
        default="baseline",
        help="Output subgroup under the experiment result root",
    )
    run_eval_parser.add_argument(
        "--label",
        default=None,
        help="Optional file label suffix to distinguish repeated evaluations",
    )

    run_qlora_parser = subparsers.add_parser("run-qlora", help="Run QLoRA for one model and dataset")
    run_qlora_parser.add_argument("--experiment", required=True, help="Path to experiment config")
    run_qlora_parser.add_argument("--model", required=True, help="Model key from configs/models/models.yaml")
    run_qlora_parser.add_argument("--dataset", required=True, help="Dataset key from configs/datasets/tasks.yaml")

    summarize_parser = subparsers.add_parser("summarize-results", help="Aggregate baseline result files")
    summarize_parser.add_argument("--experiment", required=True, help="Path to experiment config")
    summarize_parser.add_argument(
        "--output-group",
        default="baseline",
        help="Result subgroup under the experiment output root to aggregate",
    )

    export_parser = subparsers.add_parser("export-paper-tables", help="Generate paper-ready tables from results")
    export_parser.add_argument("--experiment", required=True, help="Path to experiment config")
    return parser


def cmd_print_plan(args: argparse.Namespace) -> None:
    configs = load_all_configs(args.experiment)
    payload = {
        "experiment": configs["experiment"]["experiment"],
        "baseline": configs["experiment"]["baseline"],
        "qlora": configs["experiment"]["qlora"],
        "models": list(configs["models"].keys()),
        "tasks": list(configs["tasks"].keys()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "print-plan":
        cmd_print_plan(args)
        return 0

    configs = load_all_configs(args.experiment)
    if args.command == "run-eval":
        return run_eval(
            configs,
            args.model,
            args.precision,
            peft_path=args.peft_adapter,
            output_group=args.output_group,
            label=args.label,
        )
    if args.command == "run-qlora":
        run_qlora(configs, args.model, args.dataset)
        return 0
    if args.command == "summarize-results":
        summarize_results(configs, output_group=args.output_group)
        return 0
    if args.command == "export-paper-tables":
        export_paper_tables(configs)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
