"""实验结果汇总与论文表格导出流程。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _escape_latex(value: object) -> str:
    """转义 LaTeX 表格中最常见的特殊字符。"""
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _write_simple_latex_table(df: pd.DataFrame, output_path: Path, caption: str, label: str) -> None:
    """将 DataFrame 写成简单的 booktabs 风格 LaTeX 表格。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(df.columns)
    alignment = "l" + "c" * (len(columns) - 1)

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(_escape_latex(column) for column in columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_escape_latex(row[column]) for column in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_paper_tables(configs: dict) -> None:
    """从聚合 CSV 生成论文和结果目录下的衍生表格。"""
    baseline_dir = configs["root"] / configs["experiment"]["experiment"]["output_root"] / "baseline"
    qlora_eval_dir = configs["root"] / configs["experiment"]["experiment"]["output_root"] / "qlora_eval"
    paper_tables_dir = configs["root"] / "paper" / "tables"
    results_tables_dir = baseline_dir / "tables"
    results_tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = baseline_dir / "all_metrics.csv"
    efficiency_path = baseline_dir / "all_efficiency.csv"
    if not metrics_path.exists() or not efficiency_path.exists():
        raise FileNotFoundError(
            "Expected summary CSV files under results. Run summarize-results after baseline jobs first."
        )

    metrics_df = pd.read_csv(metrics_path)
    efficiency_df = pd.read_csv(efficiency_path)

    # 主结果表按模型为行、任务为列透视，便于直接送入论文。
    main_df = (
        metrics_df.pivot_table(index="model", columns="task", values="score", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    main_output_csv = results_tables_dir / "main_results_pivot.csv"
    main_output_tex = paper_tables_dir / "generated_main_results.tex"
    main_df.to_csv(main_output_csv, index=False)
    _write_simple_latex_table(
        main_df.fillna("--"),
        main_output_tex,
        "Automatically generated main benchmark table from aggregated results.",
        "tab:generated-main-results",
    )

    efficiency_keep = [
        "model",
        "precision",
        "peak_memory_allocated_gb",
        "peak_memory_reserved_gb",
        "mean_latency_s",
        "mean_tokens_per_second",
    ]
    efficiency_pivot = efficiency_df[efficiency_keep].sort_values(by=["precision", "model"])
    efficiency_output_csv = results_tables_dir / "efficiency_results.csv"
    efficiency_output_tex = paper_tables_dir / "generated_efficiency_results.tex"
    efficiency_pivot.to_csv(efficiency_output_csv, index=False)
    _write_simple_latex_table(
        efficiency_pivot.fillna("--"),
        efficiency_output_tex,
        "Automatically generated efficiency table from aggregated results.",
        "tab:generated-efficiency-results",
    )

    qlora_metrics_path = qlora_eval_dir / "all_metrics.csv"
    if qlora_metrics_path.exists():
        qlora_metrics_df = pd.read_csv(qlora_metrics_path)
        # 仅对领域问答任务做前后对比，突出微调收益。
        baseline_domain = metrics_df[metrics_df["task"] == "domain_qa"][["model", "score"]].rename(
            columns={"score": "baseline_domain_qa"}
        )
        adapted_domain = qlora_metrics_df[qlora_metrics_df["task"] == "domain_qa"][["model", "score"]].rename(
            columns={"score": "adapted_domain_qa"}
        )
        qlora_compare_df = baseline_domain.merge(adapted_domain, on="model", how="inner")
        if not qlora_compare_df.empty:
            qlora_compare_df["domain_qa_gain"] = (
                qlora_compare_df["adapted_domain_qa"] - qlora_compare_df["baseline_domain_qa"]
            ).round(6)
            qlora_output_csv = results_tables_dir / "qlora_results.csv"
            qlora_output_tex = paper_tables_dir / "generated_qlora_results.tex"
            qlora_compare_df.to_csv(qlora_output_csv, index=False)
            _write_simple_latex_table(
                qlora_compare_df.fillna("--"),
                qlora_output_tex,
                "Automatically generated before/after QLoRA comparison on the domain benchmark.",
                "tab:generated-qlora-results",
            )
