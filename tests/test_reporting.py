from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.rc_llm_eval.pipelines.reporting import export_paper_tables


class ReportingTests(unittest.TestCase):
    def test_exports_directional_translation_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "results" / "baseline"
            baseline.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"model": "m", "precision": "int4", "task": "domain_qa:zh_to_en", "metric": "corpus_bleu", "score": 42.0},
                    {"model": "m", "precision": "int4", "task": "domain_qa:zh_to_en", "metric": "chrf_pp", "score": 55.0},
                    {"model": "m", "precision": "int4", "task": "domain_qa:en_to_zh", "metric": "terminology_success_rate", "score": 0.8},
                ]
            ).to_csv(baseline / "all_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "model": "m",
                        "precision": "int4",
                        "peak_memory_allocated_gb": 1.0,
                        "peak_memory_reserved_gb": 1.1,
                        "mean_latency_s": 0.2,
                        "mean_tokens_per_second": 10.0,
                    }
                ]
            ).to_csv(baseline / "all_efficiency.csv", index=False)
            configs = {
                "root": root,
                "experiment": {"experiment": {"output_root": "results"}},
            }
            export_paper_tables(configs)
            output = pd.read_csv(baseline / "tables" / "translation_results.csv")
            self.assertEqual(output.loc[0, "zh_to_en_corpus_bleu"], 42.0)
            self.assertEqual(output.loc[0, "zh_to_en_chrf_pp"], 55.0)
            self.assertEqual(output.loc[0, "en_to_zh_terminology_success_rate"], 0.8)
            self.assertTrue((root / "paper" / "tables" / "generated_translation_results.tex").is_file())


if __name__ == "__main__":
    unittest.main()
