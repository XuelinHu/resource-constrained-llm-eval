from __future__ import annotations

import unittest

from src.rc_llm_eval.pipelines.baseline import _domain_sample_metrics, _translation_metric_rows


class TranslationMetricTests(unittest.TestCase):
    def test_directional_metrics_are_separate_and_reproducible(self) -> None:
        samples = [
            {
                "category": "terminology_zh_to_en",
                "prediction": "contact wire",
                "reference": "contact wire",
                "metrics": _domain_sample_metrics("contact wire", "contact wire"),
            },
            {
                "category": "terminology_en_to_zh",
                "prediction": "接触线",
                "reference": "接触线",
                "metrics": _domain_sample_metrics("接触线", "接触线"),
            },
        ]
        rows = _translation_metric_rows(
            model_key="test",
            precision="bf16",
            task_key="domain_qa",
            samples=samples,
        )
        keyed = {(row["task"], row["metric"]): row for row in rows}
        for direction in ("zh_to_en", "en_to_zh"):
            task = f"domain_qa:{direction}"
            self.assertAlmostEqual(keyed[(task, "corpus_bleu")]["score"], 100.0)
            self.assertAlmostEqual(keyed[(task, "chrf_pp")]["score"], 100.0)
            self.assertAlmostEqual(keyed[(task, "terminology_success_rate")]["score"], 1.0)
            self.assertIn("version:2.6.0", keyed[(task, "corpus_bleu")]["signature"])
            self.assertIn("nw:2", keyed[(task, "chrf_pp")]["signature"])


if __name__ == "__main__":
    unittest.main()
