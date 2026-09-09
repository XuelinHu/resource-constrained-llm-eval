from __future__ import annotations

import unittest

from scripts import export_ijwis_latex as latex
from scripts import prepare_ijwis_submission as submission


class IjwisFigureExportTests(unittest.TestCase):
    def test_figure_one_matches_submission(self) -> None:
        figure = latex.FIGURES["### 3.3 Retrieval and answer generation"][0]
        self.assertEqual(figure, submission.FIGURES[0])
        self.assertEqual(figure[0], "figure_01_neural_retrieval.pdf")
        self.assertEqual(len(submission.FIGURES), 8)

    def test_revised_captions_match_submission(self) -> None:
        figures = [figure for group in latex.FIGURES.values() for figure in group]
        for index in (0, 3, 5, 6):
            self.assertEqual(figures[index], submission.FIGURES[index])

    def test_landscape_figure_precedes_governance_figure(self) -> None:
        prepared = latex.prepare_markdown(latex.SOURCE.read_text(encoding="utf-8"))
        latex.validate_citation_graph(prepared)
        result = latex.markdown_to_latex(prepared)
        self.assertEqual(result.count(r"\begin{landscape}"), 1)
        self.assertEqual(result.count(r"\end{landscape}"), 1)
        self.assertEqual(result.count(r"\afterpage{\clearpage"), 1)
        self.assertEqual(result.count(r"\begin{figure}"), 8)
        self.assertEqual(result.count("figure_01_neural_retrieval.pdf"), 1)
        self.assertNotIn("system_architecture.pdf", result)
        self.assertLess(
            result.index("figure_01_neural_retrieval.pdf"),
            result.index("figure_02_knowledge_governance.pdf"),
        )
        self.assertIn(r"\label{fig:neural-retrieval-qlora}", result)


if __name__ == "__main__":
    unittest.main()
