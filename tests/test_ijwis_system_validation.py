from __future__ import annotations

import unittest

from annotation_system.backend.app.schemas import RagAskRequest
from scripts.evaluate_rag_faithfulness import claims, evidence_text


class IjwisSystemValidationTests(unittest.TestCase):
    def test_multiple_evidence_labels_are_attached_to_the_claim(self) -> None:
        parsed = claims(
            "Catenary suspension has four anchoring forms [Evidence1, Evidence3]. "
            "吊弦包括四种类型[证据2、3]。"
        )

        self.assertEqual(parsed[0]["citation_indices"], [1, 3])
        self.assertEqual(parsed[1]["citation_indices"], [2, 3])
        self.assertNotIn("Evidence", parsed[0]["claim"])

    def test_evidence_selection_and_audio_default(self) -> None:
        sources = [{"evidence": "first"}, {"evidence": "second"}, {"evidence": "third"}]

        self.assertEqual(evidence_text(sources, [1, 3]), "first\nthird")
        self.assertTrue(RagAskRequest(question="valid question").synthesize_audio)
        self.assertFalse(
            RagAskRequest(question="valid question", synthesize_audio=False).synthesize_audio
        )


if __name__ == "__main__":
    unittest.main()
