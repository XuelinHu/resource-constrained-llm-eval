import unittest
from collections import Counter
from types import SimpleNamespace

from scripts.build_approved_bilingual_qlora_dataset import pairwise_overlaps, stratified_split
from scripts.build_rag_test_set import allocate_targets, category


def item(index: int, split: str = ""):
    return SimpleNamespace(
        external_id=f"item-{index}",
        task_type="regulation_qa",
        source_document="regulation.docx",
        metadata_json={"split": split} if split else {},
    )


class BilingualDatasetSplitTests(unittest.TestCase):
    def test_split_is_grouped_approximately_80_10_10_and_preserves_fixed_test(self):
        items = [item(index, "test" if index < 5 else "") for index in range(100)]
        split_items = stratified_split(items, seed=42)

        self.assertEqual({name: len(rows) for name, rows in split_items.items()}, {"train": 80, "valid": 10, "test": 10})
        test_ids = {row.external_id for row in split_items["test"]}
        self.assertTrue({f"item-{index}" for index in range(5)}.issubset(test_ids))

        records = {
            name: [{"pair_id": row.external_id} for row in rows]
            for name, rows in split_items.items()
        }
        self.assertEqual(pairwise_overlaps(records), {"train_valid": 0, "train_test": 0, "valid_test": 0})

    def test_rag_target_shortfall_is_reallocated_without_exceeding_availability(self):
        available = {"terminology": 450, "regulation": 190, "textbook": 123}
        targets = allocate_targets(Counter(available), {"terminology": 100, "regulation": 150, "textbook": 150}, 400)

        self.assertEqual(targets, {"terminology": 127, "regulation": 150, "textbook": 123})
        self.assertEqual(category("terminology_pair"), "terminology")
        self.assertEqual(category("regulation_clause_qa"), "regulation")
        self.assertEqual(category("textbook_operation_qa"), "textbook")


if __name__ == "__main__":
    unittest.main()
