import unittest

from datasets import Dataset

from src.rc_llm_eval.pipelines.qlora import _tokenize_completion_dataset


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, *, add_special_tokens, truncation, max_length):
        ids = ([1] if add_special_tokens else []) + [ord(char) for char in text]
        return {"input_ids": ids[:max_length]}


class CompletionTokenizationTests(unittest.TestCase):
    def test_prompt_tokens_are_masked_and_answer_tokens_are_supervised(self):
        dataset = Dataset.from_list([{"prompt_text": "Question: Q\nAnswer:", "answer": "A", "extra": "x"}])
        encoded = _tokenize_completion_dataset(dataset, FakeTokenizer(), "prompt_text", "answer", 128)[0]
        supervised = [label for label in encoded["labels"] if label != -100]

        self.assertGreater(len(supervised), 0)
        self.assertEqual(encoded["input_ids"][-len(supervised) :], supervised)
        self.assertEqual(encoded["labels"].count(-100), len("Question: Q\nAnswer:") + 1)

    def test_prompt_that_fills_context_has_no_supervised_tokens(self):
        dataset = Dataset.from_list([{"prompt_text": "long prompt", "answer": "answer"}])
        encoded = _tokenize_completion_dataset(dataset, FakeTokenizer(), "prompt_text", "answer", 4)[0]

        self.assertEqual(encoded["labels"], [-100] * 4)


if __name__ == "__main__":
    unittest.main()
