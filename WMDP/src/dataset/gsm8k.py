import csv
from collections import defaultdict
from datasets import load_dataset
from .Base import BaseDataset

import torch

class GSM8K(BaseDataset):
    def __init__(self, dataset_name, with_retain=False, if_llama=False, subset=None):
        super().__init__(dataset_name, with_retain, if_llama)
        self.subset = subset
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def _padding_fn(self, prompt, response, max_length, tokenizer):
        text = (
            self.question_start_token
            + prompt
            + self.question_end_token
            + self.answer_start_token
            + response
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
        )
        num_prompt_token = len(
            tokenizer.tokenize(
                self.question_start_token + prompt + self.question_end_token,
                add_special_tokens=True,
            )
        )
        pad_length = max_length - len(tokenized.input_ids)
        if pad_length < 0:
            return None
        pad_input_ids = tokenized.input_ids + [tokenizer.pad_token_id] * pad_length
        pad_attention_mask = tokenized.attention_mask + [0] * pad_length
        if len(tokenized.input_ids) == max_length:
            label = tokenized.input_ids
        else:
            label = (
                tokenized.input_ids
                + [tokenizer.eos_token_id]
                + [-100] * (pad_length - 1)
            )
        for i in range(num_prompt_token):
            label[i] = -100

        assert (
            len(pad_input_ids) == max_length
        ), f"input_id length mismatch: {len(pad_input_ids)} (expect: {max_length})"
        assert (
            len(pad_attention_mask) == max_length
        ), f"attention_mask length mismatch: {len(pad_attention_mask)} (expect: {max_length})"
        assert (
            len(label) == max_length
        ), f"label length mismatch: {len(label)} (expect: {max_length})"

        return {
            "input_ids": torch.tensor(pad_input_ids),
            "attention_mask": torch.tensor(pad_attention_mask),
            "label": torch.tensor(label),
            "refused_label": torch.tensor(pad_input_ids),
            "question_length": torch.tensor(pad_input_ids),
        }

    def get_dataset(self):
        if self.subset == "retain":
            train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["test"]
            train_dataset = train_dataset.add_column("dataset_name", ["wiki"] * len(train_dataset))

        else:
            train_dataset = load_dataset(
                "openai/gsm8k", "main", cache_dir="./.cache", split="train"
            )
        
        test_dataset = load_dataset(
            "openai/gsm8k", "main", cache_dir="./.cache", split="test"
        )

        dataset = defaultdict()
        dataset["train"] = train_dataset
        dataset["test"] = test_dataset
        return dataset

    def __preprocess__(self, tokenizer):
        refusal_answers = []
        with open(
            "files/data/polite_refusal_responses/polite_refusal_responses_copyright.csv",
            "r",
        ) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                refusal_answers.append(row[0])

        def preprocess(examples):
            results = {
                "input_ids": [],
                "attention_mask": [],
                "label": [],
                "refused_label": [],
                "question_length": [],
            }
            # print(examples)
            for question, answer in zip(examples["question"], examples["answer"]):
                padded = self._padding_fn(question, answer, 512, tokenizer)
                if padded:
                    for k, v in padded.items():
                        results[k].append(v)

            return results

        format_columns = [
            "input_ids",
            "attention_mask",
            "label",
            "refused_label",
            "question_length",
        ]

        for split in ["train", "test"]:
            self.dataset[split] = self.dataset[split].map(
                preprocess,
                batched=True,
                remove_columns=self.dataset[split].column_names,
                # load_from_cache_file=False,
            )

            self.dataset[split].set_format(
                type="torch",
                columns=[
                    "input_ids",
                    "attention_mask",
                    "label",
                    "refused_label",
                    "question_length",
                ],
            )

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)

        return self.dataset