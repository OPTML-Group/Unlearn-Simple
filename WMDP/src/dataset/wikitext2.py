import random
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .Base import BaseDataset, UnlearnDataset


class wikitext(BaseDataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        dataset = defaultdict()
        train_dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="train"
        )
        dataset["train"] = train_dataset
        print(f"Train dataset: {len(dataset['train'])}")
        dataset["test"] = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="test"
        )

        return dataset

    def __preprocess__(self, tokenizer):
        def preprocess(examples):
            results = {"input_ids": [], "attention_mask": [], "label": []}

            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
                max_length=512,
            )
            
            results["input_ids"] = tokenized.input_ids
            results["attention_mask"] = tokenized.attention_mask
            results["label"] = tokenized.input_ids
            return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=["text"]
        )
        test_dataset = self.dataset["test"].map(
            preprocess, batched=True, remove_columns=["text"]
        )

        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        return train_dataset, test_dataset