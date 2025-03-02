# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)


def create_pku_dataloader_from_dataset(
    tokenizer, dataset, fraction=1.0, batch_size=4, dataset_seed=8888
):
    """
    Given the PKU dataset, create the dataloader on the unlearned harmful Q&A pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded PKU dataset.
        fraction: <1 will do downsampling.
        batch_size: Batch size.

    Returns:
        Data loader of PKU harmful Q&A pairs.
    """
    torch.manual_seed(dataset_seed)
    np.random.seed(dataset_seed)
    random.seed(dataset_seed)

    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["prompt"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            prompt = examples["prompt"][i]
            response_list = []

            # Add only bad samples.
            if not examples["is_response_0_safe"][i]:
                response_list.append(examples["response_0"][i])
            if not examples["is_response_1_safe"][i]:
                response_list.append(examples["response_1"][i])

            # Add all responses to results or skip if none.
            for response in response_list:
                text = f"### Question: {prompt}\n ### Answer: {response}"
                tokenized = tokenizer(text, truncation=True, padding="max_length")
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                test_text = f"### Question: {prompt}\n ### Answer: "
                test_tokenized = tokenizer(
                    test_text, truncation=True, padding="max_length"
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

        return results

    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.
    dataset = dataset.map(
        preproccess,
        batched=True,
        remove_columns=[
            "prompt",
            "response_0",
            "response_1",
            "is_response_0_safe",
            "is_response_1_safe",
            "better_response_id",
            "safer_response_id",
        ],
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader


def get_pku_test_dataset(dataset_seed=8888, fraction=1.0):
    pku_test_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", cache_dir="./.cache",revision="ff7ba91063016c78a225b0f74e1c0860bb18230f" )[
        "test"
    ]
    torch.manual_seed(dataset_seed)
    np.random.seed(dataset_seed)
    random.seed(dataset_seed)

    # Preproccess function.
    if fraction == 1:
        return pku_test_dataset
    return pku_test_dataset.train_test_split(train_size=fraction, seed=dataset_seed)[
        "train"
    ]


def get_real_toxic_dataset(dataset_seed=8888, fraction=1.0):
    toxic_dataset = load_dataset("allenai/real-toxicity-prompts", cache_dir="./.cache")[
        "train"
    ]
    torch.manual_seed(dataset_seed)
    np.random.seed(dataset_seed)
    random.seed(dataset_seed)

    # Preproccess function.
    if fraction == 1:
        return toxic_dataset
    return toxic_dataset.train_test_split(train_size=fraction, seed=dataset_seed)[
        "train"
    ]


def bookcorpus_loaders(tokenizer, batch_size):
    dataset = load_dataset(
        "bookcorpus", "plain_text", split="train", cache_dir="./.cache"
    )

    def preproccess(examples):
        results = {"input_ids": [], "attention_mask": []}
        for i in range(len(examples["text"])):
            text = examples["text"][i]
            tokenized = tokenizer(text, truncation=True, padding="max_length")
            results["input_ids"].append(tokenized["input_ids"])
            results["attention_mask"].append(tokenized["attention_mask"])
        return results

    tokenized_datasets = dataset.map(preproccess, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_dataloader


def get_WikiMIA_dataset(LENGTH):
    return load_dataset(
        "swj0419/WikiMIA", cache_dir="./.cache", split=f"WikiMIA_length{LENGTH}"
    )


def build_unlearn_dataset(dataset, dataset_seed=8888, forget_ratio=0.1):
    torch.manual_seed(dataset_seed)
    np.random.seed(dataset_seed)
    random.seed(dataset_seed)

    Total_training_data = [item for item in dataset if item["label"] == 1]
    Total_test_data = [item for item in dataset if item["label"] == 0]

    random.shuffle(Total_training_data)

    forget_dataset = Total_training_data[: int(len(Total_training_data) * forget_ratio)]
    remain_dataset = Total_training_data[int(len(Total_training_data) * forget_ratio) :]

    return forget_dataset, remain_dataset, Total_test_data


if __name__ == "__main__":
    data = get_real_toxic_dataset()
    print(data[0])
