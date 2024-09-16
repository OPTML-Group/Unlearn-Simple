import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import os
from utils import get_model_identifiers_from_yaml

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
    


class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if './TOFU_data' not in data_path: # load dataset from hugingface hub.
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else: # load dataset from local files.
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if './TOFU_data' not in data_path:
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            
            torch.manual_seed(idx)
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if './TOFU_data' not in data_path:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if './TOFU_data' not in data_path:
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:

            torch.manual_seed(idx)
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextForgetDatasetKTOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetKTOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if './TOFU_data' not in data_path:
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        if './TOFU_data' not in data_path:
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:

            torch.manual_seed(idx)
            
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                rets.append(converted_data)
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()

                answer = self.idk[rand_pos].strip()
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                rets.append(converted_data)
        return rets


class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if './TOFU_data' not in data_path: # load dataset from hugingface hub.
            self.data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze()
    
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict

# class BaseDataset:
#     def __init__(self, if_llama=False, dataset_path=None):
#         self.dataset_path = dataset_path
#         self.if_llama = if_llama
#         self.question_start_token = "[INST] " if self.if_llama else "### Question: "
#         self.question_end_token = " [/INST]" if if_llama else "\n"
#         self.answer_start_token = " " if if_llama else "### Answer: "

#     def get_dataset(self):
#         pass

#     def __preprocess__(self, tokenizer, forget_ratio, dataset_seed):
#         pass

#     def build_dataset(self, tokenizer, forget_ratio, dataset_seed):
#         pass

#     def _padding_fn(self, prompt, response, max_length, tokenizer):
#         text = (
#             self.question_start_token
#             + prompt
#             + self.question_end_token
#             + self.answer_start_token
#             + response
#         )
#         tokenized = tokenizer(
#             text,
#             truncation=True,
#             padding="max_length",
#             add_special_tokens=True,
#         )
#         num_prompt_token = len(
#             tokenizer.tokenize(
#                 self.question_start_token + prompt + self.question_end_token,
#                 add_special_tokens=True,
#             )
#         )
#         pad_length = max_length - len(tokenized.input_ids)
#         if pad_length < 0:
#             return None
#         pad_input_ids = tokenized.input_ids + [tokenizer.pad_token_id] * pad_length
#         pad_attention_mask = tokenized.attention_mask + [0] * pad_length
#         if len(tokenized.input_ids) == max_length:
#             label = tokenized.input_ids
#         else:
#             label = (
#                 tokenized.input_ids
#                 + [tokenizer.eos_token_id]
#                 + [-100] * (pad_length - 1)
#             )
#         for i in range(num_prompt_token):
#             label[i] = -100

#         assert (
#             len(pad_input_ids) == max_length
#         ), f"input_id length mismatch: {len(pad_input_ids)} (expect: {max_length})"
#         assert (
#             len(pad_attention_mask) == max_length
#         ), f"attention_mask length mismatch: {len(pad_attention_mask)} (expect: {max_length})"
#         assert (
#             len(label) == max_length
#         ), f"label length mismatch: {len(label)} (expect: {max_length})"

#         return {
#             "input_ids": torch.tensor(pad_input_ids),
#             "attention_mask": torch.tensor(pad_attention_mask),
#             "labels": torch.tensor(label),
#         }

class WMDPDataset:
    def __init__(self, seed=42, ratio=1.0, subset="forget"):
        # super().__init__(dataset_path=dataset_path)
        self.ratio = ratio
        self.seed = seed
        self.subset = subset
        self.dataset = self.get_dataset()

    def get_dataset(self):
        
        if self.subset == "retain":
            train_dataset_cyber = load_dataset(
                "cais/wmdp-corpora", "cyber-retain-corpus", cache_dir="./.cache"
            )["train"]
            
            train_dataset_bio = load_dataset(
                "cais/wmdp-corpora", "bio-retain-corpus", cache_dir="./.cache"
            )["train"]

        else:
            train_dataset_cyber = load_dataset(
                "cais/wmdp-corpora", "cyber-forget-corpus", cache_dir="./.cache"
            )["train"]
            train_dataset_bio = load_dataset(
                "json",
                data_files="data/bio_remove_dataset.jsonl",
                split="train",
                cache_dir="./.cache",
            )


        dataset = defaultdict()
        dataset["train"] = concatenate_datasets([train_dataset_cyber, train_dataset_bio])

        return dataset
        # self.forget_data = concatenate_datasets([train_dataset_cyber, train_dataset_bio])
        
        # train_dataset = self.dataset["train"].map(
        #     preprocess, batched=True, remove_columns=self.dataset["train"].column_names
        # )
        # test_dataset = self.dataset["test"].map(
        #     preprocess, batched=True, remove_columns=self.dataset["test"].column_names
        # )

        # train_dataset.set_format(
        #     type="torch", columns=["input_ids", "attention_mask", "labels"]
        # )

        # test_dataset.set_format(
        #     type="torch", columns=["input_ids", "attention_mask", "labels"]
        # )

        # self.dataset["train"] = train_dataset
        # self.dataset["test"] = test_dataset

        # length = len(total_data)
        # total_data = total_data.shuffle(seed=self.seed)
        # total_data = total_data.select(range(int(length * self.ratio)))
        # dataset = defaultdict()
        # total_data = total_data.train_test_split(test_size=0.1, seed=self.seed)
        # dataset["test"] = total_data["test"]
        # dataset["train"] = total_data["train"]

        # dataset = defaultdict()
        # dataset["train"] = train_dataset
        # return dataset

    def __preprocess__(self, tokenizer):
        def preprocess(examples):
            results = {"input_ids": [], "attention_mask": [], "labels": []}
            tokenized = tokenizer(
                examples["text"],
                max_length=1024,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )
            results["input_ids"] = tokenized.input_ids
            results["attention_mask"] = tokenized.attention_mask
            results["labels"] = tokenized.input_ids
            # results = [
            #     tokenized.input_ids,
            #     tokenized.attention_mask,
            #     tokenized.input_ids,
            # ]

            return results
        
        # def preprocess_test(examples):
        #     prompt_template = "The followingare multiple choice questions (with answers) about Cybersecurity.\n\n {}\nA.{}\nB.{}\nC.{}\nD.{}\n\n"
        #     results = {
        #         "input_ids": [],
        #         "attention_mask": [],
        #         "answer": [],
        #     }
        #     for i in range(len(examples["question"])):
        #         question = examples["question"][i]
        #         choices = examples["choices"][i]
        #         prompt = prompt_template.format(
        #             question, choices[0], choices[1], choices[2], choices[3]
        #         )
        #         full_prompt = (
        #             self.question_start_token
        #             + prompt
        #             + self.question_end_token
        #             + self.answer_start_token
        #         )
        #         inputs = tokenizer(
        #             full_prompt, max_length=1024, padding="max_length", truncation=True
        #         )
        #         results["input_ids"].append(torch.tensor(inputs["input_ids"]))
        #         results["attention_mask"].append(torch.tensor(inputs["attention_mask"]))
        #         results["answer"].append(examples["answer"][i])
        #     return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=self.dataset["train"].column_names
        )
        # test_dataset = self.dataset["test"].map(
        #     preprocess_test, batched=True, remove_columns=self.dataset["test"].column_names
        # )

        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        # test_dataset.set_format(
        #     type="torch", columns=["input_ids", "attention_mask", "labels"]
        # )

        self.dataset["train"] = train_dataset
        # self.dataset["test"] = test_dataset

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)
        return self.dataset

import random
class UnlearnDataset(Dataset):
    def __init__(self, datasets):
        self.forget_dataset = datasets.get("forget", None)
        self.retain_dataset = datasets.get("retain", None)

    def __len__(self):
        if self.forget_dataset:
            return len(self.forget_dataset)
        if self.retain_dataset:
            return len(self.retain_dataset)
        raise ValueError("No dataset available.")

    def __getitem__(self, idx):
        # data = {"forget": None, "retain": None}

        # if self.forget_dataset:
        #     data["forget"] = self.forget_dataset[idx]
        #     if self.retain_dataset:
        #         retain_idx = random.randint(0, len(self.retain_dataset) - 1)
        #         data["retain"] = self.retain_dataset[retain_idx]
        # elif self.retain_dataset:
        #     data["retain"] = self.retain_dataset[idx]
        # print(data)

        forget_data = self.forget_dataset[idx]
        retain_idx = random.randint(0, len(self.retain_dataset) - 1)
        retain_data = self.retain_dataset[retain_idx]
        data = [forget_data, retain_data]
        return data

def unlearncollector(samples):
    res = {"forget": None, "retain": None}
    if samples["forget"]:
        forget_samples = [sample["forget"] for sample in samples]
        res["forget"] = (
            torch.stack([sample["input_ids"] for sample in forget_samples]),
            torch.stack([sample["attention_mask"] for sample in forget_samples]),
            torch.stack([sample["label"] for sample in forget_samples])
        )
    if samples["retain"]:
        retain_samples = [sample["retain"] for sample in samples]
        res["retain"] = (
            torch.stack([sample["input_ids"] for sample in retain_samples]),
            torch.stack([sample["attention_mask"] for sample in retain_samples]),
            torch.stack([sample["label"] for sample in retain_samples])
        )
    return res

# class StackUnlearnDataset(Dataset):
#     """
#     Stack multiple datasets together. If the datasets are dictionaries, the keys are used to
#     stack the datasets. Otherwise, the datasets are stacked in order. @ljcc
#     """

#     def set_epoch(self, epoch):
#         self.epoch = epoch

#     def __init__(self, *args, **kwargs):
#         if args:
#             if kwargs:
#                 raise ValueError("Arguments and keyword arguments cannot be mixed")
#             self._length = len(args[0])
#             self.datasets = args
#         else:
#             if "main_key" in kwargs:
#                 main_key = kwargs.pop("main_key")
#                 assert (
#                     main_key in kwargs
#                 ), f'key: "{main_key}" (main_key) must be provided'
#                 self._length = len(kwargs[main_key])
#             else:
#                 self._length = max(len(dataset) for dataset in kwargs.values())
#             self.datasets = kwargs
#         self.epoch = 0

#     def __getitem__(self, index):
#         idx = index + self._length * self.epoch

#         if isinstance(self.datasets, dict):
#             item = {
#                 key: dataset[idx % len(dataset)]
#                 for key, dataset in self.datasets.items()
#             }
#         else:
#             item = [dataset[idx % len(dataset)] for dataset in self.datasets]
#         return item

#     def __len__(self):
#         return self._length

# from dataclasses import dataclass
# from transformers.data.data_collator import DefaultDataCollator, default_data_collator   
# @dataclass
# class StackDataCollator(DefaultDataCollator):
#     """
#     Collator for StackUnlearnDataset. @ljcc
#     """

#     def __call__(self, features, return_tensors=None):
#         if return_tensors is None:
#             return_tensors = self.return_tensors

#         first = features[0]
#         if isinstance(first, dict):
#             collated = {
#                 key: default_data_collator([f[key] for f in features], return_tensors)
#                 for key in first.keys()
#             }
#         else:
#             collated = [
#                 default_data_collator([f[i] for f in features], return_tensors)
#                 for i in range(len(first))
#             ]
#         return collated


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
