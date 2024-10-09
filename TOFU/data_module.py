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
class WMDPDataset:
    def __init__(self, seed=42, ratio=1.0, subset="forget"):
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
            return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=self.dataset["train"].column_names
        )

        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        self.dataset["train"] = train_dataset

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
