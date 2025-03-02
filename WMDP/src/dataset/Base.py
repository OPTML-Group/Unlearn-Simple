import random
import torch
from torch.utils.data import Dataset


class BaseDataset:
    def __init__(self, dataset_name, with_retain=False, if_llama=False):
        self.dataset_name = dataset_name
        self.with_normal = with_retain
        self.if_llama = if_llama
        self.question_start_token = "### Question: "
        self.question_end_token = "\n"
        self.answer_start_token = "### Answer: "
    def get_dataset(self):
        pass

    def __preprocess__(self, tokenizer, forget_ratio, dataset_seed):
        pass

    def build_dataset(self, tokenizer, forget_ratio, dataset_seed):
        pass


class UnlearnDataset(Dataset):
    def __init__(self, datasets, forget_ratio, dataset_seed, self_retain=False):
        self.forget_ratio = forget_ratio
        self.dataset_seed = dataset_seed
        self.self_retain = self_retain

        if "forget" in datasets.keys():
            self.forget_dataset = datasets["forget"]
        else:
            self.forget_dataset = None

        if "retain" in datasets.keys():
            self.retain_dataset = datasets["retain"]
        else:
            self.retain_dataset = None

        if "perturb" in datasets.keys():
            self.perturb_dataset = datasets["perturb"]
            # print(f"perturb_dataset: {len(self.perturb_dataset)}")
        else:
            self.perturb_dataset = None

        self.build_unlearn_dataset()

    def __len__(self):
        if self.forget_dataset:
            return len(self.forget_dataset)
        elif self.retain_dataset:
            return len(self.retain_dataset)
        else:
            raise ValueError("No dataset")

    def build_unlearn_dataset(self):
        if self.forget_dataset:
            if self.forget_ratio > 1:
                length = int(self.forget_ratio)

            elif self.forget_ratio <= 1 and self.forget_ratio > 0:
                length = int(len(self.forget_dataset) * self.forget_ratio)

            # random.seed(self.dataset_seed)
            forget_index_list = random.sample(range(len(self.forget_dataset)), length)
            forget_index_list = list(range(length))
            if self.self_retain:
                retain_index_list = list(
                    set(range(len(self.forget_dataset))) - set(forget_index_list)
                )
                self.retain_dataset = self.forget_dataset.select(retain_index_list)
            self.forget_dataset = self.forget_dataset.select(forget_index_list)

    def __getitem__(self, idx):
        if self.forget_dataset:
            forget_data = self.forget_dataset[idx]
            if self.retain_dataset:
                retain_idx = random.randint(0, len(self.retain_dataset) - 1)
                retain_data = self.retain_dataset[retain_idx]
                if self.perturb_dataset:
                    perturb_idx= idx % len(self.perturb_dataset)
                    perturb_data = self.perturb_dataset[perturb_idx]

                else:
                    perturb_data = None

                return {"forget": forget_data, "retain": retain_data, "perturb": perturb_data}
            else:
                return {"forget": forget_data, "retain": None, "perturb": None}
        else:
            retain_data = self.retain_dataset[idx]
            if self.perturb_dataset:
                perturb_idx = random.randint(0, len(self.perturb_dataset) - 1)
                perturb_data = self.perturb_dataset[perturb_idx]
            else:
                perturb_data = None

            return {"forget": None, "retain": retain_data, "perturb": perturb_data}


def unlearncollector(samples):
    res = {}
    if samples[0]["forget"]:
        forget_samples = [sample["forget"] for sample in samples]
        res["forget"] = (
            torch.stack([sample["input_ids"] for sample in forget_samples]),
            torch.stack([sample["attention_mask"] for sample in forget_samples]),
            torch.stack([sample["label"] for sample in forget_samples]),
            torch.stack([sample["refused_label"] for sample in forget_samples]),
            torch.stack([sample["question_length"] for sample in forget_samples]),
        )
    else:
        res["forget"] = None
    if samples[0]["retain"]:
        retain_samples = [sample["retain"] for sample in samples]
        res["retain"] = (
            torch.stack([sample["input_ids"] for sample in retain_samples]),
            torch.stack([sample["attention_mask"] for sample in retain_samples]),
            torch.stack([sample["label"] for sample in retain_samples]),
        )
    else:
        res["retain"] = None

    # add perturb
    if samples[0]["perturb"]:
        perturb_samples = [sample["perturb"] for sample in samples] 
        res["perturb"] = (
            torch.stack([sample["input_ids"] for sample in perturb_samples]),
            torch.stack([sample["attention_mask"] for sample in perturb_samples]),
            torch.stack([sample["label"] for sample in perturb_samples]),
            torch.stack([sample["refused_label"] for sample in perturb_samples]),
            torch.stack([sample["question_length"] for sample in perturb_samples]),
        )
    else:
        res["perturb"] = None

    return res