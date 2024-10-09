from ..utils import read_text

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List, Tuple


class DefaultDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_len: int = 4096,
        add_bos_token: bool = True,
        repeats: int = 1,   # only applied to forget
    ):
        self.max_len = max_len
        self.add_bos_token = add_bos_token
        self.text = read_text(file_path)
        self.tokenizer = tokenizer
        self.encoding = self.tokenizer(self.text, add_special_tokens=False, return_tensors='pt')
        self.num_tokens = len(self.encoding.input_ids[0])

        tokens = self.encoding.input_ids[0].repeat(repeats)
        assert len(tokens.shape) == 1, "Debug error: Tokens not 1-dimensional"

        if add_bos_token:
            self.input_ids = [
                F.pad(
                    tokens[i : i + max_len - 1], (1, 0),
                    value=tokenizer.bos_token_id
                )
                for i in range(0, len(tokens), max_len - 1)
            ]
        else:
            self.input_ids = [
                tokens[i : i + max_len]
                for i in range(0, len(tokens), max_len)
            ]

        # Rotate the tokens if the last `input_ids` isn't filled to max_len
        if len(self.input_ids[-1]) < max_len:
            self.input_ids[-1] = torch.concat(
                [self.input_ids[-1], self.input_ids[0]], dim=-1
            )[:max_len]

        # Original strings
        self.strings = tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)

        pass    # def __init__()


    def __getitem__(self, index):
        return self.input_ids[index]


    def __len__(self):
        return len(self.input_ids)


    def get_collate_fn(self):

        def collate_fn(batch: List[torch.Tensor]):
            batch = torch.stack(batch)
            return {
                "input_ids": batch,
                "labels": batch.clone()
            }

        return collate_fn
        


class ForgetRetainDataset(DefaultDataset):

    def __init__(
        self,
        forget_file_path: str,
        tokenizer: AutoTokenizer,
        retain_file_path: str | None = None,
        max_len: int = 4096,
        add_bos_token: bool = True,
        repeats: int = 1
    ):
        self.forget_dataset = DefaultDataset(
            forget_file_path, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token, repeats=repeats
        )

        self.retain_exists = retain_file_path is not None
        if self.retain_exists:
            self.retain_dataset = DefaultDataset(
                retain_file_path, tokenizer,
                max_len=max_len, add_bos_token=add_bos_token
            )

        self.tokenizer = tokenizer


    def __getitem__(self, index):
        if self.retain_exists:
            return (
                self.forget_dataset[index],
                self.retain_dataset[index % len(self.retain_dataset)]
            )
        else:
            return self.forget_dataset[index], None


    def __len__(self):
        return len(self.forget_dataset)


    def get_collate_fn(self):

        def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
            batch_forget = torch.stack([pair[0] for pair in batch])
            dict_forget = {
                "input_ids": batch_forget,
                "labels": batch_forget.clone(),
                "attention_mask": torch.ones_like(batch_forget)
            }

            if self.retain_exists:
                batch_retain = torch.stack([pair[1] for pair in batch])
                dict_retain = {
                    "input_ids": batch_retain,
                    "labels": batch_retain.clone(),
                    "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
                }
            else:
                dict_retain = None

            return dict_forget, dict_retain

        return collate_fn
