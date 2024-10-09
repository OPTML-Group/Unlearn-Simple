from .utils import read_text, pad_or_trim_tensor

from typing import List, Tuple
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer


class DefaultDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True
    ):
        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data[0], str):
                self.strings = data
            elif isinstance(data[0], dict) and 'text' in data[0] \
                    and isinstance(data[0]['text'], str):
                self.strings = [d['text'] for d in data]
                if 'input_ids' in data[0]:
                    self.input_ids = [torch.tensor(d['input_ids']) for d in data]
                    return; # Done, since we have `input_ids` ready.
            else:
                raise ValueError("Format of this `.json` file is not recognized.")

            assert tokenizer is not None, "Tokenizer must be specified."

            self.input_ids = []
            for s in self.strings:
                encoding: torch.Tensor = tokenizer(
                    s,
                    add_special_tokens=add_bos_token,
                    return_tensors='pt'
                ).input_ids[0]
                encoding = pad_or_trim_tensor(
                    encoding,
                    target_length=max_len,
                    padding_value=tokenizer.pad_token_id
                )
                self.input_ids.append(encoding)

            return; # end if Path(file_path).suffix == '.json'

        assert Path(file_path).suffix == '.txt'

        tokens = tokenizer(read_text(file_path), add_special_tokens=False, return_tensors='pt').input_ids[0]
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
        add_bos_token: bool = True
    ):
        self.forget_dataset = DefaultDataset(
            forget_file_path, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token
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
