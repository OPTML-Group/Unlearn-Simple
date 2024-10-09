from .whos_harry_potter import WHPModelForCausalLM
from .task_vector import unlearn as tv_unlearn

from pathlib import Path
import os
import torch
from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re


def get_rootpath():
    return str(Path(__file__).parent.resolve())


def get_basename(file_path: str):
    return os.path.basename(os.path.normpath(file_path))


def read_text(file_path: str) -> str:
    import pandas as pd

    if Path(file_path).suffix != '.txt':
        raise ValueError

    with open(file_path, 'r') as f:
        text: str = f.read()
    return text


def read_json(fpath: str):
    fpath = str(fpath)
    with open(fpath, 'r') as f:
        return json.load(f)


def output_json(data, fpath: str):
    fpath = str(fpath)
    assert fpath.endswith('.json')
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f: json.dump(data, f)


def file_exists(dir: str) -> bool:
    return os.path.isdir(dir) and any(os.path.isfile(os.path.join(dir, f)) for f in os.listdir(dir))


def output_text(data, fpath: str):
    fpath = str(fpath)
    assert fpath.endswith('.txt')
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f: f.write(data)


def load_model(
    model_dir: str,
    model_name: str | None = None,
    quantization_config: any = None,
    reinforced_model_dir: str | None = None
) -> AutoModelForCausalLM:
    def extract_alpha(s):
        pattern = r'alpha=([+-]?\d*\.\d+|[+-]?\d+)'
        match = re.search(pattern, s)
        if match: return float(match.group(1))
        else: return None

    if model_name is not None:
        alpha = extract_alpha(model_name)
        if 'whp' in model_name:
            assert reinforced_model_dir is not None
            model = WHPModelForCausalLM(
                model_dir, reinforced_model_dir,
                alpha=alpha if alpha is not None else 1.,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            return model

        elif 'tv' in model_name:
            assert reinforced_model_dir is not None
            print(model_dir)
            print(reinforced_model_dir)
            print(alpha)
            model = tv_unlearn(model_dir=model_dir,
                               some_pt_model_dir=model_dir,
                               some_ft_model_dir=reinforced_model_dir,
                               alpha=alpha if alpha is not None else 1.)
            return model

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    return model


def load_tokenizer(
    tokenizer_dir: str,
    add_pad_token: bool = True,
    use_fast: bool = True
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast) 
    if add_pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(
    model_dir: str,
    model_name: str | None = None,
    tokenizer_dir: str | None = None,
    add_pad_token: bool = True,
    quantization_config: any = None,
    reinforced_model_dir: str | None = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = load_model(
        model_dir, model_name, quantization_config,
        reinforced_model_dir=reinforced_model_dir
    )
    tokenizer = (load_tokenizer(tokenizer_dir, add_pad_token)
                 if tokenizer_dir is not None
                 else None)
    return model, tokenizer


def estimate_steps_per_epoch(samples: int,
                             epochs: int,
                             *_,
                             per_device_batch_size: int | None = None,
                             batch_size: int | None = None):
    """Overestimates number of steps per epoch.
    """
    from torch.cuda import device_count
    from math import ceil

    if per_device_batch_size is None and batch_size is None:
        raise ValueError("Either per_device_batch_size or batch_size must be specified.")
    if batch_size is None:
        # per_device_batch_size is specified
        cnt = device_count()
        if cnt == 0:
            raise ValueError("Device not detected.")
        batch_size: int = device_count() * per_device_batch_size

    samples_per_epoch = ceil(samples / epochs)
    steps_per_epoch = ceil(samples_per_epoch / batch_size)
    return steps_per_epoch


def pad_or_trim_tensor(tensor, target_length, padding_value=0):
    current_length = tensor.size(0)
    
    if current_length < target_length:
        # Padding
        padding_size = target_length - current_length
        padding_tensor = torch.full((padding_size,), padding_value, dtype=tensor.dtype)
        padded_tensor = torch.cat((tensor, padding_tensor))
        return padded_tensor
    
    elif current_length > target_length:
        # Trimming
        trimmed_tensor = tensor[:target_length]
        return trimmed_tensor
    
    else:
        # No change needed
        return tensor
