import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pdb
import os
from config import *


def get_attr_str(model_name):
    if 'gpt2' in model_name:
        attr_dict = {
            'transformer_layer': 'transformer.h',
            'ffn_out': 'mlp.c_proj',
            'ffn_act': 'mlp.act',
            'lm_head': 'lm_head',
        }
    elif 'gpt-j' in model_name:
        attr_dict = {
            'transformer_layer': 'transformer.h',
            'ffn_out': 'mlp.fc_out',
            'ffn_act': 'mlp.act',
            'lm_head': 'lm_head',
        }
    elif 'pythia' in model_name:
        attr_dict = {
            'transformer_layer': 'gpt_neox.layers',
            'ffn_out': 'mlp.dense_4h_to_h',
            'ffn_act': 'mlp.act',
            'lm_head': 'embed_out',
        }
    else:
        raise NotImplementedError(f"{model_name} attributes unkown!")
    return attr_dict

def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.h.0.mlp.c_proj')
        should return the same as model.transformer.h.0.mlp.c_proj
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x

def set_attributes(x: nn.Module, attributes: str, values):
    attr_list = attributes.split(".")
    for attr in attr_list[:-1]:
        x = getattr(x, attr)
    setattr(x, attr_list[-1], values)


def set_model_attributes(model, model_name):
    model.config.pad_token_id = model.config.eos_token_id
    model.attr_dict = get_attr_str(model_name)
    model.inner_dim = 4 * model.config.hidden_size
    if not hasattr(model.config, "n_layer"):
        model.config.n_layer = model.config.num_hidden_layers


@torch.no_grad()
def generate(model, tokenizer, prompts, max_len, batch):
    model.eval()
    tokenizer.padding_side = "left" if batch else "right"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    gen_tokens = model.generate(
      inputs["input_ids"].to(device),
      do_sample=False,
      max_length=max_len,
    )

    return tokenizer.batch_decode(gen_tokens, skip_special_tokens=False)


def shift_position_ids(row, col, verbose=False):
    arr = torch.zeros((row,col), dtype=int)
    shift_ids = torch.arange(row-1, -1, -1)

    for i in range(row):
        j = shift_ids[i]
        arr[i][j:] = torch.arange(col-j)

    if verbose:
        print(arr)
    return arr


@torch.no_grad()
def ffn_zero(model, layer, op='zero'):
    attr_str = f"{model.attr_dict['transformer_layer']}.{layer}.{model.attr_dict['ffn_out']}.weight"
    weight = get_attributes(model, attr_str)
    clone = weight.clone()
    weight.fill_(0.)
    return clone

@torch.no_grad()
def ffn_copy(model1, model2, layer):
    # model1 <- model2; return original model1 copy
    attr_str = f"{model1.attr_dict['transformer_layer']}.{layer}.{model1.attr_dict['ffn_out']}.weight"
    clone = get_attributes(model1, attr_str).clone()
    tgt = nn.Parameter(get_attributes(model2, attr_str).to(device))
    set_attributes(model1, attr_str, tgt)
    return clone

@torch.no_grad()
def ffn_restore(model, weight, layer):
    attr_str = f"{model.attr_dict['transformer_layer']}.{layer}.{model.attr_dict['ffn_out']}.weight"
    set_attributes(model, attr_str, nn.Parameter(weight)) # requires_grad=True

@torch.no_grad()
def all_ffn_restore(model, weights):
    for ly in range(model.config.n_layer):
        ffn_restore(model, weights[ly], ly)

@torch.no_grad()
def get_all_ffn_weights(model):
    weights = []
    for layer in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{layer}.{model.attr_dict['ffn_out']}.weight"
        weight = get_attributes(model, attr_str).detach().cpu()
        weights.append(weight)
    weights = torch.stack(weights)
    print("ffn_out.shape:", weights.shape)
    return weights


@torch.no_grad()
def get_tf_probs(args, inputs, target_ids, model, tokenizer):
    model.eval()
    outputs = model(**inputs)
    probs = nn.functional.softmax(outputs.logits.squeeze(), dim=-1)
    probs = probs[-len(target_ids)-1: -1] # -1 for next token prediction
    correctness = probs.argmax(-1) == target_ids
    tgt_probs = probs[range(len(probs)), target_ids]

    return tgt_probs, correctness, outputs.loss.item()
