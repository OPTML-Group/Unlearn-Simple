# Modified from https://github.com/EleutherAI/knowledge-neurons/blob/main/knowledge_neurons/patch.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm
import pickle
import pdb
from utils import get_attributes, set_attributes


class Patch(torch.nn.Module):
    def __init__(
        self,
        ff_layer: nn.Module,
        intermediate_size: int = None,
        replacement_activations: torch.Tensor = None,
        onehot_coef: torch.Tensor = None,
    ):
        super().__init__()
        self.module = ff_layer
        if intermediate_size is not None: # slimming
            self.slim_coef = nn.Parameter(torch.ones(intermediate_size))
        self.acts = replacement_activations
        self.onehot_coef = onehot_coef

    def forward(self, x: torch.Tensor):
        hidden_states = self.module(x)
        if self.acts is not None:          # knowledge neurons
            hidden_states[:, -1, :] = self.acts # patch the last token
        elif self.onehot_coef is not None: # zero-out
            hidden_states = hidden_states * self.onehot_coef.unsqueeze(1)
        else:                              # slimming
            hidden_states *= torch.clip(self.slim_coef, 0, 1)

        return hidden_states


def patch_ff_layer(
    model: nn.Module,
    ff_attrs: str,
    intermediate_size: int = None,
    replacement_activations: torch.Tensor = None,
    onehot_coef: torch.Tensor = None,
):
    """
    replaces the ff layer at `layer_idx` with a `Patch` class - that will replace the intermediate activations at sequence position
    `mask_index` with `replacement_activations`
    """

    ff_layer = get_attributes(model, ff_attrs)
    patch = Patch(
            ff_layer,
            intermediate_size,
            replacement_activations,
            onehot_coef,
        )

    set_attributes(model, ff_attrs, patch)
    #print(f"Patch {ff_attrs}")


def unpatch_ff_layer(
    model: nn.Module,
    ff_attrs: str,
):
    """
    Removes the `Patch` applied by `patch_ff_layer`, replacing it with its original value.
    """
    ff_layer = get_attributes(model, ff_attrs)
    assert isinstance(ff_layer, Patch), "Can't unpatch a layer that hasn't been patched"
    set_attributes(model, ff_attrs, ff_layer.module)
    #print(f"Reset {ff_attrs}")


def patch_slim(model):
    for ly in range(model.config.n_layer):
        ff_attrs = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        patch_ff_layer(
            model,
            ff_attrs,
            intermediate_size=model.inner_dim,
        )

def reinit_slim(model):
    for ly in range(model.config.n_layer):
        attrs_str = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}.slim_coef"
        coef = get_attributes(model, attrs_str)
        init.ones_(coef)
