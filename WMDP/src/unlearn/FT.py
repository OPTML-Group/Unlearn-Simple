import torch
from transformers import Trainer

from .base import BaseTrainer


class FT(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        retain_data = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        outputs = model(**retain_inputs)

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
    

class Relearn(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if inputs.get("perturb") is not None:
            forget_data = inputs["perturb"]
        else:
            forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }

        outputs = model(**forget_inputs)

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss