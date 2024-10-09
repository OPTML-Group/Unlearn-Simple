import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
import torch.nn.functional as F


class WHPModelForCausalLM(PreTrainedModel):
    def __init__(self, baseline_name_or_path, reinforced_name_or_path, alpha=1., config=None, **kwargs):
        if config is None:
            config = PretrainedConfig.from_pretrained(baseline_name_or_path)
        super().__init__(config)
        self.baseline = AutoModelForCausalLM.from_pretrained(baseline_name_or_path, **kwargs)
        self.reinforced = AutoModelForCausalLM.from_pretrained(reinforced_name_or_path, **kwargs)
        self.alpha = alpha


    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=True, **kwargs):
        v_b = self.baseline(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            # return_dict=True,
                            **kwargs)
        v_r = self.reinforced(input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=labels,
                            #   return_dict=True,
                              **kwargs)
        logits = v_b.logits - self.alpha * F.relu(v_r.logits - v_b.logits)

        if not return_dict:
            return (logits,) + v_b[1:]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutput(logits=logits, loss=loss)


    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        return self.baseline.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, **model_kwargs)
    

    def _reorder_cache(self, past, beam_idx):
        return self.baseline._reorder_cache(past, beam_idx)
