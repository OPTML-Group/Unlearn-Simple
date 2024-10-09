# Modified from https://github.com/EleutherAI/knowledge-neurons/blob/main/knowledge_neurons/knowledge_neurons.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from .utils import get_attributes
from .patch import *
from .config import *


def register_hook(model, layer_idx, ori_activations, attr_str):
    ff_layer = get_attributes(model, attr_str)

    def hook_fn(m, i, o):
        ori_activations[layer_idx] = o.squeeze().cpu()

    return ff_layer.register_forward_hook(hook_fn)


@torch.no_grad()
def get_ori_activations(model, inputs):
    seq_len = inputs['input_ids'].shape[1]
    ori_activations = torch.zeros((model.config.n_layer, seq_len, model.inner_dim))

    handles = []
    for ly in range(model.config.n_layer):
        handle = register_hook(
            model,
            ly,
            ori_activations,
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}",
        )
        handles.append(handle)

    out = model(**inputs)

    for handle in handles: # detach the hooks
        handle.remove()

    return ori_activations


def largest_act(model, inputs, out_dir,
                prompt_start_i: int = 1):
    @torch.no_grad()
    def get_ffn_norms():
        all_norms = torch.zeros((model.config.n_layer, model.inner_dim))
        for ly in range(model.config.n_layer):
            attr_str = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_out']}.weight"
            weights = get_attributes(model, attr_str)
            # if 'gpt2' in args.model_name:
            #     norms = torch.norm(weights, dim=1)
            # else:
            norms = torch.norm(weights, dim=0)
            all_norms[ly] = norms.cpu()

        return all_norms

    prompt_start_i = 0

    activations = get_ori_activations(model, inputs)
    activations = activations[:, prompt_start_i: -1] # [n_layer, suffix_len, inner_dim]
    all_norms = get_ffn_norms()

    act_mean = activations.mean(1).cpu().abs() * all_norms
    torch.save(act_mean, os.path.join(out_dir, 'act-mean.pt'))
    return act_mean


def scaled_input(activations, steps, device):
    """
    Tiles activations along the batch dimension - gradually scaling them over
    `steps` steps from 0 to their original value over the batch dimensions.
    """
    tiled_activations = activations.expand((steps, len(activations)))
    scales = torch.linspace(start=0, end=1, steps=steps)[:, None] # (steps, 1)
    out = (tiled_activations * scales).to(device)
    return out # [steps, inner_dim]


def integrated_gradients(model, inputs, out_dir,
                         prompt_start_i = 0,
                         n_batches = 16,
                         device = 'cuda',
                         ig_steps = 20
                         ):
    activations = get_ori_activations(model, inputs)

    target_ids = inputs['input_ids'].squeeze()[1:].tolist() 
    seq_len = inputs['input_ids'].shape[1]

    n_layer = model.config.n_layer
    integrated_grads_ = torch.zeros((n_layer, seq_len-1-prompt_start_i, model.inner_dim))

    for ly in tqdm(range(n_layer)):
        integrated_grads = []
        for i in range(prompt_start_i, seq_len-1):
            ori_activations = activations[ly, i]

            scaled_weights = scaled_input(ori_activations, steps=ig_steps, device=device)
            scaled_weights.requires_grad_(True)

            ff_attrs = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
            integrated_grads_t = torch.zeros(model.inner_dim)
            for batch_weights in scaled_weights.chunk(n_batches): # batch ig_steps
                bs = len(batch_weights)
                cur_input_ids = inputs['input_ids'][:,:i+1].expand(bs, i+1) # [ig_steps, cur_seq_len]

                # patch the model with the scaled activations
                patch_ff_layer(
                    model,
                    ff_attrs,
                    replacement_activations=batch_weights,
                )

                outputs = model(cur_input_ids)
                probs = F.softmax(outputs.logits[:, -1, :], dim=-1) # [ig_steps, vocab]
                grad = torch.autograd.grad(
                            torch.unbind(probs[:, target_ids[i]]), batch_weights
                        )[0] # [ig_steps, inner_dim]
                integrated_grads_t += grad.sum(dim=0).cpu() # sum over ig_steps

                unpatch_ff_layer(
                    model,
                    ff_attrs,
                )
            # Eq 5, 1/m * (ori - baseline) * (\Sum grads), where we use baseline = 0
            integrated_grads_t = ori_activations * integrated_grads_t / ig_steps
            integrated_grads.append(integrated_grads_t)

        integrated_grads_[ly] = torch.stack(integrated_grads, dim=0)

    ig_mean = integrated_grads_.mean(1).cpu()
    torch.save(ig_mean, os.path.join(out_dir, 'ig-mean.pt'))
    return ig_mean


@torch.no_grad()
def apply_neuron_mask(model, values, r, start_mask_layer=1, verbose=True):

    # First, set all to ones
    reinit_slim(model)

    # Then, zero-out the identified neurons
    total = 0
    n_neurons = []
    for l in range(start_mask_layer, model.config.n_layer):
        _, indices = torch.topk(values[l], int(model.inner_dim*r))
        attrs_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_act']}.slim_coef"
        coef = get_attributes(model, attrs_str)
        coef[indices] = 0.

        n = len(indices)
        total += n
        n_neurons.append(n)

    if verbose:
        print("# Zero-Out:", f'{total/len(values.view(-1)):.1%}')
        print(n_neurons)


def download_all(model, attributions, out_dir):
    dir_name, base_name = os.path.dir_name(out_dir), os.path.base_name(out_dir)
    for r in RATIOS:
        apply_neuron_mask(model, attributions, r)
        model.save_pretrained(os.path.join(dir_name, f"{base_name}_{r}"))
