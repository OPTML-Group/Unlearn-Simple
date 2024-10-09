import sys
sys.path.append(".")
sys.path.append("../baselines")

from typing import List, Dict
import torch
from tqdm import tqdm
import zlib
import numpy as np
from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve


def compute_ppl(text: str, model, tokenizer, device='cuda'):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    ppl = torch.exp(loss).item()
    return ppl, all_prob, loss.item()


def inference(text: str, model, tokenizer) -> Dict:
    pred = {}

    _, all_prob, p1_likelihood = compute_ppl(text, model, tokenizer, device=model.device)
    _, _, p_lower_likelihood = compute_ppl(text.lower(), model, tokenizer, device=model.device)
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

    pred["PPL"] = float(p1_likelihood)
    pred["PPL/lower"] = float(p1_likelihood / p_lower_likelihood)
    pred["PPL/zlib"] = float(p1_likelihood / zlib_entropy)

    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min-{int(ratio*100)}%"] = float(-np.mean(topk_prob).item())

    return pred


def eval_data(data: List[str], model, tokenizer):
    out = []
    for text in tqdm(data):
        out.append({'text': text} | inference(text, model, tokenizer))
    return out


def sweep(ppl, y):
    fpr, tpr, _ = get_roc_curve(y, -ppl)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, get_auc(fpr, tpr), acc


def eval(
    forget_data: List[str],
    retain_data: List[str],
    holdout_data: List[str],
    model, tokenizer
):
    log = {}
    print("Evaluating on the forget set...")
    log['forget'] = eval_data(forget_data, model, tokenizer)
    print("Evaluating on the retain set...")
    log['retain'] = eval_data(retain_data, model, tokenizer)
    print("Evaluating on the holdout set...")
    log['holdout'] = eval_data(holdout_data, model, tokenizer)

    auc = {}
    ppl_types = list(log['forget'][0].keys())
    ppl_types.remove('text')
    for split0 in ['forget', 'retain', 'holdout']:
        for split1 in ['forget', 'retain', 'holdout']:
            log0, log1 = log[split0], log[split1]
            for ppl_type in ppl_types:
                ppl_nonmember = [d[ppl_type] for d in log0]
                ppl_member = [d[ppl_type] for d in log1]
                ppl = np.array(ppl_nonmember + ppl_member)
                y = np.array([0] * len(ppl_nonmember) + [1] * len(ppl_member))
                _, _, auc_score, _ = sweep(ppl, y)
                auc[f"{split0}_{split1}_{ppl_type}"] = auc_score
    return auc, log


# def compute_ppl(text: str, model, tokenizer, device='cuda'):
#     # input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
#     # input_ids = input_ids.to(device)
#     # with torch.no_grad():
#     #     outputs = model(input_ids, labels=input_ids)
#     # loss, logits = outputs[:2]

#     # probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
#     # all_prob = []
#     # input_ids_processed = input_ids[0][1:]
#     # for i, token_id in enumerate(input_ids_processed):
#     #     probability = probabilities[0, i, token_id].item()
#     #     all_prob.append(probability)

#     # ppl = torch.exp(loss).item()
#     # return ppl, all_prob, loss.item()

#     # Tokenize the input text and send it to the device
#     input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    
#     # Disable gradient calculation
#     with torch.no_grad():
#         # Forward pass with input_ids as both input and labels
#         outputs = model(input_ids, labels=input_ids)
        
#     # Get the logits and unaveraged token-level losses
#     logits = outputs.logits
#     loss_fct = torch.nn.CrossEntropyLoss(reduction='none')  # Use 'none' to avoid averaging
#     shift_logits = logits[..., :-1, :].contiguous()  # Shift logits to align with labels
#     shift_labels = input_ids[..., 1:].contiguous()  # Shift labels for token loss computation
    
#     # Compute the unaveraged token-level loss
#     unaveraged_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
#     # Reshape the loss back to the original shape of the input sequence
#     unaveraged_loss = unaveraged_loss.view(shift_labels.size())
    
#     # Compute the per-token probabilities
#     probabilities = torch.nn.functional.log_softmax(shift_logits, dim=-1)
#     all_prob = []
#     for i, token_id in enumerate(shift_labels[0]):
#         probability = probabilities[0, i, token_id].item()
#         all_prob.append(probability)

#     # Perplexity is computed on the mean loss
#     avg_loss = unaveraged_loss.mean()
#     ppl = torch.exp(avg_loss).item()
    
#     return ppl, all_prob, unaveraged_loss.cpu().numpy()



# def inference(text: str, model, tokenizer, temp_file) -> Dict:
#     pred = {}

#     _, all_prob, p1_likelihood = compute_ppl(text, model, tokenizer, device=model.device)
#     _, _, p_lower_likelihood = compute_ppl(text.lower(), model, tokenizer, device=model.device)
#     zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

#     np.save(temp_file, p1_likelihood)

#     # pred["PPL"] = float(p1_likelihood)
#     # pred["PPL/lower"] = float(p1_likelihood / p_lower_likelihood)
#     # pred["PPL/zlib"] = float(p1_likelihood / zlib_entropy)

#     # # min-k prob
#     # for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
#     #     k_length = int(len(all_prob)*ratio)
#     #     topk_prob = np.sort(all_prob)[:k_length]
#     #     pred[f"Min-{int(ratio*100)}%"] = float(-np.mean(topk_prob).item())

#     return pred


# def eval_data(data: List[str], model, tokenizer, temp_file):
#     out = []
#     for text in tqdm(data):
#         out.append({'text': text} | inference(text, model, tokenizer, temp_file))
#     return out


# def sweep(ppl, y):
#     fpr, tpr, _ = get_roc_curve(y, -ppl)
#     acc = np.max(1-(fpr+(1-tpr))/2)
#     return fpr, tpr, get_auc(fpr, tpr), acc


# def eval(
#     forget_data: List[str],
#     retain_data: List[str],
#     holdout_data: List[str],
#     model, tokenizer,

#     # for plot
#     temp_dir
# ):
#     log = {}
#     print("Evaluating on the forget set...")
#     log['forget'] = eval_data(forget_data, model, tokenizer, temp_dir + '/forget_loss.npy')
#     print("Evaluating on the retain set...")
#     log['retain'] = eval_data(retain_data, model, tokenizer, temp_dir + '/retain_loss.npy')
#     print("Evaluating on the holdout set...")
#     log['holdout'] = eval_data(holdout_data, model, tokenizer, temp_dir + '/holdout_loss.npy')

#     auc = {}
#     ppl_types = list(log['forget'][0].keys())
#     ppl_types.remove('text')
#     for split0 in ['forget', 'retain', 'holdout']:
#         for split1 in ['forget', 'retain', 'holdout']:
#             log0, log1 = log[split0], log[split1]
#             for ppl_type in ppl_types:
#                 ppl_nonmember = [d[ppl_type] for d in log0]
#                 ppl_member = [d[ppl_type] for d in log1]
#                 ppl = np.array(ppl_nonmember + ppl_member)
#                 y = np.array([0] * len(ppl_nonmember) + [1] * len(ppl_member))
#                 _, _, auc_score, _ = sweep(ppl, y)
#                 auc[f"{split0}_{split1}_{ppl_type}"] = auc_score
#     return auc, log