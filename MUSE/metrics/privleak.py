import sys
sys.path.append(".")
sys.path.append("../baselines")

from typing import List, Dict
import torch
from tqdm import tqdm
import zlib
import numpy as np
from sklearn.metrics import auc as get_auc, roc_curve as get_roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings


def compute_ppl(text: str, model, tokenizer, device='cuda'):
    # Tokenize with attention_mask and padding
    inputs = tokenizer(
        text,
        return_tensors='pt',
        add_special_tokens=True
    )
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss, logits = outputs[:2]

    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    input_ids_processed = input_ids[0][1:]
    all_prob = []
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    ppl = torch.exp(loss).item()
    return ppl, all_prob, loss.item()


def inference(text: str, model, tokenizer) -> Dict:
    pred = {}

    try:
        _, all_prob, p1_likelihood = compute_ppl(text, model, tokenizer, device=model.device)
        _, _, p_lower_likelihood = compute_ppl(text.lower(), model, tokenizer, device=model.device)
        zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

        pred["PPL"] = float(p1_likelihood)
        
        # Safe division to avoid NaNs
        if p_lower_likelihood != 0:
            pred["PPL/lower"] = float(p1_likelihood / p_lower_likelihood)
        else:
            pred["PPL/lower"] = float('nan')
            
        if zlib_entropy != 0:
            pred["PPL/zlib"] = float(p1_likelihood / zlib_entropy)
        else:
            pred["PPL/zlib"] = float('nan')

        # min-k prob
        if len(all_prob) > 0:
            for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                k_length = int(len(all_prob)*ratio)
                if k_length > 0:
                    topk_prob = np.sort(all_prob)[:k_length]
                    mean_topk = np.mean(topk_prob)
                    if np.isfinite(mean_topk):
                        pred[f"Min-{int(ratio*100)}%"] = float(-mean_topk.item())
                    else:
                        pred[f"Min-{int(ratio*100)}%"] = float('nan')
                else:
                    pred[f"Min-{int(ratio*100)}%"] = float('nan')
        else:
            for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                pred[f"Min-{int(ratio*100)}%"] = float('nan')
                
    except Exception as e:
        print(f"\033[93mWarning: Error processing text (length {len(text)}): {str(e)}. Setting all metrics to NaN.\033[0m")
        pred["PPL"] = float('nan')
        pred["PPL/lower"] = float('nan')
        pred["PPL/zlib"] = float('nan')
        for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            pred[f"Min-{int(ratio*100)}%"] = float('nan')

    return pred


def eval_data(data: List[str], model, tokenizer):
    out = []
    for text in tqdm(data):
        out.append({'text': text} | inference(text, model, tokenizer))
    return out


def filter_nans_and_balance_subsets(log):
    """
    Filter out NaN values and ensure all subsets have the same number of examples.
    Prints yellow warnings when NaNs are found.
    """
    # Get all metric keys (excluding 'text')
    ppl_types = list(log['forget'][0].keys())
    ppl_types.remove('text')
    
    # Filter out entries with NaNs for each subset
    filtered_log = {}
    original_counts = {}
    
    for split in ['forget', 'retain', 'holdout']:
        original_counts[split] = len(log[split])
        filtered_entries = []
        nan_count = 0
        
        for entry in log[split]:
            has_nan = False
            for metric in ppl_types:
                if np.isnan(entry[metric]) or np.isinf(entry[metric]):
                    has_nan = True
                    break
            
            if has_nan:
                nan_count += 1
            else:
                filtered_entries.append(entry)
        
        if nan_count > 0:
            print(f"\033[93mWarning: Found {nan_count} entries with NaN/Inf values in {split} set. "
                  f"Filtered from {original_counts[split]} to {len(filtered_entries)} examples.\033[0m")
        
        filtered_log[split] = filtered_entries
    
    # Get the minimum count across all subsets
    min_count = min(len(filtered_log[split]) for split in ['forget', 'retain', 'holdout'])
    
    if min_count == 0:
        raise ValueError("All subsets have been filtered out due to NaN/Inf values. Cannot proceed.")
    
    # Balance all subsets to have the same number of examples
    balanced_log = {}
    for split in ['forget', 'retain', 'holdout']:
        original_filtered_count = len(filtered_log[split])
        if original_filtered_count > min_count:
            print(f"\033[93mWarning: Reducing {split} set from {original_filtered_count} to {min_count} "
                  f"examples to ensure equal subset sizes.\033[0m")
            balanced_log[split] = filtered_log[split][:min_count]
        else:
            balanced_log[split] = filtered_log[split]
    
    print(f"Final balanced subset sizes: {min_count} examples each for forget, retain, and holdout sets.")
    
    return balanced_log


def sweep(ppl, y):
    # Additional safety check for NaNs
    ppl = np.array(ppl)
    y = np.array(y)
    
    # Filter out any remaining NaNs/Infs
    valid_mask = np.isfinite(ppl)
    if not np.all(valid_mask):
        nan_count = np.sum(~valid_mask)
        print(f"\033[93mWarning: Found {nan_count} NaN/Inf values in sweep function. Filtering them out.\033[0m")
        ppl = ppl[valid_mask]
        y = y[valid_mask]
    
    if len(ppl) == 0:
        print(f"\033[93mWarning: No valid data points after filtering NaNs. Returning default values.\033[0m")
        return np.array([0, 1]), np.array([0, 1]), 0.5, 0.5
    
    fpr, tpr, _ = get_roc_curve(y, -ppl)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, get_auc(fpr, tpr), acc

def plot_privleak_results(log, auc, plot_dir="privleak_plots"):
    """
    Generates and saves the following plots for privacy leakage evaluation:

    1. Histogram for each metric and split:
       - Shows the distribution of metric values (e.g., PPL, Min-k%) for each data split (forget, retain, holdout).
       - Helps visualize how well the splits are separated by each metric.

    2. ROC curve for each metric and split pair:
       - Plots the True Positive Rate vs. False Positive Rate for distinguishing between two splits using a given metric.
       - The AUC (Area Under Curve) value is shown in the legend.
       - Useful for assessing how well a metric can distinguish between, e.g., forget and retain sets.

    3. AUC heatmap for each metric:
       - Shows a matrix of AUC values for all split pairs (forget, retain, holdout) for a given metric.
       - Provides a quick overview of which splits are most/least separable for each metric.

    Returns:
        plots (dict): Mapping from plot description to saved file path.
    """
    os.makedirs(plot_dir, exist_ok=True)
    plots = {}
    ppl_types = list(log['forget'][0].keys())
    ppl_types.remove('text')
    split_names = ['forget', 'retain', 'holdout']

    # 1. Histograms for each metric and split
    for ppl_type in ppl_types:
        plt.figure()
        for split in split_names:
            values = [d[ppl_type] for d in log[split]]
            # Filter out any remaining NaNs for plotting
            values = [v for v in values if np.isfinite(v)]
            if len(values) > 0:
                sns.histplot(values, kde=True, label=split, stat="density", element="step", fill=True)
            else:
                print(f"\033[93mWarning: No valid values for {ppl_type} in {split} split for histogram.\033[0m")
        plt.title(f"Histogram: {ppl_type}")
        plt.legend()
        safe_ppl_type = ppl_type.replace("/", "_")
        hist_path = os.path.join(plot_dir, f"hist_{safe_ppl_type}.png")
        plt.savefig(hist_path)
        plt.close()
        plots[f"hist_{safe_ppl_type}"] = plt

    # 2. ROC curves and collect AUCs for heatmap
    auc_matrix = {ppl_type: np.zeros((3, 3)) for ppl_type in ppl_types}
    for i, split0 in enumerate(split_names):
        for j, split1 in enumerate(split_names):
            if i == j:
                continue
            log0, log1 = log[split0], log[split1]
            for ppl_type in ppl_types:
                auc_key = f"{split0}_{split1}_{ppl_type}"
                auc_score = auc[auc_key]
                auc_matrix[ppl_type][i, j] = auc_score

                # ROC curve
                ppl_nonmember = [d[ppl_type] for d in log0]
                ppl_member = [d[ppl_type] for d in log1]
                
                # Filter out NaNs before plotting
                ppl_nonmember = [v for v in ppl_nonmember if np.isfinite(v)]
                ppl_member = [v for v in ppl_member if np.isfinite(v)]
                
                if len(ppl_nonmember) > 0 and len(ppl_member) > 0:
                    ppl = np.array(ppl_nonmember + ppl_member)
                    y = np.array([0] * len(ppl_nonmember) + [1] * len(ppl_member))
                    fpr, tpr, _ = get_roc_curve(y, -ppl)
                    plt.figure()
                    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
                    plt.plot([0, 1], [0, 1], 'k--', label="Random")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title(f"ROC Curve: {auc_key}")
                    plt.legend(loc="lower right")
                    safe_auc_key = auc_key.replace("/", "_")
                    plot_path = os.path.join(plot_dir, f"roc_{safe_auc_key}.png")
                    plt.savefig(plot_path)
                    plt.close()
                    plots[f"roc_{safe_auc_key}"] = plt
                else:
                    print(f"\033[93mWarning: No valid values for ROC curve {auc_key}.\033[0m")

    # 3. AUC Heatmaps for each metric
    for ppl_type in ppl_types:
        plt.figure(figsize=(6, 5))
        sns.heatmap(auc_matrix[ppl_type], annot=True, fmt=".3f", xticklabels=split_names, yticklabels=split_names, cmap="viridis")
        plt.title(f"AUC Heatmap: {ppl_type}")
        safe_ppl_type = ppl_type.replace("/", "_")
        heatmap_path = os.path.join(plot_dir, f"auc_heatmap_{safe_ppl_type}.png")
        plt.savefig(heatmap_path)
        plt.close()
        plots[f"auc_heatmap_{safe_ppl_type}"] = plt

    return plots

def eval(
    forget_data: List[str],
    retain_data: List[str],
    holdout_data: List[str],
    model, tokenizer,
    plot_dir: str = "privleak_plots"
):
    log = {}
    print("Evaluating on the forget set...")
    log['forget'] = eval_data(forget_data, model, tokenizer)
    print("Evaluating on the retain set...")
    log['retain'] = eval_data(retain_data, model, tokenizer)
    print("Evaluating on the holdout set...")
    log['holdout'] = eval_data(holdout_data, model, tokenizer)

    # Filter NaNs and balance subset sizes
    print("Filtering NaN values and balancing subset sizes...")
    log = filter_nans_and_balance_subsets(log)

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
                
    plots = plot_privleak_results(log, auc, plot_dir=plot_dir)
    
    return auc, log, plots

