import sys

sys.path.append("src")
import random
import zlib
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from sklearn.svm import SVC
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.dataset import build_unlearn_dataset, get_WikiMIA_dataset


def calculatePerplexity(sentence, model, tokenizer):
    input_ids = tokenizer.encode(sentence, return_tensors="pt").cuda()
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()


def inference(model1, ref_model, tokenizer1, tokenizer2, text, ex):
    pred = {}

    p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1)
    p_lower, _, p_lower_likelihood = calculatePerplexity(
        text.lower(), model1, tokenizer1
    )

    p_ref, all_prob_ref, p_ref_likelihood = calculatePerplexity(
        text, ref_model, tokenizer2
    )

    # ppl
    pred["ppl"] = p1
    # Ratio of log ppl of large and small models
    pred["ppl/Ref_ppl (calibrate PPL to the reference model)"] = (
        p1_likelihood - p_ref_likelihood
    )

    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, "utf-8")))
    pred["ppl/zlib"] = np.log(p1) / zlib_entropy
    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob) * ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

    ex["pred"] = pred
    return ex


def Mink_MIA(model, ref_model, tokenizer, forget_dataset, test_dataset, remain_dataset):
    forget_preds = []

    remaining_preds = []

    test_preds = []

    for data in tqdm(forget_dataset, desc="forget dataset"):
        text = data["input"]
        new_ex = inference(model, ref_model, tokenizer, tokenizer, text, data)
        forget_preds.append(new_ex)

    for data in tqdm(remain_dataset, desc="remain dataset"):
        text = data["input"]
        new_ex = inference(model, ref_model, tokenizer, tokenizer, text, data)
        remaining_preds.append(new_ex)

    for data in tqdm(test_dataset, desc="test dataset"):
        if len(test_preds) == len(remaining_preds):
            break
        text = data["input"]
        new_ex = inference(model, ref_model, tokenizer, tokenizer, text, data)
        test_preds.append(new_ex)

    MIAs = defaultdict()
    train_accs = defaultdict()
    best_train_acc = 0
    for metric in forget_preds[0]["pred"].keys():
        forget_metric = [ex["pred"][metric] for ex in forget_preds]
        remaining_metric = [ex["pred"][metric] for ex in remaining_preds]
        test_metric = [ex["pred"][metric] for ex in test_preds]
        mia_curr, train_acc = SVC_fit_predict(
            remaining_metric, test_metric, forget_metric
        )
        train_accs[metric] = train_acc
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_metric = metric
        MIAs[metric] = mia_curr

    MIAs["best_metrics"] = best_metric
    MIAs["most_accurate_MIA"] = MIAs[best_metric]
    MIAs["best_train_acc"] = best_train_acc
    return MIAs


def SVC_fit_predict(shadow_train, shadow_test, target_test):
    n_shadow_train = len(shadow_train)
    n_shadow_test = len(shadow_test)
    n_target_test = len(target_test)

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    X_shadow = np.concatenate([shadow_train, shadow_test]).reshape(-1, 1)
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf.fit(X_shadow, Y_shadow)
    Training_acc = clf.score(X_shadow, Y_shadow)
    accs = []
    X_target_test = np.array(target_test).reshape(n_target_test, -1)
    acc_test = 1 - clf.predict(X_target_test).mean()
    accs.append(acc_test)

    return np.mean(accs), Training_acc


def eval_MIA(
    model_name, ref_model_name, dataset, dataset_seed=8888, fraction=0.1, output_dir="."
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        torch_dtype=torch.float16,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    forget_dataset, remain_dataset, Total_test_data = build_unlearn_dataset(
        dataset, dataset_seed, fraction
    )

    MIAs = Mink_MIA(
        model, ref_model, tokenizer, forget_dataset, Total_test_data, remain_dataset
    )

    import json

    with open(f"{output_dir}/MIA.json", "w") as f:
        json.dump(MIAs, f)


if __name__ == "__main__":
    model_ref = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-350m",
        torch_dtype=torch.float16,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b",
        torch_dtype=torch.float16,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
    test_data = get_WikiMIA_dataset(32)

    Total_training_data = [item for item in test_data if item["label"] == 1]
    Total_test_data = [item for item in test_data if item["label"] == 0]
    dataset_seed = 8888
    torch.manual_seed(dataset_seed)
    np.random.seed(dataset_seed)
    random.seed(dataset_seed)
    random.shuffle(Total_training_data)
    random.shuffle(Total_test_data)
    fraction = 0.5
    forget_dataset = Total_training_data[: int(len(Total_training_data) * fraction)]
    remain_dataset = Total_training_data[int(len(Total_training_data) * fraction) :]

    MIAs = Mink_MIA(
        model, model_ref, tokenizer, forget_dataset, Total_test_data, remain_dataset
    )
