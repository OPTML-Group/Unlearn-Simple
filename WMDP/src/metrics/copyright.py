import sys

import sacrebleu
import torch
import tqdm
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
from metrics.MIA import calculatePerplexity
sys.path.append("src")
import zlib
import numpy as np
def inference(model,tokenizer,text,ex):
    pred = {}
    p1, all_prob, p1_likelihood = calculatePerplexity(text, model, tokenizer)
    p_lower, _, p_lower_likelihood = calculatePerplexity(
        text.lower(), model, tokenizer
    )

    # ppl
    pred["ppl"] = p1
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

def eval_privacy_score(model, tokenizer, dataset):
    val_preds = []
    for idx in tqdm.tqdm(range(len(dataset)), desc="computing training data privacy score"):
        data  = dataset[idx]
        text = dataset[idx]["text"]
        preds = inference(model,  tokenizer,  text, data)
        val_preds.append(preds)
    return val_preds

def eval_leakage_rate(model, tokenizer, dataset, batch_size = 4):

    rougeLs = []
    bleus = []
    scorers = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    generated_texts = []
    for i in tqdm.tqdm(
        range(0, len(dataset), batch_size),
        desc="computing training data leakage rate",
    ):
        if i + batch_size > len(dataset):
            batch = dataset[i:]
        else:
            batch = dataset[i : i + batch_size]
        max_length = max([len(x) for x in batch["input_ids"]])
        for idx, x in enumerate(batch["input_ids"]):
            batch["input_ids"][idx] = [tokenizer.pad_token_id] * (max_length - len(x)) + x
        input_ids = torch.tensor(batch["input_ids"])
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids.cuda(),
                max_length=600,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        length = input_ids.size(1)
        decoded_outputs = tokenizer.batch_decode(
            outputs[:,length+1:], skip_special_tokens=True
        )
        ground_truth = batch["response"]
        for idx, text in enumerate(decoded_outputs):
            score = scorers.score(ground_truth[idx], text)
            rougeLs.append(score["rougeL"].recall)
            bleu = sacrebleu.corpus_bleu([text], [[ground_truth[idx]]]).score
            bleus.append(bleu)
            generated_texts.append(text)
    mean_bleu = sum(bleus) / len(bleus)
    mean_rougeL = sum(rougeLs) / len(rougeLs)

    return mean_bleu, mean_rougeL, generated_texts



def eval_copyright(
    model_name,
    batch_size=128,
    output_dir=".",
    if_llama=False,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left"
    left_pad_tokenizer.padding_size = "longest"
    try:
        left_pad_tokenizer.pad_token = left_pad_tokenizer.pad_token
        left_pad_tokenizer.pad_token_id = left_pad_tokenizer.pad_token_id
    except:
        left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
        left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    tokenizer = left_pad_tokenizer

    results = {}
    dataset = HP("HP", if_llama=if_llama)
    results["train"] = {}
    results["test"] = {}

    for key in ["train", "test"]:
        for k in [ 300]:
            path = f'files/data/hp/hp_{key}_qa_{k}.jsonl'
            eval_dataset = dataset.build_test_dataset(tokenizer, path)
            eval_dataset_with_token = dataset.build_test_dataset_without_tokenized(path)
            # MIA = eval_privacy_score(model, tokenizer, eval_dataset_with_token)
            # print(MIA)
            # mean_mia_scores = {
            #     k: sum([pred["pred"][k] for pred in MIA]) / len(MIA)
            #     for k in MIA[0]["pred"].keys()
            # }
            mean_bleu, mean_rougeL, generated_texts = eval_leakage_rate(model, tokenizer, eval_dataset, batch_size)
            # results[key][k] = {"bleu": mean_bleu, "rougeL": mean_rougeL, "mean_mia_scores": mean_mia_scores}
            results[key][k] = {"bleu": mean_bleu, "rougeL": mean_rougeL, "generated_texts": generated_texts}

    with open(f"{output_dir}/copyright.json", "w") as f:
        json.dump(results, f, indent=4)
            
        