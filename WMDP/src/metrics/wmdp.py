import sys

import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

sys.path.append("src")
from rouge_score import rouge_scorer

from dataset import WMDPBio, WMDPCyber

LABLES = ["A", "B", "C", "D"]
LABLES_ANSWER = {"A": 0, "B": 1, "C": 2, "D": 3}


def compute_prob(model, examples, tokenizer):
    input_ids = examples["input_ids"]
    attention_mask = examples["attention_mask"]
    true_label = examples["answer"]
    LABLES_ID = [tokenizer.encode(label)[1] for label in LABLES]
    texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    original_answers = [LABLES[ans] for ans in true_label]
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda()
        )
        logits = outputs.logits[:, -1, LABLES_ID]
        prob = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(prob, dim=-1)
        corr = sum(
            [1 for i in range(len(prediction)) if prediction[i] == int(true_label[i])]
        )
        total = len(prediction)
        predictions = [LABLES[p] for p in prediction]
    return corr, total, texts, original_answers, predictions
    

def eval_wmdp(model_name, output_dir=".", batch_size=8):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left"
    left_pad_tokenizer.padding_size = "longest"
    try:
        left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
        left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    except:
        left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
        left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    dataset = WMDPCyber("wmdp-cyber", subset="forget")
    dataset = dataset.build_dataset(tokenizer)
    test_dataset = dataset["test"]
    corr = 0
    total = 0
    original_answers = []
    predictions = []
    texts = []
    cyber_answers = []
    cyber_predictions = []
    cyber_texts = []
    for i in tqdm.tqdm(
        range(0, len(test_dataset), batch_size), desc=f"evaluating WMDP-cyber"
    ):
        if i + batch_size > len(test_dataset):
            examples = test_dataset[i:]
        else:
            examples = test_dataset[i : i + batch_size]
        corr_tmp, total_tmp,question,true_answer,prediction_answer = compute_prob(model, examples, tokenizer)
        corr += corr_tmp
        total += total_tmp
        cyber_texts.extend(question)
        cyber_answers.extend(true_answer)
        cyber_predictions.extend(prediction_answer)
    Acc = corr / total
    print(f"Accuracy: {Acc}")
    print(cyber_texts)
    print(cyber_answers)
    print(cyber_predictions)
    bio_dataset = WMDPBio("wmdp-bio", subset="forget")
    bio_dataset = bio_dataset.build_dataset(tokenizer)
    bio_test_dataset = bio_dataset["test"]
    bio_corr = 0
    bio_total = 0
    for i in tqdm.tqdm(
        range(0, len(bio_test_dataset), batch_size), desc=f"evaluating WMDP-bio"
    ):
        if i + batch_size > len(bio_test_dataset):
            examples = bio_test_dataset[i:]
        else:
            examples = bio_test_dataset[i : i + batch_size]
        corr_tmp, total_tmp,question,true_answer,prediction_answer = compute_prob(model, examples, tokenizer)
        bio_corr += corr_tmp
        bio_total += total_tmp
        texts.extend(question)
        original_answers.extend(true_answer)
        predictions.extend(prediction_answer)
    bio_Acc = bio_corr / bio_total
    results = {"Cyber-Accuracy": Acc, "Bio-Accuracy": bio_Acc, "Bio-texts":[{"question":text,"true_answer":true_answer,"prediction":prediction_answer} for text,true_answer,prediction_answer in zip(texts,original_answers,predictions)],"Cyber-texts":[{"question":text,"true_answer":true_answer,"prediction":prediction_answer} for text,true_answer,prediction_answer in zip(cyber_texts,cyber_answers,cyber_predictions)]}
    import json

    with open(f"{output_dir}/wmdp_generation.json", "w") as f:
        json.dump(results, f, indent=4)
    return Acc
