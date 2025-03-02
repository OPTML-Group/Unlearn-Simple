import json
import re
import sys

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

sys.path.append("src")


def generate_responses(model, tokenizer, prompts):
    inputs = tokenizer(
        prompts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            max_new_tokens=50,
            do_sample=False,
        )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, response in enumerate(responses):
        responses[i] = response.replace(prompts[i], "")
    return responses


def extract_first_email(text):
    """
    Extracts the first email address found in a given string.
    """
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(email_pattern, text)
    return match.group() if match else None


def eval_context_extraction(model, tokenizer, if_llama=False, batch_size=8):
    context_data = load_dataset(
        "json",
        data_files="files/data/PII/context.jsonl",
        split="train",
        cache_dir="./.cache",
    )

    correct = 0
    total = 0

    for i in tqdm.tqdm(range(0, len(context_data), batch_size), desc="context"):
        if i + batch_size > len(context_data):
            prompts = context_data["prompt"][i:]
            emails = context_data["email"][i:]
        else:
            prompts = context_data["prompt"][i : i + batch_size]
            emails = context_data["email"][i : i + batch_size]
        responses = generate_responses(model, tokenizer, prompts)
        for response, email in zip(responses, emails):
            extracted_email = extract_first_email(response)
            if extracted_email is None:
                total += 1
                continue
            if extracted_email == email:
                correct += 1
            total += 1

    return correct / total


def contruct_prompts(examples, email2name, prompt_template):
    prompts = examples["prompt"]
    emails = examples["email"]

    few_shots = {
        "1": [],
        "2": [],
        "3": [],
        "4": [],
        "5": [],
        "6": [],
        "7": [],
        "8": [],
        "9": [],
        "10": [],
    }
    for i in range(len(prompts)):
        prompt = prompts[i]
        email = emails[i]
        for key in prompt_template.keys():
            total_prompts = ""
            for i in range(len(prompt)):
                tmp_prompts = (
                    prompt_template[key].format(prompt[i][1]) + " " + prompt[i][0]
                )
                total_prompts = total_prompts + tmp_prompts + "\n"
            total_prompts = (
                total_prompts + prompt_template[key].format(email2name[email]) + " "
            )
            few_shots[key].append(total_prompts)
    return few_shots


def eval_few_shots_extraction(model, tokenizer, if_llama=False, batch_size=8):
    one_shot_non_domain = load_dataset(
        "json",
        data_files="files/data/PII/one_shot_non_domain.jsonl",
        split="train",
        cache_dir="./.cache",
    )
    one_shot = load_dataset(
        "json",
        data_files="files/data/PII/one_shot.jsonl",
        split="train",
        cache_dir="./.cache",
    )
    two_shot = load_dataset(
        "json",
        data_files="files/data/PII/two_shot.jsonl",
        split="train",
        cache_dir="./.cache",
    )
    two_shot_non_domain = load_dataset(
        "json",
        data_files="files/data/PII/two_shot_non_domain.jsonl",
        split="train",
        cache_dir="./.cache",
    )
    email2name = {}
    with open("files/data/PII/email2name.jsonl") as f:
        for line in f:
            data = json.loads(line)
            email2name[data["email"]] = data["name"]
    with open("files/data/PII/prompt_template.json") as f:
        prompt_template = json.load(f)
    correct = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
    }
    total = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
    }

    for i in tqdm.tqdm(
        range(0, len(one_shot_non_domain), batch_size), desc="one_shot_non_domain"
    ):
        if i + batch_size > len(one_shot_non_domain):
            data = one_shot_non_domain[i:]
        else:
            data = one_shot_non_domain[i : i + batch_size]
        emails = data["email"]
        few_shots = contruct_prompts(data, email2name, prompt_template)
        for key in few_shots.keys():
            responses = generate_responses(model, tokenizer, few_shots[key])
            for response, email in zip(responses, emails):
                extracted_email = extract_first_email(response)
                if extracted_email is None:
                    total[key] += 1
                    continue
                if extracted_email == email:
                    correct[key] += 1
                total[key] += 1
    one_shot_non_domain_acc = {key: correct[key] / total[key] for key in correct.keys()}
    correct = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
    }
    total = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
    }

    for i in tqdm.tqdm(range(0, len(one_shot), batch_size), desc="one_shot"):
        if i + batch_size > len(one_shot):
            data = one_shot[i:]
        else:
            data = one_shot[i : i + batch_size]
        emails = data["email"]
        few_shots = contruct_prompts(data, email2name, prompt_template)
        for key in few_shots.keys():
            responses = generate_responses(model, tokenizer, few_shots[key])
            for response, email in zip(responses, emails):
                extracted_email = extract_first_email(response)
                if extracted_email is None:
                    total[key] += 1
                    continue
                if extracted_email == email:
                    correct[key] += 1
                total[key] += 1
    one_shot_acc = {key: correct[key] / total[key] for key in correct.keys()}
    correct = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
    }
    total = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
    }

    for i in tqdm.tqdm(range(0, len(two_shot), batch_size), desc="two_shot"):
        if i + batch_size > len(two_shot):
            data = two_shot[i:]
        else:
            data = two_shot[i : i + batch_size]
        emails = data["email"]
        few_shots = contruct_prompts(data, email2name, prompt_template)
        for key in few_shots.keys():
            responses = generate_responses(model, tokenizer, few_shots[key])
            for response, email in zip(responses, emails):
                extracted_email = extract_first_email(response)
                if extracted_email is None:
                    total[key] += 1
                    continue
                if extracted_email == email:
                    correct[key] += 1
                total[key] += 1
    two_shot_acc = {key: correct[key] / total[key] for key in correct.keys()}
    correct = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
    }
    total = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
    }

    for i in tqdm.tqdm(
        range(0, len(two_shot_non_domain), batch_size), desc="two_shot_non_domain"
    ):
        if i + batch_size > len(two_shot_non_domain):
            data = two_shot_non_domain[i:]
        else:
            data = two_shot_non_domain[i : i + batch_size]
        emails = data["email"]
        few_shots = contruct_prompts(data, email2name, prompt_template)
        for key in few_shots.keys():
            responses = generate_responses(model, tokenizer, few_shots[key])
            for response, email in zip(responses, emails):
                extracted_email = extract_first_email(response)
                if extracted_email is None:
                    total[key] += 1
                    continue
                if extracted_email == email:
                    correct[key] += 1
                total[key] += 1
    two_shot_non_domain_acc = {key: correct[key] / total[key] for key in correct.keys()}
    return one_shot_non_domain_acc, one_shot_acc, two_shot_acc, two_shot_non_domain_acc


def eval_PII(model_name, output_dir=".", batch_size=8):
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

    context_acc = eval_context_extraction(model, tokenizer, batch_size=batch_size)
    print(context_acc)
    (
        one_shot_non_domain_acc,
        one_shot_acc,
        two_shot_acc,
        two_shot_non_domain_acc,
    ) = eval_few_shots_extraction(model, tokenizer, batch_size=batch_size)

    results = {
        "context_acc": context_acc,
        "one_shot_non_domain_acc": one_shot_non_domain_acc,
        "one_shot_acc": one_shot_acc,
        "two_shot_acc": two_shot_acc,
        "two_shot_non_domain_acc": two_shot_non_domain_acc,
    }

    with open(f"{output_dir}/PII.json", "w") as f:
        json.dump(results, f, indent=4)
